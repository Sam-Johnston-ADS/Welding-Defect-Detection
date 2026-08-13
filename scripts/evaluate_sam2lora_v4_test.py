"""
SAM 2 + LoRA Fine-tuning — Cascade Test Evaluation Script
Cascade: Scratch-Trained SegFormer-B2 -> SAM 2 box-prompt refinement

Run locally:
  python scripts/evaluate_sam2lora_v5_test.py
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import Sam2Processor, Sam2Model, SegformerForSemanticSegmentation, SegformerConfig
from peft import LoraConfig, get_peft_model

# ==========================================================
# WINDOWS PATHS & CONSTANTS
# ==========================================================
BASE_DIR          = Path(r"E:\weld-defect-detection (1)\weld-defect-detection")
COMBINED_DATA_DIR = BASE_DIR / "data" / "reviewed" / "WDXI" / "combined"

COMBINED_IMG = str(COMBINED_DATA_DIR / "images")
COMBINED_MSK = str(COMBINED_DATA_DIR / "labels")
SPLIT_PATH   = str(COMBINED_DATA_DIR / "split.json")

MODEL_DIR   = str(BASE_DIR / "models" / "sam2")
SAVE_PATH   = os.path.join(MODEL_DIR, "sam2_lora_best_fixed.pth")

SEGFORMER_CHECKPOINT = str(BASE_DIR / "models" / "segformer_v4_scratch" / "segformer_v4_scratch_best.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 4
NUM_CLASSES  = 1
LORA_R       = 4
LORA_ALPHA   = 8
LORA_DROPOUT = 0.1

# ==========================================================
# DATASET & CASCADE LOGIC
# ==========================================================
class WeldSAM2CascadeDataset(Dataset):
    def __init__(self, file_list, img_dir, mask_dir, processor, segformer_model):
        self.files           = file_list      
        self.img_dir         = img_dir
        self.mask_dir        = mask_dir
        self.processor       = processor
        self.segformer_model = segformer_model

    def __len__(self):
        return len(self.files)

    @torch.no_grad()
    def _get_segformer_box(self, image_np):
        h, w, _ = image_np.shape
        img_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(512, 512),
                                   mode="bilinear", align_corners=False).to(DEVICE)

        outputs = self.segformer_model(pixel_values=img_tensor)
        upsampled = F.interpolate(outputs.logits, size=(h, w),
                                  mode="bilinear", align_corners=False)
        pred_mask = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()

        nonzero = np.argwhere(pred_mask > 0)
        if len(nonzero) == 0:
            return [0, 0, int(w), int(h)]

        y_min, x_min = nonzero.min(axis=0)
        y_max, x_max = nonzero.max(axis=0)
        pad_x = int((x_max - x_min) * 0.05)
        pad_y = int((y_max - y_min) * 0.05)

        return [int(max(0, x_min-pad_x)), int(max(0, y_min-pad_y)),
                int(min(w, x_max+pad_x)), int(min(h, y_max+pad_y))]

    def __getitem__(self, idx):
        fname     = self.files[idx]
        image_pil = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        image_np  = np.array(image_pil)
        mask      = np.array(Image.open(os.path.join(self.mask_dir, fname)).convert("L"))
        mask_bin  = (mask > 127).astype(np.uint8)

        box = self._get_segformer_box(image_np)

        inputs = self.processor(images=image_pil, input_boxes=[[box]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(mask_bin, dtype=torch.float32)
        return inputs

# ==========================================================
# LOSS & METRICS FUNCTIONS
# ==========================================================
def dice_loss(pred, target, smooth=1.0):
    pred  = torch.sigmoid(pred)
    inter = (pred*target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3))+target.sum(dim=(2,3))
    return 1 - (2*inter+smooth)/(union+smooth)

def combined_loss(pred, target):
    target = target.unsqueeze(1).float()
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return F.binary_cross_entropy_with_logits(pred, target) + dice_loss(pred, target).mean()

def compute_test_metrics(pred_logits, target, thr=0.5):
    pred = (torch.sigmoid(pred_logits) > thr).float()
    target = target.unsqueeze(1).float()
    
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred.float(), size=target.shape[-2:], mode="nearest")
        
    # Accuracy
    correct = (pred == target).sum().float()
    total_pixels = torch.numel(pred)
    accuracy = (correct / total_pixels).item()
    
    # IoU and Dice
    inter = (pred*target).sum(dim=(2,3))
    union = (pred+target).clamp(0,1).sum(dim=(2,3))
    
    iou = (inter/(union+1e-8)).mean().item()
    dice = (2*inter/(pred.sum(dim=(2,3))+target.sum(dim=(2,3))+1e-8)).mean().item()
    
    return iou, dice, accuracy

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"split.json not found at {SPLIT_PATH}.")
        
    with open(SPLIT_PATH) as f:
        split = json.load(f)
        
    print(f"Loading Test Split: {len(split['test'])} held-out images")

    # 1. LOAD SEGFORMER (Cascade Stage 1)
    print("\nBuilding SegFormer-B2 architecture (Stage 1)...")
    segformer_config = SegformerConfig(
        num_channels=3, num_labels=2,
        depths=[3,4,6,3], hidden_sizes=[64,128,320,512],
        num_attention_heads=[1,2,5,8], decoder_hidden_size=768,
    )
    segformer_model = SegformerForSemanticSegmentation(segformer_config)

    if not os.path.exists(SEGFORMER_CHECKPOINT):
        raise FileNotFoundError(f"Missing SegFormer weights at {SEGFORMER_CHECKPOINT}")

    ckpt = torch.load(SEGFORMER_CHECKPOINT, map_location=DEVICE)
    segformer_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    segformer_model = segformer_model.to(DEVICE).eval()
    print("  Loaded SegFormer v4 for box prompting.")

    # 2. LOAD SAM 2 + LoRA (Cascade Stage 2)
    print("\nLoading SAM 2 base model + LoRA (Stage 2)...")
    MODEL_ID  = "facebook/sam2-hiera-small"
    processor = Sam2Processor.from_pretrained(MODEL_ID)
    model     = Sam2Model.from_pretrained(MODEL_ID)

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", target_modules=["attn.qkv", "attn.proj"]
    )
    model.vision_encoder = get_peft_model(model.vision_encoder, lora_config)

    # Load trained LoRA + mask_decoder weights
    if not os.path.exists(SAVE_PATH):
        raise FileNotFoundError(f"SAM2 LoRA weights not found at {SAVE_PATH}")
        
    print(f"  Loading best weights from: {SAVE_PATH}")
    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["lora_weights"], strict=False)
    model = model.to(DEVICE).eval()

    # 3. INITIALIZE DATALOADER
    test_dataset = WeldSAM2CascadeDataset(split["test"], COMBINED_IMG, COMBINED_MSK, processor, segformer_model)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print("\nStarting Test Evaluation (Cascade)...\n")
    
    test_loss = 0.0
    test_iou  = 0.0
    test_dice = 0.0
    test_acc  = 0.0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Testing]"):
            pv = batch["pixel_values"].to(DEVICE)
            ib = batch.get("input_boxes").to(DEVICE) if batch.get("input_boxes") is not None else None
            gt = batch["ground_truth_mask"].to(DEVICE)

            # Forward pass
            out = model(pixel_values=pv, input_boxes=ib, multimask_output=False)
            pm  = out.pred_masks.squeeze(1)
            
            # Loss and Metrics
            test_loss += combined_loss(pm, gt).item()
            iou, dice, accuracy = compute_test_metrics(pm, gt)
            
            test_iou  += iou
            test_dice += dice
            test_acc  += accuracy
            
    # Average across all batches
    num_batches = len(test_loader)
    test_loss /= num_batches
    test_iou  /= num_batches
    test_dice /= num_batches
    test_acc  /= num_batches

    print("\n" + "=" * 60)
    print("FINAL SAM 2 + LORA CASCADE TEST RESULTS")
    print("=" * 60)
    print(f"Test Accuracy : {test_acc * 100:.2f}%")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test IoU      : {test_iou:.4f}")
    print(f"Test Dice     : {test_dice:.4f}")
    print("=" * 60)