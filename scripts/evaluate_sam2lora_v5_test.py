"""
SAM 2 + LoRA (Stage 3.5 v5 ConvStem Cascade) Test Evaluation Script
Cascade: Scratch-Trained SegFormer-B2 (v5 ConvStem) -> SAM 2 Box-Prompt Refinement

Run locally from terminal:
  python test_sam2_v5_convstem.py
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

MODEL_DIR      = str(BASE_DIR / "models" / "sam2")
SAVE_PATH      = os.path.join(MODEL_DIR, "sam2_lora_best_v5_convstem.pth")
SEGFORMER_CKPT = str(BASE_DIR / "models" / "segformer_v5_convstem" / "segformer_v5_convstem_best.pth")

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 4
STEM_CHANNELS = 32
LORA_R       = 4
LORA_ALPHA   = 8
LORA_DROPOUT = 0.1

# ==========================================================
# 1. SEGFORMER V5 CONVSTEM ARCHITECTURE
# ==========================================================
class ConvStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=STEM_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SegformerWithConvStem(nn.Module):
    def __init__(self, segformer_config):
        super().__init__()
        self.stem      = ConvStem(in_channels=3, out_channels=STEM_CHANNELS)
        self.segformer = SegformerForSemanticSegmentation(segformer_config)

    def forward(self, pixel_values):
        stem_features = self.stem(pixel_values)
        return self.segformer(pixel_values=stem_features).logits


def load_v5_segformer(checkpoint_path, device):
    config = SegformerConfig(
        num_channels=STEM_CHANNELS,
        num_labels=2,
        depths=[3, 4, 6, 3],
        hidden_sizes=[64, 128, 320, 512],
        num_attention_heads=[1, 2, 5, 8],
        decoder_hidden_size=768,
        id2label={0: "background", 1: "defect"},
        label2id={"background": 0, "defect": 1},
    )

    model = SegformerWithConvStem(config)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing SegFormer v5 weights at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print("Loaded Custom SegFormer v5 (ConvStem) checkpoint successfully.")
    return model


# ==========================================================
# 2. CASCADE DATASET
# ==========================================================
class WeldSAM2CascadeDataset(Dataset):
    def __init__(self, file_list, img_dir, mask_dir, processor, segformer_model, device):
        self.files           = file_list
        self.img_dir         = img_dir
        self.mask_dir        = mask_dir
        self.processor       = processor
        self.segformer_model = segformer_model
        self.device          = device

    def __len__(self):
        return len(self.files)

    @torch.no_grad()
    def _get_segformer_box(self, image_np):
        h, w, _ = image_np.shape
        img_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(512, 512),
                                   mode="bilinear", align_corners=False).to(self.device)

        logits = self.segformer_model(img_tensor)
        if hasattr(logits, "logits"):
            logits = logits.logits

        upsampled = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        pred_mask = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()

        nonzero = np.argwhere(pred_mask > 0)
        if len(nonzero) == 0:
            return [0, 0, int(w), int(h)]

        y_min, x_min = nonzero.min(axis=0)
        y_max, x_max = nonzero.max(axis=0)
        pad_x = int((x_max - x_min) * 0.05)
        pad_y = int((y_max - y_min) * 0.05)

        return [int(max(0, x_min - pad_x)), int(max(0, y_min - pad_y)),
                int(min(w, x_max + pad_x)), int(min(h, y_max + pad_y))]

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
# 3. LOSS & METRIC EVALUATION
# ==========================================================
def dice_loss(pred, target, smooth=1.0):
    pred  = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - (2 * inter + smooth) / (union + smooth)

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
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).clamp(0, 1).sum(dim=(2, 3))
    
    iou = (inter / (union + 1e-8)).mean().item()
    dice = (2 * inter / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-8)).mean().item()
    
    return iou, dice, accuracy


# ==========================================================
# 4. MAIN EVALUATION LOOP
# ==========================================================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"split.json not found at {SPLIT_PATH}")

    with open(SPLIT_PATH) as f:
        split = json.load(f)

    print(f"Loading Test Split: {len(split['test'])} held-out images")

    # 1. Load SegFormer v5 ConvStem (Stage 1 Prompt Generator)
    print("\nBuilding SegFormer-B2 (v5 ConvStem) Prompt Generator...")
    segformer_model = load_v5_segformer(SEGFORMER_CKPT, DEVICE)

    # 2. Load SAM 2 Base Model + PEFT LoRA (Stage 2 Segmenter)
    print("\nLoading SAM 2 base model + LoRA...")
    MODEL_ID  = "facebook/sam2-hiera-small"
    processor = Sam2Processor.from_pretrained(MODEL_ID)
    model     = Sam2Model.from_pretrained(MODEL_ID)

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", target_modules=["attn.qkv", "attn.proj"]
    )
    model.vision_encoder = get_peft_model(model.vision_encoder, lora_config)

    if not os.path.exists(SAVE_PATH):
        raise FileNotFoundError(f"SAM 2 LoRA weights not found at {SAVE_PATH}")

    print(f"Loading best SAM 2 LoRA weights from: {SAVE_PATH}")
    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False)
    
    lora_weights = checkpoint.get("lora_weights", checkpoint)
    model.load_state_dict(lora_weights, strict=False)
    model = model.to(DEVICE).eval()

    # 3. Create Test DataLoader
    test_dataset = WeldSAM2CascadeDataset(
        split["test"], COMBINED_IMG, COMBINED_MSK, processor, segformer_model, DEVICE
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )

    print("\nStarting Test Evaluation (SAM 2 + v5 ConvStem Cascade)...\n")

    test_loss = 0.0
    test_iou  = 0.0
    test_dice = 0.0
    test_acc  = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Testing]"):
            pv = batch["pixel_values"].to(DEVICE)
            ib = batch.get("input_boxes").to(DEVICE) if batch.get("input_boxes") is not None else None
            gt = batch["ground_truth_mask"].to(DEVICE)

            # SAM 2 Forward Pass
            out = model(pixel_values=pv, input_boxes=ib, multimask_output=False)
            pm  = out.pred_masks.squeeze(1)

            # Metrics Computation
            test_loss += combined_loss(pm, gt).item()
            iou, dice, accuracy = compute_test_metrics(pm, gt)

            test_iou  += iou
            test_dice += dice
            test_acc  += accuracy

    # Compute averages across batches
    num_batches = len(test_loader)
    test_loss /= num_batches
    test_iou  /= num_batches
    test_dice /= num_batches
    test_acc  /= num_batches

    print("\n" + "=" * 60)
    print("FINAL SAM 2 + LORA (v5 ConvStem Cascade) TEST RESULTS")
    print("=" * 60)
    print(f"Test Accuracy : {test_acc * 100:.2f}%")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test IoU      : {test_iou:.4f}")
    print(f"Test Dice     : {test_dice:.4f}")
    print("=" * 60)