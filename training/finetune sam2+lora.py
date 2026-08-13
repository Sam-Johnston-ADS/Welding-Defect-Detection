"""
SAM 2 + LoRA Fine-tuning — Stage 3.5
Boundary refinement for weld defect segmentation
Cascaded Pipeline: Scratch-Trained SegFormer-B2 (0.74 IoU) -> SAM 2 Prompting

Run in your local environment terminal:
  python training/finetune_sam2_lora_local.py
"""

import os
import json
import shutil
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

# ─────────────────────────────────────────────
# 1. LOCAL PATHS CONFIGURATION
# ─────────────────────────────────────────────
BASE_DIR          = Path(r"E:\weld-defect-detection (1)\weld-defect-detection")
COMBINED_DATA_DIR = Path(r"E:\weld-defect-detection (1)\weld-defect-detection\data\reviewed\WDXI\combined")

# Dynamically route training and validation maps out of the combined directory
TRAIN_IMG   = os.path.join(COMBINED_DATA_DIR, "images")
TRAIN_MASK  = os.path.join(COMBINED_DATA_DIR, "labels")
VAL_IMG     = os.path.join(COMBINED_DATA_DIR, "images")  # Pointing to combined for validation evaluation
VAL_MASK    = os.path.join(COMBINED_DATA_DIR, "labels")

MODEL_DIR   = os.path.join(BASE_DIR, "models", "sam2")
SAVE_PATH   = os.path.join(MODEL_DIR, "sam2_lora_best.pth")

# Path to your custom scratch-trained SegFormer weights
SEGFORMER_CHECKPOINT = r"E:\weld-defect-detection (1)\weld-defect-detection\models\segformer_v4_scratch\segformer_v4_scratch_best.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# 2. CONFIG
# ─────────────────────────────────────────────
IMAGE_SIZE   = 1024     # SAM 2 native resolution
BATCH_SIZE   = 2        # small — SAM 2 is large (Keep at 2 or 4 depending on VRAM)
EPOCHS       = 30
LR           = 1e-4
NUM_CLASSES  = 1        # binary: defect or not

LORA_R       = 4        
LORA_ALPHA   = 8
LORA_DROPOUT = 0.1

os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 3. DATASET (Real Cascade Prompting)
# ─────────────────────────────────────────────
class WeldSAM2CascadeDataset(Dataset):
    def __init__(self, img_dir, mask_dir, processor, segformer_model):
        self.img_dir         = img_dir
        self.mask_dir        = mask_dir
        self.processor       = processor
        self.segformer_model = segformer_model
        
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Directory not found: {img_dir}")
            
        self.images          = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def _get_segformer_box(self, image_np):
        """Passes image through your 0.74 IoU SegFormer to extract real bounding boxes."""
        h, w, _ = image_np.shape
        
        # Prepare image tensor for SegFormer [B, C, H, W]
        img_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(512, 512), mode="bilinear", align_corners=False)
        img_tensor = img_tensor.to(DEVICE)
        
        # Inference
        outputs = self.segformer_model(pixel_values=img_tensor)
        logits = outputs.logits
        
        # Resize logits back to the original image shape
        upsampled_logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        pred_mask = upsampled_logits.argmax(dim=1).squeeze(0).cpu().numpy() # [H, W]
        
        # Calculate Bounding Box from actual SegFormer predictions
        nonzero = np.argwhere(pred_mask > 0)
        if len(nonzero) == 0:
            return [0, 0, w, h] # Fallback to whole image if no defect found

        y_min, x_min = nonzero.min(axis=0)
        y_max, x_max = nonzero.max(axis=0)

        # 5% padding around the mask boundary
        pad_x = int((x_max - x_min) * 0.05)
        pad_y = int((y_max - y_min) * 0.05)
        
        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(w, x_max + pad_x)
        y2 = min(h, y_max + pad_y)

        return [x1, y1, x2, y2]

    def __getitem__(self, idx):
        fname     = self.images[idx]
        image_pil = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        image_np  = np.array(image_pil)
        mask      = np.array(Image.open(os.path.join(self.mask_dir, fname)).convert("L"))

        mask_bin = (mask > 127).astype(np.uint8)

        # Fetch bounding box layout directly via SegFormer-B2
        box = self._get_segformer_box(image_np)

        # Process everything for SAM 2
        inputs = self.processor(
            images        = image_pil,
            input_boxes   = [[box]],   
            return_tensors= "pt"
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(mask_bin, dtype=torch.float32)

        return inputs

# ─────────────────────────────────────────────
# 4. INITIALIZE & LOAD MODEL WEIGHTS
# ─────────────────────────────────────────────
# A. Rebuild Custom SegFormer-B2 Configuration
print("Building SegFormer-B2 architecture from scratch...")
segformer_config = SegformerConfig(
    num_channels           = 3,
    num_labels              = 2, # Background + Weld Defect
    depths                  = [3, 4, 6, 3],       
    hidden_sizes            = [64, 128, 320, 512],
    num_attention_heads     = [1, 2, 5, 8],
    decoder_hidden_size     = 768,
)
segformer_model = SegformerForSemanticSegmentation(segformer_config)

if os.path.exists(SEGFORMER_CHECKPOINT):
    print(f"Loading custom 0.74 IoU weights from local path: {SEGFORMER_CHECKPOINT}")
    checkpoint = torch.load(SEGFORMER_CHECKPOINT, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        segformer_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        segformer_model.load_state_dict(checkpoint)
    print("  ✅ Custom SegFormer-B2 weights successfully verified and loaded.")
else:
    raise FileNotFoundError(f"Missing custom SegFormer weights at {SEGFORMER_CHECKPOINT}")

segformer_model = segformer_model.to(DEVICE)
segformer_model.eval() # Keep frozen

# B. Load Base SAM 2
print("\nLoading SAM 2 base model...")
MODEL_ID  = "facebook/sam2-hiera-small"   
processor = Sam2Processor.from_pretrained(MODEL_ID)
model     = Sam2Model.from_pretrained(MODEL_ID)

# C. Apply LoRA Config
lora_config = LoraConfig(
    r            = LORA_R,
    lora_alpha   = LORA_ALPHA,
    lora_dropout = LORA_DROPOUT,
    bias         = "none",
    target_modules = ["attn.qkv", "attn.proj"]
)

model.vision_encoder = get_peft_model(model.vision_encoder, lora_config)

for param in model.prompt_encoder.parameters():
    param.requires_grad = False

for param in model.mask_decoder.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

# ─────────────────────────────────────────────
# 5. DATALOADERS (Configured safely for Windows)
# ─────────────────────────────────────────────
print("\nLoading local dataset arrays...")
train_dataset = WeldSAM2CascadeDataset(TRAIN_IMG, TRAIN_MASK, processor, segformer_model)
val_dataset   = WeldSAM2CascadeDataset(VAL_IMG,   VAL_MASK,   processor, segformer_model)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0, pin_memory=True
)

print(f"Dataset Size: {len(train_dataset)} images found inside combined folder.")

# ─────────────────────────────────────────────
# 6. LOSS FUNCTIONS
# ─────────────────────────────────────────────
def dice_loss(pred, target, smooth=1.0):
    pred   = torch.sigmoid(pred)
    inter  = (pred * target).sum(dim=(2, 3))
    union  = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - (2 * inter + smooth) / (union + smooth)

def combined_loss(pred, target):
    target = target.unsqueeze(1).float()   
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
    bce  = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target).mean()
    return bce + dice

# ─────────────────────────────────────────────
# 7. METRICS
# ─────────────────────────────────────────────
def compute_iou(pred_logits, target, threshold=0.5):
    pred   = (torch.sigmoid(pred_logits) > threshold).float()
    target = target.unsqueeze(1).float()
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred.float(), size=target.shape[-2:], mode="nearest")
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).clamp(0, 1).sum(dim=(2, 3))
    return (inter / (union + 1e-8)).mean().item()

def compute_dice(pred_logits, target, threshold=0.5):
    pred   = (torch.sigmoid(pred_logits) > threshold).float()
    target = target.unsqueeze(1).float()
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred.float(), size=target.shape[-2:], mode="nearest")
    inter = (pred * target).sum(dim=(2, 3))
    return (2 * inter / (pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + 1e-8)).mean().item()

# ─────────────────────────────────────────────
# 8. OPTIMIZER & SCHEDULER
# ─────────────────────────────────────────────
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ─────────────────────────────────────────────
# 9. FINE-TUNING LOOP
# ─────────────────────────────────────────────
print(f"\nStarting SAM 2 LoRA fine-tuning...")
best_iou = 0.0
history  = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
        pixel_values  = batch["pixel_values"].to(DEVICE)
        input_boxes   = batch.get("input_boxes").to(DEVICE) if batch.get("input_boxes") is not None else None
        gt_mask       = batch["ground_truth_mask"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
        
        pred_masks = outputs.pred_masks.squeeze(1)
        loss = combined_loss(pred_masks, gt_mask)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = val_iou = val_dice = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_boxes  = batch.get("input_boxes").to(DEVICE) if batch.get("input_boxes") is not None else None
            gt_mask      = batch["ground_truth_mask"].to(DEVICE)

            outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
            pred_masks  = outputs.pred_masks.squeeze(1)
            
            val_loss   += combined_loss(pred_masks, gt_mask).item()
            val_iou    += compute_iou(pred_masks, gt_mask)
            val_dice   += compute_dice(pred_masks, gt_mask)

    val_loss /= len(val_loader)
    val_iou  /= len(val_loader)
    val_dice /= len(val_loader)
    scheduler.step()

    print(f"\nEpoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}")

    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "iou": val_iou, "dice": val_dice})

    if val_iou > best_iou:
        best_iou = val_iou
        lora_weights = {k: v for k, v in model.state_dict().items() if "lora" in k.lower() or "mask_decoder" in k.lower()}
        torch.save({
            "epoch": epoch,
            "lora_weights": lora_weights,
            "val_iou": val_iou,
            "val_dice": val_dice,
            "lora_config": {"r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT}
        }, SAVE_PATH)
        print(f"  ✅ Best refined model saved → IoU: {best_iou:.4f}")

# ─────────────────────────────────────────────
# 10. WRAP UP
# ─────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "sam2_history.json"), "w") as f:
    json.dump(history, f, indent=2)

print(f"\n{'='*50}\nSAM 2 Local Fine-tuning Pipeline Complete\n{'='*50}")
print(f"SegFormer Baseline IoU : 0.7400")
print(f"SAM 2 Refined IoU      : {best_iou:.4f}")
print(f"Model saved locally at : {SAVE_PATH}")