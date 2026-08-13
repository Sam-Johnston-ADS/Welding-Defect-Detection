"""
SAM 2 + LoRA Fine-tuning — Stage 3.5
Boundary refinement for weld defect segmentation
Following: Naddaf-Sh et al. (2025) Sensors MDPI
License: Meta SAM2 Apache 2.0

Run on Kaggle GPU:
  !pip install transformers peft accelerate -q
  !python training/finetune_sam2_lora.py
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

from transformers import Sam2Processor, Sam2Model
from peft import LoraConfig, get_peft_model, TaskType

# ─────────────────────────────────────────────
# 1. PATHS
# ─────────────────────────────────────────────
BASE_DIR    = "/kaggle/working/weld-defect-detection"
TRAIN_IMG   = f"{BASE_DIR}/data/reviewed/WDXI/datasets1/images"
TRAIN_MASK  = f"{BASE_DIR}/data/reviewed/WDXI/datasets1/labels"
VAL_IMG     = f"{BASE_DIR}/data/reviewed/WDXI/datasets2/images"
VAL_MASK    = f"{BASE_DIR}/data/reviewed/WDXI/datasets2/labels"
MODEL_DIR   = f"{BASE_DIR}/models/sam2"
SAVE_PATH   = f"{BASE_DIR}/models/sam2/sam2_lora_best.pth"
DRIVE_SAVE  = "/kaggle/working/sam2_lora_best.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# 2. CONFIG
# ─────────────────────────────────────────────
IMAGE_SIZE   = 1024     # SAM 2 native resolution
BATCH_SIZE   = 2        # small — SAM 2 is large
EPOCHS       = 30
LR           = 1e-4
NUM_CLASSES  = 1        # binary: defect or not

# LoRA config — following Naddaf-Sh et al. 2025
# Apply LoRA to image encoder + mask decoder
# Freeze prompt encoder (as in the paper)
LORA_R       = 4        # rank
LORA_ALPHA   = 8
LORA_DROPOUT = 0.1

os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 3. DATASET
# Prompting strategy: use bounding box from
# SegFormer mask as the prompt to SAM 2
# This follows the cascade design:
#   SegFormer mask → SAM 2 box prompt → refined mask
# ─────────────────────────────────────────────
class WeldSAM2Dataset(Dataset):
    def __init__(self, img_dir, mask_dir, processor):
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.processor = processor
        self.images    = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname     = self.images[idx]
        image     = np.array(Image.open(
            os.path.join(self.img_dir, fname)).convert("RGB"))
        mask      = np.array(Image.open(
            os.path.join(self.mask_dir, fname)).convert("L"))

        # binary mask
        mask_bin = (mask > 127).astype(np.uint8)

        # derive bounding box prompt from mask
        # (simulates what SegFormer would output in Stage 3)
        box = self._mask_to_box(mask_bin)

        # process with SAM 2 processor
        inputs = self.processor(
            images        = Image.fromarray(image),
            input_boxes   = [[box]],   # SAM 2 expects list of list
            return_tensors= "pt"
        )

        # squeeze batch dim added by processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(
            mask_bin, dtype=torch.float32
        )

        return inputs

    def _mask_to_box(self, mask):
        """Convert binary mask to [x1, y1, x2, y2] bounding box."""
        h, w    = mask.shape
        nonzero = np.argwhere(mask > 0)

        if len(nonzero) == 0:
            # no defect — return full image box
            return [0, 0, w, h]

        y_min, x_min = nonzero.min(axis=0)
        y_max, x_max = nonzero.max(axis=0)

        # add 5% padding
        pad_x = int((x_max - x_min) * 0.05)
        pad_y = int((y_max - y_min) * 0.05)
        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(w, x_max + pad_x)
        y2 = min(h, y_max + pad_y)

        return [x1, y1, x2, y2]


# ─────────────────────────────────────────────
# 4. LOAD SAM 2 + APPLY LORA
# ─────────────────────────────────────────────
print("Loading SAM 2 base model...")
MODEL_ID  = "facebook/sam2-hiera-small"   # smallest SAM 2 variant

processor = Sam2Processor.from_pretrained(MODEL_ID)
model     = Sam2Model.from_pretrained(MODEL_ID)

# ── apply LoRA to image encoder only ──
# (following Naddaf-Sh et al. 2025 methodology)
# Prompt encoder is FROZEN
# Mask decoder is trained normally (not LoRA)
lora_config = LoraConfig(
    r            = LORA_R,
    lora_alpha   = LORA_ALPHA,
    lora_dropout = LORA_DROPOUT,
    bias         = "none",
    target_modules = [
        # target attention projection layers in image encoder
        "attn.qkv",
        "attn.proj",
    ]
)

# apply LoRA to image encoder
model.vision_encoder = get_peft_model(model.vision_encoder, lora_config)
model.vision_encoder.print_trainable_parameters()

# freeze prompt encoder (as in the paper)
for param in model.prompt_encoder.parameters():
    param.requires_grad = False

# mask decoder trains normally
for param in model.mask_decoder.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params    : {total_params:,}")
print(f"Trainable params: {trainable_params:,} "
      f"({100*trainable_params/total_params:.2f}%)")

# ─────────────────────────────────────────────
# 5. DATALOADERS
# ─────────────────────────────────────────────
print("\nLoading datasets...")
train_dataset = WeldSAM2Dataset(TRAIN_IMG, TRAIN_MASK, processor)
val_dataset   = WeldSAM2Dataset(VAL_IMG,   VAL_MASK,   processor)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=2, pin_memory=True
)

print(f"Train: {len(train_dataset)} images")
print(f"Val  : {len(val_dataset)} images")

# ─────────────────────────────────────────────
# 6. LOSS
# Dice + BCE combined — same as Naddaf-Sh et al.
# ─────────────────────────────────────────────
def dice_loss(pred, target, smooth=1.0):
    pred   = torch.sigmoid(pred)
    inter  = (pred * target).sum(dim=(2, 3))
    union  = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - (2 * inter + smooth) / (union + smooth)

def combined_loss(pred, target):
    target = target.unsqueeze(1).float()   # (B,1,H,W)

    # resize pred to match target if needed
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(
            pred, size=target.shape[-2:],
            mode="bilinear", align_corners=False
        )

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
        pred = F.interpolate(pred.float(), size=target.shape[-2:],
                             mode="nearest")

    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).clamp(0, 1).sum(dim=(2, 3))
    iou   = (inter / (union + 1e-8)).mean()
    return iou.item()

def compute_dice(pred_logits, target, threshold=0.5):
    pred   = (torch.sigmoid(pred_logits) > threshold).float()
    target = target.unsqueeze(1).float()

    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred.float(), size=target.shape[-2:],
                             mode="nearest")

    inter = (pred * target).sum(dim=(2, 3))
    dice  = (2 * inter / (pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + 1e-8)).mean()
    return dice.item()

# ─────────────────────────────────────────────
# 8. OPTIMIZER
# ─────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=0.01
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

# ─────────────────────────────────────────────
# 9. TRAINING LOOP
# ─────────────────────────────────────────────
print(f"\nStarting SAM 2 LoRA fine-tuning...")
print(f"Epochs  : {EPOCHS}")
print(f"LoRA r  : {LORA_R}")
print(f"Device  : {DEVICE}\n")

best_iou = 0.0
history  = []

for epoch in range(1, EPOCHS + 1):

    # ── train ──
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader,
                      desc=f"Epoch {epoch}/{EPOCHS} [Train]"):

        pixel_values  = batch["pixel_values"].to(DEVICE)
        input_boxes   = batch.get("input_boxes")
        gt_mask       = batch["ground_truth_mask"].to(DEVICE)

        if input_boxes is not None:
            input_boxes = input_boxes.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            pixel_values = pixel_values,
            input_boxes  = input_boxes,
            multimask_output = False,
        )

        # pred_masks shape: (B, 1, H, W)
        pred_masks = outputs.pred_masks.squeeze(1)
        loss = combined_loss(pred_masks, gt_mask)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ── validate ──
    model.eval()
    val_loss = val_iou = val_dice = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader,
                          desc=f"Epoch {epoch}/{EPOCHS} [Val]"):

            pixel_values = batch["pixel_values"].to(DEVICE)
            input_boxes  = batch.get("input_boxes")
            gt_mask      = batch["ground_truth_mask"].to(DEVICE)

            if input_boxes is not None:
                input_boxes = input_boxes.to(DEVICE)

            outputs = model(
                pixel_values     = pixel_values,
                input_boxes      = input_boxes,
                multimask_output = False,
            )

            pred_masks  = outputs.pred_masks.squeeze(1)
            val_loss   += combined_loss(pred_masks, gt_mask).item()
            val_iou    += compute_iou(pred_masks, gt_mask)
            val_dice   += compute_dice(pred_masks, gt_mask)

    val_loss /= len(val_loader)
    val_iou  /= len(val_loader)
    val_dice /= len(val_loader)

    scheduler.step()

    print(f"\nEpoch {epoch:02d} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"IoU: {val_iou:.4f} | "
          f"Dice: {val_dice:.4f}")

    history.append({
        "epoch"     : epoch,
        "train_loss": train_loss,
        "val_loss"  : val_loss,
        "iou"       : val_iou,
        "dice"      : val_dice,
    })

    if val_iou > best_iou:
        best_iou = val_iou

        # save LoRA weights only (much smaller than full model)
        lora_weights = {
            k: v for k, v in model.state_dict().items()
            if "lora" in k.lower() or "mask_decoder" in k.lower()
        }
        torch.save({
            "epoch"      : epoch,
            "lora_weights": lora_weights,
            "val_iou"    : val_iou,
            "val_dice"   : val_dice,
            "lora_config": {
                "r"           : LORA_R,
                "lora_alpha"  : LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
            }
        }, SAVE_PATH)
        print(f"  ✅ Best model saved → IoU: {best_iou:.4f}")

# ─────────────────────────────────────────────
# 10. SAVE HISTORY + COPY TO OUTPUT
# ─────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "sam2_history.json"), "w") as f:
    json.dump(history, f, indent=2)

shutil.copy(SAVE_PATH, DRIVE_SAVE)

print(f"\n{'='*50}")
print(f"SAM 2 LoRA Fine-tuning Complete")
print(f"{'='*50}")
print(f"Best IoU        : {best_iou:.4f}")
print(f"SegFormer IoU   : 0.6085")
print(f"Improvement     : {((best_iou - 0.6085)/0.6085*100):.1f}%")
print(f"Target          : ≥ 0.70")
print(f"Model saved     : {DRIVE_SAVE}")
print(f"Download from Kaggle output panel on the right.")
