import os
import json
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)
import torch.nn.functional as F

# ─────────────────────────────────────────────
# 1. PATHS
# ─────────────────────────────────────────────
BASE_DIR     = "/content/weld-defect-detection"
TRAIN_IMG    = f"{BASE_DIR}/data/reviewed/WDXI/datasets1/images"
TRAIN_MASK   = f"{BASE_DIR}/data/reviewed/WDXI/datasets1/labels"
VAL_IMG      = f"{BASE_DIR}/data/reviewed/WDXI/datasets2/images"
VAL_MASK     = f"{BASE_DIR}/data/reviewed/WDXI/datasets2/labels"
MODEL_DIR    = f"{BASE_DIR}/models/unet"          # replacing unet slot
DRIVE_SAVE   = "/content/drive/MyDrive/segformer_best.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# 2. CONFIG
# ─────────────────────────────────────────────
IMAGE_SIZE  = 512        # SegFormer works better at 512 than 256
BATCH_SIZE  = 4          # smaller than U-Net due to transformer memory
EPOCHS      = 50
LR          = 6e-5       # SegFormer uses lower LR than CNN models
NUM_CLASSES = 2          # 0 = background, 1 = defect

# ─────────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────────
class WeldSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.images    = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname     = self.images[idx]
        img_path  = os.path.join(self.img_dir,  fname)
        mask_path = os.path.join(self.mask_dir, fname)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path).convert("L"))

        # binary mask: white=defect=1, black=background=0
        mask = (mask > 127).astype(np.int64)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image     = augmented["image"]
            mask      = augmented["mask"].long()

        return image, mask

# ─────────────────────────────────────────────
# 4. AUGMENTATIONS
# ─────────────────────────────────────────────
train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=15, p=0.4),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    A.ElasticTransform(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std =(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std =(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ─────────────────────────────────────────────
# 5. DATALOADERS
# ─────────────────────────────────────────────
train_dataset = WeldSegDataset(TRAIN_IMG, TRAIN_MASK, train_transform)
val_dataset   = WeldSegDataset(VAL_IMG,   VAL_MASK,   val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

print(f"Train images : {len(train_dataset)}")
print(f"Val   images : {len(val_dataset)}")

# ─────────────────────────────────────────────
# 6. MODEL — SegFormer-B2
# B0 = lightest, B5 = heaviest
# B2 = best balance of speed and accuracy
# for industrial weld segmentation
# ─────────────────────────────────────────────
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b2",
    num_labels          = NUM_CLASSES,
    id2label            = {0: "background", 1: "defect"},
    label2id            = {"background": 0, "defect": 1},
    ignore_mismatched_sizes = True,
)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: SegFormer-B2")
print(f"Parameters: {total_params:,}")

# ─────────────────────────────────────────────
# 7. LOSS FUNCTION
# Combined: CrossEntropy + Dice
# handles class imbalance (more background than defect)
# ─────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B, C, H, W)  targets: (B, H, W)
        probs   = F.softmax(logits, dim=1)
        defect  = probs[:, 1, :, :]          # defect channel
        targets = targets.float()
        inter   = (defect * targets).sum()
        union   = defect.sum() + targets.sum()
        return 1 - (2 * inter + self.smooth) / (union + self.smooth)

ce_loss   = nn.CrossEntropyLoss()
dice_loss = DiceLoss()

def combined_loss(logits, targets):
    # upsample logits to match target size
    logits_up = F.interpolate(
        logits, size=targets.shape[-2:],
        mode="bilinear", align_corners=False
    )
    return ce_loss(logits_up, targets) + dice_loss(logits_up, targets)

# ─────────────────────────────────────────────
# 8. OPTIMIZER + SCHEDULER
# ─────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=0.01
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

# ─────────────────────────────────────────────
# 9. METRICS
# ─────────────────────────────────────────────
def compute_metrics(logits, targets, threshold=0.5):
    logits_up = F.interpolate(
        logits, size=targets.shape[-2:],
        mode="bilinear", align_corners=False
    )
    probs  = F.softmax(logits_up, dim=1)[:, 1, :, :]  # defect prob
    preds  = (probs > threshold).long()
    target = targets.long()

    inter  = (preds * target).sum().float()
    union  = (preds + target).clamp(0, 1).sum().float()
    iou    = (inter / (union + 1e-8)).item()

    dice   = (2 * inter / (preds.sum() + target.sum() + 1e-8)).item()

    return iou, dice

# ─────────────────────────────────────────────
# 10. TRAINING LOOP
# ─────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
best_iou = 0.0
history  = []

print("\nStarting SegFormer training...")

for epoch in range(1, EPOCHS + 1):

    # ── train ──
    model.train()
    train_loss = 0.0

    for images, masks in tqdm(train_loader,
                              desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
        images = images.to(DEVICE)
        masks  = masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(pixel_values=images)
        logits  = outputs.logits          # (B, num_labels, H/4, W/4)

        loss = combined_loss(logits, masks)
        loss.backward()

        # gradient clipping — important for transformer training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ── validate ──
    model.eval()
    val_loss = val_iou = val_dice = 0.0

    with torch.no_grad():
        for images, masks in tqdm(val_loader,
                                  desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)

            outputs     = model(pixel_values=images)
            logits      = outputs.logits
            val_loss   += combined_loss(logits, masks).item()
            iou, dice   = compute_metrics(logits, masks)
            val_iou    += iou
            val_dice   += dice

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

    # save best model
    if val_iou > best_iou:
        best_iou = val_iou
        torch.save({
            "epoch"            : epoch,
            "model_state_dict" : model.state_dict(),
            "val_iou"          : val_iou,
            "val_dice"         : val_dice,
        }, os.path.join(MODEL_DIR, "segformer_best.pth"))
        print(f"  ✅ Best model saved → IoU: {best_iou:.4f}")

# save history
with open(os.path.join(MODEL_DIR, "history.json"), "w") as f:
    json.dump(history, f, indent=2)

print(f"\nTraining complete. Best IoU: {best_iou:.4f}")
print(f"U-Net target was 0.65 — SegFormer should exceed this.")

# ─────────────────────────────────────────────
# 11. SAVE TO GOOGLE DRIVE
# ─────────────────────────────────────────────
shutil.copy(
    os.path.join(MODEL_DIR, "segformer_best.pth"),
    DRIVE_SAVE
)
print(f"\nModel saved to Google Drive: {DRIVE_SAVE}")
