"""
SegFormer-B2 v5 — Conv Stem + From Scratch
Adds a small convolutional stem (3 conv-bn-relu layers) BEFORE
SegFormer's own patch embedding, following:
  Xiao et al., "Early Convolutions Help Transformers See Better" (2021)

Why this helps specifically for your case: transformers have no
built-in spatial inductive bias (locality/translation-invariance),
which is normally learned from large-scale pretraining. Since your
mentor requires NO pretrained weights, the model has to learn that
bias purely from ~2,800 training images. A conv stem gives the
transformer richer, locally-aware features to work with from the
very first layer, which measurably speeds up and improves from-
scratch convergence in the paper's own small-data experiments.

Everything else (data, split, losses, metrics) is identical to v4
so the comparison is a clean, single-variable test: does the conv
stem help, holding data/training regime constant?

Run on Kaggle GPU / Terminal:
  python training/train_segformer_v5_convstem.py
"""

import os
import json
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation, SegformerConfig

# ==========================================================
# PROJECT ROOT & PATHS
# ==========================================================
BASE_DIR = r"E:\weld-defect-detection (1)\weld-defect-detection"

COMBINED_IMG = os.path.join(
    BASE_DIR, "data", "reviewed", "WDXI", "combined", "images"
)
COMBINED_MSK = os.path.join(
    BASE_DIR, "data", "reviewed", "WDXI", "combined", "labels"
)
SPLIT_PATH = os.path.join(
    BASE_DIR, "data", "reviewed", "WDXI", "combined", "split.json"
)

MODEL_DIR = os.path.join(BASE_DIR, "models", "segformer_v5_convstem")
SAVE_PATH = os.path.join(MODEL_DIR, "segformer_v5_convstem_best.pth")
DRIVE_SAVE = SAVE_PATH

# ==========================================================
# HYPERPARAMETERS
# ==========================================================
IMAGE_SIZE    = 512
BATCH_SIZE    = 4
EPOCHS        = 100          # same budget as v4 for a fair comparison
LR            = 3e-4
NUM_CLASSES   = 2
STEM_CHANNELS = 32           # channels the conv stem hands off to SegFormer's patch embed

# ==========================================================
# DATASET & AUGMENTATIONS
# ==========================================================
class CombinedSegDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        image = np.array(Image.open(os.path.join(COMBINED_IMG, fname)).convert("RGB"))
        mask  = np.array(Image.open(os.path.join(COMBINED_MSK, fname)).convert("L"))
        mask  = (mask > 127).astype(np.int64)
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"].long()
        return image, mask

train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.5, 1.0), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=20, p=0.4),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(p=0.3),
    A.ElasticTransform(alpha=120, sigma=6, p=0.3),
    A.GridDistortion(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ==========================================================
# MODEL ARCHITECTURE
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
    """
    Conv stem -> SegFormer (patch-embeds the stem's output instead
    of raw RGB). SegFormer's config.num_channels is set to
    STEM_CHANNELS to match what the stem hands off.
    """
    def __init__(self, segformer_config):
        super().__init__()
        self.stem      = ConvStem(in_channels=3, out_channels=STEM_CHANNELS)
        self.segformer = SegformerForSemanticSegmentation(segformer_config)

    def forward(self, pixel_values):
        stem_features = self.stem(pixel_values)
        return self.segformer(pixel_values=stem_features).logits

# ==========================================================
# HELPER FUNCTIONS & LOSSES
# ==========================================================
def compute_class_weights(file_list, device, sample=300):
    bg = defect = 0
    for fname in file_list[:sample]:
        m = np.array(Image.open(os.path.join(COMBINED_MSK, fname)).convert("L"))
        d = (m > 127).sum()
        defect += d
        bg += m.size - d
    total = bg + defect
    return torch.tensor([total / (2 * bg), total / (2 * defect)], dtype=torch.float32).to(device)

def dice_loss(logits, targets, w=3.0, smooth=1.0):
    probs = F.softmax(logits, dim=1)[:, 1, :, :]
    targets = targets.float()
    inter = (probs * targets).sum()
    union = probs.sum() + targets.sum()
    return w * (1 - (2 * inter + smooth) / (union + smooth))

def combined_loss(logits, targets, ce_loss_fn):
    logits_up = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    return ce_loss_fn(logits_up, targets) + dice_loss(logits_up, targets)

def compute_metrics(logits, targets):
    logits_up = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    preds = (F.softmax(logits_up, dim=1)[:, 1, :, :] > 0.5).long()
    target = targets.long()
    inter = (preds * target).sum().float()
    union = (preds + target).clamp(0, 1).sum().float()
    iou = (inter / (union + 1e-8)).item()
    dice = (2 * inter / (preds.sum() + target.sum() + 1e-8)).item()
    return iou, dice

# ==========================================================
# MAIN EXECUTION
# ==========================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(SPLIT_PATH) as f:
        split = json.load(f)
    print(f"Split — train:{len(split['train'])} val:{len(split['val'])} test:{len(split['test'])}")

    class_weights = compute_class_weights(split["train"], device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(
        CombinedSegDataset(split["train"], train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        CombinedSegDataset(split["val"], val_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print("Building SegFormer-B2 + Conv Stem (from scratch, no pretrained weights)...")
    segformer_config = SegformerConfig(
        num_channels=STEM_CHANNELS,          # takes stem output, not raw RGB
        num_labels=NUM_CLASSES,
        depths=[3, 4, 6, 3],
        hidden_sizes=[64, 128, 320, 512],
        num_attention_heads=[1, 2, 5, 8],
        decoder_hidden_size=768,
        id2label={0: "background", 1: "defect"},
        label2id={"background": 0, "defect": 1},
    )
    model = SegformerWithConvStem(segformer_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    stem_params  = sum(p.numel() for p in model.stem.parameters())
    print(f"Total parameters : {total_params:,}")
    print(f"Conv stem params : {stem_params:,} (small addition on top of v4)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_iou = 0.0
    history  = []
    print(f"\nTraining {EPOCHS} epochs, from scratch, conv stem enabled\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = combined_loss(model(images), masks, ce_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = val_iou = val_dice = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                val_loss += combined_loss(logits, masks, ce_loss).item()
                iou, dice = compute_metrics(logits, masks)
                val_iou += iou
                val_dice += dice
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)
        scheduler.step()

        print(f"\nEpoch {epoch:02d} | Train Loss:{train_loss:.4f} | "
              f"Val Loss:{val_loss:.4f} | IoU:{val_iou:.4f} | Dice:{val_dice:.4f}")
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "iou": val_iou,
            "dice": val_dice
        })

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_iou": val_iou,
                "val_dice": val_dice,
                "stem_channels": STEM_CHANNELS
            }, SAVE_PATH)
            print(f"  Best model saved -> IoU: {best_iou:.4f}")

    with open(os.path.join(MODEL_DIR, "history_v5.json"), "w") as f:
        json.dump(history, f, indent=2)

    if SAVE_PATH != DRIVE_SAVE:
        shutil.copy(SAVE_PATH, DRIVE_SAVE)

    print(f"\n{'='*50}")
    print("SegFormer v5 (Conv Stem, from scratch) complete")
    print(f"{'='*50}")
    print(f"Best IoU (v5, conv stem)     : {best_iou:.4f}")
    print("v4 (from scratch, no stem)   : 0.7761 (val) / ~0.78 (test)")
    print(f"Improvement from conv stem   : {(best_iou - 0.7761):.4f}")
    print("NOTE: run the held-out test-split evaluation before trusting this —")
    print("      same leakage-verification discipline as every prior model.")

if __name__ == "__main__":
    main()