import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ─────────────────────────────────────────────
# 1. PATHS  ← change these in Colab
# ─────────────────────────────────────────────
TRAIN_IMG_DIR  = "data/reviewed/WDXI/datasets1/images"
TRAIN_MASK_DIR = "data/reviewed/WDXI/datasets1/labels"
VAL_IMG_DIR    = "data/reviewed/WDXI/datasets2/images"
VAL_MASK_DIR   = "data/reviewed/WDXI/datasets2/labels"
CHECKPOINT_DIR = "models/unet"

# ─────────────────────────────────────────────
# 2. CONFIG
# ─────────────────────────────────────────────
IMAGE_SIZE  = 256        # resize all images to 256x256
BATCH_SIZE  = 8
EPOCHS      = 50
LR          = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────────
class WeldDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.images    = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        img_path  = os.path.join(self.img_dir,  img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path).convert("L"))

        # convert mask to binary: white=defect=1, black=background=0
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"].unsqueeze(0)  # add channel dim

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
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ─────────────────────────────────────────────
# 5. DATALOADERS
# ─────────────────────────────────────────────
train_dataset = WeldDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
val_dataset   = WeldDataset(VAL_IMG_DIR,   VAL_MASK_DIR,   transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train images : {len(train_dataset)}")
print(f"Val   images : {len(val_dataset)}")

# ─────────────────────────────────────────────
# 6. MODEL  (U-Net with ResNet34 backbone)
# ─────────────────────────────────────────────
model = smp.Unet(
    encoder_name    = "resnet34",
    encoder_weights = "imagenet",
    in_channels     = 3,
    classes         = 1,
    activation      = None,
).to(DEVICE)

# ─────────────────────────────────────────────
# 7. LOSS + OPTIMIZER
# ─────────────────────────────────────────────
# DiceLoss + BCEWithLogitsLoss combined → handles class imbalance better
dice_loss = smp.losses.DiceLoss(mode="binary")
bce_loss  = nn.BCEWithLogitsLoss()

def combined_loss(pred, target):
    return dice_loss(pred, target) + bce_loss(pred, target)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5
)

# ─────────────────────────────────────────────
# 8. METRICS
# ─────────────────────────────────────────────
def compute_iou(pred, target, threshold=0.5):
    pred   = (torch.sigmoid(pred) > threshold).float()
    inter  = (pred * target).sum()
    union  = pred.sum() + target.sum() - inter
    return (inter / (union + 1e-8)).item()

def compute_dice(pred, target, threshold=0.5):
    pred  = (torch.sigmoid(pred) > threshold).float()
    inter = (pred * target).sum()
    return (2 * inter / (pred.sum() + target.sum() + 1e-8)).item()

# ─────────────────────────────────────────────
# 9. TRAINING LOOP
# ─────────────────────────────────────────────
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
best_iou = 0.0

for epoch in range(1, EPOCHS + 1):

    # ── train ──
    model.train()
    train_loss = 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        preds = model(images)
        loss  = combined_loss(preds, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ── validate ──
    model.eval()
    val_loss = val_iou = val_dice = 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            preds     = model(images)
            val_loss += combined_loss(preds, masks).item()
            val_iou  += compute_iou(preds, masks)
            val_dice += compute_dice(preds, masks)

    val_loss /= len(val_loader)
    val_iou  /= len(val_loader)
    val_dice /= len(val_loader)

    scheduler.step(val_loss)

    print(f"\nEpoch {epoch:02d} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"IoU: {val_iou:.4f} | "
          f"Dice: {val_dice:.4f}")

    # ── save best model ──
    if val_iou > best_iou:
        best_iou = val_iou
        save_path = os.path.join(CHECKPOINT_DIR, "unet_best.pth")
        torch.save(model.state_dict(), save_path)
        print(f"  ✅ Best model saved → IoU: {best_iou:.4f}")

print(f"\nTraining complete. Best IoU: {best_iou:.4f}")
print(f"Model saved at: {CHECKPOINT_DIR}/unet_best.pth")