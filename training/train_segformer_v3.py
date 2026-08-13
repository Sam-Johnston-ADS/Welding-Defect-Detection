"""
SegFormer-B2 training — Combined Dataset Version (v3)
Trains on the combined WDXI pool (datasets1 + datasets2) using
the reproducible split from prepare_combined_dataset.py.

Run on Kaggle GPU:
  !python training/train_segformer_v3.py
"""

import os, json, shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR    = "/kaggle/working/weld-defect-detection"
COMBINED_IMG = f"{BASE_DIR}/data/reviewed/WDXI/combined/images"
COMBINED_MSK = f"{BASE_DIR}/data/reviewed/WDXI/combined/labels"
SPLIT_PATH   = f"{BASE_DIR}/data/reviewed/WDXI/combined/split.json"
MODEL_DIR    = f"{BASE_DIR}/models/segformer_v3"
SAVE_PATH    = f"{MODEL_DIR}/segformer_v3_best.pth"
DRIVE_SAVE   = "/kaggle/working/segformer_v3_best.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

IMAGE_SIZE    = 512
BATCH_SIZE    = 4
EPOCHS        = 60
LR_HEAD       = 1e-4
LR_FULL       = 6e-5
FREEZE_EPOCHS = 10
NUM_CLASSES   = 2

os.makedirs(MODEL_DIR, exist_ok=True)

with open(SPLIT_PATH) as f:
    split = json.load(f)
print(f"Split loaded — train:{len(split['train'])} "
      f"val:{len(split['val'])} test:{len(split['test'])}")

# ─────────────────────────────────────────────
# CLASS WEIGHTS — computed from TRAIN split only
# ─────────────────────────────────────────────
def compute_class_weights(file_list, sample=300):
    bg = defect = 0
    for fname in file_list[:sample]:
        m = np.array(Image.open(os.path.join(COMBINED_MSK, fname)).convert("L"))
        d = (m > 127).sum()
        defect += d
        bg     += m.size - d
    total = bg + defect
    return torch.tensor([total/(2*bg), total/(2*defect)],
                        dtype=torch.float32).to(DEVICE)

class_weights = compute_class_weights(split["train"])
print(f"Class weights: bg={class_weights[0]:.4f} defect={class_weights[1]:.4f}")

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class CombinedSegDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.files     = file_list
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        image = np.array(Image.open(os.path.join(COMBINED_IMG, fname)).convert("RGB"))
        mask  = np.array(Image.open(os.path.join(COMBINED_MSK, fname)).convert("L"))
        mask  = (mask > 127).astype(np.int64)

        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask  = aug["mask"].long()
        return image, mask

train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.5, 1.0), p=0.5),
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3), A.Rotate(limit=20, p=0.4),
    A.RandomBrightnessContrast(p=0.4), A.GaussNoise(p=0.3),
    A.ElasticTransform(alpha=120, sigma=6, p=0.3), A.GridDistortion(p=0.2),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

train_dataset = CombinedSegDataset(split["train"], train_transform)
val_dataset   = CombinedSegDataset(split["val"],   val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b2", num_labels=NUM_CLASSES,
    id2label={0:"background",1:"defect"}, label2id={"background":0,"defect":1},
    ignore_mismatched_sizes=True,
).to(DEVICE)

def freeze_backbone(m):
    for n, p in m.named_parameters():
        if "segformer.encoder" in n: p.requires_grad = False
def unfreeze_all(m):
    for p in m.parameters(): p.requires_grad = True

freeze_backbone(model)

ce_loss = nn.CrossEntropyLoss(weight=class_weights)
def dice_loss(logits, targets, w=3.0, smooth=1.0):
    probs  = F.softmax(logits, dim=1)[:,1,:,:]
    targets = targets.float()
    inter  = (probs*targets).sum(); union = probs.sum()+targets.sum()
    return w * (1 - (2*inter+smooth)/(union+smooth))

def combined_loss(logits, targets):
    logits_up = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    return ce_loss(logits_up, targets) + dice_loss(logits_up, targets)

def compute_metrics(logits, targets):
    logits_up = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    preds  = (F.softmax(logits_up, dim=1)[:,1,:,:] > 0.5).long()
    target = targets.long()
    inter  = (preds*target).sum().float(); union = (preds+target).clamp(0,1).sum().float()
    iou    = (inter/(union+1e-8)).item()
    dice   = (2*inter/(preds.sum()+target.sum()+1e-8)).item()
    return iou, dice

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=LR_HEAD, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_iou = 0.0
history  = []

print(f"\nTraining {EPOCHS} epochs, freeze backbone first {FREEZE_EPOCHS}\n")

for epoch in range(1, EPOCHS+1):
    if epoch == FREEZE_EPOCHS+1:
        print("\n>>> Unfreezing full model <<<")
        unfreeze_all(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FULL, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS-FREEZE_EPOCHS, eta_min=1e-6)

    model.train(); train_loss = 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        loss = combined_loss(model(pixel_values=images).logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval(); val_loss = val_iou = val_dice = 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            logits = model(pixel_values=images).logits
            val_loss += combined_loss(logits, masks).item()
            iou, dice = compute_metrics(logits, masks)
            val_iou += iou; val_dice += dice
    val_loss /= len(val_loader); val_iou /= len(val_loader); val_dice /= len(val_loader)
    scheduler.step()

    print(f"\nEpoch {epoch:02d} | Train Loss:{train_loss:.4f} | "
          f"Val Loss:{val_loss:.4f} | IoU:{val_iou:.4f} | Dice:{val_dice:.4f}")
    history.append({"epoch":epoch,"train_loss":train_loss,"val_loss":val_loss,
                    "iou":val_iou,"dice":val_dice})

    if val_iou > best_iou:
        best_iou = val_iou
        torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),
                    "val_iou":val_iou,"val_dice":val_dice}, SAVE_PATH)
        print(f"  Best model saved -> IoU: {best_iou:.4f}")

with open(f"{MODEL_DIR}/history_v3.json","w") as f:
    json.dump(history, f, indent=2)
shutil.copy(SAVE_PATH, DRIVE_SAVE)

print(f"\nSegFormer v3 (combined data) complete. Best IoU: {best_iou:.4f}")
print(f"Previous (datasets1->train, datasets2->val): 0.6085")
print(f"Model saved: {DRIVE_SAVE}")
print(f"NOTE: 'test' split in split.json was NOT used here — held out for final reporting.")
