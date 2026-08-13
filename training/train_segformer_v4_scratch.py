import os, json, shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation, SegformerConfig

# 1. Let Python automatically hunt down split.json inside Kaggle's input folder
# ==========================================================
# WINDOWS PATHS
# ==========================================================

# ==========================================================
# WINDOWS PATHS
# ==========================================================
# ==========================================================
# WINDOWS PATHS
# ==========================================================

BASE_DIR = r"E:\weld-defect-detection (1)\weld-defect-detection"

COMBINED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "reviewed",
    "WDXI",
    "combined"
)

COMBINED_IMG = os.path.join(COMBINED_DIR, "images")
COMBINED_MSK = os.path.join(COMBINED_DIR, "labels")

SPLIT_PATH = os.path.join(
    COMBINED_DIR,
    "split.json"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "segformer_v4_scratch"
)

SAVE_PATH = os.path.join(
    MODEL_DIR,
    "segformer_v4_scratch_best.pth"
)

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Dataset : {COMBINED_DIR}")
print(f"Split   : {SPLIT_PATH}")
print("=" * 60)
print("PyTorch Version :", torch.__version__)
print("CUDA Available  :", torch.cuda.is_available())
print("CUDA Devices    :", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU Name        :", torch.cuda.get_device_name(0))
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

print("Training Device :", DEVICE)
print("=" * 60)
print(f"Using device: {DEVICE}")

IMAGE_SIZE  = 512
BATCH_SIZE  = 8
EPOCHS      = 100          # from-scratch needs far more epochs than fine-tuning
LR          = 3e-4         # higher LR since no pretrained features to preserve
NUM_CLASSES = 2

os.makedirs(MODEL_DIR, exist_ok=True)
with open(SPLIT_PATH) as f: split = json.load(f)
print(f"Split — train:{len(split['train'])} val:{len(split['val'])} test:{len(split['test'])}")

def compute_class_weights(file_list, sample=300):
    bg = defect = 0
    for fname in file_list[:sample]:
        m = np.array(Image.open(os.path.join(COMBINED_MSK, fname)).convert("L"))
        d = (m > 127).sum(); defect += d; bg += m.size - d
    total = bg + defect
    return torch.tensor([total/(2*bg), total/(2*defect)], dtype=torch.float32).to(DEVICE)

class_weights = compute_class_weights(split["train"])

class CombinedSegDataset(Dataset):
    def __init__(self, split_data, transform=None):
        self.split_data = split_data
        self.transform = transform

    def __len__(self):
        return len(self.split_data)

    def __getitem__(self, idx):
        fname = self.split_data[idx]

        image = np.array(
            Image.open(os.path.join(COMBINED_IMG, fname)).convert("RGB")
        )

        mask = np.array(
            Image.open(os.path.join(COMBINED_MSK, fname)).convert("L")
        )

        mask = (mask > 127).astype(np.int64)

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"].long()

        return image, mask

train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.5,1.0), p=0.5),
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3), A.Rotate(limit=20, p=0.4),
    A.RandomBrightnessContrast(p=0.4), A.GaussNoise(p=0.3),
    A.ElasticTransform(alpha=120, sigma=6, p=0.3), A.GridDistortion(p=0.2),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2(),
])
val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2(),
])

train_loader = DataLoader(
    CombinedSegDataset(split["train"], train_transform),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False
)

val_loader = DataLoader(
    CombinedSegDataset(split["val"], val_transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False
)
# ── KEY DIFFERENCE FROM v3 ──
# v3: SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b2", ...)
# v4: build the SAME architecture from a config, random weights, no download
print("Building SegFormer-B2 architecture with RANDOM initialization (no pretrained weights)...")
config = SegformerConfig(
    num_channels           = 3,
    num_labels              = NUM_CLASSES,
    depths                  = [3, 4, 6, 3],       # MiT-B2 depths
    hidden_sizes            = [64, 128, 320, 512], # MiT-B2 widths
    num_attention_heads     = [1, 2, 5, 8],
    decoder_hidden_size     = 768,
    id2label                = {0:"background",1:"defect"},
    label2id                = {"background":0,"defect":1},
)
model = SegformerForSemanticSegmentation(config).to(DEVICE)   # NOTE: no from_pretrained()

total_params = sum(p.numel() for p in model.parameters())
print(f"Model built from scratch — {total_params:,} parameters, all randomly initialized")

ce_loss = nn.CrossEntropyLoss(weight=class_weights)
def dice_loss(logits, targets, w=3.0, smooth=1.0):
    probs = F.softmax(logits, dim=1)[:,1,:,:]; targets = targets.float()
    inter = (probs*targets).sum(); union = probs.sum()+targets.sum()
    return w * (1 - (2*inter+smooth)/(union+smooth))
def combined_loss(logits, targets):
    logits_up = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    return ce_loss(logits_up, targets) + dice_loss(logits_up, targets)
def compute_metrics(logits, targets):
    logits_up = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    preds = (F.softmax(logits_up, dim=1)[:,1,:,:] > 0.5).long(); target = targets.long()
    inter = (preds*target).sum().float(); union = (preds+target).clamp(0,1).sum().float()
    iou = (inter/(union+1e-8)).item()
    dice = (2*inter/(preds.sum()+target.sum()+1e-8)).item()
    return iou, dice

# from-scratch: train ALL params from epoch 1, no freeze phase
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_iou = 0.0
history  = []
print(f"\nTraining {EPOCHS} epochs FROM SCRATCH (no freeze phase — nothing pretrained to protect)\n")

for epoch in range(1, EPOCHS+1):
    model.train(); train_loss = 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        loss = combined_loss(model(pixel_values=images).logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step(); train_loss += loss.item()
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

with open(f"{MODEL_DIR}/history_v4_scratch.json","w") as f:
    json.dump(history, f, indent=2)
print(f"\nBest model saved to:\n{SAVE_PATH}")

print("\n" + "=" * 60)
print("SegFormer-B2 Training Complete")
print("=" * 60)
print(f"Best Validation IoU : {best_iou:.4f}")
print(f"Model Saved         : {SAVE_PATH}")
print("=" * 60)