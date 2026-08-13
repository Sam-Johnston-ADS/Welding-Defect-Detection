"""
SegFormer-B2 v5 (Conv Stem) Test Evaluation Script
Run locally:
  python scripts/test_segformer_v5.py
"""
import os
import json
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
# WINDOWS PATHS & CONSTANTS[cite: 3]
# ==========================================================
BASE_DIR = r"E:\weld-defect-detection (1)\weld-defect-detection"
COMBINED_IMG = os.path.join(BASE_DIR, "data", "reviewed", "WDXI", "combined", "images")
COMBINED_MSK = os.path.join(BASE_DIR, "data", "reviewed", "WDXI", "combined", "labels")
SPLIT_PATH = os.path.join(BASE_DIR, "data", "reviewed", "WDXI", "combined", "split.json")
MODEL_DIR = os.path.join(BASE_DIR, "models", "segformer_v5_convstem")
SAVE_PATH = os.path.join(MODEL_DIR, "segformer_v5_convstem_best.pth")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512
BATCH_SIZE = 4 
NUM_CLASSES = 2
STEM_CHANNELS = 32

# ==========================================================
# DATASET & TRANSFORMS[cite: 3]
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

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ==========================================================
# V5 MODEL ARCHITECTURE (CONV STEM)[cite: 3]
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

# ==========================================================
# LOSS & METRICS FUNCTIONS[cite: 3]
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

def compute_test_metrics(logits, targets):
    logits_up = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    preds = (F.softmax(logits_up, dim=1)[:, 1, :, :] > 0.5).long()
    target = targets.long()
    
    # Accuracy
    correct = (preds == target).sum().float()
    total_pixels = torch.numel(preds)
    accuracy = (correct / total_pixels).item()
    
    # IoU and Dice
    inter = (preds * target).sum().float()
    union = (preds + target).clamp(0, 1).sum().float()
    iou = (inter / (union + 1e-8)).item()
    dice = (2 * inter / (preds.sum() + target.sum() + 1e-8)).item()
    
    return iou, dice, accuracy

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # 1. Load Split Data[cite: 3]
    with open(SPLIT_PATH) as f: 
        split = json.load(f)
        
    print(f"Loading Test Split: {len(split['test'])} held-out images")

    # 2. Compute class weights using training set to keep test loss consistent[cite: 3]
    class_weights = compute_class_weights(split["train"], DEVICE)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    test_loader = DataLoader(
        CombinedSegDataset(split["test"], val_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # 3. Initialize V5 Architecture[cite: 3]
    print("Building SegFormer-B2 + Conv Stem architecture...")
    segformer_config = SegformerConfig(
        num_channels=STEM_CHANNELS,         
        num_labels=NUM_CLASSES,
        depths=[3, 4, 6, 3],
        hidden_sizes=[64, 128, 320, 512],
        num_attention_heads=[1, 2, 5, 8],
        decoder_hidden_size=768,
        id2label={0: "background", 1: "defect"},
        label2id={"background": 0, "defect": 1},
    )
    model = SegformerWithConvStem(segformer_config).to(DEVICE)

    # 4. Load Saved Weights
    if not os.path.exists(SAVE_PATH):
        raise FileNotFoundError(f"Model weights not found at {SAVE_PATH}")
        
    print(f"Loading best weights from: {SAVE_PATH}")
    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False)
    
    # Handle potentially nested dictionaries
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    
    print("\nStarting Test Evaluation...\n")
    model.eval()
    
    test_loss = 0.0
    test_iou = 0.0
    test_dice = 0.0
    test_acc = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="[Testing]"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            # Forward pass[cite: 3]
            logits = model(images)
            
            # Calculate Loss[cite: 3]
            loss = combined_loss(logits, masks, ce_loss)
            test_loss += loss.item()
            
            # Calculate Metrics
            iou, dice, accuracy = compute_test_metrics(logits, masks)
            test_iou += iou
            test_dice += dice
            test_acc += accuracy
            
    # Average across all batches
    num_batches = len(test_loader)
    test_loss /= num_batches
    test_iou /= num_batches
    test_dice /= num_batches
    test_acc /= num_batches

    print("\n" + "=" * 60)
    print("FINAL V5 TEST SPLIT RESULTS")
    print("=" * 60)
    print(f"Test Accuracy : {test_acc * 100:.2f}%")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test IoU      : {test_iou:.4f}")
    print(f"Test Dice     : {test_dice:.4f}")
    print("=" * 60)