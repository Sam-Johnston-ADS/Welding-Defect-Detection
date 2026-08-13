"""
SegFormer-B2 v6 (Strip-Pooling + ASPP + Boundary Head) Test Evaluation Script
Run locally:
  python test_segformer_v6.py
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
from transformers import SegformerModel, SegformerConfig
from pathlib import Path

# ==========================================================
# WINDOWS PATHS & CONSTANTS[cite: 4]
# ==========================================================
BASE_DIR     = Path(r"E:\weld-defect-detection (1)\weld-defect-detection")
COMBINED_IMG = str(BASE_DIR / "data" / "reviewed" / "WDXI" / "combined" / "images")
COMBINED_MSK = str(BASE_DIR / "data" / "reviewed" / "WDXI" / "combined" / "labels")
SPLIT_PATH   = str(BASE_DIR / "data" / "reviewed" / "WDXI" / "combined" / "split.json")
MODEL_DIR    = str(BASE_DIR / "models" / "segformer_v6")
SAVE_PATH    = str(Path(MODEL_DIR) / "segformer_v6_best.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE       = 512
BATCH_SIZE       = 4
NUM_CLASSES      = 2
DECODER_CHANNELS = 256          
BOUNDARY_WEIGHT  = 0.4 

# ==========================================================
# DATASET & TRANSFORMS[cite: 4]
# ==========================================================
class CombinedSegDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files, self.transform = files, transform
    def __len__(self): return len(self.files)
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
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2(),
])

# ==========================================================
# V6 MODEL COMPONENTS[cite: 4]
# ==========================================================
class StripPooling(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        inter = max(8, channels // reduction)
        self.reduce  = nn.Conv2d(channels, inter, 1)
        self.conv_h  = nn.Conv1d(inter, inter, kernel_size=3, padding=1)
        self.conv_w  = nn.Conv1d(inter, inter, kernel_size=3, padding=1)
        self.fuse    = nn.Conv2d(inter, channels, 1)
        self.bn      = nn.BatchNorm2d(channels)

    def forward(self, x):
        b, c, h, w = x.shape
        r = self.reduce(x)
        xh = r.mean(dim=3)                       
        xh = self.conv_h(xh)
        xh = xh.unsqueeze(-1).expand(-1, -1, -1, w)
        xw = r.mean(dim=2)                       
        xw = self.conv_w(xw)
        xw = xw.unsqueeze(-2).expand(-1, -1, h, -1)
        gate = torch.sigmoid(self.bn(self.fuse(F.relu(xh + xw))))
        return x + gate * x   

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
            ) for r in rates
        ])
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(rates), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.1)
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        return self.project(torch.cat(feats, dim=1))

class SegformerV6(nn.Module):
    def __init__(self, config, decoder_channels=DECODER_CHANNELS, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = SegformerModel(config)   
        hidden_sizes = config.hidden_sizes       

        self.strip_pool_3 = StripPooling(hidden_sizes[2])
        self.strip_pool_4 = StripPooling(hidden_sizes[3])

        self.proj = nn.ModuleList([
            nn.Conv2d(hs, decoder_channels, 1) for hs in hidden_sizes
        ])

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_channels * 4, decoder_channels, 1, bias=False),
            nn.BatchNorm2d(decoder_channels), nn.ReLU(inplace=True),
        )
        self.aspp = ASPP(decoder_channels, decoder_channels)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Conv2d(decoder_channels, num_classes, 1)

        self.boundary_head = nn.Sequential(
            nn.Conv2d(decoder_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values, output_hidden_states=True)
        h1, h2, h3, h4 = outputs.hidden_states

        h3 = self.strip_pool_3(h3)
        h4 = self.strip_pool_4(h4)

        target_size = h1.shape[-2:]   
        feats = []
        for feat, proj in zip([h1, h2, h3, h4], self.proj):
            f = proj(feat)
            f = F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            feats.append(f)

        fused = self.linear_fuse(torch.cat(feats, dim=1))
        context = self.aspp(fused)
        context = self.dropout(context)

        main_logits = self.classifier(context)
        boundary_logits = self.boundary_head(fused)   

        return main_logits, boundary_logits

# ==========================================================
# LOSS & METRICS FUNCTIONS[cite: 4]
# ==========================================================
def compute_class_weights(file_list, sample=300):
    bg = defect = 0
    for fname in file_list[:sample]:
        m = np.array(Image.open(os.path.join(COMBINED_MSK, fname)).convert("L"))
        d = (m > 127).sum(); defect += d; bg += m.size - d
    total = bg + defect
    return torch.tensor([total/(2*bg), total/(2*defect)], dtype=torch.float32).to(DEVICE)

# Fixed Sobel kernels for deriving edge ground-truth
_sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3).to(DEVICE)
_sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3).to(DEVICE)

def sobel_edges(mask_float):
    m = mask_float.unsqueeze(1)  
    gx = F.conv2d(m, _sobel_x, padding=1)
    gy = F.conv2d(m, _sobel_y, padding=1)
    mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
    return (mag > 0.1).float() 

def dice_loss_fn(logits, targets, w=3.0, smooth=1.0):
    probs = F.softmax(logits, dim=1)[:,1,:,:]; targets = targets.float()
    inter = (probs*targets).sum(); union = probs.sum()+targets.sum()
    return w * (1 - (2*inter+smooth)/(union+smooth))

def compute_test_metrics(main_logits, targets):
    logits_up = F.interpolate(main_logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    preds = (F.softmax(logits_up, dim=1)[:,1,:,:] > 0.5).long()
    target = targets.long()
    
    # Accuracy
    correct = (preds == target).sum().float()
    total_pixels = torch.numel(preds)
    accuracy = (correct / total_pixels).item()
    
    # IoU and Dice
    inter = (preds*target).sum().float()
    union = (preds+target).clamp(0,1).sum().float()
    iou = (inter/(union+1e-8)).item()
    dice = (2*inter/(preds.sum()+target.sum()+1e-8)).item()
    
    return iou, dice, accuracy

# ==========================================================
# MAIN EXECUTION[cite: 4]
# ==========================================================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # 1. Load Split Data
    with open(SPLIT_PATH) as f: 
        split = json.load(f)
        
    print(f"Loading Test Split: {len(split['test'])} held-out images")

    # 2. Compute class weights using training set to keep test CE loss consistent
    class_weights = compute_class_weights(split["train"])
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def main_loss(logits, targets):
        logits_up = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        return ce_loss(logits_up, targets) + dice_loss_fn(logits_up, targets)

    def boundary_loss(boundary_logits, targets):
        targets_f = targets.float()
        boundary_up = F.interpolate(boundary_logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        edges_full = sobel_edges(targets_f)
        return F.binary_cross_entropy_with_logits(boundary_up, edges_full)

    def combined_loss(main_logits, boundary_logits, targets):
        return main_loss(main_logits, targets) + BOUNDARY_WEIGHT * boundary_loss(boundary_logits, targets)

    test_loader = DataLoader(
        CombinedSegDataset(split["test"], val_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # 3. Initialize V6 Architecture
    print("Building SegFormer v6 (Strip Pool + ASPP + Boundary head) architecture...")
    config = SegformerConfig(
        num_channels=3, num_labels=NUM_CLASSES,
        depths=[3,4,6,3], hidden_sizes=[64,128,320,512],
        num_attention_heads=[1,2,5,8],
    )
    model = SegformerV6(config).to(DEVICE)

    # 4. Load Saved Weights
    if not os.path.exists(SAVE_PATH):
        raise FileNotFoundError(f"Model weights not found at {SAVE_PATH}")
        
    print(f"Loading best weights from: {SAVE_PATH}")
    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    print("\nStarting Test Evaluation...\n")
    model.eval()
    
    test_loss = 0.0
    test_iou = 0.0
    test_dice = 0.0
    test_acc = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="[Testing]"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            # Forward pass
            main_logits, boundary_logits = model(images)
            
            # Calculate Loss
            loss = combined_loss(main_logits, boundary_logits, masks)
            test_loss += loss.item()
            
            # Calculate Metrics (evaluated on the main logits, not the boundary)
            iou, dice, accuracy = compute_test_metrics(main_logits, masks)
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
    print("FINAL V6 TEST SPLIT RESULTS")
    print("=" * 60)
    print(f"Test Accuracy : {test_acc * 100:.2f}%")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test IoU      : {test_iou:.4f}")
    print(f"Test Dice     : {test_dice:.4f}")
    print("=" * 60)