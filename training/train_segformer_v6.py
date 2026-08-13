"""
SegFormer-B2 v6 — Strip-Pooling Encoder Context + Multi-Scale
ASPP Decoder + Auxiliary Boundary Supervision (from scratch)

Run locally:
  python train_segformer_v6.py
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
from transformers import SegformerModel, SegformerConfig
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS (Updated for local Windows environment)
# ─────────────────────────────────────────────
BASE_DIR     = Path(r"E:\weld-defect-detection (1)\weld-defect-detection")
COMBINED_IMG = str(BASE_DIR / "data" / "reviewed" / "WDXI" / "combined" / "images")
COMBINED_MSK = str(BASE_DIR / "data" / "reviewed" / "WDXI" / "combined" / "labels")
SPLIT_PATH   = str(BASE_DIR / "data" / "reviewed" / "WDXI" / "combined" / "split.json")
MODEL_DIR    = str(BASE_DIR / "models" / "segformer_v6")
SAVE_PATH    = str(Path(MODEL_DIR) / "segformer_v6_best.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE       = 512
BATCH_SIZE       = 4
EPOCHS           = 100          
LR               = 3e-4
NUM_CLASSES      = 2
DECODER_CHANNELS = 256          
BOUNDARY_WEIGHT  = 0.4          

def compute_class_weights(file_list, sample=300):
    bg = defect = 0
    for fname in file_list[:sample]:
        m = np.array(Image.open(os.path.join(COMBINED_MSK, fname)).convert("L"))
        d = (m > 127).sum(); defect += d; bg += m.size - d
    total = bg + defect
    return torch.tensor([total/(2*bg), total/(2*defect)], dtype=torch.float32).to(DEVICE)

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

# ─────────────────────────────────────────────
# MODEL COMPONENTS
# ─────────────────────────────────────────────
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

# fixed Sobel kernels for deriving edge ground-truth
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

def compute_metrics(main_logits, targets):
    logits_up = F.interpolate(main_logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    preds = (F.softmax(logits_up, dim=1)[:,1,:,:] > 0.5).long(); target = targets.long()
    inter = (preds*target).sum().float(); union = (preds+target).clamp(0,1).sum().float()
    iou = (inter/(union+1e-8)).item()
    dice = (2*inter/(preds.sum()+target.sum()+1e-8)).item()
    return iou, dice


# ─────────────────────────────────────────────
# MAIN EXECUTION BLOCK (Required for Windows)
# ─────────────────────────────────────────────
def main():
    print(f"Using device: {DEVICE}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    with open(SPLIT_PATH) as f: 
        split = json.load(f)
    print(f"Split — train:{len(split['train'])} val:{len(split['val'])} test:{len(split['test'])}")

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

    train_loader = DataLoader(CombinedSegDataset(split["train"], train_transform),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(CombinedSegDataset(split["val"], val_transform),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print("Building SegFormer v6 (Strip Pooling + ASPP + Boundary head, from scratch)...")
    config = SegformerConfig(
        num_channels=3, num_labels=NUM_CLASSES,
        depths=[3,4,6,3], hidden_sizes=[64,128,320,512],
        num_attention_heads=[1,2,5,8],
    )
    model = SegformerV6(config).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_iou = 0.0
    history  = []
    print(f"\nTraining {EPOCHS} epochs, from scratch, Strip-Pool + ASPP + boundary head\n")

    for epoch in range(1, EPOCHS+1):
        model.train(); train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            main_logits, boundary_logits = model(images)
            loss = combined_loss(main_logits, boundary_logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step(); train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval(); val_loss = val_iou = val_dice = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                main_logits, boundary_logits = model(images)
                val_loss += combined_loss(main_logits, boundary_logits, masks).item()
                iou, dice = compute_metrics(main_logits, masks)
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
                        "val_iou":val_iou,"val_dice":val_dice,
                        "decoder_channels":DECODER_CHANNELS}, SAVE_PATH)
            print(f"  Best model saved -> IoU: {best_iou:.4f}")

    with open(f"{MODEL_DIR}/history_v6.json","w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*50}")
    print(f"SegFormer v6 (Strip Pool + ASPP + Boundary head) complete")
    print(f"{'='*50}")
    print(f"Best val IoU (v6)         : {best_iou:.4f}")

if __name__ == '__main__':
    # Required for Windows multiprocessing compatibility
    main()