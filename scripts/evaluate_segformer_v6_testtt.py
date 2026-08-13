"""
Evaluate SegFormer v6 (Strip Pool + ASPP + Boundary head) on the
held-out TEST split. 

Added metrics: Test Loss, Test Accuracy, Test IoU, Test Dice.
"""
import os, json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel, SegformerConfig

# ─────────────────────────────────────────────
# PATHS & CONFIG
# ─────────────────────────────────────────────
BASE_DIR          = Path(r"E:\weld-defect-detection (1)\weld-defect-detection")
COMBINED_DATA_DIR = BASE_DIR / "data" / "reviewed" / "WDXI" / "combined"
COMBINED_IMG      = str(COMBINED_DATA_DIR / "images")
COMBINED_MSK      = str(COMBINED_DATA_DIR / "labels")
SPLIT_PATH        = str(COMBINED_DATA_DIR / "split.json")
CKPT_PATH         = str(BASE_DIR / "models" / "segformer_v6" / "segformer_v6_best.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DECODER_CHANNELS = 256
NUM_CLASSES = 2

with open(SPLIT_PATH) as f:
    split = json.load(f)
test_files = split["test"]
train_files = split["train"]
print(f"Evaluating on {len(test_files)} held-out TEST images (never used in train OR val)\n")

# ─────────────────────────────────────────────
# REBUILD ARCHITECTURE (Inference Mode)
# ─────────────────────────────────────────────
class StripPooling(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        inter = max(8, channels // reduction)
        self.reduce = nn.Conv2d(channels, inter, 1)
        self.conv_h = nn.Conv1d(inter, inter, kernel_size=3, padding=1)
        self.conv_w = nn.Conv1d(inter, inter, kernel_size=3, padding=1)
        self.fuse   = nn.Conv2d(inter, channels, 1)
        self.bn     = nn.BatchNorm2d(channels)
    def forward(self, x):
        b,c,h,w = x.shape
        r = self.reduce(x)
        xh = r.mean(dim=3); xh = self.conv_h(xh); xh = xh.unsqueeze(-1).expand(-1,-1,-1,w)
        xw = r.mean(dim=2); xw = self.conv_w(xw); xw = xw.unsqueeze(-2).expand(-1,-1,h,-1)
        gate = torch.sigmoid(self.bn(self.fuse(F.relu(xh+xw))))
        return x + gate * x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=(1,6,12,18)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels,out_channels,3,padding=r,dilation=r,bias=False),
                         nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)) for r in rates
        ])
        self.project = nn.Sequential(
            nn.Conv2d(out_channels*len(rates), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.1)
        )
    def forward(self,x):
        return self.project(torch.cat([b(x) for b in self.branches], dim=1))

class SegformerV6(nn.Module):
    def __init__(self, config, decoder_channels=DECODER_CHANNELS, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = SegformerModel(config)
        hidden_sizes = config.hidden_sizes
        self.strip_pool_3 = StripPooling(hidden_sizes[2])
        self.strip_pool_4 = StripPooling(hidden_sizes[3])
        self.proj = nn.ModuleList([nn.Conv2d(hs, decoder_channels, 1) for hs in hidden_sizes])
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_channels*4, decoder_channels, 1, bias=False),
            nn.BatchNorm2d(decoder_channels), nn.ReLU(inplace=True))
        self.aspp = ASPP(decoder_channels, decoder_channels)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Conv2d(decoder_channels, num_classes, 1)
        # Boundary head is built to safely load state_dict, but skipped in forward pass below
        self.boundary_head = nn.Sequential(
            nn.Conv2d(decoder_channels,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64,1,1))

    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values, output_hidden_states=True)
        h1,h2,h3,h4 = outputs.hidden_states
        h3 = self.strip_pool_3(h3); h4 = self.strip_pool_4(h4)
        target_size = h1.shape[-2:]
        feats = []
        for feat, proj in zip([h1,h2,h3,h4], self.proj):
            f = proj(feat)
            f = F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            feats.append(f)
        fused = self.linear_fuse(torch.cat(feats, dim=1))
        context = self.aspp(fused); context = self.dropout(context)
        main_logits = self.classifier(context)
        return main_logits  # only return main_logits at inference

# ─────────────────────────────────────────────
# LOSS RECREATION (For accurate Test Loss)
# ─────────────────────────────────────────────
def compute_class_weights(file_list, sample=300):
    bg = defect = 0
    for fname in file_list[:sample]:
        m = np.array(Image.open(os.path.join(COMBINED_MSK, fname)).convert("L"))
        d = (m > 127).sum(); defect += d; bg += m.size - d
    total = bg + defect
    return torch.tensor([total/(2*bg), total/(2*defect)], dtype=torch.float32).to(DEVICE)

print("Computing class weights from train split to accurately calculate loss...")
class_weights = compute_class_weights(train_files)
ce_loss = nn.CrossEntropyLoss(weight=class_weights)

def dice_loss_fn(logits, targets, w=3.0, smooth=1.0):
    probs = F.softmax(logits, dim=1)[:,1,:,:]
    targets = targets.float()
    inter = (probs*targets).sum()
    union = probs.sum()+targets.sum()
    return w * (1 - (2*inter+smooth)/(union+smooth))

def main_loss(logits, targets):
    logits_up = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    return ce_loss(logits_up, targets) + dice_loss_fn(logits_up, targets)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
config = SegformerConfig(
    num_channels=3, num_labels=NUM_CLASSES,
    depths=[3,4,6,3], hidden_sizes=[64,128,320,512],
    num_attention_heads=[1,2,5,8],
)
model = SegformerV6(config)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(DEVICE).eval()
print(f"Loaded v6 — reported val IoU: {ckpt['val_iou']:.4f} (epoch {ckpt['epoch']})\n")

# ─────────────────────────────────────────────
# EVALUATION LOOP
# ─────────────────────────────────────────────
ious, dices, accuracies, losses = [], [], [], []

with torch.no_grad():
    for fname in test_files:
        img = Image.open(os.path.join(COMBINED_IMG, fname)).convert("RGB")
        gt  = np.array(Image.open(os.path.join(COMBINED_MSK, fname)).convert("L"))
        gt_bin = (gt > 127).astype(np.uint8)
        h, w = gt_bin.shape

        # Scale to [0, 1]
        t = torch.from_numpy(np.array(img.resize((512, 512)))).permute(2, 0, 1).float() / 255.0
        
        # Apply ImageNet Normalization to match training
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = (t - mean) / std
        
        # Add batch dimension and move to device
        t = t.unsqueeze(0).to(DEVICE)
        
        # Prepare targets for loss function
        target_tensor = torch.tensor(gt_bin, dtype=torch.long, device=DEVICE).unsqueeze(0)

        # Forward pass
        logits = model(t)
        
        # 1. Compute Loss
        loss = main_loss(logits, target_tensor)
        losses.append(loss.item())

        # 2. Compute Predictions
        logits_up = F.interpolate(logits, size=(h,w), mode="bilinear", align_corners=False)
        pred = (F.softmax(logits_up, dim=1)[0,1].cpu().numpy() > 0.5).astype(np.uint8)

        # 3. Compute Metrics
        inter = int((pred & gt_bin).sum())
        union = int((pred | gt_bin).sum())
        iou = inter / (union + 1e-8)
        dice = (2 * inter) / (pred.sum() + gt_bin.sum() + 1e-8)
        acc = (pred == gt_bin).sum() / (h * w)  # Pixel Accuracy
        
        ious.append(iou)
        dices.append(dice)
        accuracies.append(acc)

mean_iou = float(np.mean(ious))
mean_dice = float(np.mean(dices))
mean_acc = float(np.mean(accuracies))
mean_loss = float(np.mean(losses))
gap = abs(ckpt['val_iou'] - mean_iou)

# ─────────────────────────────────────────────
# RESULTS REPORT
# ─────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"TRUE HELD-OUT TEST RESULTS — SegFormer v6")
print(f"{'='*50}")
print(f"Test Loss : {mean_loss:.4f}")
print(f"Test Acc  : {mean_acc:.4f} (Pixel Accuracy)")
print(f"Test IoU  : {mean_iou:.4f}")
print(f"Test Dice : {mean_dice:.4f}")
print(f"Val IoU   : {ckpt['val_iou']:.4f}")
print(f"Gap       : {gap:.4f}")
print(f"\n{'='*50}")
print(f"VERDICT")
print(f"{'='*50}")
if gap < 0.05:
    print(f"Gap is small — this result is TRUSTWORTHY.")
    print(f"v6 vs v4 (0.7332 test)  : {'+' if mean_iou>0.7332 else ''}{mean_iou-0.7332:.4f}")
    print(f"v6 vs v5 (0.5230 test, overfit): well ahead of the discarded model")
else:
    print(f"Gap is LARGE ({gap:.4f}) — same overfitting pattern as v5.")
    print(f"Do not report the val number. Discard or investigate further.")
