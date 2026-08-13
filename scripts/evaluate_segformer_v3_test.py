"""
Evaluate SegFormer v3 on the held-out TEST split only.
This is the number that actually matters for reporting —
train/val were both used during training/model-selection,
test was never touched until now.

Run on Kaggle:
  !python scripts/evaluate_segformer_v3_test.py
"""
import os, json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation

BASE_DIR     = "/kaggle/working/weld-defect-detection"
COMBINED_IMG = f"{BASE_DIR}/data/reviewed/WDXI/combined/images"
COMBINED_MSK = f"{BASE_DIR}/data/reviewed/WDXI/combined/labels"
SPLIT_PATH   = f"{BASE_DIR}/data/reviewed/WDXI/combined/split.json"
CKPT_PATH    = f"{BASE_DIR}/models/segformer_v3/segformer_v3_best.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(SPLIT_PATH) as f:
    split = json.load(f)
test_files = split["test"]
print(f"Evaluating on {len(test_files)} held-out TEST images (never seen during training)")

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b2", num_labels=2,
    id2label={0:"background",1:"defect"}, label2id={"background":0,"defect":1},
    ignore_mismatched_sizes=True,
).to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded checkpoint — reported val IoU during training: {ckpt['val_iou']:.4f}")

tfm = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

ious, dices = [], []
with torch.no_grad():
    for fname in test_files:
        img = Image.open(os.path.join(COMBINED_IMG, fname)).convert("RGB")
        gt  = np.array(Image.open(os.path.join(COMBINED_MSK, fname)).convert("L"))
        gt_bin = (gt > 127).astype(np.uint8)
        h, w = gt_bin.shape

        tensor = tfm(img).unsqueeze(0).to(DEVICE)
        logits = model(pixel_values=tensor).logits
        logits = F.interpolate(logits, size=(h,w), mode="bilinear", align_corners=False)
        pred = (F.softmax(logits, dim=1)[0,1].cpu().numpy() > 0.5).astype(np.uint8)

        inter = int((pred & gt_bin).sum())
        union = int((pred | gt_bin).sum())
        iou = inter / (union + 1e-8)
        dice = (2*inter) / (pred.sum() + gt_bin.sum() + 1e-8)
        ious.append(iou); dices.append(dice)

mean_iou, mean_dice = float(np.mean(ious)), float(np.mean(dices))
print(f"\n{'='*50}")
print(f"TRUE HELD-OUT TEST RESULTS ({len(test_files)} images)")
print(f"{'='*50}")
print(f"Test IoU  : {mean_iou:.4f}")
print(f"Test Dice : {mean_dice:.4f}")
print(f"\nCompare against reported val IoU: {ckpt['val_iou']:.4f}")
print(f"Large gap between these two = leakage or overfitting to val split.")
