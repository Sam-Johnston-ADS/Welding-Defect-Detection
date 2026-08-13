"""
SAM 2 + LoRA Fine-tuning — Combined Dataset, Point-Prompt (v3)
Same point-prompt + LoRA r=16 approach as v2, but trained on the
combined WDXI pool (datasets1 + datasets2) instead of the
train-on-1/val-on-2 cross-condition split.

Run on Kaggle GPU:
  !pip install transformers peft accelerate -q
  !python training/finetune_sam2_lora_v3.py
"""

import os, json, shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Sam2Processor, Sam2Model
from peft import LoraConfig, get_peft_model

BASE_DIR     = "/kaggle/working/weld-defect-detection"
COMBINED_IMG = f"{BASE_DIR}/data/reviewed/WDXI/combined/images"
COMBINED_MSK = f"{BASE_DIR}/data/reviewed/WDXI/combined/labels"
SPLIT_PATH   = f"{BASE_DIR}/data/reviewed/WDXI/combined/split.json"
MODEL_DIR    = f"{BASE_DIR}/models/sam2_v3"
SAVE_PATH    = f"{MODEL_DIR}/sam2_lora_v3_best.pth"
DRIVE_SAVE   = "/kaggle/working/sam2_lora_v3_best.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

BATCH_SIZE     = 2
EPOCHS         = 40
LR             = 8e-5
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.1
NUM_POS_POINTS = 3
NUM_NEG_POINTS = 2

os.makedirs(MODEL_DIR, exist_ok=True)

with open(SPLIT_PATH) as f:
    split = json.load(f)
print(f"Split — train:{len(split['train'])} val:{len(split['val'])} "
      f"test:{len(split['test'])} (test held out)")

class CombinedSAM2PointDataset(Dataset):
    def __init__(self, file_list, processor):
        self.files     = file_list
        self.processor = processor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        image = np.array(Image.open(os.path.join(COMBINED_IMG, fname)).convert("RGB"))
        mask  = np.array(Image.open(os.path.join(COMBINED_MSK, fname)).convert("L"))
        mask_bin = (mask > 127).astype(np.uint8)

        points, labels = self._sample_points(mask_bin)

        inputs = self.processor(
            images=Image.fromarray(image),
            input_points=[[points]], input_labels=[[labels]],
            return_tensors="pt"
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(mask_bin, dtype=torch.float32)
        return inputs

    def _sample_points(self, mask):
        h, w = mask.shape
        fg = np.argwhere(mask > 0); bg = np.argwhere(mask == 0)
        points, labels = [], []

        if len(fg) > 0:
            for i in np.random.choice(len(fg), min(NUM_POS_POINTS, len(fg)), replace=False):
                y, x = fg[i]; points.append([int(x), int(y)]); labels.append(1)
        else:
            points.append([w//2, h//2]); labels.append(0)

        if len(bg) > 0:
            for i in np.random.choice(len(bg), min(NUM_NEG_POINTS, len(bg)), replace=False):
                y, x = bg[i]; points.append([int(x), int(y)]); labels.append(0)

        return points, labels

print("Loading SAM 2 base model...")
MODEL_ID  = "facebook/sam2-hiera-small"
processor = Sam2Processor.from_pretrained(MODEL_ID)
model     = Sam2Model.from_pretrained(MODEL_ID)

lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                         bias="none", target_modules=["attn.qkv", "attn.proj"])
model.vision_encoder = get_peft_model(model.vision_encoder, lora_config)
model.vision_encoder.print_trainable_parameters()

for p in model.prompt_encoder.parameters(): p.requires_grad = False
for p in model.mask_decoder.parameters():   p.requires_grad = True
model = model.to(DEVICE)

train_dataset = CombinedSAM2PointDataset(split["train"], processor)
val_dataset   = CombinedSAM2PointDataset(split["val"],   processor)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    inter = (pred*target).sum(dim=(2,3)); union = pred.sum(dim=(2,3))+target.sum(dim=(2,3))
    return 1 - (2*inter+smooth)/(union+smooth)

def combined_loss(pred, target):
    target = target.unsqueeze(1).float()
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return F.binary_cross_entropy_with_logits(pred, target) + dice_loss(pred, target).mean()

def compute_iou(pred_logits, target, thr=0.5):
    pred = (torch.sigmoid(pred_logits) > thr).float(); target = target.unsqueeze(1).float()
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred.float(), size=target.shape[-2:], mode="nearest")
    inter = (pred*target).sum(dim=(2,3)); union = (pred+target).clamp(0,1).sum(dim=(2,3))
    return (inter/(union+1e-8)).mean().item()

def compute_dice(pred_logits, target, thr=0.5):
    pred = (torch.sigmoid(pred_logits) > thr).float(); target = target.unsqueeze(1).float()
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred.float(), size=target.shape[-2:], mode="nearest")
    inter = (pred*target).sum(dim=(2,3))
    return (2*inter/(pred.sum(dim=(2,3))+target.sum(dim=(2,3))+1e-8)).mean().item()

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_iou = 0.0
history  = []
print(f"\nTraining {EPOCHS} epochs, LoRA r={LORA_R}, point prompts\n")

for epoch in range(1, EPOCHS+1):
    model.train(); train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
        pv = batch["pixel_values"].to(DEVICE)
        ip = batch.get("input_points"); il = batch.get("input_labels")
        gt = batch["ground_truth_mask"].to(DEVICE)
        if ip is not None: ip = ip.to(DEVICE)
        if il is not None: il = il.to(DEVICE)

        optimizer.zero_grad()
        out = model(pixel_values=pv, input_points=ip, input_labels=il, multimask_output=False)
        loss = combined_loss(out.pred_masks.squeeze(1), gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval(); val_loss = val_iou = val_dice = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
            pv = batch["pixel_values"].to(DEVICE)
            ip = batch.get("input_points"); il = batch.get("input_labels")
            gt = batch["ground_truth_mask"].to(DEVICE)
            if ip is not None: ip = ip.to(DEVICE)
            if il is not None: il = il.to(DEVICE)
            out = model(pixel_values=pv, input_points=ip, input_labels=il, multimask_output=False)
            pm  = out.pred_masks.squeeze(1)
            val_loss += combined_loss(pm, gt).item()
            val_iou  += compute_iou(pm, gt)
            val_dice += compute_dice(pm, gt)
    val_loss /= len(val_loader); val_iou /= len(val_loader); val_dice /= len(val_loader)
    scheduler.step()

    print(f"\nEpoch {epoch:02d} | Train Loss:{train_loss:.4f} | "
          f"Val Loss:{val_loss:.4f} | IoU:{val_iou:.4f} | Dice:{val_dice:.4f}")
    history.append({"epoch":epoch,"train_loss":train_loss,"val_loss":val_loss,
                    "iou":val_iou,"dice":val_dice})

    if val_iou > best_iou:
        best_iou = val_iou
        lora_weights = {k:v for k,v in model.state_dict().items()
                        if "lora" in k.lower() or "mask_decoder" in k.lower()}
        torch.save({"epoch":epoch,"lora_weights":lora_weights,"val_iou":val_iou,
                    "val_dice":val_dice,
                    "lora_config":{"r":LORA_R,"lora_alpha":LORA_ALPHA,"lora_dropout":LORA_DROPOUT},
                    "prompt_type":"point","data":"combined"}, SAVE_PATH)
        print(f"  Best model saved -> IoU: {best_iou:.4f}")

with open(f"{MODEL_DIR}/sam2_v3_history.json","w") as f:
    json.dump(history, f, indent=2)
shutil.copy(SAVE_PATH, DRIVE_SAVE)

print(f"\nSAM2 LoRA v3 (combined data) complete. Best IoU: {best_iou:.4f}")
print(f"v1 (box, cross-condition split)  : 0.6964")
print(f"v2 (point, cross-condition split): ~0.67 (your last run)")
print(f"v3 (point, combined pool)        : {best_iou:.4f}")
print(f"Mentor target: 0.80")
print(f"Model saved: {DRIVE_SAVE}")
print(f"Report the held-out 'test' split score separately — it's the fairest number.")
