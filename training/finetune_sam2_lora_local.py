"""
SAM 2 + LoRA Fine-tuning — Stage 3.5 (FIXED — real train/val split)
Boundary refinement for weld defect segmentation
Cascade: Scratch-Trained SegFormer-B2 -> SAM 2 box-prompt refinement

FIX from previous version: VAL_IMG/VAL_MASK were pointing at the
SAME folder as TRAIN_IMG/TRAIN_MASK with no split.json used, so the
reported 0.90 IoU was the model being scored on its own training
images (memorization), not real validation. This version loads the
actual train/val/test split so the numbers are trustworthy.

Run locally:
  python training/finetune_sam2_lora_local_fixed.py
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import Sam2Processor, Sam2Model, SegformerForSemanticSegmentation, SegformerConfig
from peft import LoraConfig, get_peft_model

# ─────────────────────────────────────────────
# 1. LOCAL PATHS
# ─────────────────────────────────────────────
BASE_DIR          = Path(r"E:\weld-defect-detection (1)\weld-defect-detection")
COMBINED_DATA_DIR = BASE_DIR / "data" / "reviewed" / "WDXI" / "combined"

COMBINED_IMG = str(COMBINED_DATA_DIR / "images")
COMBINED_MSK = str(COMBINED_DATA_DIR / "labels")
SPLIT_PATH   = str(COMBINED_DATA_DIR / "split.json")   # <- THE FIX: actually load this

MODEL_DIR   = str(BASE_DIR / "models" / "sam2")
SAVE_PATH   = os.path.join(MODEL_DIR, "sam2_lora_best_fixed.pth")

SEGFORMER_CHECKPOINT = str(BASE_DIR / "models" / "segformer_v4_scratch" / "segformer_v4_scratch_best.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

if not os.path.exists(SPLIT_PATH):
    raise FileNotFoundError(
        f"split.json not found at {SPLIT_PATH}. "
        f"Run scripts/prepare_combined_dataset.py first — it generates this file. "
        f"Without it there is no valid train/val separation."
    )

with open(SPLIT_PATH) as f:
    split = json.load(f)
print(f"Split loaded — train:{len(split['train'])} val:{len(split['val'])} "
      f"test:{len(split['test'])}  (test held out, untouched)")

# ─────────────────────────────────────────────
# 2. CONFIG
# ─────────────────────────────────────────────
BATCH_SIZE   = 4
EPOCHS       = 50
LR           = 1e-4
NUM_CLASSES  = 1
LORA_R       = 4
LORA_ALPHA   = 8
LORA_DROPOUT = 0.1

os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 3. DATASET — reads a file LIST, not a whole folder
# ─────────────────────────────────────────────
class WeldSAM2CascadeDataset(Dataset):
    def __init__(self, file_list, img_dir, mask_dir, processor, segformer_model):
        self.files           = file_list      # <- explicit split list, not os.listdir
        self.img_dir         = img_dir
        self.mask_dir        = mask_dir
        self.processor       = processor
        self.segformer_model = segformer_model

    def __len__(self):
        return len(self.files)

    @torch.no_grad()
    def _get_segformer_box(self, image_np):
        h, w, _ = image_np.shape
        img_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(512, 512),
                                   mode="bilinear", align_corners=False).to(DEVICE)

        outputs = self.segformer_model(pixel_values=img_tensor)
        upsampled = F.interpolate(outputs.logits, size=(h, w),
                                  mode="bilinear", align_corners=False)
        pred_mask = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()

        nonzero = np.argwhere(pred_mask > 0)
        if len(nonzero) == 0:
            return [0, 0, int(w), int(h)]

        y_min, x_min = nonzero.min(axis=0)
        y_max, x_max = nonzero.max(axis=0)
        pad_x = int((x_max - x_min) * 0.05)
        pad_y = int((y_max - y_min) * 0.05)

        return [int(max(0, x_min-pad_x)), int(max(0, y_min-pad_y)),
                int(min(w, x_max+pad_x)), int(min(h, y_max+pad_y))]

    def __getitem__(self, idx):
        fname     = self.files[idx]
        image_pil = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        image_np  = np.array(image_pil)
        mask      = np.array(Image.open(os.path.join(self.mask_dir, fname)).convert("L"))
        mask_bin  = (mask > 127).astype(np.uint8)

        box = self._get_segformer_box(image_np)

        inputs = self.processor(images=image_pil, input_boxes=[[box]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(mask_bin, dtype=torch.float32)
        return inputs

# ─────────────────────────────────────────────
# 4. LOAD SEGFORMER (from-scratch checkpoint)
# ─────────────────────────────────────────────
print("Building SegFormer-B2 architecture (from-scratch config)...")
segformer_config = SegformerConfig(
    num_channels=3, num_labels=2,
    depths=[3,4,6,3], hidden_sizes=[64,128,320,512],
    num_attention_heads=[1,2,5,8], decoder_hidden_size=768,
)
segformer_model = SegformerForSemanticSegmentation(segformer_config)

if not os.path.exists(SEGFORMER_CHECKPOINT):
    raise FileNotFoundError(f"Missing SegFormer weights at {SEGFORMER_CHECKPOINT}")

ckpt = torch.load(SEGFORMER_CHECKPOINT, map_location=DEVICE)
segformer_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
segformer_model = segformer_model.to(DEVICE).eval()
print(f"  Loaded SegFormer v4 — reported test IoU baseline should be verified separately")

# ─────────────────────────────────────────────
# 5. SAM 2 + LoRA
# ─────────────────────────────────────────────
print("\nLoading SAM 2 base model...")
MODEL_ID  = "facebook/sam2-hiera-small"
processor = Sam2Processor.from_pretrained(MODEL_ID)
model     = Sam2Model.from_pretrained(MODEL_ID)

lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                         bias="none", target_modules=["attn.qkv", "attn.proj"])
model.vision_encoder = get_peft_model(model.vision_encoder, lora_config)

for p in model.prompt_encoder.parameters(): p.requires_grad = False
for p in model.mask_decoder.parameters():   p.requires_grad = True
model = model.to(DEVICE)

# ─────────────────────────────────────────────
# 6. DATALOADERS — train split vs val split, genuinely different images
# ─────────────────────────────────────────────
train_dataset = WeldSAM2CascadeDataset(split["train"], COMBINED_IMG, COMBINED_MSK, processor, segformer_model)
val_dataset   = WeldSAM2CascadeDataset(split["val"],   COMBINED_IMG, COMBINED_MSK, processor, segformer_model)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images "
      f"(these are DIFFERENT images, verified against split.json)")

# ─────────────────────────────────────────────
# 7. LOSS / METRICS
# ─────────────────────────────────────────────
def dice_loss(pred, target, smooth=1.0):
    pred  = torch.sigmoid(pred)
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

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ─────────────────────────────────────────────
# 8. TRAINING LOOP
# ─────────────────────────────────────────────
print(f"\nStarting SAM 2 LoRA fine-tuning (leakage-fixed)...\n")
best_iou = 0.0
history  = []

for epoch in range(1, EPOCHS+1):
    model.train(); train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
        pv = batch["pixel_values"].to(DEVICE)
        ib = batch.get("input_boxes").to(DEVICE) if batch.get("input_boxes") is not None else None
        gt = batch["ground_truth_mask"].to(DEVICE)

        optimizer.zero_grad()
        out = model(pixel_values=pv, input_boxes=ib, multimask_output=False)
        loss = combined_loss(out.pred_masks.squeeze(1), gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step(); train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval(); val_loss = val_iou = val_dice = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
            pv = batch["pixel_values"].to(DEVICE)
            ib = batch.get("input_boxes").to(DEVICE) if batch.get("input_boxes") is not None else None
            gt = batch["ground_truth_mask"].to(DEVICE)
            out = model(pixel_values=pv, input_boxes=ib, multimask_output=False)
            pm  = out.pred_masks.squeeze(1)
            val_loss += combined_loss(pm, gt).item()
            val_iou  += compute_iou(pm, gt)
            val_dice += compute_dice(pm, gt)
    val_loss /= len(val_loader); val_iou /= len(val_loader); val_dice /= len(val_loader)
    scheduler.step()

    print(f"\nEpoch {epoch:02d} | Train Loss:{train_loss:.4f} | "
          f"Val Loss:{val_loss:.4f} | IoU:{val_iou:.4f} | Dice:{val_dice:.4f}")
    history.append({"epoch":epoch,"train_loss":train_loss,"val_loss":val_loss,"iou":val_iou,"dice":val_dice})

    if val_iou > best_iou:
        best_iou = val_iou
        lora_weights = {k:v for k,v in model.state_dict().items()
                        if "lora" in k.lower() or "mask_decoder" in k.lower()}
        torch.save({"epoch":epoch,"lora_weights":lora_weights,"val_iou":val_iou,"val_dice":val_dice,
                    "lora_config":{"r":LORA_R,"lora_alpha":LORA_ALPHA,"lora_dropout":LORA_DROPOUT}}, SAVE_PATH)
        print(f"  Best model saved -> IoU: {best_iou:.4f}")

with open(os.path.join(MODEL_DIR, "sam2_history_fixed.json"), "w") as f:
    json.dump(history, f, indent=2)

print(f"\n{'='*50}\nSAM 2 Local Fine-tuning (FIXED SPLIT) Complete\n{'='*50}")
print(f"Real validation IoU : {best_iou:.4f}")
print(f"(Previous 0.9032 was invalid — trained/validated on identical images)")
print(f"Next: evaluate on split['test'] before reporting anything to your mentor.")
