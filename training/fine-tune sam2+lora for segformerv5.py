"""
SAM 2 + LoRA Fine-tuning — Stage 3.5 (v5 ConvStem Cascade)
Boundary refinement for weld defect segmentation
Cascade: Scratch-Trained SegFormer-B2 (v5 ConvStem) -> SAM 2 Box-Prompt Refinement

Run from terminal:
  python training/finetune_sam2_v5_convstem.py
  python training/finetune_sam2_v5_convstem.py --epochs 50 --batch_size 4 --lr 1e-4
"""

import os
import json
import argparse
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
# 1. SEGFORMER V5 CONVSTEM ARCHITECTURE
# ─────────────────────────────────────────────
STEM_CHANNELS = 32

class ConvStem(nn.Module):
    """3-layer Conv Stem matching train_segformer_v5_convstem.py exactly."""
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
    """
    Conv stem -> SegFormer wrapper taking 32-channel stem output.
    """
    def __init__(self, segformer_config):
        super().__init__()
        self.stem      = ConvStem(in_channels=3, out_channels=STEM_CHANNELS)
        self.segformer = SegformerForSemanticSegmentation(segformer_config)

    def forward(self, pixel_values):
        stem_features = self.stem(pixel_values)
        return self.segformer(pixel_values=stem_features).logits


def load_v5_segformer(checkpoint_path, device):
    """Loads the v5 SegFormer weights with exact architecture match."""
    config = SegformerConfig(
        num_channels=STEM_CHANNELS, # 32 channels from ConvStem
        num_labels=2,
        depths=[3, 4, 6, 3],
        hidden_sizes=[64, 128, 320, 512],
        num_attention_heads=[1, 2, 5, 8],
        decoder_hidden_size=768,
        id2label={0: "background", 1: "defect"},
        label2id={"background": 0, "defect": 1},
    )

    model = SegformerWithConvStem(config)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing SegFormer v5 weights at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print("✅ Loaded Custom SegFormer v5 (ConvStem) checkpoint successfully.")
    return model


# ─────────────────────────────────────────────
# 2. CASCADE DATASET (SegFormer Prompt Generation -> SAM 2)
# ─────────────────────────────────────────────
class WeldSAM2CascadeDataset(Dataset):
    def __init__(self, file_list, img_dir, mask_dir, processor, segformer_model, device):
        self.files           = file_list
        self.img_dir         = img_dir
        self.mask_dir        = mask_dir
        self.processor       = processor
        self.segformer_model = segformer_model
        self.device          = device

    def __len__(self):
        return len(self.files)

    @torch.no_grad()
    def _get_segformer_box(self, image_np):
        h, w, _ = image_np.shape
        img_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(512, 512),
                                   mode="bilinear", align_corners=False).to(self.device)

        logits = self.segformer_model(img_tensor)
        if hasattr(logits, "logits"):
            logits = logits.logits

        upsampled = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        pred_mask = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()

        nonzero = np.argwhere(pred_mask > 0)
        if len(nonzero) == 0:
            return [0, 0, int(w), int(h)]

        y_min, x_min = nonzero.min(axis=0)
        y_max, x_max = nonzero.max(axis=0)
        pad_x = int((x_max - x_min) * 0.05)
        pad_y = int((y_max - y_min) * 0.05)

        return [int(max(0, x_min - pad_x)), int(max(0, y_min - pad_y)),
                int(min(w, x_max + pad_x)), int(min(h, y_max + pad_y))]

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
# 3. LOSS FUNCTIONS & METRICS
# ─────────────────────────────────────────────
def dice_loss(pred, target, smooth=1.0):
    pred  = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - (2 * inter + smooth) / (union + smooth)

def combined_loss(pred, target):
    target = target.unsqueeze(1).float()
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return F.binary_cross_entropy_with_logits(pred, target) + dice_loss(pred, target).mean()

def compute_iou(pred_logits, target, thr=0.5):
    pred = (torch.sigmoid(pred_logits) > thr).float()
    target = target.unsqueeze(1).float()
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred.float(), size=target.shape[-2:], mode="nearest")
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).clamp(0, 1).sum(dim=(2, 3))
    return (inter / (union + 1e-8)).mean().item()

def compute_dice(pred_logits, target, thr=0.5):
    pred = (torch.sigmoid(pred_logits) > thr).float()
    target = target.unsqueeze(1).float()
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred.float(), size=target.shape[-2:], mode="nearest")
    inter = (pred * target).sum(dim=(2, 3))
    return (2 * inter / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-8)).mean().item()


# ─────────────────────────────────────────────
# 4. MAIN EXECUTION PIPELINE
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SAM 2 LoRA Fine-Tuning Guided by SegFormer v5 ConvStem")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--r", type=int, default=4, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=8, help="LoRA alpha")
    parser.add_argument("--dropout", type=float, default=0.1, help="LoRA dropout")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Paths
    BASE_DIR          = Path(r"E:\weld-defect-detection (1)\weld-defect-detection")
    COMBINED_DATA_DIR = BASE_DIR / "data" / "reviewed" / "WDXI" / "combined"
    COMBINED_IMG      = str(COMBINED_DATA_DIR / "images")
    COMBINED_MSK      = str(COMBINED_DATA_DIR / "labels")
    SPLIT_PATH        = str(COMBINED_DATA_DIR / "split.json")

    MODEL_DIR         = str(BASE_DIR / "models" / "sam2")
    SAVE_PATH         = os.path.join(MODEL_DIR, "sam2_lora_best_v5_convstem.pth")
    SEGFORMER_CKPT    = str(BASE_DIR / "models" / "segformer_v5_convstem" / "segformer_v5_convstem_best.pth")

    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"split.json not found at {SPLIT_PATH}")

    with open(SPLIT_PATH) as f:
        split = json.load(f)
    print(f"Split loaded — train:{len(split['train'])} val:{len(split['val'])} test:{len(split['test'])}")

    # 1. Load SegFormer v5 ConvStem
    segformer_model = load_v5_segformer(SEGFORMER_CKPT, DEVICE)

    # 2. Load SAM 2
    print("\nLoading SAM 2 base model...")
    MODEL_ID  = "facebook/sam2-hiera-small"
    processor = Sam2Processor.from_pretrained(MODEL_ID)
    model     = Sam2Model.from_pretrained(MODEL_ID)

    # 3. Configure LoRA
    lora_config = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
        bias="none", target_modules=["attn.qkv", "attn.proj"]
    )
    model.vision_encoder = get_peft_model(model.vision_encoder, lora_config)

    for p in model.prompt_encoder.parameters(): p.requires_grad = False
    for p in model.mask_decoder.parameters():   p.requires_grad = True
    model = model.to(DEVICE)

    # 4. DataLoaders
    train_dataset = WeldSAM2CascadeDataset(split["train"], COMBINED_IMG, COMBINED_MSK, processor, segformer_model, DEVICE)
    val_dataset   = WeldSAM2CascadeDataset(split["val"],   COMBINED_IMG, COMBINED_MSK, processor, segformer_model, DEVICE)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 5. Training Loop
    print(f"\nStarting SAM 2 LoRA fine-tuning (Guided by v5 ConvStem)...\n")
    best_iou = 0.0
    history  = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]")
        
        for batch in pbar:
            pv = batch["pixel_values"].to(DEVICE)
            ib = batch.get("input_boxes").to(DEVICE) if batch.get("input_boxes") is not None else None
            gt = batch["ground_truth_mask"].to(DEVICE)

            optimizer.zero_grad()
            out = model(pixel_values=pv, input_boxes=ib, multimask_output=False)
            loss = combined_loss(out.pred_masks.squeeze(1), gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader)

        model.eval()
        val_loss = val_iou = val_dice = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Val]"):
                pv = batch["pixel_values"].to(DEVICE)
                ib = batch.get("input_boxes").to(DEVICE) if batch.get("input_boxes") is not None else None
                gt = batch["ground_truth_mask"].to(DEVICE)

                out = model(pixel_values=pv, input_boxes=ib, multimask_output=False)
                pm  = out.pred_masks.squeeze(1)

                val_loss += combined_loss(pm, gt).item()
                val_iou  += compute_iou(pm, gt)
                val_dice += compute_dice(pm, gt)

        val_loss /= len(val_loader)
        val_iou  /= len(val_loader)
        val_dice /= len(val_loader)
        scheduler.step()

        print(f"\nEpoch {epoch:02d} | Train Loss:{train_loss:.4f} | "
              f"Val Loss:{val_loss:.4f} | Val IoU:{val_iou:.4f} | Val Dice:{val_dice:.4f}")
        
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "iou": val_iou, "dice": val_dice})

        if val_iou > best_iou:
            best_iou = val_iou
            lora_weights = {k: v for k, v in model.state_dict().items()
                            if "lora" in k.lower() or "mask_decoder" in k.lower()}
            torch.save({
                "epoch": epoch,
                "lora_weights": lora_weights,
                "val_iou": val_iou,
                "val_dice": val_dice,
                "lora_config": {"r": args.r, "lora_alpha": args.alpha, "lora_dropout": args.dropout}
            }, SAVE_PATH)
            print(f"  Best model saved -> IoU: {best_iou:.4f}")

    with open(os.path.join(MODEL_DIR, "sam2_history_v5_convstem.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*50}")
    print(f"SAM 2 Local Fine-tuning (v5 ConvStem Cascade) Complete")
    print(f"{'='*50}")
    print(f"Real validation IoU : {best_iou:.4f}")


if __name__ == "__main__":
    main()