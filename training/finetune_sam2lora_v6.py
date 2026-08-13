"""
SAM 2 + LoRA Fine-tuning — Cascade Stage 3.6 (SegFormer-B2 v6)
Boundary & Context Refinement for Weld Defect Segmentation
Cascade: Scratch-Trained SegFormer-B2 v6 (Strip Pooling + ASPP) -> SAM 2 Box-Prompt Refinement

Run from terminal:
  python finetune_sam2_v6.py
  python finetune_sam2_v6.py --epochs 50 --batch_size 4 --lr 1e-4
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

from transformers import Sam2Processor, Sam2Model, SegformerModel, SegformerConfig
from peft import LoraConfig, get_peft_model

# ─────────────────────────────────────────────
# 1. SEGFORMER-B2 V6 ARCHITECTURE
# ─────────────────────────────────────────────
DECODER_CHANNELS = 256
NUM_CLASSES      = 2

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


def load_v6_segformer(checkpoint_path, device):
    """Loads trained SegFormer v6 checkpoint."""
    config = SegformerConfig(
        num_channels=3, num_labels=NUM_CLASSES,
        depths=[3, 4, 6, 3], hidden_sizes=[64, 128, 320, 512],
        num_attention_heads=[1, 2, 5, 8],
    )
    model = SegformerV6(config)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing SegFormer v6 weights at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print("✅ Loaded Custom SegFormer v6 (Strip-Pooling + ASPP) checkpoint successfully.")
    return model


# ─────────────────────────────────────────────
# 2. CASCADE DATASET (SegFormer v6 -> SAM 2)
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
        
        # ImageNet normalization matching v6 training script
        img_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(512, 512),
                                   mode="bilinear", align_corners=False).to(self.device)

        # Forward pass through SegFormer v6
        main_logits, _ = self.segformer_model(img_tensor)
        upsampled = F.interpolate(main_logits, size=(h, w), mode="bilinear", align_corners=False)
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
# 3. LOSS & METRIC COMPUTATION
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
    parser = argparse.ArgumentParser(description="SAM 2 LoRA Fine-Tuning Guided by SegFormer v6")
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
    SAVE_PATH         = os.path.join(MODEL_DIR, "sam2_lora_best_v6.pth")
    SEGFORMER_CKPT    = str(BASE_DIR / "models" / "segformer_v6" / "segformer_v6_best.pth")

    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"split.json not found at {SPLIT_PATH}")

    with open(SPLIT_PATH) as f:
        split = json.load(f)
    print(f"Split loaded — train:{len(split['train'])} val:{len(split['val'])} test:{len(split['test'])}")

    # 1. Load SegFormer v6
    segformer_model = load_v6_segformer(SEGFORMER_CKPT, DEVICE)

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
    print(f"\nStarting SAM 2 LoRA fine-tuning (Guided by v6 Cascade)...\n")
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

    with open(os.path.join(MODEL_DIR, "sam2_history_v6.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*50}")
    print(f"SAM 2 Local Fine-tuning (v6 Cascade) Complete")
    print(f"{'='*50}")
    print(f"Best validation IoU : {best_iou:.4f}")


if __name__ == "__main__":
    main()