"""
SAM 2 + LoRA Evaluation — "Downgraded" Box Prompt Baseline
Evaluates the SAM 2 model trained using loose (5% padded) bounding boxes from SegFormer v6.

Run from terminal:
  python test_sam2_v6_downgraded.py
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
    config = SegformerConfig(
        num_channels=3, num_labels=NUM_CLASSES,
        depths=[3, 4, 6, 3], hidden_sizes=[64, 128, 320, 512],
        num_attention_heads=[1, 2, 5, 8],
    )
    model = SegformerV6(config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model

# ─────────────────────────────────────────────
# 2. DATASET: Original 5% Padded Box Prompts
# ─────────────────────────────────────────────
class WeldSAM2TestDataset(Dataset):
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
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(512, 512),
                                   mode="bilinear", align_corners=False).to(self.device)

        main_logits, _ = self.segformer_model(img_tensor)
        upsampled = F.interpolate(main_logits, size=(h, w), mode="bilinear", align_corners=False)
        pred_mask = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()

        nonzero = np.argwhere(pred_mask > 0)
        if len(nonzero) == 0:
            return [0, 0, int(w), int(h)]

        y_min, x_min = nonzero.min(axis=0)
        y_max, x_max = nonzero.max(axis=0)
        
        # Original logic: 5% padding around the bounding box
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
# 3. METRICS
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

def compute_accuracy(pred_logits, target, thr=0.5):
    """Calculates overall pixel-wise accuracy."""
    pred = (torch.sigmoid(pred_logits) > thr).float()
    target = target.unsqueeze(1).float()
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred.float(), size=target.shape[-2:], mode="nearest")
    correct = (pred == target).float()
    return correct.mean().item()

# ─────────────────────────────────────────────
# 4. MAIN EXECUTION
# ─────────────────────────────────────────────
def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Paths
    BASE_DIR          = Path(r"E:\weld-defect-detection (1)\weld-defect-detection")
    COMBINED_DATA_DIR = BASE_DIR / "data" / "reviewed" / "WDXI" / "combined"
    COMBINED_IMG      = str(COMBINED_DATA_DIR / "images")
    COMBINED_MSK      = str(COMBINED_DATA_DIR / "labels")
    SPLIT_PATH        = str(COMBINED_DATA_DIR / "split.json")

    # Target the PREVIOUSLY trained weights (Downgraded Box prompting)
    SAM2_CKPT         = os.path.join(BASE_DIR, "models", "sam2", "sam2_lora_best_v6.pth")
    SEGFORMER_CKPT    = str(BASE_DIR / "models" / "segformer_v6" / "segformer_v6_best.pth")

    if not os.path.exists(SAM2_CKPT):
        raise FileNotFoundError(f"Missing fine-tuned SAM 2 weights at: {SAM2_CKPT}")
    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"Missing split.json at: {SPLIT_PATH}")

    with open(SPLIT_PATH) as f:
        split = json.load(f)
    print(f"Test split loaded: {len(split['test'])} images.")

    # 1. Load SegFormer v6
    print("\nLoading SegFormer-B2 v6 Prompt Generator...")
    segformer_model = load_v6_segformer(SEGFORMER_CKPT, DEVICE)

    # 2. Load SAM 2 and Checkpoint Data
    print("Loading SAM 2 base model and LoRA weights...")
    MODEL_ID  = "facebook/sam2-hiera-small"
    processor = Sam2Processor.from_pretrained(MODEL_ID)
    model     = Sam2Model.from_pretrained(MODEL_ID)

    ckpt_data = torch.load(SAM2_CKPT, map_location=DEVICE)
    lora_cfg_data = ckpt_data.get("lora_config", {"r": 4, "lora_alpha": 8, "lora_dropout": 0.1})

    # Apply LoRA Config
    lora_config = LoraConfig(
        r=lora_cfg_data["r"], 
        lora_alpha=lora_cfg_data["lora_alpha"], 
        lora_dropout=lora_cfg_data["lora_dropout"],
        bias="none", 
        target_modules=["attn.qkv", "attn.proj"]
    )
    model.vision_encoder = get_peft_model(model.vision_encoder, lora_config)

    # Load the fine-tuned weights
    model.load_state_dict(ckpt_data["lora_weights"], strict=False)
    model = model.to(DEVICE)
    model.eval()

    # 3. DataLoader
    test_dataset = WeldSAM2TestDataset(split["test"], COMBINED_IMG, COMBINED_MSK, processor, segformer_model, DEVICE)
    test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    # 4. Evaluation Loop
    print("\nStarting Evaluation on Test Set (Downgraded Box Prompt Baseline)...")
    test_loss = test_iou = test_dice = test_acc = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            pv = batch["pixel_values"].to(DEVICE)
            ib = batch.get("input_boxes").to(DEVICE) if batch.get("input_boxes") is not None else None
            gt = batch["ground_truth_mask"].to(DEVICE)

            out = model(pixel_values=pv, input_boxes=ib, multimask_output=False)
            pm  = out.pred_masks.squeeze(1)

            test_loss += combined_loss(pm, gt).item()
            test_iou  += compute_iou(pm, gt)
            test_dice += compute_dice(pm, gt)
            test_acc  += compute_accuracy(pm, gt)

    # Average metrics
    n_batches = len(test_loader)
    test_loss /= n_batches
    test_iou  /= n_batches
    test_dice /= n_batches
    test_acc  /= n_batches

    print("\n" + "="*50)
    print(" SAM 2 (Downgraded Box Prompt) - TEST SET RESULTS")
    print("="*50)
    print(f" Test Accuracy : {test_acc:.4f}")
    print(f" Test Loss     : {test_loss:.4f}")
    print(f" Test IoU      : {test_iou:.4f}")
    print(f" Test Dice     : {test_dice:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()