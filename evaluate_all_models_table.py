"""
Quantitative Evaluation & Tabular Image Generator — FINAL FIXED
Fixes applied:
  1. No HuggingFace security download blocks (bypassed).
  2. Strict testing on held-out test split (prevents data contamination).
  3. Float64 precision enforcement in BCE loss to prevent `NaN` errors.
"""

import os
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import (SegformerForSemanticSegmentation, SegformerConfig,
                          SegformerModel, Sam2Processor, Sam2Model)
from peft import LoraConfig, get_peft_model
import warnings

warnings.filterwarnings("ignore", message="You are using a model of type sam2_video*")

# ─────────────────────────────────────────────
# PATHS & CONFIGURATION
# ─────────────────────────────────────────────
BASE_DIR    = Path(r"D:\New folder\weld-defect-detection (4)\weld-defect-detection (1)\weld-defect-detection")
MODELS_DIR  = BASE_DIR / "models"
IMAGE_DIR   = BASE_DIR / "data" / "reviewed" / "WDXI" / "combined" / "images"
MASK_DIR    = BASE_DIR / "data" / "reviewed" / "WDXI" / "combined" / "labels"
SPLIT_PATH  = BASE_DIR / "data" / "reviewed" / "WDXI" / "combined" / "split.json"
OUTPUT_DIR  = BASE_DIR / "evaluation" / "quantitative_results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEM_CHANNELS    = 32
DECODER_CHANNELS = 256
MODEL_ID_SAM2    = "facebook/sam2-hiera-small"

def get_valid_path(primary, fallback):
    return primary if primary.exists() else fallback

CKPT = {
    "UNet"                     : MODELS_DIR / "unet" / "unet_best.pth",
    "SegFormer v2"             : MODELS_DIR / "unet" / "segformer_best_v2.pth",
    "SegFormer v3"             : get_valid_path(MODELS_DIR / "unet" / "segformer_v3_best.pth",
                                                MODELS_DIR / "segformer_v3" / "segformer_v3_best.pth"),
    "SegFormer v4 (Scratch)"   : MODELS_DIR / "segformer_v4_scratch" / "segformer_v4_scratch_best.pth",
    "SegFormer v5 (ConvStem)"  : MODELS_DIR / "segformer_v5_convstem" / "segformer_v5_convstem_best.pth",
    "SegFormer v6 (ASPP/SP)"   : MODELS_DIR / "segformer_v6" / "segformer_v6_best.pth",
    "SAM2 v1"                  : MODELS_DIR / "sam2" / "sam2_lora_best.pth",
    "SAM2 Fixed"               : MODELS_DIR / "sam2" / "sam2_lora_best_fixed.pth",
    "SAM2 v5"                  : MODELS_DIR / "sam2" / "sam2_lora_best_v5_convstem.pth",
    "SAM2 v6"                  : MODELS_DIR / "sam2" / "sam2_lora_best_v6.pth",
    "SAM2 v6 (Box-only)"       : MODELS_DIR / "sam2" / "sam2_lora_best_v6_boxonly.pth",
    "SAM2 v6 (Dense)"          : MODELS_DIR / "sam2" / "sam2_lora_best_v6_dense.pth",
}

SAM2_BOX_SOURCE = {
    "SAM2 v1": "SegFormer v2",
    "SAM2 Fixed": "SegFormer v4 (Scratch)",
    "SAM2 v5": "SegFormer v5 (ConvStem)",
    "SAM2 v6": "SegFormer v6 (ASPP/SP)",
    "SAM2 v6 (Box-only)": "SegFormer v6 (ASPP/SP)",
    "SAM2 v6 (Dense)": "SegFormer v6 (ASPP/SP)",
}

norm_transform = transforms.Compose([
    transforms.Resize((512, 512)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]), 
])

# ═══════════════════════════════════════════════
# ARCHITECTURE DEFINITIONS
# ═══════════════════════════════════════════════
class ConvStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=STEM_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels,16,3,1,1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16,24,3,1,1), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
            nn.Conv2d(24,out_channels,3,1,1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class SegformerWithConvStem(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stem = ConvStem()
        self.segformer = SegformerForSemanticSegmentation(config)
    def forward(self, pixel_values):
        return self.segformer(pixel_values=self.stem(pixel_values)).logits

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
                         nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)) for r in rates])
        self.project = nn.Sequential(
            nn.Conv2d(out_channels*len(rates), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.1))
    def forward(self,x): return self.project(torch.cat([b(x) for b in self.branches], dim=1))

class SegformerV6(nn.Module):
    def __init__(self, config, decoder_channels=DECODER_CHANNELS, num_classes=2):
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
        self.boundary_head = nn.Sequential(
            nn.Conv2d(decoder_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1))
    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values, output_hidden_states=True)
        h1,h2,h3,h4 = outputs.hidden_states
        h3 = self.strip_pool_3(h3); h4 = self.strip_pool_4(h4)
        target_size = h1.shape[-2:]
        feats = [F.interpolate(p(f), size=target_size, mode="bilinear", align_corners=False)
                for f, p in zip([h1,h2,h3,h4], self.proj)]
        fused = self.linear_fuse(torch.cat(feats, dim=1))
        context = self.dropout(self.aspp(fused))
        return self.classifier(context)

# ═══════════════════════════════════════════════
# MODEL LOADERS
# ═══════════════════════════════════════════════
def load_unet(path):
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()

def load_segformer_pretrained(path):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2", num_labels=2,
        id2label={0:"background",1:"defect"}, label2id={"background":0,"defect":1},
        ignore_mismatched_sizes=True,
    )
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"    [WARNING] {len(missing)} missing / {len(unexpected)} unexpected keys.")
    else:
        print(f"    All keys matched — loaded cleanly.")

    return model.to(DEVICE).eval()

def load_segformer_scratch(path):
    config = SegformerConfig(
        num_channels=3, num_labels=2, depths=[3,4,6,3],
        hidden_sizes=[64,128,320,512], num_attention_heads=[1,2,5,8],
        decoder_hidden_size=768,
        id2label={0:"background",1:"defect"}, label2id={"background":0,"defect":1},
    )
    model = SegformerForSemanticSegmentation(config)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()

def load_segformer_v5(path):
    config = SegformerConfig(
        num_channels=STEM_CHANNELS, num_labels=2, depths=[3,4,6,3],
        hidden_sizes=[64,128,320,512], num_attention_heads=[1,2,5,8],
        decoder_hidden_size=768,
        id2label={0:"background",1:"defect"}, label2id={"background":0,"defect":1},
    )
    model = SegformerWithConvStem(config)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()

def load_segformer_v6(path):
    config = SegformerConfig(num_channels=3, num_labels=2, depths=[3,4,6,3],
                             hidden_sizes=[64,128,320,512], num_attention_heads=[1,2,5,8])
    model = SegformerV6(config)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()

def load_sam2(path):
    processor = Sam2Processor.from_pretrained(MODEL_ID_SAM2)
    model = Sam2Model.from_pretrained(MODEL_ID_SAM2)
    ckpt = torch.load(path, map_location=DEVICE)
    lc = ckpt["lora_config"]
    lora_config = LoraConfig(r=lc["r"], lora_alpha=lc["lora_alpha"], lora_dropout=lc["lora_dropout"],
                             bias="none", target_modules=["attn.qkv","attn.proj"])
    model.vision_encoder = get_peft_model(model.vision_encoder, lora_config)
    for p in model.prompt_encoder.parameters(): p.requires_grad = False
    model.load_state_dict(ckpt["lora_weights"], strict=False)
    return processor, model.to(DEVICE).eval()

SEGFORMER_LOADERS = {
    "UNet"                     : ("unet", load_unet),
    "SegFormer v2"              : ("segformer", load_segformer_pretrained),
    "SegFormer v3"              : ("segformer", load_segformer_pretrained),
    "SegFormer v4 (Scratch)"    : ("segformer", load_segformer_scratch),
    "SegFormer v5 (ConvStem)"   : ("segformer_stem", load_segformer_v5),
    "SegFormer v6 (ASPP/SP)"    : ("segformer_v6", load_segformer_v6),
}

# ═══════════════════════════════════════════════
# INFERENCE & METRICS
# ═══════════════════════════════════════════════
def compute_metrics(prob_map, gt_binary, eps=1e-7):
    # Guard against NaN in raw prediction
    if np.isnan(prob_map).any():
        prob_map = np.nan_to_num(prob_map, nan=0.0)

    # REPAIR: Force float64 to prevent float32 rounding bugs where (1.0 - 1e-7) -> 1.0
    prob_64 = prob_map.astype(np.float64)
    prob_clamped = np.clip(prob_64, eps, 1.0 - eps)
    
    # Calculate BCE safely
    term1 = gt_binary * np.log(prob_clamped)
    term2 = (1.0 - gt_binary) * np.log(1.0 - prob_clamped)
    bce_loss = -np.mean(term1 + term2)

    # Standard metrics
    pred_binary = (prob_map > 0.5).astype(np.uint8)
    accuracy = np.mean(pred_binary == gt_binary) * 100.0

    intersection = np.sum((pred_binary == 1) & (gt_binary == 1))
    union = np.sum((pred_binary == 1) | (gt_binary == 1))
    iou = intersection / (union + eps)
    dice = (2.0 * intersection) / (np.sum(pred_binary) + np.sum(gt_binary) + eps)

    return float(bce_loss), accuracy, iou, dice, pred_binary

def infer_unet(model, image_pil, orig_size):
    tfm = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor(),
                              transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    t = tfm(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(t)
        prob = torch.sigmoid(logits)[0,0].float().cpu().numpy()
    return cv2.resize(prob, orig_size, interpolation=cv2.INTER_LINEAR)

def infer_segformer(model, image_pil, orig_size, kind="segformer"):
    t = norm_transform(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(pixel_values=t).logits if kind == "segformer" else model(t)
        logits_up = F.interpolate(logits, size=(orig_size[1], orig_size[0]), mode="bilinear", align_corners=False)
        prob = F.softmax(logits_up, dim=1)[0,1].float().cpu().numpy()
    return prob

def mask_to_box(mask_bin):
    h, w = mask_bin.shape
    nz = np.argwhere(mask_bin > 0)
    if len(nz) == 0: return [0, 0, int(w), int(h)]
    y_min, x_min = nz.min(axis=0); y_max, x_max = nz.max(axis=0)
    pad_x = int((x_max - x_min) * 0.05); pad_y = int((y_max - y_min) * 0.05)
    return [int(max(0, x_min - pad_x)), int(max(0, y_min - pad_y)),
            int(min(w, x_max + pad_x)), int(min(h, y_max + pad_y))]

def infer_sam2(processor, model, image_pil, box):
    inputs = processor(images=image_pil, input_boxes=[[box]], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, multimask_output=False)
    pred_prob = torch.sigmoid(out.pred_masks.squeeze(1))
    orig_w, orig_h = image_pil.size
    pred_prob = F.interpolate(pred_prob, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    return pred_prob.squeeze().float().cpu().numpy()

# ═══════════════════════════════════════════════
# TABLE RENDERER 
# ═══════════════════════════════════════════════
def render_metrics_table(results_dict, output_path):
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.axis('tight'); ax.axis('off')

    col_labels = ["Model Name", "Test Loss \u2193", "Accuracy (%) \u2191", "Test IoU \u2191", "Dice Score \u2191"]
    table_data = []

    best_iou, best_model_name = -1.0, ""
    for name, m in results_dict.items():
        if m["iou"] > best_iou:
            best_iou, best_model_name = m["iou"], name

    for name, m in results_dict.items():
        table_data.append([name, f"{m['loss']:.4f}", f"{m['accuracy']:.2f}%",
                           f"{m['iou']:.4f}", f"{m['dice']:.4f}"])

    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.8)

    for col_idx in range(len(col_labels)):
        cell = table[(0, col_idx)]
        cell.set_facecolor('#1F497D'); cell.get_text().set_color('white')
        cell.get_text().set_weight('bold'); cell.get_text().set_fontsize(11)

    for row_idx, (name, _) in enumerate(results_dict.items(), start=1):
        is_best = (name == best_model_name)
        bg_color = '#E2EFDA' if is_best else ('#F2F2F2' if row_idx % 2 == 0 else '#FFFFFF')
        for col_idx in range(len(col_labels)):
            cell = table[(row_idx, col_idx)]
            cell.set_facecolor(bg_color)
            if is_best: cell.get_text().set_weight('bold')

    plt.title("Quantitative Comparison of Segmentation Models on HELD-OUT TEST Split",
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nTabular image saved to: {output_path}")

# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SPLIT_PATH.exists():
        print(f"FATAL: split.json not found at {SPLIT_PATH}")
        print(f"This file defines which images are genuinely held-out for testing.")
        return

    with open(SPLIT_PATH) as f:
        split = json.load(f)
    
    # --- CONSOLE VERIFICATION ---
    train_files = split.get("train", [])
    val_files = split.get("val", [])
    test_files = split["test"]
    print("="*50)
    print(f"DATASET SPLIT LOADED SUCCESSFULLY")
    print(f"  > Ignoring Train Images: {len(train_files)}")
    print(f"  > Ignoring Val Images:   {len(val_files)}")
    print(f"  > Evaluating on HELD-OUT TEST images: {len(test_files)}")
    print("="*50 + "\n")

    results = {}
    cached_segformer_masks = {}

    for name, path in CKPT.items():
        if name.startswith("SAM2"): continue
        if not path.exists():
            print(f"[SKIP] {name} — checkpoint missing at: {path}")
            continue

        print(f"Evaluating {name}...")
        kind, loader = SEGFORMER_LOADERS[name]
        try:
            model = loader(path)
            losses, accs, ious, dices = [], [], [], []
            cached_segformer_masks[name] = {}

            # STRICT TEST SPLIT LOOP
            for fname in test_files:
                img_path = IMAGE_DIR / fname
                if not img_path.exists():
                    continue
                image_pil = Image.open(img_path).convert("RGB")
                orig_size = image_pil.size

                gt_path = MASK_DIR / fname
                gt = np.array(Image.open(gt_path).convert("L"))
                gt_bin = (gt > 127).astype(np.uint8)

                if kind == "unet":
                    prob = infer_unet(model, image_pil, orig_size)
                else:
                    prob = infer_segformer(model, image_pil, orig_size, kind=kind)

                loss, acc, iou, dice, pred_bin = compute_metrics(prob, gt_bin)
                losses.append(loss); accs.append(acc); ious.append(iou); dices.append(dice)
                cached_segformer_masks[name][fname] = pred_bin

            results[name] = {"loss": np.mean(losses), "accuracy": np.mean(accs),
                             "iou": np.mean(ious), "dice": np.mean(dices)}
            del model; torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [ERROR] Crashed while evaluating {name}: {e}")

    for name, path in CKPT.items():
        if not name.startswith("SAM2"): continue
        if not path.exists():
            print(f"[SKIP] {name} — checkpoint missing at: {path}")
            continue

        source_name = SAM2_BOX_SOURCE[name]
        if source_name not in cached_segformer_masks:
            print(f"[SKIP] {name} — upstream prompt source '{source_name}' unavailable.")
            continue

        print(f"Evaluating {name} (box prompt from {source_name})...")
        try:
            processor, model = load_sam2(path)
            losses, accs, ious, dices = [], [], [], []

            # STRICT TEST SPLIT LOOP
            for fname in test_files:
                img_path = IMAGE_DIR / fname
                if not img_path.exists() or fname not in cached_segformer_masks[source_name]:
                    continue
                image_pil = Image.open(img_path).convert("RGB")

                gt_path = MASK_DIR / fname
                gt = np.array(Image.open(gt_path).convert("L"))
                gt_bin = (gt > 127).astype(np.uint8)

                box = mask_to_box(cached_segformer_masks[source_name][fname])
                prob = infer_sam2(processor, model, image_pil, box)

                loss, acc, iou, dice, _ = compute_metrics(prob, gt_bin)
                losses.append(loss); accs.append(acc); ious.append(iou); dices.append(dice)

            results[name] = {"loss": np.mean(losses), "accuracy": np.mean(accs),
                             "iou": np.mean(ious), "dice": np.mean(dices)}
            del model; torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [ERROR] Crashed while evaluating {name}: {e}")

    json_path = OUTPUT_DIR / "quantitative_results_test_split.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    table_path = OUTPUT_DIR / "models_quantitative_comparison_table_test_split.png"
    render_metrics_table(results, table_path)

if __name__ == "__main__":
    main()