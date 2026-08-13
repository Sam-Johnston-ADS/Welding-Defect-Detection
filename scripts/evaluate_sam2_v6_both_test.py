"""
Evaluate BOTH v6-paired SAM2+LoRA checkpoints (dense-prompt and
box-only) on the held-out TEST split — the only number that
decides which one is actually better, since both are near-tied
on val (0.8404 vs 0.8396, statistical noise).

Run locally:
  python scripts/evaluate_sam2_v6_both_test.py
"""
import os, json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel, SegformerConfig, Sam2Processor, Sam2Model
from peft import LoraConfig, get_peft_model

BASE_DIR          = Path(r"E:\weld-defect-detection (1)\weld-defect-detection")
COMBINED_DATA_DIR = BASE_DIR / "data" / "reviewed" / "WDXI" / "combined"
COMBINED_IMG = str(COMBINED_DATA_DIR / "images")
COMBINED_MSK = str(COMBINED_DATA_DIR / "labels")
SPLIT_PATH   = str(COMBINED_DATA_DIR / "split.json")

SEGFORMER_V6_CKPT = str(BASE_DIR / "models" / "segformer_v6" / "segformer_v6_best.pth")
SAM2_DENSE_CKPT   = str(BASE_DIR / "models" / "sam2" / "sam2_lora_best_v6_dense.pth")            # teammate's dense-prompt version
SAM2_BOXONLY_CKPT = str(BASE_DIR / "models" / "sam2" / "sam2_lora_best_v6_boxonly.pth")    # yours, box-only

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/sam2-hiera-small"
DECODER_CHANNELS, NUM_CLASSES = 256, 2

with open(SPLIT_PATH) as f:
    split = json.load(f)
test_files = split["test"]
print(f"Evaluating on {len(test_files)} held-out TEST images\n")

# ── rebuild SegFormer v6 ──
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
        self.boundary_head = nn.Sequential(
            nn.Conv2d(decoder_channels,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64,1,1))
    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values, output_hidden_states=True)
        h1,h2,h3,h4 = outputs.hidden_states
        h3 = self.strip_pool_3(h3); h4 = self.strip_pool_4(h4)
        target_size = h1.shape[-2:]
        feats = [F.interpolate(p(f), size=target_size, mode="bilinear", align_corners=False)
                for f, p in zip([h1,h2,h3,h4], self.proj)]
        fused = self.linear_fuse(torch.cat(feats, dim=1))
        context = self.dropout(self.aspp(fused))
        return self.classifier(context), fused

config = SegformerConfig(num_channels=3, num_labels=NUM_CLASSES, depths=[3,4,6,3],
                         hidden_sizes=[64,128,320,512], num_attention_heads=[1,2,5,8])
segformer_v6 = SegformerV6(config)
seg_ckpt = torch.load(SEGFORMER_V6_CKPT, map_location=DEVICE, weights_only=False)
segformer_v6.load_state_dict(seg_ckpt["model_state_dict"])
segformer_v6 = segformer_v6.to(DEVICE).eval()
print(f"Loaded SegFormer v6 (shared upstream for both SAM2 variants)\n")

def get_v6_box(image_np):
    h, w, _ = image_np.shape
    t = torch.from_numpy(image_np).permute(2,0,1).float()/255.0
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    t = (t - mean) / std
    t = F.interpolate(t.unsqueeze(0), size=(512,512), mode="bilinear", align_corners=False).to(DEVICE)
    with torch.no_grad():
        logits, _ = segformer_v6(t)
        up = F.interpolate(logits, size=(h,w), mode="bilinear", align_corners=False)
        pred = up.argmax(dim=1).squeeze(0).cpu().numpy()
    nz = np.argwhere(pred > 0)
    if len(nz) == 0: return [0,0,int(w),int(h)]
    y_min,x_min = nz.min(axis=0); y_max,x_max = nz.max(axis=0)
    pad_x = int((x_max-x_min)*0.05); pad_y = int((y_max-y_min)*0.05)
    return [int(max(0,x_min-pad_x)), int(max(0,y_min-pad_y)),
            int(min(w,x_max+pad_x)), int(min(h,y_max+pad_y))]

def evaluate_sam2(ckpt_path, label):
    if not os.path.exists(ckpt_path):
        print(f"[skip] {label} — checkpoint not found at {ckpt_path}")
        return None

    processor = Sam2Processor.from_pretrained(MODEL_ID)
    model = Sam2Model.from_pretrained(MODEL_ID)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    lc = ckpt["lora_config"]
    lora_config = LoraConfig(r=lc["r"], lora_alpha=lc["lora_alpha"], lora_dropout=lc["lora_dropout"],
                             bias="none", target_modules=["attn.qkv","attn.proj"])
    model.vision_encoder = get_peft_model(model.vision_encoder, lora_config)
    for p in model.prompt_encoder.parameters(): p.requires_grad = False
    model.load_state_dict(ckpt["lora_weights"], strict=False)
    model = model.to(DEVICE).eval()

    ious, dices = [], []
    with torch.no_grad():
        for fname in test_files:
            image_pil = Image.open(os.path.join(COMBINED_IMG, fname)).convert("RGB")
            image_np  = np.array(image_pil)
            mask = np.array(Image.open(os.path.join(COMBINED_MSK, fname)).convert("L"))
            mask_bin = (mask > 127).astype(np.uint8)

            box = get_v6_box(image_np)
            inputs = processor(images=image_pil, input_boxes=[[box]], return_tensors="pt").to(DEVICE)
            out = model(**inputs, multimask_output=False)
            pred_logits = out.pred_masks.squeeze(1)

            target = torch.tensor(mask_bin, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            pred = (torch.sigmoid(pred_logits) > 0.5).float()
            if pred.shape[-2:] != target.shape[-2:]:
                pred = F.interpolate(pred.float(), size=target.shape[-2:], mode="nearest")

            inter = (pred*target).sum().item(); union = (pred+target).clamp(0,1).sum().item()
            iou = inter/(union+1e-8); dice = (2*inter)/(pred.sum().item()+target.sum().item()+1e-8)
            ious.append(iou); dices.append(dice)

    test_iou, test_dice = float(np.mean(ious)), float(np.mean(dices))
    gap = abs(ckpt["val_iou"] - test_iou)
    print(f"{label:30s} | Val IoU: {ckpt['val_iou']:.4f} | Test IoU: {test_iou:.4f} | Gap: {gap:.4f}")
    return {"label": label, "val_iou": ckpt["val_iou"], "test_iou": test_iou,
            "test_dice": test_dice, "gap": gap}

print(f"{'='*70}")
r1 = evaluate_sam2(SAM2_DENSE_CKPT, "Dense-prompt (teammate's)")
r2 = evaluate_sam2(SAM2_BOXONLY_CKPT, "Box-only, padded (yours)")
print(f"{'='*70}")

results = [r for r in [r1, r2] if r is not None]
if results:
    best = max(results, key=lambda r: r["test_iou"])
    print(f"\nWINNER on true held-out test: {best['label']} — test IoU {best['test_iou']:.4f}")
    print(f"\nFull leaderboard (all verified SAM2 cascades so far):")
    print(f"  v4-based  (box+pad)      : test IoU 0.8144")
    print(f"  v5-based  (box+pad)      : test IoU 0.8274")
    for r in results:
        print(f"  v6-based  ({r['label']:22s}): test IoU {r['test_iou']:.4f} "
              f"{'[TRUSTWORTHY]' if r['gap']<0.05 else '[LARGE GAP - CHECK]'}")
