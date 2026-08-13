import os
import json
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import SegformerConfig, SegformerModel
from tqdm import tqdm
from rfdetr import RFDETRBase

# ─────────────────────────────────────────────
# PATHS & CONFIG
# ─────────────────────────────────────────────
BASE_DIR     = r"E:\weld-defect-detection (1)\weld-defect-detection"
RIAWELC_DIR  = r"E:\weld-defect-detection (1)\weld-defect-detection\data\reviewed\RIAWELC\DB - Copy"

OUTPUT_DIR   = os.path.join(BASE_DIR, "data", "rfdetr_coco")
MODEL_DIR    = os.path.join(BASE_DIR, "models", "rfdetr")
LOCAL_SAVE   = os.path.join(BASE_DIR, "models", "rfdetr_best.pth")

# Path to your winning segmentation model
SEGFORMER_V6_PATH = os.path.join(BASE_DIR, "models", "segformer_v6", "segformer_v6_best.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_MAP = {
    "Difetto1"  : 0,   # Crack
    "Difetto2"  : 1,   # Porosity
    "Difetto4"  : 2,   # Lack of penetration
}
CLASS_NAMES = ["crack", "porosity", "lack_of_penetration"]

# ─────────────────────────────────────────────
# SEGFORMER V6 ARCHITECTURE (For Auto-Annotation)
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
                         nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)) for r in rates])
        self.project = nn.Sequential(
            nn.Conv2d(out_channels*len(rates), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.1))
    def forward(self,x): return self.project(torch.cat([b(x) for b in self.branches], dim=1))

class SegformerV6(nn.Module):
    def __init__(self, config, decoder_channels=256, num_classes=2):
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

def load_auto_annotator():
    """Loads SegFormer v6 to generate missing bounding boxes."""
    print("Loading SegFormer v6 for Auto-Annotation...")
    config = SegformerConfig(num_channels=3, num_labels=2, depths=[3,4,6,3],
                             hidden_sizes=[64,128,320,512], num_attention_heads=[1,2,5,8])
    model = SegformerV6(config)
    ckpt = torch.load(SEGFORMER_V6_PATH, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()

def get_pseudo_bbox(img_pil, annotator_model):
    """Uses SegFormer to predict defect mask and returns bounding box coordinates."""
    norm_transform = transforms.Compose([
        transforms.Resize((512, 512)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    t = norm_transform(img_pil).unsqueeze(0).to(DEVICE)
    w, h = img_pil.size
    
    with torch.no_grad():
        logits = annotator_model(t)
        logits_up = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        prob = F.softmax(logits_up, dim=1)[0,1].cpu().numpy()
        
    mask_bin = (prob > 0.5).astype(np.uint8)
    nz = np.argwhere(mask_bin > 0)
    
    # If no defect found, return None
    if len(nz) == 0:
        return None
        
    y_min, x_min = nz.min(axis=0)
    y_max, x_max = nz.max(axis=0)
    
    # Add 5% padding for RF-DETR context
    pad_x = int((x_max - x_min) * 0.05)
    pad_y = int((y_max - y_min) * 0.05)
    
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    box_w = min(w, x_max + pad_x) - x_min
    box_h = min(h, y_max + pad_y) - y_min
    
    return [int(x_min), int(y_min), int(box_w), int(box_h)]

# ─────────────────────────────────────────────
# DATASET CONVERTER
# ─────────────────────────────────────────────
def build_coco_dataset(split_name, rfdetr_split_name, annotator_model):
    """
    Reads RIAWELC split folder and uses SegFormer to auto-generate COCO boxes.
    """
    split_dir = os.path.join(RIAWELC_DIR, split_name)
    split_out_dir = os.path.join(OUTPUT_DIR, rfdetr_split_name)
    os.makedirs(split_out_dir, exist_ok=True)

    coco = {
        "info"       : {"description": "RIAWELC Weld Defect Auto-Annotated Dataset"},
        "categories" : [{"id": v, "name": k} for k, v in CLASS_MAP.items()],
        "images"     : [],
        "annotations": []
    }

    img_id = 0
    ann_id = 0
    skipped = 0

    for class_name, class_id in CLASS_MAP.items():
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.exists(class_dir):
            continue

        files = [f for f in os.listdir(class_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        print(f"  Auto-Annotating {split_name}/{class_name} ({len(files)} images)...")
        for fname in tqdm(files, leave=False):
            src = os.path.join(class_dir, fname)
            dst = os.path.join(split_out_dir, f"{img_id:06d}.png")

            img = Image.open(src).convert("RGB")
            w, h = img.size
            img.save(dst)

            coco["images"].append({
                "id"        : img_id,
                "file_name" : f"{img_id:06d}.png",
                "width"     : w,
                "height"    : h
            })

            # Pseudo-labeling magic
            bbox = get_pseudo_bbox(img, annotator_model)
            
            if bbox is not None:
                coco["annotations"].append({
                    "id"          : ann_id,
                    "image_id"    : img_id,
                    "category_id" : class_id,
                    "bbox"        : bbox,
                    "area"        : bbox[2] * bbox[3],
                    "iscrowd"     : 0
                })
                ann_id += 1
            else:
                skipped += 1

            img_id += 1

    json_path = os.path.join(split_out_dir, "_annotations.coco.json")
    with open(json_path, "w") as f:
        json.dump(coco, f)

    print(f"  [{rfdetr_split_name}] Generated {ann_id} valid bounding boxes. {skipped} images had undetectable defects.")
    return split_out_dir


# ─────────────────────────────────────────────
# MAIN EXECUTION BLOCK 
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    # 1. Load SegFormer to act as our automatic annotator
    annotator_model = load_auto_annotator()

    # 2. Convert and annotate the datasets
    print("\nConverting RIAWELC to Roboflow COCO format using Pseudo-labeling...")
    train_dir = build_coco_dataset("training", "train", annotator_model)
    val_dir   = build_coco_dataset("validation", "valid", annotator_model)
    test_dir  = build_coco_dataset("testing", "test", annotator_model)
    print("Auto-Annotation complete.")

    # 3. Cleanup VRAM before training RF-DETR
    del annotator_model
    torch.cuda.empty_cache()

    os.makedirs(MODEL_DIR, exist_ok=True)

    model = RFDETRBase(
        num_classes    = len(CLASS_NAMES),
        device         = DEVICE,
    )

    print("\nStarting RF-DETR training...")
    print(f"Classes : {CLASS_NAMES}")

    model.train(
        dataset_dir         = OUTPUT_DIR,         
        epochs              = 50,
        batch_size          = 8,
        grad_accum_steps    = 1,
        lr                  = 1e-4,
        output_dir          = MODEL_DIR,
        checkpoint_interval = 5,
        progress_bar        = "tqdm",
    )

    print("\nTraining complete.")

    # ─────────────────────────────────────────────
    # SAVE BEST MODEL LOCALLY
    # ─────────────────────────────────────────────
    best_path = os.path.join(MODEL_DIR, "checkpoint_best_total.pth")
    if os.path.exists(best_path):
        shutil.copy(best_path, LOCAL_SAVE)
        print(f"Best model saved locally to: {LOCAL_SAVE}")
    else:
        checkpoints = sorted(Path(MODEL_DIR).glob("*.pth"))
        if checkpoints:
            shutil.copy(str(checkpoints[-1]), LOCAL_SAVE)
            print(f"Saved latest checkpoint locally to: {LOCAL_SAVE}")
        else:
            print("No checkpoint found.")