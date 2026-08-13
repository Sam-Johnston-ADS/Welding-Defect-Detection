"""
Quality Analysis — Multi-Image Matrix Comparison Grid
Runs multiple input images through every model you've trained across
this project and saves a single large matrix comparison grid (Original, GT, 
and binary masks for all models).
"""

import os
import glob
import string
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

# Suppress harmless huggingface warnings about SAM2 video models
warnings.filterwarnings("ignore", message="You are using a model of type sam2_video*")

# ─────────────────────────────────────────────
# PATHS & CONFIGURATION
# ─────────────────────────────────────────────
# Automatically set BASE_DIR to the folder containing the 'models' directory
BASE_DIR    = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE_DIR / "models"

# Define where to get the test images and ground truth masks
IMAGE_DIR    = BASE_DIR / "data" / "reviewed" / "WDXI" / "combined" / "images"
COMBINED_MSK = BASE_DIR / "data" / "reviewed" / "WDXI" / "combined" / "labels"  
OUTPUT_DIR   = BASE_DIR / "models" / "quality_analysis"

NUM_TEST_IMAGES = 6  # Adjust this to change how many columns appear in the grid

CKPT = {
    "unet"                  : MODELS_DIR / "unet" / "unet_best.pth",
    "segformer_v4_scratch"  : MODELS_DIR / "segformer_v4_scratch" / "segformer_v4_scratch_best.pth",
    "segformer_v5_convstem" : MODELS_DIR / "segformer_v5_convstem" / "segformer_v5_convstem_best.pth",
    "segformer_v6"          : MODELS_DIR / "segformer_v6" / "segformer_v6_best.pth",
    "sam2_v1"               : MODELS_DIR / "sam2" / "sam2_lora_best.pth",
    "sam2_fixed"            : MODELS_DIR / "sam2" / "sam2_lora_best_fixed.pth",
    "sam2_v5"               : MODELS_DIR / "sam2" / "sam2_lora_best_v5_convstem.pth",
    "sam2_v6"               : MODELS_DIR / "sam2" / "sam2_lora_best_v6.pth",
    "sam2_v6_boxonly"       : MODELS_DIR / "sam2" / "sam2_lora_best_v6_boxonly.pth",
    "sam2_v6_dense"         : MODELS_DIR / "sam2" / "sam2_lora_best_v6_dense.pth"
}

# Mapping SAM2 variants to their respective upstream Segformer bounding box sources
SAM2_BOX_SOURCE = {
    "sam2_v1"         : "segformer_v4_scratch",
    "sam2_fixed"      : "segformer_v4_scratch",
    "sam2_v5"         : "segformer_v5_convstem",
    "sam2_v6"         : "segformer_v6",
    "sam2_v6_boxonly" : "segformer_v6",
    "sam2_v6_dense"   : "segformer_v6"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEM_CHANNELS    = 32
DECODER_CHANNELS = 256
MODEL_ID_SAM2    = "facebook/sam2-hiera-small"

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
def clean_state_dict(state_dict):
    """Safely strips out extraneous prefixes from state dictionaries."""
    new_state = {}
    for k, v in state_dict.items():
        clean_k = k.replace("model.", "").replace("net.", "")
        new_state[clean_k] = v
    return new_state

def load_unet(path):
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(clean_state_dict(state), strict=False)
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
    model.load_state_dict(clean_state_dict(state), strict=False)
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
    model.load_state_dict(clean_state_dict(state), strict=False)
    return model.to(DEVICE).eval()

def load_segformer_v6(path):
    config = SegformerConfig(num_channels=3, num_labels=2, depths=[3,4,6,3],
                             hidden_sizes=[64,128,320,512], num_attention_heads=[1,2,5,8])
    model = SegformerV6(config)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(clean_state_dict(state), strict=False)
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
    "unet"                 : ("unet", load_unet),
    "segformer_v4_scratch" : ("segformer", load_segformer_scratch),
    "segformer_v5_convstem": ("segformer_stem", load_segformer_v5),
    "segformer_v6"         : ("segformer_v6", load_segformer_v6),
}

# ═══════════════════════════════════════════════
# INFERENCE HELPERS
# ═══════════════════════════════════════════════
def infer_unet(model, image_pil, orig_size):
    tfm = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor(),
                              transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    t = tfm(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(t)
        prob = torch.sigmoid(logits)[0,0].cpu().numpy()
    prob_rs = cv2.resize(prob, orig_size)
    return (prob_rs > 0.5).astype(np.uint8) * 255

def infer_segformer(model, image_pil, orig_size, kind="segformer"):
    t = norm_transform(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        if kind == "segformer":
            logits = model(pixel_values=t).logits
        else:  
            logits = model(t)
        
        logits_up = F.interpolate(logits, size=(orig_size[1], orig_size[0]),
                                  mode="bilinear", align_corners=False)
        
        # Dynamically handles 1-channel vs 2-channel output formats without crashing
        if logits_up.shape[1] == 1:
            pred = (torch.sigmoid(logits_up)[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
        else:
            pred = (F.softmax(logits_up, dim=1)[0, 1].cpu().numpy() > 0.5).astype(np.uint8)
            
    return pred * 255

def mask_to_box(mask_bin):
    h, w = mask_bin.shape
    nz = np.argwhere(mask_bin > 0)
    if len(nz) == 0: return [0,0,int(w),int(h)]
    y_min,x_min = nz.min(axis=0); y_max,x_max = nz.max(axis=0)
    pad_x = int((x_max-x_min)*0.05); pad_y = int((y_max-y_min)*0.05)
    return [int(max(0,x_min-pad_x)), int(max(0,y_min-pad_y)),
            int(min(w,x_max+pad_x)), int(min(h,y_max+pad_y))]

def infer_sam2(processor, model, image_pil, box):
    inputs = processor(images=image_pil, input_boxes=[[box]], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, multimask_output=False)
    pred = torch.sigmoid(out.pred_masks.squeeze(1))
    pred_bin = (pred > 0.5).float()
    orig_w, orig_h = image_pil.size
    pred_bin = F.interpolate(pred_bin, size=(orig_h, orig_w), mode="nearest")
    return pred_bin.squeeze().cpu().numpy().astype(np.uint8) * 255

# ═══════════════════════════════════════════════
# PLOTTING GRID HELPER
# ═══════════════════════════════════════════════
def generate_matrix_grid(image_data, model_names, out_path):
    num_cols = len(image_data)
    num_rows = 2 + len(model_names) # Orig + GT + Num Models
    
    # Set up matplotlib figure
    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols, 
        figsize=(2 * num_cols, 1.8 * num_rows),
        gridspec_kw={'wspace': 0.05, 'hspace': 0.05}
    )
    
    alphabet = list(string.ascii_lowercase)

    for col_idx, data in enumerate(image_data):
        # Row 0: Original Image (A)
        ax = axes[0, col_idx] if num_cols > 1 else axes[0]
        ax.imshow(cv2.cvtColor(data['orig_cv'], cv2.COLOR_BGR2RGB))
        ax.axis('off')
        if col_idx == 0:
            ax.text(-0.15, 0.5, "A", transform=ax.transAxes, 
                    fontsize=16, fontweight='bold', va='center', ha='right')

        # Row 1: Ground Truth (B)
        ax = axes[1, col_idx] if num_cols > 1 else axes[1]
        ax.imshow(data['gt_mask'], cmap='gray')
        ax.axis('off')
        if col_idx == 0:
            ax.text(-0.15, 0.5, "B", transform=ax.transAxes, 
                    fontsize=16, fontweight='bold', va='center', ha='right')

        # Row 2+: Model Masks (a, b, c...)
        for row_offset, m_name in enumerate(model_names):
            row_idx = 2 + row_offset
            ax = axes[row_idx, col_idx] if num_cols > 1 else axes[row_idx]
            
            mask = data['predictions'].get(m_name, np.zeros_like(data['gt_mask']))
            ax.imshow(mask, cmap='gray')
            ax.axis('off')
            
            if col_idx == 0:
                letter = alphabet[row_offset] if row_offset < len(alphabet) else str(row_offset)
                label = f"({letter}) {m_name}"
                ax.text(-0.15, 0.5, label, transform=ax.transAxes, 
                        fontsize=12, fontweight='bold', va='center', ha='right')

    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Gather test images
    if not IMAGE_DIR.exists():
        print(f"Error: IMAGE_DIR not found at {IMAGE_DIR}")
        return

    all_images = glob.glob(str(IMAGE_DIR / "*.png")) + glob.glob(str(IMAGE_DIR / "*.jpg"))
    test_images = all_images[:NUM_TEST_IMAGES]
    
    if not test_images:
        print("No images found in the directory.")
        return

    image_data_list = []
    
    # 2. Iterate through each image and run inference
    for img_path_str in test_images:
        img_path = Path(img_path_str)
        fname = img_path.name
        print(f"\nProcessing Image: {fname}")
        
        image_pil = Image.open(img_path).convert("RGB")
        img_cv = cv2.imread(str(img_path))
        orig_size = image_pil.size  
        
        # Load Ground truth
        gt_path = COMBINED_MSK / fname
        if gt_path.exists():
            gt = np.array(Image.open(gt_path).convert("L"))
            gt_bin = (gt > 127).astype(np.uint8) * 255
        else:
            gt_bin = np.zeros((orig_size[1], orig_size[0]), dtype=np.uint8)

        data = {
            'fname': fname,
            'orig_cv': img_cv,
            'gt_mask': gt_bin,
            'predictions': {}
        }

        segformer_masks = {}  

        # ── run all SegFormer-family + U-Net models ──
        for name, path in CKPT.items():
            if name.startswith("sam2"): continue
            if not path.exists():
                print(f"  [SKIP] {name} — checkpoint not found at {path}")
                continue

            print(f"  -> Running {name}...")
            try:
                kind, loader = SEGFORMER_LOADERS[name]
                model = loader(path)
                if kind == "unet":
                    mask = infer_unet(model, image_pil, orig_size)
                else:
                    mask = infer_segformer(model, image_pil, orig_size, kind=kind)
                
                segformer_masks[name] = mask
                data['predictions'][name] = mask
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                # We added this print to catch exactly what the error is going forward
                print(f"    [ERROR] {name} encountered an issue: {e}")

        # ── run SAM2 variants ──
        for name, path in CKPT.items():
            if not name.startswith("sam2"): continue
            if not path.exists(): continue
                
            source_name = SAM2_BOX_SOURCE[name]
            if source_name not in segformer_masks:
                print(f"  [SKIP] {name} — box source '{source_name}' did not run successfully")
                continue

            print(f"  -> Running {name} (box source: {source_name})...")
            try:
                processor, model = load_sam2(path)
                box = mask_to_box((segformer_masks[source_name] > 0).astype(np.uint8))
                mask = infer_sam2(processor, model, image_pil, box)
                
                data['predictions'][name] = mask
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    [ERROR] {name}: {e}")

        image_data_list.append(data)

    # 3. Generate the final matrix grid
    print(f"\n{'='*60}")
    print("Generating Comparison Matrix Grid...")
    
    # Get a list of all models successfully executed (to set the rows)
    successful_models = list(CKPT.keys()) 
    out_img_path = OUTPUT_DIR / "final_matrix_comparison_grid.png"
    
    generate_matrix_grid(image_data_list, successful_models, out_img_path)
    
    print(f"Matrix grid saved successfully to: {out_img_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()