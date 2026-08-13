"""
SAM 2 Stage 3.5 — Inference Module
Boundary refinement after SegFormer segmentation.

Pipeline flow:
  Stage 3 (SegFormer) → binary mask
  Stage 3.5 (SAM 2)   → refined boundary mask
  Stage 4 (Post-processing) → cleaned mask

Usage in cascade.py:
  from pipeline.stage35_sam2_refine import load_sam2, run_sam2_refinement
"""

import os
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import Sam2Processor, Sam2Model
from peft import LoraConfig, get_peft_model

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_ID     = "facebook/sam2-hiera-small"
LORA_R       = 4
LORA_ALPHA   = 8
LORA_DROPOUT = 0.1
THRESHOLD    = 0.5
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# LOAD SAM 2 WITH LORA WEIGHTS
# ─────────────────────────────────────────────
def load_sam2(checkpoint_path, cache_dir=None):
    """
    Load SAM 2 with LoRA weights from fine-tuning checkpoint.

    Args:
        checkpoint_path: path to sam2_lora_best.pth
        cache_dir: optional local cache for model weights

    Returns:
        processor, model (ready for inference)
    """
    print("Loading SAM 2 + LoRA (Stage 3.5)...")

    processor = Sam2Processor.from_pretrained(
        MODEL_ID, cache_dir=cache_dir
    )
    model = Sam2Model.from_pretrained(
        MODEL_ID, cache_dir=cache_dir
    )

    # re-apply LoRA structure so weights can be loaded
    lora_config = LoraConfig(
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        target_modules = ["attn.qkv", "attn.proj"]
    )
    model.vision_encoder = get_peft_model(
        model.vision_encoder, lora_config
    )

    # load fine-tuned LoRA + mask decoder weights
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    lora_weights = checkpoint["lora_weights"]

    # load into model — strict=False handles any minor key mismatches
    missing, unexpected = model.load_state_dict(
        lora_weights, strict=False
    )
    if missing:
        print(f"  Missing keys  : {len(missing)} (expected for frozen layers)")
    if unexpected:
        print(f"  Unexpected    : {len(unexpected)}")

    model = model.to(DEVICE)
    model.eval()

    val_iou = checkpoint.get("val_iou", "unknown")
    print(f"  SAM 2 loaded — fine-tuned Val IoU: {val_iou}")
    return processor, model


# ─────────────────────────────────────────────
# MASK → BOUNDING BOX HELPER
# ─────────────────────────────────────────────
def mask_to_boxes(mask_np, min_area=50, padding=0.05):
    """
    Convert binary mask to list of bounding boxes,
    one per connected component (defect blob).
    """
    import cv2
    h, w = mask_np.shape

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_np, connectivity=8
    )

    boxes = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])

        # add padding — all values converted to Python int
        pad_x = int(bw * padding)
        pad_y = int(bh * padding)
        x1 = int(max(0, bx - pad_x))
        y1 = int(max(0, by - pad_y))
        x2 = int(min(w, bx + bw + pad_x))
        y2 = int(min(h, by + bh + pad_y))

        boxes.append([x1, y1, x2, y2])

    return boxes if boxes else [[0, 0, int(w), int(h)]]


# ─────────────────────────────────────────────
# STAGE 3.5 INFERENCE
# ─────────────────────────────────────────────
def run_sam2_refinement(processor, model, image_pil, segformer_mask_np):
    """
    Refine SegFormer segmentation mask using SAM 2.

    Strategy (follows Naddaf-Sh et al. 2025):
    - Use SegFormer mask blobs as box prompts to SAM 2
    - SAM 2 produces sharper, more accurate boundaries
    - Combine all per-blob SAM 2 outputs into final mask

    Args:
        processor         : Sam2Processor
        model             : Sam2Model with LoRA weights
        image_pil         : original PIL image (RGB)
        segformer_mask_np : numpy (H, W) binary uint8 from SegFormer

    Returns:
        refined_mask_np : numpy (H, W) binary uint8
        improved        : bool — True if SAM 2 refined successfully
    """
    orig_w, orig_h = image_pil.size

    # get bounding boxes from SegFormer mask blobs
    boxes = mask_to_boxes(segformer_mask_np)

    if not boxes:
        return segformer_mask_np, False

    # run SAM 2 with all boxes as prompts
    try:
        inputs = processor(
            images      = image_pil,
            input_boxes = [boxes],      # list of list of boxes
            return_tensors = "pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(
                **{k: v for k, v in inputs.items()
                   if k != "ground_truth_mask"},
                multimask_output = False,
            )

        # pred_masks: (1, num_boxes, 1, H, W)
        pred_masks = outputs.pred_masks

        # combine all per-box masks with OR
        # (each box → one mask → union = full defect region)
        combined = torch.zeros(
            1, 1, pred_masks.shape[-2], pred_masks.shape[-1],
            device=DEVICE
        )
        for b in range(pred_masks.shape[1]):
            mask_b    = pred_masks[:, b, :, :, :]   # (1,1,H,W)
            prob_b    = torch.sigmoid(mask_b)
            combined  = torch.maximum(combined, prob_b)

        # threshold and resize to original image size
        refined = (combined > THRESHOLD).float()
        refined = F.interpolate(
            refined,
            size=(orig_h, orig_w),
            mode="nearest"
        )
        refined_np = refined.squeeze().cpu().numpy().astype(np.uint8) * 255

        return refined_np, True

    except Exception as e:
        print(f"  SAM 2 refinement error: {e} — using SegFormer mask")
        return segformer_mask_np * 255, False


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys, cv2

    BASE_DIR = r"C:\Users\Sam\Desktop\weld-defect-detection"
    CKPT     = os.path.join(BASE_DIR, "models/sam2/sam2_lora_best.pth")
    CACHE    = os.path.join(BASE_DIR, "models/sam2")

    if not os.path.exists(CKPT):
        print(f"Checkpoint not found: {CKPT}")
        print("Train SAM 2 first: python training/finetune_sam2_lora.py")
        sys.exit()

    # load models
    processor, model = load_sam2(CKPT, cache_dir=CACHE)

    # test on one WDXI image
    img_dir  = os.path.join(BASE_DIR, "data/reviewed/WDXI/datasets1/images")
    mask_dir = os.path.join(BASE_DIR, "data/reviewed/WDXI/datasets1/labels")
    fname    = sorted(os.listdir(img_dir))[0]

    image_pil  = Image.open(os.path.join(img_dir,  fname)).convert("RGB")
    mask_np    = np.array(Image.open(
        os.path.join(mask_dir, fname)).convert("L"))
    mask_bin   = (mask_np > 127).astype(np.uint8)

    print(f"\nTesting on: {fname}")
    print(f"SegFormer defect pixels: {mask_bin.sum()}")

    refined_mask, improved = run_sam2_refinement(
        processor, model, image_pil, mask_bin
    )

    print(f"SAM 2 defect pixels   : {(refined_mask > 0).sum()}")
    print(f"Refined successfully  : {improved}")

    # save comparison
    out_dir = os.path.join(BASE_DIR, "data/auto_labeled")
    os.makedirs(out_dir, exist_ok=True)

    # side by side: original mask vs SAM 2 refined
    h, w   = mask_bin.shape
    canvas = np.zeros((h, w*2 + 10), dtype=np.uint8)
    canvas[:, :w]        = mask_bin * 255
    canvas[:, w+10:]     = refined_mask

    cv2.imwrite(os.path.join(out_dir, "sam2_comparison.png"), canvas)
    print(f"\nComparison saved: {out_dir}/sam2_comparison.png")
    print("Left = SegFormer mask | Right = SAM 2 refined mask")