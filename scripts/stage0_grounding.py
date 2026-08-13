"""
Grounding DINO — Stage 0 for Cascade Pipeline
Pre-localizes the weld region BEFORE the EfficientNetB3
pre-classifier, so all downstream models only see the
weld area, not the full image.
License: Apache 2.0 (safe for industrial use)
"""

import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────
MODEL_ID        = "IDEA-Research/grounding-dino-tiny"
TEXT_PROMPT     = "dark horizontal band. linear dark region. horizontal stripe."
BOX_THRESHOLD   = 0.20    # lower threshold for X-ray images
TEXT_THRESHOLD  = 0.15
PADDING_RATIO   = 0.20    # 20% padding — more generous for X-ray
MIN_CROP_RATIO  = 0.30    # crop must be at least 30% of image — else use full image
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# 2. LOAD STAGE 0 MODEL
# ─────────────────────────────────────────────
def load_grounding_dino(cache_dir=None):
    print("Loading Grounding DINO (Stage 0)...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, cache_dir=cache_dir
    )
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_ID, cache_dir=cache_dir
    ).to(DEVICE)
    model.eval()
    print("  Grounding DINO loaded (Apache 2.0)")
    return processor, model


# ─────────────────────────────────────────────
# 3. STAGE 0 INFERENCE
# ─────────────────────────────────────────────
def run_stage0_grounding(processor, model, image_pil):
    """
    Finds the weld region in the image using text prompt.

    Returns:
        crop_pil    : PIL Image — cropped weld region
        crop_box    : (x1, y1, x2, y2) in original image coordinates
        confidence  : float — best detection score
        found       : bool — True if weld region found
    """
    w, h = image_pil.size

    inputs = processor(
        images=image_pil,
        text=TEXT_PROMPT,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold  = BOX_THRESHOLD,
        text_threshold = TEXT_THRESHOLD,
        target_sizes   = [(h, w)]
    )[0]

    boxes  = result["boxes"].cpu().numpy()
    scores = result["scores"].cpu().numpy()

    if len(boxes) == 0:
        # no detection — return full image
        print("  Stage 0: No detection — using full image")
        return image_pil, (0, 0, w, h), 0.0, False

    # pick largest area box (not highest confidence)
    # on X-ray images the weld region is large, not a tiny spot
    areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in boxes]
    best_idx = int(np.argmax(areas))
    x1, y1, x2, y2 = boxes[best_idx]
    confidence = float(scores[best_idx])

    # add generous padding
    pad_x = int((x2 - x1) * PADDING_RATIO)
    pad_y = int((y2 - y1) * PADDING_RATIO)
    x1 = max(0, int(x1) - pad_x)
    y1 = max(0, int(y1) - pad_y)
    x2 = min(w, int(x2) + pad_x)
    y2 = min(h, int(y2) + pad_y)

    crop_w = x2 - x1
    crop_h = y2 - y1

    # MINIMUM CROP SIZE CHECK
    # if crop is less than 30% of image in either dimension
    # the detection is unreliable — use full image instead
    if (crop_w / w) < MIN_CROP_RATIO or (crop_h / h) < MIN_CROP_RATIO:
        print(f"  Stage 0: Crop too small ({crop_w}x{crop_h} vs {w}x{h}) "
              f"— using full image")
        return image_pil, (0, 0, w, h), confidence, False

    crop_pil = image_pil.crop((x1, y1, x2, y2))
    return crop_pil, (x1, y1, x2, y2), confidence, True


# ─────────────────────────────────────────────
# 4. UPDATED CASCADE WITH STAGE 0
# Drop this into cascade.py by replacing the
# WeldDefectPipeline.__init__ and predict methods
# ─────────────────────────────────────────────
UPDATED_CASCADE_SNIPPET = """
# In cascade.py — add Stage 0 to __init__:

from scripts.stage0_grounding import load_grounding_dino, run_stage0_grounding

class WeldDefectPipeline:
    def __init__(self):
        # Stage 0 — Grounding DINO
        self.gdino_processor, self.gdino_model = load_grounding_dino(
            cache_dir=f"{BASE_DIR}/models/grounding"
        )
        # ... rest of existing __init__ ...

    def predict(self, image_path):
        image_pil = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image_pil.size

        # ── Stage 0: Grounding DINO pre-localization ──
        crop_pil, crop_box, gdino_conf, weld_found = run_stage0_grounding(
            self.gdino_processor, self.gdino_model, image_pil
        )
        if not weld_found:
            print("  Stage 0: No weld region found — using full image")
        else:
            print(f"  Stage 0: Weld region found (conf: {gdino_conf:.3f}) "
                  f"→ crop {crop_box}")
            image_pil = crop_pil   # downstream stages work on the crop

        # ── Stage 1: Pre-classifier (EfficientNetB3) ──
        # ... rest of existing predict() unchanged ...
"""

# ─────────────────────────────────────────────
# 5. QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    BASE_DIR = r"C:\Users\Sam\Desktop\weld-defect-detection"

    # quick test on one image
    test_folder = os.path.join(
        BASE_DIR,
        "data/reviewed/WDXI/datasets1/images"
    )
    images = [f for f in os.listdir(test_folder)
              if f.lower().endswith((".png", ".jpg"))]

    if not images:
        print("No test images found.")
        sys.exit()

    test_img_path = os.path.join(test_folder, images[0])
    print(f"Testing on: {images[0]}")

    processor, model = load_grounding_dino(
        cache_dir=os.path.join(BASE_DIR, "models/grounding")
    )

    image_pil = Image.open(test_img_path).convert("RGB")
    crop, box, conf, found = run_stage0_grounding(processor, model, image_pil)

    print(f"\nResult:")
    print(f"  Weld found  : {found}")
    print(f"  Confidence  : {conf:.3f}")
    print(f"  Crop box    : {box}")
    print(f"  Crop size   : {crop.size}")

    # save crop to check it
    save_path = os.path.join(BASE_DIR, "data/auto_labeled/test_crop.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    crop.save(save_path)
    print(f"  Crop saved  : {save_path}")
    print("\nStage 0 working correctly.")
