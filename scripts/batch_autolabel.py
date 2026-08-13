"""
Grounding DINO — Auto-labeling Script
Runs on unlabeled images in data/raw/ and saves
bounding box labels to data/auto_labeled/
License: Apache 2.0 (safe for industrial use)
"""

import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ─────────────────────────────────────────────
# 1. PATHS
# ─────────────────────────────────────────────
BASE_DIR        = r"C:\Users\Sam\Desktop\weld-defect-detection"
INPUT_DIR       = f"{BASE_DIR}/data/raw"           # unlabeled images go here
OUTPUT_LABELS   = f"{BASE_DIR}/data/auto_labeled"  # COCO json output
OUTPUT_VIZ      = f"{BASE_DIR}/data/auto_labeled/visualizations"  # preview images
MODEL_CACHE     = f"{BASE_DIR}/models/grounding"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# 2. TEXT PROMPTS
# What to search for in each image.
# Use "." to separate multiple prompts.
# ─────────────────────────────────────────────
TEXT_PROMPT     = "weld seam. weld joint. weld defect. crack. porosity."
BOX_THRESHOLD   = 0.30   # lower = more detections (adjust if missing real welds)
TEXT_THRESHOLD  = 0.25

# ─────────────────────────────────────────────
# 3. LOAD MODEL
# ─────────────────────────────────────────────
print("Loading Grounding DINO 1.5 Edge (Apache 2.0)...")
MODEL_ID = "IDEA-Research/grounding-dino-tiny"   # tiny = fast, good for auto-labeling

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    cache_dir=MODEL_CACHE
)
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    MODEL_ID,
    cache_dir=MODEL_CACHE
).to(DEVICE)
model.eval()
print("Model loaded.\n")

# ─────────────────────────────────────────────
# 4. SETUP OUTPUT DIRS
# ─────────────────────────────────────────────
os.makedirs(OUTPUT_LABELS, exist_ok=True)
os.makedirs(OUTPUT_VIZ,    exist_ok=True)

# ─────────────────────────────────────────────
# 5. COCO FORMAT INIT
# ─────────────────────────────────────────────
coco = {
    "info"        : {"description": "Grounding DINO auto-labels for weld images"},
    "categories"  : [{"id": 1, "name": "weld_region", "supercategory": "weld"}],
    "images"      : [],
    "annotations" : []
}
img_id = 0
ann_id = 0

# ─────────────────────────────────────────────
# 6. PROCESS IMAGES
# ─────────────────────────────────────────────
extensions = (".png", ".jpg", ".jpeg", ".bmp")
images     = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(extensions)
]

if not images:
    print(f"No images found in {INPUT_DIR}")
    print("Add your unlabeled shop-floor images there and run again.")
    exit()

print(f"Processing {len(images)} images...\n")

results_summary = {"total": len(images), "with_detections": 0, "no_detection": 0}

for fname in tqdm(images, desc="Auto-labeling"):
    img_path = os.path.join(INPUT_DIR, fname)

    try:
        image = Image.open(img_path).convert("RGB")
        w, h  = image.size

        # run inference
        inputs  = processor(
            images=image,
            text=TEXT_PROMPT,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        # post-process
        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold  = BOX_THRESHOLD,
            text_threshold = TEXT_THRESHOLD,
            target_sizes   = [(h, w)]
        )[0]

        boxes  = result["boxes"].cpu().numpy()    # (N, 4) xyxy
        scores = result["scores"].cpu().numpy()   # (N,)
        labels = result["labels"]                 # list of str

        # add to COCO
        coco["images"].append({
            "id"        : img_id,
            "file_name" : fname,
            "width"     : w,
            "height"    : h
        })

        if len(boxes) > 0:
            results_summary["with_detections"] += 1
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                bw = float(x2 - x1)
                bh = float(y2 - y1)
                coco["annotations"].append({
                    "id"          : ann_id,
                    "image_id"    : img_id,
                    "category_id" : 1,
                    "bbox"        : [float(x1), float(y1), bw, bh],
                    "area"        : bw * bh,
                    "score"       : float(score),
                    "label"       : label,
                    "iscrowd"     : 0
                })
                ann_id += 1

            # save visualization
            draw = ImageDraw.Draw(image)
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
                draw.text((x1, max(0, y1 - 16)),
                          f"{label} {score:.2f}", fill="lime")
            viz_path = os.path.join(OUTPUT_VIZ, f"viz_{fname}")
            image.save(viz_path)

        else:
            results_summary["no_detection"] += 1

        img_id += 1

    except Exception as e:
        print(f"  Error on {fname}: {e}")
        continue

# ─────────────────────────────────────────────
# 7. SAVE COCO JSON
# ─────────────────────────────────────────────
json_path = os.path.join(OUTPUT_LABELS, "_auto_labels.coco.json")
with open(json_path, "w") as f:
    json.dump(coco, f, indent=2)

print(f"\n{'='*50}")
print(f"AUTO-LABELING COMPLETE")
print(f"{'='*50}")
print(f"Total images      : {results_summary['total']}")
print(f"With detections   : {results_summary['with_detections']}")
print(f"No detection      : {results_summary['no_detection']}")
print(f"Total annotations : {ann_id}")
print(f"COCO JSON saved   : {json_path}")
print(f"Visualizations    : {OUTPUT_VIZ}")
print(f"\nNext step: Review images in data/auto_labeled/visualizations/")
print(f"Then move reviewed images to data/reviewed/ for training.")
