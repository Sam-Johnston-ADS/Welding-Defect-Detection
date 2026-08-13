# Industrial Weld Defect Detection System

An end-to-end, cascade-based computer vision pipeline for detecting, segmenting, and
classifying defects in industrial X-ray weld radiographs — built with commercially
licensed, non-pretrained-capable model architectures suitable for production deployment.

---

## ⚠️ Current Status Note

The `cascade.py` currently in this repo runs **SegFormer v2** (pretrained MiT-B2 backbone,
verified test IoU **0.6085**) as its segmentation stage. Through extensive experimentation
documented below, **SegFormer v6** (Strip Pooling + ASPP + Boundary Head, trained fully
from scratch) was found to perform significantly better — verified test IoU **~0.87** across
three independent held-out evaluations — while satisfying a stricter "no pretrained weights"
constraint. **Before production use, update `SEGFORMER_PATH` in `cascade.py` to point to
`models/segformer_v6/segformer_v6_best.pth`** and load it via the `SegformerV6` architecture
class (see `training/train_segformer_v6.py`). This is tracked as an open item.

---

## Pipeline Architecture

```
Raw X-ray Image
      │
      ▼
Stage 1 — Pre-Classifier Gatekeeper (EfficientNetB3)
  Rejects non-weld images. Confidence ≥ 0.5 → continue
      │
      ▼
Stage 2 — Segmentation (SegFormer-B2)
  Predicts a binary defect mask + probability map
      │
      ▼
Stage 3 — Morphological Post-Processing (OpenCV)
  Elliptical open/close + connected-component filtering (< 20px removed)
      │
      ▼
Stage 4 — Confidence & Decision Routing
  Mean region probability over predicted defect pixels
  ACCEPT (≥ 0.70)  |  REVIEW (< 0.70)  |  NO_DEFECT (empty mask)
      │
      ▼
Output: _bbox.png · _overlay.png · _mask.png · _gradcam.png (XAI heatmap)
```

An optional **Stage 3.5 — SAM 2 + LoRA boundary refinement** can be inserted between
Stages 2 and 3 (see `pipeline/stage35_sam2_refine.py`); it auto-activates in earlier
pipeline versions when a paired checkpoint is present.

---

## Verified Model Results

All numbers below were measured on a **held-out test split** (312 images, never used in
training or validation, defined in `data/reviewed/WDXI/combined/split.json`) — not on
validation data. Val-only numbers are explicitly excluded from this table because this
project found real cases (see Known Findings) where val performance did not reflect true
generalization.

| Model | Architecture | Pretrained? | Test IoU | Test Dice |
|---|---|---|---|---|
| U-Net | SeResNet34 backbone | Yes | 0.5601 | 0.6871 |
| SegFormer v2 | MiT-B2, standard decoder | Yes | 0.6085 | — |
| SegFormer v3 | MiT-B2, standard decoder, combined dataset | Yes | 0.8915 | 0.9418 |
| SegFormer v4 | MiT-B2, standard decoder | **No (from scratch)** | 0.7332 | 0.8414 |
| SegFormer v5 | MiT-B2 + Conv Stem | No | 0.5230* | — |
| **SegFormer v6** | **MiT-B2 + Strip Pooling + ASPP + Boundary Head** | **No** | **~0.87** | **~0.93** |
| SAM2 + LoRA (v4-paired) | Box-prompt refinement | No | 0.8144 | 0.8922 |
| SAM2 + LoRA (v6-paired, dense prompt) | Box + mask-prompt refinement | No | 0.8438 | — |

\* v5's checkpoint file was later overwritten by a subsequent training run with different
convergence behavior — the 0.5230 result reflects the *original* run, which showed a large
val→test gap (0.7843 → 0.5230) and was diagnosed as overfitting. This is preserved here as
a documented, honest experimental finding, not hidden.

**Non-segmentation models:**

| Model | Task | Test Accuracy | Test Precision | Test Recall |
|---|---|---|---|---|
| EfficientNetB3 (pre-classifier) | Weld / non-weld gate | 0.9996 | 0.9995 | 1.0000 |
| RF-DETR | Defect detection (crack/porosity/lack-of-penetration) | — | — | mAP@0.5:0.95 = 0.9938 |

RF-DETR was later **removed from the production cascade** per project guidance to simplify
the pipeline to EfficientNetB3 → SegFormer → routing; its detection-confidence and defect-type
classification signal was replaced with SegFormer's own mean-region-probability confidence.

---

## Datasets

| Dataset | Purpose | License | Size |
|---|---|---|---|
| [RIAWELC](https://github.com/stefyste/RIAWELC) | Classification (crack/porosity/lack-of-penetration/no-defect) | Research/educational | 24,407 images |
| WDXI (`admin1523/Weld-defect-detection-datasets`) | Pixel-level segmentation masks | Research/educational | ~3,100 images |

Neither dataset provides commercial-use rights — this project is scoped as research/academic
work. See `docs/LICENSE_NOTES.md` for the full model + dataset license audit trail, including
why NVIDIA's LocateAnything-3B (non-commercial) and standard YOLO were avoided in favor of
Apache 2.0 / BSD-licensed alternatives (RF-DETR, SegFormer, SAM 2, EfficientNetB3).

---

## Known Findings (Worth Reading Before Extending This Work)

- **One of the eleven papers used in the initial literature review was fabricated** — a
  scene-classification paper with "welding defect" substituted for "scene" throughout,
  using unrelated benchmarks (Places, ImageNet, Scene-15). It was excluded from all analysis;
  see the original literature review report for the full audit.
- **Combining train/val across dataset sources can silently inflate reported metrics** if
  done via naive `os.listdir()` scanning instead of an explicit train/val/test split file —
  this project hit that bug twice in independently-written evaluation scripts. Always verify
  `split.json` is actually loaded and filtered against before trusting a "test" number.
- **`strict=False` in `model.load_state_dict()` can silently leave layers at random
  initialization** if checkpoint keys don't match exactly, producing plausible-looking but
  meaningless (near-zero IoU) results with no error raised. Always inspect the `missing`/
  `unexpected` return values.
- **SegFormer v5 (Conv Stem)** looked like a val-time improvement over v4 (0.7843 vs 0.7761)
  but collapsed on the true test split (0.5230) — a clean example of why val-only comparison
  is insufficient for architecture decisions in this project.
- **SegFormer v6 (Strip Pooling + ASPP + Boundary Head)** is the strongest verified result,
  reproduced consistently (0.8716–0.8744 IoU) across three independent evaluation scripts.

---

## Project Structure

```
weld-defect-detection/
├── data/
│   ├── raw/                    # unlabeled images awaiting review
│   ├── reviewed/                # RIAWELC + WDXI datasets, combined pool, split.json
│   ├── auto_labeled/            # Grounding DINO batch-labeling output (future work)
│   └── splits/
├── models/
│   ├── preclassifier/           # EfficientNetB3
│   ├── unet/                    # U-Net baseline + SegFormer v2 (legacy)
│   ├── segformer_v3/ ... v6/    # SegFormer variants, one folder per version
│   ├── yolov8_seg/               # RF-DETR (legacy, removed from production cascade)
│   └── sam2/                     # SAM2 + LoRA checkpoints
├── pipeline/
│   ├── cascade.py                # production inference pipeline
│   └── stage35_sam2_refine.py    # optional SAM2 refinement stage
├── training/
│   ├── train_segformer_v2.py ... v6.py
│   ├── train_segformer_v6_optimized.py   # Tversky loss + LR warmup + grad accumulation
│   └── finetune_sam2_lora*.py
├── scripts/
│   ├── audit_labels.py           # ground-truth mask quality audit
│   ├── prepare_combined_dataset.py
│   └── evaluate_*_test.py        # held-out test-split evaluation scripts (one per model)
├── evaluation/
│   ├── reports/                  # per-image inference outputs
│   └── quantitative_results/     # comparison tables, JSON metrics
├── app_streamlit.py               # Streamlit inspection dashboard
└── docs/
    └── LICENSE_NOTES.md
```

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate segmentation-models-pytorch albumentations \
            opencv-python streamlit pandas rfdetr
```

Model checkpoints are not included in this repo due to size — train from scratch using the
scripts in `training/`, or request pretrained checkpoints separately.

## Running the Dashboard

```bash
streamlit run app_streamlit.py
```

Upload an X-ray weld image and click **Run Inspection**. Outputs (bounding box, overlay,
binary mask, XAI heatmap) are generated per image and saved to `evaluation/reports/`.

## Running Batch Inference

```bash
python pipeline/cascade.py
```

Processes every image in the configured test folder and prints a pipeline summary
(accepted / review / no-defect / rejected counts).

## Evaluating a Model on the Held-Out Test Set

```bash
python scripts/evaluate_all_models_table.py
```

Generates a publication-ready comparison table across every trained checkpoint, filtered
strictly to `split["test"]` — never the full dataset.

---

## License

Model architectures used in this project (EfficientNetB3, RF-DETR, SegFormer, SAM 2) are
Apache 2.0 / BSD-licensed and support commercial use. Datasets (RIAWELC, WDXI) are licensed
for research and educational use only — see `docs/LICENSE_NOTES.md` before any commercial
deployment.
