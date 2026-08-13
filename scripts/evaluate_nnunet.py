"""
Evaluate a trained nnU-Net model on the WDXI held-out test set
(datasets2) and compute IoU/Dice using the SAME formulas as the
SegFormer/SAM2 scripts, so the numbers are directly comparable.

Run AFTER nnU-Net training + prediction:
  nnUNetv2_predict -i nnUNet_raw/Dataset501_WeldDefect/imagesTs \
                    -o nnUNet_predictions \
                    -d 501 -c 2d -f 0

Then:
  python scripts/evaluate_nnunet.py
"""

import os
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = r"C:\Users\Sam\Desktop\weld-defect-detection"
# BASE_DIR = "/kaggle/working/weld-defect-detection"

TEST_IMG_SRC   = f"{BASE_DIR}/data/reviewed/WDXI/datasets2/images"
TEST_MASK_SRC  = f"{BASE_DIR}/data/reviewed/WDXI/datasets2/labels"   # ground truth
PRED_DIR       = f"{BASE_DIR}/nnUNet_predictions"                    # nnU-Net output
RESULTS_PATH   = f"{BASE_DIR}/evaluation/reports/nnunet_results.json"


def compute_iou_dice(pred_bin, gt_bin):
    """Same formulas used for SegFormer/SAM2 evaluation."""
    inter = int((pred_bin & gt_bin).sum())
    union = int((pred_bin | gt_bin).sum())
    iou   = inter / (union + 1e-8)

    pred_sum = int(pred_bin.sum())
    gt_sum   = int(gt_bin.sum())
    dice     = (2 * inter) / (pred_sum + gt_sum + 1e-8)

    return iou, dice


def main():
    # map nnU-Net prediction filenames back to original test filenames
    # (prepare_nnunet_dataset.py named them test_0000, test_0001, ...
    #  in the SAME sorted order as the original datasets2/images folder)
    orig_files = sorted([f for f in os.listdir(TEST_IMG_SRC)
                          if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    pred_files = sorted([f for f in os.listdir(PRED_DIR)
                          if f.lower().endswith(".png")])

    if len(pred_files) != len(orig_files):
        print(f"WARNING: {len(pred_files)} predictions found but "
              f"{len(orig_files)} original test images exist.")
        print("Make sure nnUNetv2_predict has finished running on the full test set.")

    ious, dices = [], []
    per_image   = []

    n = min(len(pred_files), len(orig_files))
    for i in range(n):
        orig_fname = orig_files[i]
        pred_fname = pred_files[i]

        gt_mask = np.array(Image.open(
            os.path.join(TEST_MASK_SRC, orig_fname)).convert("L"))
        gt_bin  = (gt_mask > 127).astype(np.uint8)

        pred_mask = np.array(Image.open(
            os.path.join(PRED_DIR, pred_fname)).convert("L"))
        # nnU-Net outputs label maps with values 0/1 directly
        pred_bin  = (pred_mask > 0).astype(np.uint8)

        if pred_bin.shape != gt_bin.shape:
            pred_img = Image.fromarray(pred_bin * 255).resize(
                (gt_bin.shape[1], gt_bin.shape[0]), Image.NEAREST)
            pred_bin = (np.array(pred_img) > 127).astype(np.uint8)

        iou, dice = compute_iou_dice(pred_bin, gt_bin)
        ious.append(iou)
        dices.append(dice)
        per_image.append({"file": orig_fname, "iou": round(iou, 4), "dice": round(dice, 4)})

    mean_iou  = float(np.mean(ious))
    mean_dice = float(np.mean(dices))

    print(f"\n{'='*50}")
    print(f"nnU-Net Evaluation Results (WDXI datasets2 test set)")
    print(f"{'='*50}")
    print(f"Images evaluated : {n}")
    print(f"Mean IoU         : {mean_iou:.4f}")
    print(f"Mean Dice        : {mean_dice:.4f}")
    print(f"\nComparison:")
    print(f"  U-Net (baseline)      : IoU 0.5601")
    print(f"  SegFormer-B2          : IoU 0.6085")
    print(f"  SAM 2 + LoRA (v1)     : IoU 0.6964")
    print(f"  nnU-Net (this run)    : IoU {mean_iou:.4f}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    import json
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "mean_iou"  : mean_iou,
            "mean_dice" : mean_dice,
            "n_images"  : n,
            "per_image" : per_image
        }, f, indent=2)
    print(f"\nFull results saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
