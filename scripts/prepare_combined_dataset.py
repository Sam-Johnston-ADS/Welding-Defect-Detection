"""
Combine WDXI datasets1 + datasets2 into a single pool and
generate a reproducible random train/val/test split.

NOTE: datasets1 (standard) and datasets2 (cross-condition, harder)
were originally meant as separate train/cross-dataset-test sets.
Combining them removes that built-in domain-shift evaluation —
document this choice in your report.

Run once:
  python scripts/prepare_combined_dataset.py
"""

import os
import json
import random
import shutil
from PIL import Image

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = r"C:\Users\Sam\Desktop\weld-defect-detection"
# BASE_DIR = "/kaggle/working/weld-defect-detection"

DS1_IMG = f"{BASE_DIR}/data/reviewed/WDXI/datasets1/images"
DS1_MSK = f"{BASE_DIR}/data/reviewed/WDXI/datasets1/labels"
DS2_IMG = f"{BASE_DIR}/data/reviewed/WDXI/datasets2/images"
DS2_MSK = f"{BASE_DIR}/data/reviewed/WDXI/datasets2/labels"

COMBINED_IMG = f"{BASE_DIR}/data/reviewed/WDXI/combined/images"
COMBINED_MSK = f"{BASE_DIR}/data/reviewed/WDXI/combined/labels"
SPLIT_PATH   = f"{BASE_DIR}/data/reviewed/WDXI/combined/split.json"

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10   # held out, untouched during training — final reporting only
SEED        = 42


def merge_source(img_src, mask_src, prefix, img_dst, mask_dst):
    files = sorted([f for f in os.listdir(img_src)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    merged_names = []
    for i, fname in enumerate(files):
        new_name = f"{prefix}_{i:04d}.png"
        Image.open(os.path.join(img_src, fname)).convert("RGB").save(
            os.path.join(img_dst, new_name))
        Image.open(os.path.join(mask_src, fname)).convert("L").save(
            os.path.join(mask_dst, new_name))
        merged_names.append(new_name)
    return merged_names


def main():
    os.makedirs(COMBINED_IMG, exist_ok=True)
    os.makedirs(COMBINED_MSK, exist_ok=True)

    print("Merging datasets1 (standard)...")
    names1 = merge_source(DS1_IMG, DS1_MSK, "ds1", COMBINED_IMG, COMBINED_MSK)
    print(f"  {len(names1)} images")

    print("Merging datasets2 (cross-condition)...")
    names2 = merge_source(DS2_IMG, DS2_MSK, "ds2", COMBINED_IMG, COMBINED_MSK)
    print(f"  {len(names2)} images")

    all_names = names1 + names2
    print(f"\nTotal combined pool: {len(all_names)} images")

    # reproducible random split
    rng = random.Random(SEED)
    shuffled = all_names[:]
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)

    train_files = shuffled[:n_train]
    val_files   = shuffled[n_train:n_train + n_val]
    test_files  = shuffled[n_train + n_val:]

    split = {
        "seed"  : SEED,
        "train" : train_files,
        "val"   : val_files,
        "test"  : test_files,
        "source_counts": {"datasets1": len(names1), "datasets2": len(names2)}
    }
    with open(SPLIT_PATH, "w") as f:
        json.dump(split, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Combined dataset ready")
    print(f"{'='*50}")
    print(f"Train : {len(train_files)} ({TRAIN_RATIO*100:.0f}%)")
    print(f"Val   : {len(val_files)} ({VAL_RATIO*100:.0f}%)")
    print(f"Test  : {len(test_files)} ({TEST_RATIO*100:.0f}%)  <- held out, report on this only")
    print(f"Images : {COMBINED_IMG}")
    print(f"Masks  : {COMBINED_MSK}")
    print(f"Split  : {SPLIT_PATH}")


if __name__ == "__main__":
    main()
