"""
Convert WDXI dataset to nnU-Net v2 raw dataset format.

nnU-Net requires a strict folder structure:
  nnUNet_raw/DatasetXXX_NAME/
    imagesTr/case_0000_0000.png   <- training images (channel suffix _0000)
    labelsTr/case_0000.png        <- training labels (values 0/1, NOT 0/255)
    imagesTs/case_0000_0000.png   <- test images (optional, we use for eval)
    dataset.json

We use WDXI datasets1 as nnU-Net's training set (it will do its own
internal 5-fold cross-validation within this set) and datasets2 as
a held-out test set for final comparison against SegFormer — same
split roles SegFormer used, so the comparison is fair.

Run once locally or on Kaggle before training:
  python scripts/prepare_nnunet_dataset.py
"""

import os
import shutil
import json
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────
# PATHS — set BASE_DIR for your environment
# ─────────────────────────────────────────────
BASE_DIR = r"C:\Users\Sam\Desktop\weld-defect-detection"
# BASE_DIR = "/kaggle/working/weld-defect-detection"    # Kaggle version

TRAIN_IMG_SRC  = f"{BASE_DIR}/data/reviewed/WDXI/datasets1/images"
TRAIN_MASK_SRC = f"{BASE_DIR}/data/reviewed/WDXI/datasets1/labels"
TEST_IMG_SRC   = f"{BASE_DIR}/data/reviewed/WDXI/datasets2/images"
TEST_MASK_SRC  = f"{BASE_DIR}/data/reviewed/WDXI/datasets2/labels"   # kept aside for eval, not given to nnU-Net

# nnU-Net expects these three env-var directories to exist
NNUNET_RAW          = f"{BASE_DIR}/nnUNet_raw"
NNUNET_PREPROCESSED = f"{BASE_DIR}/nnUNet_preprocessed"
NNUNET_RESULTS      = f"{BASE_DIR}/nnUNet_results"

DATASET_ID   = 501                      # any 3-digit id not already used
DATASET_NAME = f"Dataset{DATASET_ID}_WeldDefect"
DATASET_DIR  = f"{NNUNET_RAW}/{DATASET_NAME}"


def ensure_dirs():
    for d in [NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS,
              f"{DATASET_DIR}/imagesTr", f"{DATASET_DIR}/labelsTr",
              f"{DATASET_DIR}/imagesTs"]:
        os.makedirs(d, exist_ok=True)


def convert_split(img_src, mask_src, img_dst, label_dst, prefix, save_labels=True):
    """Copy images with _0000 suffix, convert masks to 0/1 label maps."""
    files = sorted([f for f in os.listdir(img_src)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    case_ids = []
    for i, fname in enumerate(files):
        case_id = f"{prefix}_{i:04d}"
        case_ids.append(case_id)

        # image — grayscale, channel 0
        img = Image.open(os.path.join(img_src, fname)).convert("L")
        img.save(os.path.join(img_dst, f"{case_id}_0000.png"))

        if save_labels:
            mask_path = os.path.join(mask_src, fname)
            mask = np.array(Image.open(mask_path).convert("L"))
            label = (mask > 127).astype(np.uint8)   # 0/1, NOT 0/255
            Image.fromarray(label).save(os.path.join(label_dst, f"{case_id}.png"))

    return case_ids


def write_dataset_json(num_training):
    dataset_json = {
        "channel_names": {"0": "grayscale"},
        "labels": {
            "background": 0,
            "defect": 1
        },
        "numTraining": num_training,
        "file_ending": ".png"
    }
    with open(f"{DATASET_DIR}/dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)


def main():
    print("Preparing nnU-Net raw dataset...")
    ensure_dirs()

    print("Converting training set (WDXI datasets1)...")
    train_ids = convert_split(
        TRAIN_IMG_SRC, TRAIN_MASK_SRC,
        f"{DATASET_DIR}/imagesTr", f"{DATASET_DIR}/labelsTr",
        prefix="train", save_labels=True
    )
    print(f"  {len(train_ids)} training cases")

    print("Converting held-out test set (WDXI datasets2, images only)...")
    test_ids = convert_split(
        TEST_IMG_SRC, None,
        f"{DATASET_DIR}/imagesTs", None,
        prefix="test", save_labels=False
    )
    print(f"  {len(test_ids)} test cases (labels kept aside for evaluation)")

    write_dataset_json(num_training=len(train_ids))

    print(f"\n{'='*50}")
    print("nnU-Net dataset ready")
    print(f"{'='*50}")
    print(f"Dataset dir : {DATASET_DIR}")
    print(f"Dataset ID  : {DATASET_ID}")
    print(f"\nSet these environment variables before running nnU-Net CLI:")
    print(f'  export nnUNet_raw="{NNUNET_RAW}"')
    print(f'  export nnUNet_preprocessed="{NNUNET_PREPROCESSED}"')
    print(f'  export nnUNet_results="{NNUNET_RESULTS}"')
    print(f"\n(On Windows CMD use 'set' instead of 'export')")
    print(f"\nNext steps:")
    print(f"  1. nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity")
    print(f"  2. nnUNetv2_train {DATASET_ID} 2d 0")
    print(f"  3. python scripts/evaluate_nnunet.py")


if __name__ == "__main__":
    main()
