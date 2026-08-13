import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import json

# ─────────────────────────────────────────────
# 1. PATHS
# ─────────────────────────────────────────────
BASE_DIR     = "/content/weld-defect-detection"
RIAWELC_DIR  = f"{BASE_DIR}/data/reviewed/RIAWELC/DB - Copy"
MODEL_DIR    = f"{BASE_DIR}/models/preclassifier"
DRIVE_SAVE   = "/content/drive/MyDrive/preclassifier_best.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# 2. CONFIG
# ─────────────────────────────────────────────
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS     = 20
LR         = 1e-4

# ─────────────────────────────────────────────
# 3. DATASET
# Binary classification:
#   label 0 = no defect (NoDifetto)
#   label 1 = defect    (Difetto1 + Difetto2 + Difetto4)
# ─────────────────────────────────────────────
DEFECT_FOLDERS    = ["Difetto1", "Difetto2", "Difetto4"]
NO_DEFECT_FOLDER  = "NoDifetto"

class PreClassifierDataset(Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform
        self.samples   = []   # list of (image_path, label)

        split_dir = os.path.join(RIAWELC_DIR, split)

        # defect images → label 1
        for folder in DEFECT_FOLDERS:
            folder_path = os.path.join(split_dir, folder)
            if not os.path.exists(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append(
                        (os.path.join(folder_path, fname), 1)
                    )

        # no defect images → label 0
        no_defect_path = os.path.join(split_dir, NO_DEFECT_FOLDER)
        if os.path.exists(no_defect_path):
            for fname in os.listdir(no_defect_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append(
                        (os.path.join(no_defect_path, fname), 0)
                    )

        print(f"  {split}: {len(self.samples)} images "
              f"({sum(1 for _,l in self.samples if l==1)} defect, "
              f"{sum(1 for _,l in self.samples if l==0)} no-defect)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# ─────────────────────────────────────────────
# 4. TRANSFORMS
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# 5. DATALOADERS
# ─────────────────────────────────────────────
print("Loading datasets...")
train_dataset = PreClassifierDataset("training",   train_transform)
val_dataset   = PreClassifierDataset("validation", val_transform)
test_dataset  = PreClassifierDataset("testing",    val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2)

# ─────────────────────────────────────────────
# 6. MODEL
# MobileNetV3 — lightweight, fast, good for binary gate
# ─────────────────────────────────────────────
model = models.efficientnet_b3(weights="IMAGENET1K_V1")

# replace final layer with binary output
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features, 1
)
model = model.to(DEVICE)

print(f"\nModel: EfficientNetB3")
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

# ─────────────────────────────────────────────
# 7. LOSS + OPTIMIZER
# ─────────────────────────────────────────────
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=3, factor=0.5
)

# ─────────────────────────────────────────────
# 8. TRAINING LOOP
# ─────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
best_val_acc = 0.0
history      = []

print("\nStarting training...")
for epoch in range(1, EPOCHS + 1):

    # ── train ──
    model.train()
    train_loss = correct = total = 0

    for images, labels in tqdm(train_loader,
                                desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds       = (torch.sigmoid(outputs) > 0.5).float()
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    train_acc  = correct / total
    train_loss /= len(train_loader)

    # ── validate ──
    model.eval()
    val_loss = val_correct = val_total = 0
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            outputs   = model(images)
            val_loss += criterion(outputs, labels).item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

            # confusion matrix components
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    val_acc        = val_correct / val_total
    val_loss      /= len(val_loader)
    precision      = tp / (tp + fp + 1e-8)
    recall         = tp / (tp + fn + 1e-8)
    rejection_rate = tn / (tn + fp + 1e-8)  # non-weld rejection rate

    scheduler.step(val_acc)

    print(f"\nEpoch {epoch:02d} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
          f"Precision: {precision:.4f} | Recall: {recall:.4f} | "
          f"Rejection Rate: {rejection_rate:.4f}")

    history.append({
        "epoch": epoch,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "precision": precision,
        "recall": recall,
        "rejection_rate": rejection_rate
    })

    # save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "epoch"     : epoch,
            "model_state_dict": model.state_dict(),
            "val_acc"   : val_acc,
            "precision" : precision,
            "recall"    : recall,
            "rejection_rate": rejection_rate,
        }, os.path.join(MODEL_DIR, "preclassifier_best.pth"))
        print(f"  ✅ Best model saved → Val Acc: {best_val_acc:.4f}")

# save training history
with open(os.path.join(MODEL_DIR, "history.json"), "w") as f:
    json.dump(history, f, indent=2)

print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.4f}")

# ─────────────────────────────────────────────
# 9. FINAL TEST EVALUATION
# ─────────────────────────────────────────────
print("\nRunning final test evaluation...")

checkpoint = torch.load(
    os.path.join(MODEL_DIR, "preclassifier_best.pth"),
    map_location=DEVICE
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2)

tp = fp = tn = fn = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)
        preds  = (torch.sigmoid(model(images)) > 0.5).float()
        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

test_acc       = (tp + tn) / (tp + fp + tn + fn)
test_precision = tp / (tp + fp + 1e-8)
test_recall    = tp / (tp + fn + 1e-8)
test_rejection = tn / (tn + fp + 1e-8)

print(f"\n{'='*50}")
print(f"FINAL TEST RESULTS")
print(f"{'='*50}")
print(f"Accuracy        : {test_acc:.4f}")
print(f"Precision       : {test_precision:.4f}")
print(f"Recall          : {test_recall:.4f}")
print(f"Rejection Rate  : {test_rejection:.4f}  (target ≥ 0.98)")
print(f"{'='*50}")

# ─────────────────────────────────────────────
# 10. SAVE TO GOOGLE DRIVE
# ─────────────────────────────────────────────
import shutil
shutil.copy(
    os.path.join(MODEL_DIR, "preclassifier_best.pth"),
    DRIVE_SAVE
)
print(f"\nModel saved to Google Drive: {DRIVE_SAVE}")
