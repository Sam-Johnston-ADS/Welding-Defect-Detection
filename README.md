# Industrial-WeldVision-AI

**Industrial-WeldVision-AI** is an AI-based welding inspection system designed for surface weld defect segmentation, severity analysis, region-wise defect reporting, and inspection report generation.

This project is developed as an **industrial-level AI inspection framework**, not as a basic hackathon demo. The first completed module focuses on **surface weld defect segmentation using U-Net**. The system is designed to be extended with weld/non-weld validation, weld bead inspection, defect type classification, X-ray inspection, ultrasonic inspection, and multimodal fusion.

---

## 1. Project Objective

The main objective of this project is to support automated weld inspection by using deep learning to detect surface defect regions from weld images and generate inspection outputs that are useful for industrial review.

The system answers:

- Is there a visible surface defect?
- Where is the defect located?
- How much area does the defect cover?
- How many defect regions are present?
- What is the severity level?
- What action is recommended?
- Can the result be exported as an inspection report?

---

## 2. Current Completed Module

### Surface Weld Defect Segmentation Module

The current implemented module uses a U-Net segmentation model to predict surface defect masks from weld images.

Current capabilities:

- Dataset split generation
- U-Net model training
- Independent model evaluation
- Single-image inference
- Batch-folder inference
- Predicted defect mask generation
- Probability map generation
- Defect overlay generation
- Region-wise bounding box detection
- Defect percentage calculation
- Severity classification
- JSON and text inspection reports
- Batch severity summary
- Professional report generation
- Streamlit dashboard interface

---

## 3. System Workflow

```text
Surface Weld Image
        ↓
Image Preprocessing
        ↓
U-Net Defect Segmentation
        ↓
Predicted Defect Mask
        ↓
Region-wise Defect Extraction
        ↓
Defect Percentage Calculation
        ↓
Severity Analysis
        ↓
Inspection Recommendation
        ↓
JSON / TXT / HTML / Markdown Reports
        ↓
Streamlit Dashboard Visualization
```

---

## 4. Final Surface Module Results

The trained U-Net model was independently evaluated on 152 unseen test images.

| Metric | Result |
|---|---:|
| Test Images | 152 |
| Dice Score | 0.9489 |
| IoU Score | 0.9027 |
| Precision | 0.9499 |
| Recall | 0.9479 |
| F1 Score | 0.9489 |
| Accuracy | 0.9982 |

The model was also applied to 1512 weld images for batch-level severity analysis.

| Severity | Count |
|---|---:|
| No Defect | 0 |
| Minor | 504 |
| Moderate | 719 |
| Serious | 269 |
| Critical | 20 |

Batch defect statistics:

| Item | Value |
|---|---:|
| Mean Defect Percentage | 1.8923% |
| Minimum Defect Percentage | 0.1137% |
| Maximum Defect Percentage | 6.9344% |

---

## 5. Directory Structure

```text
Industrial-WeldVision-AI/
│
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── main.py
│
├── configs/
│   ├── surface_unet.yaml
│   ├── surface_yolov8.yaml
│   ├── xray_classifier.yaml
│   ├── ultrasonic_model.yaml
│   └── fusion_config.yaml
│
├── datasets/
│   ├── raw/
│   │   ├── surface/
│   │   │   ├── images/
│   │   │   └── masks/
│   │   ├── xray/
│   │   └── ultrasonic/
│   │
│   ├── processed/
│   └── splits/
│       ├── surface_train.csv
│       ├── surface_val.csv
│       └── surface_test.csv
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_surface_unet_training.ipynb
│   └── 03_surface_model_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   └── create_splits.py
│   │
│   ├── preprocessing/
│   │   ├── image_preprocessing.py
│   │   └── mask_preprocessing.py
│   │
│   ├── models/
│   │   └── unet.py
│   │
│   ├── training/
│   │   └── train_surface_unet.py
│   │
│   ├── evaluation/
│   │   └── evaluate_surface_model.py
│   │
│   ├── inference/
│   │   └── predict_surface.py
│   │
│   ├── reporting/
│   │   ├── severity_analysis.py
│   │   └── report_generator.py
│   │
│   └── utils/
│
├── models/
│   └── surface/
│       └── unet/
│           └── checkpoints/
│               ├── best_model.pth
│               └── latest_checkpoint.pth
│
├── results/
│   └── surface/
│       ├── reports/
│       ├── inference/
│       ├── evaluation/
│       └── plots/
│
├── dashboard/
│   └── app.py
│
├── api/
├── tests/
├── docs/
└── deployment/
```

---

## 6. Installation

### Step 1: Create virtual environment

```bat
cd /d D:\Project\Industrial-WeldVision-AI
py -3.12 -m venv .venv
.venv\Scripts\activate
```

### Step 2: Upgrade pip

```bat
python -m pip install --upgrade pip setuptools wheel
```

### Step 3: Install dependencies

```bat
pip install numpy==1.26.4 pandas==2.2.2 matplotlib==3.8.4 opencv-python==4.9.0.80 scikit-learn==1.4.2 pillow tqdm pyyaml streamlit
```

### Step 4: Install PyTorch

For CPU:

```bat
pip install torch torchvision
```

For NVIDIA GPU, install the correct CUDA-supported PyTorch version from the official PyTorch installation command.

### Step 5: Verify environment

```bat
python -c "import numpy, pandas, matplotlib, cv2, torch; print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('matplotlib OK'); print('opencv', cv2.__version__); print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
```

---

## 7. Dataset Format

The current surface segmentation module expects:

```text
datasets/raw/surface/
├── images/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
│
└── masks/
    ├── 00000.png
    ├── 00001.png
    └── ...
```

Mask convention:

```text
0   = background
255 = defect region
```

During training, masks are converted into:

```text
0 = background
1 = defect
```

---

## 8. Create Dataset Splits

Run from project root:

```bat
python src\data\create_splits.py
```

This creates:

```text
datasets/splits/
├── surface_all.csv
├── surface_train.csv
├── surface_val.csv
├── surface_test.csv
└── surface_split_summary.csv
```

For old dataset path:

```bat
python src\data\create_splits.py --image-dir datasets\surface\images --mask-dir datasets\surface\masks
```

---

## 9. Train Surface U-Net Model

### Quick test training

```bat
python src\training\train_surface_unet.py --epochs 3 --batch-size 2 --image-width 256 --image-height 128 --model-type small_unet
```

### Full training

```bat
python src\training\train_surface_unet.py --epochs 50 --batch-size 4 --image-width 512 --image-height 256 --model-type unet
```

### Resume training

```bat
python src\training\train_surface_unet.py --resume models\surface\unet\checkpoints\latest_checkpoint.pth --epochs 55
```

Training outputs:

```text
models/surface/unet/checkpoints/best_model.pth
models/surface/unet/checkpoints/latest_checkpoint.pth
results/surface/reports/surface_unet_training_history.csv
results/surface/reports/surface_unet_training_summary.txt
results/surface/plots/
```

---

## 10. Evaluate Trained Model

Run:

```bat
python src\evaluation\evaluate_surface_model.py
```

This generates:

```text
results/surface/reports/surface_evaluation_metrics.csv
results/surface/reports/surface_per_image_metrics.csv
results/surface/reports/surface_evaluation_prediction_report.csv
results/surface/reports/surface_model_evaluation_summary.txt

results/surface/evaluation/masks/
results/surface/evaluation/overlays/
results/surface/evaluation/comparisons/
results/surface/evaluation/plots/
```

---

## 11. Run Inference

### Single image prediction

```bat
python src\inference\predict_surface.py --image datasets\raw\surface\images\00000.png
```

### Folder prediction

```bat
python src\inference\predict_surface.py --input-dir datasets\raw\surface\images
```

### Industrial-style region-wise prediction

```bat
python src\inference\predict_surface.py --input-dir datasets\raw\surface\images --min-region-area 25
```

Inference outputs:

```text
results/surface/inference/
├── masks/
├── probability_maps/
├── overlays/
└── reports/
    ├── image_name_inspection_report.json
    ├── image_name_inspection_summary.txt
    └── surface_batch_prediction_report.csv
```

---

## 12. Severity Analysis

Analyze full batch prediction report:

```bat
python src\reporting\severity_analysis.py --batch-csv results\surface\inference\reports\surface_batch_prediction_report.csv --output results\surface\inference\reports\batch_severity_summary.txt
```

Analyze one predicted mask:

```bat
python src\reporting\severity_analysis.py --mask results\surface\inference\masks\00000_pred_mask.png --output results\surface\inference\reports\00000_severity.json
```

Default severity rule:

| Defect Percentage | Severity |
|---|---|
| 0% | No Defect |
| 0 - 1% | Minor |
| 1 - 3% | Moderate |
| 3 - 6% | Serious |
| > 6% | Critical |

---

## 13. Generate Professional Reports

### Batch report

```bat
python src\reporting\report_generator.py --mode batch --batch-csv results\surface\inference\reports\surface_batch_prediction_report.csv --severity-summary results\surface\inference\reports\batch_severity_summary.txt --evaluation-metrics results\surface\reports\surface_evaluation_metrics.csv --output-dir results\surface\reports\generated
```

### Final surface module report

```bat
python src\reporting\report_generator.py --mode final --training-summary results\surface\reports\surface_unet_training_summary.txt --evaluation-summary results\surface\reports\surface_model_evaluation_summary.txt --batch-summary results\surface\inference\reports\batch_severity_summary.txt --output-dir results\surface\reports\generated
```

Generated outputs:

```text
results/surface/reports/generated/
├── surface_batch_professional_inspection_report.md
├── surface_batch_professional_inspection_report.txt
├── surface_batch_professional_inspection_report.html
├── surface_batch_professional_inspection_report.json
├── surface_module_final_project_report.md
├── surface_module_final_project_report.txt
├── surface_module_final_project_report.html
└── surface_module_final_project_report.json
```

---

## 14. Run Streamlit Dashboard

```bat
streamlit run dashboard\app.py
```

Dashboard features:

- Upload weld image
- Run surface defect segmentation
- View original image
- View predicted defect mask
- View probability map
- View overlay with defect-region boxes
- View defect percentage
- View severity level
- View repair recommendation
- View individual defect-region table
- Download JSON report
- Download text report
- Download mask
- Download overlay
- View batch severity summary
- View project status

Local URL:

```text
http://localhost:8501
```

---

## 15. Important Limitation

The current model is a binary surface defect segmentation model. It predicts only the defect regions that are represented in the training masks.

It may not detect:

- weld bead geometry issues
- overlap defects
- undercut defects
- poor bead formation
- defect type categories
- non-weld images
- internal defects

This is expected because the current dataset uses binary masks:

```text
background vs defect
```

For industrial deployment, future versions should include:

- weld bead segmentation
- weld/non-weld validation
- normal/defective weld classification
- multi-class defect labeling
- X-ray defect detection
- ultrasonic flaw analysis
- fusion decision engine

---

## 16. Future Scope

Planned industrial extensions:

### 1. Weld / Non-Weld Validation

```text
Input image
    ↓
Weld / Non-weld classifier
    ↓
If weld image → continue inspection
If non-weld image → reject input
```

### 2. Weld Bead Segmentation

Detect the actual weld bead region before defect segmentation.

### 3. Defect Type Classification

Classify defect type:

```text
crack
porosity
undercut
overlap
slag
spatter
```

### 4. X-ray Inspection Module

Detect internal weld defects from radiographic images.

### 5. Ultrasonic Inspection Module

Analyze ultrasonic B-scan or signal data for internal flaw detection.

### 6. Fusion Decision Engine

Combine:

```text
surface inspection
+ x-ray inspection
+ ultrasonic inspection
= final weld quality decision
```

### 7. API Deployment

Deploy model using FastAPI.

### 8. Docker Deployment

Package the system using Docker for production deployment.

---

## 17. Current Project Status

| Module | Status |
|---|---|
| Dataset split generation | Completed |
| Image preprocessing | Completed |
| Mask preprocessing | Completed |
| U-Net model | Completed |
| Training pipeline | Completed |
| Evaluation pipeline | Completed |
| Inference pipeline | Completed |
| Region-wise reporting | Completed |
| Severity analysis | Completed |
| Report generation | Completed |
| Streamlit dashboard | Completed |
| Weld/non-weld validation | Planned |
| Weld bead segmentation | Planned |
| Defect type classification | Planned |
| X-ray module | Planned |
| Ultrasonic module | Planned |
| Fusion module | Planned |

---

## 18. Industrial Use Note

This system is designed as a decision-support tool for weld inspection. It can assist inspectors by highlighting defect regions, calculating defect area, and generating structured reports.

For real industrial deployment, the model must be validated using production-grade weld images, calibrated against applicable welding standards, and reviewed by certified inspection professionals.

---

## 19. Author

**Sam Johnston C**  
B.Tech Artificial Intelligence and Data Science  
St. Joseph College of Engineering

---

## 20. Project Version

```text
Current Version: Surface Inspection Module v1.0
Status: Completed
Next Version: Weld validation + bead inspection module
```
