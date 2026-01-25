# Therapy Response Prediction Module

This module predicts therapy response (Regression/Stable/Progression) from pre-treatment brain MRI scans.

## Overview

The therapy response prediction pipeline analyzes pre-treatment MRI scans and predicts how the tumor will respond to therapy by comparing with post-treatment outcomes.

### Data Sources

| Dataset | Role | Description |
|---------|------|-------------|
| BraTS2025-GLI-PRE-TrainingData | PRE-treatment | Input features for training |
| BraTS2025-GLI-PRE-ValidationData | PRE-treatment | Input features for validation |
| BraTS2024-BraTS-GLI-TrainingData | POST-treatment | Outcome measurement (ground truth) |
| BraTS2024-BraTS-GLI-ValidationData | POST-treatment | Outcome measurement for validation |

### Response Labels

Labels are assigned based on tumor volume change between PRE and POST treatment:

| Label | Criteria | Interpretation |
|-------|----------|----------------|
| **Regression** | Volume decrease > 25% | Tumor responded well to therapy |
| **Stable** | Volume change ±25% | No significant change |
| **Progression** | Volume increase > 25% | Tumor grew despite therapy |

## Pipeline Phases

### Phase 0: Label Generation (Current)

Generate ground-truth response labels by comparing PRE and POST treatment tumor volumes.

```
PRE-treatment MRI → Extract Volume → Compare → Response Label
                                        ↑
POST-treatment MRI → Extract Volume ────┘
```

### Phase 1: Feature Extraction (Planned)

Extract radiomic and deep learning features from PRE-treatment scans.

### Phase 2: Model Training (Planned)

Train ML models to predict therapy response from PRE-treatment features only.

### Phase 3: Inference (Planned)

Predict therapy response for new patients using only PRE-treatment scans.

## Quick Start

### 1. Extract Datasets

```bash
cd "Therapy Prediction"
python extract_datasets.py
```

This extracts the zip files and organizes them into:
- `data/pre_treatment/` - BraTS 2025 PRE scans
- `data/post_treatment/` - BraTS 2024 POST scans

### 2. Generate Response Labels

```bash
python generate_response_labels.py
```

This creates `response_labels.csv` with columns:
- `Patient_ID` - Unique patient identifier
- `Pre_Volume_ml` - Tumor volume before treatment (ml)
- `Post_Volume_ml` - Tumor volume after treatment (ml)
- `Volume_Change_Percent` - Percentage volume change
- `Response_Label` - Regression/Stable/Progression

### Custom Directories

```bash
python generate_response_labels.py \
    --pre_dir /path/to/pre \
    --post_dir /path/to/post \
    --output custom_labels.csv
```

## File Structure

```
Therapy Prediction/
├── README.md                    # This file
├── extract_datasets.py          # Dataset extraction script
├── generate_response_labels.py  # Phase 0: Label generation
├── response_labels.csv          # Generated labels (output)
├── dataset/                     # Raw zip files
│   ├── BraTS2025-GLI-PRE-Challenge-TrainingData (1).zip
│   ├── BraTS2025-GLI-PRE-Challenge-ValidationData.zip
│   ├── BraTS2024-BraTS-GLI-TrainingData.zip
│   └── BraTS2024-BraTS-GLI-ValidationData.zip
└── data/                        # Extracted data
    ├── pre_treatment/           # PRE-treatment cases
    │   ├── BraTS-GLI-00001-000/
    │   └── ...
    └── post_treatment/          # POST-treatment cases
        ├── BraTS-GLI-00001-000/
        └── ...
```

## Requirements

- Python 3.8+
- nibabel
- numpy

Install dependencies:
```bash
pip install nibabel numpy
```

## Technical Details

### Volume Calculation

Whole tumor volume is computed as the sum of all tumor labels:
- NCR (Necrotic Core) - Label 1
- ED (Edema) - Label 2
- ET (Enhancing Tumor) - Label 3 or 4

Volume (ml) = Voxel Count × Voxel Volume (mm³) / 1000

### Patient Matching

Patients are matched between PRE and POST datasets using the 5-digit patient ID extracted from folder names:
- `BraTS-GLI-00001-000` → Patient ID: `00001`
- `BraTS2024-GLI-00001-000` → Patient ID: `00001`

## Next Steps

After generating labels, you can:
1. Analyze the label distribution
2. Extract features from PRE-treatment scans
3. Train ML models for response prediction
4. Validate on held-out test set
