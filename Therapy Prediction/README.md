# LUMIERE Therapy Response Prediction

## Overview

This module processes the LUMIERE longitudinal glioma MRI dataset from the MICCAI Brain Tumor Progression Challenge to create a model-ready CSV for therapy response prediction.

## Dataset Structure

The LUMIERE dataset follows this structure:

```
dataset/
├── LUMIERE-ExpertRating-v202211.csv    # Expert RANO annotations
├── Imaging-v202211.zip                  # Imaging data (31 GB)
└── Imaging/                             # Extracted imaging data
    └── Patient-XXX/
        ├── week-000-1/                  # Baseline scan 1 (Pre-Op)
        │   ├── T1.nii.gz
        │   ├── CT1.nii.gz               # Contrast T1
        │   ├── T2.nii.gz
        │   ├── FLAIR.nii.gz
        │   └── HD-GLIO-AUTO-segmentation/
        │       └── registered/
        │           └── segmentation.nii.gz
        ├── week-000-2/                  # Baseline scan 2 (Post-Op)
        ├── week-044/                    # Follow-up week 44
        └── week-056/                    # Follow-up week 56
```

## Phase 0 - Dataset Preparation (No ML)

The `prepare_lumiere_dataset.py` script creates a model-ready CSV by:

1. **Identifying scans**: For each patient, finds baseline scans (`week-000-*`) and follow-up scans
2. **Loading segmentations**: Uses HD-GLIO-AUTO or DeepBraTumIA automatic segmentations
3. **Computing volumes**: Calculates whole tumor volume in milliliters
4. **Calculating delta**: Computes relative volume change: `(FollowUp - Baseline) / Baseline`
5. **Mapping RANO labels**: Simplifies expert ratings to three classes:
   - **Response**: Complete Response (CR) or Partial Response (PR)
   - **Stable**: Stable Disease (SD)
   - **Progression**: Progressive Disease (PD)

### Usage

```bash
# First, extract the imaging data (one-time, ~31 GB)
python prepare_lumiere_dataset.py --extract

# Or, if data is already extracted elsewhere
python prepare_lumiere_dataset.py --data_dir /path/to/Imaging

# Generate the output CSV
python prepare_lumiere_dataset.py
```

### Output

The script generates `lumiere_phase0.csv` with columns:

| Column | Description |
|--------|-------------|
| `Patient_ID` | Patient identifier (e.g., Patient-001) |
| `Baseline_Week` | Baseline scan week folder (e.g., week-000-1) |
| `Followup_Week` | Follow-up scan week folder (e.g., week-044) |
| `Baseline_Volume_ml` | Baseline tumor volume in milliliters |
| `Followup_Volume_ml` | Follow-up tumor volume in milliliters |
| `Delta_Volume` | Relative volume change: (followup - baseline) / baseline |
| `Response_Label` | Simplified RANO label: Response, Stable, or Progression |

### Label Distribution (Expected)

Based on the RANO ratings in the dataset:
- **Progression (PD)**: ~253 timepoints
- **Stable (SD)**: ~97 timepoints
- **Response (CR + PR)**: ~47 timepoints

Note: Pre-Op and Post-Op ratings are excluded as they don't represent treatment response.

## Requirements

- Python 3.8+
- numpy
- pandas
- nibabel

Install with:
```bash
pip install numpy pandas nibabel
```

## Next Steps (Future Phases)

After Phase 0, you can train an XGBoost model:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load the prepared data
df = pd.read_csv('lumiere_phase0.csv')

# Features and labels
X = df[['Baseline_Volume_ml', 'Followup_Volume_ml', 'Delta_Volume']]
y = df['Response_Label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
model = XGBClassifier()
model.fit(X_train, y_train)
```

## References

- [LUMIERE Dataset](https://www.synapse.org/#!Synapse:syn51156910/wiki/)
- [MICCAI BraTS Challenge](https://www.synapse.org/#!Synapse:syn51156910)
- [RANO Criteria](https://doi.org/10.1200/JCO.2009.26.3988)
