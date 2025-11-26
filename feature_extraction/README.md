# Feature Extraction Module

This module provides tools to extract clinically relevant features from brain MRI scans and tumor segmentation masks for automated radiology report generation.

## Structure

```
feature_extraction/
├── __init__.py                   # Module initialization
├── utils.py                      # Shared utility functions
├── step1_sequence_findings.py    # T1/T2/FLAIR/contrast analysis
├── step2_mass_effect.py          # Midline shift, ventricular compression
├── step3_anatomical_context.py   # Brain atlas mapping, lobe identification
├── step4_multiplicity.py         # Single vs multiple lesions detection
├── step5_morphology.py           # Shape, margins, necrosis analysis
├── step6_quality_control.py      # Confidence scores, artifact detection
├── step7_normal_structures.py    # Ventricles, vessels assessment
├── run_all_features.py           # Master script to run all steps
└── README.md                     # This file
```

## Usage

### Run Individual Steps

```bash
# Step 1: Sequence-specific findings
python step1_sequence_findings.py --input <mri_folder> --segmentation <seg.nii.gz> --output <output.json>

# Step 2: Mass effect metrics
python step2_mass_effect.py --input <mri_folder> --segmentation <seg.nii.gz> --output <output.json>

# ... additional steps
```

### Run All Steps

```bash
python run_all_features.py --input <mri_folder> --segmentation <seg.nii.gz> --output-dir <output_folder>
```

## Output

Each step produces a JSON file with:
- Structured data for programmatic use
- `text_summary` field with human-readable summary for report generation

## Features Extracted

### Step 1: Sequence-specific Findings
- T1, T2, FLAIR intensity analysis
- Contrast enhancement ratios
- Signal characteristics relative to normal brain

### Step 2: Mass Effect Metrics
- Midline shift (mm)
- Ventricular compression/asymmetry
- Herniation risk assessment
- Sulcal effacement

### Step 3: Anatomical Context
- Brain lobe identification
- Hemisphere dominance
- Deep vs superficial location
- Proximity to eloquent areas

### Step 4: Multiplicity
- Single vs multiple lesions
- Connected component analysis
- Satellite lesion detection

### Step 5: Morphology
- Shape descriptors (sphericity, elongation)
- Margin characteristics
- Necrosis percentage
- Cystic vs solid components

### Step 6: Quality Control
- Segmentation confidence metrics
- Missing sequence detection
- Artifact flags

### Step 7: Normal Structures
- Ventricular system assessment
- Major vessel proximity
- Parenchyma outside tumor

## Requirements

- Python 3.10+
- numpy
- nibabel
- scipy
- (optional) scikit-image for advanced morphology

## Author

AI-Powered Brain MRI Assistant
November 2025
