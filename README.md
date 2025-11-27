# AI-Powered Brain MRI Assistant

An automated brain tumor segmentation and analysis system that helps doctors analyze MRI scans faster and more accurately. Includes AI-powered tumor segmentation and comprehensive feature extraction for radiology report generation.

## What It Does

### 1. Tumor Segmentation
Takes 4 different types of brain MRI scans (T1, T1ce, T2, FLAIR) and automatically identifies three tumor regions:
- **Enhancing Tumor (ET)**: The most active part of the tumor
- **Edema (ED)**: Swelling around the tumor  
- **Necrotic Core (NCR)**: Dead tissue in the tumor center

### 2. Feature Extraction Pipeline (NEW)
Extracts 50+ radiology-relevant features from the segmentation, including:
- **Signal characteristics** on each MRI sequence
- **Tumor volumes** and measurements
- **Anatomical location** (hemisphere, lobe, depth)
- **Mass effect** (midline shift, ventricular compression)
- **Morphology** (shape, margins, necrosis pattern)
- **Quality metrics** and confidence scores

Outputs are optimized for LLM-based radiology report generation with safeguards against hallucination.

## How It Works

1. **Segmentation Model**: BraTS 2021 Challenge winning model (ranked #1 globally)
2. **Technology**: nnU-Net deep learning framework
3. **Feature Extraction**: 6-step pipeline analyzing all clinically relevant tumor characteristics
4. **Output**: JSON summaries ready for LLM report generation

## Performance

Tested on real brain tumor cases:
- **92.95% accuracy** on enhancing tumor detection ⭐
- **83.24% accuracy** on edema detection
- **67.34% accuracy** on necrotic core detection

## How to Use

### Step 1: Set Up Environment
```powershell
$env:nnUNet_raw_data_base="C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant\nnUNet_raw"
$env:nnUNet_preprocessed="C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant\nnUNet_preprocessed"
$env:RESULTS_FOLDER="C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant\nnUNet_results"
```

### Step 2: Run Segmentation
```bash
python run_brats2021_inference_singlethread.py --input <input_folder> --output <output_folder>
```

### Step 3: Run Feature Extraction (NEW)
```bash
cd feature_extraction
python run_all.py --input <mri_folder> --segmentation <segmentation.nii.gz> --output <output_folder>
```

## What You Get

- **3D Segmentation File**: Color-coded tumor regions in NIfTI format
- **Comprehensive Analysis** (NEW):
  - `llm_ready_summary.json` - Structured data for LLM report generation
  - `radiology_report.txt` - Human-readable analysis report
  - `comprehensive_analysis.json` - Complete technical data
- **Volume Measurements**: Automatic calculation of tumor sizes
- **Quality Metrics**: Confidence scores and reliability warnings

## Key Features for LLM Integration

The feature extraction pipeline includes safeguards for accurate report generation:
- **Patient info placeholders**: Prevents LLM from fabricating demographics
- **Technique documentation**: Lists exactly which sequences were available
- **Guarded language**: Uses non-diagnostic phrasing (e.g., "can be seen with" not "suggests")
- **Reliability warnings**: Flags measurements affected by poor image quality
- **Laterality validation**: Cross-checks hemisphere determination

## Important Notes

- Close other applications while running for best performance
- Don't interrupt the process once it starts
- Segmentation takes ~30 minutes; feature extraction takes ~2 minutes
- Results are saved in standard medical format (NIfTI + JSON)

## Technical Details

See `PROJECT_DOCUMENTATION.md` for:
- Complete model architecture details
- Feature extraction pipeline documentation
- All bug fixes and solutions
- Performance benchmarks
- Troubleshooting guide

## Requirements

- Windows 10/11 (tested)
- Python 3.10
- 16GB RAM minimum (32GB recommended)
- 10GB free disk space



