# AI-Powered Brain MRI Assistant

An automated brain tumor segmentation and analysis system that helps doctors analyze MRI scans faster and more accurately. Includes AI-powered tumor segmentation, comprehensive feature extraction, and a **fully automated pipeline** for end-to-end analysis.

## What It Does

### 1. Tumor Segmentation
Takes 4 different types of brain MRI scans (T1, T1ce, T2, FLAIR) and automatically identifies three tumor regions:
- **Enhancing Tumor (ET)**: The most active part of the tumor
- **Edema (ED)**: Swelling around the tumor  
- **Necrotic Core (NCR)**: Dead tissue in the tumor center

### 2. Feature Extraction Pipeline
Extracts 50+ radiology-relevant features from the segmentation, including:
- **Signal characteristics** on each MRI sequence
- **Tumor volumes** and measurements
- **Anatomical location** (hemisphere, lobe, depth)
- **Mass effect** (midline shift, ventricular compression)
- **Morphology** (shape, margins, necrosis pattern)
- **Quality metrics** and confidence scores

Outputs are optimized for LLM-based radiology report generation with safeguards against hallucination.

### 3. Fully Automated Pipeline (NEW)
One-command automation that runs the complete workflow:
1. **File Renaming**: Converts BraTS 2025 naming to BraTS 2021 format
2. **Segmentation**: Runs dual-model ensemble inference
3. **Label Conversion**: Converts model output to standard BraTS labels
4. **Evaluation**: Computes Dice scores against ground truth
5. **Feature Extraction**: Generates comprehensive analysis and LLM-ready JSON

## How It Works

1. **Segmentation Model**: BraTS 2021 Challenge winning model (ranked #1 globally)
2. **Technology**: nnU-Net deep learning framework
3. **Feature Extraction**: 6-step pipeline analyzing all clinically relevant tumor characteristics
4. **Output**: JSON summaries ready for LLM report generation

## Performance

Tested on multiple BraTS 2025 cases:

| Case | Mean Dice | WT | TC | ET |
|------|-----------|----|----|----|
| BraTS-GLI-00003-000 | **97.41%** | 98.43% | 98.03% | 95.76% |
| BraTS-GLI-00005-000 | **96.17%** | 96.79% | 97.86% | 93.87% |
| BraTS-GLI-00009-000 | **92.60%** | - | - | - |

**Average Performance**: 95%+ Mean Dice Score ⭐ Excellent

## How to Use

### Option 1: Fully Automated Pipeline (Recommended)

Run the complete analysis with a single command:

```bash
python run_full_pipeline.py <input_folder>
```

**Example**:
```bash
python run_full_pipeline.py BraTS-GLI-00009-000
```

This automatically:
1. Renames BraTS 2025 files to BraTS 2021 format
2. Runs segmentation (~5-6 minutes on CPU)
3. Converts labels to match ground truth
4. Evaluates against ground truth (if available)
5. Runs 6-step feature extraction pipeline
6. Generates LLM-ready JSON and radiology report

### Option 2: Manual Step-by-Step

#### Step 1: Set Up Environment
```powershell
$env:nnUNet_raw_data_base="C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant\nnUNet_raw"
$env:nnUNet_preprocessed="C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant\nnUNet_preprocessed"
$env:RESULTS_FOLDER="C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant\nnUNet_results"
```

#### Step 2: Convert BraTS 2025 Naming (if needed)
```bash
python convert_brats2025_naming.py <input_folder>
```

#### Step 3: Run Segmentation
```bash
python run_brats2021_inference_singlethread.py --input <input_folder> --output <output_folder>
```

#### Step 4: Convert Labels
```bash
python convert_labels_to_brats.py <segmentation.nii.gz> <output_brats.nii.gz>
```

#### Step 5: Run Feature Extraction
```bash
python feature_extraction/run_all.py --input <mri_folder> --segmentation <seg.nii.gz> --output <output_folder>
```

## What You Get

- **3D Segmentation File**: Color-coded tumor regions in NIfTI format
- **Pipeline Summary**: `pipeline_summary.json` with timing and metrics
- **Comprehensive Analysis**:
  - `llm_ready_summary.json` - Structured data for LLM report generation
  - `radiology_report.txt` - Human-readable analysis report
  - `comprehensive_analysis.json` - Complete technical data
  - `step1-6 JSON files` - Detailed per-step analysis
- **Volume Measurements**: Automatic calculation of tumor sizes
- **Quality Metrics**: Confidence scores and reliability warnings
- **Evaluation Metrics**: Dice scores compared to ground truth (when available)

## Key Features for LLM Integration

The feature extraction pipeline includes safeguards for accurate report generation:
- **Patient info placeholders**: Prevents LLM from fabricating demographics
- **Technique documentation**: Lists exactly which sequences were available
- **Guarded language**: Uses non-diagnostic phrasing (e.g., "can be seen with" not "suggests")
- **Reliability warnings**: Flags measurements affected by poor image quality
- **Laterality validation**: Cross-checks hemisphere determination

## Important Notes

- **Full pipeline takes ~5-6 minutes** on CPU (optimized from 30 minutes)
- Supports both **BraTS 2025** and **BraTS 2021** file naming formats
- Results are saved in standard medical format (NIfTI + JSON)

## Supported Data Formats

### BraTS 2025 (Auto-converted)
```
CaseID-t1n.nii.gz, CaseID-t1c.nii.gz, CaseID-t2w.nii.gz, CaseID-t2f.nii.gz, CaseID-seg.nii.gz
```

### BraTS 2021 (Native)
```
CaseID_t1.nii.gz, CaseID_t1ce.nii.gz, CaseID_t2.nii.gz, CaseID_flair.nii.gz, CaseID_seg.nii.gz
```

## Technical Details

See `PROJECT_DOCUMENTATION.md` for:
- Complete model architecture details
- Feature extraction pipeline documentation
- All bug fixes and solutions
- Performance benchmarks
- Troubleshooting guide




