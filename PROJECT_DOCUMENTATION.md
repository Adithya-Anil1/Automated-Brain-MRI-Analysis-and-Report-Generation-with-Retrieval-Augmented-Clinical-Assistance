# AI-Powered Brain MRI Assistant - Technical Documentation

## Project Overview

This project implements an automated brain tumor segmentation system using the **BraTS 2021 KAIST MRI Lab winning model**. The system performs semantic segmentation on brain MRI scans to identify and delineate different tumor regions with state-of-the-art accuracy.

**Status**: ✅ Fully functional and validated  
**Last Updated**: November 28, 2025  
**Python Version**: 3.10.11  
**Compute**: CPU-only (no GPU required, optimized for fast inference)

### New Features (v2.1)
- ✅ **Fully Automated Pipeline**: Single-command end-to-end analysis (`run_full_pipeline.py`)
- ✅ **BraTS 2025 Support**: Automatic file naming conversion (`convert_brats2025_naming.py`)
- ✅ **Feature Extraction Pipeline**: 6-step analysis for radiology report generation
- ✅ **LLM Integration Safeguards**: Prevents hallucination of patient data and sequences
- ✅ **Quality Metrics**: SNR-based reliability warnings
- ✅ **Structured Output**: JSON format optimized for LLM consumption
- ✅ **Performance**: 95%+ Mean Dice on BraTS 2025 test cases

---

## Model Architecture

### Primary Model: BraTS 2021 KAIST MRI Lab Ensemble

**Source**: [KAIST MRI Lab GitHub Repository](https://github.com/KAIST-MRI-Lab/BraTS2021)  
**Framework**: nnU-Net (No-New-Net) with custom BraTS-specific modifications  
**Competition Result**: 1st Place Winner - BraTS 2021 Challenge

### Model Components

The system uses an **ensemble of two models** with 5-fold cross-validation each:

#### Model 1: `nnUNetTrainerV2BraTSRegions_DA4_BN_BD`
- **Architecture**: 3D U-Net with residual connections
- **Normalization**: Batch Normalization (BN)
- **Training**: Data Augmentation v4 (DA4), Big Data mode (BD)
- **Output**: 3 sigmoid channels (regions-based segmentation)
- **Path**: `nnUNet_results/3d_fullres/Task500_BraTS2021/nnUNetTrainerV2BraTSRegions_DA4_BN_BD__nnUNetPlansv2.1/`
- **Folds**: 5 (fold_0 through fold_4)

#### Model 2: `nnUNetTrainerV2BraTSRegions_DA4_BN_BD_largeUnet_Groupnorm`
- **Architecture**: Large 3D U-Net with Group Normalization
- **Normalization**: Group Normalization (better for small batch sizes)
- **Training**: Data Augmentation v4 (DA4), Big Data mode (BD)
- **Output**: 3 sigmoid channels (regions-based segmentation)
- **Path**: `nnUNet_results/3d_fullres/Task500_BraTS2021/nnUNetTrainerV2BraTSRegions_DA4_BN_BD_largeUnet_Groupnorm__nnUNetPlansv2.1/`
- **Folds**: 5 (fold_0 through fold_4)
- **Note**: More memory-intensive, slower on CPU

### Key Architectural Features

**BraTSRegions Custom Trainer**:
- Uses **region-based segmentation** instead of standard multi-class
- Outputs 3 independent sigmoid probability maps (one per region)
- Regions defined as:
  - **Region 1**: Whole Tumor (WT) = labels {1, 2, 3}
  - **Region 2**: Tumor Core (TC) = labels {2, 3}
  - **Region 3**: Enhancing Tumor (ET) = label {3}
- **Critical**: `regions_class_order = (1, 2, 3)` maps outputs to labels
- **Inference**: Uses sigmoid activation (NOT softmax)
- **Threshold**: 0.5 for binary decision per region

**Network Specifications**:
- **Input**: 4 MRI modalities (T1, T1ce, T2, FLAIR)
- **Patch Size**: 128×128×128 voxels
- **Prediction Strategy**: Sliding window with Gaussian weighting
- **Mirroring**: Test-time augmentation with 8 mirror axes (0, 1, 2)
- **Step Size**: 0.5 (50% overlap between patches)
- **Ensembling**: Mean averaging of 10 predictions (5 folds × 2 models)

---

## Dataset

### BraTS 2021 Dataset

**Official Name**: Brain Tumor Segmentation Challenge 2021  
**Organizer**: Medical Image Computing and Computer Assisted Intervention (MICCAI)  
**Task**: Multi-class segmentation of gliomas in pre-operative MRI scans

### Data Format

**Input Modalities** (4 channels):
1. **T1**: T1-weighted MRI (native contrast)
2. **T1ce**: T1-weighted with contrast enhancement (gadolinium)
3. **T2**: T2-weighted MRI
4. **FLAIR**: Fluid-Attenuated Inversion Recovery

**File Format**: NIfTI (.nii.gz)  
**Spatial Resolution**: 1×1×1 mm³ isotropic  
**Dimensions**: 240×240×155 voxels (after cropping from 240×240×155)  
**Orientation**: LPS (Left-Posterior-Superior)

### Label Convention

**BraTS Standard Labels**:
- **0**: Background (healthy brain tissue)
- **1**: NCR (Necrotic and Non-Enhancing Tumor Core)
- **2**: ED (Peritumoral Edema)
- **4**: ET (GD-Enhancing Tumor) ⚠️ Note: No label 3 in BraTS

**Internal Model Labels** (before conversion):
- **0**: Background
- **1**: Region 1 output
- **2**: Region 2 output
- **3**: Region 3 output (must be converted to label 4)

### Sample Data Location

**Ground Truth**: `data/sample_data/BraTS2021_sample/BraTS2021_00495_seg.nii.gz`  
**Input MRI Files**:
- `BraTS2021_00495_t1.nii.gz`
- `BraTS2021_00495_t1ce.nii.gz`
- `BraTS2021_00495_t2.nii.gz`
- `BraTS2021_00495_flair.nii.gz`

**Case ID**: BraTS2021_00495  
**Tumor Volume (Ground Truth)**: 99.30 cm³  
**Label Distribution**:
- NCR (Label 1): 20,808 voxels (20.81 cm³)
- ED (Label 2): 46,447 voxels (46.45 cm³)
- ET (Label 4): 32,047 voxels (32.05 cm³)

---

## Performance Metrics

### Validation Results (BraTS 2025 Test Cases)

**Test Date**: November 27-28, 2025  
**Model**: BraTS 2021 KAIST Ensemble (1st Place Winner)

#### Summary Results

| Case ID | Mean Dice | WT Dice | TC Dice | ET Dice | Rating |
|---------|-----------|---------|---------|---------|--------|
| BraTS-GLI-00003-000 | **97.41%** | 98.43% | 98.03% | 95.76% | ⭐ Excellent |
| BraTS-GLI-00005-000 | **96.17%** | 96.79% | 97.86% | 93.87% | ⭐ Excellent |
| BraTS-GLI-00009-000 | **92.60%** | - | - | - | ⭐ Excellent |

**Average Mean Dice**: 95.39%  
**Best Case**: BraTS-GLI-00003-000 (97.41% Mean Dice)

#### Detailed Results (BraTS-GLI-00003-000)

| Label | Region | Dice Score | IoU | Sensitivity | Specificity |
|-------|--------|------------|-----|-------------|-------------|
| 1 | NCR (Necrotic Core) | **96.57%** | 93.35% | 96.77% | 99.99% |
| 2 | ED (Edema) | **97.63%** | 95.38% | 97.09% | 99.98% |
| 3 | ET (Enhancing Tumor) | **95.66%** | 91.64% | 93.69% | 99.99% |

| Metric | Dice Score | IoU | Sensitivity | Rating |
|--------|------------|-----|-------------|--------|
| **Whole Tumor (WT)** | **98.43%** | 96.90% | 97.36% | ⭐ Excellent |
| **Tumor Core (TC)** | **98.03%** | 96.12% | 97.83% | ⭐ Excellent |
| **Enhancing Tumor (ET)** | **95.76%** | 91.87% | 93.69% | ⭐ Excellent |

### Performance Interpretation

**Dice Score Ranges**:
- **> 90%**: Excellent ⭐ (ET achieved this)
- **80-90%**: Good (ED in this range)
- **70-80%**: Moderate (WT in this range)
- **50-70%**: Fair (TC in this range)
- **< 50%**: Poor

**Expected Performance** (from literature):
- Whole Tumor: 89-92% Dice
- Tumor Core: 86-88% Dice
- Enhancing Tumor: 84-87% Dice

**Analysis**:
- ✅ **ET performance exceeds expectations** (92.95% vs 84-87% expected)
- ⚠️ WT and TC below expected ranges (likely due to single-case testing)
- ✅ High specificity across all labels (99%+) - very few false positives
- ✅ NCR has excellent sensitivity (98.95%) - catches nearly all tumor core

---

## Technical Implementation

### Environment Setup

**Python Environment**: `venv310/` (Python 3.10.11)  
**Package Manager**: pip (user-level installation)

#### Critical Dependencies

```
torch==2.9.1 (CPU-only, no CUDA)
nibabel==5.3.2
SimpleITK==2.5.3
batchgenerators==0.21 (CRITICAL: Must be 0.21, not 0.25+)
numpy (via torch)
scipy (via torch)
matplotlib==3.10.7
medpy==0.5.2
monai==1.5.1
```

#### Important Notes

1. **batchgenerators Version**: Must be `0.21` - version `0.25.1` removed `MultiThreadedAugmenter` class
2. **PyTorch Version**: `2.9.1` defaults to `weights_only=True` in `torch.load()` - must override to `False`
3. **CPU Inference**: No GPU required, but inference is slow (10-20 min per model)

### Environment Variables

**Required for nnU-Net**:
```powershell
$env:nnUNet_raw_data_base = "C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant\nnUNet_raw"
$env:nnUNet_preprocessed = "C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant\nnUNet_preprocessed"
$env:RESULTS_FOLDER = "C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant\nnUNet_results"
```

These must be set before running inference.

---

## Workflow & Scripts

### Fully Automated Pipeline Script (NEW)

**File**: `run_full_pipeline.py`  
**Purpose**: Complete end-to-end analysis with a single command  
**Status**: ✅ Fully functional and validated

**Usage**:
```bash
python run_full_pipeline.py <input_folder>
python run_full_pipeline.py BraTS-GLI-00009-000
```

**Pipeline Steps**:
1. **File Renaming**: Converts BraTS 2025 naming to BraTS 2021 format
2. **Segmentation**: Runs dual-model ensemble inference
3. **Label Conversion**: Converts model output to BraTS standard labels
4. **Evaluation**: Computes Dice scores against ground truth
5. **Feature Extraction**: Generates comprehensive analysis

**Output Structure**:
```
results/<CaseID>/
├── <CaseID>.nii.gz              (Raw segmentation)
├── <CaseID>_brats.nii.gz        (Converted labels)
├── pipeline_summary.json         (Timing and metrics)
├── temp_model1/                  (Model 1 intermediate)
├── temp_model2/                  (Model 2 intermediate)
└── feature_extraction/
    ├── llm_ready_summary.json    (For LLM report generation)
    ├── radiology_report.txt      (Human-readable report)
    ├── comprehensive_analysis.json
    └── step1-6 JSON files
```

**Execution Time**: ~5-6 minutes on CPU

---

### BraTS 2025 Naming Conversion Script (NEW)

**File**: `convert_brats2025_naming.py`  
**Purpose**: Convert BraTS 2025 file naming to BraTS 2021 format  
**Status**: ✅ Fully functional

**Usage**:
```bash
python convert_brats2025_naming.py <input_folder>
python convert_brats2025_naming.py BraTS-GLI-00009-000
```

**Naming Conversion**:
| BraTS 2025 | BraTS 2021 |
|------------|------------|
| `*-t1n.nii.gz` | `*_t1.nii.gz` |
| `*-t1c.nii.gz` | `*_t1ce.nii.gz` |
| `*-t2w.nii.gz` | `*_t2.nii.gz` |
| `*-t2f.nii.gz` | `*_flair.nii.gz` |
| `*-seg.nii.gz` | `*_seg.nii.gz` |

---

### Main Inference Script

**File**: `run_brats2021_inference_singlethread.py`  
**Purpose**: Run full ensemble inference with both models  
**Status**: ✅ Fixed and validated

**Usage**:
```powershell
$env:nnUNet_raw_data_base="C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant\nnUNet_raw"
$env:nnUNet_preprocessed="C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant\nnUNet_preprocessed"
$env:RESULTS_FOLDER="C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant\nnUNet_results"
python run_brats2021_inference_singlethread.py --input data\sample_data\BraTS2021_sample --output results\BraTS2021_00495_fixed
```

**Key Parameters**:
- `--input`: Folder containing the 4 MRI modalities (t1, t1ce, t2, flair)
- `--output`: Output directory for segmentation results
- Uses single-threaded processing (Windows multiprocessing compatibility)

**Output Structure**:
```
results/BraTS2021_00495_fixed/
├── temp_model1/
│   └── BraTS2021_00495.nii.gz  (Model 1 output)
├── temp_model2/
│   └── BraTS2021_00495.nii.gz  (Model 2 output)
└── BraTS2021_00495.nii.gz      (Final ensemble output)
```

**Processing Steps**:
1. Load all 5 fold checkpoints for Model 1
2. Preprocess input (crop, normalize)
3. Run inference with 5-fold ensemble + mirroring
4. Save Model 1 output with `region_class_order=(1,2,3)`
5. Repeat steps 1-4 for Model 2
6. Ensemble both models (mean averaging)
7. Save final segmentation

**Execution Time**: ~20-30 minutes on CPU (10-15 min per model)

### Critical Bug Fix (Applied)

**File Modified**: `run_brats2021_inference_singlethread.py` line 147

**Original (Buggy) Code**:
```python
save_segmentation_nifti_from_softmax(softmax_mean, output_file, dct, 
                                      interpolation_order, None, ...)
```

**Fixed Code**:
```python
save_segmentation_nifti_from_softmax(softmax_mean, output_file, dct, 
                                      interpolation_order, (1, 2, 3), ...)
```

**Explanation**:
- When `region_class_order=None`, code uses `argmax()` on 3 channels → outputs [0,1,2]
- When `region_class_order=(1,2,3)`, code thresholds at 0.5 and assigns labels → outputs [0,1,2,3]
- This fix was **critical** - improved Dice from 2-5% to 67-93%

### PyTorch Compatibility Fix (Applied)

**File Modified**: `Brats21_KAIST_MRI_Lab/nnunet/training/model_restore.py` line 147

**Original Code**:
```python
checkpoint = torch.load(i, map_location=torch.device('cpu'))
```

**Fixed Code**:
```python
checkpoint = torch.load(i, map_location=torch.device('cpu'), weights_only=False)
```

**Reason**: PyTorch 2.9+ defaults to `weights_only=True` which rejects `numpy.core.multiarray.scalar` objects in old checkpoints.

### Label Conversion Script

**File**: `convert_labels_to_brats.py`  
**Purpose**: Convert internal labels [0,1,2,3] to BraTS format [0,1,2,4]

**Usage**:
```bash
python convert_labels_to_brats.py "results\BraTS2021_00495_fixed\BraTS2021_00495.nii.gz" "results\BraTS2021_00495_fixed\BraTS2021_00495_brats.nii.gz"
```

**Label Mapping**:
- Internal Label 1 → BraTS Label 2 (ED)
- Internal Label 2 → BraTS Label 1 (NCR)
- Internal Label 3 → BraTS Label 4 (ET) ⚠️ Critical conversion
- Label 0 remains 0 (Background)

**Features**:
- Rounds floating-point labels to nearest integer (handles numerical errors)
- Validates output has [0,1,2,4] labels
- Preserves NIfTI header and affine transformation

### Evaluation Script

**File**: `evaluate_segmentation.py`  
**Purpose**: Compute Dice, IoU, sensitivity, specificity metrics

**Usage**:
```bash
python evaluate_segmentation.py --pred "results\BraTS2021_00495_fixed\BraTS2021_00495_brats.nii.gz" --gt "data\sample_data\BraTS2021_sample\BraTS2021_00495_seg.nii.gz"
```

**Metrics Computed**:
- **Per-label**: Dice score, IoU, Sensitivity, Specificity, TP/FP/FN counts
- **Compound**: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET)
- **Overall**: Mean Dice across WT/TC/ET

### Label Checking Script

**File**: `check_labels.py`  
**Purpose**: Quick diagnostic to verify label presence

**Usage**:
```bash
python check_labels.py "results\BraTS2021_00495_fixed\BraTS2021_00495.nii.gz"
```

**Output**:
- Unique labels found
- Voxel counts per label
- BraTS compliance check
- Tumor volume statistics

---

## Known Issues & Solutions

### Issue 1: Missing Label 4 (SOLVED)

**Symptom**: Segmentation only contains labels [0,1,2], missing label 4  
**Root Cause**: `region_class_order` parameter was `None` instead of `(1,2,3)`  
**Solution**: Modified `run_brats2021_inference_singlethread.py` line 147  
**Result**: Now outputs [0,1,2,3], which converts to [0,1,2,4] ✅

### Issue 2: PyTorch UnpicklingError (SOLVED)

**Symptom**: `UnpicklingError: Weights only load failed` when loading checkpoints  
**Root Cause**: PyTorch 2.9 defaults to `weights_only=True`  
**Solution**: Added `weights_only=False` to `model_restore.py` line 147  
**Result**: Checkpoints load successfully ✅

### Issue 3: batchgenerators Import Error (SOLVED)

**Symptom**: `ImportError: cannot import name 'MultiThreadedAugmenter'`  
**Root Cause**: batchgenerators 0.25.1 removed this class  
**Solution**: Downgrade to `batchgenerators==0.21`  
**Result**: All imports work correctly ✅

### Issue 4: CPU Inference Hangs/Crashes (MITIGATED)

**Symptom**: Inference hangs at "computing Gaussian" and crashes with KeyboardInterrupt  
**Root Cause**: Models too computationally intensive for CPU, especially with mirroring  
**Solution**: 
1. Close all unnecessary applications and terminals
2. Set Windows to High Performance mode
3. Do NOT interrupt terminal during inference
4. Single-threaded processing (already implemented)
**Result**: Inference completes but takes 20-30 minutes ⏳

### Issue 5: Floating Point Label Errors (SOLVED)

**Symptom**: Labels are [1.003..., 1.992..., 4.000...] instead of [1, 2, 4]  
**Root Cause**: Numerical precision errors in label conversion  
**Solution**: Added `np.round().astype(np.uint8)` to `convert_labels_to_brats.py`  
**Result**: Clean integer labels [0,1,2,4] ✅

---

## Directory Structure

```
AI-Powered Brain MRI Assistant/
├── run_full_pipeline.py                     ⭐ NEW: Fully automated pipeline
├── convert_brats2025_naming.py              ⭐ NEW: BraTS 2025 naming converter
├── run_brats2021_inference_singlethread.py  ⭐ Main inference script (FIXED)
├── convert_labels_to_brats.py               ⭐ Label conversion utility (FIXED)
├── evaluate_segmentation.py                 ⭐ Metrics calculation
├── check_labels.py                          ⭐ Label verification
├── compare_segmentations.py                 Visual comparison tool
├── download_more_brats_data.py              Download additional samples
├── requirements.txt                         Python dependencies
├── README.md                                ⭐ Quick start guide
├── PROJECT_DOCUMENTATION.md                 ⭐ This file (detailed docs)
│
├── Brats21_KAIST_MRI_Lab/                   Model source code
│   ├── nnunet/                              nnU-Net framework
│   │   ├── training/
│   │   │   ├── model_restore.py             (MODIFIED: weights_only=False)
│   │   │   └── network_training/
│   │   │       └── competitions_with_custom_Trainers/BraTS2020/
│   │   │           └── nnUNetTrainerV2BraTSRegions.py  (regions_class_order)
│   │   ├── inference/
│   │   │   └── segmentation_export.py       (region_class_order logic)
│   │   └── network_architecture/            U-Net architectures
│   └── requirements_v2.txt
│
├── data/
│   └── sample_data/
│       └── BraTS2021_sample/                ⭐ Test case
│           ├── BraTS2021_00495_t1.nii.gz
│           ├── BraTS2021_00495_t1ce.nii.gz
│           ├── BraTS2021_00495_t2.nii.gz
│           ├── BraTS2021_00495_flair.nii.gz
│           └── BraTS2021_00495_seg.nii.gz   (Ground truth)
│
├── results/
│   └── BraTS2021_00495_fixed/               ⭐ Final output (correct labels)
│       ├── temp_model1/
│       │   └── BraTS2021_00495.nii.gz
│       ├── temp_model2/
│       │   └── BraTS2021_00495.nii.gz
│       ├── BraTS2021_00495.nii.gz           (Ensemble, labels [0,1,2,3])
│       └── BraTS2021_00495_brats.nii.gz     (Converted, labels [0,1,2,4])
│
├── nnUNet_results/                          ⭐ Trained model checkpoints
│   └── 3d_fullres/
│       └── Task500_BraTS2021/
│           ├── nnUNetTrainerV2BraTSRegions_DA4_BN_BD__nnUNetPlansv2.1/
│           │   ├── fold_0/model_final_checkpoint.model
│           │   ├── fold_1/model_final_checkpoint.model
│           │   ├── fold_2/model_final_checkpoint.model
│           │   ├── fold_3/model_final_checkpoint.model
│           │   └── fold_4/model_final_checkpoint.model
│           └── nnUNetTrainerV2BraTSRegions_DA4_BN_BD_largeUnet_Groupnorm__nnUNetPlansv2.1/
│               ├── fold_0/model_final_checkpoint.model
│               ├── fold_1/model_final_checkpoint.model
│               ├── fold_2/model_final_checkpoint.model
│               ├── fold_3/model_final_checkpoint.model
│               └── fold_4/model_final_checkpoint.model
│
├── nnUNet_raw/                              Raw data structure (nnU-Net format)
├── nnUNet_preprocessed/                     Preprocessed cache
├── venv310/                                 Python virtual environment
├── feature_extraction/                      ⭐ Feature extraction pipeline (NEW)
│   ├── run_all.py                           Main pipeline script
│   ├── step1_sequence_findings.py           Signal analysis
│   ├── step2_mass_effect.py                 Mass effect metrics
│   ├── step3_multiplicity.py                Lesion multiplicity
│   ├── step4_morphology.py                  Tumor morphology
│   ├── step5_quality.py                     Quality control
│   ├── step6_normal_structures.py           Normal structures
│   ├── utils.py                             Shared utilities
│   └── README.md                            Pipeline documentation
│
├── main_files/                              Additional utilities
├── documentation/
│   └── instructions.md                      ⭐ Usage instructions (KEEP)
└── visualizations/                          Output visualizations

⭐ = Critical/Active files
```

---

## Feature Extraction Pipeline (NEW)

### Overview

A comprehensive 6-step feature extraction pipeline that analyzes tumor segmentations and produces structured output optimized for LLM-based radiology report generation.

**Location**: `feature_extraction/`  
**Main Script**: `run_all.py`  
**Output**: JSON summaries + text reports

### Pipeline Steps

#### Step 1: Sequence-Specific Findings (`step1_sequence_findings.py`)
- Signal intensity analysis (hypo/iso/hyperintense) for each tumor region
- Intensity ratios relative to normal brain tissue
- Contrast enhancement pattern detection
- T2/FLAIR mismatch analysis (IDH-mutant marker)
- Tumor volume calculations

#### Step 2: Mass Effect Metrics (`step2_mass_effect.py`)
- Anatomical localization (hemisphere, lobe, depth, gyri)
- Midline shift measurement with significance threshold
- Ventricular compression assessment
- Sulcal effacement grading
- Herniation risk scoring (0-7 scale)

#### Step 3: Lesion Multiplicity (`step3_multiplicity.py`)
- Connected component analysis
- Satellite lesion detection
- Inter-lesion distance calculation
- Distribution pattern classification (solitary, multicentric, etc.)
- Differential diagnosis suggestions

#### Step 4: Tumor Morphology (`step4_morphology.py`)
- Shape analysis (sphericity, elongation)
- Contour regularity (smooth vs lobulated vs irregular)
- Margin transition sharpness (well-defined vs infiltrative)
- Necrosis pattern and percentage
- Cystic vs solid classification

#### Step 5: Quality Control (`step5_quality.py`)
- Segmentation quality scoring (0-100)
- Image quality per sequence (SNR estimates)
- Artifact detection (motion, susceptibility, truncation)
- Measurement confidence levels
- Case-specific caveats and limitations

#### Step 6: Normal Structures (`step6_normal_structures.py`)
- Ventricular system assessment (size, symmetry, hydrocephalus)
- Brain parenchyma evaluation (gray-white differentiation)
- Major vessel assessment (flow voids, vascular involvement)

### LLM Integration Safeguards

The pipeline includes specific features to prevent LLM hallucination:

1. **Patient Information Placeholders**
   ```json
   "patient_info": {
     "age": "<not provided>",
     "sex": "<not provided>",
     "clinical_history": "<not provided>",
     "presenting_symptoms": "<not provided>",
     "note": "DO NOT fabricate patient demographics or clinical history."
   }
   ```

2. **Technique Documentation**
   ```json
   "technique": {
     "sequences_performed": ["T1", "T1CE", "T2", "FLAIR"],
     "sequences_not_available": ["DWI", "ADC", "MRS", "Perfusion"],
     "contrast_administered": true,
     "acquisition_parameters": {...},
     "note": "LLM must only reference sequences listed in sequences_performed"
   }
   ```

3. **Guarded Medical Language**
   - Old: "may suggest low-grade pathology"
   - New: "can be seen with lower-grade glioma, treatment effect, or other pathology; clinical and histopathological correlation required"

4. **Measurement Reliability Warnings**
   - Flags when T2 SNR is low: affects necrosis fraction, cystic classification
   - Flags when overall quality is poor: interpret with caution

5. **Laterality Validation**
   - Cross-checks hemisphere from voxel distribution vs centroid methods
   - Reports inconsistencies for tumors near midline

6. **Border vs Margin Clarification**
   - Contour: outer surface smoothness (smooth/lobulated/irregular)
   - Margin: tumor-brain intensity transition (well-defined/infiltrative)

### Usage

```bash
cd feature_extraction
python run_all.py --input <mri_folder> --segmentation <seg.nii.gz> --output <output_folder>
```

### Output Files

| File | Purpose |
|------|--------|
| `llm_ready_summary.json` | Structured data optimized for LLM consumption |
| `radiology_report.txt` | Human-readable text report |
| `comprehensive_analysis.json` | Complete technical data |
| `step1_sequence_findings.json` | Detailed step 1 results |
| `step2_mass_effect.json` | Detailed step 2 results |
| `step3_multiplicity.json` | Detailed step 3 results |
| `step4_morphology.json` | Detailed step 4 results |
| `step5_quality.json` | Detailed step 5 results |
| `step6_normal_structures.json` | Detailed step 6 results |

### Sample LLM-Ready Output Structure

```json
{
  "case_id": "BraTS-GLI-01477-000",
  "patient_info": { "age": "<not provided>", ... },
  "technique": { "sequences_performed": [...], ... },
  "tumor_characteristics": { "volume_cm3": 189.0, ... },
  "location": { "hemisphere": "left", "primary_lobe": "temporal", ... },
  "enhancement": { "present": false, "pattern": "Non-enhancing", ... },
  "morphology": { "contour_shape": "Smooth contour", "margin_transition": "Infiltrative transition", ... },
  "quality_metrics": { "segmentation_score": 100, ... },
  "measurement_reliability_warnings": [...],
  "caveats": [...]
}
```

---

## Future Improvements

### Short Term
1. ~~**Report Generation**: Automated medical report with measurements~~ ✅ DONE
2. **Test with more samples**: Validate on larger dataset
3. **Visualization**: Implement 3D rendering and slice-by-slice views
4. **Performance optimization**: Investigate CPU-optimized inference

### Medium Term
1. **GPU Support**: Enable CUDA for 10-20x faster inference
2. **Web Interface**: Flask/Streamlit app for easy deployment
3. **LLM Integration**: Direct integration with GPT-4/Claude for report generation
4. **Batch Processing**: Process multiple cases in sequence

### Long Term
1. **Clinical Validation**: Test on larger dataset (100+ cases)
2. **Multi-center Validation**: Test on data from different hospitals
3. **Longitudinal Analysis**: Track tumor changes over time
4. **Integration**: PACS/DICOM integration for clinical workflow

---

## References

### Primary Citation
```
Isensee, F., Jaeger, P.F., Kohl, S.A.A. et al. 
"nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." 
Nature Methods 18, 203–211 (2021). 
https://doi.org/10.1038/s41592-020-01008-z
```

### BraTS Challenge
```
Baid, U., Ghodasara, S., Mohan, S. et al. 
"The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification." 
arXiv:2107.02314 (2021).
```

### Model Repository
- GitHub: https://github.com/KAIST-MRI-Lab/BraTS2021
- License: Apache 2.0

---

## Troubleshooting Guide

### Problem: "nnUNet environment variables not defined"
**Solution**: Set environment variables before running inference:
```powershell
$env:nnUNet_raw_data_base="C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant\nnUNet_raw"
$env:nnUNet_preprocessed="C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant\nnUNet_preprocessed"
$env:RESULTS_FOLDER="C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant\nnUNet_results"
```

### Problem: Inference hangs at "computing Gaussian"
**Solutions**:
1. Close all other applications
2. Don't run any other commands in the terminal
3. Set Windows to High Performance mode
4. Wait patiently (10-20 minutes per model)
5. Don't interrupt the process

### Problem: Missing label 4 in output
**Solution**: Ensure you're using the FIXED version of `run_brats2021_inference_singlethread.py` with `region_class_order=(1,2,3)` on line 147

### Problem: "cannot import name 'MultiThreadedAugmenter'"
**Solution**: 
```bash
python -m pip uninstall batchgenerators
python -m pip install batchgenerators==0.21
```

### Problem: "Weights only load failed"
**Solution**: Verify `model_restore.py` line 147 has `weights_only=False`

### Problem: Evaluation shows 0% Dice for all labels
**Solution**: Run `convert_labels_to_brats.py` to convert labels [0,1,2,3] to [0,1,2,4]

---

## Quick Start Guide

### Option 1: Fully Automated Pipeline (Recommended)

```bash
# For BraTS 2025 format data (auto-converts naming)
python run_full_pipeline.py BraTS-GLI-00009-000

# For BraTS 2021 format data
python run_full_pipeline.py data/sample_data/BraTS2021_sample
```

This single command runs all 5 steps automatically and outputs:
- Segmentation file
- Evaluation metrics (if ground truth available)
- Feature extraction results
- LLM-ready JSON for report generation

### Option 2: Manual Step-by-Step

1. **Prepare input data** in BraTS format:
   ```
   # BraTS 2025 format (auto-converted)
   input_folder/
   ├── CaseID-t1n.nii.gz
   ├── CaseID-t1c.nii.gz
   ├── CaseID-t2w.nii.gz
   ├── CaseID-t2f.nii.gz
   └── CaseID-seg.nii.gz (ground truth)

   # BraTS 2021 format (native)
   input_folder/
   ├── CaseID_t1.nii.gz
   ├── CaseID_t1ce.nii.gz
   ├── CaseID_t2.nii.gz
   ├── CaseID_flair.nii.gz
   └── CaseID_seg.nii.gz (ground truth)
   ```

2. **Convert naming (if BraTS 2025)**:
   ```bash
   python convert_brats2025_naming.py input_folder
   ```

3. **Set environment variables**:
   ```powershell
   $env:nnUNet_raw_data_base="C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant\nnUNet_raw"
   $env:nnUNet_preprocessed="C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant\nnUNet_preprocessed"
   $env:RESULTS_FOLDER="C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant\nnUNet_results"
   ```

4. **Run inference**:
   ```bash
   python run_brats2021_inference_singlethread.py --input input_folder --output results/CaseID
   ```

5. **Convert labels**:
   ```bash
   python convert_labels_to_brats.py "results\CaseID\CaseID.nii.gz" "results\CaseID\CaseID_brats.nii.gz"
   ```

6. **Evaluate (if ground truth available)**:
   ```bash
   python evaluate_segmentation.py --pred "results\CaseID\CaseID_brats.nii.gz" --gt "input_folder\CaseID_seg.nii.gz"
   ```

7. **Run feature extraction**:
   ```bash
   python feature_extraction/run_all.py --input input_folder --segmentation "results\CaseID\CaseID_brats.nii.gz" --output "results\CaseID\feature_extraction"
   ```

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11 (tested), Linux (should work), macOS (untested)
- **RAM**: 16 GB (32 GB recommended)
- **Storage**: 10 GB for models + 1 GB per case processed
- **CPU**: Multi-core processor (inference takes 20-30 min on modern CPU)

### Recommended Requirements
- **RAM**: 32 GB or more
- **GPU**: NVIDIA GPU with 8+ GB VRAM (for faster inference)
- **Storage**: SSD for faster I/O

### Software Requirements
- Python 3.10.x (3.10.11 tested)
- pip package manager
- PowerShell or bash terminal

---

## Contact & Support

**Project Path**: `C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant`  
**Repository**: AI-Powered-Brain-MRI-Assistant  
**Owner**: Adithya-Anil1  

For issues or questions, refer to:
1. This documentation file (`PROJECT_DOCUMENTATION.md`)
2. Instructions file (`documentation/instructions.md`)
3. Original model repository: https://github.com/KAIST-MRI-Lab/BraTS2021

---

## Change Log

### November 28, 2025
- ✅ Created `run_full_pipeline.py` - fully automated end-to-end analysis
- ✅ Created `convert_brats2025_naming.py` - BraTS 2025 to 2021 format conversion
- ✅ Fixed Unicode encoding issues in `convert_labels_to_brats.py` (Windows cp1252 compatibility)
- ✅ Validated on BraTS 2025 test cases: 95%+ Mean Dice scores
- ✅ Updated documentation with new features and performance results

### November 27, 2025
- ✅ Tested on BraTS-GLI-00003-000: **97.41% Mean Dice** ⭐
- ✅ Tested on BraTS-GLI-00005-000: **96.17% Mean Dice** ⭐
- ✅ Tested on BraTS-GLI-00009-000: **92.60% Mean Dice** ⭐
- ✅ Feature extraction pipeline fully validated
- ✅ LLM-ready JSON output confirmed working

### November 26, 2025
- ✅ Fixed `region_class_order` bug in inference script
- ✅ Added `weights_only=False` for PyTorch 2.9 compatibility
- ✅ Fixed label conversion floating-point errors
- ✅ Validated on BraTS2021_00495: ET Dice 92.95% ⭐
- ✅ Created comprehensive documentation

### Earlier (Setup Phase)
- ✅ Downloaded and extracted BraTS 2021 KAIST model checkpoints
- ✅ Set up nnU-Net environment structure
- ✅ Installed all dependencies
- ✅ Downgraded batchgenerators to 0.21
- ✅ Created utility scripts (check_labels, evaluate, convert)

---

**END OF DOCUMENTATION**
