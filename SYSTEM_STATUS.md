# AI-Powered Brain MRI Assistant - Current Status

**Date**: November 26, 2025

## ✅ Working Components

### 1. Segmentation Model
- **Model**: BraTS 2021 KAIST MRI Lab (Model 1: nnUNetTrainerV2BraTSRegions_DA4_BN_BD)
- **Status**: Functional
- **Output Labels**: 
  - 0: Background
  - 1: NCR (Necrotic Tumor Core) 
  - 2: ED (Peritumoral Edema)
- **Missing**: Label 4 (Enhancing Tumor) - not being detected by model

### 2. Utilities
- ✅ `check_labels.py` - Verify segmentation labels
- ✅ `evaluate_segmentation.py` - Calculate Dice scores
- ✅ `visualize_segmentation.py` - Generate 3D visualizations
- ✅ `convert_labels_to_brats.py` - Label format conversion (when needed)

### 3. Test Data
- **Available**: 1 sample (BraTS2021_00495)
- **Location**: `data\sample_data\BraTS2021_sample\`
- **Files**: T1, T1CE, T2, FLAIR, segmentation ground truth

## ⚠️ Known Issues

### Model Limitations
1. **Enhancing Tumor (ET) not detected**: Model outputs only labels 0, 1, 2 instead of 0, 1, 2, 4
   - This causes low Dice scores (~2-5%) when comparing to ground truth
   - Ground truth has label 4 (32,047 voxels) but prediction has 0 voxels
   
2. **Model 2 crashes on CPU**: largeUnet_Groupnorm model is too memory-intensive
   - Hangs during "computing Gaussian" step
   - KeyboardInterrupt occurs during convolution operations

### Architecture Issues
- BraTS models use internal labels (1, 2, 3) that should convert to BraTS format (1, 2, 4)
- Label 3 (internal ET) → Label 4 (BraTS ET) conversion expected but not happening
- Model appears to not generate label 3 at all

## 📊 Current Results

### Segmentation Output (Model 1)
```
Location: results\BraTS2021_00495\BraTS2021_00495.nii.gz
Labels: [0, 1, 2]
- Background: 8,898,240 voxels (99.67%)
- NCR: 20,993 voxels (0.24%)
- ED: 8,767 voxels (0.10%)
Total tumor: 29.76 cm³
```

### Ground Truth
```
Location: data\sample_data\BraTS2021_sample\BraTS2021_00495_seg.nii.gz
Labels: [0, 1, 2, 4]
- Background: 8,828,698 voxels (98.89%)
- NCR: 20,808 voxels (0.23%)
- ED: 46,447 voxels (0.52%)
- ET: 32,047 voxels (0.36%)
Total tumor: 99.30 cm³
```

### Accuracy Metrics
- **Whole Tumor (WT)**: ~2-5% Dice (very low due to missing ET)
- **Tumor Core (TC)**: ~2-5% Dice
- **Enhancing Tumor (ET)**: 0% (not detected)

## 🎯 Next Steps (Options)

### Option 1: Get More Test Cases
- Download additional BraTS 2021/2024 samples
- Test if other cases produce label 3/4
- Verify if issue is case-specific or model-wide

### Option 2: Accept Current Limitations
- Use segmentation for Whole Tumor (WT) only
- Extract features from NCR and ED regions
- Build report generation based on available labels

### Option 3: Find Working Model
- Download alternative BraTS 2021 pre-trained model
- Verify it outputs all 4 labels before using
- Re-run inference pipeline

### Option 4: Focus on Visualization & Reporting
- Generate medical reports from available segmentation
- Create 3D visualizations of detected tumor regions
- Calculate volumes and statistics for NCR and ED

## 📝 Project Goals Alignment

**Original Goal**: AI-Powered Brain MRI Assistant with segmentation and report generation

**Current Capability**: 
- ✅ Segmentation working (partial - missing ET)
- ✅ Volume calculations
- ✅ Visualization
- ⏳ Report generation (ready to implement with current data)
- ❌ Classification (skipped per user request)

## 🔧 Technical Environment

- **Python**: 3.10.11
- **PyTorch**: 2.9.1 (CPU only)
- **Key Packages**: nibabel 5.3.2, SimpleITK 2.5.3, matplotlib 3.10.7
- **Model Framework**: nnU-Net (modified for BraTS 2021)
- **Hardware**: CPU-based inference (no GPU)

## 💡 Recommendations

1. **Short-term**: Proceed with report generation using available segmentation (NCR + ED)
2. **Medium-term**: Download 5-10 more test cases to verify model behavior
3. **Long-term**: Consider getting GPU access or finding pre-trained model that works on CPU

---

**Status**: System functional but limited by model's inability to detect Enhancing Tumor (ET)
