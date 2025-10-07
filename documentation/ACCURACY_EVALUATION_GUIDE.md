# Segmentation Accuracy Evaluation - Summary Report

## Overview
This document summarizes the evaluation of your brain tumor segmentation model against ground truth data.

## Files Generated
1. **`evaluate_segmentation.py`** - Script to calculate accuracy metrics (Dice, IoU, Sensitivity, Specificity)
2. **`compare_segmentations.py`** - Script to create visual comparisons
3. **Comparison visualizations** in `results/BraTS2021_00495/comparison/`

## How to Use

### 1. Calculate Accuracy Metrics
```bash
# Activate your virtual environment
venv310\Scripts\activate

# Run evaluation
python evaluate_segmentation.py --pred "path/to/prediction.nii.gz" --gt "path/to/ground_truth.nii.gz"
```

Example:
```bash
python evaluate_segmentation.py --pred "results\BraTS2021_00495\BraTS2021_00495.nii.gz" --gt "sample_data\BraTS2021_sample\BraTS2021_00495_seg.nii.gz"
```

### 2. Create Visual Comparisons
```bash
python compare_segmentations.py --pred "path/to/prediction.nii.gz" --gt "path/to/ground_truth.nii.gz" --output "output/directory"
```

## Current Results (BraTS2021_00495)

### Metrics by Class
- **NCR (Necrotic Tumor Core - Label 1)**: Dice = 5.29%
- **ED (Peritumoral Edema - Label 2)**: Dice = 0.12%
- **ET (Enhancing Tumor - Label 4)**: Dice = 0.00%

### Compound Metrics (BraTS Standard)
- **Whole Tumor (WT)**: Dice = 2.45%
- **Tumor Core (TC)**: Dice = 5.29%
- **Mean Dice**: 2.58%

### Performance Classification
**POOR** - Current performance is significantly below clinical standards.

Expected performance for medical-grade models:
- Excellent: > 90%
- Good: 80-90%
- Moderate: 70-80%
- Fair: 50-70%
- **Poor: < 50%** ← Your current result

## Analysis

### Issues Identified
1. **Label Mismatch**: 
   - Ground truth contains labels: 0, 1, 2, 4
   - Prediction contains labels: 0, 1, 2 only
   - Missing label 3 (ET - Enhancing Tumor)

2. **Very Low Accuracy**: 
   - Dice scores < 6% indicate the model is not performing segmentation correctly
   - High false positive and false negative rates

3. **Possible Causes**:
   - Model may not have been properly trained
   - Incorrect preprocessing of input data
   - Model architecture mismatch with task
   - Training data quality issues
   - Inference configuration errors

## Recommendations

### Immediate Actions

1. **Verify Ground Truth Labels**
   ```bash
   python -c "import nibabel as nib; import numpy as np; gt=nib.load('sample_data/BraTS2021_sample/BraTS2021_00495_seg.nii.gz'); print('Unique labels:', np.unique(gt.get_fdata()))"
   ```

2. **Check Model Training**
   - Verify the model completed training successfully
   - Check training loss curves and validation metrics
   - Ensure the model was trained on similar data format

3. **Review Preprocessing**
   - Ensure input data is properly normalized
   - Check if the input modalities (T1, T1CE, T2, FLAIR) are in the correct order
   - Verify spatial dimensions match training data

4. **Try a Different Model**
   - Consider using a pre-trained model from the BraTS competition
   - Check if nnU-Net default settings work better
   - Verify model checkpoint files are correct

### Long-term Solutions

1. **Retrain the Model**
   - Use a larger, more diverse training dataset
   - Train for more epochs until convergence
   - Use data augmentation
   - Monitor validation metrics during training

2. **Use Transfer Learning**
   - Start from a pre-trained BraTS model
   - Fine-tune on your specific data

3. **Ensemble Methods**
   - Combine predictions from multiple models
   - Can improve accuracy by 2-5%

## Metrics Explanation

### Dice Score (Sørensen-Dice Coefficient)
- Measures overlap between prediction and ground truth
- Formula: `2 × TP / (2 × TP + FP + FN)`
- Range: 0 (no overlap) to 1 (perfect match)
- Most commonly used metric in medical image segmentation

### IoU (Intersection over Union / Jaccard Index)
- Similar to Dice but more strict
- Formula: `TP / (TP + FP + FN)`
- Range: 0 to 1

### Sensitivity (Recall / True Positive Rate)
- Measures how well the model detects actual tumor
- Formula: `TP / (TP + FN)`
- Important for not missing tumors

### Specificity
- Measures how well the model identifies non-tumor regions
- Formula: `TN / (TN + FP)`
- Important for avoiding false alarms

### BraTS Standard Regions
1. **WT (Whole Tumor)**: All tumor labels combined (1, 2, 3/4)
2. **TC (Tumor Core)**: NCR + ET (labels 1, 3/4)
3. **ET (Enhancing Tumor)**: Only enhancing tumor (label 3 or 4)

## Next Steps

1. **Investigate the low accuracy** - Check model training logs
2. **Verify data preprocessing** - Ensure inputs match model expectations
3. **Test with different samples** - Check if issue is sample-specific or systematic
4. **Consider retraining** - If model is fundamentally flawed

## Visualizations

Check the comparison folder:
- `segmentation_comparison.png` - Side-by-side view of GT vs Prediction
- `label_distribution.png` - Shows distribution of labels in each

Color coding in comparisons:
- **Red** = NCR (Label 1)
- **Green** = ED (Label 2)
- **Blue** = ET (Label 3)
- **Yellow** = ET (Label 4)

In difference maps:
- **Green** = Correct prediction
- **Yellow** = Approximate (detected tumor, wrong class)
- **Red** = False Positive (predicted tumor where none exists)
- **Blue** = False Negative (missed tumor)

## References
- BraTS Challenge: https://www.synapse.org/#!Synapse:syn51156910
- nnU-Net Paper: https://arxiv.org/abs/1809.10486
- Dice Score: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
