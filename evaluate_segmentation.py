"""
Evaluate segmentation accuracy by comparing predicted segmentation with ground truth.
Calculates Dice Score, IoU (Jaccard Index), Sensitivity, and Specificity for each tumor class.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import argparse


def calculate_metrics(pred, gt, label):
    """
    Calculate segmentation metrics for a specific label.
    
    Args:
        pred: Predicted segmentation array
        gt: Ground truth segmentation array
        label: The label value to evaluate
        
    Returns:
        dict: Dictionary containing Dice, IoU, Sensitivity, and Specificity
    """
    # Create binary masks for the specific label
    pred_mask = (pred == label).astype(np.float32)
    gt_mask = (gt == label).astype(np.float32)
    
    # Calculate True Positives, False Positives, False Negatives, True Negatives
    tp = np.sum(pred_mask * gt_mask)
    fp = np.sum(pred_mask * (1 - gt_mask))
    fn = np.sum((1 - pred_mask) * gt_mask)
    tn = np.sum((1 - pred_mask) * (1 - gt_mask))
    
    # Calculate metrics
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)  # Also called Recall
    specificity = tn / (tn + fp + 1e-8)
    
    return {
        'dice': dice,
        'iou': iou,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def evaluate_segmentation(pred_path, gt_path):
    """
    Evaluate segmentation by comparing prediction with ground truth.
    
    Args:
        pred_path: Path to predicted segmentation file
        gt_path: Path to ground truth segmentation file
    """
    print("=" * 80)
    print("SEGMENTATION EVALUATION")
    print("=" * 80)
    print(f"\nPredicted file: {pred_path}")
    print(f"Ground truth file: {gt_path}")
    
    # Load the NIfTI files
    print("\nLoading files...")
    pred_nii = nib.load(pred_path)
    gt_nii = nib.load(gt_path)
    
    pred_data = pred_nii.get_fdata()
    gt_data = gt_nii.get_fdata()
    
    print(f"Prediction shape: {pred_data.shape}")
    print(f"Ground truth shape: {gt_data.shape}")
    
    # Check if shapes match
    if pred_data.shape != gt_data.shape:
        print("\n⚠️  WARNING: Shape mismatch! Attempting to resize...")
        # You might need to handle this case
        return
    
    # Get unique labels in both
    pred_labels = np.unique(pred_data)
    gt_labels = np.unique(gt_data)
    
    print(f"\nUnique labels in prediction: {pred_labels}")
    print(f"Unique labels in ground truth: {gt_labels}")
    
    # BraTS label definitions
    label_names = {
        0: 'Background',
        1: 'NCR (Necrotic Tumor Core)',
        2: 'ED (Peritumoral Edema)',
        3: 'ET (Enhancing Tumor)',
        4: 'ET (Enhancing Tumor - alternate)'  # Sometimes label 4 is used
    }
    
    # Calculate metrics for each label
    print("\n" + "=" * 80)
    print("RESULTS BY CLASS")
    print("=" * 80)
    
    all_metrics = {}
    labels_to_evaluate = sorted(set(pred_labels) | set(gt_labels))
    
    for label in labels_to_evaluate:
        if label == 0:  # Skip background
            continue
            
        label_name = label_names.get(label, f'Label {label}')
        metrics = calculate_metrics(pred_data, gt_data, label)
        all_metrics[label] = metrics
        
        print(f"\n{label_name} (Label {label}):")
        print(f"  Dice Score:      {metrics['dice']:.4f} ({metrics['dice']*100:.2f}%)")
        print(f"  IoU (Jaccard):   {metrics['iou']:.4f} ({metrics['iou']*100:.2f}%)")
        print(f"  Sensitivity:     {metrics['sensitivity']:.4f} ({metrics['sensitivity']*100:.2f}%)")
        print(f"  Specificity:     {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
        print(f"  True Positives:  {int(metrics['tp']):,}")
        print(f"  False Positives: {int(metrics['fp']):,}")
        print(f"  False Negatives: {int(metrics['fn']):,}")
    
    # Calculate compound metrics (commonly used in BraTS)
    print("\n" + "=" * 80)
    print("COMPOUND METRICS (BraTS Standard)")
    print("=" * 80)
    
    # Whole Tumor (WT): combines labels 1, 2, and 3
    pred_wt = np.isin(pred_data, [1, 2, 3]).astype(np.float32)
    gt_wt = np.isin(gt_data, [1, 2, 3]).astype(np.float32)
    wt_metrics = calculate_metrics_binary(pred_wt, gt_wt)
    
    print("\nWhole Tumor (WT) - Labels 1, 2, 3 combined:")
    print(f"  Dice Score:      {wt_metrics['dice']:.4f} ({wt_metrics['dice']*100:.2f}%)")
    print(f"  IoU:             {wt_metrics['iou']:.4f} ({wt_metrics['iou']*100:.2f}%)")
    print(f"  Sensitivity:     {wt_metrics['sensitivity']:.4f} ({wt_metrics['sensitivity']*100:.2f}%)")
    
    # Tumor Core (TC): combines labels 1 and 3
    pred_tc = np.isin(pred_data, [1, 3]).astype(np.float32)
    gt_tc = np.isin(gt_data, [1, 3]).astype(np.float32)
    tc_metrics = calculate_metrics_binary(pred_tc, gt_tc)
    
    print("\nTumor Core (TC) - Labels 1, 3 combined:")
    print(f"  Dice Score:      {tc_metrics['dice']:.4f} ({tc_metrics['dice']*100:.2f}%)")
    print(f"  IoU:             {tc_metrics['iou']:.4f} ({tc_metrics['iou']*100:.2f}%)")
    print(f"  Sensitivity:     {tc_metrics['sensitivity']:.4f} ({tc_metrics['sensitivity']*100:.2f}%)")
    
    # Enhancing Tumor (ET): label 3 only
    if 3 in all_metrics:
        et_metrics = all_metrics[3]
        print("\nEnhancing Tumor (ET) - Label 3 only:")
        print(f"  Dice Score:      {et_metrics['dice']:.4f} ({et_metrics['dice']*100:.2f}%)")
        print(f"  IoU:             {et_metrics['iou']:.4f} ({et_metrics['iou']*100:.2f}%)")
        print(f"  Sensitivity:     {et_metrics['sensitivity']:.4f} ({et_metrics['sensitivity']*100:.2f}%)")
    
    # Calculate mean Dice score across all regions
    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)
    mean_dice = np.mean([wt_metrics['dice'], tc_metrics['dice'], et_metrics['dice'] if 3 in all_metrics else 0])
    print(f"\nMean Dice Score (WT, TC, ET): {mean_dice:.4f} ({mean_dice*100:.2f}%)")
    
    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("\nDice Score Interpretation:")
    print("  > 0.90: Excellent")
    print("  0.80 - 0.90: Good")
    print("  0.70 - 0.80: Moderate")
    print("  0.50 - 0.70: Fair")
    print("  < 0.50: Poor")
    
    print("\nNote: BraTS competition typically reports Dice scores for WT, TC, and ET.")
    print("      State-of-the-art models achieve Dice scores of 0.85-0.92 for these regions.")
    
    return all_metrics


def calculate_metrics_binary(pred_mask, gt_mask):
    """Calculate metrics for binary masks."""
    tp = np.sum(pred_mask * gt_mask)
    fp = np.sum(pred_mask * (1 - gt_mask))
    fn = np.sum((1 - pred_mask) * gt_mask)
    
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    
    return {
        'dice': dice,
        'iou': iou,
        'sensitivity': sensitivity
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate brain tumor segmentation')
    parser.add_argument('--pred', type=str, required=True,
                        help='Path to predicted segmentation file (.nii.gz)')
    parser.add_argument('--gt', type=str, required=True,
                        help='Path to ground truth segmentation file (.nii.gz)')
    
    args = parser.parse_args()
    
    pred_path = Path(args.pred)
    gt_path = Path(args.gt)
    
    if not pred_path.exists():
        print(f"Error: Predicted file not found: {pred_path}")
        exit(1)
        
    if not gt_path.exists():
        print(f"Error: Ground truth file not found: {gt_path}")
        exit(1)
    
    evaluate_segmentation(pred_path, gt_path)
