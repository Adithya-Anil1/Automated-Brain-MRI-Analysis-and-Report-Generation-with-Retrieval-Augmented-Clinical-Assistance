"""
Visualize and compare predicted segmentation with ground truth side-by-side.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def compare_segmentations(pred_path, gt_path, output_dir=None):
    """
    Create side-by-side comparison of predicted and ground truth segmentations.
    """
    print("Loading segmentations...")
    pred_nii = nib.load(pred_path)
    gt_nii = nib.load(gt_path)
    
    pred_data = pred_nii.get_fdata()
    gt_data = gt_nii.get_fdata()
    
    print(f"Prediction shape: {pred_data.shape}")
    print(f"Ground truth shape: {gt_data.shape}")
    print(f"Unique labels in prediction: {np.unique(pred_data)}")
    print(f"Unique labels in ground truth: {np.unique(gt_data)}")
    
    # Create output directory
    if output_dir is None:
        output_dir = Path(pred_path).parent / "comparison"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Define color maps for visualization
    colors = {
        0: [0, 0, 0],        # Background - black
        1: [255, 0, 0],      # NCR - red
        2: [0, 255, 0],      # ED - green
        3: [0, 0, 255],      # ET - blue
        4: [255, 255, 0]     # ET (alternate) - yellow
    }
    
    def create_rgb_mask(mask):
        """Convert label mask to RGB image."""
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for label, color in colors.items():
            rgb[mask == label] = color
        return rgb
    
    # Select slices with tumor present
    tumor_slices = np.where(np.sum(gt_data, axis=(0, 1)) > 0)[0]
    
    if len(tumor_slices) == 0:
        print("No tumor found in ground truth!")
        return
    
    # Select evenly spaced slices
    num_slices = min(12, len(tumor_slices))
    selected_slices = tumor_slices[np.linspace(0, len(tumor_slices)-1, num_slices, dtype=int)]
    
    print(f"\nCreating comparison visualizations for {num_slices} slices...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_slices, 3, figsize=(15, 5*num_slices))
    if num_slices == 1:
        axes = axes.reshape(1, -1)
    
    for idx, slice_num in enumerate(selected_slices):
        # Ground truth
        gt_slice = gt_data[:, :, slice_num]
        gt_rgb = create_rgb_mask(gt_slice)
        
        # Prediction
        pred_slice = pred_data[:, :, slice_num]
        pred_rgb = create_rgb_mask(pred_slice)
        
        # Overlay (difference map)
        # Green: Correct, Red: False Positive, Blue: False Negative
        diff = np.zeros((*gt_slice.shape, 3), dtype=np.uint8)
        
        # Correct predictions (both have tumor)
        correct_mask = (gt_slice > 0) & (pred_slice > 0) & (gt_slice == pred_slice)
        diff[correct_mask] = [0, 255, 0]  # Green
        
        # Approximate correct (tumor detected but wrong class)
        approx_correct = (gt_slice > 0) & (pred_slice > 0) & (gt_slice != pred_slice)
        diff[approx_correct] = [255, 255, 0]  # Yellow
        
        # False positives (predicted tumor where there isn't)
        fp_mask = (gt_slice == 0) & (pred_slice > 0)
        diff[fp_mask] = [255, 0, 0]  # Red
        
        # False negatives (missed tumor)
        fn_mask = (gt_slice > 0) & (pred_slice == 0)
        diff[fn_mask] = [0, 0, 255]  # Blue
        
        # Plot
        axes[idx, 0].imshow(np.rot90(gt_rgb))
        axes[idx, 0].set_title(f'Ground Truth - Slice {slice_num}')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(np.rot90(pred_rgb))
        axes[idx, 1].set_title(f'Prediction - Slice {slice_num}')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(np.rot90(diff))
        axes[idx, 2].set_title(f'Difference - Slice {slice_num}')
        axes[idx, 2].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='NCR (Label 1)'),
        Patch(facecolor='green', label='ED (Label 2)'),
        Patch(facecolor='blue', label='ET (Label 3)'),
        Patch(facecolor='yellow', label='ET (Label 4)'),
        Patch(facecolor='black', label='Background')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Add difference legend
    diff_legend = [
        Patch(facecolor='green', label='Correct (exact match)'),
        Patch(facecolor='yellow', label='Approx. Correct (wrong class)'),
        Patch(facecolor='red', label='False Positive'),
        Patch(facecolor='blue', label='False Negative')
    ]
    fig.legend(handles=diff_legend, loc='lower right', bbox_to_anchor=(0.98, 0.02))
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "segmentation_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison saved to: {output_path}")
    
    plt.close()
    
    # Create summary statistics visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Label distribution
    pred_labels, pred_counts = np.unique(pred_data, return_counts=True)
    gt_labels, gt_counts = np.unique(gt_data, return_counts=True)
    
    label_names = {0: 'Background', 1: 'NCR', 2: 'ED', 3: 'ET', 4: 'ET'}
    
    # Plot prediction distribution
    axes[0].bar([label_names.get(l, f'L{l}') for l in pred_labels], pred_counts, color='skyblue', alpha=0.7, label='Prediction')
    axes[0].set_xlabel('Label')
    axes[0].set_ylabel('Voxel Count')
    axes[0].set_title('Prediction Label Distribution')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Plot ground truth distribution
    axes[1].bar([label_names.get(l, f'L{l}') for l in gt_labels], gt_counts, color='coral', alpha=0.7, label='Ground Truth')
    axes[1].set_xlabel('Label')
    axes[1].set_ylabel('Voxel Count')
    axes[1].set_title('Ground Truth Label Distribution')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    stats_path = output_dir / "label_distribution.png"
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    print(f"Label distribution saved to: {stats_path}")
    
    plt.close()
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print("\nFiles created:")
    print(f"  1. segmentation_comparison.png - Side-by-side comparison")
    print(f"  2. label_distribution.png - Label statistics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare segmentations visually')
    parser.add_argument('--pred', type=str, required=True,
                        help='Path to predicted segmentation')
    parser.add_argument('--gt', type=str, required=True,
                        help='Path to ground truth segmentation')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    compare_segmentations(args.pred, args.gt, args.output)
