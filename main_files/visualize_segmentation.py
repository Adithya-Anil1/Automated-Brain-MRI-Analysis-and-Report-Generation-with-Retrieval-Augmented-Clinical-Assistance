"""
Visualize BraTS segmentation results
Overlays the predicted segmentation mask on MRI images
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_nifti(filepath):
    """Load NIfTI file and return data array."""
    img = nib.load(filepath)
    return img.get_fdata()


def create_rgb_overlay(mri_slice, seg_slice, alpha=0.4):
    """
    Create RGB visualization with segmentation overlay.
    
    Args:
        mri_slice: 2D MRI slice (grayscale)
        seg_slice: 2D segmentation slice (labels)
        alpha: Transparency of overlay (0=transparent, 1=opaque)
    
    Returns:
        RGB image with overlay
    """
    # Normalize MRI to 0-1 range
    mri_norm = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-8)
    
    # Create RGB image from grayscale MRI
    rgb = np.stack([mri_norm, mri_norm, mri_norm], axis=-1)
    
    # Define colors for each tumor region
    colors = {
        0: [0, 0, 0],           # Background - black (invisible)
        1: [1, 0, 0],           # NCR - red
        2: [0, 1, 0],           # ED - green  
        4: [0, 0, 1],           # ET - blue
    }
    
    # Create overlay
    for label, color in colors.items():
        if label == 0:
            continue
        mask = (seg_slice == label)
        for c in range(3):
            rgb[:, :, c] = np.where(mask, 
                                   (1 - alpha) * rgb[:, :, c] + alpha * color[c],
                                   rgb[:, :, c])
    
    return rgb


def visualize_case(mri_dir, seg_path, output_dir, num_slices=9):
    """
    Visualize segmentation results for a case.
    
    Args:
        mri_dir: Directory containing MRI modalities
        seg_path: Path to segmentation file
        output_dir: Directory to save visualizations
        num_slices: Number of slices to show
    """
    mri_dir = Path(mri_dir)
    seg_path = Path(seg_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find MRI files
    case_name = seg_path.stem.replace('.nii', '')
    
    # Try to find modality files
    modalities = {}
    for mod_name, patterns in [
        ('FLAIR', ['*flair.nii.gz', '*_0003.nii.gz']),
        ('T1', ['*t1.nii.gz', '*_0000.nii.gz']),
        ('T1CE', ['*t1ce.nii.gz', '*_0001.nii.gz']),
        ('T2', ['*t2.nii.gz', '*_0002.nii.gz'])
    ]:
        for pattern in patterns:
            files = list(mri_dir.glob(pattern))
            if files:
                modalities[mod_name] = load_nifti(files[0])
                print(f"Loaded {mod_name}: {files[0].name}")
                break
    
    if not modalities:
        print("[ERROR] No MRI modalities found!")
        return
    
    # Load segmentation
    seg = load_nifti(seg_path)
    print(f"Loaded segmentation: {seg_path.name}")
    print(f"Segmentation shape: {seg.shape}")
    print(f"Unique labels: {np.unique(seg)}")
    
    # Get dimensions
    depth = seg.shape[2]
    
    # Select slices evenly distributed through the volume
    # Focus on middle slices where tumor is usually visible
    start_slice = depth // 4
    end_slice = 3 * depth // 4
    slice_indices = np.linspace(start_slice, end_slice, num_slices, dtype=int)
    
    # Create figure for each modality
    for mod_name, mri_data in modalities.items():
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f'{case_name} - {mod_name} with Segmentation Overlay', fontsize=16)
        
        for idx, slice_num in enumerate(slice_indices):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # Get slices
            mri_slice = mri_data[:, :, slice_num]
            seg_slice = seg[:, :, slice_num]
            
            # Create overlay
            overlay = create_rgb_overlay(mri_slice, seg_slice, alpha=0.4)
            
            # Display
            ax.imshow(overlay)
            ax.set_title(f'Slice {slice_num}/{depth}')
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        output_file = output_dir / f'{case_name}_{mod_name}_overlay.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    # Create a comparison figure showing all modalities for one slice
    mid_slice = depth // 2
    
    # Find slice with most tumor
    tumor_counts = []
    for s in range(depth):
        tumor_counts.append(np.sum(seg[:, :, s] > 0))
    best_slice = np.argmax(tumor_counts)
    
    print(f"\nCreating detailed view of slice {best_slice} (most tumor visible)")
    
    fig, axes = plt.subplots(2, len(modalities), figsize=(20, 10))
    fig.suptitle(f'{case_name} - Slice {best_slice} - All Modalities', fontsize=16)
    
    for idx, (mod_name, mri_data) in enumerate(modalities.items()):
        # Original MRI
        axes[0, idx].imshow(mri_data[:, :, best_slice], cmap='gray')
        axes[0, idx].set_title(f'{mod_name} (Original)')
        axes[0, idx].axis('off')
        
        # With overlay
        mri_slice = mri_data[:, :, best_slice]
        seg_slice = seg[:, :, best_slice]
        overlay = create_rgb_overlay(mri_slice, seg_slice, alpha=0.5)
        axes[1, idx].imshow(overlay)
        axes[1, idx].set_title(f'{mod_name} + Segmentation')
        axes[1, idx].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='NCR (Necrotic Core)'),
        Patch(facecolor='green', label='ED (Edema)'),
        Patch(facecolor='blue', label='ET (Enhancing Tumor)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12)
    
    plt.tight_layout()
    output_file = output_dir / f'{case_name}_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()
    
    print(f"\n{'='*70}")
    print("VISUALIZATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize BraTS segmentation results')
    parser.add_argument('--mri', type=str, required=True, help='Directory with MRI files')
    parser.add_argument('--seg', type=str, required=True, help='Path to segmentation file')
    parser.add_argument('--output', type=str, required=True, help='Output directory for visualizations')
    parser.add_argument('--slices', type=int, default=9, help='Number of slices to visualize')
    
    args = parser.parse_args()
    
    visualize_case(args.mri, args.seg, args.output, args.slices)


if __name__ == "__main__":
    main()
