"""
Quick utility to check what labels are in a segmentation file.
Useful for debugging label mismatches.
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path


def check_labels(file_path):
    """Check what labels are present in a segmentation file"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ Error: File not found: {file_path}")
        return
    
    print("=" * 70)
    print(f"CHECKING LABELS IN: {file_path.name}")
    print("=" * 70)
    
    # Load file
    nii = nib.load(file_path)
    data = nii.get_fdata()
    
    print(f"\n📊 File Information:")
    print(f"   Shape: {data.shape}")
    print(f"   Data type: {data.dtype}")
    print(f"   File size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Get unique labels
    unique_labels = np.unique(data)
    
    print(f"\n🏷️  Unique Labels Found: {len(unique_labels)}")
    print(f"   {unique_labels}")
    
    # BraTS label definitions
    brats_labels = {
        0: 'Background',
        1: 'NCR (Necrotic Tumor Core)',
        2: 'ED (Peritumoral Edema)',
        3: 'ET (Enhancing Tumor) - Alternative',
        4: 'ET (Enhancing Tumor) - BraTS2021'
    }
    
    print(f"\n📋 Label Details:")
    for label in unique_labels:
        count = np.sum(data == label)
        percentage = (count / data.size) * 100
        label_name = brats_labels.get(label, f'Unknown (Label {label})')
        
        print(f"   Label {int(label):2d}: {label_name:40s} "
              f"| Count: {count:,} ({percentage:.2f}%)")
    
    # Check for expected BraTS labels
    print(f"\n✅ Expected BraTS2021 Labels Check:")
    expected = [0, 1, 2, 4]
    for exp_label in expected:
        if exp_label in unique_labels:
            print(f"   ✓ Label {exp_label} - FOUND")
        else:
            if exp_label == 0:
                print(f"   ⚠️  Label {exp_label} - NOT FOUND (unusual for background)")
            else:
                print(f"   ✗ Label {exp_label} - MISSING")
    
    # Check if this looks correct
    print(f"\n🔍 Analysis:")
    
    has_all_tumor_labels = all(l in unique_labels for l in [1, 2, 4])
    if has_all_tumor_labels:
        print("   ✅ All tumor labels present (1, 2, 4)")
        print("   ✅ This looks like CORRECT BraTS2021 segmentation")
    else:
        print("   ⚠️  Some tumor labels are missing!")
        missing = [l for l in [1, 2, 4] if l not in unique_labels]
        print(f"   Missing labels: {missing}")
        
        if 4 not in unique_labels:
            print("\n   ⚠️  CRITICAL: Label 4 (Enhancing Tumor) is missing!")
            print("   This is the main label for BraTS2021 model.")
            print("   Possible causes:")
            print("      • Wrong model was used for inference")
            print("      • Postprocessing script modified labels")
            print("      • Inference didn't complete properly")
        
        if 3 in unique_labels and 4 not in unique_labels:
            print("\n   ℹ️  Note: Found label 3 instead of 4")
            print("   Some BraTS models use label 3 for ET instead of 4.")
            print("   You may need to remap: 3 → 4")
    
    # Tumor volume statistics
    if any(l in unique_labels for l in [1, 2, 3, 4]):
        print(f"\n📏 Tumor Volume Statistics:")
        voxel_volume = np.prod(nii.header.get_zooms())  # mm³ per voxel
        
        tumor_mask = np.isin(data, [1, 2, 3, 4])
        tumor_voxels = np.sum(tumor_mask)
        tumor_volume_mm3 = tumor_voxels * voxel_volume
        tumor_volume_cm3 = tumor_volume_mm3 / 1000
        
        print(f"   Total tumor voxels: {tumor_voxels:,}")
        print(f"   Total tumor volume: {tumor_volume_cm3:.2f} cm³")
        
        for label in [1, 2, 3, 4]:
            if label in unique_labels:
                label_voxels = np.sum(data == label)
                label_volume = label_voxels * voxel_volume / 1000
                print(f"   Label {label} volume: {label_volume:.2f} cm³")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_labels.py <segmentation_file.nii.gz>")
        print("\nExample:")
        print("  python check_labels.py results\\BraTS2021_00495\\BraTS2021_00495.nii.gz")
        sys.exit(1)
    
    file_path = sys.argv[1]
    check_labels(file_path)
