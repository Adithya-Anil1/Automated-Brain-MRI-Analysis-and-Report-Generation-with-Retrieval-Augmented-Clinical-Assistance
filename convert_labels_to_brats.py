"""
Convert nnU-Net internal labels (0,1,2,3) back to BraTS convention (0,1,2,4)

nnU-Net internal:
- 0: Background
- 1: Edema (ED)
- 2: Non-enhancing tumor core (NCR) 
- 3: Enhancing tumor (ET)

BraTS convention:
- 0: Background
- 1: NCR (Necrotic/Non-enhancing tumor core)
- 2: ED (Peritumoral edema)  
- 4: ET (GD-enhancing tumor)
"""

import nibabel as nib
import numpy as np
import sys
from pathlib import Path


def convert_labels_back_to_brats(seg: np.ndarray):
    """Convert from nnU-Net labels (0,1,2,3) to BraTS labels (0,1,2,4)"""
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 2  # ED: nnU-Net 1 → BraTS 2
    new_seg[seg == 2] = 1  # NCR: nnU-Net 2 → BraTS 1
    new_seg[seg == 3] = 4  # ET: nnU-Net 3 → BraTS 4
    return new_seg


def convert_file(input_path, output_path):
    """Convert a single NIfTI file"""
    print(f"\n{'='*70}")
    print(f"Converting: {input_path}")
    print(f"{'='*70}")
    
    # Load the segmentation
    img = nib.load(input_path)
    data = img.get_fdata()
    
    # Check current labels
    unique_before = np.unique(data)
    print(f"\nLabels before conversion: {unique_before}")
    
    # Convert labels
    data_converted = convert_labels_back_to_brats(data)
    
    # Check new labels
    unique_after = np.unique(data_converted)
    print(f"Labels after conversion:  {unique_after}")
    
    # Show label mapping
    print(f"\nLabel mapping applied:")
    if 1 in unique_before:
        print(f"  1 (ED) → 2 (ED)")
    if 2 in unique_before:
        print(f"  2 (NCR) → 1 (NCR)")
    if 3 in unique_before:
        print(f"  3 (ET) → 4 (ET)  ✓ CRITICAL CONVERSION")
    
    # Save converted segmentation
    img_converted = nib.Nifti1Image(data_converted, img.affine, img.header)
    nib.save(img_converted, output_path)
    
    print(f"\n✅ Saved converted segmentation to: {output_path}")
    print(f"\nExpected BraTS labels: [0, 1, 2, 4]")
    print(f"Actual labels in output: {unique_after}")
    
    if set(unique_after) == {0, 1, 2, 4}:
        print("✅ SUCCESS: All BraTS labels present!")
    elif 4 not in unique_after:
        print("⚠️  WARNING: Label 4 still missing - check if input had label 3")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_labels_to_brats.py <input_file> [output_file]")
        print("\nExample:")
        print('  python convert_labels_to_brats.py "results\\temp_model1\\BraTS2021_00495.nii.gz" "results\\BraTS2021_00495_fixed.nii.gz"')
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    
    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2])
    else:
        # Default: add _brats suffix
        output_file = input_file.parent / (input_file.stem.replace('.nii', '_brats.nii') + '.gz')
    
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    convert_file(str(input_file), str(output_file))
