"""
Convert nnU-Net internal labels (0,1,2,3) to BraTS convention

Supports two output formats:
- BraTS 2025 (default): Labels [0, 1, 2, 3] - ET uses label 3
- BraTS 2021 (legacy):  Labels [0, 1, 2, 4] - ET uses label 4

nnU-Net internal:
- 0: Background
- 1: Edema (ED)
- 2: Non-enhancing tumor core (NCR) 
- 3: Enhancing tumor (ET)

BraTS 2025 convention:
- 0: Background
- 1: NCR (Necrotic/Non-enhancing tumor core)
- 2: ED (Peritumoral edema)  
- 3: ET (GD-enhancing tumor)

BraTS 2021 convention (legacy):
- 0: Background
- 1: NCR (Necrotic/Non-enhancing tumor core)
- 2: ED (Peritumoral edema)  
- 4: ET (GD-enhancing tumor)
"""

import nibabel as nib
import numpy as np
import sys
import argparse
from pathlib import Path


def convert_labels_to_brats2025(seg: np.ndarray):
    """Convert from nnU-Net labels (0,1,2,3) to BraTS 2025 labels (0,1,2,3)"""
    # Round to nearest integer first to handle floating point errors
    seg = np.round(seg).astype(np.uint8)
    
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 2  # ED: nnU-Net 1 → BraTS 2
    new_seg[seg == 2] = 1  # NCR: nnU-Net 2 → BraTS 1
    new_seg[seg == 3] = 3  # ET: nnU-Net 3 → BraTS 3 (unchanged)
    return new_seg


def convert_labels_to_brats2021(seg: np.ndarray):
    """Convert from nnU-Net labels (0,1,2,3) to BraTS 2021 labels (0,1,2,4)"""
    # Round to nearest integer first to handle floating point errors
    seg = np.round(seg).astype(np.uint8)
    
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 2  # ED: nnU-Net 1 → BraTS 2
    new_seg[seg == 2] = 1  # NCR: nnU-Net 2 → BraTS 1
    new_seg[seg == 3] = 4  # ET: nnU-Net 3 → BraTS 4
    return new_seg


def convert_file(input_path, output_path, format="brats2025"):
    """Convert a single NIfTI file"""
    print(f"\n{'='*70}")
    print(f"Converting: {input_path}")
    print(f"Format: {format.upper()}")
    print(f"{'='*70}")
    
    # Load the segmentation
    img = nib.load(input_path)
    data = img.get_fdata()
    
    # Check current labels
    unique_before = np.unique(data)
    print(f"\nLabels before conversion: {unique_before}")
    
    # Convert labels based on format
    if format == "brats2025":
        data_converted = convert_labels_to_brats2025(data)
        expected_labels = {0, 1, 2, 3}
        et_label = 3
    else:  # brats2021
        data_converted = convert_labels_to_brats2021(data)
        expected_labels = {0, 1, 2, 4}
        et_label = 4
    
    # Check new labels
    unique_after = np.unique(data_converted)
    print(f"Labels after conversion:  {unique_after}")
    
    # Show label mapping
    print(f"\nLabel mapping applied ({format.upper()}):")
    if 1 in unique_before:
        print(f"  1 (ED) → 2 (ED)")
    if 2 in unique_before:
        print(f"  2 (NCR) → 1 (NCR)")
    if 3 in unique_before:
        print(f"  3 (ET) → {et_label} (ET)  ✓ CRITICAL CONVERSION")
    
    # Save converted segmentation
    img_converted = nib.Nifti1Image(data_converted, img.affine, img.header)
    nib.save(img_converted, output_path)
    
    print(f"\n✅ Saved converted segmentation to: {output_path}")
    print(f"\nExpected {format.upper()} labels: {sorted(expected_labels)}")
    print(f"Actual labels in output: {unique_after}")
    
    if set(unique_after) == expected_labels:
        print(f"✅ SUCCESS: All {format.upper()} labels present!")
    elif et_label not in unique_after:
        print(f"⚠️  WARNING: Label {et_label} missing - check if input had label 3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert nnU-Net labels to BraTS format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # BraTS 2025 format (default) - outputs labels [0,1,2,3]
  python convert_labels_to_brats.py results/case.nii.gz results/case_brats.nii.gz

  # BraTS 2021 format (legacy) - outputs labels [0,1,2,4]  
  python convert_labels_to_brats.py results/case.nii.gz results/case_brats.nii.gz --format brats2021
        """
    )
    parser.add_argument("input", help="Input NIfTI file with nnU-Net labels [0,1,2,3]")
    parser.add_argument("output", nargs="?", help="Output NIfTI file (optional, defaults to input_brats.nii.gz)")
    parser.add_argument("--format", choices=["brats2025", "brats2021"], default="brats2025",
                        help="Output format: brats2025 (default, ET=3) or brats2021 (legacy, ET=4)")
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    
    if args.output:
        output_file = Path(args.output)
    else:
        # Default: add _brats suffix
        output_file = input_file.parent / (input_file.stem.replace('.nii', '_brats.nii') + '.gz')
    
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    convert_file(str(input_file), str(output_file), format=args.format)
