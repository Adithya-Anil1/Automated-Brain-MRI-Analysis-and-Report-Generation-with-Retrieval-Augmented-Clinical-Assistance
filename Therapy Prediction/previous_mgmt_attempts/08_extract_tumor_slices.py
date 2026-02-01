"""
08_extract_tumor_slices.py
Level 4: Extract Tumor-Intersecting Axial Slices for Deep Learning

This script extracts 2D axial slices from MRI volumes where the tumor
segmentation mask is non-empty. Only tumor-containing slices are saved
to enable deep learning analysis of tumor regions.

Input:
    - Level4_MGMT_Dataset.csv (Patient_ID, Baseline_Scan_Path, MGMT_Label)
    - MRI volumes (FLAIR, T1c) and segmentation masks

Output:
    - Extracted slices as NPY files in slices_output/
    - slice_metadata.csv with slice-to-patient mapping
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Try to import nibabel for NIfTI loading
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("ERROR: nibabel is required. Install with: pip install nibabel")

# Configuration
SCRIPT_DIR = Path(__file__).parent
INPUT_CSV = SCRIPT_DIR / "Level4_MGMT_Dataset.csv"
OUTPUT_DIR = SCRIPT_DIR / "slices_output"
METADATA_CSV = OUTPUT_DIR / "slice_metadata.csv"

# Slice extraction parameters
MIN_TUMOR_PIXELS = 50  # Minimum tumor pixels to include slice
TARGET_SIZE = (224, 224)  # ResNet input size


def find_image_file(scan_folder: Path, modality: str) -> Optional[Path]:
    """
    Find an MRI image file for specified modality.
    
    Args:
        scan_folder: Path to patient scan folder
        modality: One of 'FLAIR', 'T1', 'CT1', 'T2'
    
    Returns:
        Path to image file or None
    """
    # Try direct file in scan folder
    direct_path = scan_folder / f"{modality}.nii.gz"
    if direct_path.exists():
        return direct_path
    
    # Try lowercase
    direct_path_lower = scan_folder / f"{modality.lower()}.nii.gz"
    if direct_path_lower.exists():
        return direct_path_lower
    
    return None


def find_mask_file(scan_folder: Path) -> Optional[Path]:
    """
    Find the tumor segmentation mask file.
    Prioritizes FLAIR segmentation mask.
    
    Returns:
        Path to mask file or None
    """
    # Priority 1: FLAIR segmentation mask
    flair_mask = scan_folder / "DeepBraTumIA-segmentation" / "native" / "segmentation" / "flair_seg_mask.nii.gz"
    if flair_mask.exists():
        return flair_mask
    
    # Priority 2: T1c segmentation mask
    t1c_mask = scan_folder / "DeepBraTumIA-segmentation" / "native" / "segmentation" / "ct1_seg_mask.nii.gz"
    if t1c_mask.exists():
        return t1c_mask
    
    # Priority 3: T1 mask
    t1_mask = scan_folder / "DeepBraTumIA-segmentation" / "native" / "segmentation" / "t1_seg_mask.nii.gz"
    if t1_mask.exists():
        return t1_mask
    
    return None


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    """
    Normalize a 2D slice to [0, 1] range with z-score normalization.
    
    Args:
        slice_2d: 2D numpy array
    
    Returns:
        Normalized slice
    """
    # Get non-zero values (brain tissue)
    brain_mask = slice_2d > 0
    if brain_mask.sum() == 0:
        return slice_2d
    
    # Z-score normalization on brain voxels
    brain_values = slice_2d[brain_mask]
    mean_val = np.mean(brain_values)
    std_val = np.std(brain_values)
    
    if std_val < 1e-8:
        return slice_2d
    
    # Normalize
    normalized = (slice_2d - mean_val) / std_val
    
    # Clip to reasonable range and scale to [0, 1]
    normalized = np.clip(normalized, -3, 3)
    normalized = (normalized + 3) / 6  # Map [-3, 3] to [0, 1]
    
    return normalized


def resize_slice(slice_2d: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize a 2D slice to target size using simple interpolation.
    
    Args:
        slice_2d: 2D numpy array
        target_size: (height, width) target dimensions
    
    Returns:
        Resized slice
    """
    # Use scipy if available, otherwise use numpy-based resize
    try:
        from scipy.ndimage import zoom
        h, w = slice_2d.shape
        th, tw = target_size
        zoom_factors = (th / h, tw / w)
        return zoom(slice_2d, zoom_factors, order=1)
    except ImportError:
        # Simple numpy-based resize (nearest neighbor)
        h, w = slice_2d.shape
        th, tw = target_size
        row_indices = (np.arange(th) * h / th).astype(int)
        col_indices = (np.arange(tw) * w / tw).astype(int)
        return slice_2d[row_indices[:, None], col_indices]


def extract_tumor_slices(
    patient_id: str,
    scan_folder: Path,
    output_dir: Path,
    min_tumor_pixels: int = MIN_TUMOR_PIXELS
) -> List[dict]:
    """
    Extract tumor-intersecting axial slices from a patient's MRI.
    
    Args:
        patient_id: Patient identifier
        scan_folder: Path to scan folder
        output_dir: Directory to save slices
        min_tumor_pixels: Minimum tumor pixels to include slice
    
    Returns:
        List of slice metadata dictionaries
    """
    slice_records = []
    
    # Find FLAIR image (primary modality)
    flair_path = find_image_file(scan_folder, "FLAIR")
    if flair_path is None:
        print(f"  WARNING: No FLAIR found for {patient_id}")
        return slice_records
    
    # Find mask
    mask_path = find_mask_file(scan_folder)
    if mask_path is None:
        print(f"  WARNING: No mask found for {patient_id}")
        return slice_records
    
    # Load volumes
    try:
        flair_nii = nib.load(str(flair_path))
        mask_nii = nib.load(str(mask_path))
        
        # Reorient both to canonical (RAS+) orientation to fix axis mismatches
        flair_canonical = nib.as_closest_canonical(flair_nii)
        mask_canonical = nib.as_closest_canonical(mask_nii)
        
        flair_data = flair_canonical.get_fdata()
        mask_data = mask_canonical.get_fdata()
        
        # Handle shape mismatches - first try to find correct axis permutation
        if flair_data.shape != mask_data.shape:
            flair_shape = flair_data.shape
            mask_shape = mask_data.shape
            
            # Sort mask shape dimensions and match to FLAIR shape
            # This finds the permutation that makes mask dimensions align with FLAIR
            flair_sorted_dims = sorted(enumerate(flair_shape), key=lambda x: x[1])
            mask_sorted_dims = sorted(enumerate(mask_shape), key=lambda x: x[1])
            
            # Check if dimensions can be matched by permutation
            flair_sorted_vals = [x[1] for x in flair_sorted_dims]
            mask_sorted_vals = [x[1] for x in mask_sorted_dims]
            
            if flair_sorted_vals == mask_sorted_vals:
                # Dimensions match - find the permutation
                # Create permutation: for each FLAIR axis, which mask axis has the same size?
                perm = [None, None, None]
                for fi, fv in enumerate(flair_shape):
                    for mi, mv in enumerate(mask_shape):
                        if fv == mv and mi not in perm[:fi]:
                            perm[fi] = mi
                            break
                
                if None not in perm:
                    mask_data = np.transpose(mask_data, perm)
            else:
                # Dimensions don't match exactly - try best-effort permutation
                # Find permutation that minimizes total dimension differences
                from itertools import permutations
                best_perm = None
                best_diff = float('inf')
                
                for perm in permutations([0, 1, 2]):
                    permuted_shape = tuple(mask_shape[p] for p in perm)
                    diff = sum(abs(f - m) for f, m in zip(flair_shape, permuted_shape))
                    if diff < best_diff:
                        best_diff = diff
                        best_perm = perm
                
                if best_perm:
                    mask_data = np.transpose(mask_data, best_perm)
            
            # After permutation, resample if still different
            if flair_data.shape != mask_data.shape:
                try:
                    from scipy.ndimage import zoom
                    zoom_factors = tuple(f / m for f, m in zip(flair_data.shape, mask_data.shape))
                    mask_data = zoom(mask_data, zoom_factors, order=0)  # Nearest neighbor for mask
                except ImportError:
                    # Fallback: use minimum common shape
                    min_shape = tuple(min(f, m) for f, m in zip(flair_data.shape, mask_data.shape))
                    flair_data = flair_data[:min_shape[0], :min_shape[1], :min_shape[2]]
                    mask_data = mask_data[:min_shape[0], :min_shape[1], :min_shape[2]]
    
    except Exception as e:
        print(f"  ERROR loading {patient_id}: {e}")
        return slice_records
    
    # Create patient output directory
    patient_output = output_dir / patient_id
    patient_output.mkdir(parents=True, exist_ok=True)
    
    # Extract axial slices (along z-axis, last dimension)
    n_slices = flair_data.shape[2]
    slices_saved = 0
    
    for z in range(n_slices):
        # Get axial slice
        flair_slice = flair_data[:, :, z]
        mask_slice = mask_data[:, :, z]
        
        # Check if tumor is present in this slice
        tumor_pixels = np.sum(mask_slice > 0)
        if tumor_pixels < min_tumor_pixels:
            continue
        
        # Normalize slice
        normalized_slice = normalize_slice(flair_slice)
        
        # Resize to target size
        resized_slice = resize_slice(normalized_slice, TARGET_SIZE)
        
        # Save as NPY
        slice_filename = f"slice_{z:03d}.npy"
        slice_path = patient_output / slice_filename
        np.save(str(slice_path), resized_slice.astype(np.float32))
        
        # Record metadata
        slice_records.append({
            'Patient_ID': patient_id,
            'Slice_Index': z,
            'Slice_Path': str(slice_path),
            'Tumor_Pixels': int(tumor_pixels)
        })
        slices_saved += 1
    
    print(f"  {patient_id}: Extracted {slices_saved} tumor slices (of {n_slices} total)")
    return slice_records


def main():
    """Main pipeline to extract tumor slices from all patients."""
    print("=" * 70)
    print("Level 4: Extract Tumor-Intersecting Slices for Deep Learning")
    print("=" * 70)
    
    if not NIBABEL_AVAILABLE:
        print("ERROR: nibabel is required. Install with: pip install nibabel")
        return
    
    # Load dataset
    if not INPUT_CSV.exists():
        print(f"ERROR: Dataset not found: {INPUT_CSV}")
        return
    
    df = pd.read_csv(INPUT_CSV)
    print(f"\nLoaded {len(df)} patients from {INPUT_CSV}")
    print(f"  Methylated: {(df['MGMT_Label'] == 1).sum()}")
    print(f"  Unmethylated: {(df['MGMT_Label'] == 0).sum()}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Extract slices for each patient
    all_slice_records = []
    successful_patients = 0
    
    print("\n" + "-" * 70)
    print("Extracting tumor slices...")
    print("-" * 70)
    
    for idx, row in df.iterrows():
        patient_id = row['Patient_ID']
        scan_folder = Path(row['Baseline_Scan_Path'])
        mgmt_label = row['MGMT_Label']
        
        if not scan_folder.exists():
            print(f"  WARNING: Scan folder not found: {scan_folder}")
            continue
        
        # Extract slices
        slice_records = extract_tumor_slices(patient_id, scan_folder, OUTPUT_DIR)
        
        # Add MGMT label to each slice record
        for record in slice_records:
            record['MGMT_Label'] = mgmt_label
        
        all_slice_records.extend(slice_records)
        
        if len(slice_records) > 0:
            successful_patients += 1
    
    # Create metadata DataFrame
    if len(all_slice_records) == 0:
        print("\nERROR: No slices extracted!")
        return
    
    metadata_df = pd.DataFrame(all_slice_records)
    metadata_df = metadata_df[['Patient_ID', 'Slice_Index', 'Slice_Path', 'Tumor_Pixels', 'MGMT_Label']]
    
    # Save metadata
    metadata_df.to_csv(METADATA_CSV, index=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Total patients processed: {successful_patients}")
    print(f"Total slices extracted: {len(metadata_df)}")
    print(f"\nSlices per class:")
    for label in [0, 1]:
        class_slices = len(metadata_df[metadata_df['MGMT_Label'] == label])
        class_patients = metadata_df[metadata_df['MGMT_Label'] == label]['Patient_ID'].nunique()
        status = "Methylated" if label == 1 else "Unmethylated"
        print(f"  {status}: {class_slices} slices from {class_patients} patients")
    
    print(f"\nAverage slices per patient: {len(metadata_df) / successful_patients:.1f}")
    print(f"\nMetadata saved to: {METADATA_CSV}")
    
    return metadata_df


if __name__ == "__main__":
    main()
