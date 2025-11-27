"""
Shared utility functions for feature extraction.

Author: AI-Powered Brain MRI Assistant
Date: November 27, 2025
"""

import numpy as np
import nibabel as nib
import json
from pathlib import Path
from scipy import ndimage


def load_nifti(filepath):
    """Load a NIfTI file and return data array, affine, and header."""
    img = nib.load(filepath)
    return img.get_fdata(), img.affine, img.header


def save_nifti(data, affine, header, filepath):
    """Save data as a NIfTI file."""
    img = nib.Nifti1Image(data, affine, header)
    nib.save(img, filepath)


def get_intensity_stats(data, mask):
    """Calculate intensity statistics within a masked region."""
    if mask.sum() == 0:
        return {
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'median': None,
            'q25': None,
            'q75': None,
            'voxel_count': 0
        }
    
    values = data[mask > 0]
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75)),
        'voxel_count': int(mask.sum())
    }


def get_normal_brain_stats(data, seg_mask):
    """Get intensity statistics from normal brain tissue (non-tumor regions)."""
    # Create mask for non-tumor, non-background regions
    brain_mask = data > np.percentile(data[data > 0], 5) if data.max() > 0 else data > 0
    normal_mask = brain_mask & (seg_mask == 0)
    
    return get_intensity_stats(data, normal_mask)


def get_brain_mask(data, threshold_percentile=5):
    """Create a brain mask from MRI data."""
    if data.max() == 0:
        return data > 0
    threshold = np.percentile(data[data > 0], threshold_percentile)
    return data > threshold


def get_case_id(input_folder):
    """Extract case ID from input folder."""
    input_folder = Path(input_folder)
    
    # Try BraTS 2021 format first
    files = list(input_folder.glob("*_t1.nii.gz"))
    if files:
        return files[0].name.split('_t1')[0]
    
    # Try BraTS 2025 format
    files = list(input_folder.glob("*-t1n.nii.gz"))
    if files:
        return files[0].name.split('-t1')[0]
    
    # Fallback to folder name
    return input_folder.name


def get_mri_paths(input_folder, case_id=None):
    """Get paths to all MRI modality files."""
    input_folder = Path(input_folder)
    
    if case_id is None:
        case_id = get_case_id(input_folder)
    
    # Check BraTS 2021 format
    if (input_folder / f"{case_id}_t1.nii.gz").exists():
        return {
            't1': input_folder / f"{case_id}_t1.nii.gz",
            't1ce': input_folder / f"{case_id}_t1ce.nii.gz",
            't2': input_folder / f"{case_id}_t2.nii.gz",
            'flair': input_folder / f"{case_id}_flair.nii.gz"
        }
    
    # Check BraTS 2025 format
    if (input_folder / f"{case_id}-t1n.nii.gz").exists():
        return {
            't1': input_folder / f"{case_id}-t1n.nii.gz",
            't1ce': input_folder / f"{case_id}-t1c.nii.gz",
            't2': input_folder / f"{case_id}-t2w.nii.gz",
            'flair': input_folder / f"{case_id}-t2f.nii.gz"
        }
    
    raise ValueError(f"Could not find MRI files in {input_folder}")


def get_voxel_dimensions(header):
    """Get voxel dimensions from NIfTI header."""
    dims = header.get_zooms()[:3]
    return {
        'dimensions_mm': list(dims),
        'volume_mm3': float(np.prod(dims)),
        'volume_cm3': float(np.prod(dims) / 1000)
    }


def get_acquisition_details(header):
    """
    Extract acquisition details from NIfTI header.
    
    Returns slice thickness, voxel dimensions, and any available
    acquisition parameters from the header.
    """
    dims = header.get_zooms()[:3]
    shape = header.get_data_shape()[:3]
    
    # Slice thickness is typically the largest dimension (axial acquisition)
    # or explicitly the z-dimension spacing
    slice_thickness_mm = float(dims[2])
    in_plane_resolution = (float(dims[0]), float(dims[1]))
    
    # Get matrix size
    matrix_size = (int(shape[0]), int(shape[1]), int(shape[2]))
    
    # Check for additional info in header extensions or description
    try:
        descrip_raw = header.get('descrip', b'')
        if isinstance(descrip_raw, bytes):
            descrip = descrip_raw.decode('utf-8', errors='ignore').strip()
        elif isinstance(descrip_raw, np.ndarray):
            descrip = descrip_raw.tobytes().decode('utf-8', errors='ignore').strip()
        else:
            descrip = str(descrip_raw).strip()
    except Exception:
        descrip = None
    
    return {
        'slice_thickness_mm': slice_thickness_mm,
        'in_plane_resolution_mm': in_plane_resolution,
        'voxel_size_mm': [float(d) for d in dims],
        'matrix_size': matrix_size,
        'num_slices': int(shape[2]),
        'description': descrip if descrip else None
    }


def get_tumor_masks(seg_data):
    """Get masks for different tumor regions from segmentation."""
    seg_data = np.round(seg_data).astype(np.int32)
    
    return {
        'background': seg_data == 0,
        'ncr': seg_data == 1,  # Necrotic core
        'ed': seg_data == 2,   # Edema
        'et': (seg_data == 3) | (seg_data == 4),  # Enhancing tumor (BraTS 2021/2025)
        'tc': (seg_data == 1) | (seg_data == 3) | (seg_data == 4),  # Tumor core
        'wt': seg_data > 0  # Whole tumor
    }


def calculate_volume(mask, voxel_volume_cm3):
    """Calculate volume in cm³ from a binary mask."""
    return float(mask.sum() * voxel_volume_cm3)


def get_centroid(mask):
    """Calculate the centroid of a binary mask."""
    if mask.sum() == 0:
        return None
    
    coords = np.array(np.where(mask)).T
    centroid = coords.mean(axis=0)
    return {
        'x': float(centroid[0]),
        'y': float(centroid[1]),
        'z': float(centroid[2])
    }


def get_bounding_box(mask):
    """Get bounding box of a binary mask."""
    if mask.sum() == 0:
        return None
    
    coords = np.where(mask)
    return {
        'min_x': int(coords[0].min()),
        'max_x': int(coords[0].max()),
        'min_y': int(coords[1].min()),
        'max_y': int(coords[1].max()),
        'min_z': int(coords[2].min()),
        'max_z': int(coords[2].max()),
        'size_x': int(coords[0].max() - coords[0].min() + 1),
        'size_y': int(coords[1].max() - coords[1].min() + 1),
        'size_z': int(coords[2].max() - coords[2].min() + 1)
    }


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_results(results, output_path):
    """Save results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"Results saved to: {output_path}")


def load_results(json_path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)
