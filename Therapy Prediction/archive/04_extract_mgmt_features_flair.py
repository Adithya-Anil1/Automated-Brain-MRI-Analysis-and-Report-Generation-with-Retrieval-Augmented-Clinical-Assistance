"""
04_extract_mgmt_features_flair.py
Level 4: Advanced Radiomic Feature Extraction from FLAIR Images

This script extracts comprehensive radiomic features from FLAIR MRI scans
using all available feature classes in pyradiomics (or pure Python fallback).

Input:
    - Level4_MGMT_Dataset.csv (Patient_ID, Baseline_Scan_Path, MGMT_Label)

Output:
    - Level4_Radiomic_Features_FLAIR.csv (Patient_ID, MGMT_Label, + all radiomic features)

Features Extracted (if pyradiomics available):
    - Shape: 3D morphology
    - FirstOrder: Intensity statistics
    - GLCM: Gray Level Co-occurrence Matrix
    - GLRLM: Gray Level Run Length Matrix
    - GLSZM: Gray Level Size Zone Matrix
    - NGTDM: Neighbouring Gray Tone Difference Matrix
    - GLDM: Gray Level Dependence Matrix

If pyradiomics unavailable, uses pure Python implementation with extended texture features.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from scipy.stats import skew, kurtosis, entropy

warnings.filterwarnings('ignore')


# Configuration
SCRIPT_DIR = Path(__file__).parent
INPUT_CSV = SCRIPT_DIR / "Level4_MGMT_Dataset.csv"
OUTPUT_CSV = SCRIPT_DIR / "Level4_Radiomic_Features_FLAIR.csv"


# Check if pyradiomics is available
PYRADIOMICS_AVAILABLE = False
try:
    from radiomics import featureextractor
    import logging
    logging.getLogger('radiomics').setLevel(logging.ERROR)
    PYRADIOMICS_AVAILABLE = True
    print("✓ pyradiomics detected - using full feature extraction")
except ImportError:
    print("! pyradiomics not available - using pure Python implementation")


def find_flair_file(scan_folder: Path) -> Optional[Path]:
    """
    Find the FLAIR image file.
    
    Search priority:
    1. FLAIR registered (in same space as segmentation)
    2. FLAIR in main folder
    """
    # First try to find the registered FLAIR that matches the segmentation
    registered_folder = scan_folder / "HD-GLIO-AUTO-segmentation" / "registered"
    if registered_folder.exists():
        files = list(registered_folder.glob("*.nii.gz")) + list(registered_folder.glob("*.nii"))
        # Look for FLAIR_r2s_bet_reg.nii.gz (the registered version)
        for f in files:
            fname_lower = f.name.lower()
            if 'flair' in fname_lower and 'reg' in fname_lower:
                return f
        # Fallback to any FLAIR in registered folder
        for f in files:
            fname_lower = f.name.lower()
            if 'flair' in fname_lower:
                return f
    
    # Try DeepBraTumIA
    deepbratumia_folder = scan_folder / "DeepBraTumIA-segmentation" / "registered"
    if deepbratumia_folder.exists():
        files = list(deepbratumia_folder.glob("*.nii.gz")) + list(deepbratumia_folder.glob("*.nii"))
        for f in files:
            fname_lower = f.name.lower()
            if 'flair' in fname_lower:
                return f
    
    # Fallback to main folder
    files = list(scan_folder.glob("*.nii.gz")) + list(scan_folder.glob("*.nii"))
    for f in files:
        fname_lower = f.name.lower()
        if 'flair' in fname_lower:
            return f
    
    return None


def find_mask_file(scan_folder: Path) -> Optional[Path]:
    """
    Find the segmentation mask file.
    """
    search_locations = [
        scan_folder / "HD-GLIO-AUTO-segmentation" / "registered",
        scan_folder / "HD-GLIO-AUTO-segmentation" / "native",
        scan_folder / "DeepBraTumIA-segmentation" / "registered",
        scan_folder / "DeepBraTumIA-segmentation" / "native",
        scan_folder,
    ]
    
    for location in search_locations:
        if not location.exists():
            continue
            
        files = list(location.glob("*.nii.gz")) + list(location.glob("*.nii"))
        
        for f in files:
            fname_lower = f.name.lower()
            if "seg" in fname_lower:
                return f
    
    return None


# ============================================================
# PYRADIOMICS EXTRACTION (if available)
# ============================================================

def create_full_feature_extractor():
    """
    Create a pyradiomics feature extractor with ALL features enabled.
    """
    extractor = featureextractor.RadiomicsFeatureExtractor()
    
    # Enable ALL feature classes
    extractor.enableAllFeatures()
    
    # Settings for SPEED and robustness
    # Force resampling to 1mm isotropic - reduces voxel count significantly
    extractor.settings['resampledPixelSpacing'] = [1, 1, 1]
    extractor.settings['interpolator'] = 'sitkBSpline'
    extractor.settings['resampledPixelSpacing'] = [1, 1, 1]
    
    # Use binCount instead of binWidth - guarantees exactly 32 bins (faster)
    extractor.settings['binCount'] = 32
    
    # Other speed optimizations
    extractor.settings['preCrop'] = True  # Crop to mask bounding box first
    extractor.settings['force2D'] = False
    extractor.settings['label'] = 1
    
    return extractor


def extract_with_pyradiomics(extractor, image_path: Path, mask_path: Path) -> Dict[str, Any]:
    """
    Extract features using pyradiomics with all feature classes.
    """
    import SimpleITK as sitk
    
    # Load images
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    
    # Get unique labels in mask (excluding 0)
    mask_array = sitk.GetArrayFromImage(mask)
    unique_labels = [int(l) for l in set(mask_array.flatten()) if l > 0]
    
    if not unique_labels:
        raise ValueError("Mask is empty (no tumor voxels)")
    
    # Use the first non-zero label
    extractor.settings['label'] = unique_labels[0]
    
    # Extract features
    result = extractor.execute(image, mask)
    
    # Filter to keep only feature values (exclude diagnostics)
    features = {}
    for key, value in result.items():
        if not key.startswith('diagnostics_'):
            if hasattr(value, 'item'):
                features[key] = value.item()
            else:
                features[key] = value
    
    return features


# ============================================================
# PURE PYTHON EXTRACTION (fallback)
# ============================================================

def extract_shape_features(mask: np.ndarray, voxel_spacing: Tuple[float, float, float]) -> Dict[str, float]:
    """Extract 3D shape features."""
    features = {}
    tumor_mask = mask > 0
    voxel_volume = np.prod(voxel_spacing)
    
    num_voxels = np.sum(tumor_mask)
    volume_mm3 = num_voxels * voxel_volume
    features['shape_Volume_mm3'] = volume_mm3
    features['shape_VoxelCount'] = int(num_voxels)
    
    eroded = ndimage.binary_erosion(tumor_mask)
    surface_voxels = tumor_mask & ~eroded
    surface_area_approx = np.sum(surface_voxels) * (voxel_spacing[0] * voxel_spacing[1])
    features['shape_SurfaceArea_mm2'] = surface_area_approx
    
    if volume_mm3 > 0:
        features['shape_SurfaceVolumeRatio'] = surface_area_approx / volume_mm3
    else:
        features['shape_SurfaceVolumeRatio'] = 0
    
    if surface_area_approx > 0:
        sphericity = (np.pi ** (1/3) * (6 * volume_mm3) ** (2/3)) / surface_area_approx
        features['shape_Sphericity'] = min(sphericity, 1.0)
    else:
        features['shape_Sphericity'] = 0
    
    if num_voxels > 0:
        coords = np.where(tumor_mask)
        bbox_dims = []
        for i in range(3):
            dim_range = (coords[i].max() - coords[i].min() + 1) * voxel_spacing[i]
            bbox_dims.append(dim_range)
        
        bbox_dims = sorted(bbox_dims, reverse=True)
        features['shape_BoundingBoxDim_Major'] = bbox_dims[0]
        features['shape_BoundingBoxDim_Middle'] = bbox_dims[1]
        features['shape_BoundingBoxDim_Minor'] = bbox_dims[2]
        
        if bbox_dims[0] > 0:
            features['shape_Elongation'] = bbox_dims[1] / bbox_dims[0]
            features['shape_Flatness'] = bbox_dims[2] / bbox_dims[0]
        else:
            features['shape_Elongation'] = 0
            features['shape_Flatness'] = 0
    
    return features


def extract_firstorder_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Extract first-order intensity statistics."""
    features = {}
    tumor_mask = mask > 0
    intensities = image[tumor_mask]
    
    if len(intensities) == 0:
        return {f'firstorder_{k}': 0 for k in ['Mean', 'Median', 'Std', 'Variance', 'Min', 'Max', 
                'Range', 'Skewness', 'Kurtosis', 'Energy', 'Entropy', 'P10', 'P25', 'P75', 'P90', 
                'IQR', 'MAD', 'RobustMAD', 'Uniformity', 'RootMeanSquared', 'MeanAbsoluteDeviation']}
    
    features['firstorder_Mean'] = np.mean(intensities)
    features['firstorder_Median'] = np.median(intensities)
    features['firstorder_Std'] = np.std(intensities)
    features['firstorder_Variance'] = np.var(intensities)
    features['firstorder_Min'] = np.min(intensities)
    features['firstorder_Max'] = np.max(intensities)
    features['firstorder_Range'] = features['firstorder_Max'] - features['firstorder_Min']
    features['firstorder_Skewness'] = skew(intensities)
    features['firstorder_Kurtosis'] = kurtosis(intensities)
    features['firstorder_Energy'] = np.sum(intensities ** 2)
    features['firstorder_RootMeanSquared'] = np.sqrt(np.mean(intensities ** 2))
    
    hist, _ = np.histogram(intensities, bins=256)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    features['firstorder_Entropy'] = entropy(hist)
    features['firstorder_Uniformity'] = np.sum(hist ** 2)
    
    features['firstorder_P10'] = np.percentile(intensities, 10)
    features['firstorder_P25'] = np.percentile(intensities, 25)
    features['firstorder_P75'] = np.percentile(intensities, 75)
    features['firstorder_P90'] = np.percentile(intensities, 90)
    features['firstorder_IQR'] = features['firstorder_P75'] - features['firstorder_P25']
    
    features['firstorder_MAD'] = np.mean(np.abs(intensities - features['firstorder_Mean']))
    features['firstorder_MeanAbsoluteDeviation'] = features['firstorder_MAD']
    
    p10, p90 = features['firstorder_P10'], features['firstorder_P90']
    robust_intensities = intensities[(intensities >= p10) & (intensities <= p90)]
    if len(robust_intensities) > 0:
        robust_mean = np.mean(robust_intensities)
        features['firstorder_RobustMAD'] = np.mean(np.abs(robust_intensities - robust_mean))
    else:
        features['firstorder_RobustMAD'] = 0
    
    return features


def compute_glcm_features(image: np.ndarray, mask: np.ndarray, num_levels: int = 32) -> Dict[str, float]:
    """Compute GLCM (Gray Level Co-occurrence Matrix) features."""
    features = {}
    tumor_mask = mask > 0
    
    if np.sum(tumor_mask) < 10:
        return {f'glcm_{k}': 0 for k in ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 
                'Correlation', 'ASM', 'Entropy', 'MaxProbability', 'ClusterShade', 'ClusterProminence']}
    
    # Quantize intensities
    tumor_intensities = image[tumor_mask]
    min_val, max_val = tumor_intensities.min(), tumor_intensities.max()
    if max_val > min_val:
        quantized = np.floor((image - min_val) / (max_val - min_val + 1e-10) * (num_levels - 1)).astype(int)
        quantized = np.clip(quantized, 0, num_levels - 1)
    else:
        quantized = np.zeros_like(image, dtype=int)
    
    # Build GLCM for offset (1,0,0)
    glcm = np.zeros((num_levels, num_levels), dtype=float)
    coords = np.where(tumor_mask)
    
    for z, y, x in zip(coords[0], coords[1], coords[2]):
        for dz, dy, dx in [(0, 0, 1), (0, 1, 0), (1, 0, 0)]:
            nz, ny, nx = z + dz, y + dy, x + dx
            if 0 <= nz < mask.shape[0] and 0 <= ny < mask.shape[1] and 0 <= nx < mask.shape[2]:
                if tumor_mask[nz, ny, nx]:
                    i, j = quantized[z, y, x], quantized[nz, ny, nx]
                    glcm[i, j] += 1
                    glcm[j, i] += 1
    
    # Normalize
    if glcm.sum() > 0:
        glcm = glcm / glcm.sum()
    
    # Compute features
    i_idx, j_idx = np.meshgrid(range(num_levels), range(num_levels), indexing='ij')
    
    features['glcm_Contrast'] = np.sum(glcm * (i_idx - j_idx) ** 2)
    features['glcm_Dissimilarity'] = np.sum(glcm * np.abs(i_idx - j_idx))
    features['glcm_Homogeneity'] = np.sum(glcm / (1 + (i_idx - j_idx) ** 2))
    features['glcm_Energy'] = np.sum(glcm ** 2)
    features['glcm_ASM'] = features['glcm_Energy']  # Angular Second Moment
    
    glcm_nonzero = glcm[glcm > 0]
    features['glcm_Entropy'] = -np.sum(glcm_nonzero * np.log2(glcm_nonzero + 1e-10))
    features['glcm_MaxProbability'] = np.max(glcm)
    
    # Correlation
    mu_i = np.sum(i_idx * glcm)
    mu_j = np.sum(j_idx * glcm)
    sigma_i = np.sqrt(np.sum((i_idx - mu_i) ** 2 * glcm))
    sigma_j = np.sqrt(np.sum((j_idx - mu_j) ** 2 * glcm))
    
    if sigma_i > 0 and sigma_j > 0:
        features['glcm_Correlation'] = np.sum((i_idx - mu_i) * (j_idx - mu_j) * glcm) / (sigma_i * sigma_j)
    else:
        features['glcm_Correlation'] = 0
    
    # Cluster features
    features['glcm_ClusterShade'] = np.sum(((i_idx + j_idx - mu_i - mu_j) ** 3) * glcm)
    features['glcm_ClusterProminence'] = np.sum(((i_idx + j_idx - mu_i - mu_j) ** 4) * glcm)
    
    return features


def compute_glrlm_features(image: np.ndarray, mask: np.ndarray, num_levels: int = 32) -> Dict[str, float]:
    """Compute GLRLM (Gray Level Run Length Matrix) features - simplified."""
    features = {}
    tumor_mask = mask > 0
    
    if np.sum(tumor_mask) < 10:
        return {f'glrlm_{k}': 0 for k in ['ShortRunEmphasis', 'LongRunEmphasis', 'GrayLevelNonUniformity',
                'RunLengthNonUniformity', 'RunPercentage', 'LowGrayLevelRunEmphasis', 
                'HighGrayLevelRunEmphasis', 'ShortRunLowGrayLevelEmphasis', 'ShortRunHighGrayLevelEmphasis']}
    
    # Quantize intensities
    tumor_intensities = image[tumor_mask]
    min_val, max_val = tumor_intensities.min(), tumor_intensities.max()
    if max_val > min_val:
        quantized = np.floor((image - min_val) / (max_val - min_val + 1e-10) * (num_levels - 1)).astype(int)
        quantized = np.clip(quantized, 0, num_levels - 1)
    else:
        quantized = np.zeros_like(image, dtype=int)
    
    # Maximum possible run length
    max_run = max(mask.shape)
    
    # Build GLRLM (simplified - horizontal runs only for speed)
    glrlm = np.zeros((num_levels, max_run), dtype=float)
    
    for z in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            run_length = 0
            current_level = -1
            for x in range(mask.shape[2]):
                if tumor_mask[z, y, x]:
                    level = quantized[z, y, x]
                    if level == current_level:
                        run_length += 1
                    else:
                        if current_level >= 0 and run_length > 0:
                            glrlm[current_level, min(run_length - 1, max_run - 1)] += 1
                        current_level = level
                        run_length = 1
                else:
                    if current_level >= 0 and run_length > 0:
                        glrlm[current_level, min(run_length - 1, max_run - 1)] += 1
                    current_level = -1
                    run_length = 0
            if current_level >= 0 and run_length > 0:
                glrlm[current_level, min(run_length - 1, max_run - 1)] += 1
    
    total_runs = glrlm.sum()
    if total_runs == 0:
        return {f'glrlm_{k}': 0 for k in ['ShortRunEmphasis', 'LongRunEmphasis', 'GrayLevelNonUniformity',
                'RunLengthNonUniformity', 'RunPercentage', 'LowGrayLevelRunEmphasis', 
                'HighGrayLevelRunEmphasis', 'ShortRunLowGrayLevelEmphasis', 'ShortRunHighGrayLevelEmphasis']}
    
    i_idx, j_idx = np.meshgrid(range(num_levels), range(max_run), indexing='ij')
    j_idx = j_idx + 1  # Run lengths start at 1
    
    # Run Length features
    features['glrlm_ShortRunEmphasis'] = np.sum(glrlm / (j_idx ** 2)) / total_runs
    features['glrlm_LongRunEmphasis'] = np.sum(glrlm * (j_idx ** 2)) / total_runs
    features['glrlm_GrayLevelNonUniformity'] = np.sum(np.sum(glrlm, axis=1) ** 2) / total_runs
    features['glrlm_RunLengthNonUniformity'] = np.sum(np.sum(glrlm, axis=0) ** 2) / total_runs
    features['glrlm_RunPercentage'] = total_runs / np.sum(tumor_mask)
    
    i_weights = (i_idx + 1)
    features['glrlm_LowGrayLevelRunEmphasis'] = np.sum(glrlm / (i_weights ** 2)) / total_runs
    features['glrlm_HighGrayLevelRunEmphasis'] = np.sum(glrlm * (i_weights ** 2)) / total_runs
    features['glrlm_ShortRunLowGrayLevelEmphasis'] = np.sum(glrlm / ((j_idx ** 2) * (i_weights ** 2))) / total_runs
    features['glrlm_ShortRunHighGrayLevelEmphasis'] = np.sum(glrlm * (i_weights ** 2) / (j_idx ** 2)) / total_runs
    
    return features


def compute_glszm_features(image: np.ndarray, mask: np.ndarray, num_levels: int = 32) -> Dict[str, float]:
    """Compute GLSZM (Gray Level Size Zone Matrix) features - simplified."""
    features = {}
    tumor_mask = mask > 0
    
    if np.sum(tumor_mask) < 10:
        return {f'glszm_{k}': 0 for k in ['SmallAreaEmphasis', 'LargeAreaEmphasis', 'GrayLevelNonUniformity',
                'SizeZoneNonUniformity', 'ZonePercentage', 'LowGrayLevelZoneEmphasis', 
                'HighGrayLevelZoneEmphasis', 'GrayLevelVariance', 'ZoneVariance']}
    
    # Quantize intensities
    tumor_intensities = image[tumor_mask]
    min_val, max_val = tumor_intensities.min(), tumor_intensities.max()
    if max_val > min_val:
        quantized = np.floor((image - min_val) / (max_val - min_val + 1e-10) * (num_levels - 1)).astype(int)
        quantized = np.clip(quantized, 0, num_levels - 1)
    else:
        quantized = np.zeros_like(image, dtype=int)
    
    # Use connected components to find zones
    from scipy.ndimage import label as ndimage_label
    
    zone_sizes = {i: [] for i in range(num_levels)}
    
    for level in range(num_levels):
        level_mask = (quantized == level) & tumor_mask
        if np.sum(level_mask) > 0:
            labeled, num_zones = ndimage_label(level_mask)
            for zone_id in range(1, num_zones + 1):
                zone_size = np.sum(labeled == zone_id)
                zone_sizes[level].append(zone_size)
    
    # Build GLSZM
    max_size = max(max(sizes) if sizes else 1 for sizes in zone_sizes.values())
    max_size = min(max_size, 1000)  # Cap for memory
    
    glszm = np.zeros((num_levels, max_size), dtype=float)
    for level, sizes in zone_sizes.items():
        for size in sizes:
            if size <= max_size:
                glszm[level, size - 1] += 1
    
    total_zones = glszm.sum()
    if total_zones == 0:
        return {f'glszm_{k}': 0 for k in ['SmallAreaEmphasis', 'LargeAreaEmphasis', 'GrayLevelNonUniformity',
                'SizeZoneNonUniformity', 'ZonePercentage', 'LowGrayLevelZoneEmphasis', 
                'HighGrayLevelZoneEmphasis', 'GrayLevelVariance', 'ZoneVariance']}
    
    i_idx, j_idx = np.meshgrid(range(num_levels), range(max_size), indexing='ij')
    j_idx = j_idx + 1
    
    features['glszm_SmallAreaEmphasis'] = np.sum(glszm / (j_idx ** 2)) / total_zones
    features['glszm_LargeAreaEmphasis'] = np.sum(glszm * (j_idx ** 2)) / total_zones
    features['glszm_GrayLevelNonUniformity'] = np.sum(np.sum(glszm, axis=1) ** 2) / total_zones
    features['glszm_SizeZoneNonUniformity'] = np.sum(np.sum(glszm, axis=0) ** 2) / total_zones
    features['glszm_ZonePercentage'] = total_zones / np.sum(tumor_mask)
    
    i_weights = (i_idx + 1)
    features['glszm_LowGrayLevelZoneEmphasis'] = np.sum(glszm / (i_weights ** 2)) / total_zones
    features['glszm_HighGrayLevelZoneEmphasis'] = np.sum(glszm * (i_weights ** 2)) / total_zones
    
    # Variance features
    p = glszm / total_zones
    mu_i = np.sum(i_idx * p)
    mu_j = np.sum(j_idx * p)
    features['glszm_GrayLevelVariance'] = np.sum(((i_idx - mu_i) ** 2) * p)
    features['glszm_ZoneVariance'] = np.sum(((j_idx - mu_j) ** 2) * p)
    
    return features


def compute_ngtdm_features(image: np.ndarray, mask: np.ndarray, num_levels: int = 32) -> Dict[str, float]:
    """Compute NGTDM (Neighbouring Gray Tone Difference Matrix) features."""
    features = {}
    tumor_mask = mask > 0
    
    if np.sum(tumor_mask) < 27:  # Need 3x3x3 neighborhood
        return {f'ngtdm_{k}': 0 for k in ['Coarseness', 'Contrast', 'Busyness', 'Complexity', 'Strength']}
    
    # Quantize intensities
    tumor_intensities = image[tumor_mask]
    min_val, max_val = tumor_intensities.min(), tumor_intensities.max()
    if max_val > min_val:
        quantized = np.floor((image - min_val) / (max_val - min_val + 1e-10) * (num_levels - 1)).astype(int)
        quantized = np.clip(quantized, 0, num_levels - 1)
    else:
        quantized = np.zeros_like(image, dtype=int)
    
    # Compute average neighborhood for each voxel
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = 0
    kernel = kernel / 26  # Normalize
    
    avg_neighborhood = ndimage.convolve(quantized.astype(float), kernel, mode='constant')
    
    # NGTDM: sum of |i - avg| for each gray level
    s = np.zeros(num_levels)  # Sum of differences
    n = np.zeros(num_levels)  # Count
    
    coords = np.where(tumor_mask)
    for z, y, x in zip(coords[0], coords[1], coords[2]):
        # Check if voxel has valid neighborhood
        if 1 <= z < mask.shape[0]-1 and 1 <= y < mask.shape[1]-1 and 1 <= x < mask.shape[2]-1:
            i = quantized[z, y, x]
            diff = abs(i - avg_neighborhood[z, y, x])
            s[i] += diff
            n[i] += 1
    
    N = np.sum(n)
    if N == 0:
        return {f'ngtdm_{k}': 0 for k in ['Coarseness', 'Contrast', 'Busyness', 'Complexity', 'Strength']}
    
    p = n / N
    
    # Coarseness
    eps = 1e-10
    coarseness_sum = np.sum(p * s)
    features['ngtdm_Coarseness'] = 1 / (coarseness_sum + eps)
    
    # Contrast
    n_g = np.sum(p > 0)
    if n_g > 1:
        contrast = 0
        for i in range(num_levels):
            for j in range(num_levels):
                if p[i] > 0 and p[j] > 0:
                    contrast += p[i] * p[j] * (i - j) ** 2
        contrast *= np.sum(s) / (n_g * (n_g - 1) * N)
        features['ngtdm_Contrast'] = contrast
    else:
        features['ngtdm_Contrast'] = 0
    
    # Busyness
    busyness_num = np.sum(p * s)
    busyness_den = 0
    for i in range(num_levels):
        for j in range(num_levels):
            if p[i] > 0 and p[j] > 0:
                busyness_den += abs(i * p[i] - j * p[j])
    features['ngtdm_Busyness'] = busyness_num / (busyness_den + eps)
    
    # Complexity
    complexity = 0
    for i in range(num_levels):
        for j in range(num_levels):
            if p[i] > 0 and p[j] > 0 and n[i] > 0 and n[j] > 0:
                complexity += abs(i - j) * (p[i] * s[i] + p[j] * s[j]) / (n[i] + n[j])
    features['ngtdm_Complexity'] = complexity / N
    
    # Strength
    strength_num = 0
    for i in range(num_levels):
        for j in range(num_levels):
            if p[i] > 0 and p[j] > 0:
                strength_num += (p[i] + p[j]) * (i - j) ** 2
    features['ngtdm_Strength'] = strength_num / (np.sum(s) + eps)
    
    return features


def extract_gradient_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Extract gradient/edge-based texture features."""
    features = {}
    tumor_mask = mask > 0
    
    if np.sum(tumor_mask) == 0:
        return {f'gradient_{k}': 0 for k in ['Mean', 'Std', 'Max', 'Min', 'Range', 'Energy']}
    
    masked_image = np.where(tumor_mask, image, 0).astype(float)
    
    gradient_z = ndimage.sobel(masked_image, axis=0)
    gradient_y = ndimage.sobel(masked_image, axis=1)
    gradient_x = ndimage.sobel(masked_image, axis=2)
    gradient_magnitude = np.sqrt(gradient_z**2 + gradient_y**2 + gradient_x**2)
    
    tumor_gradients = gradient_magnitude[tumor_mask]
    
    features['gradient_Mean'] = np.mean(tumor_gradients)
    features['gradient_Std'] = np.std(tumor_gradients)
    features['gradient_Max'] = np.max(tumor_gradients)
    features['gradient_Min'] = np.min(tumor_gradients)
    features['gradient_Range'] = features['gradient_Max'] - features['gradient_Min']
    features['gradient_Energy'] = np.sum(tumor_gradients ** 2)
    
    return features


def extract_all_features_pure_python(image_path: Path, mask_path: Path) -> Dict[str, Any]:
    """Extract all features using pure Python implementation."""
    img_nii = nib.load(str(image_path))
    mask_nii = nib.load(str(mask_path))
    
    image = img_nii.get_fdata().astype(float)
    mask = mask_nii.get_fdata().astype(int)
    voxel_spacing = img_nii.header.get_zooms()[:3]
    
    # Handle dimension mismatch
    if image.shape != mask.shape:
        from scipy.ndimage import zoom
        zoom_factors = [image.shape[i] / mask.shape[i] for i in range(3)]
        mask = zoom(mask, zoom_factors, order=0).astype(int)
    
    if mask.max() == 0:
        raise ValueError("Mask is empty (no tumor voxels)")
    
    features = {}
    features.update(extract_shape_features(mask, voxel_spacing))
    features.update(extract_firstorder_features(image, mask))
    features.update(compute_glcm_features(image, mask))
    features.update(compute_glrlm_features(image, mask))
    features.update(compute_glszm_features(image, mask))
    features.update(compute_ngtdm_features(image, mask))
    features.update(extract_gradient_features(image, mask))
    
    return features


def main():
    """Main pipeline to extract advanced radiomic features from FLAIR images."""
    print("=" * 70)
    print("Level 4: Advanced Radiomic Feature Extraction (FLAIR)")
    print("=" * 70)
    
    # Load dataset
    print(f"\nLoading dataset from: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        print("ERROR: Input CSV not found!")
        print("Run 03_create_mgmt_dataset.py first.")
        sys.exit(1)
    
    df = pd.read_csv(INPUT_CSV)
    print(f"  Total patients to process: {len(df)}")
    
    # Initialize extractor if pyradiomics available
    extractor = None
    if PYRADIOMICS_AVAILABLE:
        print("\nInitializing pyradiomics with ALL features enabled...")
        extractor = create_full_feature_extractor()
    else:
        print("\nUsing pure Python implementation with extended features...")
    
    # Process patients
    print("\n" + "-" * 70)
    print("Processing patients (FLAIR images)...")
    print("-" * 70)
    
    results = []
    successful = 0
    failed = 0
    failed_patients = []
    
    for idx, row in df.iterrows():
        patient_id = row['Patient_ID']
        scan_path = Path(row['Baseline_Scan_Path'])
        mgmt_label = row['MGMT_Label']
        
        print(f"\n[{idx + 1}/{len(df)}] Processing: {patient_id}")
        
        try:
            # Find FLAIR and mask
            flair_path = find_flair_file(scan_path)
            mask_path = find_mask_file(scan_path)
            
            if flair_path is None:
                raise FileNotFoundError(f"No FLAIR image found in {scan_path}")
            
            if mask_path is None:
                raise FileNotFoundError(f"No segmentation mask found in {scan_path}")
            
            print(f"  FLAIR: {flair_path.name}")
            print(f"  Mask:  {mask_path.name}")
            
            # Extract features
            if PYRADIOMICS_AVAILABLE and extractor is not None:
                features = extract_with_pyradiomics(extractor, flair_path, mask_path)
            else:
                features = extract_all_features_pure_python(flair_path, mask_path)
            
            features['Patient_ID'] = patient_id
            features['MGMT_Label'] = mgmt_label
            
            results.append(features)
            successful += 1
            print(f"  ✓ Extracted {len(features) - 2} features")
            
        except Exception as e:
            failed += 1
            failed_patients.append(patient_id)
            print(f"  ✗ FAILED: {str(e)}")
            continue
    
    # Create output
    print("\n" + "-" * 70)
    print("Creating output dataset...")
    print("-" * 70)
    
    if not results:
        print("ERROR: No patients were successfully processed!")
        sys.exit(1)
    
    df_features = pd.DataFrame(results)
    
    cols = ['Patient_ID', 'MGMT_Label'] + [c for c in df_features.columns 
                                            if c not in ['Patient_ID', 'MGMT_Label']]
    df_features = df_features[cols]
    
    df_features.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFeatures saved to: {OUTPUT_CSV}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successfully processed: {successful}/{len(df)} patients")
    print(f"Failed: {failed} patients")
    
    feature_cols = [c for c in df_features.columns if c not in ['Patient_ID', 'MGMT_Label']]
    print(f"\nTotal features extracted: {len(feature_cols)}")
    
    # Group by type
    shape_f = [c for c in feature_cols if 'shape' in c.lower()]
    firstorder_f = [c for c in feature_cols if 'firstorder' in c.lower()]
    glcm_f = [c for c in feature_cols if 'glcm' in c.lower()]
    glrlm_f = [c for c in feature_cols if 'glrlm' in c.lower()]
    glszm_f = [c for c in feature_cols if 'glszm' in c.lower()]
    ngtdm_f = [c for c in feature_cols if 'ngtdm' in c.lower()]
    gradient_f = [c for c in feature_cols if 'gradient' in c.lower()]
    
    print(f"  - Shape: {len(shape_f)}")
    print(f"  - FirstOrder: {len(firstorder_f)}")
    print(f"  - GLCM: {len(glcm_f)}")
    print(f"  - GLRLM: {len(glrlm_f)}")
    print(f"  - GLSZM: {len(glszm_f)}")
    print(f"  - NGTDM: {len(ngtdm_f)}")
    print(f"  - Gradient: {len(gradient_f)}")
    
    print(f"\nMGMT distribution: Methylated={sum(df_features['MGMT_Label'])}, "
          f"Unmethylated={len(df_features) - sum(df_features['MGMT_Label'])}")
    
    return df_features


if __name__ == "__main__":
    main()
