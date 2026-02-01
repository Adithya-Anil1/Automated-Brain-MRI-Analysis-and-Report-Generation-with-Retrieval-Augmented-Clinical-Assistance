"""
04_extract_mgmt_features.py
Level 4: Radiomic Feature Extraction for MGMT Prediction (Pure Python Version)

This script extracts radiomic features from baseline MRI scans for MGMT
methylation prediction using numpy/scipy/nibabel (no pyradiomics required).

Input:
    - Level4_MGMT_Dataset.csv (Patient_ID, Baseline_Scan_Path, MGMT_Label)

Output:
    - Level4_Radiomic_Features.csv (Patient_ID, MGMT_Label, + radiomic features)

Features Extracted:
    - Shape: 3D tumor morphology (volume, surface area, sphericity, etc.)
    - FirstOrder: Intensity statistics (mean, std, skewness, kurtosis, etc.)
    - Texture (GLCM-like): Simplified texture heterogeneity metrics
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
OUTPUT_CSV = SCRIPT_DIR / "Level4_Radiomic_Features.csv"


def find_image_file(scan_folder: Path) -> Optional[Path]:
    """
    Find the T1CE (contrast-enhanced T1) or T1 image file.
    
    Search priority:
    1. CT1 registered (in same space as segmentation)
    2. CT1 in main folder
    3. T1 registered
    4. T1 in main folder
    
    We prioritize registered images as they match the segmentation space.
    """
    # First try to find the registered CT1 that matches the segmentation
    registered_folder = scan_folder / "HD-GLIO-AUTO-segmentation" / "registered"
    if registered_folder.exists():
        files = list(registered_folder.glob("*.nii.gz")) + list(registered_folder.glob("*.nii"))
        # Look for CT1_r2s_bet_reg.nii.gz (the registered version)
        for f in files:
            fname_lower = f.name.lower()
            if 'ct1' in fname_lower and 'reg' in fname_lower:
                return f
        # Fallback to any CT1 in registered folder
        for f in files:
            fname_lower = f.name.lower()
            if 'ct1' in fname_lower:
                return f
        # Try T1 registered
        for f in files:
            fname_lower = f.name.lower()
            if 't1' in fname_lower and 'reg' in fname_lower:
                return f
    
    # Try DeepBraTumIA
    deepbratumia_folder = scan_folder / "DeepBraTumIA-segmentation" / "registered"
    if deepbratumia_folder.exists():
        files = list(deepbratumia_folder.glob("*.nii.gz")) + list(deepbratumia_folder.glob("*.nii"))
        for f in files:
            fname_lower = f.name.lower()
            if 'ct1' in fname_lower and 'reg' in fname_lower:
                return f
        for f in files:
            fname_lower = f.name.lower()
            if 'ct1' in fname_lower:
                return f
    
    # Fallback to main folder (native space)
    files = list(scan_folder.glob("*.nii.gz")) + list(scan_folder.glob("*.nii"))
    for f in files:
        fname_lower = f.name.lower()
        if 'ct1' in fname_lower or 't1ce' in fname_lower:
            return f
    for f in files:
        fname_lower = f.name.lower()
        if 't1' in fname_lower and 't2' not in fname_lower:
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


def extract_shape_features(mask: np.ndarray, voxel_spacing: Tuple[float, float, float]) -> Dict[str, float]:
    """
    Extract 3D shape features from the tumor mask.
    
    Features:
    - Volume (in mm³)
    - Surface Area (approximate)
    - Sphericity
    - Compactness
    - Elongation
    - Flatness
    """
    features = {}
    
    # Get tumor voxels (any non-zero label)
    tumor_mask = mask > 0
    
    # Voxel volume in mm³
    voxel_volume = np.prod(voxel_spacing)
    
    # Volume
    num_voxels = np.sum(tumor_mask)
    volume_mm3 = num_voxels * voxel_volume
    features['shape_Volume_mm3'] = volume_mm3
    features['shape_VoxelCount'] = int(num_voxels)
    
    # Surface area (approximate using gradient magnitude)
    # Count boundary voxels (voxels that have at least one non-tumor neighbor)
    eroded = ndimage.binary_erosion(tumor_mask)
    surface_voxels = tumor_mask & ~eroded
    surface_area_approx = np.sum(surface_voxels) * (voxel_spacing[0] * voxel_spacing[1])
    features['shape_SurfaceArea_mm2'] = surface_area_approx
    
    # Surface to Volume Ratio
    if volume_mm3 > 0:
        features['shape_SurfaceVolumeRatio'] = surface_area_approx / volume_mm3
    else:
        features['shape_SurfaceVolumeRatio'] = 0
    
    # Sphericity: how sphere-like is the tumor
    # Sphericity = (pi^(1/3) * (6*V)^(2/3)) / A
    if surface_area_approx > 0:
        sphericity = (np.pi ** (1/3) * (6 * volume_mm3) ** (2/3)) / surface_area_approx
        features['shape_Sphericity'] = min(sphericity, 1.0)  # Cap at 1.0
    else:
        features['shape_Sphericity'] = 0
    
    # Bounding box dimensions
    if num_voxels > 0:
        coords = np.where(tumor_mask)
        bbox_dims = []
        for i in range(3):
            dim_range = (coords[i].max() - coords[i].min() + 1) * voxel_spacing[i]
            bbox_dims.append(dim_range)
        
        bbox_dims = sorted(bbox_dims, reverse=True)  # [major, middle, minor]
        
        features['shape_BoundingBoxDim_Major'] = bbox_dims[0]
        features['shape_BoundingBoxDim_Middle'] = bbox_dims[1]
        features['shape_BoundingBoxDim_Minor'] = bbox_dims[2]
        
        # Elongation: ratio of middle to major axis
        if bbox_dims[0] > 0:
            features['shape_Elongation'] = bbox_dims[1] / bbox_dims[0]
        else:
            features['shape_Elongation'] = 0
        
        # Flatness: ratio of minor to major axis
        if bbox_dims[0] > 0:
            features['shape_Flatness'] = bbox_dims[2] / bbox_dims[0]
        else:
            features['shape_Flatness'] = 0
        
        # Centroid location (normalized to bounding box)
        centroid = [np.mean(coords[i]) for i in range(3)]
        features['shape_Centroid_Z'] = centroid[0] * voxel_spacing[0]
        features['shape_Centroid_Y'] = centroid[1] * voxel_spacing[1]
        features['shape_Centroid_X'] = centroid[2] * voxel_spacing[2]
    else:
        features['shape_BoundingBoxDim_Major'] = 0
        features['shape_BoundingBoxDim_Middle'] = 0
        features['shape_BoundingBoxDim_Minor'] = 0
        features['shape_Elongation'] = 0
        features['shape_Flatness'] = 0
        features['shape_Centroid_Z'] = 0
        features['shape_Centroid_Y'] = 0
        features['shape_Centroid_X'] = 0
    
    return features


def extract_firstorder_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Extract first-order (intensity) statistics from the tumor region.
    
    Features:
    - Mean, Median, Std, Min, Max
    - Skewness, Kurtosis
    - Energy, Entropy
    - Percentiles (10th, 25th, 75th, 90th)
    - Interquartile Range
    - Range
    - Mean Absolute Deviation
    - Robust Mean Absolute Deviation
    """
    features = {}
    
    # Get tumor voxel intensities
    tumor_mask = mask > 0
    intensities = image[tumor_mask]
    
    if len(intensities) == 0:
        # Return zeros for all features
        for key in ['Mean', 'Median', 'Std', 'Min', 'Max', 'Skewness', 'Kurtosis',
                    'Energy', 'Entropy', 'P10', 'P25', 'P75', 'P90', 'IQR', 'Range',
                    'MAD', 'RobustMAD', 'Variance', 'Uniformity']:
            features[f'firstorder_{key}'] = 0
        return features
    
    # Basic statistics
    features['firstorder_Mean'] = np.mean(intensities)
    features['firstorder_Median'] = np.median(intensities)
    features['firstorder_Std'] = np.std(intensities)
    features['firstorder_Variance'] = np.var(intensities)
    features['firstorder_Min'] = np.min(intensities)
    features['firstorder_Max'] = np.max(intensities)
    features['firstorder_Range'] = features['firstorder_Max'] - features['firstorder_Min']
    
    # Higher order statistics
    features['firstorder_Skewness'] = skew(intensities)
    features['firstorder_Kurtosis'] = kurtosis(intensities)
    
    # Energy (sum of squared intensities)
    features['firstorder_Energy'] = np.sum(intensities ** 2)
    
    # Entropy
    # Bin the intensities for histogram
    hist, _ = np.histogram(intensities, bins=256)
    hist = hist / hist.sum()  # Normalize to probabilities
    hist = hist[hist > 0]  # Remove zeros
    features['firstorder_Entropy'] = entropy(hist)
    
    # Percentiles
    features['firstorder_P10'] = np.percentile(intensities, 10)
    features['firstorder_P25'] = np.percentile(intensities, 25)
    features['firstorder_P75'] = np.percentile(intensities, 75)
    features['firstorder_P90'] = np.percentile(intensities, 90)
    features['firstorder_IQR'] = features['firstorder_P75'] - features['firstorder_P25']
    
    # Mean Absolute Deviation
    features['firstorder_MAD'] = np.mean(np.abs(intensities - features['firstorder_Mean']))
    
    # Robust Mean Absolute Deviation (using only 10-90 percentile)
    p10, p90 = features['firstorder_P10'], features['firstorder_P90']
    robust_intensities = intensities[(intensities >= p10) & (intensities <= p90)]
    if len(robust_intensities) > 0:
        robust_mean = np.mean(robust_intensities)
        features['firstorder_RobustMAD'] = np.mean(np.abs(robust_intensities - robust_mean))
    else:
        features['firstorder_RobustMAD'] = 0
    
    # Uniformity (sum of squared histogram probabilities)
    features['firstorder_Uniformity'] = np.sum(hist ** 2)
    
    return features


def extract_texture_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Extract simplified texture features (GLCM-like metrics).
    
    Features:
    - Gradient statistics (edge/texture information)
    - Local variance (heterogeneity)
    - Coarseness metrics
    """
    features = {}
    
    tumor_mask = mask > 0
    
    if np.sum(tumor_mask) == 0:
        for key in ['GradientMean', 'GradientStd', 'GradientMax', 
                    'LocalVarianceMean', 'LocalVarianceStd',
                    'Coarseness', 'Contrast']:
            features[f'texture_{key}'] = 0
        return features
    
    # Create a masked version of the image (set non-tumor to 0)
    masked_image = np.where(tumor_mask, image, 0).astype(float)
    
    # Gradient magnitude (edge strength - measure of texture)
    gradient_z = ndimage.sobel(masked_image, axis=0)
    gradient_y = ndimage.sobel(masked_image, axis=1)
    gradient_x = ndimage.sobel(masked_image, axis=2)
    gradient_magnitude = np.sqrt(gradient_z**2 + gradient_y**2 + gradient_x**2)
    
    tumor_gradients = gradient_magnitude[tumor_mask]
    features['texture_GradientMean'] = np.mean(tumor_gradients)
    features['texture_GradientStd'] = np.std(tumor_gradients)
    features['texture_GradientMax'] = np.max(tumor_gradients)
    
    # Local variance (heterogeneity measure)
    # Use a small neighborhood
    local_mean = ndimage.uniform_filter(masked_image, size=3)
    local_sq_mean = ndimage.uniform_filter(masked_image**2, size=3)
    local_variance = local_sq_mean - local_mean**2
    local_variance = np.maximum(local_variance, 0)  # Ensure non-negative
    
    tumor_local_var = local_variance[tumor_mask]
    features['texture_LocalVarianceMean'] = np.mean(tumor_local_var)
    features['texture_LocalVarianceStd'] = np.std(tumor_local_var)
    
    # Coarseness (inverse of sum of gradients)
    gradient_sum = np.sum(tumor_gradients)
    if gradient_sum > 0:
        features['texture_Coarseness'] = len(tumor_gradients) / gradient_sum
    else:
        features['texture_Coarseness'] = 0
    
    # Contrast (related to intensity differences between neighbors)
    # Simplified: use range of local means
    tumor_local_mean = local_mean[tumor_mask]
    features['texture_Contrast'] = np.max(tumor_local_mean) - np.min(tumor_local_mean)
    
    # Additional GLCM-like features using simplified approach
    # Homogeneity proxy: inverse of variance
    if features['texture_LocalVarianceMean'] > 0:
        features['texture_Homogeneity'] = 1 / (1 + features['texture_LocalVarianceMean'])
    else:
        features['texture_Homogeneity'] = 1
    
    # Dissimilarity proxy: mean absolute difference from neighbors
    diff_z = np.abs(np.diff(masked_image, axis=0))
    diff_y = np.abs(np.diff(masked_image, axis=1))
    diff_x = np.abs(np.diff(masked_image, axis=2))
    
    # Need to handle mask for differences
    mask_z = tumor_mask[:-1, :, :] & tumor_mask[1:, :, :]
    mask_y = tumor_mask[:, :-1, :] & tumor_mask[:, 1:, :]
    mask_x = tumor_mask[:, :, :-1] & tumor_mask[:, :, 1:]
    
    all_diffs = np.concatenate([
        diff_z[mask_z],
        diff_y[mask_y],
        diff_x[mask_x]
    ])
    
    if len(all_diffs) > 0:
        features['texture_Dissimilarity'] = np.mean(all_diffs)
    else:
        features['texture_Dissimilarity'] = 0
    
    return features


def extract_features_for_patient(image_path: Path, mask_path: Path) -> Dict[str, Any]:
    """
    Extract all radiomic features for a single patient.
    """
    # Load NIfTI files
    img_nii = nib.load(str(image_path))
    mask_nii = nib.load(str(mask_path))
    
    image = img_nii.get_fdata().astype(float)
    mask = mask_nii.get_fdata().astype(int)
    
    # Get voxel spacing from header
    voxel_spacing = img_nii.header.get_zooms()[:3]
    
    # Check dimensions match
    if image.shape != mask.shape:
        # Try to resample mask to image space using nearest neighbor
        from scipy.ndimage import zoom
        
        # Calculate zoom factors
        zoom_factors = [image.shape[i] / mask.shape[i] for i in range(3)]
        
        # Resample mask (use order=0 for nearest neighbor to preserve labels)
        mask = zoom(mask, zoom_factors, order=0).astype(int)
        
        # Verify shapes now match
        if image.shape != mask.shape:
            raise ValueError(f"Shape mismatch after resampling: image {image.shape} vs mask {mask.shape}")
    
    # Check if mask has any tumor voxels
    if mask.max() == 0:
        raise ValueError("Mask is empty (no tumor voxels)")
    
    # Extract features
    features = {}
    
    # Shape features
    shape_features = extract_shape_features(mask, voxel_spacing)
    features.update(shape_features)
    
    # First-order features
    firstorder_features = extract_firstorder_features(image, mask)
    features.update(firstorder_features)
    
    # Texture features
    texture_features = extract_texture_features(image, mask)
    features.update(texture_features)
    
    return features


def main():
    """Main pipeline to extract radiomic features for all patients."""
    print("=" * 70)
    print("Level 4: Radiomic Feature Extraction for MGMT Prediction")
    print("(Pure Python Implementation - No pyradiomics required)")
    print("=" * 70)
    
    # Step 1: Load the dataset
    print(f"\nLoading dataset from: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        print("ERROR: Input CSV not found!")
        print("Run 03_create_mgmt_dataset.py first.")
        sys.exit(1)
    
    df = pd.read_csv(INPUT_CSV)
    print(f"  Total patients to process: {len(df)}")
    
    # Step 2: Process each patient
    print("\n" + "-" * 70)
    print("Processing patients...")
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
            # Find image and mask files
            image_path = find_image_file(scan_path)
            mask_path = find_mask_file(scan_path)
            
            if image_path is None:
                raise FileNotFoundError(f"No T1CE/T1 image found in {scan_path}")
            
            if mask_path is None:
                raise FileNotFoundError(f"No segmentation mask found in {scan_path}")
            
            print(f"  Image: {image_path.name}")
            print(f"  Mask:  {mask_path.name}")
            
            # Extract features
            features = extract_features_for_patient(image_path, mask_path)
            
            # Add patient info
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
    
    # Step 3: Create output DataFrame
    print("\n" + "-" * 70)
    print("Creating output dataset...")
    print("-" * 70)
    
    if not results:
        print("ERROR: No patients were successfully processed!")
        sys.exit(1)
    
    df_features = pd.DataFrame(results)
    
    # Reorder columns: Patient_ID, MGMT_Label first, then features
    cols = ['Patient_ID', 'MGMT_Label'] + [c for c in df_features.columns 
                                            if c not in ['Patient_ID', 'MGMT_Label']]
    df_features = df_features[cols]
    
    # Step 4: Save to CSV
    df_features.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFeatures saved to: {OUTPUT_CSV}")
    
    # Step 5: Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successfully processed: {successful}/{len(df)} patients")
    print(f"Failed: {failed} patients")
    
    if failed_patients:
        print(f"\nFailed patients: {failed_patients[:10]}")
        if len(failed_patients) > 10:
            print(f"  ... and {len(failed_patients) - 10} more")
    
    # Feature summary
    feature_cols = [c for c in df_features.columns if c not in ['Patient_ID', 'MGMT_Label']]
    print(f"\nTotal features extracted: {len(feature_cols)}")
    
    # Group by feature type
    shape_features = [c for c in feature_cols if c.startswith('shape_')]
    firstorder_features = [c for c in feature_cols if c.startswith('firstorder_')]
    texture_features = [c for c in feature_cols if c.startswith('texture_')]
    
    print(f"  - Shape features: {len(shape_features)}")
    print(f"  - FirstOrder features: {len(firstorder_features)}")
    print(f"  - Texture features: {len(texture_features)}")
    
    # Class distribution
    print(f"\nMGMT Label distribution (in extracted features):")
    print(f"  - Methylated (1): {(df_features['MGMT_Label'] == 1).sum()}")
    print(f"  - Unmethylated (0): {(df_features['MGMT_Label'] == 0).sum()}")
    
    print("\nFeature names:")
    for feat in feature_cols:
        print(f"  - {feat}")
    
    return df_features


if __name__ == "__main__":
    main()
