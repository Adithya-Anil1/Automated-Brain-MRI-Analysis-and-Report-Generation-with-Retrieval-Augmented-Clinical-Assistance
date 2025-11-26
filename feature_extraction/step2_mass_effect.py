"""
Step 2: Mass Effect Metrics

Analyzes mass effect indicators:
- Midline shift calculation
- Ventricular compression analysis
- Herniation signs detection
- Sulcal effacement

Author: AI-Powered Brain MRI Assistant
Date: November 27, 2025
"""

import numpy as np
import argparse
from pathlib import Path
from scipy import ndimage

from utils import (
    load_nifti, get_case_id, get_mri_paths, get_voxel_dimensions,
    get_tumor_masks, get_centroid, get_bounding_box,
    get_brain_mask, save_results
)


def calculate_midline_shift(brain_data, seg_data, voxel_dims):
    """
    Calculate midline shift by comparing tumor location to brain center.
    
    Midline shift is the displacement of brain midline structures
    from their normal position, typically measured in mm.
    """
    # Get brain mask (non-zero voxels)
    brain_mask = get_brain_mask(brain_data)
    
    # Get tumor mask
    tumor_mask = seg_data > 0
    
    if tumor_mask.sum() == 0:
        return {
            'shift_mm': 0,
            'shift_direction': 'None',
            'severity': 'No tumor detected',
            'details': 'No tumor present to cause mass effect'
        }
    
    # Find brain boundaries in the left-right axis (typically x-axis)
    # Assuming standard radiological orientation: x = left-right
    brain_coords = np.where(brain_mask)
    
    if len(brain_coords[0]) == 0:
        return {
            'shift_mm': None,
            'shift_direction': 'Unknown',
            'severity': 'Could not calculate',
            'details': 'Brain mask could not be determined'
        }
    
    # Find the anatomical midline (center of brain in x-axis)
    brain_x_min = brain_coords[0].min()
    brain_x_max = brain_coords[0].max()
    anatomical_midline_x = (brain_x_min + brain_x_max) / 2
    
    # Find tumor centroid
    tumor_coords = np.where(tumor_mask)
    tumor_centroid_x = tumor_coords[0].mean()
    tumor_centroid_y = tumor_coords[1].mean()
    
    # For midline shift, we look at the septum pellucidum / third ventricle region
    # Approximate by looking at central brain structures
    brain_width = brain_x_max - brain_x_min
    
    # Check which hemisphere the tumor is in
    if tumor_centroid_x < anatomical_midline_x:
        tumor_side = 'left'
    else:
        tumor_side = 'right'
    
    # Estimate shift based on tumor volume and proximity to midline
    # Large tumors close to midline cause more shift
    tumor_volume_voxels = tumor_mask.sum()
    distance_to_midline = abs(tumor_centroid_x - anatomical_midline_x) * voxel_dims[0]
    
    # Simple model: shift is proportional to tumor mass and inversely to distance
    # This is a simplified estimate - true midline shift requires identifying
    # specific anatomical structures (septum pellucidum, falx cerebri)
    
    # Calculate approximate shift based on tumor presence
    # Look at asymmetry in brain tissue around the midline
    left_half = brain_mask.copy()
    right_half = brain_mask.copy()
    
    midline_idx = int(anatomical_midline_x)
    left_half[midline_idx:, :, :] = 0
    right_half[:midline_idx, :, :] = 0
    
    # Calculate center of mass for each hemisphere
    if left_half.sum() > 0 and right_half.sum() > 0:
        left_com = ndimage.center_of_mass(left_half)
        right_com = ndimage.center_of_mass(right_half)
        
        # Expected COM positions (symmetric around midline)
        expected_left_x = anatomical_midline_x - (brain_width / 4)
        expected_right_x = anatomical_midline_x + (brain_width / 4)
        
        # Calculate deviation from expected
        left_shift = (left_com[0] - expected_left_x) * voxel_dims[0]
        right_shift = (right_com[0] - expected_right_x) * voxel_dims[0]
        
        # Net shift (positive = shift to right, negative = shift to left)
        # When tumor is on left, midline shifts right (positive)
        # When tumor is on right, midline shifts left (negative)
        estimated_shift = (left_shift + right_shift) / 2
    else:
        estimated_shift = 0
    
    # Determine direction and severity
    shift_mm = abs(estimated_shift)
    
    if estimated_shift > 0:
        shift_direction = 'Right to left' if tumor_side == 'right' else 'Left to right'
    elif estimated_shift < 0:
        shift_direction = 'Left to right' if tumor_side == 'right' else 'Right to left'
    else:
        shift_direction = 'None'
    
    # Clinical severity classification
    if shift_mm < 3:
        severity = 'Minimal/None'
        clinical = 'No significant midline shift'
    elif shift_mm < 5:
        severity = 'Mild'
        clinical = 'Mild midline shift, typically not requiring urgent intervention'
    elif shift_mm < 10:
        severity = 'Moderate'
        clinical = 'Moderate midline shift, close monitoring recommended'
    else:
        severity = 'Severe'
        clinical = 'Severe midline shift, may require urgent intervention'
    
    return {
        'shift_mm': float(shift_mm),
        'shift_direction': shift_direction,
        'tumor_hemisphere': tumor_side,
        'severity': severity,
        'clinical_significance': clinical,
        'brain_midline_x': float(anatomical_midline_x),
        'tumor_centroid_x': float(tumor_centroid_x),
        'note': 'Estimated from tissue asymmetry - clinical correlation recommended'
    }


def analyze_ventricular_compression(brain_data, seg_data, voxel_dims):
    """
    Analyze ventricular system for signs of compression.
    
    Uses intensity analysis to identify ventricles (CSF = dark on T1, bright on T2)
    and checks for asymmetry or compression.
    """
    brain_mask = get_brain_mask(brain_data)
    tumor_mask = seg_data > 0
    
    # CSF appears as very low intensity on T1 (or high on T2)
    # Estimate CSF threshold
    brain_values = brain_data[brain_mask]
    
    if len(brain_values) == 0:
        return {
            'compression_detected': False,
            'details': 'Could not analyze - no brain tissue detected'
        }
    
    # CSF typically in lowest intensity quartile
    csf_threshold = np.percentile(brain_values, 15)
    
    # Potential CSF/ventricle mask
    potential_csf = (brain_data < csf_threshold) & (brain_data > 0)
    
    # Exclude regions that overlap with tumor
    csf_mask = potential_csf & ~tumor_mask
    
    # Get image dimensions
    dims = brain_data.shape
    midline_x = dims[0] // 2
    
    # Split into hemispheres
    left_csf = csf_mask.copy()
    left_csf[midline_x:, :, :] = 0
    
    right_csf = csf_mask.copy()
    right_csf[:midline_x, :, :] = 0
    
    left_csf_volume = left_csf.sum() * np.prod(voxel_dims) / 1000  # cm³
    right_csf_volume = right_csf.sum() * np.prod(voxel_dims) / 1000
    
    # Calculate asymmetry ratio
    total_csf = left_csf_volume + right_csf_volume
    if total_csf > 0:
        asymmetry_ratio = abs(left_csf_volume - right_csf_volume) / total_csf
    else:
        asymmetry_ratio = 0
    
    # Determine which side is compressed (smaller ventricle)
    if left_csf_volume < right_csf_volume * 0.7:
        compressed_side = 'left'
        compression_detected = True
    elif right_csf_volume < left_csf_volume * 0.7:
        compressed_side = 'right'
        compression_detected = True
    else:
        compressed_side = 'none'
        compression_detected = False
    
    # Check proximity of tumor to ventricles
    tumor_coords = np.where(tumor_mask)
    csf_coords = np.where(csf_mask)
    
    if len(tumor_coords[0]) > 0 and len(csf_coords[0]) > 0:
        # Find minimum distance between tumor and ventricles
        # Sample points for efficiency
        n_samples = min(1000, len(tumor_coords[0]))
        sample_idx = np.random.choice(len(tumor_coords[0]), n_samples, replace=False)
        
        tumor_points = np.array([tumor_coords[0][sample_idx],
                                  tumor_coords[1][sample_idx],
                                  tumor_coords[2][sample_idx]]).T
        
        csf_points = np.array([csf_coords[0], csf_coords[1], csf_coords[2]]).T
        
        if len(csf_points) > 1000:
            csf_sample_idx = np.random.choice(len(csf_points), 1000, replace=False)
            csf_points = csf_points[csf_sample_idx]
        
        # Calculate minimum distances
        min_distances = []
        for tp in tumor_points:
            distances = np.sqrt(np.sum((csf_points - tp)**2, axis=1))
            min_distances.append(distances.min())
        
        min_tumor_ventricle_distance = np.min(min_distances) * voxel_dims[0]
        mean_tumor_ventricle_distance = np.mean(min_distances) * voxel_dims[0]
    else:
        min_tumor_ventricle_distance = None
        mean_tumor_ventricle_distance = None
    
    # Severity assessment
    if asymmetry_ratio > 0.5:
        severity = 'Severe'
    elif asymmetry_ratio > 0.3:
        severity = 'Moderate'
    elif asymmetry_ratio > 0.15:
        severity = 'Mild'
    else:
        severity = 'None/Minimal'
    
    return {
        'compression_detected': compression_detected,
        'compressed_side': compressed_side,
        'asymmetry_ratio': float(asymmetry_ratio),
        'left_ventricle_volume_cm3': float(left_csf_volume),
        'right_ventricle_volume_cm3': float(right_csf_volume),
        'severity': severity,
        'tumor_to_ventricle_distance_mm': float(min_tumor_ventricle_distance) if min_tumor_ventricle_distance else None,
        'note': 'Based on CSF intensity analysis - MRI sequence-dependent'
    }


def detect_herniation_signs(brain_data, seg_data, voxel_dims):
    """
    Detect signs of brain herniation.
    
    Types of herniation:
    - Subfalcine: midline shift > 5mm
    - Transtentorial: tumor in temporal lobe with brainstem compression
    - Tonsillar: cerebellar tonsils below foramen magnum
    """
    tumor_mask = seg_data > 0
    brain_mask = get_brain_mask(brain_data)
    
    herniation_signs = []
    risk_level = 'Low'
    
    if tumor_mask.sum() == 0:
        return {
            'herniation_signs': [],
            'risk_level': 'None',
            'details': 'No tumor detected'
        }
    
    # Get tumor location
    tumor_bbox = get_bounding_box(tumor_mask)
    tumor_centroid = get_centroid(tumor_mask)
    
    dims = brain_data.shape
    
    # Check for subfalcine herniation risk (large tumor near midline)
    midline_x = dims[0] // 2
    tumor_distance_to_midline = abs(tumor_centroid['x'] - midline_x) * voxel_dims[0]
    
    tumor_volume = tumor_mask.sum() * np.prod(voxel_dims) / 1000  # cm³
    
    if tumor_distance_to_midline < 30 and tumor_volume > 30:
        herniation_signs.append('Subfalcine herniation risk - large tumor near midline')
        risk_level = 'Moderate'
    
    # Check for transtentorial herniation risk (tumor in lower brain regions)
    # Lower z-values indicate inferior brain position
    z_threshold = dims[2] * 0.3  # Lower 30% of brain
    tumor_in_lower_brain = tumor_bbox['min_z'] < z_threshold
    
    if tumor_in_lower_brain and tumor_volume > 20:
        herniation_signs.append('Transtentorial herniation risk - tumor in lower brain region')
        if risk_level != 'High':
            risk_level = 'Moderate'
    
    # Check for tonsillar herniation risk (tumor in posterior fossa)
    # Posterior = high y values, low z values
    y_threshold = dims[1] * 0.7
    posterior_fossa_tumor = (tumor_centroid['y'] > y_threshold) and (tumor_centroid['z'] < z_threshold)
    
    if posterior_fossa_tumor and tumor_volume > 10:
        herniation_signs.append('Tonsillar herniation risk - posterior fossa mass')
        risk_level = 'High'
    
    # Check for uncal herniation (temporal lobe tumor)
    # Temporal = lateral position, middle-inferior z level
    lateral_threshold = dims[0] * 0.3
    temporal_position = (tumor_centroid['x'] < lateral_threshold or 
                         tumor_centroid['x'] > dims[0] - lateral_threshold)
    middle_z = dims[2] * 0.3 < tumor_centroid['z'] < dims[2] * 0.6
    
    if temporal_position and middle_z and tumor_volume > 25:
        herniation_signs.append('Uncal herniation risk - temporal lobe mass')
        if risk_level == 'Low':
            risk_level = 'Moderate'
    
    if not herniation_signs:
        herniation_signs.append('No obvious herniation signs detected')
    
    return {
        'herniation_signs': herniation_signs,
        'risk_level': risk_level,
        'tumor_volume_cm3': float(tumor_volume),
        'tumor_location': {
            'centroid': tumor_centroid,
            'bounding_box': tumor_bbox,
            'distance_to_midline_mm': float(tumor_distance_to_midline)
        },
        'note': 'Automated estimation - clinical and imaging correlation essential'
    }


def analyze_sulcal_effacement(brain_data, seg_data, voxel_dims):
    """
    Analyze sulcal effacement (compression of brain sulci by mass effect).
    """
    tumor_mask = seg_data > 0
    brain_mask = get_brain_mask(brain_data)
    
    if tumor_mask.sum() == 0:
        return {
            'effacement_detected': False,
            'details': 'No tumor detected'
        }
    
    # Get tumor bounding box
    tumor_bbox = get_bounding_box(tumor_mask)
    
    # Analyze brain tissue density around tumor
    # Create expanded region around tumor
    expanded_tumor = ndimage.binary_dilation(tumor_mask, iterations=10)
    peritumoral_region = expanded_tumor & ~tumor_mask & brain_mask
    
    if peritumoral_region.sum() == 0:
        return {
            'effacement_detected': False,
            'details': 'Could not analyze peritumoral region'
        }
    
    # Compare tissue density in peritumoral region vs distant brain
    distant_brain = brain_mask & ~expanded_tumor
    
    if distant_brain.sum() == 0:
        return {
            'effacement_detected': True,
            'severity': 'Severe',
            'details': 'Tumor occupies majority of brain volume'
        }
    
    # Calculate intensity variance (sulci have different intensity than gyri)
    peritumoral_std = np.std(brain_data[peritumoral_region])
    distant_std = np.std(brain_data[distant_brain])
    
    # Lower variance in peritumoral region suggests effaced sulci
    if distant_std > 0:
        variance_ratio = peritumoral_std / distant_std
    else:
        variance_ratio = 1.0
    
    if variance_ratio < 0.6:
        effacement = True
        severity = 'Moderate to Severe'
    elif variance_ratio < 0.8:
        effacement = True
        severity = 'Mild to Moderate'
    else:
        effacement = False
        severity = 'None/Minimal'
    
    return {
        'effacement_detected': effacement,
        'severity': severity,
        'variance_ratio': float(variance_ratio),
        'peritumoral_intensity_std': float(peritumoral_std),
        'normal_brain_intensity_std': float(distant_std),
        'note': 'Based on intensity variance analysis'
    }


def generate_summary(results):
    """Generate text summary for radiology report."""
    lines = []
    
    lines.append("MASS EFFECT ANALYSIS:")
    lines.append("")
    
    # Midline shift
    ms = results['midline_shift']
    lines.append(f"Midline Shift: {ms['severity']}")
    if ms['shift_mm'] is not None and ms['shift_mm'] > 0:
        lines.append(f"  - Estimated shift: {ms['shift_mm']:.1f} mm ({ms['shift_direction']})")
        lines.append(f"  - Tumor hemisphere: {ms['tumor_hemisphere']}")
    lines.append(f"  - {ms.get('clinical_significance', '')}")
    
    # Ventricular compression
    lines.append("")
    vc = results['ventricular_compression']
    lines.append(f"Ventricular System: {vc['severity']} compression")
    if vc['compression_detected']:
        lines.append(f"  - Compressed side: {vc['compressed_side']}")
        lines.append(f"  - Asymmetry ratio: {vc['asymmetry_ratio']:.2f}")
    if vc.get('tumor_to_ventricle_distance_mm'):
        lines.append(f"  - Tumor-ventricle distance: {vc['tumor_to_ventricle_distance_mm']:.1f} mm")
    
    # Herniation
    lines.append("")
    hr = results['herniation_risk']
    lines.append(f"Herniation Risk: {hr['risk_level']}")
    for sign in hr['herniation_signs']:
        lines.append(f"  - {sign}")
    
    # Sulcal effacement
    lines.append("")
    se = results['sulcal_effacement']
    lines.append(f"Sulcal Effacement: {se['severity']}")
    
    return "\n".join(lines)


def analyze_mass_effect(input_folder, segmentation_path, output_path=None):
    """
    Main function to analyze mass effect metrics.
    """
    input_folder = Path(input_folder)
    case_id = get_case_id(input_folder)
    print(f"Analyzing case: {case_id}")
    
    # Get MRI paths - use T1 for structural analysis
    mri_paths = get_mri_paths(input_folder, case_id)
    
    # Load T1 for structural analysis
    print("Loading MRI data...")
    t1_data, t1_affine, t1_header = load_nifti(mri_paths['t1'])
    
    # Load segmentation
    print("Loading segmentation mask...")
    seg_data, _, _ = load_nifti(segmentation_path)
    seg_data = np.round(seg_data).astype(np.int32)
    
    # Get voxel dimensions
    voxel_info = get_voxel_dimensions(t1_header)
    voxel_dims = voxel_info['dimensions_mm']
    
    print("\n" + "="*60)
    print("STEP 2: MASS EFFECT METRICS")
    print("="*60)
    
    # Calculate midline shift
    print("\n--- Midline Shift Analysis ---")
    midline_results = calculate_midline_shift(t1_data, seg_data, voxel_dims)
    print(f"  Severity: {midline_results['severity']}")
    if midline_results['shift_mm']:
        print(f"  Estimated shift: {midline_results['shift_mm']:.1f} mm")
    
    # Analyze ventricular compression
    print("\n--- Ventricular Compression Analysis ---")
    ventricle_results = analyze_ventricular_compression(t1_data, seg_data, voxel_dims)
    print(f"  Compression: {ventricle_results['severity']}")
    if ventricle_results['compression_detected']:
        print(f"  Compressed side: {ventricle_results['compressed_side']}")
    
    # Detect herniation signs
    print("\n--- Herniation Risk Assessment ---")
    herniation_results = detect_herniation_signs(t1_data, seg_data, voxel_dims)
    print(f"  Risk level: {herniation_results['risk_level']}")
    for sign in herniation_results['herniation_signs']:
        print(f"  - {sign}")
    
    # Analyze sulcal effacement
    print("\n--- Sulcal Effacement Analysis ---")
    sulcal_results = analyze_sulcal_effacement(t1_data, seg_data, voxel_dims)
    print(f"  Effacement: {sulcal_results['severity']}")
    
    # Compile results
    results = {
        'case_id': case_id,
        'step': 'Step 2 - Mass effect metrics',
        'voxel_info': voxel_info,
        'midline_shift': midline_results,
        'ventricular_compression': ventricle_results,
        'herniation_risk': herniation_results,
        'sulcal_effacement': sulcal_results
    }
    
    # Generate summary
    summary = generate_summary(results)
    results['text_summary'] = summary
    
    print("\n" + "="*60)
    print("TEXT SUMMARY FOR RADIOLOGY REPORT")
    print("="*60)
    print(summary)
    
    # Save results
    if output_path:
        save_results(results, output_path)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Step 2: Analyze mass effect metrics from brain MRI'
    )
    parser.add_argument('--input', required=True, 
                        help='Input folder containing MRI sequences')
    parser.add_argument('--segmentation', required=True,
                        help='Path to segmentation mask (NIfTI)')
    parser.add_argument('--output', default=None,
                        help='Output path for JSON results')
    
    args = parser.parse_args()
    
    analyze_mass_effect(args.input, args.segmentation, args.output)


if __name__ == "__main__":
    main()
