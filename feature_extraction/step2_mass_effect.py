"""
Step 2: Mass Effect Metrics

Analyzes mass effect indicators:
- Midline shift calculation (evidence-based)
- Ventricular compression analysis
- Herniation risk (derived from measurable displacement, not proximity)
- Sulcal effacement
- Anatomical localization

Author: AI-Powered Brain MRI Assistant
Date: November 27, 2025
"""

import numpy as np
import argparse
from pathlib import Path
from scipy import ndimage
from scipy.ndimage import binary_erosion

from utils import (
    load_nifti, get_case_id, get_mri_paths, get_voxel_dimensions,
    get_tumor_masks, get_centroid, get_bounding_box,
    get_brain_mask, save_results
)


# Minimum threshold for reporting directional shift (below this is noise)
SHIFT_NOISE_THRESHOLD_MM = 1.0


def calculate_midline_shift(brain_data, seg_data, voxel_dims):
    """
    Calculate midline shift by comparing tumor location to brain center.
    
    Midline shift is the displacement of brain midline structures
    from their normal position, typically measured in mm.
    
    Below SHIFT_NOISE_THRESHOLD_MM, direction is not reported as it's
    below measurement precision.
    """
    brain_mask = get_brain_mask(brain_data)
    tumor_mask = seg_data > 0
    
    if tumor_mask.sum() == 0:
        return {
            'shift_mm': 0,
            'shift_direction': 'Not applicable',
            'severity': 'No tumor detected',
            'clinical_significance': 'No tumor present to cause mass effect',
            'is_significant': False
        }
    
    brain_coords = np.where(brain_mask)
    
    if len(brain_coords[0]) == 0:
        return {
            'shift_mm': None,
            'shift_direction': 'Unknown',
            'severity': 'Could not calculate',
            'clinical_significance': 'Brain mask could not be determined',
            'is_significant': False
        }
    
    # Find the anatomical midline
    brain_x_min = brain_coords[0].min()
    brain_x_max = brain_coords[0].max()
    anatomical_midline_x = (brain_x_min + brain_x_max) / 2
    brain_width = brain_x_max - brain_x_min
    
    # Find tumor centroid
    tumor_coords = np.where(tumor_mask)
    tumor_centroid_x = tumor_coords[0].mean()
    
    # Determine which hemisphere the tumor is in
    if tumor_centroid_x < anatomical_midline_x:
        tumor_side = 'left'
    else:
        tumor_side = 'right'
    
    distance_to_midline = abs(tumor_centroid_x - anatomical_midline_x) * voxel_dims[0]
    
    # Calculate shift based on hemispheric asymmetry
    left_half = brain_mask.copy()
    right_half = brain_mask.copy()
    
    midline_idx = int(anatomical_midline_x)
    left_half[midline_idx:, :, :] = 0
    right_half[:midline_idx, :, :] = 0
    
    if left_half.sum() > 0 and right_half.sum() > 0:
        left_com = ndimage.center_of_mass(left_half)
        right_com = ndimage.center_of_mass(right_half)
        
        expected_left_x = anatomical_midline_x - (brain_width / 4)
        expected_right_x = anatomical_midline_x + (brain_width / 4)
        
        left_shift = (left_com[0] - expected_left_x) * voxel_dims[0]
        right_shift = (right_com[0] - expected_right_x) * voxel_dims[0]
        
        estimated_shift = (left_shift + right_shift) / 2
    else:
        estimated_shift = 0
    
    shift_mm = abs(estimated_shift)
    
    # Determine if shift is significant (above noise threshold)
    is_significant = shift_mm >= SHIFT_NOISE_THRESHOLD_MM
    
    # Only report direction if shift is significant
    if not is_significant:
        shift_direction = 'Not applicable (below measurement threshold)'
        severity = 'None'
        clinical = 'No significant midline shift detected'
    elif shift_mm < 3:
        if estimated_shift > 0:
            shift_direction = 'Left to right' if tumor_side == 'left' else 'Right to left'
        else:
            shift_direction = 'Right to left' if tumor_side == 'left' else 'Left to right'
        severity = 'Minimal'
        clinical = 'No significant midline shift detected'
    elif shift_mm < 5:
        if estimated_shift > 0:
            shift_direction = 'Left to right' if tumor_side == 'left' else 'Right to left'
        else:
            shift_direction = 'Right to left' if tumor_side == 'left' else 'Left to right'
        severity = 'Mild'
        clinical = 'Mild midline shift, close monitoring recommended'
    elif shift_mm < 10:
        if estimated_shift > 0:
            shift_direction = 'Left to right' if tumor_side == 'left' else 'Right to left'
        else:
            shift_direction = 'Right to left' if tumor_side == 'left' else 'Left to right'
        severity = 'Moderate'
        clinical = 'Moderate midline shift, close monitoring recommended'
    else:
        if estimated_shift > 0:
            shift_direction = 'Left to right' if tumor_side == 'left' else 'Right to left'
        else:
            shift_direction = 'Right to left' if tumor_side == 'left' else 'Left to right'
        severity = 'Severe'
        clinical = 'Severe midline shift, may require urgent intervention'
    
    return {
        'shift_mm': float(shift_mm),
        'shift_direction': shift_direction,
        'tumor_hemisphere': tumor_side,
        'severity': severity,
        'clinical_significance': clinical,
        'is_significant': is_significant,
        'brain_midline_x': float(anatomical_midline_x),
        'tumor_centroid_x': float(tumor_centroid_x),
        'distance_to_midline_mm': float(distance_to_midline),
        'measurement_threshold_mm': SHIFT_NOISE_THRESHOLD_MM,
        'note': 'Estimated from tissue asymmetry - clinical correlation recommended'
    }


def analyze_ventricular_compression(brain_data, seg_data, voxel_dims):
    """
    Analyze ventricular system for signs of compression.
    
    Returns asymmetry ratio and severity based on measurable distortion.
    """
    brain_mask = get_brain_mask(brain_data)
    tumor_mask = seg_data > 0
    
    brain_values = brain_data[brain_mask]
    
    if len(brain_values) == 0:
        return {
            'compression_detected': False,
            'severity': 'Could not analyze',
            'asymmetry_ratio': 0,
            'details': 'Could not analyze - no brain tissue detected'
        }
    
    # CSF threshold
    csf_threshold = np.percentile(brain_values, 15)
    potential_csf = (brain_data < csf_threshold) & (brain_data > 0)
    csf_mask = potential_csf & ~tumor_mask
    
    dims = brain_data.shape
    midline_x = dims[0] // 2
    
    left_csf = csf_mask.copy()
    left_csf[midline_x:, :, :] = 0
    
    right_csf = csf_mask.copy()
    right_csf[:midline_x, :, :] = 0
    
    left_csf_volume = left_csf.sum() * np.prod(voxel_dims) / 1000
    right_csf_volume = right_csf.sum() * np.prod(voxel_dims) / 1000
    
    total_csf = left_csf_volume + right_csf_volume
    asymmetry_ratio = abs(left_csf_volume - right_csf_volume) / total_csf if total_csf > 0 else 0
    
    if left_csf_volume < right_csf_volume * 0.7:
        compressed_side = 'left'
        compression_detected = True
    elif right_csf_volume < left_csf_volume * 0.7:
        compressed_side = 'right'
        compression_detected = True
    else:
        compressed_side = 'none'
        compression_detected = False
    
    # Calculate tumor-ventricle distance
    tumor_coords = np.where(tumor_mask)
    csf_coords = np.where(csf_mask)
    
    min_tumor_ventricle_distance = None
    if len(tumor_coords[0]) > 0 and len(csf_coords[0]) > 0:
        n_samples = min(1000, len(tumor_coords[0]))
        sample_idx = np.random.choice(len(tumor_coords[0]), n_samples, replace=False)
        
        tumor_points = np.array([tumor_coords[0][sample_idx],
                                  tumor_coords[1][sample_idx],
                                  tumor_coords[2][sample_idx]]).T
        
        csf_points = np.array([csf_coords[0], csf_coords[1], csf_coords[2]]).T
        
        if len(csf_points) > 1000:
            csf_sample_idx = np.random.choice(len(csf_points), 1000, replace=False)
            csf_points = csf_points[csf_sample_idx]
        
        min_distances = []
        for tp in tumor_points:
            distances = np.sqrt(np.sum((csf_points - tp)**2, axis=1))
            min_distances.append(distances.min())
        
        min_tumor_ventricle_distance = float(np.min(min_distances) * voxel_dims[0])
    
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
        'tumor_to_ventricle_distance_mm': min_tumor_ventricle_distance,
        'note': 'Based on CSF intensity analysis - MRI sequence-dependent'
    }


def assess_herniation_risk(midline_shift_results, ventricular_results, sulcal_results, tumor_volume_cm3, tumor_location):
    """
    Assess herniation risk based on MEASURABLE displacement metrics.
    
    Risk is derived from:
    1. Midline shift magnitude (primary)
    2. Ventricular distortion (secondary)
    3. Sulcal effacement (tertiary)
    
    NOT derived from:
    - Tumor size alone
    - Proximity to midline without displacement
    
    This is clinically consistent: you cannot have herniation risk without mass effect.
    """
    risk_factors = []
    herniation_signs = []
    
    # Get the actual measurements
    shift_mm = midline_shift_results.get('shift_mm', 0) or 0
    shift_is_significant = midline_shift_results.get('is_significant', False)
    ventricular_asymmetry = ventricular_results.get('asymmetry_ratio', 0) or 0
    ventricular_severity = ventricular_results.get('severity', 'None/Minimal')
    sulcal_severity = sulcal_results.get('severity', 'None/Minimal')
    
    # Calculate mass effect score from measurable metrics
    mass_effect_score = 0
    
    # Primary: Midline shift
    if shift_mm >= 10:
        mass_effect_score += 4
        risk_factors.append(f'Midline shift: {shift_mm:.1f}mm (severe)')
        herniation_signs.append('Severe midline shift (>10mm) - high subfalcine herniation risk')
    elif shift_mm >= 5:
        mass_effect_score += 3
        risk_factors.append(f'Midline shift: {shift_mm:.1f}mm (moderate)')
        herniation_signs.append('Moderate midline shift (5-10mm) - subfalcine herniation possible')
    elif shift_mm >= 3:
        mass_effect_score += 2
        risk_factors.append(f'Midline shift: {shift_mm:.1f}mm (mild)')
        herniation_signs.append('Mild midline shift (3-5mm) - early mass effect')
    elif shift_mm >= 1:
        mass_effect_score += 1
        risk_factors.append(f'Midline shift: {shift_mm:.1f}mm (minimal)')
    # Below 1mm: no contribution
    
    # Secondary: Ventricular distortion
    if ventricular_asymmetry > 0.5:
        mass_effect_score += 2
        risk_factors.append(f'Ventricular asymmetry: {ventricular_asymmetry:.2f} (severe)')
        herniation_signs.append('Severe ventricular asymmetry - significant mass effect')
    elif ventricular_asymmetry > 0.3:
        mass_effect_score += 1
        risk_factors.append(f'Ventricular asymmetry: {ventricular_asymmetry:.2f} (moderate)')
    elif ventricular_asymmetry > 0.15:
        mass_effect_score += 0.5
        risk_factors.append(f'Ventricular asymmetry: {ventricular_asymmetry:.2f} (mild)')
    
    # Tertiary: Sulcal effacement
    if sulcal_severity in ['Moderate to Severe', 'Severe']:
        mass_effect_score += 1
        risk_factors.append(f'Sulcal effacement: {sulcal_severity}')
    elif sulcal_severity in ['Mild to Moderate']:
        mass_effect_score += 0.5
        risk_factors.append(f'Sulcal effacement: {sulcal_severity}')
    
    # Determine risk level based on composite score
    if mass_effect_score >= 5:
        risk_level = 'High'
    elif mass_effect_score >= 3:
        risk_level = 'Moderate'
    elif mass_effect_score >= 1.5:
        risk_level = 'Mild'
    else:
        risk_level = 'Low'
    
    # If no significant mass effect, risk is low regardless of tumor size
    if not herniation_signs:
        if tumor_volume_cm3 > 50:
            herniation_signs.append(f'Large tumor ({tumor_volume_cm3:.1f}cm³) without significant mass effect currently')
            # Add monitoring recommendation
            herniation_signs.append('Recommend close monitoring for interval mass effect development')
        else:
            herniation_signs.append('No significant herniation risk - no measurable mass effect')
    
    return {
        'risk_level': risk_level,
        'herniation_signs': herniation_signs,
        'risk_factors': risk_factors,
        'mass_effect_score': float(mass_effect_score),
        'mass_effect_metrics': {
            'midline_shift_mm': float(shift_mm),
            'midline_shift_significant': shift_is_significant,
            'ventricular_asymmetry': float(ventricular_asymmetry),
            'ventricular_severity': ventricular_severity,
            'sulcal_effacement_severity': sulcal_severity
        },
        'tumor_volume_cm3': float(tumor_volume_cm3),
        'clinical_note': 'Risk derived from measurable displacement metrics, not tumor proximity alone'
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
            'severity': 'No tumor detected',
            'details': 'No tumor detected'
        }
    
    # Create expanded region around tumor
    expanded_tumor = ndimage.binary_dilation(tumor_mask, iterations=10)
    peritumoral_region = expanded_tumor & ~tumor_mask & brain_mask
    
    if peritumoral_region.sum() == 0:
        return {
            'effacement_detected': False,
            'severity': 'Could not analyze',
            'details': 'Could not analyze peritumoral region'
        }
    
    distant_brain = brain_mask & ~expanded_tumor
    
    if distant_brain.sum() == 0:
        return {
            'effacement_detected': True,
            'severity': 'Severe',
            'details': 'Tumor occupies majority of brain volume'
        }
    
    peritumoral_std = np.std(brain_data[peritumoral_region])
    distant_std = np.std(brain_data[distant_brain])
    
    variance_ratio = peritumoral_std / distant_std if distant_std > 0 else 1.0
    
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


def determine_anatomical_location(seg_data, voxel_dims):
    """
    Determine detailed anatomical location of tumor.
    
    Returns:
    - Hemisphere (left/right/bilateral)
    - Lobe(s) (frontal, parietal, temporal, occipital)
    - Depth (cortical/superficial, subcortical, deep/periventricular)
    - Approximate gyri (estimated)
    """
    tumor_mask = seg_data > 0
    
    if tumor_mask.sum() == 0:
        return {
            'hemisphere': 'None',
            'laterality': 'N/A',
            'lobes': [],
            'primary_lobe': 'None',
            'depth': 'No tumor detected',
            'approximate_gyri': [],
            'details': 'No tumor present'
        }
    
    dims = seg_data.shape
    tumor_centroid = get_centroid(tumor_mask)
    tumor_bbox = get_bounding_box(tumor_mask)
    total_voxels = tumor_mask.sum()
    
    # Determine hemisphere
    midline_x = dims[0] / 2
    left_voxels = tumor_mask[:int(midline_x), :, :].sum()
    right_voxels = tumor_mask[int(midline_x):, :, :].sum()
    
    if left_voxels > 0.9 * total_voxels:
        hemisphere = 'left'
        laterality = 'Unilateral (left hemisphere)'
    elif right_voxels > 0.9 * total_voxels:
        hemisphere = 'right'
        laterality = 'Unilateral (right hemisphere)'
    elif left_voxels > 0.6 * total_voxels:
        hemisphere = 'left-predominant'
        laterality = 'Bilateral, left-predominant'
    elif right_voxels > 0.6 * total_voxels:
        hemisphere = 'right-predominant'
        laterality = 'Bilateral, right-predominant'
    else:
        hemisphere = 'bilateral'
        laterality = 'Bilateral (crosses midline)'
    
    # Determine lobes based on standard brain atlas approximations
    lobes = []
    lobe_percentages = {}
    lobe_details = {}
    
    # Frontal lobe: anterior 40% of brain (y-axis in standard orientation)
    frontal_mask = np.zeros_like(tumor_mask)
    frontal_mask[:, :int(dims[1]*0.45), int(dims[2]*0.3):] = True
    frontal_overlap = (tumor_mask & frontal_mask).sum()
    if frontal_overlap > 0.05 * total_voxels:
        lobes.append('frontal')
        pct = (frontal_overlap / total_voxels) * 100
        lobe_percentages['frontal'] = pct
        lobe_details['frontal'] = f'{pct:.0f}% of tumor in frontal lobe'
    
    # Parietal lobe: superior-posterior region
    parietal_mask = np.zeros_like(tumor_mask)
    parietal_mask[:, int(dims[1]*0.3):int(dims[1]*0.7), int(dims[2]*0.5):] = True
    parietal_overlap = (tumor_mask & parietal_mask).sum()
    if parietal_overlap > 0.05 * total_voxels:
        lobes.append('parietal')
        pct = (parietal_overlap / total_voxels) * 100
        lobe_percentages['parietal'] = pct
        lobe_details['parietal'] = f'{pct:.0f}% of tumor in parietal lobe'
    
    # Temporal lobe: lateral inferior regions
    temporal_mask = np.zeros_like(tumor_mask)
    temporal_mask[:int(dims[0]*0.35), int(dims[1]*0.2):int(dims[1]*0.7), :int(dims[2]*0.55)] = True
    temporal_mask[int(dims[0]*0.65):, int(dims[1]*0.2):int(dims[1]*0.7), :int(dims[2]*0.55)] = True
    temporal_overlap = (tumor_mask & temporal_mask).sum()
    if temporal_overlap > 0.05 * total_voxels:
        lobes.append('temporal')
        pct = (temporal_overlap / total_voxels) * 100
        lobe_percentages['temporal'] = pct
        lobe_details['temporal'] = f'{pct:.0f}% of tumor in temporal lobe'
    
    # Occipital lobe: posterior region
    occipital_mask = np.zeros_like(tumor_mask)
    occipital_mask[:, int(dims[1]*0.65):, :] = True
    occipital_overlap = (tumor_mask & occipital_mask).sum()
    if occipital_overlap > 0.05 * total_voxels:
        lobes.append('occipital')
        pct = (occipital_overlap / total_voxels) * 100
        lobe_percentages['occipital'] = pct
        lobe_details['occipital'] = f'{pct:.0f}% of tumor in occipital lobe'
    
    # Deep structures: central regions (basal ganglia, thalamus, insula)
    deep_mask = np.zeros_like(tumor_mask)
    x_center = slice(int(dims[0]*0.3), int(dims[0]*0.7))
    y_center = slice(int(dims[1]*0.3), int(dims[1]*0.6))
    z_center = slice(int(dims[2]*0.25), int(dims[2]*0.6))
    deep_mask[x_center, y_center, z_center] = True
    deep_overlap = (tumor_mask & deep_mask).sum()
    if deep_overlap > 0.1 * total_voxels:
        if 'deep structures' not in lobes:
            lobes.append('deep structures')
        pct = (deep_overlap / total_voxels) * 100
        lobe_percentages['deep_structures'] = pct
        lobe_details['deep_structures'] = f'{pct:.0f}% involving deep structures (basal ganglia/thalamus)'
    
    # Primary lobe
    if lobe_percentages:
        primary_lobe = max(lobe_percentages, key=lobe_percentages.get)
        primary_percentage = lobe_percentages[primary_lobe]
    else:
        primary_lobe = 'indeterminate'
        primary_percentage = 0
        lobes = ['location indeterminate']
    
    # Determine depth
    # Check tumor overlap with surface
    if tumor_centroid:
        center = np.array([dims[0]/2, dims[1]/2, dims[2]/2])
        tumor_center = np.array([tumor_centroid['x'], tumor_centroid['y'], tumor_centroid['z']])
        distance_from_center = np.linalg.norm((tumor_center - center) * voxel_dims)
        brain_radius = min(dims) * min(voxel_dims) / 2
        relative_depth = 1 - (distance_from_center / brain_radius)
    else:
        relative_depth = 0.5
    
    # Determine depth category
    if relative_depth > 0.7:
        depth = 'Deep (periventricular/central)'
        depth_detail = 'Tumor located in deep brain structures near ventricles'
    elif relative_depth > 0.4:
        depth = 'Subcortical'
        depth_detail = 'Tumor located in subcortical white matter'
    else:
        depth = 'Cortical/Superficial'
        depth_detail = 'Tumor involves cortical surface or is superficially located'
    
    # Approximate gyri based on lobe and position
    gyri = []
    if 'frontal' in lobes:
        if tumor_centroid and tumor_centroid['z'] > dims[2] * 0.7:
            gyri.append('superior frontal gyrus region')
        elif tumor_centroid and tumor_centroid['z'] > dims[2] * 0.5:
            gyri.append('middle frontal gyrus region')
        else:
            gyri.append('inferior frontal gyrus region')
    
    if 'parietal' in lobes:
        if tumor_centroid and tumor_centroid['z'] > dims[2] * 0.65:
            gyri.append('superior parietal lobule region')
        else:
            gyri.append('inferior parietal lobule region')
    
    if 'temporal' in lobes:
        if tumor_centroid and tumor_centroid['z'] > dims[2] * 0.45:
            gyri.append('superior temporal gyrus region')
        elif tumor_centroid and tumor_centroid['z'] > dims[2] * 0.3:
            gyri.append('middle temporal gyrus region')
        else:
            gyri.append('inferior temporal gyrus region')
    
    if 'occipital' in lobes:
        gyri.append('occipital cortex region')
    
    if not gyri:
        gyri = ['gyral localization not determined']
    
    return {
        'hemisphere': hemisphere,
        'laterality': laterality,
        'lobes': lobes,
        'lobe_percentages': lobe_percentages,
        'lobe_details': lobe_details,
        'primary_lobe': primary_lobe,
        'primary_lobe_percentage': float(primary_percentage) if primary_percentage else 0,
        'depth': depth,
        'depth_detail': depth_detail,
        'relative_depth_score': float(relative_depth),
        'approximate_gyri': gyri,
        'tumor_centroid': tumor_centroid,
        'tumor_bounding_box': tumor_bbox,
        'note': 'Anatomical localization estimated from standard brain atlas coordinates - clinical correlation recommended'
    }


def generate_summary(results):
    """Generate text summary for radiology report."""
    lines = []
    
    lines.append("MASS EFFECT ANALYSIS:")
    lines.append("")
    
    # Anatomical location
    lines.append("Anatomical Location:")
    loc = results['anatomical_location']
    lines.append(f"  - Laterality: {loc['laterality']}")
    lines.append(f"  - Primary lobe: {loc['primary_lobe'].capitalize()} ({loc['primary_lobe_percentage']:.0f}%)")
    if len(loc['lobes']) > 1:
        lines.append(f"  - Additional involvement: {', '.join([l for l in loc['lobes'] if l != loc['primary_lobe']])}")
    lines.append(f"  - Depth: {loc['depth']}")
    if loc['approximate_gyri'] and loc['approximate_gyri'][0] != 'gyral localization not determined':
        lines.append(f"  - Gyri (estimated): {', '.join(loc['approximate_gyri'])}")
    
    # Midline shift
    lines.append("")
    ms = results['midline_shift']
    if ms['is_significant']:
        lines.append(f"Midline Shift: {ms['severity']} ({ms['shift_mm']:.1f} mm)")
        lines.append(f"  - Direction: {ms['shift_direction']}")
    else:
        lines.append("Midline Shift: None")
        lines.append("  - No significant midline shift detected")
    lines.append(f"  - Tumor hemisphere: {ms['tumor_hemisphere']}")
    
    # Ventricular compression
    lines.append("")
    vc = results['ventricular_compression']
    lines.append(f"Ventricular System: {vc['severity']}")
    if vc['compression_detected']:
        lines.append(f"  - Compressed side: {vc['compressed_side']}")
        lines.append(f"  - Asymmetry ratio: {vc['asymmetry_ratio']:.2f}")
    if vc.get('tumor_to_ventricle_distance_mm'):
        lines.append(f"  - Tumor-ventricle distance: {vc['tumor_to_ventricle_distance_mm']:.1f} mm")
    
    # Sulcal effacement
    lines.append("")
    se = results['sulcal_effacement']
    lines.append(f"Sulcal Effacement: {se['severity']}")
    
    # Herniation (risk derived from above metrics)
    lines.append("")
    hr = results['herniation_risk']
    lines.append(f"Herniation Risk Assessment: {hr['risk_level']}")
    lines.append(f"  - Mass effect score: {hr['mass_effect_score']:.1f}/7")
    for sign in hr['herniation_signs']:
        lines.append(f"  - {sign}")
    
    return "\n".join(lines)


def analyze_mass_effect(input_folder, segmentation_path, output_path=None):
    """
    Main function to analyze mass effect metrics.
    """
    input_folder = Path(input_folder)
    case_id = get_case_id(input_folder)
    print(f"Analyzing case: {case_id}")
    
    mri_paths = get_mri_paths(input_folder, case_id)
    
    print("Loading MRI data...")
    t1_data, t1_affine, t1_header = load_nifti(mri_paths['t1'])
    
    print("Loading segmentation mask...")
    seg_data, _, _ = load_nifti(segmentation_path)
    seg_data = np.round(seg_data).astype(np.int32)
    
    voxel_info = get_voxel_dimensions(t1_header)
    voxel_dims = voxel_info['dimensions_mm']
    
    print("\n" + "="*60)
    print("STEP 2: MASS EFFECT METRICS")
    print("="*60)
    
    # Anatomical location (done first for context)
    print("\n--- Anatomical Localization ---")
    anatomical_location = determine_anatomical_location(seg_data, voxel_dims)
    print(f"  Hemisphere: {anatomical_location['laterality']}")
    print(f"  Primary lobe: {anatomical_location['primary_lobe']}")
    if anatomical_location['lobes']:
        print(f"  All lobes: {', '.join(anatomical_location['lobes'])}")
    print(f"  Depth: {anatomical_location['depth']}")
    
    # Midline shift
    print("\n--- Midline Shift Analysis ---")
    midline_results = calculate_midline_shift(t1_data, seg_data, voxel_dims)
    if midline_results['is_significant']:
        print(f"  Shift: {midline_results['shift_mm']:.1f} mm ({midline_results['severity']})")
    else:
        print(f"  No significant midline shift detected")
    
    # Ventricular compression
    print("\n--- Ventricular Compression Analysis ---")
    ventricle_results = analyze_ventricular_compression(t1_data, seg_data, voxel_dims)
    print(f"  Compression: {ventricle_results['severity']}")
    
    # Sulcal effacement
    print("\n--- Sulcal Effacement Analysis ---")
    sulcal_results = analyze_sulcal_effacement(t1_data, seg_data, voxel_dims)
    print(f"  Effacement: {sulcal_results['severity']}")
    
    # Calculate tumor volume for herniation assessment
    tumor_mask = seg_data > 0
    tumor_volume_cm3 = tumor_mask.sum() * np.prod(voxel_dims) / 1000
    
    # Herniation risk (derived from measurable metrics)
    print("\n--- Herniation Risk Assessment ---")
    herniation_results = assess_herniation_risk(
        midline_results, 
        ventricle_results, 
        sulcal_results,
        tumor_volume_cm3,
        anatomical_location
    )
    print(f"  Risk level: {herniation_results['risk_level']}")
    print(f"  Mass effect score: {herniation_results['mass_effect_score']:.1f}/7")
    for sign in herniation_results['herniation_signs']:
        print(f"  - {sign}")
    
    # Compile results
    results = {
        'case_id': case_id,
        'step': 'Step 2 - Mass effect metrics',
        'voxel_info': voxel_info,
        'anatomical_location': anatomical_location,
        'midline_shift': midline_results,
        'ventricular_compression': ventricle_results,
        'sulcal_effacement': sulcal_results,
        'herniation_risk': herniation_results
    }
    
    summary = generate_summary(results)
    results['text_summary'] = summary
    
    print("\n" + "="*60)
    print("TEXT SUMMARY FOR RADIOLOGY REPORT")
    print("="*60)
    print(summary)
    
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
