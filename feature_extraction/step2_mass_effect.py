"""
Step 2: Mass Effect Metrics

Analyzes mass effect indicators:
- Midline shift calculation
- Ventricular compression analysis
- Herniation signs detection
- Sulcal effacement
- Anatomical context (lobe, depth, hemisphere)

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


def determine_anatomical_context(brain_data, seg_data, voxel_dims):
    """
    Determine anatomical context of the tumor:
    - Lobe (frontal, parietal, temporal, occipital)
    - Depth (cortical/superficial, subcortical, periventricular/deep)
    - Hemisphere (left, right, bilateral)
    - Approximate gyri based on location
    
    Uses anatomical landmarks based on standard brain atlas proportions.
    """
    tumor_mask = seg_data > 0
    brain_mask = get_brain_mask(brain_data)
    
    if tumor_mask.sum() == 0:
        return {
            'hemisphere': None,
            'lobe': None,
            'depth': None,
            'gyri': None,
            'details': 'No tumor detected'
        }
    
    dims = brain_data.shape
    
    # Get tumor centroid and bounding box
    tumor_coords = np.where(tumor_mask)
    tumor_centroid = {
        'x': tumor_coords[0].mean(),
        'y': tumor_coords[1].mean(),
        'z': tumor_coords[2].mean()
    }
    
    # Determine hemisphere
    # X-axis: left-right (0 = left, max = right in radiological convention)
    midline_x = dims[0] // 2
    left_tumor_voxels = ((tumor_coords[0] < midline_x)).sum()
    right_tumor_voxels = ((tumor_coords[0] >= midline_x)).sum()
    total_tumor_voxels = tumor_mask.sum()
    
    if left_tumor_voxels > 0.8 * total_tumor_voxels:
        hemisphere = 'Left'
    elif right_tumor_voxels > 0.8 * total_tumor_voxels:
        hemisphere = 'Right'
    else:
        hemisphere = 'Bilateral (crossing midline)'
    
    # Determine lobe based on Y (anterior-posterior) and Z (inferior-superior) position
    # Y-axis: anterior (0) to posterior (max)
    # Z-axis: inferior (0) to superior (max)
    y_normalized = tumor_centroid['y'] / dims[1]  # 0 = most anterior, 1 = most posterior
    z_normalized = tumor_centroid['z'] / dims[2]  # 0 = most inferior, 1 = most superior
    
    # Lobe determination using anatomical proportions
    # Frontal: anterior 40%, superior 60%
    # Parietal: middle-posterior (40-70%), superior 60%
    # Temporal: middle (30-70% AP), inferior 40%
    # Occipital: posterior 30%, any height
    
    lobe_candidates = []
    
    if y_normalized < 0.4 and z_normalized > 0.4:
        lobe_candidates.append('Frontal')
    if 0.3 <= y_normalized <= 0.7 and z_normalized > 0.5:
        lobe_candidates.append('Parietal')
    if 0.3 <= y_normalized <= 0.7 and z_normalized < 0.5:
        lobe_candidates.append('Temporal')
    if y_normalized > 0.7:
        lobe_candidates.append('Occipital')
    
    # If tumor spans multiple regions, list primary
    if not lobe_candidates:
        # Default based on closest match
        if y_normalized < 0.5:
            lobe = 'Frontal' if z_normalized > 0.4 else 'Temporal'
        else:
            lobe = 'Parietal' if z_normalized > 0.5 else 'Occipital'
    elif len(lobe_candidates) == 1:
        lobe = lobe_candidates[0]
    else:
        lobe = f"{lobe_candidates[0]} (extending into {', '.join(lobe_candidates[1:])})"
    
    # Determine depth (superficial vs deep)
    # Calculate distance from brain surface
    brain_coords = np.where(brain_mask)
    if len(brain_coords[0]) > 0:
        # Find brain boundaries
        brain_x_min, brain_x_max = brain_coords[0].min(), brain_coords[0].max()
        brain_y_min, brain_y_max = brain_coords[1].min(), brain_coords[1].max()
        brain_z_min, brain_z_max = brain_coords[2].min(), brain_coords[2].max()
        
        # Calculate tumor distance from all brain surfaces
        tumor_bbox = get_bounding_box(tumor_mask)
        
        # Distance to nearest cortical surface (in voxels)
        dist_to_surfaces = [
            tumor_bbox['min_x'] - brain_x_min,  # Left surface
            brain_x_max - tumor_bbox['max_x'],  # Right surface
            tumor_bbox['min_y'] - brain_y_min,  # Anterior surface
            brain_y_max - tumor_bbox['max_y'],  # Posterior surface
            brain_z_max - tumor_bbox['max_z']   # Superior surface
        ]
        min_surface_dist = min(dist_to_surfaces) * min(voxel_dims)  # Convert to mm
        
        # Also check proximity to ventricles (center of brain)
        center_x = (brain_x_min + brain_x_max) / 2
        center_y = (brain_y_min + brain_y_max) / 2
        center_z = (brain_z_min + brain_z_max) / 2
        
        dist_to_center = np.sqrt(
            ((tumor_centroid['x'] - center_x) * voxel_dims[0])**2 +
            ((tumor_centroid['y'] - center_y) * voxel_dims[1])**2 +
            ((tumor_centroid['z'] - center_z) * voxel_dims[2])**2
        )
        
        # Classify depth
        brain_radius = min(brain_x_max - brain_x_min, 
                          brain_y_max - brain_y_min) * min(voxel_dims) / 2
        
        if min_surface_dist < 10:  # Within 10mm of cortical surface
            depth = 'Cortical (superficial)'
        elif dist_to_center < brain_radius * 0.3:  # Within 30% of center
            depth = 'Periventricular (deep)'
        else:
            depth = 'Subcortical'
    else:
        depth = 'Unable to determine'
    
    # Determine approximate gyri based on lobe and position
    gyri = []
    if 'Frontal' in lobe:
        if tumor_centroid['z'] > dims[2] * 0.7:
            gyri.append('Superior frontal gyrus')
        elif tumor_centroid['z'] > dims[2] * 0.5:
            gyri.append('Middle frontal gyrus')
        else:
            gyri.append('Inferior frontal gyrus')
        if tumor_centroid['y'] < dims[1] * 0.2:
            gyri.append('Precentral gyrus region')
    
    if 'Parietal' in lobe:
        if tumor_centroid['z'] > dims[2] * 0.7:
            gyri.append('Superior parietal lobule')
        else:
            gyri.append('Inferior parietal lobule')
        if tumor_centroid['y'] < dims[1] * 0.5:
            gyri.append('Postcentral gyrus region')
    
    if 'Temporal' in lobe:
        if tumor_centroid['z'] > dims[2] * 0.35:
            gyri.append('Superior temporal gyrus')
        elif tumor_centroid['z'] > dims[2] * 0.25:
            gyri.append('Middle temporal gyrus')
        else:
            gyri.append('Inferior temporal gyrus')
    
    if 'Occipital' in lobe:
        gyri.append('Occipital cortex')
        if tumor_centroid['z'] > dims[2] * 0.5:
            gyri.append('Cuneus region')
        else:
            gyri.append('Lingual gyrus region')
    
    if not gyri:
        gyri = ['Location requires clinical correlation']
    
    return {
        'hemisphere': hemisphere,
        'lobe': lobe,
        'depth': depth,
        'gyri': gyri,
        'tumor_centroid_normalized': {
            'x': float(tumor_centroid['x'] / dims[0]),
            'y': float(tumor_centroid['y'] / dims[1]),
            'z': float(tumor_centroid['z'] / dims[2])
        },
        'note': 'Anatomical location estimated from tumor position - clinical correlation recommended'
    }


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
    
    # Clinical severity classification
    # Below 3mm is considered within measurement noise and clinically insignificant
    if shift_mm < 3:
        severity = 'None'
        clinical = 'No significant midline shift detected'
        # Direction is irrelevant at negligible shifts
        shift_direction = 'None (below measurement threshold)'
    elif shift_mm < 5:
        severity = 'Mild'
        clinical = 'Mild midline shift, typically not requiring urgent intervention'
        if estimated_shift > 0:
            shift_direction = 'Right to left' if tumor_side == 'right' else 'Left to right'
        else:
            shift_direction = 'Left to right' if tumor_side == 'right' else 'Right to left'
    elif shift_mm < 10:
        severity = 'Moderate'
        clinical = 'Moderate midline shift, close monitoring recommended'
        if estimated_shift > 0:
            shift_direction = 'Right to left' if tumor_side == 'right' else 'Left to right'
        else:
            shift_direction = 'Left to right' if tumor_side == 'right' else 'Right to left'
    else:
        severity = 'Severe'
        clinical = 'Severe midline shift, may require urgent intervention'
        if estimated_shift > 0:
            shift_direction = 'Right to left' if tumor_side == 'right' else 'Left to right'
        else:
            shift_direction = 'Left to right' if tumor_side == 'right' else 'Right to left'
    
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


def detect_herniation_signs(brain_data, seg_data, voxel_dims, midline_results=None, ventricle_results=None, sulcal_results=None):
    """
    Detect signs of brain herniation based on objective mass effect measurements.
    
    Herniation risk is calculated from:
    - Actual midline shift magnitude
    - Ventricular compression severity
    - Sulcal effacement
    - Tumor location (secondary factor)
    
    Types of herniation:
    - Subfalcine: midline shift > 5mm
    - Transtentorial: significant mass effect in supratentorial compartment
    - Tonsillar: posterior fossa mass with significant compression
    - Uncal: temporal lobe mass with brainstem compression signs
    
    Risk is NOT determined by tumor proximity alone - must have measurable displacement.
    """
    tumor_mask = seg_data > 0
    brain_mask = get_brain_mask(brain_data)
    
    herniation_signs = []
    risk_factors = []
    risk_score = 0  # Accumulative score based on objective findings
    
    if tumor_mask.sum() == 0:
        return {
            'herniation_signs': [],
            'risk_level': 'None',
            'risk_factors': [],
            'details': 'No tumor detected'
        }
    
    # Get tumor location and volume
    tumor_bbox = get_bounding_box(tumor_mask)
    tumor_centroid = get_centroid(tumor_mask)
    dims = brain_data.shape
    midline_x = dims[0] // 2
    tumor_distance_to_midline = abs(tumor_centroid['x'] - midline_x) * voxel_dims[0]
    tumor_volume = tumor_mask.sum() * np.prod(voxel_dims) / 1000  # cm³
    
    # ========================================================================
    # CRITICAL: Herniation risk is based on OBJECTIVE MEASUREMENTS
    # ========================================================================
    
    # Factor 1: Midline shift (most important indicator)
    if midline_results:
        shift_mm = midline_results.get('shift_mm')
        if shift_mm is None:
            shift_mm = 0
        shift_severity = midline_results.get('severity', 'None')
        
        if shift_mm >= 10:
            risk_score += 4
            risk_factors.append(f'Severe midline shift: {shift_mm:.1f} mm')
            herniation_signs.append('Subfalcine herniation - significant midline displacement')
        elif shift_mm >= 5:
            risk_score += 2
            risk_factors.append(f'Moderate midline shift: {shift_mm:.1f} mm')
            herniation_signs.append('Subfalcine herniation risk - measurable midline shift > 5mm')
        elif shift_mm >= 3:
            risk_score += 1
            risk_factors.append(f'Mild midline shift: {shift_mm:.1f} mm')
        # Note: shift < 3mm does NOT contribute to herniation risk
    
    # Factor 2: Ventricular compression
    if ventricle_results:
        vent_severity = ventricle_results.get('severity', 'None/Minimal')
        asymmetry = ventricle_results.get('asymmetry_ratio')
        if asymmetry is None:
            asymmetry = 0
        
        if vent_severity == 'Severe' or asymmetry > 0.5:
            risk_score += 3
            risk_factors.append(f'Severe ventricular compression (asymmetry: {asymmetry:.2f})')
        elif vent_severity == 'Moderate' or asymmetry > 0.3:
            risk_score += 2
            risk_factors.append(f'Moderate ventricular compression (asymmetry: {asymmetry:.2f})')
        elif vent_severity == 'Mild' or asymmetry > 0.15:
            risk_score += 1
            risk_factors.append(f'Mild ventricular compression (asymmetry: {asymmetry:.2f})')
    
    # Factor 3: Sulcal effacement
    if sulcal_results:
        sulcal_severity = sulcal_results.get('severity', 'None/Minimal')
        
        if 'Severe' in sulcal_severity:
            risk_score += 2
            risk_factors.append('Severe sulcal effacement')
        elif 'Moderate' in sulcal_severity:
            risk_score += 1
            risk_factors.append('Moderate sulcal effacement')
    
    # Factor 4: Tumor location (contributory but NOT sufficient alone)
    # Location only adds to risk if there are OTHER mass effect signs
    z_threshold = dims[2] * 0.3
    y_threshold = dims[1] * 0.7
    
    # Posterior fossa (tonsillar herniation risk)
    posterior_fossa_tumor = (tumor_centroid['y'] > y_threshold) and (tumor_centroid['z'] < z_threshold)
    if posterior_fossa_tumor and tumor_volume > 10:
        # Only add risk if there's actual mass effect
        if risk_score > 0:
            risk_score += 1
            herniation_signs.append('Tonsillar herniation concern - posterior fossa location with mass effect')
        else:
            # Note location but don't escalate risk without mass effect
            risk_factors.append('Posterior fossa location (monitor for mass effect)')
    
    # Temporal lobe (uncal herniation risk)
    lateral_threshold = dims[0] * 0.3
    temporal_position = (tumor_centroid['x'] < lateral_threshold or 
                         tumor_centroid['x'] > dims[0] - lateral_threshold)
    middle_z = dims[2] * 0.3 < tumor_centroid['z'] < dims[2] * 0.6
    
    if temporal_position and middle_z and tumor_volume > 25:
        if risk_score > 0:
            risk_score += 1
            herniation_signs.append('Uncal herniation concern - temporal location with mass effect')
        else:
            risk_factors.append('Temporal lobe location (monitor for mass effect)')
    
    # Lower brain regions (transtentorial risk)
    tumor_in_lower_brain = tumor_bbox['min_z'] < z_threshold
    if tumor_in_lower_brain and tumor_volume > 20 and risk_score >= 2:
        herniation_signs.append('Transtentorial herniation concern - inferior mass with significant effect')
    
    # ========================================================================
    # DETERMINE FINAL RISK LEVEL FROM OBJECTIVE SCORE
    # ========================================================================
    # Risk is derived from measurable findings, NOT tumor proximity alone
    
    if risk_score >= 6:
        risk_level = 'High'
    elif risk_score >= 3:
        risk_level = 'Moderate'
    elif risk_score >= 1:
        risk_level = 'Low'
    else:
        risk_level = 'Very Low'
        herniation_signs.append('No significant mass effect to suggest herniation')
    
    # Add explanatory note if tumor is large but mass effect is minimal
    if tumor_volume > 30 and risk_score < 2:
        risk_factors.append(f'Note: Large tumor ({tumor_volume:.1f} cm³) with minimal measured mass effect')
    
    if not herniation_signs:
        herniation_signs.append('No herniation signs based on current measurements')
    
    return {
        'herniation_signs': herniation_signs,
        'risk_level': risk_level,
        'risk_score': risk_score,
        'risk_factors': risk_factors,
        'tumor_volume_cm3': float(tumor_volume),
        'tumor_location': {
            'centroid': tumor_centroid,
            'bounding_box': tumor_bbox,
            'distance_to_midline_mm': float(tumor_distance_to_midline)
        },
        'calculation_method': 'Derived from objective measurements (midline shift, ventricular compression, sulcal effacement)',
        'note': 'Risk calculated from measurable displacement, not tumor proximity alone'
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
    
    # Anatomical Context
    if 'anatomical_context' in results:
        ac = results['anatomical_context']
        lines.append("Anatomical Location:")
        if ac.get('hemisphere'):
            lines.append(f"  - Hemisphere: {ac['hemisphere']}")
        if ac.get('lobe'):
            lines.append(f"  - Lobe: {ac['lobe']}")
        if ac.get('depth'):
            lines.append(f"  - Depth: {ac['depth']}")
        if ac.get('gyri') and ac['gyri']:
            lines.append(f"  - Approximate region: {', '.join(ac['gyri'])}")
        lines.append("")
    
    # Midline shift
    ms = results['midline_shift']
    lines.append(f"Midline Shift: {ms['severity']}")
    if ms['shift_mm'] is not None:
        if ms['shift_mm'] < 3:
            # For negligible shifts, don't report direction
            lines.append(f"  - {ms.get('clinical_significance', 'No significant midline shift detected')}")
        else:
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
    
    # Sulcal effacement
    lines.append("")
    se = results['sulcal_effacement']
    lines.append(f"Sulcal Effacement: {se['severity']}")
    
    # Herniation - now with risk factors
    lines.append("")
    hr = results['herniation_risk']
    risk_score = hr.get('risk_score', '')
    if risk_score:
        lines.append(f"Herniation Risk: {hr['risk_level']} (objective score: {risk_score})")
    else:
        lines.append(f"Herniation Risk: {hr['risk_level']}")
    
    for sign in hr['herniation_signs']:
        lines.append(f"  - {sign}")
    
    # Show risk factors if present
    if hr.get('risk_factors'):
        lines.append("  Contributing factors:")
        for factor in hr['risk_factors']:
            lines.append(f"    • {factor}")
    
    # Add note about calculation method
    if hr.get('calculation_method'):
        lines.append("")
        lines.append(f"Note: {hr['calculation_method']}")
    
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
    
    # Analyze anatomical context first
    print("\n--- Anatomical Context ---")
    anatomical_context = determine_anatomical_context(t1_data, seg_data, voxel_dims)
    print(f"  Hemisphere: {anatomical_context['hemisphere']}")
    print(f"  Lobe: {anatomical_context['lobe']}")
    print(f"  Depth: {anatomical_context['depth']}")
    if anatomical_context['gyri']:
        print(f"  Gyri: {', '.join(anatomical_context['gyri'])}")
    
    # Calculate midline shift
    print("\n--- Midline Shift Analysis ---")
    midline_results = calculate_midline_shift(t1_data, seg_data, voxel_dims)
    print(f"  Severity: {midline_results['severity']}")
    if midline_results['shift_mm']:
        if midline_results['shift_mm'] < 3:
            print(f"  {midline_results['clinical_significance']}")
        else:
            print(f"  Estimated shift: {midline_results['shift_mm']:.1f} mm")
    
    # Analyze ventricular compression
    print("\n--- Ventricular Compression Analysis ---")
    ventricle_results = analyze_ventricular_compression(t1_data, seg_data, voxel_dims)
    print(f"  Compression: {ventricle_results['severity']}")
    if ventricle_results['compression_detected']:
        print(f"  Compressed side: {ventricle_results['compressed_side']}")
    
    # Analyze sulcal effacement
    print("\n--- Sulcal Effacement Analysis ---")
    sulcal_results = analyze_sulcal_effacement(t1_data, seg_data, voxel_dims)
    print(f"  Effacement: {sulcal_results['severity']}")
    
    # Detect herniation signs - NOW BASED ON OBJECTIVE MEASUREMENTS
    print("\n--- Herniation Risk Assessment ---")
    herniation_results = detect_herniation_signs(
        t1_data, seg_data, voxel_dims,
        midline_results=midline_results,
        ventricle_results=ventricle_results,
        sulcal_results=sulcal_results
    )
    print(f"  Risk level: {herniation_results['risk_level']} (score: {herniation_results.get('risk_score', 'N/A')})")
    for sign in herniation_results['herniation_signs']:
        print(f"  - {sign}")
    if herniation_results.get('risk_factors'):
        print("  Risk factors considered:")
        for factor in herniation_results['risk_factors']:
            print(f"    • {factor}")
    
    # Compile results
    results = {
        'case_id': case_id,
        'step': 'Step 2 - Mass effect metrics',
        'voxel_info': voxel_info,
        'anatomical_context': anatomical_context,
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
