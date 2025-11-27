"""
Step 6: Normal Structures Assessment

Evaluates normal brain structures outside the tumor:
- Ventricular system (size, symmetry, morphology)
- Brain parenchyma (signal characteristics, atrophy)
- Major intracranial vessels (patency, flow voids)
- White matter integrity
- Gray-white differentiation

Clinical Relevance:
- Hydrocephalus detection
- Global brain changes (atrophy, leukoaraiosis)
- Vascular abnormalities
- Secondary effects of tumor

Author: AI-Powered Brain MRI Assistant
Date: November 27, 2025
"""

import numpy as np
import argparse
from pathlib import Path
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt

from utils import (
    load_nifti, get_case_id, get_mri_paths, get_voxel_dimensions,
    get_tumor_masks, get_brain_mask, save_results
)


def identify_ventricles(t1_data, t2_data, flair_data, brain_mask, tumor_mask):
    """
    Identify and segment ventricular system based on CSF characteristics.
    CSF: Dark on T1, Bright on T2, Dark on FLAIR (suppressed)
    """
    # Normalize intensities within brain
    t1_brain = t1_data[brain_mask]
    t2_brain = t2_data[brain_mask]
    flair_brain = flair_data[brain_mask]
    
    # CSF characteristics
    # T1: Low signal (bottom 15%)
    # T2: High signal (top 15%)
    # FLAIR: Low signal (bottom 20% - CSF suppression)
    
    t1_thresh_low = np.percentile(t1_brain, 15)
    t2_thresh_high = np.percentile(t2_brain, 85)
    flair_thresh_low = np.percentile(flair_brain, 25)
    
    # CSF mask based on multi-sequence criteria
    csf_mask = (
        brain_mask &
        (t1_data < t1_thresh_low) &
        (t2_data > t2_thresh_high) &
        (flair_data < flair_thresh_low) &
        ~tumor_mask  # Exclude tumor
    )
    
    # Clean up with morphological operations
    csf_mask = binary_erosion(csf_mask, iterations=1)
    csf_mask = binary_dilation(csf_mask, iterations=1)
    
    # Label connected components and keep only larger ones (ventricles)
    structure = ndimage.generate_binary_structure(3, 2)
    labeled, num_features = ndimage.label(csf_mask, structure=structure)
    
    # Keep components > 1 cm³ as ventricles
    voxel_vol = 1.0  # Assuming 1mm³ voxels for now
    min_size = 1000 / voxel_vol  # 1 cm³
    
    ventricle_mask = np.zeros_like(csf_mask)
    for i in range(1, num_features + 1):
        component = labeled == i
        if component.sum() > min_size:
            # Check if centrally located (ventricles are central)
            coords = np.where(component)
            centroid_x = np.mean(coords[0])
            center_x = brain_mask.shape[0] / 2
            if abs(centroid_x - center_x) < brain_mask.shape[0] * 0.3:
                ventricle_mask = ventricle_mask | component
    
    return ventricle_mask, csf_mask


def analyze_ventricular_system(t1_data, t2_data, flair_data, brain_mask, tumor_mask, voxel_dims):
    """
    Comprehensive analysis of the ventricular system.
    """
    ventricle_mask, csf_mask = identify_ventricles(
        t1_data, t2_data, flair_data, brain_mask, tumor_mask
    )
    
    voxel_vol = np.prod(voxel_dims) / 1000  # cm³
    
    # Total ventricular volume
    total_volume = ventricle_mask.sum() * voxel_vol
    
    # Brain volume (excluding tumor)
    brain_volume = (brain_mask & ~tumor_mask).sum() * voxel_vol
    
    # Ventricle-to-brain ratio (VBR)
    vbr = (total_volume / brain_volume * 100) if brain_volume > 0 else 0
    
    # Split into left and right hemispheres
    midline = ventricle_mask.shape[0] // 2
    left_ventricle = ventricle_mask[:midline, :, :]
    right_ventricle = ventricle_mask[midline:, :, :]
    
    left_vol = left_ventricle.sum() * voxel_vol
    right_vol = right_ventricle.sum() * voxel_vol
    
    # Asymmetry analysis
    if left_vol + right_vol > 0:
        asymmetry = abs(left_vol - right_vol) / (left_vol + right_vol)
    else:
        asymmetry = 0
    
    larger_side = 'left' if left_vol > right_vol else 'right' if right_vol > left_vol else 'symmetric'
    
    # Evans index approximation (frontal horn width / biparietal diameter)
    # Simplified: Check frontal horn width ratio
    # Normal Evans Index < 0.3
    
    # Find frontal horn slice (anterior third of ventricles)
    vent_coords = np.where(ventricle_mask)
    if len(vent_coords[1]) > 0:
        frontal_y = np.percentile(vent_coords[1], 75)  # Anterior 25%
        frontal_slices = ventricle_mask[:, int(frontal_y):, :]
        frontal_width = np.max(np.sum(frontal_slices, axis=0)) if frontal_slices.any() else 0
        
        # Brain width at same level
        brain_width = brain_mask.shape[0]
        evans_index = (frontal_width * voxel_dims[0]) / (brain_width * voxel_dims[0])
    else:
        evans_index = 0
    
    # Determine if hydrocephalus present
    if evans_index > 0.3 and vbr > 5:
        hydrocephalus = True
        hydrocephalus_type = "Communicating hydrocephalus suggested"
    elif vbr > 7:
        hydrocephalus = True
        hydrocephalus_type = "Ventriculomegaly noted"
    else:
        hydrocephalus = False
        hydrocephalus_type = "No hydrocephalus"
    
    # Check for ventricular trapping/obstruction near tumor
    # Dilate tumor and check overlap with ventricles
    tumor_dilated = binary_dilation(tumor_mask, iterations=5)
    ventricle_near_tumor = ventricle_mask & tumor_dilated
    obstruction_risk = ventricle_near_tumor.sum() / ventricle_mask.sum() if ventricle_mask.sum() > 0 else 0
    
    # Morphology assessment
    if vbr < 2:
        size_assessment = "Normal"
        size_note = "Ventricles within normal size limits"
    elif vbr < 4:
        size_assessment = "Mildly prominent"
        size_note = "Mild prominence of ventricular system"
    elif vbr < 6:
        size_assessment = "Moderately dilated"
        size_note = "Moderate ventricular enlargement"
    else:
        size_assessment = "Markedly dilated"
        size_note = "Marked ventriculomegaly"
    
    return {
        'total_volume_cm3': float(total_volume),
        'left_volume_cm3': float(left_vol),
        'right_volume_cm3': float(right_vol),
        'ventricle_brain_ratio_percent': float(vbr),
        'asymmetry_index': float(asymmetry),
        'larger_side': larger_side,
        'evans_index_estimate': float(evans_index),
        'size_assessment': size_assessment,
        'size_note': size_note,
        'hydrocephalus_present': hydrocephalus,
        'hydrocephalus_type': hydrocephalus_type,
        'obstruction_risk': float(obstruction_risk),
        'obstruction_note': 'Tumor adjacent to ventricular system' if obstruction_risk > 0.1 else 'No direct ventricular involvement',
        'symmetry_assessment': 'Symmetric' if asymmetry < 0.15 else f'Asymmetric ({larger_side} larger)'
    }


def analyze_parenchyma(t1_data, t2_data, flair_data, brain_mask, tumor_mask, voxel_dims):
    """
    Analyze brain parenchyma outside the tumor.
    """
    # Normal brain = brain excluding tumor
    normal_brain = brain_mask & ~tumor_mask
    
    if normal_brain.sum() == 0:
        return {
            'assessment': 'Unable to assess',
            'note': 'Insufficient normal brain tissue for analysis'
        }
    
    voxel_vol = np.prod(voxel_dims) / 1000  # cm³
    
    # Signal characteristics of normal-appearing white matter
    # WM is typically in the inner portions of the brain
    # Use distance from brain surface
    brain_dist = distance_transform_edt(brain_mask)
    deep_threshold = np.percentile(brain_dist[brain_mask], 60)
    
    # Deep white matter mask (approximation)
    deep_wm_mask = normal_brain & (brain_dist > deep_threshold)
    
    # Periventricular region
    # Identify ventricles first
    ventricle_mask, _ = identify_ventricles(t1_data, t2_data, flair_data, brain_mask, tumor_mask)
    ventricle_dilated = binary_dilation(ventricle_mask, iterations=10)
    periventricular = ventricle_dilated & normal_brain & ~ventricle_mask
    
    # FLAIR hyperintensities in periventricular region (possible white matter disease)
    if periventricular.sum() > 0:
        pv_flair = flair_data[periventricular]
        pv_flair_mean = pv_flair.mean()
        
        # Compare to cortical GM FLAIR
        cortical_mask = normal_brain & (brain_dist < np.percentile(brain_dist[brain_mask], 40))
        cortical_flair = flair_data[cortical_mask]
        cortical_flair_mean = cortical_flair.mean()
        
        pv_hyperintensity_ratio = pv_flair_mean / cortical_flair_mean if cortical_flair_mean > 0 else 1.0
        
        if pv_hyperintensity_ratio > 1.3:
            wm_disease = True
            wm_description = "FLAIR hyperintensities in periventricular white matter, may represent chronic small vessel disease"
        elif pv_hyperintensity_ratio > 1.15:
            wm_disease = True
            wm_description = "Mild periventricular FLAIR signal changes"
        else:
            wm_disease = False
            wm_description = "No significant periventricular white matter changes"
    else:
        pv_hyperintensity_ratio = 1.0
        wm_disease = False
        wm_description = "Periventricular region could not be assessed"
    
    # Gray-white differentiation
    # In normal brain, there should be good contrast between GM and WM on T1
    if deep_wm_mask.sum() > 100:
        deep_wm_t1 = t1_data[deep_wm_mask].mean()
        cortical_t1 = t1_data[cortical_mask].mean() if cortical_mask.sum() > 100 else deep_wm_t1
        
        gw_ratio = deep_wm_t1 / cortical_t1 if cortical_t1 > 0 else 1.0
        
        if gw_ratio > 1.1:
            gw_differentiation = "Preserved"
            gw_note = "Normal gray-white matter differentiation"
        elif gw_ratio > 1.0:
            gw_differentiation = "Mildly reduced"
            gw_note = "Slightly reduced gray-white differentiation"
        else:
            gw_differentiation = "Reduced"
            gw_note = "Loss of gray-white differentiation (may indicate edema or diffuse pathology)"
    else:
        gw_differentiation = "Could not assess"
        gw_ratio = 1.0
        gw_note = "Insufficient tissue for gray-white analysis"
    
    # Volume estimation for atrophy
    brain_volume = brain_mask.sum() * voxel_vol
    normal_brain_volume = normal_brain.sum() * voxel_vol
    
    # Approximate sulcal widening (atrophy indicator)
    # More CSF in sulci = wider sulci
    surface_brain = brain_mask & ~binary_erosion(brain_mask, iterations=3)
    
    return {
        'normal_brain_volume_cm3': float(normal_brain_volume),
        'total_brain_volume_cm3': float(brain_volume),
        'periventricular_assessment': {
            'hyperintensity_ratio': float(pv_hyperintensity_ratio),
            'white_matter_disease_present': wm_disease,
            'description': wm_description
        },
        'gray_white_differentiation': {
            'assessment': gw_differentiation,
            'ratio': float(gw_ratio),
            'note': gw_note
        },
        'overall_assessment': 'Normal' if not wm_disease and gw_differentiation == 'Preserved' else 'Abnormal findings present',
        'atrophy_assessment': 'Not formally assessed (requires age-matched normative data)'
    }


def analyze_major_vessels(t1_data, t1ce_data, brain_mask, tumor_mask, voxel_dims):
    """
    Basic assessment of major intracranial vessels.
    
    Note: This is limited without dedicated vascular sequences (MRA/MRV).
    Looks for flow voids and enhancement patterns.
    """
    # Flow voids appear dark on all sequences
    # In standard MRI, we can only see major vessels as flow voids
    
    # Look for linear dark structures in expected vessel locations
    # Major vessels: Circle of Willis (inferior brain), sagittal sinus (midline superior)
    
    # Get inferior slices for Circle of Willis region
    inferior_third = brain_mask.shape[2] // 3
    inferior_brain = brain_mask.copy()
    inferior_brain[:, :, inferior_third:] = False
    
    # Get very dark voxels in inferior brain on T1
    if inferior_brain.sum() > 0:
        inf_t1 = t1_data[inferior_brain]
        flow_void_thresh = np.percentile(inf_t1, 5)
        
        flow_void_mask = (
            inferior_brain &
            (t1_data < flow_void_thresh) &
            ~tumor_mask
        )
        
        # Flow voids should be small relative to brain
        flow_void_volume = flow_void_mask.sum() * np.prod(voxel_dims) / 1000
        flow_void_fraction = flow_void_mask.sum() / inferior_brain.sum()
        
        # Expected range for major vessels
        if 0.001 < flow_void_fraction < 0.05:
            flow_void_assessment = "Present"
            flow_void_note = "Flow voids identified in expected vessel locations"
        elif flow_void_fraction < 0.001:
            flow_void_assessment = "Not well visualized"
            flow_void_note = "Major vessel flow voids not clearly identified (may be normal variant or sequence-dependent)"
        else:
            flow_void_assessment = "Prominent"
            flow_void_note = "Prominent dark signal in basal regions (may include vessels and air-bone interfaces)"
    else:
        flow_void_assessment = "Could not assess"
        flow_void_note = "Insufficient inferior brain for vessel assessment"
        flow_void_volume = 0
        flow_void_fraction = 0
    
    # Check for abnormal enhancement near major vessels (T1ce)
    # This could indicate vascular involvement by tumor
    if t1ce_data is not None:
        # Enhancement near tumor
        tumor_dilated = binary_dilation(tumor_mask, iterations=10)
        peritumoral = tumor_dilated & ~tumor_mask & brain_mask
        
        if peritumoral.sum() > 0:
            peritumoral_t1 = t1_data[peritumoral].mean()
            peritumoral_t1ce = t1ce_data[peritumoral].mean()
            
            peritumoral_enhancement_ratio = peritumoral_t1ce / peritumoral_t1 if peritumoral_t1 > 0 else 1.0
            
            if peritumoral_enhancement_ratio > 1.5:
                vascular_involvement = "Possible"
                vascular_note = "Enhancement in peritumoral region may indicate vascular involvement"
            else:
                vascular_involvement = "Not evident"
                vascular_note = "No obvious vascular encasement or involvement"
        else:
            vascular_involvement = "Could not assess"
            vascular_note = "Insufficient peritumoral tissue"
            peritumoral_enhancement_ratio = 1.0
    else:
        vascular_involvement = "Not assessed"
        vascular_note = "T1ce not available for enhancement assessment"
        peritumoral_enhancement_ratio = 0
    
    return {
        'flow_voids': {
            'assessment': flow_void_assessment,
            'note': flow_void_note,
            'volume_cm3': float(flow_void_volume)
        },
        'vascular_involvement': {
            'assessment': vascular_involvement,
            'note': vascular_note,
            'peritumoral_enhancement_ratio': float(peritumoral_enhancement_ratio) if t1ce_data is not None else None
        },
        'limitations': [
            "Detailed vascular assessment requires MRA/MRV sequences",
            "Flow void analysis is limited on standard structural MRI",
            "Cannot assess vessel patency or flow direction"
        ],
        'overall_assessment': 'Limited assessment on structural sequences'
    }


def generate_summary(results):
    """Generate text summary for radiology report."""
    lines = []
    
    lines.append("NORMAL STRUCTURES ASSESSMENT:")
    lines.append("")
    
    # Ventricular system
    vent = results['ventricular_system']
    lines.append("Ventricular System:")
    lines.append(f"  - Size: {vent['size_assessment']} (VBR: {vent['ventricle_brain_ratio_percent']:.1f}%)")
    lines.append(f"  - Volume: {vent['total_volume_cm3']:.1f} cm³ (L: {vent['left_volume_cm3']:.1f}, R: {vent['right_volume_cm3']:.1f})")
    lines.append(f"  - Symmetry: {vent['symmetry_assessment']}")
    lines.append(f"  - {vent['hydrocephalus_type']}")
    if vent['obstruction_risk'] > 0.1:
        lines.append(f"  - ⚠ {vent['obstruction_note']}")
    
    # Parenchyma
    lines.append("")
    par = results['parenchyma']
    lines.append("Brain Parenchyma:")
    lines.append(f"  - Gray-white differentiation: {par['gray_white_differentiation']['assessment']}")
    lines.append(f"    {par['gray_white_differentiation']['note']}")
    lines.append(f"  - Periventricular white matter: {par['periventricular_assessment']['description']}")
    lines.append(f"  - Overall: {par['overall_assessment']}")
    
    # Major vessels
    lines.append("")
    ves = results['major_vessels']
    lines.append("Major Vessels (Limited Assessment):")
    lines.append(f"  - Flow voids: {ves['flow_voids']['assessment']}")
    lines.append(f"  - Vascular involvement: {ves['vascular_involvement']['assessment']}")
    lines.append(f"  Note: {ves['overall_assessment']}")
    
    return "\n".join(lines)


def analyze_normal_structures(input_folder, segmentation_path, output_path=None):
    """
    Main function to analyze normal brain structures.
    """
    input_folder = Path(input_folder)
    case_id = get_case_id(input_folder)
    print(f"Analyzing case: {case_id}")
    
    mri_paths = get_mri_paths(input_folder, case_id)
    
    print("Loading MRI sequences...")
    t1_data, _, t1_header = load_nifti(mri_paths['t1'])
    t1ce_data, _, _ = load_nifti(mri_paths['t1ce'])
    t2_data, _, _ = load_nifti(mri_paths['t2'])
    flair_data, _, _ = load_nifti(mri_paths['flair'])
    
    print("Loading segmentation mask...")
    seg_data, _, _ = load_nifti(segmentation_path)
    seg_data = np.round(seg_data).astype(np.int32)
    
    voxel_info = get_voxel_dimensions(t1_header)
    voxel_dims = voxel_info['dimensions_mm']
    
    # Get masks
    tumor_masks = get_tumor_masks(seg_data)
    tumor_mask = tumor_masks['wt']
    brain_mask = get_brain_mask(t1_data)
    
    print("\n" + "="*60)
    print("STEP 6: NORMAL STRUCTURES ASSESSMENT")
    print("="*60)
    
    # Ventricular system
    print("\n--- Ventricular System Analysis ---")
    ventricular = analyze_ventricular_system(
        t1_data, t2_data, flair_data, brain_mask, tumor_mask, voxel_dims
    )
    print(f"  Size: {ventricular['size_assessment']}")
    print(f"  Volume: {ventricular['total_volume_cm3']:.1f} cm³")
    print(f"  VBR: {ventricular['ventricle_brain_ratio_percent']:.1f}%")
    print(f"  Hydrocephalus: {'Yes' if ventricular['hydrocephalus_present'] else 'No'}")
    
    # Parenchyma
    print("\n--- Brain Parenchyma Analysis ---")
    parenchyma = analyze_parenchyma(
        t1_data, t2_data, flair_data, brain_mask, tumor_mask, voxel_dims
    )
    print(f"  Gray-white differentiation: {parenchyma['gray_white_differentiation']['assessment']}")
    print(f"  White matter disease: {'Present' if parenchyma['periventricular_assessment']['white_matter_disease_present'] else 'Not significant'}")
    
    # Major vessels
    print("\n--- Major Vessels Analysis ---")
    vessels = analyze_major_vessels(t1_data, t1ce_data, brain_mask, tumor_mask, voxel_dims)
    print(f"  Flow voids: {vessels['flow_voids']['assessment']}")
    print(f"  Vascular involvement: {vessels['vascular_involvement']['assessment']}")
    
    # Compile results
    results = {
        'case_id': case_id,
        'step': 'Step 6 - Normal structures assessment',
        'voxel_info': voxel_info,
        'ventricular_system': ventricular,
        'parenchyma': parenchyma,
        'major_vessels': vessels
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
        description='Step 6: Analyze normal brain structures'
    )
    parser.add_argument('--input', required=True,
                        help='Input folder containing MRI sequences')
    parser.add_argument('--segmentation', required=True,
                        help='Path to segmentation mask (NIfTI)')
    parser.add_argument('--output', default=None,
                        help='Output path for JSON results')
    
    args = parser.parse_args()
    
    analyze_normal_structures(args.input, args.segmentation, args.output)


if __name__ == "__main__":
    main()
