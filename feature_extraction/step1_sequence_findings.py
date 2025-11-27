"""
Step 1: Sequence-specific findings

Analyzes MRI sequence characteristics:
- Semantic signal labels (hypo/iso/hyperintense) for each tumor region on each sequence
- Intensity ratios relative to normal brain tissue for all sequences
- T1, T1ce, T2, FLAIR intensity behavior in tumor regions
- Contrast enhancement analysis (T1 vs T1ce)

NORMAL BRAIN REFERENCE METHODOLOGY:
- Reference: All non-tumor brain tissue (combined gray matter + white matter)
- Excludes: Tumor regions (labels 1,2,3), CSF, and background voxels
- Selection: Voxels with intensity > 5th percentile of non-zero brain values
- Rationale: Combined GM+WM provides a stable reference that accounts for
  partial volume effects and is robust across different MRI acquisitions.
  Using both tissues together avoids the need for tissue segmentation and
  provides consistent reference values across different scan parameters.

SEMANTIC SIGNAL LABELS:
- markedly hypointense: ratio < 0.6 (>40% darker than normal brain)
- hypointense: ratio 0.6-0.85 (15-40% darker)
- isointense: ratio 0.85-1.15 (within 15% of normal)
- hyperintense: ratio 1.15-1.5 (15-50% brighter)
- markedly hyperintense: ratio > 1.5 (>50% brighter)

Author: AI-Powered Brain MRI Assistant
Date: November 27, 2025
"""

import numpy as np
import argparse
from pathlib import Path

from utils import (
    load_nifti, get_intensity_stats, get_normal_brain_stats,
    get_case_id, get_mri_paths, get_voxel_dimensions, get_acquisition_details,
    get_tumor_masks, calculate_volume, save_results
)


def get_signal_label(ratio):
    """
    Convert intensity ratio to semantic signal label.
    
    Args:
        ratio: Intensity ratio (tumor region mean / normal brain mean)
    
    Returns:
        str: Semantic label
    """
    if ratio < 0.6:
        return "markedly hypointense"
    elif ratio < 0.85:
        return "hypointense"
    elif ratio < 1.15:
        return "isointense"
    elif ratio < 1.5:
        return "hyperintense"
    else:
        return "markedly hyperintense"


def get_signal_summary(t1_label, t2_label, flair_label, t1ce_label=None):
    """Generate a one-line signal summary for a region."""
    parts = [f"T1 {t1_label}", f"T2 {t2_label}", f"FLAIR {flair_label}"]
    if t1ce_label:
        parts.append(f"T1ce {t1ce_label}")
    return ", ".join(parts)


def analyze_region_signals(region_name, region_mask, 
                           t1_data, t2_data, flair_data, t1ce_data,
                           normal_t1, normal_t2, normal_flair, normal_t1ce):
    """
    Analyze signal characteristics for a specific tumor region across all sequences.
    
    Returns complete intensity metrics, ratios, and semantic labels for each sequence.
    """
    if region_mask.sum() == 0:
        return None
    
    # Get intensity stats for this region in each sequence
    t1_stats = get_intensity_stats(t1_data, region_mask)
    t2_stats = get_intensity_stats(t2_data, region_mask)
    flair_stats = get_intensity_stats(flair_data, region_mask)
    t1ce_stats = get_intensity_stats(t1ce_data, region_mask)
    
    # Calculate ratios relative to normal brain
    t1_ratio = t1_stats['mean'] / normal_t1['mean'] if normal_t1['mean'] and normal_t1['mean'] > 0 else 1.0
    t2_ratio = t2_stats['mean'] / normal_t2['mean'] if normal_t2['mean'] and normal_t2['mean'] > 0 else 1.0
    flair_ratio = flair_stats['mean'] / normal_flair['mean'] if normal_flair['mean'] and normal_flair['mean'] > 0 else 1.0
    t1ce_ratio = t1ce_stats['mean'] / normal_t1ce['mean'] if normal_t1ce['mean'] and normal_t1ce['mean'] > 0 else 1.0
    
    # Get semantic labels
    t1_label = get_signal_label(t1_ratio)
    t2_label = get_signal_label(t2_ratio)
    flair_label = get_signal_label(flair_ratio)
    t1ce_label = get_signal_label(t1ce_ratio)
    
    # Enhancement ratio (T1ce vs T1 pre-contrast)
    enhancement_ratio = t1ce_stats['mean'] / t1_stats['mean'] if t1_stats['mean'] and t1_stats['mean'] > 0 else 1.0
    
    return {
        'region': region_name,
        'voxel_count': int(region_mask.sum()),
        'T1': {
            'mean_intensity': float(t1_stats['mean']),
            'std': float(t1_stats['std']),
            'ratio_to_normal': round(float(t1_ratio), 3),
            'signal_label': t1_label
        },
        'T2': {
            'mean_intensity': float(t2_stats['mean']),
            'std': float(t2_stats['std']),
            'ratio_to_normal': round(float(t2_ratio), 3),
            'signal_label': t2_label
        },
        'FLAIR': {
            'mean_intensity': float(flair_stats['mean']),
            'std': float(flair_stats['std']),
            'ratio_to_normal': round(float(flair_ratio), 3),
            'signal_label': flair_label
        },
        'T1ce': {
            'mean_intensity': float(t1ce_stats['mean']),
            'std': float(t1ce_stats['std']),
            'ratio_to_normal': round(float(t1ce_ratio), 3),
            'signal_label': t1ce_label,
            'enhancement_ratio': round(float(enhancement_ratio), 3)
        },
        'signal_summary': get_signal_summary(t1_label, t2_label, flair_label, t1ce_label)
    }


def analyze_all_region_signals(t1_data, t2_data, flair_data, t1ce_data, tumor_masks, seg_data):
    """
    Analyze signal characteristics for all tumor regions.
    
    Returns comprehensive signal analysis with ratios and semantic labels.
    """
    # Get normal brain stats for all sequences
    normal_t1 = get_normal_brain_stats(t1_data, seg_data)
    normal_t2 = get_normal_brain_stats(t2_data, seg_data)
    normal_flair = get_normal_brain_stats(flair_data, seg_data)
    normal_t1ce = get_normal_brain_stats(t1ce_data, seg_data)
    
    results = {
        'normal_brain_reference': {
            'methodology': 'Combined gray matter + white matter (non-tumor, non-CSF brain tissue)',
            'T1_mean': normal_t1['mean'],
            'T2_mean': normal_t2['mean'],
            'FLAIR_mean': normal_flair['mean'],
            'T1ce_mean': normal_t1ce['mean'],
            'voxel_count': normal_t1['voxel_count']
        },
        'regions': {}
    }
    
    # Analyze each region
    region_labels = {
        'ncr': 'Necrotic Core (NCR)',
        'ed': 'Peritumoral Edema (ED)', 
        'et': 'Enhancing Tumor (ET)'
    }
    
    for key, display_name in region_labels.items():
        mask = tumor_masks[key]
        region_analysis = analyze_region_signals(
            display_name, mask,
            t1_data, t2_data, flair_data, t1ce_data,
            normal_t1, normal_t2, normal_flair, normal_t1ce
        )
        if region_analysis:
            results['regions'][key] = region_analysis
    
    return results


def analyze_contrast_enhancement(t1_data, t1ce_data, tumor_masks, region_signals):
    """
    Analyze contrast enhancement patterns using pre-computed region signals.
    """
    et_mask = tumor_masks['et']
    ncr_mask = tumor_masks['ncr']
    
    results = {
        'enhancement_present': bool(et_mask.sum() > 0),
        'pattern': None,
        'heterogeneity': None,
        'metrics': {}
    }
    
    if not results['enhancement_present']:
        results['pattern'] = 'Non-enhancing'
        results['heterogeneity'] = 'Not applicable'
        results['description'] = 'Non-enhancing pattern can be seen with lower-grade glioma, treatment effect, or other pathology; clinical and histopathological correlation required'
        return results
    
    # Get enhancement metrics from region signals
    et_signals = region_signals['regions'].get('et', {})
    if et_signals:
        enhancement_ratio = et_signals['T1ce'].get('enhancement_ratio', 1.0)
        results['metrics']['enhancement_ratio_T1ce_over_T1'] = enhancement_ratio
        results['metrics']['T1ce_ratio_to_normal'] = et_signals['T1ce']['ratio_to_normal']
        
        # Get heterogeneity from standard deviation
        t1ce_mean = et_signals['T1ce']['mean_intensity']
        t1ce_std = et_signals['T1ce']['std']
        if t1ce_mean > 0:
            cv = t1ce_std / t1ce_mean
            results['metrics']['coefficient_of_variation'] = round(float(cv), 3)
            
            if cv > 0.35:
                results['heterogeneity'] = 'Markedly heterogeneous'
            elif cv > 0.25:
                results['heterogeneity'] = 'Heterogeneous'
            elif cv > 0.15:
                results['heterogeneity'] = 'Mildly heterogeneous'
            else:
                results['heterogeneity'] = 'Homogeneous'
    
    # Determine enhancement pattern (ring vs solid)
    if ncr_mask.sum() > 0 and et_mask.sum() > 0:
        from scipy.ndimage import binary_dilation
        dilated_ncr = binary_dilation(ncr_mask, iterations=2)
        enhancement_around_ncr = np.logical_and(dilated_ncr, et_mask).sum()
        
        if enhancement_around_ncr > 0.3 * et_mask.sum():
            results['pattern'] = 'Ring-enhancing'
            results['description'] = 'Peripheral rim enhancement surrounding central non-enhancing core, characteristic of high-grade glioma or metastasis'
        else:
            results['pattern'] = 'Solid/nodular enhancing'
            results['description'] = 'Solid pattern of enhancement without central necrosis'
    else:
        results['pattern'] = 'Solid/nodular enhancing'
        results['description'] = 'Solid pattern of enhancement without central necrosis'
    
    # Add enhancement strength description
    if 'enhancement_ratio_T1ce_over_T1' in results['metrics']:
        ratio = results['metrics']['enhancement_ratio_T1ce_over_T1']
        if ratio > 2.0:
            results['enhancement_strength'] = 'Marked enhancement'
        elif ratio > 1.5:
            results['enhancement_strength'] = 'Strong enhancement'
        elif ratio > 1.2:
            results['enhancement_strength'] = 'Moderate enhancement'
        elif ratio > 1.05:
            results['enhancement_strength'] = 'Mild enhancement'
        else:
            results['enhancement_strength'] = 'Minimal/equivocal enhancement'
    
    return results


def detect_t2_flair_mismatch(region_signals):
    """
    Detect T2/FLAIR mismatch sign (suggestive of IDH-mutant glioma).
    
    T2/FLAIR mismatch: Region that is hyperintense on T2 but relatively 
    hypointense on FLAIR compared to surrounding edema.
    """
    results = {
        'mismatch_detected': False,
        'description': None
    }
    
    # Check each region for mismatch
    for key, region in region_signals['regions'].items():
        t2_ratio = region['T2']['ratio_to_normal']
        flair_ratio = region['FLAIR']['ratio_to_normal']
        
        # Mismatch: T2 hyperintense (>1.3) but FLAIR relatively lower (ratio < 0.8 of T2 ratio)
        if t2_ratio > 1.3 and flair_ratio < t2_ratio * 0.7:
            results['mismatch_detected'] = True
            results['region'] = key
            results['t2_ratio'] = t2_ratio
            results['flair_ratio'] = flair_ratio
            results['description'] = f"Possible T2/FLAIR mismatch in {region['region']}: T2 hyperintense (ratio {t2_ratio:.2f}) with relatively suppressed FLAIR (ratio {flair_ratio:.2f}). May suggest IDH-mutant lower-grade glioma."
            break
    
    if not results['mismatch_detected']:
        results['description'] = "No T2/FLAIR mismatch detected. Signal intensity patterns concordant between T2 and FLAIR sequences."
    
    return results


def generate_summary(results):
    """Generate text summary for radiology report."""
    lines = []
    
    # CLINICAL CONTEXT - Placeholders to prevent LLM fabrication
    lines.append("CLINICAL INFORMATION:")
    lines.append("  Patient age: <not provided>")
    lines.append("  Patient sex: <not provided>")
    lines.append("  Clinical history: <not provided>")
    lines.append("  Presenting symptoms: <not provided>")
    lines.append("  [Note: Do not fabricate - include only if provided in clinical records]")
    lines.append("")
    
    # TECHNIQUE SECTION - Always include at the top
    lines.append("TECHNIQUE:")
    technique = results.get('technique', {})
    
    # Sequences performed
    sequences = technique.get('sequences_performed', [])
    if sequences:
        lines.append(f"  Sequences performed: {', '.join(sequences)}")
    else:
        lines.append("  Sequences performed: <not provided>")
    
    # Contrast
    if technique.get('contrast_administered', False):
        lines.append(f"  Contrast: Administered ({technique.get('contrast_note', 'Gadolinium-based')})")
    else:
        lines.append("  Contrast: Not administered or not available")
    
    # Acquisition details
    acq = technique.get('acquisition_parameters', {})
    if acq:
        slice_thick = acq.get('slice_thickness_mm', 'N/A')
        in_plane = acq.get('in_plane_resolution_mm', ('N/A', 'N/A'))
        matrix = acq.get('matrix_size', ('N/A', 'N/A', 'N/A'))
        lines.append(f"  Slice thickness: {slice_thick} mm")
        lines.append(f"  In-plane resolution: {in_plane[0]:.2f} × {in_plane[1]:.2f} mm")
        lines.append(f"  Matrix size: {matrix[0]} × {matrix[1]} × {matrix[2]}")
    
    # Not available sequences
    not_avail = technique.get('sequences_not_available', [])
    if not_avail:
        lines.append(f"  Not available: {', '.join(not_avail)}")
    
    lines.append("")
    lines.append("SEQUENCE-SPECIFIC FINDINGS:")
    lines.append("")
    lines.append("Reference: Normal brain tissue (combined GM+WM, excluding tumor and CSF)")
    lines.append("")
    
    # Signal characteristics for each region
    lines.append("Signal Characteristics by Region:")
    region_signals = results['region_signal_analysis']['regions']
    
    for key in ['ncr', 'ed', 'et']:
        if key in region_signals:
            region = region_signals[key]
            lines.append(f"  {region['region']}:")
            lines.append(f"    {region['signal_summary']}")
            lines.append(f"    Ratios - T1: {region['T1']['ratio_to_normal']:.2f}, T2: {region['T2']['ratio_to_normal']:.2f}, FLAIR: {region['FLAIR']['ratio_to_normal']:.2f}, T1ce: {region['T1ce']['ratio_to_normal']:.2f}")
    
    # Enhancement pattern
    lines.append("")
    enhancement = results['contrast_enhancement']
    lines.append(f"Contrast Enhancement: {enhancement['pattern']}")
    if 'enhancement_strength' in enhancement:
        lines.append(f"  Strength: {enhancement['enhancement_strength']}")
    if 'heterogeneity' in enhancement and enhancement['heterogeneity']:
        lines.append(f"  Heterogeneity: {enhancement['heterogeneity']}")
    if 'description' in enhancement:
        lines.append(f"  {enhancement['description']}")
    
    # T2/FLAIR mismatch
    lines.append("")
    mismatch = results['t2_flair_mismatch']
    lines.append(f"T2/FLAIR Mismatch: {'Present' if mismatch['mismatch_detected'] else 'Not detected'}")
    lines.append(f"  {mismatch['description']}")
    
    # Volume summary
    lines.append("")
    lines.append("Tumor Volumes:")
    for region, vol in results['volumes'].items():
        if vol > 0:
            lines.append(f"  - {region}: {vol:.2f} cm³")
    
    # Note about DWI
    lines.append("")
    lines.append("Note: Diffusion-weighted imaging (DWI/ADC) not available in standard BraTS dataset")
    
    return "\n".join(lines)


def analyze_sequence_findings(input_folder, segmentation_path, output_path=None):
    """
    Main function to analyze all sequence-specific findings.
    """
    input_folder = Path(input_folder)
    case_id = get_case_id(input_folder)
    print(f"Analyzing case: {case_id}")
    
    # Get MRI paths
    mri_paths = get_mri_paths(input_folder, case_id)
    
    # Load all sequences
    print("Loading MRI sequences...")
    t1_data, t1_affine, t1_header = load_nifti(mri_paths['t1'])
    t1ce_data, _, _ = load_nifti(mri_paths['t1ce'])
    t2_data, _, _ = load_nifti(mri_paths['t2'])
    flair_data, _, _ = load_nifti(mri_paths['flair'])
    
    # Load segmentation
    print("Loading segmentation mask...")
    seg_data, _, _ = load_nifti(segmentation_path)
    seg_data = np.round(seg_data).astype(np.int32)
    
    # Get voxel info
    voxel_info = get_voxel_dimensions(t1_header)
    
    # Get acquisition details for technique section
    acquisition_details = get_acquisition_details(t1_header)
    
    # Build technique section - document exactly what sequences are available
    sequences_available = []
    sequences_detail = {}
    contrast_administered = False
    
    for seq_name, seq_path in mri_paths.items():
        if seq_path.exists():
            seq_upper = seq_name.upper()
            sequences_available.append(seq_upper)
            
            # Load header for each sequence to get its specific details
            _, _, seq_header = load_nifti(seq_path)
            seq_acq = get_acquisition_details(seq_header)
            
            sequences_detail[seq_upper] = {
                'available': True,
                'file': seq_path.name,
                'slice_thickness_mm': seq_acq['slice_thickness_mm'],
                'in_plane_resolution_mm': seq_acq['in_plane_resolution_mm'],
                'matrix_size': seq_acq['matrix_size'],
                'num_slices': seq_acq['num_slices']
            }
            
            # Check if contrast was used (T1ce present indicates contrast)
            if seq_name.lower() == 't1ce':
                contrast_administered = True
        else:
            sequences_detail[seq_name.upper()] = {
                'available': False,
                'file': None
            }
    
    # Add DWI/ADC status (not in BraTS)
    sequences_detail['DWI'] = {'available': False, 'note': 'Not included in BraTS dataset'}
    sequences_detail['ADC'] = {'available': False, 'note': 'Not included in BraTS dataset'}
    
    technique_section = {
        'sequences_performed': sequences_available,
        'sequences_detail': sequences_detail,
        'contrast_administered': contrast_administered,
        'contrast_note': 'Gadolinium-based contrast agent (inferred from T1ce sequence presence)' if contrast_administered else 'No post-contrast imaging available',
        'acquisition_parameters': {
            'slice_thickness_mm': acquisition_details['slice_thickness_mm'],
            'in_plane_resolution_mm': acquisition_details['in_plane_resolution_mm'],
            'voxel_size_mm': acquisition_details['voxel_size_mm'],
            'matrix_size': acquisition_details['matrix_size'],
            'num_slices': acquisition_details['num_slices']
        },
        'sequences_not_available': ['DWI', 'ADC', 'MRS', 'Perfusion'],
        'note': 'Acquisition parameters extracted from NIfTI headers; original scanner parameters may differ'
    }
    
    # Get tumor masks
    tumor_masks = get_tumor_masks(seg_data)
    
    print("\n" + "="*60)
    print("STEP 1: SEQUENCE-SPECIFIC FINDINGS")
    print("="*60)
    
    # Analyze all region signals with semantic labels and ratios
    print("\n--- Regional Signal Analysis ---")
    region_signals = analyze_all_region_signals(
        t1_data, t2_data, flair_data, t1ce_data, tumor_masks, seg_data
    )
    
    print(f"\nNormal brain reference (GM+WM combined):")
    print(f"  T1 mean: {region_signals['normal_brain_reference']['T1_mean']:.1f}")
    print(f"  T2 mean: {region_signals['normal_brain_reference']['T2_mean']:.1f}")
    print(f"  FLAIR mean: {region_signals['normal_brain_reference']['FLAIR_mean']:.1f}")
    print(f"  T1ce mean: {region_signals['normal_brain_reference']['T1ce_mean']:.1f}")
    print(f"  Voxel count: {region_signals['normal_brain_reference']['voxel_count']:,}")
    
    print("\nSignal characteristics by region:")
    for key in ['ncr', 'ed', 'et']:
        if key in region_signals['regions']:
            region = region_signals['regions'][key]
            print(f"\n  {region['region']}:")
            print(f"    {region['signal_summary']}")
            print(f"    T1 ratio: {region['T1']['ratio_to_normal']:.3f}, T2 ratio: {region['T2']['ratio_to_normal']:.3f}")
            print(f"    FLAIR ratio: {region['FLAIR']['ratio_to_normal']:.3f}, T1ce ratio: {region['T1ce']['ratio_to_normal']:.3f}")
            print(f"    Enhancement ratio (T1ce/T1): {region['T1ce']['enhancement_ratio']:.3f}")
    
    # Analyze contrast enhancement
    print("\n--- Contrast Enhancement Analysis ---")
    enhancement_results = analyze_contrast_enhancement(
        t1_data, t1ce_data, tumor_masks, region_signals
    )
    print(f"  Pattern: {enhancement_results['pattern']}")
    if 'enhancement_strength' in enhancement_results:
        print(f"  Strength: {enhancement_results['enhancement_strength']}")
    if 'heterogeneity' in enhancement_results:
        print(f"  Heterogeneity: {enhancement_results['heterogeneity']}")
    
    # T2/FLAIR mismatch
    print("\n--- T2/FLAIR Mismatch Analysis ---")
    mismatch_results = detect_t2_flair_mismatch(region_signals)
    print(f"  {mismatch_results['description']}")
    
    # Calculate volumes
    volumes = {
        'Whole Tumor (WT)': calculate_volume(tumor_masks['wt'], voxel_info['volume_cm3']),
        'Tumor Core (TC)': calculate_volume(tumor_masks['tc'], voxel_info['volume_cm3']),
        'Enhancing Tumor (ET)': calculate_volume(tumor_masks['et'], voxel_info['volume_cm3']),
        'Necrotic Core (NCR)': calculate_volume(tumor_masks['ncr'], voxel_info['volume_cm3']),
        'Peritumoral Edema (ED)': calculate_volume(tumor_masks['ed'], voxel_info['volume_cm3'])
    }
    
    print("\n--- Tumor Volumes ---")
    for region, vol in volumes.items():
        if vol > 0:
            print(f"  {region}: {vol:.2f} cm³")
    
    # Compile results
    results = {
        'case_id': case_id,
        'step': 'Step 1 - Sequence-specific findings',
        'technique': technique_section,
        'voxel_info': voxel_info,
        'region_signal_analysis': region_signals,
        'contrast_enhancement': enhancement_results,
        't2_flair_mismatch': mismatch_results,
        'volumes': volumes,
        'sequences_analyzed': ['T1', 'T1ce', 'T2', 'FLAIR'],
        'diffusion_available': False,
        'diffusion_note': 'DWI/ADC not available in standard BraTS dataset'
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
        description='Step 1: Extract sequence-specific findings from brain MRI'
    )
    parser.add_argument('--input', required=True, 
                        help='Input folder containing MRI sequences')
    parser.add_argument('--segmentation', required=True,
                        help='Path to segmentation mask (NIfTI)')
    parser.add_argument('--output', default=None,
                        help='Output path for JSON results')
    
    args = parser.parse_args()
    
    analyze_sequence_findings(args.input, args.segmentation, args.output)


if __name__ == "__main__":
    main()
