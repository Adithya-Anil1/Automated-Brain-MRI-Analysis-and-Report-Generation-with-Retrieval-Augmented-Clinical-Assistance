"""
Step 1: Sequence-specific findings

Analyzes MRI sequence characteristics:
- T1, T2, FLAIR intensity behavior in tumor regions
- Contrast enhancement analysis (T1 vs T1ce)
- Signal characteristics relative to normal brain

Author: AI-Powered Brain MRI Assistant
Date: November 27, 2025
"""

import numpy as np
import argparse
from pathlib import Path

from utils import (
    load_nifti, get_intensity_stats, get_normal_brain_stats,
    get_case_id, get_mri_paths, get_voxel_dimensions,
    get_tumor_masks, calculate_volume, save_results
)


def analyze_contrast_enhancement(t1_data, t1ce_data, tumor_masks):
    """
    Analyze contrast enhancement by comparing T1 pre and post contrast.
    Enhancement indicates blood-brain barrier breakdown.
    """
    et_mask = tumor_masks['et']
    tc_mask = tumor_masks['tc']
    wt_mask = tumor_masks['wt']
    
    # Calculate enhancement ratio: T1ce / T1 in tumor regions
    t1_safe = np.where(t1_data > 0, t1_data, 1)
    enhancement_ratio = t1ce_data / t1_safe
    
    results = {
        'enhancing_tumor': {
            't1_stats': get_intensity_stats(t1_data, et_mask),
            't1ce_stats': get_intensity_stats(t1ce_data, et_mask),
            'enhancement_ratio': get_intensity_stats(enhancement_ratio, et_mask) if et_mask.sum() > 0 else None
        },
        'tumor_core': {
            't1_stats': get_intensity_stats(t1_data, tc_mask),
            't1ce_stats': get_intensity_stats(t1ce_data, tc_mask),
            'enhancement_ratio': get_intensity_stats(enhancement_ratio, tc_mask) if tc_mask.sum() > 0 else None
        },
        'whole_tumor': {
            't1_stats': get_intensity_stats(t1_data, wt_mask),
            't1ce_stats': get_intensity_stats(t1ce_data, wt_mask),
            'enhancement_ratio': get_intensity_stats(enhancement_ratio, wt_mask) if wt_mask.sum() > 0 else None
        }
    }
    
    # Determine enhancement pattern
    if et_mask.sum() > 0 and results['enhancing_tumor']['enhancement_ratio']:
        et_enhancement = results['enhancing_tumor']['enhancement_ratio']['mean']
        if et_enhancement > 1.5:
            results['enhancement_pattern'] = 'Strong enhancement'
            results['enhancement_description'] = 'Marked contrast uptake indicating significant BBB breakdown'
        elif et_enhancement > 1.2:
            results['enhancement_pattern'] = 'Moderate enhancement'
            results['enhancement_description'] = 'Moderate contrast uptake suggesting partial BBB disruption'
        elif et_enhancement > 1.0:
            results['enhancement_pattern'] = 'Mild enhancement'
            results['enhancement_description'] = 'Subtle contrast uptake with minimal BBB compromise'
        else:
            results['enhancement_pattern'] = 'No significant enhancement'
            results['enhancement_description'] = 'No appreciable contrast uptake'
    else:
        results['enhancement_pattern'] = 'No enhancing tumor detected'
        results['enhancement_description'] = 'Non-enhancing lesion, may suggest low-grade pathology'
    
    return results


def analyze_t2_flair_behavior(t2_data, flair_data, tumor_masks, seg_data):
    """
    Analyze T2 and FLAIR signal characteristics.
    """
    ed_mask = tumor_masks['ed']
    ncr_mask = tumor_masks['ncr']
    et_mask = tumor_masks['et']
    wt_mask = tumor_masks['wt']
    
    # Get normal brain stats for comparison
    t2_normal = get_normal_brain_stats(t2_data, seg_data)
    flair_normal = get_normal_brain_stats(flair_data, seg_data)
    
    results = {
        'normal_brain': {
            't2': t2_normal,
            'flair': flair_normal
        },
        'edema': {
            't2': get_intensity_stats(t2_data, ed_mask),
            'flair': get_intensity_stats(flair_data, ed_mask)
        },
        'necrotic_core': {
            't2': get_intensity_stats(t2_data, ncr_mask),
            'flair': get_intensity_stats(flair_data, ncr_mask)
        },
        'enhancing_tumor': {
            't2': get_intensity_stats(t2_data, et_mask),
            'flair': get_intensity_stats(flair_data, et_mask)
        },
        'whole_tumor': {
            't2': get_intensity_stats(t2_data, wt_mask),
            'flair': get_intensity_stats(flair_data, wt_mask)
        }
    }
    
    # Characterize signal behavior
    signal_behavior = []
    signal_details = []
    
    # Edema characterization
    if ed_mask.sum() > 0 and t2_normal['mean'] is not None and t2_normal['mean'] > 0:
        ed_t2_ratio = results['edema']['t2']['mean'] / t2_normal['mean']
        ed_flair_ratio = results['edema']['flair']['mean'] / flair_normal['mean']
        
        results['edema']['t2_ratio'] = ed_t2_ratio
        results['edema']['flair_ratio'] = ed_flair_ratio
        
        if ed_t2_ratio > 1.3:
            signal_behavior.append('Edema: T2 hyperintense')
            signal_details.append(f'Peritumoral edema shows {ed_t2_ratio:.1f}x T2 signal vs normal brain')
        if ed_flair_ratio > 1.3:
            signal_behavior.append('Edema: FLAIR hyperintense')
            signal_details.append(f'Peritumoral edema shows {ed_flair_ratio:.1f}x FLAIR signal vs normal brain')
    
    # Enhancing tumor characterization
    if et_mask.sum() > 0 and t2_normal['mean'] is not None and t2_normal['mean'] > 0:
        et_t2_ratio = results['enhancing_tumor']['t2']['mean'] / t2_normal['mean']
        et_flair_ratio = results['enhancing_tumor']['flair']['mean'] / flair_normal['mean']
        
        results['enhancing_tumor']['t2_ratio'] = et_t2_ratio
        results['enhancing_tumor']['flair_ratio'] = et_flair_ratio
        
        if et_t2_ratio > 1.3:
            signal_behavior.append('Enhancing tumor: T2 hyperintense')
        elif et_t2_ratio < 0.7:
            signal_behavior.append('Enhancing tumor: T2 hypointense')
        else:
            signal_behavior.append('Enhancing tumor: T2 isointense')
            
        if et_flair_ratio > 1.3:
            signal_behavior.append('Enhancing tumor: FLAIR hyperintense')
        elif et_flair_ratio < 0.7:
            signal_behavior.append('Enhancing tumor: FLAIR hypointense')
    
    # Necrotic core characterization
    if ncr_mask.sum() > 0 and t2_normal['mean'] is not None and t2_normal['mean'] > 0:
        ncr_t2_ratio = results['necrotic_core']['t2']['mean'] / t2_normal['mean']
        ncr_flair_ratio = results['necrotic_core']['flair']['mean'] / flair_normal['mean']
        
        results['necrotic_core']['t2_ratio'] = ncr_t2_ratio
        results['necrotic_core']['flair_ratio'] = ncr_flair_ratio
        
        if ncr_t2_ratio > 1.5:
            signal_behavior.append('Necrotic core: T2 hyperintense (fluid/necrosis)')
        if ncr_flair_ratio < 0.8:
            signal_behavior.append('Necrotic core: FLAIR hypointense (suggests fluid)')
    
    results['signal_behavior'] = signal_behavior
    results['signal_details'] = signal_details
    
    return results


def analyze_t1_behavior(t1_data, tumor_masks, seg_data):
    """Analyze T1-weighted signal characteristics."""
    t1_normal = get_normal_brain_stats(t1_data, seg_data)
    
    results = {
        'normal_brain': t1_normal
    }
    
    signal_behavior = []
    
    for region_name, mask in [('ncr', tumor_masks['ncr']), 
                               ('et', tumor_masks['et']),
                               ('ed', tumor_masks['ed'])]:
        if mask.sum() > 0:
            stats = get_intensity_stats(t1_data, mask)
            results[region_name] = stats
            
            if t1_normal['mean'] and t1_normal['mean'] > 0:
                ratio = stats['mean'] / t1_normal['mean']
                results[region_name]['ratio_to_normal'] = ratio
                
                region_label = {'ncr': 'Necrotic core', 'et': 'Enhancing tumor', 'ed': 'Edema'}[region_name]
                if ratio > 1.2:
                    signal_behavior.append(f'{region_label}: T1 hyperintense')
                elif ratio < 0.8:
                    signal_behavior.append(f'{region_label}: T1 hypointense')
                else:
                    signal_behavior.append(f'{region_label}: T1 isointense')
    
    results['signal_behavior'] = signal_behavior
    return results


def generate_summary(results):
    """Generate text summary for radiology report."""
    lines = []
    
    lines.append("SEQUENCE-SPECIFIC FINDINGS:")
    lines.append("")
    
    # Enhancement pattern
    enhancement = results['contrast_enhancement']
    lines.append(f"Contrast Enhancement: {enhancement['enhancement_pattern']}")
    lines.append(f"  {enhancement['enhancement_description']}")
    
    if enhancement['enhancing_tumor']['enhancement_ratio']:
        ratio = enhancement['enhancing_tumor']['enhancement_ratio']['mean']
        lines.append(f"  Enhancement ratio: {ratio:.2f}x (T1ce/T1)")
    
    # T2/FLAIR behavior
    lines.append("")
    lines.append("T2/FLAIR Signal Characteristics:")
    for behavior in results['t2_flair_analysis']['signal_behavior']:
        lines.append(f"  - {behavior}")
    
    for detail in results['t2_flair_analysis'].get('signal_details', []):
        lines.append(f"  - {detail}")
    
    # T1 behavior
    lines.append("")
    lines.append("T1 Signal Characteristics:")
    for behavior in results['t1_analysis']['signal_behavior']:
        lines.append(f"  - {behavior}")
    
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
    
    # Get tumor masks
    tumor_masks = get_tumor_masks(seg_data)
    
    print("\n" + "="*60)
    print("STEP 1: SEQUENCE-SPECIFIC FINDINGS")
    print("="*60)
    
    # Analyze contrast enhancement
    print("\n--- Contrast Enhancement Analysis ---")
    enhancement_results = analyze_contrast_enhancement(t1_data, t1ce_data, tumor_masks)
    print(f"Enhancement pattern: {enhancement_results['enhancement_pattern']}")
    
    # Analyze T2/FLAIR
    print("\n--- T2/FLAIR Signal Analysis ---")
    t2_flair_results = analyze_t2_flair_behavior(t2_data, flair_data, tumor_masks, seg_data)
    for behavior in t2_flair_results['signal_behavior']:
        print(f"  - {behavior}")
    
    # Analyze T1
    print("\n--- T1 Signal Analysis ---")
    t1_results = analyze_t1_behavior(t1_data, tumor_masks, seg_data)
    for behavior in t1_results['signal_behavior']:
        print(f"  - {behavior}")
    
    # Calculate volumes
    volumes = {
        'Whole Tumor (WT)': calculate_volume(tumor_masks['wt'], voxel_info['volume_cm3']),
        'Tumor Core (TC)': calculate_volume(tumor_masks['tc'], voxel_info['volume_cm3']),
        'Enhancing Tumor (ET)': calculate_volume(tumor_masks['et'], voxel_info['volume_cm3']),
        'Necrotic Core (NCR)': calculate_volume(tumor_masks['ncr'], voxel_info['volume_cm3']),
        'Edema (ED)': calculate_volume(tumor_masks['ed'], voxel_info['volume_cm3'])
    }
    
    print("\n--- Tumor Volumes ---")
    for region, vol in volumes.items():
        if vol > 0:
            print(f"  {region}: {vol:.2f} cm³")
    
    # Compile results
    results = {
        'case_id': case_id,
        'step': 'Step 1 - Sequence-specific findings',
        'voxel_info': voxel_info,
        'contrast_enhancement': enhancement_results,
        't2_flair_analysis': t2_flair_results,
        't1_analysis': t1_results,
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
