"""
MRI Feature Extraction for Brain Tumor Analysis
Step 1: Sequence-specific findings
- T1, T2, FLAIR intensity behavior
- Contrast enhancement analysis
- Diffusion restriction (if DWI available)

Author: AI-Powered Brain MRI Assistant
Date: November 27, 2025
"""

import numpy as np
import nibabel as nib
import os
import argparse
from pathlib import Path
import json
from scipy import ndimage


def load_nifti(filepath):
    """Load a NIfTI file and return data array and affine."""
    img = nib.load(filepath)
    return img.get_fdata(), img.affine, img.header


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
    # Assuming data > 0 is brain tissue
    brain_mask = data > np.percentile(data[data > 0], 5) if data.max() > 0 else data > 0
    normal_mask = brain_mask & (seg_mask == 0)
    
    return get_intensity_stats(data, normal_mask)


def analyze_contrast_enhancement(t1_data, t1ce_data, seg_mask):
    """
    Analyze contrast enhancement by comparing T1 pre and post contrast.
    Enhancement indicates blood-brain barrier breakdown.
    """
    # Get enhancing tumor region (label 3 in BraTS 2025, label 4 in BraTS 2021)
    et_mask = (seg_mask == 3) | (seg_mask == 4)
    
    # Get tumor core (NCR + ET)
    tc_mask = (seg_mask == 1) | (seg_mask == 3) | (seg_mask == 4)
    
    # Get whole tumor
    wt_mask = seg_mask > 0
    
    # Calculate enhancement ratio: T1ce / T1 in tumor regions
    # Avoid division by zero
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
    if et_mask.sum() > 0:
        et_enhancement = results['enhancing_tumor']['enhancement_ratio']['mean']
        if et_enhancement > 1.5:
            results['enhancement_pattern'] = 'Strong enhancement'
        elif et_enhancement > 1.2:
            results['enhancement_pattern'] = 'Moderate enhancement'
        elif et_enhancement > 1.0:
            results['enhancement_pattern'] = 'Mild enhancement'
        else:
            results['enhancement_pattern'] = 'No significant enhancement'
    else:
        results['enhancement_pattern'] = 'No enhancing tumor detected'
    
    return results


def analyze_t2_flair_behavior(t2_data, flair_data, seg_mask):
    """
    Analyze T2 and FLAIR signal characteristics.
    - Edema typically hyperintense on T2 and FLAIR
    - Tumor core may show variable signal
    """
    # Region masks
    ed_mask = seg_mask == 2  # Edema
    ncr_mask = seg_mask == 1  # Necrotic core
    et_mask = (seg_mask == 3) | (seg_mask == 4)  # Enhancing tumor
    wt_mask = seg_mask > 0  # Whole tumor
    
    # Get normal brain stats for comparison
    t2_normal = get_normal_brain_stats(t2_data, seg_mask)
    flair_normal = get_normal_brain_stats(flair_data, seg_mask)
    
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
    
    # Edema characterization
    if ed_mask.sum() > 0 and t2_normal['mean'] is not None:
        ed_t2_ratio = results['edema']['t2']['mean'] / t2_normal['mean']
        ed_flair_ratio = results['edema']['flair']['mean'] / flair_normal['mean']
        
        if ed_t2_ratio > 1.3:
            signal_behavior.append('Edema: T2 hyperintense')
        if ed_flair_ratio > 1.3:
            signal_behavior.append('Edema: FLAIR hyperintense')
    
    # Enhancing tumor characterization
    if et_mask.sum() > 0 and t2_normal['mean'] is not None:
        et_t2_ratio = results['enhancing_tumor']['t2']['mean'] / t2_normal['mean']
        et_flair_ratio = results['enhancing_tumor']['flair']['mean'] / flair_normal['mean']
        
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
    
    results['signal_behavior'] = signal_behavior
    
    return results


def analyze_sequence_findings(input_folder, segmentation_path, output_path=None):
    """
    Main function to analyze all sequence-specific findings.
    
    Args:
        input_folder: Folder containing T1, T1ce, T2, FLAIR NIfTI files
        segmentation_path: Path to segmentation mask
        output_path: Optional path to save JSON results
    
    Returns:
        Dictionary with all sequence-specific findings
    """
    input_folder = Path(input_folder)
    
    # Find the case ID
    files = list(input_folder.glob("*_t1.nii.gz"))
    if not files:
        files = list(input_folder.glob("*-t1n.nii.gz"))  # BraTS 2025 format
    
    if not files:
        raise ValueError(f"No T1 files found in {input_folder}")
    
    case_id = files[0].name.split('_t1')[0].split('-t1')[0]
    print(f"Analyzing case: {case_id}")
    
    # Determine file naming convention (BraTS 2021 vs 2025)
    if (input_folder / f"{case_id}_t1.nii.gz").exists():
        # BraTS 2021 format
        t1_path = input_folder / f"{case_id}_t1.nii.gz"
        t1ce_path = input_folder / f"{case_id}_t1ce.nii.gz"
        t2_path = input_folder / f"{case_id}_t2.nii.gz"
        flair_path = input_folder / f"{case_id}_flair.nii.gz"
    else:
        # BraTS 2025 format
        t1_path = input_folder / f"{case_id}-t1n.nii.gz"
        t1ce_path = input_folder / f"{case_id}-t1c.nii.gz"
        t2_path = input_folder / f"{case_id}-t2w.nii.gz"
        flair_path = input_folder / f"{case_id}-t2f.nii.gz"
    
    # Load all sequences
    print("Loading MRI sequences...")
    t1_data, t1_affine, t1_header = load_nifti(t1_path)
    t1ce_data, _, _ = load_nifti(t1ce_path)
    t2_data, _, _ = load_nifti(t2_path)
    flair_data, _, _ = load_nifti(flair_path)
    
    # Load segmentation
    print("Loading segmentation mask...")
    seg_data, _, _ = load_nifti(segmentation_path)
    seg_data = np.round(seg_data).astype(np.int32)
    
    # Get voxel dimensions
    voxel_dims = t1_header.get_zooms()[:3]
    voxel_volume_mm3 = np.prod(voxel_dims)
    voxel_volume_cm3 = voxel_volume_mm3 / 1000
    
    print("\n" + "="*60)
    print("STEP 1: SEQUENCE-SPECIFIC FINDINGS")
    print("="*60)
    
    # 1. Analyze contrast enhancement
    print("\n--- Contrast Enhancement Analysis ---")
    enhancement_results = analyze_contrast_enhancement(t1_data, t1ce_data, seg_data)
    
    print(f"Enhancement pattern: {enhancement_results['enhancement_pattern']}")
    if enhancement_results['enhancing_tumor']['enhancement_ratio']:
        print(f"Mean enhancement ratio (ET): {enhancement_results['enhancing_tumor']['enhancement_ratio']['mean']:.2f}")
    
    # 2. Analyze T2/FLAIR behavior
    print("\n--- T2/FLAIR Signal Analysis ---")
    t2_flair_results = analyze_t2_flair_behavior(t2_data, flair_data, seg_data)
    
    for behavior in t2_flair_results['signal_behavior']:
        print(f"  - {behavior}")
    
    # 3. Compile intensity profiles for all regions
    print("\n--- Intensity Profiles by Region ---")
    
    regions = {
        'Background': 0,
        'Necrotic Core (NCR)': 1,
        'Edema (ED)': 2,
        'Enhancing Tumor (ET)': [3, 4]  # Handle both BraTS conventions
    }
    
    intensity_profiles = {}
    for region_name, label in regions.items():
        if isinstance(label, list):
            mask = np.isin(seg_data, label)
        else:
            mask = seg_data == label
        
        if mask.sum() == 0:
            continue
            
        intensity_profiles[region_name] = {
            'T1': get_intensity_stats(t1_data, mask),
            'T1ce': get_intensity_stats(t1ce_data, mask),
            'T2': get_intensity_stats(t2_data, mask),
            'FLAIR': get_intensity_stats(flair_data, mask)
        }
        
        vol_cm3 = mask.sum() * voxel_volume_cm3
        print(f"\n{region_name} (Volume: {vol_cm3:.2f} cm³, {mask.sum()} voxels):")
        print(f"  T1:    mean={intensity_profiles[region_name]['T1']['mean']:.1f}, std={intensity_profiles[region_name]['T1']['std']:.1f}")
        print(f"  T1ce:  mean={intensity_profiles[region_name]['T1ce']['mean']:.1f}, std={intensity_profiles[region_name]['T1ce']['std']:.1f}")
        print(f"  T2:    mean={intensity_profiles[region_name]['T2']['mean']:.1f}, std={intensity_profiles[region_name]['T2']['std']:.1f}")
        print(f"  FLAIR: mean={intensity_profiles[region_name]['FLAIR']['mean']:.1f}, std={intensity_profiles[region_name]['FLAIR']['std']:.1f}")
    
    # 4. Normal brain reference
    print("\n--- Normal Brain Reference ---")
    normal_brain = {
        'T1': get_normal_brain_stats(t1_data, seg_data),
        'T1ce': get_normal_brain_stats(t1ce_data, seg_data),
        'T2': get_normal_brain_stats(t2_data, seg_data),
        'FLAIR': get_normal_brain_stats(flair_data, seg_data)
    }
    
    print(f"Normal brain T1:    mean={normal_brain['T1']['mean']:.1f}")
    print(f"Normal brain T1ce:  mean={normal_brain['T1ce']['mean']:.1f}")
    print(f"Normal brain T2:    mean={normal_brain['T2']['mean']:.1f}")
    print(f"Normal brain FLAIR: mean={normal_brain['FLAIR']['mean']:.1f}")
    
    # Compile all results
    results = {
        'case_id': case_id,
        'step': 'Step 1 - Sequence-specific findings',
        'voxel_dimensions_mm': list(voxel_dims),
        'voxel_volume_cm3': voxel_volume_cm3,
        'contrast_enhancement': enhancement_results,
        't2_flair_analysis': t2_flair_results,
        'intensity_profiles': intensity_profiles,
        'normal_brain_reference': normal_brain,
        'sequences_analyzed': ['T1', 'T1ce', 'T2', 'FLAIR'],
        'diffusion_restriction': 'DWI/ADC not available in standard BraTS dataset'
    }
    
    # Generate text summary
    summary = generate_sequence_summary(results)
    results['text_summary'] = summary
    
    print("\n" + "="*60)
    print("TEXT SUMMARY FOR RADIOLOGY REPORT")
    print("="*60)
    print(summary)
    
    # Save results if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Custom JSON encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"\nResults saved to: {output_path}")
    
    return results


def generate_sequence_summary(results):
    """Generate a text summary suitable for a radiology report."""
    lines = []
    
    lines.append("SEQUENCE-SPECIFIC FINDINGS:")
    lines.append("")
    
    # Enhancement pattern
    enhancement = results['contrast_enhancement']
    lines.append(f"Contrast Enhancement: {enhancement['enhancement_pattern']}")
    
    if enhancement['enhancing_tumor']['enhancement_ratio']:
        ratio = enhancement['enhancing_tumor']['enhancement_ratio']['mean']
        lines.append(f"  - Enhancement ratio in enhancing tumor: {ratio:.2f}x relative to pre-contrast")
    
    # T2/FLAIR behavior
    lines.append("")
    lines.append("T2/FLAIR Signal Characteristics:")
    for behavior in results['t2_flair_analysis']['signal_behavior']:
        lines.append(f"  - {behavior}")
    
    # Key findings
    lines.append("")
    lines.append("Key Intensity Observations:")
    
    profiles = results['intensity_profiles']
    normal = results['normal_brain_reference']
    
    if 'Edema (ED)' in profiles and normal['FLAIR']['mean']:
        ed_flair = profiles['Edema (ED)']['FLAIR']['mean']
        normal_flair = normal['FLAIR']['mean']
        ratio = ed_flair / normal_flair
        lines.append(f"  - Peritumoral edema shows {ratio:.1f}x FLAIR signal vs normal brain")
    
    if 'Enhancing Tumor (ET)' in profiles:
        et_t1ce = profiles['Enhancing Tumor (ET)']['T1ce']['mean']
        et_t1 = profiles['Enhancing Tumor (ET)']['T1']['mean']
        if et_t1 > 0:
            ratio = et_t1ce / et_t1
            lines.append(f"  - Enhancing tumor shows {ratio:.1f}x T1 signal increase post-contrast")
    
    if 'Necrotic Core (NCR)' in profiles and normal['T1']['mean']:
        ncr_t1 = profiles['Necrotic Core (NCR)']['T1']['mean']
        normal_t1 = normal['T1']['mean']
        if ncr_t1 < normal_t1 * 0.8:
            lines.append("  - Necrotic core appears T1 hypointense (consistent with necrosis)")
    
    # Note about DWI
    lines.append("")
    lines.append("Note: Diffusion-weighted imaging (DWI/ADC) analysis not available")
    lines.append("      (not included in standard BraTS dataset)")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Extract sequence-specific findings from brain MRI'
    )
    parser.add_argument('--input', required=True, 
                        help='Input folder containing MRI sequences')
    parser.add_argument('--segmentation', required=True,
                        help='Path to segmentation mask (NIfTI)')
    parser.add_argument('--output', default=None,
                        help='Output path for JSON results (optional)')
    
    args = parser.parse_args()
    
    results = analyze_sequence_findings(args.input, args.segmentation, args.output)
    
    return results


if __name__ == "__main__":
    main()
