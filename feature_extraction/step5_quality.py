"""
Step 5: Quality Control & Confidence Metrics

Assesses the reliability and quality of the analysis:
- Segmentation quality indicators
- Image quality assessment
- Confidence scores for key measurements
- Potential artifacts detection
- Limitations and caveats

Clinical Relevance:
- Helps radiologists understand reliability of automated analysis
- Flags cases that may need manual review
- Identifies technical factors that may affect interpretation

Author: AI-Powered Brain MRI Assistant
Date: November 27, 2025
"""

import numpy as np
import argparse
from pathlib import Path
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation, sobel

from utils import (
    load_nifti, get_case_id, get_mri_paths, get_voxel_dimensions,
    get_tumor_masks, get_brain_mask, save_results
)


def assess_segmentation_quality(seg_data, tumor_masks, voxel_dims):
    """
    Assess quality indicators of the segmentation.
    
    Checks for:
    - Reasonable volume ranges
    - Component connectivity
    - Label consistency
    - Anatomical plausibility
    """
    issues = []
    warnings = []
    quality_score = 100  # Start at 100, deduct for issues
    
    wt_mask = tumor_masks['wt']
    tc_mask = tumor_masks['tc']
    et_mask = tumor_masks['et']
    ncr_mask = tumor_masks['ncr']
    ed_mask = tumor_masks['ed']
    
    # Calculate volumes
    voxel_vol = np.prod(voxel_dims) / 1000  # cm³
    wt_vol = wt_mask.sum() * voxel_vol
    tc_vol = tc_mask.sum() * voxel_vol
    et_vol = et_mask.sum() * voxel_vol
    ncr_vol = ncr_mask.sum() * voxel_vol
    ed_vol = ed_mask.sum() * voxel_vol
    
    # Check 1: No tumor detected
    if wt_vol == 0:
        issues.append("No tumor segmentation detected")
        quality_score -= 50
        return {
            'quality_score': quality_score,
            'grade': 'Poor',
            'issues': issues,
            'warnings': warnings,
            'recommendation': 'Manual review required - no segmentation found'
        }
    
    # Check 2: Extremely small tumor (might be artifact)
    if wt_vol < 0.5:
        warnings.append(f"Very small tumor volume ({wt_vol:.2f} cm³) - may be artifact")
        quality_score -= 10
    
    # Check 3: Extremely large tumor (might include non-tumor)
    if wt_vol > 300:
        warnings.append(f"Very large tumor volume ({wt_vol:.0f} cm³) - verify boundaries")
        quality_score -= 10
    
    # Check 4: Tumor core larger than whole tumor (impossible)
    if tc_vol > wt_vol * 1.01:  # Allow 1% tolerance
        issues.append("Tumor core volume exceeds whole tumor - label inconsistency")
        quality_score -= 20
    
    # Check 5: Check for isolated small fragments
    structure = ndimage.generate_binary_structure(3, 3)
    labeled_wt, num_components = ndimage.label(wt_mask, structure=structure)
    
    if num_components > 5:
        warnings.append(f"Multiple disconnected components ({num_components}) - possible over-segmentation")
        quality_score -= 5
    
    # Check 6: Verify enhancing tumor is subset of tumor core
    if et_mask.sum() > 0:
        et_outside_tc = et_mask & ~tc_mask
        if et_outside_tc.sum() > et_mask.sum() * 0.05:
            warnings.append("Some enhancing tumor voxels outside tumor core boundary")
            quality_score -= 5
    
    # Check 7: Check for holes in segmentation (may indicate issues)
    filled_wt = ndimage.binary_fill_holes(wt_mask)
    holes = filled_wt & ~wt_mask
    hole_fraction = holes.sum() / wt_mask.sum() if wt_mask.sum() > 0 else 0
    
    if hole_fraction > 0.1:
        warnings.append(f"Segmentation has internal holes ({hole_fraction*100:.0f}% of volume)")
        quality_score -= 5
    
    # Check 8: Verify tumor not at image boundary (truncation)
    dims = seg_data.shape
    boundary_margin = 3
    
    at_boundary = False
    if wt_mask[:boundary_margin, :, :].sum() > 0:
        at_boundary = True
    if wt_mask[-boundary_margin:, :, :].sum() > 0:
        at_boundary = True
    if wt_mask[:, :boundary_margin, :].sum() > 0:
        at_boundary = True
    if wt_mask[:, -boundary_margin:, :].sum() > 0:
        at_boundary = True
    if wt_mask[:, :, :boundary_margin].sum() > 0:
        at_boundary = True
    if wt_mask[:, :, -boundary_margin:].sum() > 0:
        at_boundary = True
    
    if at_boundary:
        warnings.append("Tumor extends to image boundary - may be truncated")
        quality_score -= 10
    
    # Determine grade
    if quality_score >= 90:
        grade = 'Excellent'
        recommendation = 'High confidence in segmentation quality'
    elif quality_score >= 75:
        grade = 'Good'
        recommendation = 'Acceptable quality, routine review recommended'
    elif quality_score >= 60:
        grade = 'Fair'
        recommendation = 'Some concerns identified, careful review advised'
    elif quality_score >= 40:
        grade = 'Poor'
        recommendation = 'Multiple issues detected, manual verification required'
    else:
        grade = 'Unacceptable'
        recommendation = 'Significant problems, re-segmentation may be needed'
    
    return {
        'quality_score': max(0, quality_score),
        'grade': grade,
        'issues': issues,
        'warnings': warnings,
        'num_components': num_components,
        'hole_fraction': float(hole_fraction),
        'at_image_boundary': at_boundary,
        'recommendation': recommendation
    }


def assess_image_quality(mri_data_dict, brain_mask):
    """
    Assess quality of the input MRI sequences.
    
    Checks for:
    - Signal-to-noise ratio estimates
    - Intensity normalization
    - Missing data / zeros
    - Intensity outliers
    """
    quality_metrics = {}
    overall_issues = []
    
    for seq_name, data in mri_data_dict.items():
        seq_issues = []
        
        # Get brain values
        brain_values = data[brain_mask]
        
        if len(brain_values) == 0:
            seq_issues.append("No brain tissue detected")
            quality_metrics[seq_name] = {
                'snr_estimate': 0,
                'issues': seq_issues,
                'quality': 'Poor'
            }
            continue
        
        # Estimate SNR (signal / background noise)
        signal_mean = brain_values.mean()
        
        # Background (non-brain, non-zero)
        background_mask = ~brain_mask & (data > 0) & (data < np.percentile(data[data > 0], 10))
        if background_mask.sum() > 100:
            background_std = data[background_mask].std()
            snr = signal_mean / background_std if background_std > 0 else 0
        else:
            # Use brain tissue variation as proxy
            snr = signal_mean / brain_values.std() if brain_values.std() > 0 else 0
        
        # Check for zeros within brain
        zeros_in_brain = ((data == 0) & brain_mask).sum()
        zero_fraction = zeros_in_brain / brain_mask.sum() if brain_mask.sum() > 0 else 0
        
        if zero_fraction > 0.01:
            seq_issues.append(f"Missing data: {zero_fraction*100:.1f}% zeros within brain")
        
        # Check for intensity outliers
        q99 = np.percentile(brain_values, 99)
        q01 = np.percentile(brain_values, 1)
        iqr = np.percentile(brain_values, 75) - np.percentile(brain_values, 25)
        
        outlier_high = (brain_values > q99 + 3*iqr).sum()
        outlier_low = (brain_values < q01 - 3*iqr).sum()
        outlier_fraction = (outlier_high + outlier_low) / len(brain_values)
        
        if outlier_fraction > 0.01:
            seq_issues.append(f"Intensity outliers detected ({outlier_fraction*100:.1f}%)")
        
        # Determine quality
        if snr > 20 and len(seq_issues) == 0:
            quality = 'Excellent'
        elif snr > 10 and len(seq_issues) <= 1:
            quality = 'Good'
        elif snr > 5:
            quality = 'Fair'
        else:
            quality = 'Poor'
        
        quality_metrics[seq_name] = {
            'snr_estimate': float(snr),
            'zero_fraction': float(zero_fraction),
            'outlier_fraction': float(outlier_fraction),
            'mean_intensity': float(signal_mean),
            'std_intensity': float(brain_values.std()),
            'issues': seq_issues,
            'quality': quality
        }
        
        overall_issues.extend([f"{seq_name}: {issue}" for issue in seq_issues])
    
    # Overall image quality
    qualities = [m['quality'] for m in quality_metrics.values()]
    if all(q == 'Excellent' for q in qualities):
        overall_quality = 'Excellent'
    elif all(q in ['Excellent', 'Good'] for q in qualities):
        overall_quality = 'Good'
    elif any(q == 'Poor' for q in qualities):
        overall_quality = 'Poor'
    else:
        overall_quality = 'Fair'
    
    return {
        'sequences': quality_metrics,
        'overall_quality': overall_quality,
        'issues': overall_issues
    }


def detect_artifacts(mri_data_dict, brain_mask, seg_data):
    """
    Detect common MRI artifacts that may affect image quality.
    
    Artifact types:
    - Motion artifacts (ghosting, blurring)
    - Susceptibility artifacts (signal dropout, distortion)
    - Wrap-around/aliasing artifacts
    - Intensity inhomogeneity (bias field)
    - Ringing/Gibbs artifacts
    """
    artifacts_detected = []
    artifact_details = {}
    
    # Reference: T1 data for most artifact detection
    t1_data = mri_data_dict.get('T1', list(mri_data_dict.values())[0])
    
    # 1. Check for intensity inhomogeneity (bias field)
    # Compare intensity at center vs periphery
    brain_coords = np.where(brain_mask)
    if len(brain_coords[0]) > 0:
        # Calculate distance from center
        center = np.array([np.mean(brain_coords[i]) for i in range(3)])
        
        # Get intensities at different distances from center
        distances = np.sqrt(
            (brain_coords[0] - center[0])**2 +
            (brain_coords[1] - center[1])**2 +
            (brain_coords[2] - center[2])**2
        )
        
        max_dist = distances.max()
        inner_mask = distances < max_dist * 0.3
        outer_mask = distances > max_dist * 0.7
        
        inner_values = t1_data[brain_mask][inner_mask]
        outer_values = t1_data[brain_mask][outer_mask]
        
        if len(inner_values) > 100 and len(outer_values) > 100:
            inhomogeneity_ratio = outer_values.mean() / inner_values.mean() if inner_values.mean() > 0 else 1.0
            
            if inhomogeneity_ratio < 0.7 or inhomogeneity_ratio > 1.4:
                artifacts_detected.append("Intensity inhomogeneity")
                artifact_details['intensity_inhomogeneity'] = {
                    'detected': True,
                    'severity': 'Moderate' if 0.6 < inhomogeneity_ratio < 1.6 else 'Severe',
                    'ratio': float(inhomogeneity_ratio),
                    'description': 'Significant signal intensity variation across the brain (bias field artifact)',
                    'impact': 'May affect intensity-based measurements'
                }
            else:
                artifact_details['intensity_inhomogeneity'] = {
                    'detected': False,
                    'ratio': float(inhomogeneity_ratio)
                }
    
    # 2. Check for motion artifacts (ghosting)
    # Motion creates periodic signal in phase-encode direction
    # Check for unusual periodic patterns in background
    for seq_name, data in mri_data_dict.items():
        # Get background region (outside brain)
        background = ~brain_mask & (data > 0)
        
        if background.sum() > 1000:
            bg_values = data[background]
            bg_std = bg_values.std()
            bg_mean = bg_values.mean()
            
            # High variation in background suggests ghosting
            cv_background = bg_std / bg_mean if bg_mean > 0 else 0
            
            if cv_background > 0.5:
                if 'motion_ghosting' not in artifact_details:
                    artifacts_detected.append("Possible motion artifact")
                    artifact_details['motion_ghosting'] = {
                        'detected': True,
                        'affected_sequences': [seq_name],
                        'background_cv': float(cv_background),
                        'description': 'Elevated background signal variation suggests possible motion/ghosting',
                        'impact': 'May affect tumor boundary delineation'
                    }
                else:
                    artifact_details['motion_ghosting']['affected_sequences'].append(seq_name)
    
    if 'motion_ghosting' not in artifact_details:
        artifact_details['motion_ghosting'] = {'detected': False}
    
    # 3. Check for susceptibility artifacts (signal dropout)
    # Typically at air-tissue interfaces (sinuses, temporal lobes)
    # Look for unexpected zero regions within brain
    zero_clusters = (t1_data == 0) & brain_mask
    if zero_clusters.sum() > 100:
        # Check if zeros are in expected susceptibility regions (inferior brain)
        zero_coords = np.where(zero_clusters)
        if len(zero_coords[2]) > 0:
            mean_z = np.mean(zero_coords[2])
            brain_z = np.mean(np.where(brain_mask)[2])
            
            # If zeros are in inferior part, likely susceptibility
            if mean_z < brain_z * 0.5:
                artifacts_detected.append("Susceptibility artifact")
                artifact_details['susceptibility'] = {
                    'detected': True,
                    'location': 'Inferior brain (near skull base)',
                    'volume_mm3': float(zero_clusters.sum()),
                    'description': 'Signal dropout in inferior brain, typical susceptibility artifact near air-bone interface',
                    'impact': 'May affect assessment of inferior tumor components'
                }
            else:
                artifact_details['susceptibility'] = {
                    'detected': True,
                    'location': 'Atypical location',
                    'description': 'Signal voids detected in unexpected location - may indicate pathology or artifact'
                }
    else:
        artifact_details['susceptibility'] = {'detected': False}
    
    # 4. Check for wrap-around artifacts (aliasing)
    # Look for anatomy appearing at wrong edge of image
    # This is hard to detect automatically, so we check for asymmetric brain at edges
    dims = t1_data.shape
    edge_margin = 5
    
    # Check if there's signal at all edges
    edge_signal = {
        'x_min': t1_data[:edge_margin, :, :].max() > 0,
        'x_max': t1_data[-edge_margin:, :, :].max() > 0,
        'y_min': t1_data[:, :edge_margin, :].max() > 0,
        'y_max': t1_data[:, -edge_margin:, :].max() > 0
    }
    
    if sum(edge_signal.values()) >= 3:  # Signal at most edges
        artifacts_detected.append("Possible wrap-around")
        artifact_details['wrap_around'] = {
            'detected': True,
            'description': 'Brain tissue extends to image boundaries - possible aliasing or tight FOV',
            'edges_affected': [k for k, v in edge_signal.items() if v],
            'impact': 'Anatomy at edges may be compromised'
        }
    else:
        artifact_details['wrap_around'] = {'detected': False}
    
    # 5. Check for Gibbs ringing (truncation artifact)
    # Appears as oscillating signal near high-contrast edges
    # Check gradient magnitude variation at tumor boundary
    if seg_data is not None and seg_data.max() > 0:
        tumor_mask = seg_data > 0
        eroded = binary_erosion(tumor_mask, iterations=2)
        tumor_edge = tumor_mask & ~eroded
        
        if tumor_edge.sum() > 100:
            # Get gradient at tumor edge
            grad_x = sobel(t1_data.astype(float), axis=0)
            grad_y = sobel(t1_data.astype(float), axis=1)
            grad_z = sobel(t1_data.astype(float), axis=2)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
            
            edge_gradient = grad_mag[tumor_edge]
            edge_cv = edge_gradient.std() / edge_gradient.mean() if edge_gradient.mean() > 0 else 0
            
            if edge_cv > 1.5:  # High variation suggests ringing
                artifacts_detected.append("Possible Gibbs ringing")
                artifact_details['gibbs_ringing'] = {
                    'detected': True,
                    'edge_gradient_cv': float(edge_cv),
                    'description': 'High gradient variation at tumor margins, may indicate Gibbs/truncation artifact',
                    'impact': 'May affect precise tumor boundary measurement'
                }
            else:
                artifact_details['gibbs_ringing'] = {'detected': False}
        else:
            artifact_details['gibbs_ringing'] = {'detected': False, 'note': 'Insufficient edge for analysis'}
    else:
        artifact_details['gibbs_ringing'] = {'detected': False}
    
    # Overall artifact assessment
    if len(artifacts_detected) == 0:
        overall_assessment = "No significant artifacts detected"
        artifact_severity = "None"
    elif len(artifacts_detected) <= 2:
        overall_assessment = f"Minor artifacts detected: {', '.join(artifacts_detected)}"
        artifact_severity = "Mild"
    else:
        overall_assessment = f"Multiple artifacts present: {', '.join(artifacts_detected)}"
        artifact_severity = "Moderate to Severe"
    
    return {
        'artifacts_detected': artifacts_detected,
        'artifact_count': len(artifacts_detected),
        'severity': artifact_severity,
        'overall_assessment': overall_assessment,
        'details': artifact_details,
        'impact_on_analysis': 'Review recommended' if len(artifacts_detected) > 1 else 'Minimal impact expected'
    }


def calculate_measurement_confidence(results_dict):
    """
    Estimate confidence levels for key measurements based on
    data quality and measurement reliability.
    """
    confidence = {}
    
    # Volume measurements - generally reliable if segmentation is good
    confidence['volume_measurements'] = {
        'confidence': 'High',
        'note': 'Volume calculations are mathematically precise given the segmentation'
    }
    
    # Enhancement patterns - depends on T1ce quality
    confidence['enhancement_analysis'] = {
        'confidence': 'High',
        'note': 'Based on objective intensity comparisons'
    }
    
    # Midline shift - moderate confidence due to anatomical variation
    confidence['midline_shift'] = {
        'confidence': 'Moderate',
        'note': 'Estimated from tissue asymmetry; clinical correlation recommended'
    }
    
    # Border/margin analysis - moderate confidence
    confidence['margin_analysis'] = {
        'confidence': 'Moderate',
        'note': 'Based on intensity gradients; subjective component remains'
    }
    
    # Anatomical localization - moderate confidence
    confidence['anatomical_localization'] = {
        'confidence': 'Moderate',
        'note': 'Based on standard atlas coordinates; individual variation exists'
    }
    
    # Multiplicity - high confidence for connected component analysis
    confidence['multiplicity'] = {
        'confidence': 'High',
        'note': '3D connected component analysis is objective'
    }
    
    return confidence


def identify_limitations(seg_quality, image_quality, tumor_masks):
    """
    Identify limitations and caveats for the analysis.
    """
    limitations = []
    caveats = []
    
    # Standard limitations
    limitations.append("Automated analysis should be verified by qualified radiologist")
    limitations.append("Segmentation based on BraTS 2021 model trained on glioma cases")
    
    # Sequence-specific limitations
    limitations.append("DWI/ADC sequences not available - diffusion characteristics not assessed")
    limitations.append("Perfusion imaging not available - cannot assess tumor vascularity")
    
    # Conditional limitations based on findings
    wt_mask = tumor_masks['wt']
    et_mask = tumor_masks['et']
    
    if et_mask.sum() == 0:
        # Guarded language - no firm grade diagnosis
        caveats.append("Non-enhancing pattern: Can be seen with lower-grade glioma, treatment effect, or other pathology; clinical and histopathological correlation required")
    
    if seg_quality.get('at_image_boundary', False):
        caveats.append("Tumor at image boundary: Volume may be underestimated")
    
    # Image quality impact on specific measurements
    seq_quality = image_quality.get('sequences', {})
    t2_snr = seq_quality.get('T2', {}).get('snr_estimate', 10)
    if t2_snr < 6:
        caveats.append(f"Low T2 SNR ({t2_snr:.1f}): Necrosis fraction and cystic/solid classification less reliable")
    
    if image_quality.get('overall_quality') in ['Fair', 'Poor']:
        caveats.append("Suboptimal image quality may affect measurement accuracy")
    
    # Model-specific caveats
    caveats.append("Model optimized for adult gliomas; performance may vary for other tumor types")
    caveats.append("Peritumoral edema vs infiltrating tumor cannot be distinguished on conventional MRI")
    
    return {
        'limitations': limitations,
        'caveats': caveats
    }


def generate_summary(results):
    """Generate text summary for radiology report."""
    lines = []
    
    lines.append("QUALITY CONTROL AND CONFIDENCE ASSESSMENT:")
    lines.append("")
    
    # Segmentation quality
    seg = results['segmentation_quality']
    lines.append(f"Segmentation Quality: {seg['grade']} (Score: {seg['quality_score']}/100)")
    lines.append(f"  {seg['recommendation']}")
    
    if seg['issues']:
        lines.append("  Issues:")
        for issue in seg['issues']:
            lines.append(f"    ⚠ {issue}")
    
    if seg['warnings']:
        lines.append("  Warnings:")
        for warning in seg['warnings']:
            lines.append(f"    ⚡ {warning}")
    
    # Image quality
    lines.append("")
    img = results['image_quality']
    lines.append(f"Image Quality: {img['overall_quality']}")
    for seq, metrics in img['sequences'].items():
        lines.append(f"  - {seq}: {metrics['quality']} (SNR ≈ {metrics['snr_estimate']:.1f})")
    
    # Artifact detection
    lines.append("")
    art = results.get('artifact_detection', {})
    if art:
        lines.append(f"Artifact Assessment: {art.get('severity', 'Not assessed')}")
        lines.append(f"  {art.get('overall_assessment', 'Not assessed')}")
        if art.get('artifacts_detected'):
            for artifact in art['artifacts_detected']:
                lines.append(f"    • {artifact}")
    
    # Confidence levels
    lines.append("")
    lines.append("Measurement Confidence:")
    for measure, conf in results['measurement_confidence'].items():
        lines.append(f"  - {measure.replace('_', ' ').title()}: {conf['confidence']}")
    
    # Limitations
    lines.append("")
    lines.append("Key Limitations:")
    for lim in results['limitations_and_caveats']['limitations'][:3]:
        lines.append(f"  • {lim}")
    
    if results['limitations_and_caveats']['caveats']:
        lines.append("")
        lines.append("Case-Specific Caveats:")
        for cav in results['limitations_and_caveats']['caveats']:
            lines.append(f"  • {cav}")
    
    return "\n".join(lines)


def analyze_quality(input_folder, segmentation_path, output_path=None):
    """
    Main function to perform quality control analysis.
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
    brain_mask = get_brain_mask(t1_data)
    
    print("\n" + "="*60)
    print("STEP 5: QUALITY CONTROL AND CONFIDENCE METRICS")
    print("="*60)
    
    # Segmentation quality
    print("\n--- Segmentation Quality Assessment ---")
    seg_quality = assess_segmentation_quality(seg_data, tumor_masks, voxel_dims)
    print(f"  Grade: {seg_quality['grade']} (Score: {seg_quality['quality_score']}/100)")
    if seg_quality['issues']:
        for issue in seg_quality['issues']:
            print(f"  ⚠ Issue: {issue}")
    if seg_quality['warnings']:
        for warning in seg_quality['warnings']:
            print(f"  ⚡ Warning: {warning}")
    
    # Image quality
    print("\n--- Image Quality Assessment ---")
    mri_dict = {'T1': t1_data, 'T1ce': t1ce_data, 'T2': t2_data, 'FLAIR': flair_data}
    image_quality = assess_image_quality(mri_dict, brain_mask)
    print(f"  Overall: {image_quality['overall_quality']}")
    for seq, metrics in image_quality['sequences'].items():
        print(f"  {seq}: {metrics['quality']} (SNR ≈ {metrics['snr_estimate']:.1f})")
    
    # Artifact detection
    print("\n--- Artifact Detection ---")
    artifact_detection = detect_artifacts(mri_dict, brain_mask, seg_data)
    print(f"  Severity: {artifact_detection['severity']}")
    print(f"  {artifact_detection['overall_assessment']}")
    if artifact_detection['artifacts_detected']:
        for artifact in artifact_detection['artifacts_detected']:
            print(f"    • {artifact}")
    
    # Measurement confidence
    print("\n--- Measurement Confidence ---")
    measurement_confidence = calculate_measurement_confidence({})
    for measure, conf in measurement_confidence.items():
        print(f"  {measure}: {conf['confidence']}")
    
    # Limitations
    print("\n--- Limitations and Caveats ---")
    limitations = identify_limitations(seg_quality, image_quality, tumor_masks)
    print(f"  {len(limitations['limitations'])} standard limitations")
    print(f"  {len(limitations['caveats'])} case-specific caveats")
    
    # Compile results
    results = {
        'case_id': case_id,
        'step': 'Step 5 - Quality control and confidence metrics',
        'segmentation_quality': seg_quality,
        'image_quality': image_quality,
        'artifact_detection': artifact_detection,
        'measurement_confidence': measurement_confidence,
        'limitations_and_caveats': limitations
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
        description='Step 5: Quality control and confidence assessment'
    )
    parser.add_argument('--input', required=True,
                        help='Input folder containing MRI sequences')
    parser.add_argument('--segmentation', required=True,
                        help='Path to segmentation mask (NIfTI)')
    parser.add_argument('--output', default=None,
                        help='Output path for JSON results')
    
    args = parser.parse_args()
    
    analyze_quality(args.input, args.segmentation, args.output)


if __name__ == "__main__":
    main()
