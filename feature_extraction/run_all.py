"""
Step 6: Run All Steps & Compile Comprehensive Report

Executes all feature extraction steps and compiles results into
a unified, radiology-ready report suitable for LLM processing.

Features:
- Runs Steps 1-5 sequentially
- Compiles all results into unified JSON
- Generates structured text report
- Creates LLM-ready summary

Author: AI-Powered Brain MRI Assistant
Date: November 27, 2025
"""

import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime

from utils import (
    load_nifti, get_case_id, get_mri_paths, get_voxel_dimensions,
    get_tumor_masks, save_results
)

# Import step modules
from step1_sequence_findings import analyze_sequence_findings
from step2_mass_effect import analyze_mass_effect
from step3_multiplicity import analyze_multiplicity
from step4_morphology import analyze_morphology
from step5_quality import analyze_quality
from step6_normal_structures import analyze_normal_structures


def compile_comprehensive_report(all_results):
    """
    Compile all step results into a comprehensive radiology report.
    """
    case_id = all_results['case_id']
    
    lines = []
    lines.append("=" * 70)
    lines.append("BRAIN MRI TUMOR ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Case ID: {case_id}")
    lines.append(f"Analysis Date: {all_results['analysis_timestamp']}")
    lines.append(f"Model: BraTS 2021 KAIST MRI Lab (1st Place)")
    lines.append("")
    
    # Executive Summary
    lines.append("-" * 70)
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 70)
    
    step1 = all_results['step1_sequence_findings']
    step2 = all_results['step2_mass_effect']
    step3 = all_results['step3_multiplicity']
    step4 = all_results['step4_morphology']
    step5 = all_results['step5_quality']
    
    # Key findings summary
    volumes = step1.get('volumes', {})
    wt_vol = volumes.get('Whole Tumor (WT)', 0)
    et_vol = volumes.get('Enhancing Tumor (ET)', 0)
    
    enhancement = step1.get('contrast_enhancement', {})
    pattern = enhancement.get('pattern', 'Unknown')
    
    location = step2.get('anatomical_location', {})
    primary_lobe = location.get('primary_lobe', 'Unknown')
    laterality = location.get('laterality', 'Unknown')
    
    lines.append(f"• Tumor Volume: {wt_vol:.1f} cm³")
    lines.append(f"• Location: {primary_lobe.capitalize()} lobe, {laterality}")
    lines.append(f"• Enhancement: {pattern}")
    lines.append(f"• Lesion Count: {step3.get('component_analysis', {}).get('num_components', 1)}")
    lines.append(f"• Quality Score: {step5.get('segmentation_quality', {}).get('quality_score', 0)}/100")
    lines.append("")
    
    # Detailed findings by section
    lines.append("-" * 70)
    lines.append("1. SEQUENCE-SPECIFIC FINDINGS")
    lines.append("-" * 70)
    lines.append(step1.get('text_summary', 'Not available'))
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("2. MASS EFFECT AND ANATOMICAL LOCATION")
    lines.append("-" * 70)
    lines.append(step2.get('text_summary', 'Not available'))
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("3. LESION MULTIPLICITY AND DISTRIBUTION")
    lines.append("-" * 70)
    lines.append(step3.get('text_summary', 'Not available'))
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("4. TUMOR MORPHOLOGY AND MARGINS")
    lines.append("-" * 70)
    lines.append(step4.get('text_summary', 'Not available'))
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("5. QUALITY CONTROL")
    lines.append("-" * 70)
    lines.append(step5.get('text_summary', 'Not available'))
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("6. NORMAL STRUCTURES")
    lines.append("-" * 70)
    step6 = all_results.get('step6_normal_structures', {})
    lines.append(step6.get('text_summary', 'Not available'))
    lines.append("")
    
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def generate_llm_summary(all_results):
    """
    Generate a concise, structured summary optimized for LLM consumption.
    This format is designed to be easily parsed by language models for
    generating natural language radiology reports.
    """
    step1 = all_results['step1_sequence_findings']
    step2 = all_results['step2_mass_effect']
    step3 = all_results['step3_multiplicity']
    step4 = all_results['step4_morphology']
    step5 = all_results['step5_quality']
    step6 = all_results.get('step6_normal_structures', {})
    
    # Extract key values
    volumes = step1.get('volumes', {})
    region_signals = step1.get('region_signal_analysis', {}).get('regions', {})
    enhancement = step1.get('contrast_enhancement', {})
    location = step2.get('anatomical_location', {})
    midline = step2.get('midline_shift', {})
    herniation = step2.get('herniation_risk', {})
    multiplicity = step3.get('component_analysis', {})
    distribution = step3.get('distribution_pattern', {})
    shape = step4.get('shape_descriptors', {})
    margins = step4.get('margin_definition', {})
    necrosis = step4.get('necrosis_pattern', {})
    cystic_solid = step4.get('cystic_solid_classification', {})
    quality = step5.get('segmentation_quality', {})
    artifacts = step5.get('artifact_detection', {})
    ventricular = step6.get('ventricular_system', {})
    parenchyma = step6.get('parenchyma', {})
    
    # Extract technique details
    technique = step1.get('technique', {})
    acq_params = technique.get('acquisition_parameters', {})
    
    summary = {
        'case_id': all_results['case_id'],
        'analysis_date': all_results['analysis_timestamp'],
        
        # PATIENT DEMOGRAPHICS - Placeholders to prevent LLM fabrication
        # These should be populated from clinical records if available
        'patient_info': {
            'age': '<not provided>',
            'sex': '<not provided>',
            'clinical_history': '<not provided>',
            'presenting_symptoms': '<not provided>',
            'relevant_prior_imaging': '<not provided>',
            'note': 'DO NOT fabricate patient demographics or clinical history. Only include information explicitly provided in clinical records.'
        },
        
        # TECHNIQUE SECTION - Critical for LLM to not guess sequences
        'technique': {
            'sequences_performed': technique.get('sequences_performed', []),
            'sequences_not_available': technique.get('sequences_not_available', ['DWI', 'ADC', 'MRS', 'Perfusion']),
            'contrast_administered': technique.get('contrast_administered', False),
            'contrast_note': technique.get('contrast_note', 'Unknown'),
            'acquisition_parameters': {
                'slice_thickness_mm': acq_params.get('slice_thickness_mm', 'Not available'),
                'in_plane_resolution_mm': acq_params.get('in_plane_resolution_mm', 'Not available'),
                'voxel_size_mm': acq_params.get('voxel_size_mm', 'Not available'),
                'matrix_size': acq_params.get('matrix_size', 'Not available'),
                'num_slices': acq_params.get('num_slices', 'Not available')
            },
            'note': 'LLM must only reference sequences listed in sequences_performed; do not infer or guess additional sequences'
        },
        
        'tumor_characteristics': {
            'volume_cm3': volumes.get('Whole Tumor (WT)', 0),
            'enhancing_volume_cm3': volumes.get('Enhancing Tumor (ET)', 0),
            'necrotic_volume_cm3': volumes.get('Necrotic Core (NCR)', 0),
            'edema_volume_cm3': volumes.get('Peritumoral Edema (ED)', 0),
            'max_diameter_mm': multiplicity.get('components', [{}])[0].get('max_diameter_mm', 0) if multiplicity.get('components') else 0
        },
        
        'location': {
            'hemisphere': location.get('hemisphere', 'Unknown'),
            'laterality': location.get('laterality', 'Unknown'),
            'primary_lobe': location.get('primary_lobe', 'Unknown'),
            'involved_lobes': location.get('lobes', []),
            'depth': location.get('depth', 'Unknown'),
            'gyri': location.get('approximate_gyri', []),
            # Laterality validation: cross-check hemisphere determination
            'laterality_validation': _validate_laterality(location, midline)
        },
        
        'signal_characteristics': {
            region: {
                'signal_summary': data.get('signal_summary', ''),
                'T1_ratio': data.get('T1', {}).get('ratio_to_normal', 0),
                'T2_ratio': data.get('T2', {}).get('ratio_to_normal', 0),
                'FLAIR_ratio': data.get('FLAIR', {}).get('ratio_to_normal', 0),
                'T1ce_ratio': data.get('T1ce', {}).get('ratio_to_normal', 0)
            }
            for region, data in region_signals.items()
        },
        
        'enhancement': {
            'present': enhancement.get('enhancement_present', False),
            'pattern': enhancement.get('pattern', 'None'),
            'strength': enhancement.get('enhancement_strength', 'None'),
            'heterogeneity': enhancement.get('heterogeneity', 'N/A')
        },
        
        'mass_effect': {
            'midline_shift_mm': midline.get('shift_mm', 0),
            'shift_significant': midline.get('is_significant', False),
            'shift_direction': midline.get('shift_direction', 'None'),
            'herniation_risk': herniation.get('risk_level', 'Low'),
            'mass_effect_score': herniation.get('mass_effect_score', 0)
        },
        
        'morphology': {
            'shape': shape.get('shape_classification', 'Unknown'),
            'sphericity': shape.get('sphericity', 0),
            'elongation': shape.get('elongation', 1),
            # Explicit separation of contour vs margin concepts
            'contour_shape': step4.get('border_regularity', {}).get('classification', 'Unknown'),
            'contour_concept': 'outer_surface_smoothness',
            'margin_transition': margins.get('classification', 'Unknown'),
            'margin_concept': 'intensity_transition_sharpness',
            'margin_sharpness': margins.get('margin_sharpness', 0),
            'combined_description': f"{step4.get('border_regularity', {}).get('classification', 'Unknown')} contour with {margins.get('classification', 'unknown').lower()} margins"
        },
        
        'necrosis': {
            'present': necrosis.get('necrosis_present', False),
            'pattern': necrosis.get('pattern', 'None'),
            'percentage': necrosis.get('necrosis_percentage', 0),
            'location': necrosis.get('location', 'N/A')
        },
        
        'cystic_solid': {
            'classification': cystic_solid.get('classification', 'Unknown'),
            'cystic_percentage': cystic_solid.get('cystic_percentage', 0),
            'solid_percentage': cystic_solid.get('solid_percentage', 100),
            'description': cystic_solid.get('description', '')
        },
        
        'multiplicity': {
            'lesion_count': multiplicity.get('num_components', 1),
            'is_single_lesion': multiplicity.get('is_single_lesion', True),
            'distribution_pattern': distribution.get('pattern', 'Solitary'),
            'has_satellites': step3.get('satellite_analysis', {}).get('has_satellites', False)
        },
        
        'differential_considerations': distribution.get('differential_considerations', []),
        
        'normal_structures': {
            'ventricular_system': {
                'size': ventricular.get('size_assessment', 'Not assessed'),
                'volume_cm3': ventricular.get('total_volume_cm3', 0),
                'hydrocephalus': ventricular.get('hydrocephalus_present', False),
                'symmetry': ventricular.get('symmetry_assessment', 'Unknown')
            },
            'parenchyma': {
                'gray_white_differentiation': parenchyma.get('gray_white_differentiation', {}).get('assessment', 'Unknown'),
                'white_matter_disease': parenchyma.get('periventricular_assessment', {}).get('white_matter_disease_present', False),
                'overall': parenchyma.get('overall_assessment', 'Unknown')
            }
        },
        
        # Enhanced quality metrics with T2 SNR impact
        'quality_metrics': {
            'segmentation_score': quality.get('quality_score', 0),
            'segmentation_grade': quality.get('grade', 'Unknown'),
            'image_quality': step5.get('image_quality', {}).get('overall_quality', 'Unknown'),
            'sequence_quality': {
                seq: data.get('quality', 'Unknown')
                for seq, data in step5.get('image_quality', {}).get('sequences', {}).items()
            },
            'artifacts': {
                'detected': artifacts.get('artifacts_detected', []),
                'severity': artifacts.get('severity', 'None'),
                'impact': artifacts.get('impact_on_analysis', 'Unknown')
            },
            'confidence_high': ['volume_measurements', 'multiplicity', 'enhancement_analysis'],
            'confidence_moderate': ['midline_shift', 'margin_analysis', 'anatomical_localization']
        },
        
        # Explicit reliability warnings based on image quality
        'measurement_reliability_warnings': _generate_reliability_warnings(step5),
        
        'caveats': step5.get('limitations_and_caveats', {}).get('caveats', [])
    }
    
    return summary


def _validate_laterality(location, midline):
    """
    Cross-check hemisphere determination between location and midline shift analyses.
    
    Returns consistency status and any discrepancies.
    """
    location_hemisphere = location.get('hemisphere', 'Unknown')
    midline_tumor_side = midline.get('tumor_hemisphere', 'Unknown')
    
    # Normalize values for comparison
    location_side = location_hemisphere.split('-')[0] if '-' in location_hemisphere else location_hemisphere
    
    # Check consistency
    if location_side == 'bilateral' or 'bilateral' in location_hemisphere:
        # Bilateral tumors won't match single-side determination
        return {
            'consistent': True,
            'note': 'Bilateral tumor - crosses midline',
            'location_method': location_hemisphere,
            'centroid_method': midline_tumor_side
        }
    elif location_side.lower() == midline_tumor_side.lower():
        return {
            'consistent': True,
            'note': 'Hemisphere determination consistent across methods',
            'location_method': location_hemisphere,
            'centroid_method': midline_tumor_side
        }
    else:
        return {
            'consistent': False,
            'warning': f'Hemisphere mismatch: location analysis suggests {location_hemisphere}, centroid analysis suggests {midline_tumor_side}. Tumor may be near midline.',
            'location_method': location_hemisphere,
            'centroid_method': midline_tumor_side
        }


def _generate_reliability_warnings(step5):
    """Generate explicit warnings about measurement reliability based on image quality."""
    warnings = []
    
    seq_quality = step5.get('image_quality', {}).get('sequences', {})
    
    # T2 SNR impact
    t2_quality = seq_quality.get('T2', {})
    t2_snr = t2_quality.get('snr_estimate', 10)
    if t2_snr < 6:
        warnings.append({
            'sequence': 'T2',
            'snr': t2_snr,
            'affected_measurements': ['necrosis_fraction', 'cystic_solid_classification', 'edema_extent'],
            'warning': f'Low T2 SNR ({t2_snr:.1f}) may reduce reliability of necrosis/cystic fraction and edema measurements'
        })
    
    # T1ce SNR impact  
    t1ce_quality = seq_quality.get('T1ce', {})
    t1ce_snr = t1ce_quality.get('snr_estimate', 10)
    if t1ce_snr < 6:
        warnings.append({
            'sequence': 'T1ce',
            'snr': t1ce_snr,
            'affected_measurements': ['enhancement_analysis', 'margin_sharpness'],
            'warning': f'Low T1ce SNR ({t1ce_snr:.1f}) may reduce reliability of enhancement and margin measurements'
        })
    
    # Overall quality impact
    overall_quality = step5.get('image_quality', {}).get('overall_quality', 'Good')
    if overall_quality in ['Poor', 'Fair']:
        warnings.append({
            'overall': True,
            'warning': f'Overall image quality is {overall_quality}; interpret quantitative measurements with caution'
        })
    
    return warnings


def run_all_steps(input_folder, segmentation_path, output_folder):
    """
    Run all feature extraction steps and compile comprehensive results.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    case_id = get_case_id(input_folder)
    
    print("=" * 70)
    print("BRAIN MRI FEATURE EXTRACTION PIPELINE")
    print("=" * 70)
    print(f"\nCase ID: {case_id}")
    print(f"Input: {input_folder}")
    print(f"Segmentation: {segmentation_path}")
    print(f"Output: {output_folder}")
    print("")
    
    # Run each step
    print("\n" + "=" * 70)
    print("RUNNING STEP 1: Sequence-Specific Findings")
    print("=" * 70)
    step1_output = output_folder / "step1_sequence_findings.json"
    step1_results = analyze_sequence_findings(input_folder, segmentation_path, step1_output)
    
    print("\n" + "=" * 70)
    print("RUNNING STEP 2: Mass Effect Metrics")
    print("=" * 70)
    step2_output = output_folder / "step2_mass_effect.json"
    step2_results = analyze_mass_effect(input_folder, segmentation_path, step2_output)
    
    print("\n" + "=" * 70)
    print("RUNNING STEP 3: Lesion Multiplicity")
    print("=" * 70)
    step3_output = output_folder / "step3_multiplicity.json"
    step3_results = analyze_multiplicity(input_folder, segmentation_path, step3_output)
    
    print("\n" + "=" * 70)
    print("RUNNING STEP 4: Tumor Morphology")
    print("=" * 70)
    step4_output = output_folder / "step4_morphology.json"
    step4_results = analyze_morphology(input_folder, segmentation_path, step4_output)
    
    print("\n" + "=" * 70)
    print("RUNNING STEP 5: Quality Control")
    print("=" * 70)
    step5_output = output_folder / "step5_quality.json"
    step5_results = analyze_quality(input_folder, segmentation_path, step5_output)
    
    print("\n" + "=" * 70)
    print("RUNNING STEP 6: Normal Structures Assessment")
    print("=" * 70)
    step6_output = output_folder / "step6_normal_structures.json"
    step6_results = analyze_normal_structures(input_folder, segmentation_path, step6_output)
    
    # Compile all results
    print("\n" + "=" * 70)
    print("COMPILING COMPREHENSIVE REPORT")
    print("=" * 70)
    
    all_results = {
        'case_id': case_id,
        'analysis_timestamp': datetime.now().isoformat(),
        'input_folder': str(input_folder),
        'segmentation_path': str(segmentation_path),
        'step1_sequence_findings': step1_results,
        'step2_mass_effect': step2_results,
        'step3_multiplicity': step3_results,
        'step4_morphology': step4_results,
        'step5_quality': step5_results,
        'step6_normal_structures': step6_results
    }
    
    # Generate comprehensive report
    comprehensive_report = compile_comprehensive_report(all_results)
    all_results['comprehensive_report'] = comprehensive_report
    
    # Generate LLM-ready summary
    llm_summary = generate_llm_summary(all_results)
    all_results['llm_summary'] = llm_summary
    
    # Save comprehensive results
    comprehensive_output = output_folder / "comprehensive_analysis.json"
    save_results(all_results, comprehensive_output)
    
    # Save LLM summary separately for easy access
    llm_output = output_folder / "llm_ready_summary.json"
    save_results(llm_summary, llm_output)
    
    # Save text report
    report_output = output_folder / "radiology_report.txt"
    with open(report_output, 'w', encoding='utf-8') as f:
        f.write(comprehensive_report)
    print(f"\nText report saved to: {report_output}")
    
    # Print the comprehensive report
    print("\n" + comprehensive_report)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  • Comprehensive analysis: {comprehensive_output}")
    print(f"  • LLM-ready summary: {llm_output}")
    print(f"  • Text report: {report_output}")
    print(f"  • Individual step results: step1-5 JSON files")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Run all feature extraction steps and compile comprehensive report'
    )
    parser.add_argument('--input', required=True,
                        help='Input folder containing MRI sequences')
    parser.add_argument('--segmentation', required=True,
                        help='Path to segmentation mask (NIfTI)')
    parser.add_argument('--output', required=True,
                        help='Output folder for all results')
    
    args = parser.parse_args()
    
    run_all_steps(args.input, args.segmentation, args.output)


if __name__ == "__main__":
    main()
