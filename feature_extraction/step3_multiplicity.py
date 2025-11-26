"""
Step 3: Lesion Multiplicity & Distribution

Analyzes tumor distribution patterns:
- Single vs multiple lesions detection
- Spatial clustering analysis
- Connectivity assessment (contiguous vs separate foci)
- Satellite lesion detection
- Distribution pattern classification

Clinical Relevance:
- Multiple lesions may suggest metastatic disease or multicentric glioma
- Satellite lesions within 2cm suggest local tumor spread
- Distant separate lesions may indicate different pathology

Author: AI-Powered Brain MRI Assistant
Date: November 27, 2025
"""

import numpy as np
import argparse
from pathlib import Path
from scipy import ndimage
from scipy.ndimage import label, binary_dilation, binary_erosion

from utils import (
    load_nifti, get_case_id, get_mri_paths, get_voxel_dimensions,
    get_tumor_masks, get_centroid, get_bounding_box,
    calculate_volume, save_results
)


# Distance thresholds in mm
SATELLITE_DISTANCE_MM = 20  # Lesions within 20mm are satellites
SEPARATE_DISTANCE_MM = 40   # Lesions >40mm apart are clearly separate

# Minimum volume to be considered a real lesion (filter segmentation noise)
MIN_LESION_VOLUME_CM3 = 0.1  # 100 mm³ minimum


def detect_connected_components(seg_data, voxel_dims):
    """
    Detect separate tumor components using 3D connected component analysis.
    
    Returns list of components with their properties.
    """
    tumor_mask = seg_data > 0
    
    if tumor_mask.sum() == 0:
        return {
            'num_components': 0,
            'components': [],
            'is_single_lesion': True,
            'description': 'No tumor detected'
        }
    
    # Use 3D connectivity (26-connected for full 3D neighborhood)
    structure = ndimage.generate_binary_structure(3, 3)
    labeled_array, num_components = label(tumor_mask, structure=structure)
    
    components = []
    
    for comp_id in range(1, num_components + 1):
        comp_mask = labeled_array == comp_id
        comp_voxels = comp_mask.sum()
        
        # Calculate volume
        volume_cm3 = comp_voxels * np.prod(voxel_dims) / 1000
        
        # Get centroid
        coords = np.where(comp_mask)
        centroid = {
            'x': float(np.mean(coords[0])),
            'y': float(np.mean(coords[1])),
            'z': float(np.mean(coords[2]))
        }
        
        # Get centroid in mm (physical coordinates)
        centroid_mm = {
            'x': centroid['x'] * voxel_dims[0],
            'y': centroid['y'] * voxel_dims[1],
            'z': centroid['z'] * voxel_dims[2]
        }
        
        # Get bounding box
        bbox = {
            'x_min': int(coords[0].min()),
            'x_max': int(coords[0].max()),
            'y_min': int(coords[1].min()),
            'y_max': int(coords[1].max()),
            'z_min': int(coords[2].min()),
            'z_max': int(coords[2].max())
        }
        
        # Calculate max diameter
        max_diameter_mm = max(
            (bbox['x_max'] - bbox['x_min']) * voxel_dims[0],
            (bbox['y_max'] - bbox['y_min']) * voxel_dims[1],
            (bbox['z_max'] - bbox['z_min']) * voxel_dims[2]
        )
        
        # Analyze component composition (which labels are present)
        comp_labels = seg_data[comp_mask]
        unique_labels, label_counts = np.unique(comp_labels, return_counts=True)
        composition = {
            'ncr': int((comp_labels == 1).sum()),
            'ed': int((comp_labels == 2).sum()),
            'et': int((comp_labels == 3).sum())
        }
        
        components.append({
            'id': comp_id,
            'voxel_count': int(comp_voxels),
            'volume_cm3': float(volume_cm3),
            'centroid_voxel': centroid,
            'centroid_mm': centroid_mm,
            'bounding_box': bbox,
            'max_diameter_mm': float(max_diameter_mm),
            'composition': composition,
            'has_enhancement': composition['et'] > 0
        })
    
    # Filter out tiny components (likely segmentation noise)
    significant_components = [c for c in components if c['volume_cm3'] >= MIN_LESION_VOLUME_CM3]
    noise_components = [c for c in components if c['volume_cm3'] < MIN_LESION_VOLUME_CM3]
    
    # Sort by volume (largest first)
    significant_components.sort(key=lambda x: x['volume_cm3'], reverse=True)
    
    # Renumber after sorting
    for i, comp in enumerate(significant_components):
        comp['rank'] = i + 1
        if i == 0:
            comp['classification'] = 'Primary lesion'
        else:
            comp['classification'] = f'Secondary lesion #{i}'
    
    # Build description
    num_significant = len(significant_components)
    if noise_components:
        noise_note = f' ({len(noise_components)} sub-threshold fragments excluded, <{MIN_LESION_VOLUME_CM3} cm³)'
    else:
        noise_note = ''
    
    return {
        'num_components': num_significant,
        'components': significant_components,
        'is_single_lesion': num_significant == 1,
        'description': f'{num_significant} lesion(s) detected{noise_note}',
        'excluded_fragments': len(noise_components),
        'minimum_volume_threshold_cm3': MIN_LESION_VOLUME_CM3
    }


def calculate_inter_lesion_distances(components, voxel_dims):
    """
    Calculate distances between all pairs of lesion components.
    """
    if len(components) < 2:
        return {
            'distances': [],
            'min_distance_mm': None,
            'max_distance_mm': None,
            'mean_distance_mm': None
        }
    
    distances = []
    
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            c1 = components[i]['centroid_mm']
            c2 = components[j]['centroid_mm']
            
            dist = np.sqrt(
                (c1['x'] - c2['x'])**2 +
                (c1['y'] - c2['y'])**2 +
                (c1['z'] - c2['z'])**2
            )
            
            distances.append({
                'component_1': components[i]['id'],
                'component_2': components[j]['id'],
                'distance_mm': float(dist),
                'relationship': classify_distance_relationship(dist)
            })
    
    all_distances = [d['distance_mm'] for d in distances]
    
    return {
        'distances': distances,
        'min_distance_mm': float(min(all_distances)) if all_distances else None,
        'max_distance_mm': float(max(all_distances)) if all_distances else None,
        'mean_distance_mm': float(np.mean(all_distances)) if all_distances else None
    }


def classify_distance_relationship(distance_mm):
    """Classify the relationship between lesions based on distance."""
    if distance_mm < SATELLITE_DISTANCE_MM:
        return 'Satellite/adjacent'
    elif distance_mm < SEPARATE_DISTANCE_MM:
        return 'Regional spread'
    else:
        return 'Distant/separate'


def analyze_enhancing_components(seg_data, voxel_dims):
    """
    Separately analyze enhancing tumor components.
    Multiple enhancing foci may have different clinical significance.
    """
    et_mask = seg_data == 3  # Enhancing tumor label
    
    if et_mask.sum() == 0:
        return {
            'num_enhancing_foci': 0,
            'enhancing_components': [],
            'pattern': 'Non-enhancing',
            'description': 'No enhancing tumor components detected'
        }
    
    structure = ndimage.generate_binary_structure(3, 3)
    labeled_et, num_et_components = label(et_mask, structure=structure)
    
    et_components = []
    
    for comp_id in range(1, num_et_components + 1):
        comp_mask = labeled_et == comp_id
        volume_cm3 = comp_mask.sum() * np.prod(voxel_dims) / 1000
        
        coords = np.where(comp_mask)
        centroid_mm = {
            'x': float(np.mean(coords[0]) * voxel_dims[0]),
            'y': float(np.mean(coords[1]) * voxel_dims[1]),
            'z': float(np.mean(coords[2]) * voxel_dims[2])
        }
        
        et_components.append({
            'id': comp_id,
            'volume_cm3': float(volume_cm3),
            'centroid_mm': centroid_mm
        })
    
    # Sort by volume
    et_components.sort(key=lambda x: x['volume_cm3'], reverse=True)
    
    # Determine pattern
    if num_et_components == 0:
        pattern = 'Non-enhancing'
    elif num_et_components == 1:
        pattern = 'Single enhancing focus'
    elif num_et_components <= 3:
        pattern = 'Few enhancing foci'
    else:
        pattern = 'Multiple/scattered enhancing foci'
    
    return {
        'num_enhancing_foci': num_et_components,
        'enhancing_components': et_components,
        'pattern': pattern,
        'total_enhancing_volume_cm3': float(sum(c['volume_cm3'] for c in et_components)),
        'description': f'{num_et_components} separate enhancing focus/foci detected'
    }


def detect_satellite_lesions(components, primary_component, voxel_dims):
    """
    Detect satellite lesions around the primary tumor.
    
    Satellites are small lesions within SATELLITE_DISTANCE_MM of the primary tumor.
    """
    if len(components) < 2:
        return {
            'satellite_count': 0,
            'satellites': [],
            'has_satellites': False,
            'description': 'Single lesion, no satellites'
        }
    
    primary_centroid = primary_component['centroid_mm']
    satellites = []
    
    for comp in components[1:]:  # Skip primary (first component)
        comp_centroid = comp['centroid_mm']
        
        dist = np.sqrt(
            (primary_centroid['x'] - comp_centroid['x'])**2 +
            (primary_centroid['y'] - comp_centroid['y'])**2 +
            (primary_centroid['z'] - comp_centroid['z'])**2
        )
        
        if dist < SATELLITE_DISTANCE_MM:
            satellites.append({
                'component_id': comp['id'],
                'volume_cm3': comp['volume_cm3'],
                'distance_from_primary_mm': float(dist),
                'has_enhancement': comp['has_enhancement']
            })
    
    if satellites:
        description = f'{len(satellites)} satellite lesion(s) within {SATELLITE_DISTANCE_MM}mm of primary tumor'
    else:
        description = 'No satellite lesions detected'
    
    return {
        'satellite_count': len(satellites),
        'satellites': satellites,
        'has_satellites': len(satellites) > 0,
        'satellite_threshold_mm': SATELLITE_DISTANCE_MM,
        'description': description
    }


def classify_distribution_pattern(component_analysis, distance_analysis, satellite_analysis, enhancing_analysis):
    """
    Classify overall tumor distribution pattern.
    """
    num_components = component_analysis['num_components']
    
    if num_components == 0:
        return {
            'pattern': 'No tumor',
            'classification': 'No lesion detected',
            'clinical_implication': 'N/A',
            'differential_considerations': []
        }
    
    if num_components == 1:
        pattern = 'Solitary'
        classification = 'Single contiguous lesion'
        clinical_implication = 'Unifocal disease, typical for primary brain tumor'
        differentials = ['Primary glioma', 'Solitary metastasis', 'Lymphoma', 'Abscess']
    
    elif satellite_analysis['has_satellites']:
        pattern = 'Primary with satellites'
        classification = 'Main lesion with satellite nodules'
        clinical_implication = 'Suggests local tumor spread or infiltrative growth pattern'
        differentials = ['High-grade glioma with infiltration', 'Multicentric glioma', 'Inflammatory process']
    
    elif num_components <= 3:
        if distance_analysis['max_distance_mm'] and distance_analysis['max_distance_mm'] < SEPARATE_DISTANCE_MM:
            pattern = 'Regional multifocal'
            classification = 'Few lesions in regional distribution'
            clinical_implication = 'Regional disease, may be contiguous or multicentric'
            differentials = ['Multicentric glioma', 'Regional metastases', 'Demyelinating disease']
        else:
            pattern = 'Distant multifocal'
            classification = 'Separate lesions in different brain regions'
            clinical_implication = 'Multifocal disease, consider metastatic process'
            differentials = ['Metastatic disease', 'Multicentric glioma', 'CNS lymphoma', 'Multifocal infection']
    
    else:  # More than 3 components
        pattern = 'Diffuse/scattered'
        classification = 'Multiple lesions throughout brain'
        clinical_implication = 'Diffuse disease pattern, high probability of metastatic or systemic process'
        differentials = ['Metastatic carcinoma', 'CNS lymphoma', 'Miliary tuberculosis', 'Septic emboli']
    
    # Add enhancement pattern context
    if enhancing_analysis['num_enhancing_foci'] == 0:
        enhancement_note = 'Non-enhancing pattern may suggest low-grade pathology'
    elif enhancing_analysis['num_enhancing_foci'] > num_components:
        enhancement_note = 'Multiple enhancing foci within lesions suggest heterogeneous enhancement'
    else:
        enhancement_note = 'Enhancement pattern consistent with lesion count'
    
    return {
        'pattern': pattern,
        'classification': classification,
        'clinical_implication': clinical_implication,
        'differential_considerations': differentials,
        'enhancement_note': enhancement_note,
        'lesion_count': num_components,
        'enhancing_foci_count': enhancing_analysis['num_enhancing_foci']
    }


def generate_summary(results):
    """Generate text summary for radiology report."""
    lines = []
    
    lines.append("LESION MULTIPLICITY AND DISTRIBUTION:")
    lines.append("")
    
    # Lesion count
    comp = results['component_analysis']
    lines.append(f"Lesion Count: {comp['num_components']}")
    
    if comp['num_components'] == 0:
        lines.append("  No tumor lesions detected")
        return "\n".join(lines)
    
    # Primary lesion
    if comp['components']:
        primary = comp['components'][0]
        lines.append(f"  Primary lesion: {primary['volume_cm3']:.2f} cm³, max diameter {primary['max_diameter_mm']:.1f} mm")
        if primary['has_enhancement']:
            lines.append("    - Contains enhancing component")
        else:
            lines.append("    - Non-enhancing")
    
    # Secondary lesions
    if comp['num_components'] > 1:
        lines.append(f"  Secondary lesions: {comp['num_components'] - 1}")
        for lesion in comp['components'][1:]:
            lines.append(f"    - Lesion #{lesion['rank']}: {lesion['volume_cm3']:.2f} cm³")
    
    # Distances
    if results['distance_analysis']['distances']:
        lines.append("")
        lines.append("Inter-lesion Distances:")
        dist = results['distance_analysis']
        lines.append(f"  - Minimum: {dist['min_distance_mm']:.1f} mm")
        lines.append(f"  - Maximum: {dist['max_distance_mm']:.1f} mm")
    
    # Satellites
    sat = results['satellite_analysis']
    if sat['has_satellites']:
        lines.append("")
        lines.append(f"Satellite Lesions: {sat['satellite_count']}")
        for s in sat['satellites']:
            lines.append(f"  - {s['distance_from_primary_mm']:.1f} mm from primary, {s['volume_cm3']:.2f} cm³")
    
    # Enhancing foci
    lines.append("")
    enh = results['enhancing_analysis']
    lines.append(f"Enhancing Foci: {enh['pattern']}")
    if enh['num_enhancing_foci'] > 0:
        lines.append(f"  - Total enhancing volume: {enh['total_enhancing_volume_cm3']:.2f} cm³")
    
    # Distribution pattern
    lines.append("")
    dist_pattern = results['distribution_pattern']
    lines.append(f"Distribution Pattern: {dist_pattern['pattern']}")
    lines.append(f"  {dist_pattern['classification']}")
    lines.append(f"  Clinical implication: {dist_pattern['clinical_implication']}")
    
    if dist_pattern['differential_considerations']:
        lines.append("  Differential considerations:")
        for diff in dist_pattern['differential_considerations']:
            lines.append(f"    - {diff}")
    
    return "\n".join(lines)


def analyze_multiplicity(input_folder, segmentation_path, output_path=None):
    """
    Main function to analyze lesion multiplicity and distribution.
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
    print("STEP 3: LESION MULTIPLICITY AND DISTRIBUTION")
    print("="*60)
    
    # Detect connected components
    print("\n--- Connected Component Analysis ---")
    component_analysis = detect_connected_components(seg_data, voxel_dims)
    print(f"  Number of separate lesions: {component_analysis['num_components']}")
    
    if component_analysis['num_components'] > 0:
        primary = component_analysis['components'][0]
        print(f"  Primary lesion: {primary['volume_cm3']:.2f} cm³")
        if component_analysis['num_components'] > 1:
            print(f"  Secondary lesions: {component_analysis['num_components'] - 1}")
    
    # Calculate inter-lesion distances
    print("\n--- Inter-lesion Distance Analysis ---")
    distance_analysis = calculate_inter_lesion_distances(
        component_analysis['components'], voxel_dims
    )
    if distance_analysis['distances']:
        print(f"  Min distance: {distance_analysis['min_distance_mm']:.1f} mm")
        print(f"  Max distance: {distance_analysis['max_distance_mm']:.1f} mm")
    else:
        print("  Single lesion - no inter-lesion distances")
    
    # Detect satellites
    print("\n--- Satellite Lesion Detection ---")
    if component_analysis['components']:
        satellite_analysis = detect_satellite_lesions(
            component_analysis['components'],
            component_analysis['components'][0],
            voxel_dims
        )
    else:
        satellite_analysis = {
            'satellite_count': 0,
            'satellites': [],
            'has_satellites': False,
            'description': 'No tumor detected'
        }
    print(f"  {satellite_analysis['description']}")
    
    # Analyze enhancing components
    print("\n--- Enhancing Foci Analysis ---")
    enhancing_analysis = analyze_enhancing_components(seg_data, voxel_dims)
    print(f"  Pattern: {enhancing_analysis['pattern']}")
    if enhancing_analysis['num_enhancing_foci'] > 0:
        print(f"  Total enhancing volume: {enhancing_analysis['total_enhancing_volume_cm3']:.2f} cm³")
    
    # Classify distribution pattern
    print("\n--- Distribution Pattern Classification ---")
    distribution_pattern = classify_distribution_pattern(
        component_analysis, distance_analysis, satellite_analysis, enhancing_analysis
    )
    print(f"  Pattern: {distribution_pattern['pattern']}")
    print(f"  {distribution_pattern['classification']}")
    
    # Compile results
    results = {
        'case_id': case_id,
        'step': 'Step 3 - Lesion multiplicity and distribution',
        'voxel_info': voxel_info,
        'component_analysis': component_analysis,
        'distance_analysis': distance_analysis,
        'satellite_analysis': satellite_analysis,
        'enhancing_analysis': enhancing_analysis,
        'distribution_pattern': distribution_pattern
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
        description='Step 3: Analyze lesion multiplicity and distribution'
    )
    parser.add_argument('--input', required=True,
                        help='Input folder containing MRI sequences')
    parser.add_argument('--segmentation', required=True,
                        help='Path to segmentation mask (NIfTI)')
    parser.add_argument('--output', default=None,
                        help='Output path for JSON results')
    
    args = parser.parse_args()
    
    analyze_multiplicity(args.input, args.segmentation, args.output)


if __name__ == "__main__":
    main()
