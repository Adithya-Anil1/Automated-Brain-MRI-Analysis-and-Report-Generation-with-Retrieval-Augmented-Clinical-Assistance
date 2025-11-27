"""
Step 4: Tumor Morphology & Margins

Analyzes tumor shape and border characteristics:
- Shape descriptors (sphericity, elongation, irregularity)
- Margin analysis (well-defined vs infiltrative)
- Border regularity assessment
- Surface characteristics
- Necrosis patterns

Clinical Relevance:
- Irregular margins suggest infiltrative growth (high-grade glioma)
- Well-defined margins may indicate lower grade or encapsulated lesion
- Necrosis patterns correlate with tumor grade and prognosis

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
    get_tumor_masks, get_centroid, get_bounding_box,
    calculate_volume, save_results
)


def calculate_surface_area(mask, voxel_dims):
    """
    Estimate surface area using the marching cubes concept.
    Counts boundary voxels and estimates surface contribution.
    """
    if mask.sum() == 0:
        return 0.0
    
    # Create eroded version
    eroded = binary_erosion(mask)
    
    # Surface voxels are those in original but not in eroded
    surface_voxels = mask & ~eroded
    
    # Each surface voxel contributes approximately one face
    # Average voxel face area
    avg_face_area = (voxel_dims[0] * voxel_dims[1] + 
                     voxel_dims[1] * voxel_dims[2] + 
                     voxel_dims[0] * voxel_dims[2]) / 3
    
    surface_area = surface_voxels.sum() * avg_face_area
    
    return float(surface_area)


def calculate_sphericity(volume_mm3, surface_area_mm2):
    """
    Calculate sphericity: ratio of surface area of equivalent sphere 
    to actual surface area.
    
    Sphericity = 1.0 for perfect sphere, <1.0 for irregular shapes.
    """
    if surface_area_mm2 == 0 or volume_mm3 == 0:
        return 0.0
    
    # Surface area of sphere with same volume
    radius = (3 * volume_mm3 / (4 * np.pi)) ** (1/3)
    sphere_surface = 4 * np.pi * radius ** 2
    
    sphericity = sphere_surface / surface_area_mm2
    
    # Clamp to [0, 1] (can exceed 1 due to discretization)
    return float(min(1.0, max(0.0, sphericity)))


def calculate_elongation(mask, voxel_dims):
    """
    Calculate elongation using principal component analysis.
    
    Returns ratio of longest to shortest axis.
    """
    coords = np.where(mask)
    
    if len(coords[0]) < 10:
        return 1.0, [1.0, 1.0, 1.0]
    
    # Scale by voxel dimensions
    points = np.array([
        coords[0] * voxel_dims[0],
        coords[1] * voxel_dims[1],
        coords[2] * voxel_dims[2]
    ]).T
    
    # Center the points
    centered = points - points.mean(axis=0)
    
    # Compute covariance matrix
    cov = np.cov(centered.T)
    
    # Get eigenvalues (principal axes lengths squared)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
    
    # Elongation is ratio of largest to smallest
    if eigenvalues[-1] > 0:
        elongation = np.sqrt(eigenvalues[0] / eigenvalues[-1])
    else:
        elongation = 1.0
    
    # Axis lengths (standard deviations along principal axes)
    axis_lengths = [float(np.sqrt(e) * 2) for e in eigenvalues]  # 2 sigma for approximate extent
    
    return float(elongation), axis_lengths


def calculate_compactness(volume_mm3, surface_area_mm2):
    """
    Calculate compactness (isoperimetric quotient).
    
    Compactness = 36 * pi * V^2 / S^3
    = 1.0 for sphere, <1.0 for less compact shapes
    """
    if surface_area_mm2 == 0:
        return 0.0
    
    compactness = (36 * np.pi * volume_mm3**2) / (surface_area_mm2**3)
    
    return float(min(1.0, compactness))


def analyze_border_regularity(mask, voxel_dims):
    """
    Analyze border CONTOUR regularity by examining surface curvature variation.
    
    This measures whether the outer surface is smooth vs lobulated/spiculated.
    Note: This is distinct from margin definition (intensity transition sharpness).
    A tumor can have smooth contours but infiltrative margins.
    """
    if mask.sum() == 0:
        return {
            'regularity_score': 0,
            'classification': 'No tumor',
            'description': 'No tumor detected'
        }
    
    # Get surface voxels
    eroded = binary_erosion(mask)
    surface = mask & ~eroded
    
    if surface.sum() < 10:
        return {
            'regularity_score': 1.0,
            'classification': 'Too small to assess',
            'description': 'Tumor too small for border analysis'
        }
    
    # Calculate distance transform for gradient estimation
    dist_inside = distance_transform_edt(mask)
    dist_outside = distance_transform_edt(~mask)
    
    # Signed distance
    signed_dist = dist_inside - dist_outside
    
    # Get gradient (surface normals)
    grad_x = np.gradient(signed_dist, axis=0)
    grad_y = np.gradient(signed_dist, axis=1)
    grad_z = np.gradient(signed_dist, axis=2)
    
    # Magnitude of gradient at surface
    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    
    # Get gradient values at surface
    surface_grad = grad_mag[surface]
    
    # Regularity: low variation in gradient magnitude indicates smooth border
    if surface_grad.std() > 0:
        # Coefficient of variation of gradient magnitude
        cv = surface_grad.std() / surface_grad.mean()
        regularity = 1.0 / (1.0 + cv)
    else:
        regularity = 1.0
    
    # Classify (contour smoothness, not margin sharpness)
    if regularity > 0.7:
        classification = 'Smooth contour'
        description = 'Smooth, regular outer contour (note: does not indicate margin sharpness)'
    elif regularity > 0.5:
        classification = 'Mildly lobulated'
        description = 'Some contour irregularity with mild lobulation'
    elif regularity > 0.3:
        classification = 'Lobulated'
        description = 'Lobulated/irregular outer contour'
    else:
        classification = 'Highly irregular'
        description = 'Highly irregular/spiculated outer contour'
    
    return {
        'regularity_score': float(regularity),
        'classification': classification,
        'description': description,
        'surface_voxel_count': int(surface.sum()),
        'concept': 'contour_smoothness'  # Explicit: this is about shape, not intensity transition
    }


def analyze_margin_definition(t1ce_data, seg_data, tumor_masks, voxel_dims):
    """
    Analyze tumor-brain TRANSITION ZONE (margin definition).
    
    This measures how sharply the tumor intensity transitions to normal brain.
    Sharp transitions = well-demarcated margins.
    Gradual transitions = infiltrative margins (tumor blends with brain).
    Note: Distinct from contour regularity (surface smoothness).
    """
    wt_mask = tumor_masks['wt']
    
    if wt_mask.sum() == 0:
        return {
            'margin_sharpness': 0,
            'classification': 'No tumor',
            'description': 'No tumor detected'
        }
    
    # Create band around tumor (peritumoral zone)
    dilated = binary_dilation(wt_mask, iterations=5)
    peritumoral = dilated & ~wt_mask
    
    # Get intensities
    tumor_intensities = t1ce_data[wt_mask]
    peritumoral_intensities = t1ce_data[peritumoral]
    
    if len(peritumoral_intensities) == 0:
        return {
            'margin_sharpness': 0.5,
            'classification': 'Could not assess',
            'description': 'Insufficient peritumoral tissue for analysis'
        }
    
    # Calculate intensity contrast at border
    tumor_mean = tumor_intensities.mean()
    peritumoral_mean = peritumoral_intensities.mean()
    
    # Contrast ratio
    if peritumoral_mean > 0:
        contrast = abs(tumor_mean - peritumoral_mean) / peritumoral_mean
    else:
        contrast = 0
    
    # Get intensities at immediate border
    eroded = binary_erosion(wt_mask)
    border_inner = wt_mask & ~eroded
    border_outer = binary_dilation(wt_mask) & ~wt_mask
    
    inner_intensities = t1ce_data[border_inner]
    outer_intensities = t1ce_data[border_outer]
    
    # Gradient at border
    if len(inner_intensities) > 0 and len(outer_intensities) > 0:
        border_gradient = abs(inner_intensities.mean() - outer_intensities.mean())
        border_gradient_normalized = border_gradient / (inner_intensities.std() + outer_intensities.std() + 1e-6)
    else:
        border_gradient_normalized = 0
    
    # Sharpness score (higher = sharper margins)
    sharpness = min(1.0, (contrast + border_gradient_normalized) / 2)
    
    # Classify (intensity transition sharpness)
    if sharpness > 0.6:
        classification = 'Sharp transition'
        description = 'Abrupt tumor-brain intensity transition, well-demarcated margin'
    elif sharpness > 0.4:
        classification = 'Moderate transition'
        description = 'Moderately distinct margin with some gradual transition zones'
    elif sharpness > 0.2:
        classification = 'Gradual transition'
        description = 'Indistinct margin with gradual intensity blending into brain'
    else:
        classification = 'Infiltrative transition'
        description = 'No clear intensity demarcation, tumor infiltrates surrounding parenchyma'
    
    return {
        'margin_sharpness': float(sharpness),
        'contrast_ratio': float(contrast),
        'border_gradient': float(border_gradient_normalized),
        'classification': classification,
        'description': description,
        'concept': 'intensity_transition'  # Explicit: this is about signal transition, not shape
    }


def analyze_cystic_vs_solid(t1_data, t2_data, flair_data, seg_data, tumor_masks, voxel_dims):
    """
    Classify tumor as cystic, solid, or mixed based on signal characteristics.
    
    Cystic components: Follow CSF signal (T1 dark, T2 bright, FLAIR dark/suppressed)
    Solid components: Do not follow CSF signal
    Necrosis: May have variable signal, often with debris
    """
    ncr_mask = tumor_masks['ncr']
    wt_mask = tumor_masks['wt']
    tc_mask = tumor_masks['tc']
    
    if wt_mask.sum() == 0:
        return {
            'classification': 'No tumor',
            'cystic_percentage': 0,
            'solid_percentage': 0,
            'description': 'No tumor detected'
        }
    
    voxel_vol = np.prod(voxel_dims) / 1000  # cm³
    
    # Get reference CSF values from ventricles (if identifiable)
    # CSF: T1 very low, T2 very high, FLAIR suppressed
    brain_values = t1_data[t1_data > 0]
    csf_t1_upper = np.percentile(brain_values, 10)  # CSF is in bottom 10%
    csf_t2_lower = np.percentile(t2_data[t2_data > 0], 85)  # CSF is in top 15%
    csf_flair_upper = np.percentile(flair_data[flair_data > 0], 20)  # FLAIR suppressed
    
    # Analyze NCR region for cystic vs necrotic
    if ncr_mask.sum() > 0:
        ncr_t1 = t1_data[ncr_mask]
        ncr_t2 = t2_data[ncr_mask]
        ncr_flair = flair_data[ncr_mask]
        
        # Count voxels with CSF-like signal (cystic)
        cystic_like = (
            (ncr_t1 < csf_t1_upper * 1.5) &  # Low T1
            (ncr_t2 > csf_t2_lower * 0.8) &  # High T2
            (ncr_flair < csf_flair_upper * 2)  # Relatively suppressed on FLAIR
        )
        
        cystic_fraction_in_ncr = cystic_like.sum() / len(ncr_t1) if len(ncr_t1) > 0 else 0
        
        # T2 signal variation (cysts are homogeneous, necrosis is heterogeneous)
        t2_cv = ncr_t2.std() / ncr_t2.mean() if ncr_t2.mean() > 0 else 0
        
        # FLAIR suppression (true cysts suppress, necrosis with protein doesn't)
        flair_t2_ratio = ncr_flair.mean() / ncr_t2.mean() if ncr_t2.mean() > 0 else 1
        
    else:
        cystic_fraction_in_ncr = 0
        t2_cv = 0
        flair_t2_ratio = 1
    
    # Calculate component volumes
    ncr_volume = ncr_mask.sum() * voxel_vol
    wt_volume = wt_mask.sum() * voxel_vol
    
    # Cystic volume estimate
    cystic_volume = ncr_volume * cystic_fraction_in_ncr
    cystic_percentage = (cystic_volume / wt_volume * 100) if wt_volume > 0 else 0
    
    # Solid volume (non-cystic tumor)
    solid_volume = wt_volume - cystic_volume
    solid_percentage = 100 - cystic_percentage
    
    # Classification based on proportions
    if cystic_percentage > 70:
        classification = 'Predominantly cystic'
        description = 'Large cystic component with thin wall/rim'
    elif cystic_percentage > 40:
        classification = 'Cystic with solid component'
        description = 'Mixed cystic and solid tumor with significant cystic component'
    elif cystic_percentage > 15:
        classification = 'Solid with cystic component'
        description = 'Predominantly solid tumor with cystic/necrotic areas'
    elif ncr_mask.sum() > 0:
        # Has necrosis but not truly cystic
        if t2_cv > 0.3:  # Heterogeneous
            classification = 'Solid with necrosis'
            description = 'Solid tumor with central necrotic (non-cystic) component'
        else:
            classification = 'Solid with possible cyst'
            description = 'Solid tumor with possible small cystic component'
    else:
        classification = 'Solid'
        description = 'Homogeneous solid tumor without significant cystic component'
    
    # Additional signal characteristics
    signal_characteristics = {
        't2_homogeneity': 'Homogeneous' if t2_cv < 0.2 else ('Mildly heterogeneous' if t2_cv < 0.4 else 'Heterogeneous'),
        'flair_suppression': 'Present (suggests true cyst)' if flair_t2_ratio < 0.7 else 'Absent (suggests necrosis/protein)',
        'csf_like_signal_fraction': float(cystic_fraction_in_ncr)
    }
    
    return {
        'classification': classification,
        'cystic_volume_cm3': float(cystic_volume),
        'cystic_percentage': float(cystic_percentage),
        'solid_volume_cm3': float(solid_volume),
        'solid_percentage': float(solid_percentage),
        'signal_characteristics': signal_characteristics,
        'description': description
    }


def analyze_necrosis_pattern(seg_data, tumor_masks, voxel_dims):
    """
    Analyze necrosis distribution within the tumor.
    """
    ncr_mask = tumor_masks['ncr']
    tc_mask = tumor_masks['tc']
    wt_mask = tumor_masks['wt']
    
    ncr_volume = ncr_mask.sum() * np.prod(voxel_dims) / 1000  # cm³
    tc_volume = tc_mask.sum() * np.prod(voxel_dims) / 1000
    wt_volume = wt_mask.sum() * np.prod(voxel_dims) / 1000
    
    if wt_volume == 0:
        return {
            'necrosis_present': False,
            'pattern': 'No tumor',
            'description': 'No tumor detected'
        }
    
    if ncr_volume == 0:
        return {
            'necrosis_present': False,
            'necrosis_volume_cm3': 0,
            'necrosis_percentage': 0,
            'pattern': 'No necrosis',
            'description': 'No central necrosis identified, solid tumor'
        }
    
    necrosis_pct = (ncr_volume / wt_volume) * 100
    
    # Analyze necrosis location (central vs peripheral)
    if ncr_mask.sum() > 0 and tc_mask.sum() > 0:
        # Get centroids
        ncr_coords = np.where(ncr_mask)
        tc_coords = np.where(tc_mask)
        
        ncr_centroid = np.array([np.mean(ncr_coords[i]) for i in range(3)])
        tc_centroid = np.array([np.mean(tc_coords[i]) for i in range(3)])
        
        # Distance from necrosis center to tumor core center
        dist = np.linalg.norm((ncr_centroid - tc_centroid) * voxel_dims)
        
        # Tumor core radius approximation
        tc_radius = (3 * tc_volume * 1000 / (4 * np.pi)) ** (1/3)  # in mm
        
        if dist < tc_radius * 0.3:
            location = 'Central'
            location_description = 'Necrosis centered within tumor'
        elif dist < tc_radius * 0.6:
            location = 'Eccentric'
            location_description = 'Necrosis somewhat offset from tumor center'
        else:
            location = 'Peripheral'
            location_description = 'Necrosis located eccentrically'
    else:
        location = 'Undetermined'
        location_description = 'Could not determine necrosis location'
    
    # Pattern classification
    if necrosis_pct > 50:
        pattern = 'Extensive necrosis'
        description = f'Large central necrotic component ({necrosis_pct:.0f}% of tumor), characteristic of high-grade glioma'
    elif necrosis_pct > 25:
        pattern = 'Moderate necrosis'
        description = f'Moderate central necrosis ({necrosis_pct:.0f}% of tumor), suggests high-grade pathology'
    elif necrosis_pct > 10:
        pattern = 'Focal necrosis'
        description = f'Focal areas of necrosis ({necrosis_pct:.0f}% of tumor)'
    else:
        pattern = 'Minimal necrosis'
        description = f'Small necrotic foci ({necrosis_pct:.0f}% of tumor)'
    
    return {
        'necrosis_present': True,
        'necrosis_volume_cm3': float(ncr_volume),
        'necrosis_percentage': float(necrosis_pct),
        'pattern': pattern,
        'location': location,
        'location_description': location_description,
        'description': description
    }


def calculate_shape_descriptors(seg_data, tumor_masks, voxel_dims):
    """
    Calculate comprehensive shape descriptors for the tumor.
    """
    wt_mask = tumor_masks['wt']
    
    if wt_mask.sum() == 0:
        return {
            'volume_cm3': 0,
            'surface_area_mm2': 0,
            'sphericity': 0,
            'compactness': 0,
            'elongation': 1.0,
            'principal_axes_mm': [0, 0, 0]
        }
    
    # Volume
    volume_mm3 = wt_mask.sum() * np.prod(voxel_dims)
    volume_cm3 = volume_mm3 / 1000
    
    # Surface area
    surface_area = calculate_surface_area(wt_mask, voxel_dims)
    
    # Sphericity
    sphericity = calculate_sphericity(volume_mm3, surface_area)
    
    # Compactness
    compactness = calculate_compactness(volume_mm3, surface_area)
    
    # Elongation
    elongation, principal_axes = calculate_elongation(wt_mask, voxel_dims)
    
    # Shape classification
    if sphericity > 0.8:
        shape_class = 'Spherical/round'
    elif sphericity > 0.6:
        shape_class = 'Ovoid'
    elif sphericity > 0.4:
        shape_class = 'Irregular'
    else:
        shape_class = 'Highly irregular/complex'
    
    if elongation > 2.5:
        elongation_class = 'Elongated'
    elif elongation > 1.5:
        elongation_class = 'Mildly elongated'
    else:
        elongation_class = 'Roughly isotropic'
    
    return {
        'volume_cm3': float(volume_cm3),
        'surface_area_mm2': float(surface_area),
        'sphericity': float(sphericity),
        'compactness': float(compactness),
        'elongation': float(elongation),
        'principal_axes_mm': principal_axes,
        'shape_classification': shape_class,
        'elongation_classification': elongation_class
    }


def generate_summary(results):
    """Generate text summary for radiology report."""
    lines = []
    
    lines.append("TUMOR MORPHOLOGY AND MARGINS:")
    lines.append("")
    
    # Shape descriptors
    shape = results['shape_descriptors']
    lines.append(f"Shape: {shape['shape_classification']}")
    lines.append(f"  - Volume: {shape['volume_cm3']:.2f} cm³")
    lines.append(f"  - Surface area: {shape['surface_area_mm2']:.0f} mm²")
    lines.append(f"  - Sphericity: {shape['sphericity']:.2f} ({shape['shape_classification'].lower()})")
    lines.append(f"  - Elongation: {shape['elongation']:.2f} ({shape['elongation_classification'].lower()})")
    
    # Border regularity (contour smoothness)
    lines.append("")
    border = results['border_regularity']
    lines.append(f"Contour Shape: {border['classification']}")
    lines.append(f"  (Measures outer surface smoothness, not intensity transition)")
    lines.append(f"  {border['description']}")
    
    # Margin definition (intensity transition)
    lines.append("")
    margin = results['margin_definition']
    lines.append(f"Margin Transition: {margin['classification']}")
    lines.append(f"  (Measures tumor-brain intensity demarcation)")
    lines.append(f"  {margin['description']}")
    
    # Combined interpretation for LLM
    lines.append("")
    lines.append(f"Morphology Summary: {border['classification']} contour with {margin['classification'].lower()} margins")
    
    # Necrosis
    lines.append("")
    necro = results['necrosis_pattern']
    if necro['necrosis_present']:
        lines.append(f"Necrosis: {necro['pattern']}")
        lines.append(f"  - Volume: {necro['necrosis_volume_cm3']:.2f} cm³ ({necro['necrosis_percentage']:.0f}% of tumor)")
        lines.append(f"  - Location: {necro['location']}")
        lines.append(f"  {necro['description']}")
    else:
        lines.append(f"Necrosis: {necro['pattern']}")
        lines.append(f"  {necro['description']}")
    
    # Cystic vs Solid
    lines.append("")
    cystic = results['cystic_solid_classification']
    lines.append(f"Cystic/Solid: {cystic['classification']}")
    lines.append(f"  - Solid: {cystic['solid_percentage']:.0f}%, Cystic: {cystic['cystic_percentage']:.0f}%")
    lines.append(f"  {cystic['description']}")
    if 'signal_characteristics' in cystic:
        lines.append(f"  - T2 signal: {cystic['signal_characteristics']['t2_homogeneity']}")
        lines.append(f"  - FLAIR: {cystic['signal_characteristics']['flair_suppression']}")
    
    return "\n".join(lines)


def analyze_morphology(input_folder, segmentation_path, output_path=None):
    """
    Main function to analyze tumor morphology and margins.
    """
    input_folder = Path(input_folder)
    case_id = get_case_id(input_folder)
    print(f"Analyzing case: {case_id}")
    
    mri_paths = get_mri_paths(input_folder, case_id)
    
    print("Loading MRI data...")
    t1_data, _, t1_header = load_nifti(mri_paths['t1'])
    t1ce_data, _, _ = load_nifti(mri_paths['t1ce'])
    t2_data, _, _ = load_nifti(mri_paths['t2'])
    flair_data, _, _ = load_nifti(mri_paths['flair'])
    
    print("Loading segmentation mask...")
    seg_data, _, _ = load_nifti(segmentation_path)
    seg_data = np.round(seg_data).astype(np.int32)
    
    voxel_info = get_voxel_dimensions(t1_header)
    voxel_dims = voxel_info['dimensions_mm']
    
    # Get tumor masks
    tumor_masks = get_tumor_masks(seg_data)
    
    print("\n" + "="*60)
    print("STEP 4: TUMOR MORPHOLOGY AND MARGINS")
    print("="*60)
    
    # Shape descriptors
    print("\n--- Shape Analysis ---")
    shape_descriptors = calculate_shape_descriptors(seg_data, tumor_masks, voxel_dims)
    print(f"  Shape: {shape_descriptors['shape_classification']}")
    print(f"  Sphericity: {shape_descriptors['sphericity']:.2f}")
    print(f"  Elongation: {shape_descriptors['elongation']:.2f} ({shape_descriptors['elongation_classification']})")
    
    # Border regularity
    print("\n--- Border Regularity Analysis ---")
    border_regularity = analyze_border_regularity(tumor_masks['wt'], voxel_dims)
    print(f"  Classification: {border_regularity['classification']}")
    print(f"  Regularity score: {border_regularity['regularity_score']:.2f}")
    
    # Margin definition
    print("\n--- Margin Definition Analysis ---")
    margin_definition = analyze_margin_definition(t1ce_data, seg_data, tumor_masks, voxel_dims)
    print(f"  Classification: {margin_definition['classification']}")
    print(f"  Sharpness: {margin_definition['margin_sharpness']:.2f}")
    
    # Necrosis pattern
    print("\n--- Necrosis Pattern Analysis ---")
    necrosis_pattern = analyze_necrosis_pattern(seg_data, tumor_masks, voxel_dims)
    print(f"  Pattern: {necrosis_pattern['pattern']}")
    if necrosis_pattern['necrosis_present']:
        print(f"  Percentage: {necrosis_pattern['necrosis_percentage']:.0f}%")
    
    # Cystic vs Solid classification
    print("\n--- Cystic vs Solid Classification ---")
    cystic_solid = analyze_cystic_vs_solid(t1_data, t2_data, flair_data, seg_data, tumor_masks, voxel_dims)
    print(f"  Classification: {cystic_solid['classification']}")
    print(f"  Solid: {cystic_solid['solid_percentage']:.0f}%, Cystic: {cystic_solid['cystic_percentage']:.0f}%")
    
    # Compile results
    results = {
        'case_id': case_id,
        'step': 'Step 4 - Tumor morphology and margins',
        'voxel_info': voxel_info,
        'shape_descriptors': shape_descriptors,
        'border_regularity': border_regularity,
        'margin_definition': margin_definition,
        'necrosis_pattern': necrosis_pattern,
        'cystic_solid_classification': cystic_solid
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
        description='Step 4: Analyze tumor morphology and margins'
    )
    parser.add_argument('--input', required=True,
                        help='Input folder containing MRI sequences')
    parser.add_argument('--segmentation', required=True,
                        help='Path to segmentation mask (NIfTI)')
    parser.add_argument('--output', default=None,
                        help='Output path for JSON results')
    
    args = parser.parse_args()
    
    analyze_morphology(args.input, args.segmentation, args.output)


if __name__ == "__main__":
    main()
