TITLE: Tumor Volume Measurement
KEYWORDS: volume, measurement, cubic centimeters, segmentation, voxel, BraTS
VERSION: 1.0
---

# Tumor Volume Measurement

Tumor volume measurement is a quantitative assessment performed on brain MRI scans to determine the size of different tumor sub-regions. In automated brain tumor analysis, volumes are calculated from segmentation masks produced by deep learning models.

## How Volume Is Calculated

1. **Segmentation**: A trained neural network (e.g., nnU-Net) assigns a label to each voxel (3D pixel) in the MRI scan, classifying it as enhancing tumor, non-enhancing tumor, peritumoral edema, or normal tissue.
2. **Voxel counting**: The number of voxels assigned to each label is counted.
3. **Volume conversion**: The voxel count is multiplied by the volume of a single voxel (determined by the MRI scan's spatial resolution, typically 1mm × 1mm × 1mm = 1 mm³).
4. **Unit conversion**: The result is converted to cubic centimeters (cm³), where 1 cm³ = 1000 mm³.

## Reported Volumes in Brain Tumor Analysis

Typical volumetric measurements include:

- **Enhancing Tumor (ET)**: Volume of the contrast-enhancing tumor component.
- **Non-Enhancing Tumor Core (NCET)**: Volume of the non-enhancing solid tumor and necrotic regions.
- **Peritumoral Edema (ED)**: Volume of the surrounding edema/invaded tissue.
- **Whole Tumor (WT)**: Sum of all three components (ET + NCET + ED), representing the total lesion volume.

## Clinical Significance

- **Baseline assessment**: Initial tumor volumes provide a reference for monitoring disease progression.
- **Treatment response**: Volume changes over serial MRI scans can indicate whether a tumor is growing, stable, or responding to intervention.
- **Research**: Volumetric data from standardized segmentation frameworks (like BraTS) enable comparison across studies and institutions.

## Limitations

- Volume measurements depend on the accuracy of the segmentation model.
- Partial volume effects at tumor boundaries can introduce small measurement errors.
- Different segmentation approaches may produce slightly different volume estimates for the same scan.
