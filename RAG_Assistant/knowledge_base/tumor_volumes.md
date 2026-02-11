TITLE: Tumor Volume Measurement
KEYWORDS: volume, measurement, cubic centimeters, segmentation, voxel, BraTS, quantitative, volumetric
VERSION: 1.0
---

Definition:
Tumor volume measurement is a quantitative assessment performed on brain MRI scans to determine the size of different tumor sub-regions. In automated brain tumor analysis, volumes are calculated from segmentation masks produced by deep learning models. A trained neural network (such as nnU-Net) assigns a label to each voxel (3D pixel) in the MRI scan, classifying it as enhancing tumor, non-enhancing tumor, peritumoral edema, or normal tissue. The number of voxels assigned to each label is counted and multiplied by the volume of a single voxel (determined by the scan's spatial resolution, typically 1 mm x 1 mm x 1 mm = 1 mm³), then converted to cubic centimeters where 1 cm³ = 1000 mm³. Standard volumetric measurements include enhancing tumor (ET), non-enhancing tumor core (NCET), peritumoral edema (ED), and whole tumor (WT), which is the sum of all three components.

Why It Appears in the Report:
Volume measurements from the BraTS segmentation framework provide objective, reproducible quantification of each tumor sub-region. These values characterize lesion extent and enable standardized comparison across scans.

Typical Reporting Units:
Cubic centimeters (cm³).

What This Does NOT Mean:
This finding alone does not determine tumor type, grade, prognosis, or required treatment.

Source Authority:
Adapted from standard neuroradiology references and simplified for educational use.
