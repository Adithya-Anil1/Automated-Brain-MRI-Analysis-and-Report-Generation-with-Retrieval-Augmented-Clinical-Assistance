TITLE: MRI Sequences for Brain Tumor Imaging
KEYWORDS: T1, T2, FLAIR, T1CE, contrast, sequences, modalities, MRI
VERSION: 1.0
---

# MRI Sequences for Brain Tumor Imaging

Brain tumor MRI protocols typically include four standard sequences, each providing different tissue contrast characteristics. Together, these sequences allow radiologists and automated segmentation algorithms to identify and delineate tumor sub-regions.

## T1-Weighted (T1)

- **Appearance**: Gray matter appears darker than white matter. CSF appears very dark (hypointense).
- **Strengths**: Excellent anatomical detail; good for visualizing brain structure and identifying mass effect.
- **Tumor relevance**: Tumors may appear iso- to hypointense. Used as a baseline for comparison with post-contrast images.

## T1-Weighted Post-Contrast (T1CE / T1Gd)

- **Appearance**: Same as T1, but after intravenous gadolinium contrast administration. Areas with blood-brain barrier breakdown appear bright.
- **Strengths**: Best sequence for identifying enhancing tumor tissue.
- **Tumor relevance**: Enhancing tumor regions appear hyperintense. Ring enhancement with a dark center suggests central necrosis, a hallmark of glioblastoma.

## T2-Weighted (T2)

- **Appearance**: CSF appears very bright (hyperintense). Gray matter is brighter than white matter.
- **Strengths**: Sensitive to water content; excellent for detecting edema and fluid-containing lesions.
- **Tumor relevance**: Both tumor and surrounding edema appear hyperintense, making it difficult to distinguish between the two on T2 alone.

## T2-FLAIR (Fluid-Attenuated Inversion Recovery)

- **Appearance**: Similar to T2, but with CSF signal suppressed (appears dark).
- **Strengths**: Edema and tumor infiltration in periventricular regions are more conspicuous because the bright CSF signal is removed.
- **Tumor relevance**: Peritumoral edema and non-enhancing tumor infiltration are best visualized on FLAIR. Enhancing tumor is better characterized on T1CE.

## Why All Four Sequences Are Needed

Each MRI sequence highlights different tissue properties. By combining information from all four sequences, segmentation algorithms (such as nnU-Net used in this project) can accurately distinguish between:

- Enhancing tumor (best seen on T1CE)
- Non-enhancing tumor / necrotic core (T1 vs T1CE comparison)
- Peritumoral edema (best seen on T2/FLAIR)
- Normal brain tissue (all sequences combined)
