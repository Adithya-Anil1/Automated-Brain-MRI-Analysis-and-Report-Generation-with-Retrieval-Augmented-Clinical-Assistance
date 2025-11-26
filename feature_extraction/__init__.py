"""
Feature Extraction Module for Brain Tumor MRI Analysis

This module provides tools to extract clinically relevant features from 
brain MRI scans and tumor segmentation masks for radiology report generation.

Steps:
1. Sequence-specific findings (T1, T2, FLAIR, contrast)
2. Mass effect metrics (midline shift, ventricular compression)
3. Anatomical context (lobe identification, hemisphere)
4. Multiplicity (single vs multiple lesions)
5. Morphology (shape, margins, necrosis)
6. Quality control (confidence, artifacts)
7. Normal structures assessment

Author: AI-Powered Brain MRI Assistant
Date: November 27, 2025
"""

from .utils import load_nifti, get_intensity_stats, NumpyEncoder

__version__ = "1.0.0"
__all__ = [
    'load_nifti',
    'get_intensity_stats', 
    'NumpyEncoder'
]
