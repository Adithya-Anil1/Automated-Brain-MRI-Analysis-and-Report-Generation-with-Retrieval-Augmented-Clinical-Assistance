# 4-Level Therapy Prediction Framework

## Overview

This project implements a hierarchical AI pipeline for analyzing longitudinal Brain MRI (Glioblastoma) data from the BraTS 2025 (LUMIERE) dataset. The system moves from retrospective assessment to prospective prediction, with increasing layers of interpretability and complexity.

## Level 1: Therapy Response Assessment (Retrospective)
*"Did the tumor respond to treatment?"*

**Goal:** Classify patient outcomes as "Progression" (Failure) or "Stable/Response" (Success) based on longitudinal changes.

**Input Data:**
- Timepoint 1: Pre-treatment MRI (Baseline)
- Timepoint 2: Post-treatment Follow-up MRI
- Features: "Delta" features (Volume Change, Midline Shift Change, Necrosis Growth)

**Method:**
- Rule-Based Logic: Immediate "Progression" flag if Midline Shift > 10mm or Volume Increase > 25%
- XGBoost Classifier: Validates the assessment for borderline cases

**Ground Truth:** RANO (Response Assessment in Neuro-Oncology) labels from the dataset

**Output:** Binary Classification (Progressive Disease vs. Stable Disease)

## Level 2: Explainable AI (Evidence & Reasoning)
*"Why did the model make this decision?"*

**Goal:** Convert "Black Box" predictions into clinically interpretable reasoning.

**Input Data:** The trained XGBoost models from Level 1 and Level 4.

**Method:**
- SHAP (SHapley Additive exPlanations): Calculates the marginal contribution of each radiomic feature to the final prediction.

**Output:**
- Feature Importance Plots: Global ranking of top biomarkers (e.g., "Necrosis Volume was the #1 predictor")
- Individual Force Plots: Patient-specific explanations (e.g., "For Patient-007, high Sphericity pushed the probability towards 'Methylated'")

## Level 3: Uncertainty Quantification (Confidence)
*"How sure is the system in this assessment?"*

**Goal:** Prevent overconfidence in clinical decision support by flagging "Gray Zone" cases.

**Input Data:** The raw probability scores (predict_proba) from the XGBoost models.

**Method:**
- Confidence Stratification:
  - High Confidence: Probability > 80% or < 20%
  - Moderate Confidence: Probability 60-80% or 20-40%
  - Indeterminate: Probability 40-60% (The "Gray Zone")

**Output:** A confidence band accompanying every prediction (e.g., "Prediction: Methylated (Moderate Confidence)")

## Level 4: Early Molecular Prediction (Prospective)
*"Will the tumor respond to therapy? (The Virtual Biopsy)"*

**Goal:** Predict the tumor's genetic sensitivity to chemotherapy before treatment begins.

**Input Data:**
- Timepoint 1 ONLY: Pre-treatment MRI (Baseline)
- Features: Static Radiomics (Texture Heterogeneity, Shape Irregularity, Intensity distributions)
- Target Label: MGMT Promoter Methylation Status (Methylated = Sensitive, Unmethylated = Resistant)

**Method:**
- Radiogenomics: Train XGBoost to map imaging phenotypes to genetic genotypes
- Validation: Leave-One-Out Cross-Validation (LOOCV) due to dataset size (N=80)

**Output:** "Predicted MGMT Status" (Chemo-Sensitive vs. Chemo-Resistant)

## Dataset Specifications (BraTS 2025 / LUMIERE)

**Source:** LUMIERE Dataset (Longitudinal Glioblastoma MRI with Expert RANO Evaluation)

**Sample Size:** ~91 Patients total

**Level 1 Subset:** Patients with valid Pre+Post pairs (~91 cases)

**Level 4 Subset:** Patients with valid MGMT labels (~80 cases: 37 Methylated, 43 Unmethylated)

**Key Files:**
- Imaging-v202211: Raw NIfTI MRI scans
- LUMIERE-ExpertRating.csv: RANO labels (Target for Level 1)
- LUMIERE_dataset_-_Demographics_and_pathology_information.csv: MGMT labels (Target for Level 4)
