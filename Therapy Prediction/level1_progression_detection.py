#!/usr/bin/env python3
"""
level1_progression_detection.py

LEVEL 1: Baseline Tumor Progression Detection

This script implements a retrospective progression detection baseline that 
classifies whether a tumor shows Progression vs Non-Progression by comparing 
baseline and follow-up MRI scans.

OBJECTIVE:
- Binary classification: Progression (1) vs Non-Progression (0)
- Uses ONLY volumetric features from tumor segmentations
- Labels derived EXCLUSIVELY from RANO clinical assessment file

CRITICAL CONSTRAINTS:
- No deep learning / CNNs
- No radiomics libraries  
- No black-box features (no embeddings, no latent vectors)
- No single-timepoint inference
- No therapy response prediction
- No MGMT or molecular inference

Author: AI-Powered Brain MRI Assistant
Date: 2026-02-02
"""

import os
import warnings
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
BASE_DIR = Path(__file__).parent
LUMIERE_DATA_PATH = BASE_DIR / "lumiere_phase0.csv"
RANO_EXPERT_PATH = BASE_DIR / "dataset" / "LUMIERE-ExpertRating-v202211.csv"
OUTPUT_DIR = BASE_DIR / "level1_progression_results"

# Volume change threshold heuristic (RANO-inspired, NOT a clinical rule)
VOLUME_CHANGE_THRESHOLD = 0.25  # 25% volume increase as progression heuristic


# ============================================================================
# DATA LOADING AND PATIENT PAIRING
# ============================================================================

def load_rano_labels(filepath: Path) -> pd.DataFrame:
    """
    Load RANO clinical assessment labels.
    
    The RANO (Response Assessment in Neuro-Oncology) ratings are:
    - PD: Progressive Disease → Progression (1)
    - SD: Stable Disease → Non-Progression (0)  
    - PR: Partial Response → Non-Progression (0)
    - CR: Complete Response → Non-Progression (0)
    - Pre-Op/Post-Op: Excluded (not treatment timepoints)
    
    Returns:
        DataFrame with Patient, Date, and binary Progression_Label
    """
    print("\n" + "="*70)
    print("LOADING RANO CLINICAL ASSESSMENT LABELS")
    print("="*70)
    
    df = pd.read_csv(filepath)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Extract rating column (handle long column name)
    rating_col = [c for c in df.columns if 'Rating' in c and 'rationale' not in c.lower()][0]
    df = df.rename(columns={rating_col: 'RANO_Rating'})
    
    print(f"\nRaw RANO data: {len(df)} records")
    
    # Show RANO rating distribution
    print("\nRANO Rating Distribution (Original):")
    print(df['RANO_Rating'].value_counts())
    
    # Filter out Pre-Op and Post-Op (not valid treatment assessment timepoints)
    df = df[~df['RANO_Rating'].isin(['Pre-Op', 'Post-Op'])].copy()
    print(f"\nAfter excluding Pre-Op/Post-Op: {len(df)} records")
    
    # Create binary progression label from RANO ratings
    # PD (Progressive Disease) = 1 (Progression)
    # SD, PR, CR = 0 (Non-Progression)
    df['Progression_Label'] = (df['RANO_Rating'] == 'PD').astype(int)
    
    # Show binary label distribution
    print("\nBinary Progression Label Distribution:")
    prog_count = df['Progression_Label'].sum()
    non_prog_count = len(df) - prog_count
    print(f"  Progression (PD): {prog_count} ({100*prog_count/len(df):.1f}%)")
    print(f"  Non-Progression (SD/PR/CR): {non_prog_count} ({100*non_prog_count/len(df):.1f}%)")
    
    return df[['Patient', 'Date', 'RANO_Rating', 'Progression_Label']]


def load_volume_data(filepath: Path) -> pd.DataFrame:
    """
    Load volumetric data from LUMIERE phase0 file.
    
    This file contains pre-computed tumor volumes at baseline and follow-up.
    
    Returns:
        DataFrame with patient volumes and timepoint information
    """
    print("\n" + "="*70)
    print("LOADING VOLUMETRIC DATA")
    print("="*70)
    
    df = pd.read_csv(filepath)
    
    print(f"\nVolumetric data: {len(df)} patient-timepoint pairs")
    print(f"Unique patients: {df['Patient_ID'].nunique()}")
    print(f"\nColumns: {list(df.columns)}")
    
    # Show volume statistics
    print("\nVolume Statistics (ml):")
    print(f"  Baseline: mean={df['Baseline_Volume_ml'].mean():.2f}, "
          f"std={df['Baseline_Volume_ml'].std():.2f}, "
          f"range=[{df['Baseline_Volume_ml'].min():.2f}, {df['Baseline_Volume_ml'].max():.2f}]")
    print(f"  Follow-up: mean={df['Followup_Volume_ml'].mean():.2f}, "
          f"std={df['Followup_Volume_ml'].std():.2f}, "
          f"range=[{df['Followup_Volume_ml'].min():.2f}, {df['Followup_Volume_ml'].max():.2f}]")
    
    return df


def pair_patients_with_labels(
    volume_df: pd.DataFrame, 
    rano_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Pair volumetric data with RANO clinical assessment labels.
    
    CRITICAL: Labels come from RANO file, NOT derived from volume changes.
    
    This function:
    1. Matches each baseline-followup pair with the RANO assessment at follow-up
    2. Excludes patients missing either timepoint
    3. Uses ONLY RANO-derived progression labels
    
    Returns:
        DataFrame with paired volumetric data and RANO labels
    """
    print("\n" + "="*70)
    print("PAIRING PATIENTS: VOLUMES + RANO LABELS")
    print("="*70)
    
    # Standardize patient ID format
    volume_df = volume_df.copy()
    volume_df['Patient_ID'] = volume_df['Patient_ID'].str.strip()
    
    rano_df = rano_df.copy()
    rano_df['Patient'] = rano_df['Patient'].str.strip()
    
    # Match on patient ID and follow-up week
    volume_df['Followup_Week_Clean'] = volume_df['Followup_Week'].str.strip()
    rano_df['Date_Clean'] = rano_df['Date'].str.strip()
    
    # Merge volumetric data with RANO labels
    merged = volume_df.merge(
        rano_df[['Patient', 'Date_Clean', 'RANO_Rating', 'Progression_Label']],
        left_on=['Patient_ID', 'Followup_Week_Clean'],
        right_on=['Patient', 'Date_Clean'],
        how='inner'
    )
    
    print(f"\nSuccessfully matched {len(merged)} baseline-followup pairs with RANO labels")
    print(f"Unique patients: {merged['Patient_ID'].nunique()}")
    
    # Verify label distribution
    print("\nMatched Label Distribution:")
    prog = merged['Progression_Label'].sum()
    non_prog = len(merged) - prog
    print(f"  Progression: {prog} ({100*prog/len(merged):.1f}%)")
    print(f"  Non-Progression: {non_prog} ({100*non_prog/len(merged):.1f}%)")
    
    # CRITICAL CHECK: Ensure we're using RANO labels, NOT volume-derived labels
    # The Response_Label column in volume_df should match (but we use Progression_Label)
    if 'Response_Label' in merged.columns:
        response_map = {'Progression': 1, 'Stable': 0, 'Response': 0}
        merged['Response_Binary'] = merged['Response_Label'].map(response_map)
        agreement = (merged['Progression_Label'] == merged['Response_Binary']).mean()
        print(f"\nLabel agreement check (RANO vs file labels): {100*agreement:.1f}%")
        if agreement < 0.95:
            print("  WARNING: Some discrepancy between RANO and file labels!")
            print("  Using RANO clinical assessment labels as ground truth.")
    
    return merged


# ============================================================================
# FEATURE ENGINEERING (VOLUME-BASED ONLY)
# ============================================================================

def extract_volumetric_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extract volumetric features from paired baseline-followup data.
    
    Features computed (as per requirements):
    1. V_base: Tumor volume at baseline (ml)
    2. V_follow: Tumor volume at follow-up (ml)
    3. Delta_V: Absolute volume change = V_follow - V_base (ml)
    4. Delta_V_percent: Relative volume change = (V_follow - V_base) / V_base
    
    Edge-case handling:
    - If V_base = 0: Skip sample (cannot compute relative change)
    - Exclude patients missing either timepoint (already handled in pairing)
    
    Returns:
        X: Feature matrix (n_samples, 4)
        y: Binary progression labels
        feature_df: DataFrame with features and metadata
    """
    print("\n" + "="*70)
    print("EXTRACTING VOLUMETRIC FEATURES")
    print("="*70)
    
    df = df.copy()
    
    # Extract raw volumes
    df['V_base'] = df['Baseline_Volume_ml']
    df['V_follow'] = df['Followup_Volume_ml']
    
    # Compute absolute change
    df['Delta_V'] = df['V_follow'] - df['V_base']
    
    # Compute relative change with edge-case handling
    # Handle V_base = 0: these samples cannot have valid relative change
    zero_baseline_mask = df['V_base'] == 0
    n_zero_baseline = zero_baseline_mask.sum()
    
    if n_zero_baseline > 0:
        print(f"\nWARNING: Found {n_zero_baseline} samples with V_base = 0")
        print("  These samples will be EXCLUDED (cannot compute relative change)")
        df = df[~zero_baseline_mask].copy()
    
    # Compute relative change (safe now that V_base > 0)
    df['Delta_V_percent'] = (df['V_follow'] - df['V_base']) / df['V_base']
    
    # Feature names
    feature_names = ['V_base', 'V_follow', 'Delta_V', 'Delta_V_percent']
    
    print(f"\nFinal dataset: {len(df)} samples")
    print(f"Features: {feature_names}")
    
    # Feature statistics
    print("\nFeature Statistics:")
    for feat in feature_names:
        print(f"  {feat}: mean={df[feat].mean():.4f}, std={df[feat].std():.4f}, "
              f"range=[{df[feat].min():.4f}, {df[feat].max():.4f}]")
    
    # Prepare feature matrix and labels
    X = df[feature_names].values
    y = df['Progression_Label'].values
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Label distribution: {np.sum(y)} Progression, {len(y) - np.sum(y)} Non-Progression")
    
    return X, y, df


# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

def train_logistic_regression(
    X: np.ndarray, 
    y: np.ndarray,
    n_folds: int = 5
) -> Dict:
    """
    Train Logistic Regression classifier with cross-validation.
    
    This is the preferred model per Level 1 requirements:
    - Simple and explainable
    - No deep learning
    - Coefficients provide direct feature importance
    
    Returns:
        Dictionary with model, predictions, and metrics
    """
    print("\n" + "="*70)
    print("TRAINING: LOGISTIC REGRESSION")
    print("="*70)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize model
    model = LogisticRegression(
        random_state=RANDOM_SEED,
        max_iter=1000,
        class_weight='balanced'  # Handle any class imbalance
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    # Get cross-validated predictions
    y_pred_proba = cross_val_predict(model, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Fit final model on all data for coefficient extraction
    model.fit(X_scaled, y)
    
    # Compute metrics
    metrics = compute_metrics(y, y_pred, y_pred_proba)
    
    # Extract feature importance (coefficients)
    feature_names = ['V_base', 'V_follow', 'Delta_V', 'Delta_V_percent']
    coefficients = dict(zip(feature_names, model.coef_[0]))
    
    print(f"\n{n_folds}-Fold Cross-Validation Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print("\nFeature Coefficients (Importance):")
    for feat, coef in sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat}: {coef:+.4f}")
    
    # DATA LEAKAGE CHECK
    check_for_data_leakage(metrics, "Logistic Regression")
    
    return {
        'model': model,
        'scaler': scaler,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'metrics': metrics,
        'coefficients': coefficients,
        'feature_names': feature_names
    }


def train_random_forest(
    X: np.ndarray, 
    y: np.ndarray,
    n_folds: int = 5
) -> Dict:
    """
    Train shallow Random Forest classifier with cross-validation.
    
    Allowed per Level 1 requirements with constraint:
    - max_depth ≤ 3 (shallow trees for explainability)
    
    Returns:
        Dictionary with model, predictions, and metrics
    """
    print("\n" + "="*70)
    print("TRAINING: RANDOM FOREST (Shallow, max_depth=3)")
    print("="*70)
    
    # Initialize model (shallow for explainability)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,  # CRITICAL: Shallow as per requirements
        random_state=RANDOM_SEED,
        class_weight='balanced',
        n_jobs=-1
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    # Get cross-validated predictions
    y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Fit final model on all data for feature importance
    model.fit(X, y)
    
    # Compute metrics
    metrics = compute_metrics(y, y_pred, y_pred_proba)
    
    # Extract feature importance
    feature_names = ['V_base', 'V_follow', 'Delta_V', 'Delta_V_percent']
    importances = dict(zip(feature_names, model.feature_importances_))
    
    print(f"\n{n_folds}-Fold Cross-Validation Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print("\nFeature Importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {imp:.4f}")
    
    # DATA LEAKAGE CHECK
    check_for_data_leakage(metrics, "Random Forest")
    
    return {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'metrics': metrics,
        'importances': importances,
        'feature_names': feature_names
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
    """Compute classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }


def check_for_data_leakage(metrics: Dict, model_name: str):
    """
    Check for potential data leakage if model accuracy is suspiciously high.
    
    Per Level 1 requirements: "If model accuracy approaches near-perfect levels, 
    explicitly flag and investigate data leakage"
    """
    LEAKAGE_THRESHOLD = 0.95
    
    if metrics['accuracy'] >= LEAKAGE_THRESHOLD:
        print("\n" + "!"*70)
        print("⚠️  DATA LEAKAGE WARNING ⚠️")
        print("!"*70)
        print(f"\n{model_name} achieved accuracy of {metrics['accuracy']:.4f}")
        print(f"This exceeds the suspicion threshold of {LEAKAGE_THRESHOLD:.2f}")
        print("\nPossible causes to investigate:")
        print("  1. Label leakage: Are labels derived from volume features?")
        print("  2. Feature leakage: Are future values included in features?")
        print("  3. Overfitting: Model memorizing training data")
        print("  4. Data duplication: Same samples in train and test")
        print("\nRECOMMENDATION: Review data pipeline for information leakage.")
        print("!"*70)
    elif metrics['accuracy'] >= 0.85:
        print(f"\n⚡ Note: High accuracy ({metrics['accuracy']:.4f}). "
              "Likely legitimate but worth monitoring.")


# ============================================================================
# CLINICAL SUMMARY AND EXPLANATIONS
# ============================================================================

def generate_clinical_summary(
    df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    model_results: Dict
) -> List[Dict]:
    """
    Generate per-sample clinical summaries with rule-based explanations.
    
    Each summary contains:
    - Prediction (Progression / Non-Progression)
    - Volume change values (absolute and relative)
    - Explanation text (rule-based)
    - Confidence score
    
    Returns:
        List of clinical summary dictionaries
    """
    print("\n" + "="*70)
    print("GENERATING CLINICAL SUMMARIES")
    print("="*70)
    
    summaries = []
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        pred = predictions[idx]
        prob = probabilities[idx]
        
        # Extract values
        patient_id = row['Patient_ID']
        v_base = row['V_base']
        v_follow = row['V_follow']
        delta_v = row['Delta_V']
        delta_v_pct = row['Delta_V_percent']
        true_label = row['Progression_Label']
        rano_rating = row['RANO_Rating']
        
        # Generate prediction label
        pred_label = "Progression" if pred == 1 else "Non-Progression"
        true_label_str = "Progression" if true_label == 1 else "Non-Progression"
        
        # Compute confidence
        confidence = prob if pred == 1 else (1 - prob)
        
        # Generate rule-based explanation
        explanation = generate_explanation(
            pred, delta_v_pct, v_base, v_follow, confidence
        )
        
        summary = {
            'patient_id': patient_id,
            'baseline_week': row.get('Baseline_Week', 'N/A'),
            'followup_week': row.get('Followup_Week', 'N/A'),
            'V_base_ml': round(v_base, 2),
            'V_follow_ml': round(v_follow, 2),
            'Delta_V_ml': round(delta_v, 2),
            'Delta_V_percent': round(delta_v_pct * 100, 1),
            'prediction': pred_label,
            'true_label': true_label_str,
            'rano_rating': rano_rating,
            'confidence': round(confidence, 3),
            'correct': pred == true_label,
            'explanation': explanation
        }
        
        summaries.append(summary)
    
    return summaries


def generate_explanation(
    prediction: int,
    delta_v_pct: float,
    v_base: float,
    v_follow: float,
    confidence: float
) -> str:
    """
    Generate rule-based explanation for each prediction.
    
    Uses a 25% volume-change heuristic inspired by RANO progression criteria.
    NOTE: This is a heuristic reference, NOT a clinical rule.
    """
    pct_change = delta_v_pct * 100  # Convert to percentage
    threshold_pct = VOLUME_CHANGE_THRESHOLD * 100  # 25%
    
    if prediction == 1:  # Predicted Progression
        if delta_v_pct > VOLUME_CHANGE_THRESHOLD:
            explanation = (
                f"Predicted PROGRESSION because tumor volume increased by "
                f"{pct_change:.1f}% between baseline ({v_base:.1f} ml) and "
                f"follow-up ({v_follow:.1f} ml), exceeding the {threshold_pct:.0f}% "
                f"volume-change heuristic inspired by RANO progression criteria. "
                f"Model confidence: {confidence:.1%}."
            )
        elif delta_v_pct > 0:
            explanation = (
                f"Predicted PROGRESSION with tumor volume increase of {pct_change:.1f}% "
                f"(baseline: {v_base:.1f} ml → follow-up: {v_follow:.1f} ml). "
                f"Although below the {threshold_pct:.0f}% heuristic threshold, "
                f"the model identified progression patterns. Confidence: {confidence:.1%}."
            )
        else:
            explanation = (
                f"Predicted PROGRESSION despite apparent volume decrease of {pct_change:.1f}% "
                f"(baseline: {v_base:.1f} ml → follow-up: {v_follow:.1f} ml). "
                f"This may indicate non-enhancing progression or measurement variability. "
                f"Confidence: {confidence:.1%}. Clinical review recommended."
            )
    else:  # Predicted Non-Progression
        if delta_v_pct < -0.3:  # > 30% decrease
            explanation = (
                f"Predicted NON-PROGRESSION (likely response) with significant tumor "
                f"volume decrease of {pct_change:.1f}% (baseline: {v_base:.1f} ml → "
                f"follow-up: {v_follow:.1f} ml). This suggests treatment response. "
                f"Confidence: {confidence:.1%}."
            )
        elif delta_v_pct < 0:
            explanation = (
                f"Predicted NON-PROGRESSION with tumor volume decrease of {pct_change:.1f}% "
                f"(baseline: {v_base:.1f} ml → follow-up: {v_follow:.1f} ml). "
                f"Volume remains stable or reduced. Confidence: {confidence:.1%}."
            )
        elif delta_v_pct <= VOLUME_CHANGE_THRESHOLD:
            explanation = (
                f"Predicted NON-PROGRESSION (stable disease) with volume change of "
                f"{pct_change:.1f}% (baseline: {v_base:.1f} ml → follow-up: {v_follow:.1f} ml). "
                f"Change is within the {threshold_pct:.0f}% stability threshold "
                f"inspired by RANO criteria. Confidence: {confidence:.1%}."
            )
        else:
            explanation = (
                f"Predicted NON-PROGRESSION despite volume increase of {pct_change:.1f}% "
                f"(baseline: {v_base:.1f} ml → follow-up: {v_follow:.1f} ml). "
                f"Model suggests stability based on learned patterns. "
                f"Confidence: {confidence:.1%}. Close monitoring advised."
            )
    
    return explanation


def print_clinical_summaries(summaries: List[Dict], n_samples: int = 10):
    """Print clinical summaries for a subset of samples."""
    print("\n" + "="*70)
    print(f"SAMPLE CLINICAL SUMMARIES (First {n_samples} samples)")
    print("="*70)
    
    for i, s in enumerate(summaries[:n_samples]):
        print(f"\n{'─'*70}")
        print(f"PATIENT: {s['patient_id']}")
        print(f"{'─'*70}")
        print(f"Timepoints: {s['baseline_week']} → {s['followup_week']}")
        print(f"Baseline Volume:  {s['V_base_ml']:.2f} ml")
        print(f"Follow-up Volume: {s['V_follow_ml']:.2f} ml")
        print(f"Volume Change:    {s['Delta_V_ml']:+.2f} ml ({s['Delta_V_percent']:+.1f}%)")
        print(f"\n📊 PREDICTION: {s['prediction']}")
        print(f"📋 TRUE LABEL: {s['true_label']} (RANO: {s['rano_rating']})")
        print(f"✓ CORRECT: {'Yes ✅' if s['correct'] else 'No ❌'}")
        print(f"🎯 CONFIDENCE: {s['confidence']:.1%}")
        print(f"\n💡 EXPLANATION:")
        print(f"   {s['explanation']}")


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_results(
    summaries: List[Dict],
    lr_results: Dict,
    rf_results: Dict,
    output_dir: Path
):
    """Save all results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save clinical summaries
    summaries_df = pd.DataFrame(summaries)
    summaries_path = output_dir / "clinical_summaries.csv"
    summaries_df.to_csv(summaries_path, index=False)
    print(f"\nSaved clinical summaries to: {summaries_path}")
    
    # Save model metrics
    metrics_data = {
        'Model': ['Logistic Regression', 'Random Forest (depth=3)'],
        'Accuracy': [lr_results['metrics']['accuracy'], rf_results['metrics']['accuracy']],
        'Precision': [lr_results['metrics']['precision'], rf_results['metrics']['precision']],
        'Recall': [lr_results['metrics']['recall'], rf_results['metrics']['recall']],
        'F1_Score': [lr_results['metrics']['f1'], rf_results['metrics']['f1']],
        'ROC_AUC': [lr_results['metrics']['roc_auc'], rf_results['metrics']['roc_auc']]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = output_dir / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved model metrics to: {metrics_path}")
    
    # Save feature importance comparison
    feature_names = lr_results['feature_names']
    importance_data = {
        'Feature': feature_names,
        'LR_Coefficient': [lr_results['coefficients'][f] for f in feature_names],
        'RF_Importance': [rf_results['importances'][f] for f in feature_names]
    }
    importance_df = pd.DataFrame(importance_data)
    importance_path = output_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"Saved feature importance to: {importance_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline for Level 1 progression detection."""
    
    print("\n" + "="*70)
    print("LEVEL 1: BASELINE TUMOR PROGRESSION DETECTION")
    print("="*70)
    print("\nObjective: Binary classification of Progression vs Non-Progression")
    print("Method: Volumetric features from baseline-followup MRI pairs")
    print("Labels: RANO clinical assessment (NOT derived from volumes)")
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    
    # Load RANO clinical labels
    rano_df = load_rano_labels(RANO_EXPERT_PATH)
    
    # Load volumetric data
    volume_df = load_volume_data(LUMIERE_DATA_PATH)
    
    # ========================================================================
    # STEP 2: Patient Pairing
    # ========================================================================
    
    # Pair volumes with RANO labels
    paired_df = pair_patients_with_labels(volume_df, rano_df)
    
    if len(paired_df) == 0:
        print("\n❌ ERROR: No matched patient pairs found!")
        print("Check that patient IDs and timepoint labels match between files.")
        return
    
    # ========================================================================
    # STEP 3: Feature Extraction
    # ========================================================================
    
    X, y, feature_df = extract_volumetric_features(paired_df)
    
    if len(X) < 10:
        print(f"\n⚠️ WARNING: Only {len(X)} samples available. Results may be unreliable.")
    
    # ========================================================================
    # STEP 4: Model Training
    # ========================================================================
    
    # Train Logistic Regression (preferred)
    lr_results = train_logistic_regression(X, y)
    
    # Train Random Forest (alternative)
    rf_results = train_random_forest(X, y)
    
    # ========================================================================
    # STEP 5: Generate Clinical Summaries
    # ========================================================================
    
    # Use Logistic Regression predictions (preferred model)
    summaries = generate_clinical_summary(
        feature_df,
        lr_results['predictions'],
        lr_results['probabilities'],
        lr_results
    )
    
    # Print sample summaries
    print_clinical_summaries(summaries, n_samples=10)
    
    # ========================================================================
    # STEP 6: Model Comparison
    # ========================================================================
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print("\n{:<25} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"
    ))
    print("-" * 75)
    
    for name, results in [("Logistic Regression", lr_results), 
                          ("Random Forest (d=3)", rf_results)]:
        m = results['metrics']
        print("{:<25} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            name, m['accuracy'], m['precision'], m['recall'], m['f1'], m['roc_auc']
        ))
    
    # ========================================================================
    # STEP 7: Save Results
    # ========================================================================
    
    save_results(summaries, lr_results, rf_results, OUTPUT_DIR)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("LEVEL 1 PIPELINE COMPLETE")
    print("="*70)
    print(f"\n✅ Processed {len(feature_df)} baseline-followup pairs")
    print(f"✅ Trained 2 explainable models (Logistic Regression, Random Forest)")
    print(f"✅ Generated {len(summaries)} clinical summaries with explanations")
    print(f"✅ Results saved to: {OUTPUT_DIR}")
    
    # Constraints verification
    print("\n📋 CONSTRAINT VERIFICATION:")
    print("  ✓ Used ONLY volumetric features (V_base, V_follow, Delta_V, Delta_V%)")
    print("  ✓ Labels from RANO clinical assessment (NOT derived from volumes)")
    print("  ✓ Simple classifiers (Logistic Regression, shallow Random Forest)")
    print("  ✓ No deep learning / CNNs")
    print("  ✓ No radiomics libraries")
    print("  ✓ Rule-based explanations for each prediction")
    print("  ✓ Two-timepoint inference (baseline + follow-up)")
    
    return {
        'feature_df': feature_df,
        'lr_results': lr_results,
        'rf_results': rf_results,
        'summaries': summaries
    }


if __name__ == "__main__":
    results = main()
