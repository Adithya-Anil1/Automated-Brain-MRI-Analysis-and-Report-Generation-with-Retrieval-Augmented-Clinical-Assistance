#!/usr/bin/env python3
"""
10_xgboost_rano.py

Treatment Response Prediction using XGBoost and Radiomics.

CRITICAL: Uses PRE-TREATMENT radiomics to predict POST-TREATMENT RANO response.

Pipeline:
1. Load pre-treatment radiomics, post-treatment RANO labels, and clinical data
2. Feature preprocessing (low-variance, correlation, standardization)
3. XGBoost classification with class balancing
4. 5-fold stratified cross-validation
5. Feature importance analysis

Author: AI-Powered Brain MRI Assistant
Date: 2026-02-01
"""

import os
import re
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix
)

import xgboost as xgb

warnings.filterwarnings('ignore')

# Set random seed
SEED = 42
np.random.seed(SEED)

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent
RADIOMICS_PATH = BASE_DIR / "previous_mgmt_attempts" / "Level4_Radiomic_Features_Enhancing.csv"
RANO_PATH = BASE_DIR / "dataset" / "LUMIERE-ExpertRating-v202211.csv"
CLINICAL_PATH = BASE_DIR / "LUMIERE-Demographics_Pathology.csv"
OUTPUT_DIR = BASE_DIR / "xgboost_rano_results"

# Minimum week for post-treatment RANO rating
MIN_WEEK_FOR_LABEL = 12


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_week_number(week_str):
    """Extract numeric week from string like 'week-044' or 'week-000-1'."""
    if pd.isna(week_str):
        return None
    match = re.search(r'week-(\d+)', str(week_str))
    if match:
        return int(match.group(1))
    return None


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "-"*60)
    print(title)
    print("-"*60)


# =============================================================================
# STEP 1: DATA LOADING (LEAKAGE-FREE)
# =============================================================================

def load_radiomics():
    """
    Load pre-treatment radiomics features.
    
    Returns:
        pd.DataFrame: Radiomics features with Patient_ID
    """
    print("\nLoading radiomics features...")
    
    if not RADIOMICS_PATH.exists():
        raise FileNotFoundError(f"Radiomics file not found: {RADIOMICS_PATH}")
    
    df = pd.read_csv(RADIOMICS_PATH)
    
    # The radiomics CSV contains pre-treatment features (baseline scans)
    # Column structure: Patient_ID, Used_Label, [radiomic features...]
    
    # Identify feature columns (exclude Patient_ID and non-feature columns)
    non_feature_cols = ['Patient_ID', 'Used_Label', 'Timepoint', 'Week']
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    
    # Keep only Patient_ID and features
    keep_cols = ['Patient_ID'] + feature_cols
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]
    
    # Drop columns with >50% missing values
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()
    if 'Patient_ID' in cols_to_drop:
        cols_to_drop.remove('Patient_ID')
    df = df.drop(columns=cols_to_drop)
    
    n_patients = df['Patient_ID'].nunique()
    n_features = len(df.columns) - 1  # Exclude Patient_ID
    
    print(f"  Loaded: {n_patients} patients, {n_features} features")
    if cols_to_drop:
        print(f"  Dropped {len(cols_to_drop)} columns with >50% missing values")
    
    return df


def load_rano_labels():
    """
    Load post-treatment RANO labels (Week >= 12).
    
    Returns:
        pd.DataFrame: Patient_ID, RANO_Label (binary)
    """
    print("\nLoading RANO labels (post-treatment, Week >= 12)...")
    
    if not RANO_PATH.exists():
        raise FileNotFoundError(f"RANO file not found: {RANO_PATH}")
    
    df = pd.read_csv(RANO_PATH)
    
    # Rename columns for consistency
    df = df.rename(columns={'Patient': 'Patient_ID'})
    
    # Find the RANO rating column
    rano_col = None
    for col in df.columns:
        if "rating" in col.lower() and "rationale" not in col.lower():
            rano_col = col
            break
    
    if rano_col is None:
        raise ValueError("Could not find RANO rating column!")
    
    # Filter for valid treatment responses (exclude Pre-Op, Post-Op)
    valid_ratings = ['CR', 'PR', 'SD', 'PD']
    df = df[df[rano_col].isin(valid_ratings)].copy()
    
    # Parse week numbers
    df['Week_Num'] = df['Date'].apply(parse_week_number)
    
    # Filter for Week >= MIN_WEEK_FOR_LABEL (post-treatment)
    df = df[df['Week_Num'] >= MIN_WEEK_FOR_LABEL]
    
    # For each patient, get the EARLIEST post-treatment RANO rating
    patient_labels = []
    for patient_id in df['Patient_ID'].unique():
        patient_data = df[df['Patient_ID'] == patient_id]
        earliest = patient_data.loc[patient_data['Week_Num'].idxmin()]
        
        rano_rating = earliest[rano_col]
        week = earliest['Week_Num']
        
        # Binary label: CR/PR = 1 (Responder), SD/PD = 0 (Non-Responder)
        if rano_rating in ['CR', 'PR']:
            label = 1
        else:  # SD, PD
            label = 0
        
        patient_labels.append({
            'Patient_ID': patient_id,
            'RANO': rano_rating,
            'Week': week,
            'Label': label
        })
    
    labels_df = pd.DataFrame(patient_labels)
    
    # Summary
    responders = (labels_df['Label'] == 1).sum()
    non_responders = (labels_df['Label'] == 0).sum()
    
    print(f"  Loaded: {len(labels_df)} patients with valid RANO (Week >= {MIN_WEEK_FOR_LABEL})")
    print(f"  RANO breakdown:")
    for rating in ['CR', 'PR', 'SD', 'PD']:
        count = (labels_df['RANO'] == rating).sum()
        print(f"    {rating}: {count}")
    
    return labels_df[['Patient_ID', 'Label']]


def load_clinical_data():
    """
    Load clinical data (Age, Sex, MGMT).
    
    Returns:
        pd.DataFrame: Patient_ID, Age, Sex, MGMT_Status
    """
    print("\nLoading clinical data...")
    
    if not CLINICAL_PATH.exists():
        raise FileNotFoundError(f"Clinical file not found: {CLINICAL_PATH}")
    
    df = pd.read_csv(CLINICAL_PATH)
    
    # Rename columns
    df = df.rename(columns={
        'Patient': 'Patient_ID',
        'Age at surgery (years)': 'Age',
        'MGMT qualitative': 'MGMT_Status'
    })
    
    # Select relevant columns
    keep_cols = ['Patient_ID', 'Age', 'Sex', 'MGMT_Status']
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
    
    # Encode Sex: female=0, male=1
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
    
    # Encode MGMT: methylated=1, not methylated=0
    if 'MGMT_Status' in df.columns:
        df['MGMT_Status'] = df['MGMT_Status'].map({
            'methylated': 1, 
            'not methylated': 0
        })
    
    print(f"  Loaded: {len(df)} patients with clinical data")
    
    return df


def merge_datasets(radiomics_df, labels_df, clinical_df):
    """
    Merge all datasets on Patient_ID (inner join).
    
    Returns:
        tuple: (merged_df, feature_columns, label_column)
    """
    print("\nMerging datasets (inner join on Patient_ID)...")
    
    # Merge radiomics with labels
    merged = radiomics_df.merge(labels_df, on='Patient_ID', how='inner')
    print(f"  After radiomics + labels: {len(merged)} patients")
    
    # Merge with clinical data
    merged = merged.merge(clinical_df, on='Patient_ID', how='inner')
    print(f"  After adding clinical: {len(merged)} patients")
    
    # Identify feature columns
    non_feature_cols = ['Patient_ID', 'Label', 'Used_Label']
    feature_cols = [c for c in merged.columns if c not in non_feature_cols]
    
    # Class distribution
    responders = (merged['Label'] == 1).sum()
    non_responders = (merged['Label'] == 0).sum()
    
    print(f"\nFinal dataset:")
    print(f"  Patients: {len(merged)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Class distribution:")
    print(f"    Responders (CR/PR): {responders}")
    print(f"    Non-Responders (SD/PD): {non_responders}")
    
    return merged, feature_cols


# =============================================================================
# STEP 2: FEATURE PREPROCESSING
# =============================================================================

def preprocess_features(df, feature_cols):
    """
    Preprocess features:
    1. Remove low-variance features (variance < 0.01)
    2. Remove highly correlated features (correlation > 0.95)
    3. Return preprocessed feature matrix
    
    Returns:
        tuple: (X, y, final_feature_names, preprocessing_info)
    """
    print_section("FEATURE PREPROCESSING")
    
    original_count = len(feature_cols)
    print(f"Original features: {original_count}")
    
    # Extract features and labels
    X = df[feature_cols].copy()
    y = df['Label'].values
    
    # Step 1: Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Step 2: Remove low-variance features (variance < 0.01 after standardization)
    # First standardize to compare variances fairly
    scaler_temp = StandardScaler()
    X_scaled_temp = scaler_temp.fit_transform(X_imputed)
    variances = np.var(X_scaled_temp, axis=0)
    
    low_var_mask = variances >= 0.01
    low_var_removed = (~low_var_mask).sum()
    
    X_filtered = X_imputed.iloc[:, low_var_mask]
    feature_names = X_filtered.columns.tolist()
    
    print(f"After low-variance removal: {len(feature_names)} (removed {low_var_removed})")
    
    # Step 3: Remove highly correlated features (correlation > 0.95)
    if len(feature_names) > 1:
        corr_matrix = X_filtered.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = set()
        for col in upper.columns:
            high_corr = upper[col][upper[col] > 0.95].index.tolist()
            to_drop.update(high_corr)
        
        X_filtered = X_filtered.drop(columns=list(to_drop))
        feature_names = X_filtered.columns.tolist()
        
        print(f"After correlation removal: {len(feature_names)} (removed {len(to_drop)})")
    
    # Step 4: Final standardization
    scaler = StandardScaler()
    X_final = scaler.fit_transform(X_filtered)
    
    print(f"Final features: {len(feature_names)}")
    
    preprocessing_info = {
        'original_count': original_count,
        'after_low_variance': original_count - low_var_removed,
        'after_correlation': len(feature_names),
        'final_count': len(feature_names)
    }
    
    return X_final, y, feature_names, preprocessing_info


# =============================================================================
# STEP 3 & 4: CLASS IMBALANCE AND XGBOOST CONFIGURATION
# =============================================================================

def calculate_class_imbalance(y):
    """Calculate and print class imbalance ratio."""
    n_responders = (y == 1).sum()
    n_non_responders = (y == 0).sum()
    
    if n_responders > 0:
        ratio = n_non_responders / n_responders
    else:
        ratio = 1.0
    
    print(f"\nClass imbalance ratio: {ratio:.2f} (Non-Responders / Responders)")
    
    return ratio


def get_xgboost_model(class_weight_ratio):
    """
    Create XGBoost classifier with specified configuration.
    
    Args:
        class_weight_ratio: scale_pos_weight value
        
    Returns:
        XGBClassifier
    """
    model = xgb.XGBClassifier(
        scale_pos_weight=class_weight_ratio,
        max_depth=3,
        n_estimators=100,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )
    
    return model


# =============================================================================
# STEP 5: CROSS-VALIDATION EVALUATION
# =============================================================================

def cross_validate(X, y, class_weight_ratio, feature_names):
    """
    Perform 5-fold stratified cross-validation.
    
    Returns:
        dict: Metrics and predictions
    """
    print_section("CROSS-VALIDATION RESULTS (5-FOLD)")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    # Store metrics for each fold
    metrics = {
        'roc_auc': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # Store predictions for aggregated confusion matrix
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        model = get_xgboost_model(class_weight_ratio)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = 0.5
        
        acc = accuracy_score(y_test, y_pred)
        
        # Handle case where no positive predictions
        try:
            prec = precision_score(y_test, y_pred, zero_division=0)
        except:
            prec = 0.0
        
        try:
            rec = recall_score(y_test, y_pred, zero_division=0)
        except:
            rec = 0.0
        
        try:
            f1 = f1_score(y_test, y_pred, zero_division=0)
        except:
            f1 = 0.0
        
        metrics['roc_auc'].append(auc)
        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1'].append(f1)
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        
        print(f"Fold {fold}: AUC={auc:.4f}, Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    
    # Print summary
    print("\n" + "-"*40)
    print("Summary (Mean ± Std):")
    for metric_name, values in metrics.items():
        # Filter out NaN values for mean/std calculation
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            print(f"  {metric_name.upper():12s}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print(f"  {metric_name.upper():12s}: N/A (no valid folds)")
    
    # Aggregated confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(f"\nAggregated Confusion Matrix:")
    print(f"  [[TN={cm[0,0]:3d}  FP={cm[0,1]:3d}]")
    print(f"   [FN={cm[1,0]:3d}  TP={cm[1,1]:3d}]]")
    
    results = {
        'metrics': {k: {
            'mean': float(np.nanmean(v)), 
            'std': float(np.nanstd(v)), 
            'folds': [float(x) if not np.isnan(x) else None for x in v]
        } for k, v in metrics.items()},
        'confusion_matrix': cm.tolist(),
        'all_y_true': [int(x) for x in all_y_true],
        'all_y_pred': [int(x) for x in all_y_pred],
        'all_y_proba': [float(x) for x in all_y_proba]
    }
    
    return results


# =============================================================================
# STEP 6: FEATURE IMPORTANCE
# =============================================================================

def analyze_feature_importance(X, y, class_weight_ratio, feature_names):
    """
    Train model on entire dataset and extract feature importance.
    
    Returns:
        pd.DataFrame: Features ranked by importance
    """
    print_section("TOP 10 IMPORTANT FEATURES")
    
    # Train on full dataset
    model = get_xgboost_model(class_weight_ratio)
    model.fit(X, y)
    
    # Get feature importance (gain)
    importance = model.get_booster().get_score(importance_type='gain')
    
    # Convert to DataFrame
    importance_df = pd.DataFrame([
        {'feature': feature_names[int(k.replace('f', ''))], 'importance': v}
        for k, v in importance.items()
    ])
    
    # If no importance scores (rare), create empty DataFrame
    if len(importance_df) == 0:
        print("  No features had importance > 0")
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': [0.0] * len(feature_names)
        })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Print top 10
    top_10 = importance_df.head(10)
    for i, row in top_10.iterrows():
        print(f"  {i+1:2d}. {row['feature'][:50]:50s}: {row['importance']:.4f}")
    
    return importance_df, model


# =============================================================================
# STEP 7: SAVE RESULTS
# =============================================================================

def save_results(cv_results, importance_df, preprocessing_info, class_ratio):
    """Save all results to files."""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Convert confusion matrix to native Python types
    cm_list = [[int(x) for x in row] for row in cv_results['confusion_matrix']]
    
    # 1. Save results.json
    results_json = {
        'config': {
            'seed': SEED,
            'n_splits': 5,
            'min_week_for_label': MIN_WEEK_FOR_LABEL,
            'class_imbalance_ratio': float(class_ratio),
            'xgboost_params': {
                'max_depth': 3,
                'n_estimators': 100,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        },
        'preprocessing': {k: int(v) if isinstance(v, (np.integer, np.int32, np.int64)) else v 
                         for k, v in preprocessing_info.items()},
        'metrics': cv_results['metrics'],
        'confusion_matrix': cm_list
    }
    
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR / 'results.json'}")
    
    # 2. Save feature_importance.csv
    importance_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
    print(f"Saved: {OUTPUT_DIR / 'feature_importance.csv'}")
    
    # 3. Save feature_importance.png
    plt.figure(figsize=(10, 8))
    top_20 = importance_df.head(20)
    
    if len(top_20) > 0 and top_20['importance'].sum() > 0:
        # Truncate long feature names
        top_20 = top_20.copy()
        top_20['feature_short'] = top_20['feature'].apply(lambda x: x[:40] + '...' if len(x) > 40 else x)
        
        plt.barh(range(len(top_20)), top_20['importance'].values, color='steelblue')
        plt.yticks(range(len(top_20)), top_20['feature_short'].values)
        plt.gca().invert_yaxis()
        plt.xlabel('Importance (Gain)')
        plt.title('Top 20 Features by XGBoost Importance')
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, 'No feature importance available', ha='center', va='center')
    
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'feature_importance.png'}")
    
    # 4. Save confusion_matrix.png
    cm = np.array(cv_results['confusion_matrix'])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Responder', 'Responder'],
                yticklabels=['Non-Responder', 'Responder'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Aggregated Confusion Matrix (5-Fold CV)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'confusion_matrix.png'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("XGBOOST TREATMENT RESPONSE PREDICTION")
    print("Pre-treatment Radiomics → Post-treatment RANO Response")
    print("="*60)
    
    try:
        # =====================================================================
        # STEP 1: DATA LOADING
        # =====================================================================
        print_section("DATA LOADING")
        
        radiomics_df = load_radiomics()
        labels_df = load_rano_labels()
        clinical_df = load_clinical_data()
        
        merged_df, feature_cols = merge_datasets(radiomics_df, labels_df, clinical_df)
        
        if len(merged_df) < 10:
            print("\nERROR: Not enough patients for cross-validation!")
            return
        
        # =====================================================================
        # STEP 2: FEATURE PREPROCESSING
        # =====================================================================
        
        X, y, feature_names, preprocessing_info = preprocess_features(merged_df, feature_cols)
        
        if len(feature_names) == 0:
            print("\nERROR: No features remaining after preprocessing!")
            return
        
        # =====================================================================
        # STEP 3: CLASS IMBALANCE
        # =====================================================================
        
        class_ratio = calculate_class_imbalance(y)
        
        # =====================================================================
        # STEP 4 & 5: CROSS-VALIDATION
        # =====================================================================
        
        cv_results = cross_validate(X, y, class_ratio, feature_names)
        
        # =====================================================================
        # STEP 6: FEATURE IMPORTANCE
        # =====================================================================
        
        importance_df, final_model = analyze_feature_importance(X, y, class_ratio, feature_names)
        
        # =====================================================================
        # STEP 7: SAVE RESULTS
        # =====================================================================
        
        save_results(cv_results, importance_df, preprocessing_info, class_ratio)
        
        # =====================================================================
        # FINAL SUMMARY
        # =====================================================================
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        
        # Handle NaN in final summary
        auc_mean = cv_results['metrics']['roc_auc']['mean']
        auc_std = cv_results['metrics']['roc_auc']['std']
        if np.isnan(auc_mean):
            print(f"\nFinal ROC-AUC: N/A (insufficient positive samples in some folds)")
        else:
            print(f"\nFinal ROC-AUC: {auc_mean:.4f} ± {auc_std:.4f}")
        print(f"Results saved to: {OUTPUT_DIR}")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please ensure all required data files exist.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
