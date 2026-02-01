#!/usr/bin/env python3
"""
11_predict_survival.py

Survival Prediction from Pre-treatment Radiomics + Clinical Features.

BEST OPTION: Perfect class balance (43 vs 43)!
- Target: Long survival (>72 weeks) vs Short survival (≤72 weeks)
- Features: Pre-treatment radiomics + Age + Sex + MGMT status

This is the most viable prediction task in the LUMIERE dataset because:
1. Perfect 1:1 class balance (no imbalance issues!)
2. 85 patients with both radiomics and survival data
3. Clinically meaningful endpoint

Author: AI-Powered Brain MRI Assistant
Date: 2026-02-01
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, roc_curve,
    classification_report
)

import xgboost as xgb

warnings.filterwarnings('ignore')

# Random seed
SEED = 42
np.random.seed(SEED)

# Paths
BASE_DIR = Path(__file__).parent
RADIOMICS_PATH = BASE_DIR / "previous_mgmt_attempts" / "Level4_Radiomic_Features_Enhancing.csv"
CLINICAL_PATH = BASE_DIR / "LUMIERE-Demographics_Pathology.csv"
OUTPUT_DIR = BASE_DIR / "survival_prediction_results"


def load_data():
    """Load and merge radiomics with clinical data."""
    
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Load radiomics
    radiomics = pd.read_csv(RADIOMICS_PATH)
    print(f"\nRadiomics: {len(radiomics)} patients, {len(radiomics.columns)-2} features")
    
    # Load clinical data
    clinical = pd.read_csv(CLINICAL_PATH)
    clinical = clinical.rename(columns={
        'Patient': 'Patient_ID',
        'Survival time (weeks)': 'Survival_Weeks',
        'Age at surgery (years)': 'Age',
        'MGMT qualitative': 'MGMT'
    })
    
    # Parse survival
    clinical['Survival_Weeks'] = pd.to_numeric(clinical['Survival_Weeks'], errors='coerce')
    
    # Filter patients with valid survival
    clinical = clinical[clinical['Survival_Weeks'].notna()].copy()
    print(f"Clinical (with survival): {len(clinical)} patients")
    
    # Create binary survival label (median split)
    median_survival = clinical['Survival_Weeks'].median()
    clinical['Survival_Binary'] = (clinical['Survival_Weeks'] > median_survival).astype(int)
    
    print(f"\nMedian survival: {median_survival:.1f} weeks")
    long_surv = (clinical['Survival_Binary'] == 1).sum()
    short_surv = (clinical['Survival_Binary'] == 0).sum()
    print(f"Long survival (>{median_survival:.0f}w): {long_surv}")
    print(f"Short survival (≤{median_survival:.0f}w): {short_surv}")
    print(f"Class ratio: {max(long_surv, short_surv)/min(long_surv, short_surv):.2f}:1")
    
    # Encode clinical features
    clinical['Sex_Binary'] = clinical['Sex'].map({'female': 0, 'male': 1})
    clinical['MGMT_Binary'] = clinical['MGMT'].map({'methylated': 1, 'not methylated': 0})
    
    # Merge
    merged = radiomics.merge(
        clinical[['Patient_ID', 'Age', 'Sex_Binary', 'MGMT_Binary', 'Survival_Binary', 'Survival_Weeks']],
        on='Patient_ID',
        how='inner'
    )
    
    print(f"\nMerged dataset: {len(merged)} patients")
    
    return merged, median_survival


def preprocess_features(df):
    """Preprocess features."""
    
    print("\n" + "="*60)
    print("FEATURE PREPROCESSING")
    print("="*60)
    
    # Identify feature columns
    exclude_cols = ['Patient_ID', 'Used_Label', 'Survival_Binary', 'Survival_Weeks']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"Total features: {len(feature_cols)}")
    
    X = df[feature_cols].copy()
    y = df['Survival_Binary'].values
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    
    # Remove low-variance features
    variances = X_imputed.var()
    low_var_cols = variances[variances < 0.01].index.tolist()
    X_filtered = X_imputed.drop(columns=low_var_cols)
    print(f"After low-variance removal: {len(X_filtered.columns)}")
    
    # Remove highly correlated features
    corr_matrix = X_filtered.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    X_filtered = X_filtered.drop(columns=to_drop)
    print(f"After correlation removal: {len(X_filtered.columns)}")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)
    
    feature_names = X_filtered.columns.tolist()
    
    return X_scaled, y, feature_names, scaler


def evaluate_models(X, y, feature_names):
    """Evaluate multiple ML models using cross-validation."""
    
    print("\n" + "="*60)
    print("MODEL EVALUATION (5-Fold Stratified CV)")
    print("="*60)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(
            C=0.1, penalty='l2', solver='saga', max_iter=5000, 
            random_state=SEED, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_split=5,
            random_state=SEED, class_weight='balanced', n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, use_label_encoder=False,
            eval_metric='logloss', verbosity=0
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=SEED
        ),
        'SVM (RBF)': SVC(
            C=1.0, kernel='rbf', probability=True,
            random_state=SEED, class_weight='balanced'
        )
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    results = {}
    all_predictions = {}
    
    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        
        # Store metrics per fold
        fold_metrics = {
            'roc_auc': [], 'accuracy': [], 'precision': [], 
            'recall': [], 'f1': []
        }
        
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Feature selection using L1 (inside fold)
            selector = SelectFromModel(
                LogisticRegression(C=0.1, penalty='l1', solver='saga', 
                                   max_iter=5000, random_state=SEED),
                threshold='median'
            )
            X_train_sel = selector.fit_transform(X_train, y_train)
            X_test_sel = selector.transform(X_test)
            
            # Train
            model.fit(X_train_sel, y_train)
            
            # Predict
            y_pred = model.predict(X_test_sel)
            y_proba = model.predict_proba(X_test_sel)[:, 1]
            
            # Metrics
            fold_metrics['roc_auc'].append(roc_auc_score(y_test, y_proba))
            fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            fold_metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            fold_metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            fold_metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_proba.extend(y_proba)
        
        # Summary
        mean_auc = np.mean(fold_metrics['roc_auc'])
        std_auc = np.std(fold_metrics['roc_auc'])
        mean_acc = np.mean(fold_metrics['accuracy'])
        mean_f1 = np.mean(fold_metrics['f1'])
        
        print(f"  ROC-AUC:  {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"  Accuracy: {mean_acc:.4f}")
        print(f"  F1-Score: {mean_f1:.4f}")
        
        results[model_name] = {
            'metrics': {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in fold_metrics.items()},
            'confusion_matrix': confusion_matrix(all_y_true, all_y_pred).tolist()
        }
        
        all_predictions[model_name] = {
            'y_true': all_y_true,
            'y_pred': all_y_pred,
            'y_proba': all_y_proba
        }
    
    return results, all_predictions


def get_feature_importance(X, y, feature_names):
    """Get feature importance from best models."""
    
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Train XGBoost on full data
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, use_label_encoder=False,
        eval_metric='logloss', verbosity=0
    )
    model.fit(X, y)
    
    # Get importance
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Important Features:")
    for i, row in importance_df.head(15).iterrows():
        print(f"  {row['feature'][:50]:50s}: {row['importance']:.4f}")
    
    return importance_df


def save_results(results, predictions, importance_df, median_survival):
    """Save all results."""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Results JSON
    output = {
        'task': 'Survival Prediction (Long vs Short)',
        'threshold_weeks': float(median_survival),
        'seed': SEED,
        'models': {
            name: {
                'roc_auc_mean': data['metrics']['roc_auc']['mean'],
                'roc_auc_std': data['metrics']['roc_auc']['std'],
                'accuracy_mean': data['metrics']['accuracy']['mean'],
                'f1_mean': data['metrics']['f1']['mean'],
                'confusion_matrix': data['confusion_matrix']
            }
            for name, data in results.items()
        }
    }
    
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR / 'results.json'}")
    
    # 2. Feature importance
    importance_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
    print(f"Saved: {OUTPUT_DIR / 'feature_importance.csv'}")
    
    # 3. ROC curves
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(predictions)))
    
    for (name, pred), color in zip(predictions.items(), colors):
        fpr, tpr, _ = roc_curve(pred['y_true'], pred['y_proba'])
        auc = roc_auc_score(pred['y_true'], pred['y_proba'])
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Survival Prediction\n(Long vs Short Survival)', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'roc_curves.png'}")
    
    # 4. Feature importance plot
    plt.figure(figsize=(10, 8))
    top_20 = importance_df.head(20).copy()
    top_20['feature_short'] = top_20['feature'].apply(lambda x: x[:35] + '...' if len(x) > 35 else x)
    
    plt.barh(range(len(top_20)), top_20['importance'].values, color='steelblue')
    plt.yticks(range(len(top_20)), top_20['feature_short'].values)
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')
    plt.title('Top 20 Features for Survival Prediction')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'feature_importance.png'}")
    
    # 5. Model comparison bar chart
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    aucs = [results[m]['metrics']['roc_auc']['mean'] for m in model_names]
    stds = [results[m]['metrics']['roc_auc']['std'] for m in model_names]
    
    bars = plt.barh(model_names, aucs, xerr=stds, color='steelblue', capsize=5)
    plt.axvline(x=0.5, color='red', linestyle='--', label='Random (0.5)')
    plt.xlabel('ROC-AUC')
    plt.title('Model Comparison - Survival Prediction')
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'model_comparison.png'}")


def main():
    print("="*60)
    print("SURVIVAL PREDICTION FROM PRE-TREATMENT RADIOMICS")
    print("Target: Long (>72w) vs Short (≤72w) Survival")
    print("="*60)
    
    # Load data
    df, median_survival = load_data()
    
    if len(df) < 20:
        print("\nERROR: Not enough data!")
        return
    
    # Preprocess
    X, y, feature_names, scaler = preprocess_features(df)
    
    # Evaluate models
    results, predictions = evaluate_models(X, y, feature_names)
    
    # Feature importance
    importance_df = get_feature_importance(X, y, feature_names)
    
    # Save results
    save_results(results, predictions, importance_df, median_survival)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    best_model = max(results.keys(), key=lambda m: results[m]['metrics']['roc_auc']['mean'])
    best_auc = results[best_model]['metrics']['roc_auc']['mean']
    best_std = results[best_model]['metrics']['roc_auc']['std']
    
    print(f"\nBest Model: {best_model}")
    print(f"ROC-AUC: {best_auc:.4f} ± {best_std:.4f}")
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
