"""
05_train_mgmt_model.py
Level 4: MGMT Prediction Model Training with Leave-One-Out Cross-Validation

This script trains an XGBoost classifier to predict MGMT methylation status
from radiomic features, using proper LOOCV with no data leakage.

Input:
    - Level4_Radiomic_Features.csv (Patient_ID, MGMT_Label, + 41 radiomic features)

Output:
    - Level4_ROC_Curve.png (ROC curve visualization)
    - Console output with AUC score and performance metrics

Key Design Principles:
    - All preprocessing (scaling, feature selection) happens INSIDE the LOOCV loop
    - Only training data is used to fit preprocessors
    - Prevents data leakage for unbiased performance estimation
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
except ImportError:
    print("ERROR: xgboost not installed!")
    print("Install with: pip install xgboost")
    sys.exit(1)


# Configuration
SCRIPT_DIR = Path(__file__).parent
INPUT_CSV = SCRIPT_DIR / "Level4_Radiomic_Features.csv"
OUTPUT_ROC = SCRIPT_DIR / "Level4_ROC_Curve.png"

# Model parameters
N_FEATURES = 5          # Number of top features to select
MAX_DEPTH = 2           # XGBoost max depth (shallow to prevent overfitting)
N_ESTIMATORS = 100      # Number of boosting rounds
RANDOM_STATE = 42       # For reproducibility


def load_and_prepare_data(csv_path: Path):
    """
    Load the radiomic features CSV and prepare X (features) and y (labels).
    
    Returns:
        X: DataFrame of numeric features
        y: numpy array of binary labels
        feature_names: list of feature column names
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"  Total samples: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
    
    # Separate target and features
    y = df['MGMT_Label'].values
    
    # Drop non-numeric and identifier columns
    drop_cols = ['Patient_ID', 'MGMT_Label']
    X = df.drop(columns=drop_cols)
    
    # Ensure all columns are numeric
    X = X.select_dtypes(include=[np.number])
    feature_names = X.columns.tolist()
    
    print(f"  Feature columns: {len(feature_names)}")
    print(f"  Class distribution: Methylated={sum(y)}, Unmethylated={len(y)-sum(y)}")
    
    return X, y, feature_names


def train_with_loocv(X: pd.DataFrame, y: np.ndarray, feature_names: list):
    """
    Train and evaluate using Leave-One-Out Cross-Validation.
    
    All preprocessing (scaling, feature selection) is done INSIDE the loop
    using only training data to prevent data leakage.
    
    Returns:
        y_true: list of true labels
        y_probs: list of predicted probabilities
        selected_features_count: dict counting how often each feature was selected
    """
    print("\n" + "=" * 60)
    print("Starting Leave-One-Out Cross-Validation")
    print("=" * 60)
    
    loo = LeaveOneOut()
    n_splits = loo.get_n_splits(X)
    print(f"Total LOOCV iterations: {n_splits}")
    
    # Storage for predictions
    y_true = []
    y_probs = []
    y_preds = []
    
    # Track which features are selected most often
    selected_features_count = {feat: 0 for feat in feature_names}
    
    # LOOCV Loop
    for fold_idx, (train_idx, test_idx) in enumerate(loo.split(X)):
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Progress indicator (every 10 folds)
        if (fold_idx + 1) % 10 == 0 or fold_idx == 0:
            print(f"  Processing fold {fold_idx + 1}/{n_splits}...")
        
        # ============================================================
        # STEP 1: Scaling (fit on training data ONLY)
        # ============================================================
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # ============================================================
        # STEP 2: Feature Selection (fit on training data ONLY)
        # ============================================================
        selector = SelectKBest(score_func=f_classif, k=N_FEATURES)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        # Track selected features
        selected_mask = selector.get_support()
        for i, is_selected in enumerate(selected_mask):
            if is_selected:
                selected_features_count[feature_names[i]] += 1
        
        # ============================================================
        # STEP 3: Train XGBoost Classifier
        # ============================================================
        model = XGBClassifier(
            max_depth=MAX_DEPTH,
            n_estimators=N_ESTIMATORS,
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            verbosity=0
        )
        model.fit(X_train_selected, y_train)
        
        # ============================================================
        # STEP 4: Predict probability for test sample
        # ============================================================
        prob = model.predict_proba(X_test_selected)[0, 1]  # Probability of class 1 (Methylated)
        pred = model.predict(X_test_selected)[0]
        
        # ============================================================
        # STEP 5: Store results
        # ============================================================
        y_true.append(y_test[0])
        y_probs.append(prob)
        y_preds.append(pred)
    
    print(f"  Completed all {n_splits} folds!")
    
    return y_true, y_probs, y_preds, selected_features_count


def evaluate_and_plot(y_true: list, y_probs: list, y_preds: list, 
                      selected_features_count: dict, output_path: Path):
    """
    Calculate metrics and plot ROC curve.
    """
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_preds = np.array(y_preds)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # ============================================================
    # Calculate ROC-AUC Score
    # ============================================================
    auc_score = roc_auc_score(y_true, y_probs)
    print(f"\n*** ROC-AUC Score: {auc_score:.4f} ***")
    
    # Interpret AUC
    if auc_score >= 0.9:
        interpretation = "Excellent discrimination"
    elif auc_score >= 0.8:
        interpretation = "Good discrimination"
    elif auc_score >= 0.7:
        interpretation = "Fair discrimination"
    elif auc_score >= 0.6:
        interpretation = "Poor discrimination"
    else:
        interpretation = "No discrimination (random)"
    print(f"    Interpretation: {interpretation}")
    
    # ============================================================
    # Additional Metrics
    # ============================================================
    accuracy = accuracy_score(y_true, y_preds)
    cm = confusion_matrix(y_true, y_preds)
    
    print(f"\nAccuracy: {accuracy:.4f} ({int(accuracy * len(y_true))}/{len(y_true)} correct)")
    
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Unmeth  Meth")
    print(f"  Actual Unmeth    {cm[0,0]:3d}   {cm[0,1]:3d}")
    print(f"  Actual Meth      {cm[1,0]:3d}   {cm[1,1]:3d}")
    
    # Sensitivity and Specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nSensitivity (Methylated recall): {sensitivity:.4f}")
    print(f"Specificity (Unmethylated recall): {specificity:.4f}")
    
    # ============================================================
    # Top Selected Features
    # ============================================================
    print("\nTop 10 Most Frequently Selected Features:")
    sorted_features = sorted(selected_features_count.items(), key=lambda x: x[1], reverse=True)
    for i, (feat, count) in enumerate(sorted_features[:10]):
        pct = count / len(y_true) * 100
        print(f"  {i+1}. {feat}: {count}/{len(y_true)} folds ({pct:.1f}%)")
    
    # ============================================================
    # Plot ROC Curve
    # ============================================================
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.3f})')
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier (AUC = 0.5)')
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Level 4: MGMT Methylation Prediction\nROC Curve (LOOCV)', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add AUC annotation
    plt.annotate(f'AUC = {auc_score:.3f}', 
                 xy=(0.6, 0.3), fontsize=16, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nROC curve saved to: {output_path}")
    
    return auc_score


def main():
    """Main pipeline for MGMT prediction model training."""
    print("=" * 60)
    print("Level 4: MGMT Methylation Prediction Model")
    print("XGBoost with Leave-One-Out Cross-Validation")
    print("=" * 60)
    
    # Step 1: Load data
    if not INPUT_CSV.exists():
        print(f"ERROR: Input file not found: {INPUT_CSV}")
        print("Run 04_extract_mgmt_features.py first.")
        sys.exit(1)
    
    X, y, feature_names = load_and_prepare_data(INPUT_CSV)
    
    # Step 2: Train with LOOCV
    y_true, y_probs, y_preds, selected_features = train_with_loocv(X, y, feature_names)
    
    # Step 3: Evaluate and plot
    auc_score = evaluate_and_plot(y_true, y_probs, y_preds, selected_features, OUTPUT_ROC)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Dataset: {len(y)} patients")
    print(f"Features: {len(feature_names)} radiomic features")
    print(f"Selected features per fold: {N_FEATURES}")
    print(f"Model: XGBoost (max_depth={MAX_DEPTH}, n_estimators={N_ESTIMATORS})")
    print(f"\n*** FINAL ROC-AUC: {auc_score:.4f} ***")
    
    return auc_score


if __name__ == "__main__":
    main()
