#!/usr/bin/env python3
"""
Level 1 Baseline Model: Progression Detection from Delta Volume

Binary XGBoost classifier to detect Progression vs Non-Progression
using only Delta_Volume as the input feature.

Features:
- Patient-level train/test split (no data leakage)
- Class imbalance handling via scale_pos_weight
- Evaluation: ROC-AUC, Precision, Recall, F1, Confusion Matrix
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier

# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_FILE = SCRIPT_DIR / "lumiere_phase0.csv"


def load_and_prepare_data(filepath: Path) -> pd.DataFrame:
    """
    Load CSV and create binary target column.
    
    Progression → 1
    Stable or Response → 0
    """
    df = pd.read_csv(filepath)
    
    # Create binary target
    df['Target'] = (df['Response_Label'] == 'Progression').astype(int)
    
    return df


def patient_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split data by Patient_ID to prevent data leakage.
    No patient appears in both train and test sets.
    """
    # Get unique patients
    patients = df['Patient_ID'].unique()
    
    # Use GroupShuffleSplit to split by patient
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    # Get train/test indices
    train_idx, test_idx = next(gss.split(df, groups=df['Patient_ID']))
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    return train_df, test_df


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """
    Train XGBoost binary classifier with class imbalance handling.
    """
    # Calculate scale_pos_weight for class imbalance
    n_negative = np.sum(y_train == 0)
    n_positive = np.sum(y_train == 1)
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
    
    # Initialize XGBoost with reasonable defaults
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Train
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model: XGBClassifier, X_test: np.ndarray, y_test: np.ndarray):
    """
    Evaluate model and print results.
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print("\n" + "=" * 60)
    print("LEVEL 1 BASELINE MODEL EVALUATION")
    print("=" * 60)
    
    print("\n📊 Dataset Summary:")
    print(f"   Test samples: {len(y_test)}")
    print(f"   Progression (1): {np.sum(y_test == 1)}")
    print(f"   Non-Progression (0): {np.sum(y_test == 0)}")
    
    print("\n📈 Performance Metrics:")
    print(f"   ROC-AUC Score: {roc_auc:.4f}")
    print(f"   Precision (Progression): {precision:.4f}")
    print(f"   Recall (Progression): {recall:.4f}")
    print(f"   F1-Score (Progression): {f1:.4f}")
    
    print("\n📋 Confusion Matrix:")
    print(f"   {'':>20} Predicted")
    print(f"   {'':>15} Non-Prog  Prog")
    print(f"   Actual Non-Prog  {cm[0, 0]:>5}  {cm[0, 1]:>5}")
    print(f"   Actual Prog      {cm[1, 0]:>5}  {cm[1, 1]:>5}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Progression', 'Progression']))
    
    return {
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def main():
    print("=" * 60)
    print("LEVEL 1: PROGRESSION DETECTION BASELINE")
    print("Feature: Delta_Volume only")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading data...")
    df = load_and_prepare_data(DATA_FILE)
    print(f"   Total samples: {len(df)}")
    print(f"   Unique patients: {df['Patient_ID'].nunique()}")
    print(f"   Label distribution:")
    print(f"      Progression: {(df['Target'] == 1).sum()} ({(df['Target'] == 1).mean()*100:.1f}%)")
    print(f"      Non-Progression: {(df['Target'] == 0).sum()} ({(df['Target'] == 0).mean()*100:.1f}%)")
    
    # Split by patient
    print("\n🔀 Splitting data by Patient_ID...")
    train_df, test_df = patient_train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"   Train: {len(train_df)} samples from {train_df['Patient_ID'].nunique()} patients")
    print(f"   Test: {len(test_df)} samples from {test_df['Patient_ID'].nunique()} patients")
    
    # Verify no patient overlap
    train_patients = set(train_df['Patient_ID'].unique())
    test_patients = set(test_df['Patient_ID'].unique())
    overlap = train_patients.intersection(test_patients)
    assert len(overlap) == 0, f"Patient overlap detected: {overlap}"
    print("   ✅ No patient overlap between train and test")
    
    # Prepare features (Delta_Volume only)
    X_train = train_df[['Delta_Volume']].values
    y_train = train_df['Target'].values
    X_test = test_df[['Delta_Volume']].values
    y_test = test_df['Target'].values
    
    # Train model
    print("\n🚀 Training XGBoost classifier...")
    model = train_model(X_train, y_train)
    print("   ✅ Model trained")
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    print("\n" + "=" * 60)
    print("✅ Level 1 Baseline Complete")
    print("=" * 60)
    
    return model, metrics


if __name__ == "__main__":
    main()
