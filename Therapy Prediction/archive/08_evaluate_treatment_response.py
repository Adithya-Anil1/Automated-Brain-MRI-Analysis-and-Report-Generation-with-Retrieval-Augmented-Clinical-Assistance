#!/usr/bin/env python3
"""
08_evaluate_treatment_response.py

Evaluates whether pre-treatment radiomics + age can predict clinical treatment response (RANO-based).

Data sources:
- Radiomics features: Level4_Radiomic_Features_Enhancing.csv
- Clinical data (labels + age): LUMIERE-Demographics_Pathology.csv
- RANO ratings: dataset/LUMIERE-ExpertRating-v202211.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """Load radiomics, clinical, and RANO rating data."""
    
    # Paths
    radiomics_path = "previous_mgmt_attempts/Level4_Radiomic_Features_Enhancing.csv"
    clinical_path = "LUMIERE-Demographics_Pathology.csv"
    rano_path = "dataset/LUMIERE-ExpertRating-v202211.csv"
    
    # Load radiomics features
    print("Loading radiomics features...")
    radiomics_df = pd.read_csv(radiomics_path)
    print(f"  Radiomics: {len(radiomics_df)} patients, {len(radiomics_df.columns)-2} features")
    
    # Load clinical data
    print("Loading clinical data...")
    clinical_df = pd.read_csv(clinical_path)
    clinical_df = clinical_df.rename(columns={'Patient': 'Patient_ID'})
    print(f"  Clinical: {len(clinical_df)} patients")
    
    # Load RANO ratings
    print("Loading RANO ratings...")
    rano_df = pd.read_csv(rano_path)
    rano_df = rano_df.rename(columns={'Patient': 'Patient_ID'})
    
    # Find the RANO rating column
    response_columns = ["RANO", "Response", "TreatmentResponse"]
    rano_col = None
    
    for col in rano_df.columns:
        # Check if any of the expected names are in the column name
        for expected in response_columns:
            if expected.lower() in col.lower():
                rano_col = col
                break
        # Also check for "Rating" which is in the actual column name
        if "rating" in col.lower() and "rationale" not in col.lower():
            rano_col = col
            break
        if rano_col:
            break
    
    if rano_col is None:
        raise ValueError(f"No response column found! Searched for: {response_columns}")
    
    print(f"  Using RANO column: '{rano_col}'")
    
    return radiomics_df, clinical_df, rano_df, rano_col


def extract_best_response_per_patient(rano_df, rano_col):
    """
    Extract one RANO response per patient.
    We use the BEST post-treatment response (excluding Pre-Op and Post-Op).
    Priority: CR > PR > SD > PD
    """
    
    # Filter out Pre-Op and Post-Op entries
    treatment_responses = rano_df[~rano_df[rano_col].isin(['Pre-Op', 'Post-Op'])].copy()
    
    # Create a priority mapping (lower = better response)
    priority_map = {'CR': 1, 'PR': 2, 'SD': 3, 'PD': 4}
    treatment_responses['priority'] = treatment_responses[rano_col].map(priority_map)
    
    # Drop rows without valid RANO ratings
    treatment_responses = treatment_responses.dropna(subset=['priority'])
    
    # Get the best response for each patient
    best_response = treatment_responses.loc[
        treatment_responses.groupby('Patient_ID')['priority'].idxmin()
    ][['Patient_ID', rano_col]].copy()
    
    best_response = best_response.rename(columns={rano_col: 'RANO_Response'})
    
    print(f"  Extracted best response for {len(best_response)} patients")
    
    return best_response


def map_labels(df):
    """
    Map RANO responses to binary labels.
    CR/PR -> 1 (Responder)
    SD/PD -> 0 (Non-Responder)
    """
    
    label_mapping = {
        'CR': 1,  # Complete Response -> Responder
        'PR': 1,  # Partial Response -> Responder
        'SD': 0,  # Stable Disease -> Non-Responder
        'PD': 0   # Progressive Disease -> Non-Responder
    }
    
    # Map labels
    df['Response_Label'] = df['RANO_Response'].map(label_mapping)
    
    # Count before dropping
    print("\nLabel mapping summary:")
    for response, label in label_mapping.items():
        count = (df['RANO_Response'] == response).sum()
        label_str = "Responder (1)" if label == 1 else "Non-Responder (0)"
        print(f"  {response} -> {label_str}: {count}")
    
    # Count invalid/missing
    invalid_count = df['Response_Label'].isna().sum()
    if invalid_count > 0:
        print(f"  Dropped (invalid values): {invalid_count}")
    
    # Drop rows with invalid labels
    df = df.dropna(subset=['Response_Label'])
    df['Response_Label'] = df['Response_Label'].astype(int)
    
    print(f"\nFinal class distribution:")
    print(f"  Responders (CR/PR): {(df['Response_Label'] == 1).sum()}")
    print(f"  Non-Responders (SD/PD): {(df['Response_Label'] == 0).sum()}")
    
    return df


def prepare_features_and_labels(radiomics_df, clinical_df, response_df):
    """Merge all data and prepare features and labels."""
    
    # Merge radiomics with response data
    merged = radiomics_df.merge(response_df, on='Patient_ID', how='inner')
    print(f"\nAfter merging radiomics with response: {len(merged)} patients")
    
    # Merge with clinical data to get Age
    merged = merged.merge(
        clinical_df[['Patient_ID', 'Age at surgery (years)']],
        on='Patient_ID',
        how='inner'
    )
    merged = merged.rename(columns={'Age at surgery (years)': 'Age'})
    print(f"After merging with clinical (Age): {len(merged)} patients")
    
    # Map labels
    merged = map_labels(merged)
    
    # Identify feature columns (exclude Patient_ID, Used_Label, RANO_Response, Response_Label, Age)
    exclude_cols = ['Patient_ID', 'Used_Label', 'RANO_Response', 'Response_Label', 'Age']
    radiomics_feature_cols = [c for c in merged.columns if c not in exclude_cols]
    
    print(f"\nFeature summary:")
    print(f"  Radiomic features: {len(radiomics_feature_cols)}")
    print(f"  Age feature: 1 (always retained)")
    
    return merged, radiomics_feature_cols


def evaluate_with_cross_validation(data_df, radiomics_feature_cols):
    """
    Perform 5-fold stratified cross-validation.
    Feature selection and preprocessing happen inside each fold.
    """
    
    X_radiomics = data_df[radiomics_feature_cols].values
    X_age = data_df['Age'].values.reshape(-1, 1)
    y = data_df['Response_Label'].values
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION EVALUATION")
    print("="*60)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_radiomics, y), 1):
        # Split data
        X_train_rad, X_val_rad = X_radiomics[train_idx], X_radiomics[val_idx]
        X_train_age, X_val_age = X_age[train_idx], X_age[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Step 1: Standardize radiomics features (fit on train only)
        scaler_rad = StandardScaler()
        X_train_rad_scaled = scaler_rad.fit_transform(X_train_rad)
        X_val_rad_scaled = scaler_rad.transform(X_val_rad)
        
        # Step 2: L1 feature selection on radiomics (Lasso)
        lasso = LogisticRegression(
            penalty='l1',
            solver='saga',
            C=0.1,
            max_iter=5000,
            random_state=42,
            class_weight='balanced'
        )
        lasso.fit(X_train_rad_scaled, y_train)
        
        # Get selected feature mask (non-zero coefficients)
        selected_mask = np.abs(lasso.coef_[0]) > 1e-6
        n_selected = selected_mask.sum()
        
        if n_selected == 0:
            # If no features selected, use top 10 by coefficient magnitude
            top_k = min(10, len(radiomics_feature_cols))
            top_indices = np.argsort(np.abs(lasso.coef_[0]))[-top_k:]
            selected_mask = np.zeros(len(radiomics_feature_cols), dtype=bool)
            selected_mask[top_indices] = True
            n_selected = top_k
        
        # Apply feature selection
        X_train_rad_selected = X_train_rad_scaled[:, selected_mask]
        X_val_rad_selected = X_val_rad_scaled[:, selected_mask]
        
        # Step 3: Standardize age (fit on train only)
        scaler_age = StandardScaler()
        X_train_age_scaled = scaler_age.fit_transform(X_train_age)
        X_val_age_scaled = scaler_age.transform(X_val_age)
        
        # Step 4: Concatenate selected radiomics with age
        X_train_final = np.hstack([X_train_rad_selected, X_train_age_scaled])
        X_val_final = np.hstack([X_val_rad_selected, X_val_age_scaled])
        
        # Step 5: Train Random Forest with class balancing
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train_final, y_train)
        
        # Step 6: Predict and evaluate
        y_pred_proba = rf.predict_proba(X_val_final)[:, 1]
        
        try:
            auc = roc_auc_score(y_val, y_pred_proba)
        except ValueError:
            # If only one class in validation set
            auc = 0.5
        
        fold_aucs.append(auc)
        
        print(f"\nFold {fold}:")
        print(f"  Train size: {len(y_train)} | Val size: {len(y_val)}")
        print(f"  Val class distribution: {(y_val == 0).sum()} Non-Resp, {(y_val == 1).sum()} Resp")
        print(f"  Selected radiomics features: {n_selected}")
        print(f"  Total features (radiomics + age): {X_train_final.shape[1]}")
        print(f"  ROC-AUC: {auc:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Per-fold ROC-AUC: {[f'{auc:.4f}' for auc in fold_aucs]}")
    print(f"Mean ROC-AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print("="*60)
    
    return fold_aucs


def main():
    print("="*60)
    print("TREATMENT RESPONSE PREDICTION (RANO-based)")
    print("Radiomics + Age -> Response (CR/PR vs SD/PD)")
    print("="*60)
    
    # Load data
    radiomics_df, clinical_df, rano_df, rano_col = load_and_prepare_data()
    
    # Extract best response per patient
    response_df = extract_best_response_per_patient(rano_df, rano_col)
    
    # Prepare features and labels
    data_df, radiomics_feature_cols = prepare_features_and_labels(
        radiomics_df, clinical_df, response_df
    )
    
    # Check if we have enough data
    if len(data_df) < 10:
        print(f"\nERROR: Only {len(data_df)} samples available. Need at least 10 for cross-validation.")
        return
    
    # Check class balance
    class_counts = data_df['Response_Label'].value_counts()
    if len(class_counts) < 2:
        print(f"\nERROR: Only one class present. Cannot perform classification.")
        return
    
    min_class_count = class_counts.min()
    if min_class_count < 5:
        print(f"\nWARNING: Minority class has only {min_class_count} samples. Results may be unreliable.")
    
    # Run cross-validation evaluation
    fold_aucs = evaluate_with_cross_validation(data_df, radiomics_feature_cols)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
