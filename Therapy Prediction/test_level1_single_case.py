#!/usr/bin/env python3
"""
test_level1_single_case.py

Test Level 1 Progression Detection on a random case from the LUMIERE dataset.
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Paths
BASE_DIR = Path(__file__).parent
LUMIERE_DATA_PATH = BASE_DIR / "lumiere_phase0.csv"
RANO_EXPERT_PATH = BASE_DIR / "dataset" / "LUMIERE-ExpertRating-v202211.csv"

RANDOM_SEED = 42
VOLUME_CHANGE_THRESHOLD = 0.25


def load_and_prepare_data():
    """Load and prepare the full dataset."""
    # Load RANO labels
    rano_df = pd.read_csv(RANO_EXPERT_PATH)
    rano_df.columns = rano_df.columns.str.strip()
    rating_col = [c for c in rano_df.columns if 'Rating' in c and 'rationale' not in c.lower()][0]
    rano_df = rano_df.rename(columns={rating_col: 'RANO_Rating'})
    rano_df = rano_df[~rano_df['RANO_Rating'].isin(['Pre-Op', 'Post-Op'])].copy()
    rano_df['Progression_Label'] = (rano_df['RANO_Rating'] == 'PD').astype(int)
    rano_df['Patient'] = rano_df['Patient'].str.strip()
    rano_df['Date_Clean'] = rano_df['Date'].str.strip()
    
    # Load volume data
    volume_df = pd.read_csv(LUMIERE_DATA_PATH)
    volume_df['Patient_ID'] = volume_df['Patient_ID'].str.strip()
    volume_df['Followup_Week_Clean'] = volume_df['Followup_Week'].str.strip()
    
    # Merge
    merged = volume_df.merge(
        rano_df[['Patient', 'Date_Clean', 'RANO_Rating', 'Progression_Label']],
        left_on=['Patient_ID', 'Followup_Week_Clean'],
        right_on=['Patient', 'Date_Clean'],
        how='inner'
    )
    
    # Compute features
    merged['V_base'] = merged['Baseline_Volume_ml']
    merged['V_follow'] = merged['Followup_Volume_ml']
    merged['Delta_V'] = merged['V_follow'] - merged['V_base']
    merged = merged[merged['V_base'] > 0].copy()
    merged['Delta_V_percent'] = (merged['V_follow'] - merged['V_base']) / merged['V_base']
    
    return merged


def train_model(df):
    """Train the logistic regression model on the full dataset."""
    feature_names = ['V_base', 'V_follow', 'Delta_V', 'Delta_V_percent']
    X = df[feature_names].values
    y = df['Progression_Label'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, class_weight='balanced')
    model.fit(X_scaled, y)
    
    return model, scaler, feature_names


def predict_single_case(model, scaler, case_data, feature_names):
    """Make prediction on a single case."""
    X = case_data[feature_names].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]
    
    return pred, prob


def generate_explanation(pred, delta_v_pct, v_base, v_follow, confidence):
    """Generate rule-based explanation."""
    pct_change = delta_v_pct * 100
    threshold_pct = VOLUME_CHANGE_THRESHOLD * 100
    
    if pred == 1:
        if delta_v_pct > VOLUME_CHANGE_THRESHOLD:
            return (f"Predicted PROGRESSION because tumor volume increased by "
                   f"{pct_change:.1f}% between baseline ({v_base:.1f} ml) and "
                   f"follow-up ({v_follow:.1f} ml), exceeding the {threshold_pct:.0f}% "
                   f"volume-change heuristic inspired by RANO progression criteria.")
        elif delta_v_pct > 0:
            return (f"Predicted PROGRESSION with tumor volume increase of {pct_change:.1f}% "
                   f"(baseline: {v_base:.1f} ml → follow-up: {v_follow:.1f} ml). "
                   f"Model identified progression patterns despite being below threshold.")
        else:
            return (f"Predicted PROGRESSION despite volume decrease of {pct_change:.1f}% "
                   f"(baseline: {v_base:.1f} ml → follow-up: {v_follow:.1f} ml). "
                   f"May indicate non-enhancing progression. Clinical review recommended.")
    else:
        if delta_v_pct < -0.3:
            return (f"Predicted NON-PROGRESSION (likely response) with significant tumor "
                   f"volume decrease of {pct_change:.1f}% (baseline: {v_base:.1f} ml → "
                   f"follow-up: {v_follow:.1f} ml). Suggests treatment response.")
        elif delta_v_pct < 0:
            return (f"Predicted NON-PROGRESSION with tumor volume decrease of {pct_change:.1f}% "
                   f"(baseline: {v_base:.1f} ml → follow-up: {v_follow:.1f} ml). "
                   f"Volume remains stable or reduced.")
        elif delta_v_pct <= VOLUME_CHANGE_THRESHOLD:
            return (f"Predicted NON-PROGRESSION (stable disease) with volume change of "
                   f"{pct_change:.1f}% (baseline: {v_base:.1f} ml → follow-up: {v_follow:.1f} ml). "
                   f"Within {threshold_pct:.0f}% stability threshold.")
        else:
            return (f"Predicted NON-PROGRESSION despite volume increase of {pct_change:.1f}% "
                   f"(baseline: {v_base:.1f} ml → follow-up: {v_follow:.1f} ml). "
                   f"Close monitoring advised.")


def main():
    print("\n" + "="*70)
    print("LEVEL 1 TEST: RANDOM CASE FROM LUMIERE DATASET")
    print("="*70)
    
    # Load data
    print("\n📂 Loading LUMIERE dataset...")
    df = load_and_prepare_data()
    print(f"   Loaded {len(df)} patient-timepoint pairs")
    
    # Train model
    print("\n🧠 Training Logistic Regression model...")
    model, scaler, feature_names = train_model(df)
    print("   Model trained successfully")
    
    # Select random case
    random.seed()  # Use current time for true randomness
    random_idx = random.randint(0, len(df) - 1)
    case = df.iloc[random_idx]
    
    print("\n" + "="*70)
    print("🎲 RANDOMLY SELECTED TEST CASE")
    print("="*70)
    
    # Make prediction
    pred, prob = predict_single_case(model, scaler, case, feature_names)
    confidence = prob[1] if pred == 1 else prob[0]
    
    # Display case details
    print(f"\n┌{'─'*68}┐")
    print(f"│ {'PATIENT INFORMATION':^66} │")
    print(f"├{'─'*68}┤")
    print(f"│  Patient ID:      {case['Patient_ID']:<48} │")
    print(f"│  Baseline Week:   {case['Baseline_Week']:<48} │")
    print(f"│  Follow-up Week:  {case['Followup_Week']:<48} │")
    print(f"├{'─'*68}┤")
    print(f"│ {'VOLUMETRIC DATA':^66} │")
    print(f"├{'─'*68}┤")
    print(f"│  Baseline Volume:    {case['V_base']:>10.2f} ml{' '*41} │")
    print(f"│  Follow-up Volume:   {case['V_follow']:>10.2f} ml{' '*41} │")
    print(f"│  Absolute Change:    {case['Delta_V']:>+10.2f} ml{' '*41} │")
    print(f"│  Relative Change:    {case['Delta_V_percent']*100:>+10.1f} %{' '*42} │")
    print(f"├{'─'*68}┤")
    print(f"│ {'PREDICTION RESULTS':^66} │")
    print(f"├{'─'*68}┤")
    
    pred_label = "PROGRESSION" if pred == 1 else "NON-PROGRESSION"
    true_label = "PROGRESSION" if case['Progression_Label'] == 1 else "NON-PROGRESSION"
    correct = "✅ CORRECT" if pred == case['Progression_Label'] else "❌ INCORRECT"
    
    print(f"│  🔮 Model Prediction: {pred_label:<44} │")
    print(f"│  📋 Ground Truth:     {true_label:<44} │")
    print(f"│  📊 RANO Rating:      {case['RANO_Rating']:<44} │")
    print(f"│  🎯 Confidence:       {confidence*100:.1f}%{' '*50} │")
    print(f"│  ✓  Result:          {correct:<44} │")
    print(f"├{'─'*68}┤")
    print(f"│ {'CLINICAL EXPLANATION':^66} │")
    print(f"├{'─'*68}┤")
    
    explanation = generate_explanation(
        pred, case['Delta_V_percent'], case['V_base'], case['V_follow'], confidence
    )
    
    # Word wrap explanation
    words = explanation.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= 64:
            current_line += (" " if current_line else "") + word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    
    for line in lines:
        print(f"│  {line:<64} │")
    
    print(f"└{'─'*68}┘")
    
    # Summary
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print(f"\n{'Result:':<20} {correct}")
    print(f"{'Prediction:':<20} {pred_label}")
    print(f"{'True Label:':<20} {true_label} (RANO: {case['RANO_Rating']})")
    print(f"{'Confidence:':<20} {confidence*100:.1f}%")
    print(f"{'Volume Change:':<20} {case['Delta_V_percent']*100:+.1f}%")


if __name__ == "__main__":
    main()
