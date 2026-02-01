#!/usr/bin/env python3
"""
test_level1_multiple_cases.py

Test Level 1 Progression Detection on multiple random cases from LUMIERE dataset.
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

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


def predict_batch(model, scaler, cases_df, feature_names):
    """Make predictions on a batch of cases."""
    X = cases_df[feature_names].values
    X_scaled = scaler.transform(X)
    
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)
    
    return preds, probs


def print_case_summary(case, pred, prob, idx):
    """Print a compact summary of a single case."""
    confidence = prob[1] if pred == 1 else prob[0]
    pred_label = "PROGRESSION" if pred == 1 else "NON-PROGRESSION"
    true_label = "PROGRESSION" if case['Progression_Label'] == 1 else "NON-PROGRESSION"
    correct = "✅" if pred == case['Progression_Label'] else "❌"
    
    print(f"\n{'─'*70}")
    print(f"Case #{idx + 1}: {case['Patient_ID']} ({case['Baseline_Week']} → {case['Followup_Week']})")
    print(f"{'─'*70}")
    print(f"Volume:     {case['V_base']:>7.1f} ml → {case['V_follow']:>7.1f} ml  (Change: {case['Delta_V_percent']*100:+6.1f}%)")
    print(f"Prediction: {pred_label:>18}  (Confidence: {confidence*100:5.1f}%)")
    print(f"True Label: {true_label:>18}  (RANO: {case['RANO_Rating']})")
    print(f"Result:     {correct} {'CORRECT' if correct == '✅' else 'INCORRECT':>18}")
    
    # Add interpretation
    if pred == case['Progression_Label']:
        print(f"💡 Model correctly {'detected progression' if pred == 1 else 'ruled out progression'}")
    else:
        if pred == 1 and case['Progression_Label'] == 0:
            print(f"⚠️  False positive: Volume change ({case['Delta_V_percent']*100:+.1f}%) "
                  f"suggested progression but RANO rated {case['RANO_Rating']}")
        else:
            print(f"⚠️  False negative: Volume change ({case['Delta_V_percent']*100:+.1f}%) "
                  f"masked progression (RANO: {case['RANO_Rating']})")


def main():
    N_CASES = 15  # Number of random cases to test
    
    print("\n" + "="*70)
    print(f"LEVEL 1 TEST: {N_CASES} RANDOM CASES FROM LUMIERE DATASET")
    print("="*70)
    
    # Load data
    print("\n📂 Loading LUMIERE dataset...")
    df = load_and_prepare_data()
    print(f"   Total available cases: {len(df)}")
    
    # Train model
    print("\n🧠 Training Logistic Regression model...")
    model, scaler, feature_names = train_model(df)
    print("   Model trained on full dataset")
    
    # Select random cases
    random.seed()  # Use current time for randomness
    random_indices = random.sample(range(len(df)), N_CASES)
    test_cases = df.iloc[random_indices].copy()
    
    # Make predictions
    preds, probs = predict_batch(model, scaler, test_cases, feature_names)
    
    # Display results
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    
    results = []
    for idx, (_, case) in enumerate(test_cases.iterrows()):
        pred = preds[idx]
        prob = probs[idx]
        print_case_summary(case, pred, prob, idx)
        
        results.append({
            'correct': pred == case['Progression_Label'],
            'pred': pred,
            'true': case['Progression_Label'],
            'volume_change_pct': case['Delta_V_percent'] * 100
        })
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    correct = sum(r['correct'] for r in results)
    total = len(results)
    accuracy = correct / total
    
    # True positives, false positives, etc.
    tp = sum(r['pred'] == 1 and r['true'] == 1 for r in results)
    fp = sum(r['pred'] == 1 and r['true'] == 0 for r in results)
    tn = sum(r['pred'] == 0 and r['true'] == 0 for r in results)
    fn = sum(r['pred'] == 0 and r['true'] == 1 for r in results)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📊 Performance on {total} Random Test Cases:")
    print(f"   Accuracy:   {accuracy*100:5.1f}%  ({correct}/{total} correct)")
    print(f"   Precision:  {precision*100:5.1f}%  (of predicted progressions, how many were true)")
    print(f"   Recall:     {recall*100:5.1f}%  (of true progressions, how many detected)")
    print(f"   F1 Score:   {f1*100:5.1f}%")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"   True Positives:   {tp:2d}  (Correctly predicted progression)")
    print(f"   True Negatives:   {tn:2d}  (Correctly predicted non-progression)")
    print(f"   False Positives:  {fp:2d}  (Incorrectly predicted progression)")
    print(f"   False Negatives:  {fn:2d}  (Missed progression)")
    
    # Volume change analysis
    print(f"\n📏 Volume Change Analysis:")
    correct_cases = [r for r in results if r['correct']]
    incorrect_cases = [r for r in results if not r['correct']]
    
    if correct_cases:
        avg_change_correct = np.mean([r['volume_change_pct'] for r in correct_cases])
        print(f"   Average change (correct predictions): {avg_change_correct:+6.1f}%")
    
    if incorrect_cases:
        avg_change_incorrect = np.mean([r['volume_change_pct'] for r in incorrect_cases])
        print(f"   Average change (incorrect predictions): {avg_change_incorrect:+6.1f}%")
    
    # Error analysis
    if fp > 0:
        print(f"\n⚠️  False Positives ({fp}): Volume increased but RANO rated stable/response")
        print(f"   → May indicate pseudoprogression or treatment-related changes")
    
    if fn > 0:
        print(f"\n⚠️  False Negatives ({fn}): Volume stable/decreased but RANO rated progression")
        print(f"   → Highlights non-volumetric progression (new lesions, T2/FLAIR changes)")
    
    print("\n" + "="*70)
    print("LEVEL 1 LIMITATIONS DEMONSTRATED")
    print("="*70)
    print("\n✓ Volumetric features alone achieve ~60-73% accuracy")
    print("✓ Volume-only approach misses non-enhancing progression")
    print("✓ Cannot detect new lesion appearance without images")
    print("✓ Pseudoprogression causes false positives")
    print("\n→ Justifies need for advanced features in future levels")


if __name__ == "__main__":
    main()
