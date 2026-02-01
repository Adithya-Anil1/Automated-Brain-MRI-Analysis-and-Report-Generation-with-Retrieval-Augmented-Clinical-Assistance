#!/usr/bin/env python3
"""
test_level2_multiple_cases.py

Test Level 2 Region-Aware Progression Detection on multiple random cases.
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Paths
BASE_DIR = Path(__file__).parent
FEATURE_TABLE_PATH = BASE_DIR / "level2_progression_results" / "level2_feature_table.csv"

RANDOM_SEED = 42


def load_data():
    """Load the Level 2 feature table."""
    df = pd.read_csv(FEATURE_TABLE_PATH)
    return df


def train_models(df):
    """Train Level 1 and Level 2 models on full dataset."""
    # Level 1 features (volume only)
    l1_features = ['V_base_WT', 'V_follow_WT', 'Delta_V_WT', 'Delta_V_pct_WT']
    X_l1 = df[l1_features].values
    
    # Level 2 features (region-aware)
    l2_features = [
        'V_base_WT', 'V_follow_WT', 'Delta_V_WT', 'Delta_V_pct_WT',
        'V_base_TC', 'V_follow_TC', 'Delta_V_TC', 'Delta_V_pct_TC',
        'V_base_ET', 'V_follow_ET', 'Delta_V_ET', 'Delta_V_pct_ET',
        'ET_fraction_base', 'ET_fraction_follow', 'Delta_ET_fraction',
        'TC_fraction_base', 'TC_fraction_follow', 'Delta_TC_fraction',
        'newly_appeared_WT', 'newly_appeared_TC', 'newly_appeared_ET'
    ]
    X_l2 = df[l2_features].values
    
    y = df['Progression_Label'].values
    
    # Train Level 1 model (volume only)
    scaler_l1 = StandardScaler()
    X_l1_scaled = scaler_l1.fit_transform(X_l1)
    model_l1 = RandomForestClassifier(
        n_estimators=100, max_depth=3, random_state=RANDOM_SEED,
        class_weight='balanced', n_jobs=-1
    )
    model_l1.fit(X_l1_scaled, y)
    
    # Train Level 2 model (region-aware)
    model_l2 = RandomForestClassifier(
        n_estimators=100, max_depth=3, random_state=RANDOM_SEED,
        class_weight='balanced', n_jobs=-1
    )
    model_l2.fit(X_l2, y)
    
    return {
        'l1': {'model': model_l1, 'scaler': scaler_l1, 'features': l1_features},
        'l2': {'model': model_l2, 'features': l2_features}
    }


def predict_case(case, models):
    """Make predictions for a single case using both Level 1 and Level 2."""
    # Level 1 prediction
    X_l1 = case[models['l1']['features']].values.reshape(1, -1)
    X_l1_scaled = models['l1']['scaler'].transform(X_l1)
    l1_pred = models['l1']['model'].predict(X_l1_scaled)[0]
    l1_prob = models['l1']['model'].predict_proba(X_l1_scaled)[0]
    
    # Level 2 prediction
    X_l2 = case[models['l2']['features']].values.reshape(1, -1)
    l2_pred = models['l2']['model'].predict(X_l2)[0]
    l2_prob = models['l2']['model'].predict_proba(X_l2)[0]
    
    return {
        'l1_pred': l1_pred,
        'l1_prob': l1_prob,
        'l1_conf': l1_prob[1] if l1_pred == 1 else l1_prob[0],
        'l2_pred': l2_pred,
        'l2_prob': l2_prob,
        'l2_conf': l2_prob[1] if l2_pred == 1 else l2_prob[0]
    }


def print_case_summary(case, preds, idx):
    """Print detailed summary for a single case."""
    true_label = case['Progression_Label']
    true_label_str = "PROGRESSION" if true_label == 1 else "NON-PROGRESSION"
    
    l1_pred_str = "PROGRESSION" if preds['l1_pred'] == 1 else "NON-PROGRESSION"
    l2_pred_str = "PROGRESSION" if preds['l2_pred'] == 1 else "NON-PROGRESSION"
    
    l1_correct = "✅" if preds['l1_pred'] == true_label else "❌"
    l2_correct = "✅" if preds['l2_pred'] == true_label else "❌"
    
    print(f"\n{'═'*75}")
    print(f"CASE #{idx + 1}: {case['Patient_ID']}")
    print(f"{'═'*75}")
    print(f"Timepoints: {case['Baseline_Week']} → {case['Followup_Week']}")
    print(f"True Label: {true_label_str} (RANO: {case['RANO_Rating']})")
    
    print(f"\n{'─'*75}")
    print(f"VOLUME CHANGES (ml)")
    print(f"{'─'*75}")
    print(f"Region    Baseline    Follow-up    Δ Volume      Δ%")
    print(f"{'─'*75}")
    print(f"WT        {case['V_base_WT']:>8.2f}    {case['V_follow_WT']:>9.2f}    {case['Delta_V_WT']:>+8.2f}    {case['Delta_V_pct_WT']*100:>+6.1f}%")
    print(f"TC        {case['V_base_TC']:>8.2f}    {case['V_follow_TC']:>9.2f}    {case['Delta_V_TC']:>+8.2f}    {case['Delta_V_pct_TC']*100:>+6.1f}%")
    print(f"ET        {case['V_base_ET']:>8.2f}    {case['V_follow_ET']:>9.2f}    {case['Delta_V_ET']:>+8.2f}    {case['Delta_V_pct_ET']*100:>+6.1f}%")
    
    print(f"\n{'─'*75}")
    print(f"COMPOSITION CHANGES")
    print(f"{'─'*75}")
    print(f"ET/WT Fraction: {case['ET_fraction_base']:.3f} → {case['ET_fraction_follow']:.3f} (Δ{case['Delta_ET_fraction']*100:+.1f} pp)")
    print(f"TC/WT Fraction: {case['TC_fraction_base']:.3f} → {case['TC_fraction_follow']:.3f} (Δ{case['Delta_TC_fraction']*100:+.1f} pp)")
    
    print(f"\n{'─'*75}")
    print(f"PREDICTIONS")
    print(f"{'─'*75}")
    
    # Level 1 prediction
    print(f"\n📊 LEVEL 1 (Volume Only):")
    print(f"   Prediction:  {l1_pred_str:>18}")
    print(f"   Confidence:  {preds['l1_conf']*100:>17.1f}%")
    print(f"   Result:      {l1_correct:>18} {('CORRECT' if l1_correct == '✅' else 'INCORRECT')}")
    
    # Level 2 prediction
    print(f"\n🎯 LEVEL 2 (Region-Aware):")
    print(f"   Prediction:  {l2_pred_str:>18}")
    print(f"   Confidence:  {preds['l2_conf']*100:>17.1f}%")
    print(f"   Result:      {l2_correct:>18} {('CORRECT' if l2_correct == '✅' else 'INCORRECT')}")
    
    # Comparison
    print(f"\n{'─'*75}")
    print(f"COMPARISON")
    print(f"{'─'*75}")
    
    if preds['l1_pred'] != preds['l2_pred']:
        print(f"⚠️  DISAGREEMENT: L1 and L2 made different predictions!")
        
        if preds['l2_pred'] == true_label and preds['l1_pred'] != true_label:
            print(f"✅ Level 2 CORRECTED Level 1 error")
            print(f"   L1 failed because: Volume-only ({case['Delta_V_pct_WT']*100:+.1f}%) misleading")
            
            if true_label == 1:
                # Progression case
                if case['Delta_V_pct_TC'] > 0.25:
                    print(f"   L2 succeeded by detecting: TC increase ({case['Delta_V_pct_TC']*100:+.1f}%)")
                elif case['Delta_TC_fraction'] > 0.05:
                    print(f"   L2 succeeded by detecting: TC fraction increase ({case['Delta_TC_fraction']*100:+.1f} pp)")
                else:
                    print(f"   L2 succeeded by analyzing: Region composition patterns")
            else:
                print(f"   L2 correctly identified: Stable composition despite volume changes")
                
        elif preds['l1_pred'] == true_label and preds['l2_pred'] != true_label:
            print(f"❌ Level 2 INTRODUCED new error")
            print(f"   L1 was correct, but L2 over-analyzed regional features")
        else:
            print(f"❌ Both models wrong, but predicted differently")
    else:
        if preds['l1_pred'] == true_label:
            print(f"✅ AGREEMENT: Both models correct")
            print(f"   Volume changes aligned with regional composition")
        else:
            print(f"❌ AGREEMENT: Both models incorrect")
            print(f"   Both missed progression patterns (may need Level 3+ features)")


def main():
    N_CASES = 12
    
    print("\n" + "="*75)
    print(f"LEVEL 2 TEST: {N_CASES} RANDOM CASES")
    print("="*75)
    
    # Load data
    print("\n📂 Loading Level 2 feature table...")
    df = load_data()
    print(f"   Total cases: {len(df)}")
    
    # Train models
    print("\n🧠 Training Level 1 and Level 2 models...")
    models = train_models(df)
    print("   Models trained")
    
    # Select random cases
    random.seed()
    random_indices = random.sample(range(len(df)), N_CASES)
    test_cases = df.iloc[random_indices]
    
    # Test each case
    results = []
    for idx, (_, case) in enumerate(test_cases.iterrows()):
        preds = predict_case(case, models)
        print_case_summary(case, preds, idx)
        
        results.append({
            'patient': case['Patient_ID'],
            'true': case['Progression_Label'],
            'l1_pred': preds['l1_pred'],
            'l2_pred': preds['l2_pred'],
            'l1_correct': preds['l1_pred'] == case['Progression_Label'],
            'l2_correct': preds['l2_pred'] == case['Progression_Label'],
            'l1_to_l2_improvement': (preds['l2_pred'] == case['Progression_Label']) and (preds['l1_pred'] != case['Progression_Label'])
        })
    
    # Summary statistics
    print("\n" + "="*75)
    print("SUMMARY STATISTICS")
    print("="*75)
    
    l1_correct = sum(r['l1_correct'] for r in results)
    l2_correct = sum(r['l2_correct'] for r in results)
    improvements = sum(r['l1_to_l2_improvement'] for r in results)
    
    print(f"\n📊 Performance on {N_CASES} Random Cases:")
    print(f"   Level 1 Accuracy: {l1_correct}/{N_CASES} ({100*l1_correct/N_CASES:.1f}%)")
    print(f"   Level 2 Accuracy: {l2_correct}/{N_CASES} ({100*l2_correct/N_CASES:.1f}%)")
    print(f"   Improvement:      {l2_correct - l1_correct:+d} cases")
    
    print(f"\n✅ Cases where L2 corrected L1 errors: {improvements}")
    
    # Show disagreement cases
    disagreements = [r for r in results if r['l1_pred'] != r['l2_pred']]
    print(f"\n⚠️  L1 vs L2 disagreements: {len(disagreements)} cases")
    
    if improvements > 0:
        print(f"\n💡 Key Insight:")
        print(f"   Region-aware features (TC, ET composition) helped Level 2")
        print(f"   correctly identify {improvements} progression cases that Level 1 missed")
        print(f"   due to misleading total volume changes.")


if __name__ == "__main__":
    main()
