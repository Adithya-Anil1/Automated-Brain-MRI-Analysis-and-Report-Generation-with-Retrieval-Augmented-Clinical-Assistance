#!/usr/bin/env python3
"""
test_integrated_random_patient.py

Test the integrated decision flow on a random patient,
showing the complete decision trace through L1 → L2 → L3.
"""

import random
import warnings
from pathlib import Path
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent

# Load results
INTEGRATED_RESULTS = BASE_DIR / "integrated_results" / "integrated_decisions_v2.csv"
INTEGRATED_TRACES = BASE_DIR / "integrated_results" / "decision_traces_v2.csv"

# Load L3 features for more detail
L3_FEATURES = BASE_DIR / "level3_progression_results" / "level3_temporal_features.csv"


def main():
    print("\n" + "="*70)
    print("TESTING INTEGRATED DECISION FLOW ON RANDOM PATIENT")
    print("="*70)
    
    # Load data
    results_df = pd.read_csv(INTEGRATED_RESULTS)
    traces_df = pd.read_csv(INTEGRATED_TRACES)
    
    # Filter to patients with complete data (not uncertain or insufficient)
    valid_patients = results_df[
        (results_df['Final_Label'] != 'INSUFFICIENT DATA') & 
        (results_df['Final_Label'] != 'UNCERTAIN')
    ]
    
    print(f"\nTotal patients: {len(results_df)}")
    print(f"Valid for testing: {len(valid_patients)}")
    
    # Pick random patient
    random_idx = random.randint(0, len(valid_patients) - 1)
    patient_row = valid_patients.iloc[random_idx]
    patient_id = patient_row['Patient_ID']
    
    # Get trace
    trace_row = traces_df[traces_df['Patient_ID'] == patient_id].iloc[0]
    
    print(f"\n🎲 Randomly selected: {patient_id}")
    
    # Display detailed results
    print("\n" + "━"*70)
    print(f"PATIENT: {patient_id}")
    print("━"*70)
    
    # True label
    true_label = "PROGRESSION" if patient_row['True_Label'] == 1 else "NON-PROGRESSION"
    print(f"\n📋 GROUND TRUTH (from RANO): {true_label}")
    
    # Level predictions
    print("\n" + "─"*70)
    print("LEVEL-BY-LEVEL DECISIONS")
    print("─"*70)
    
    # Level 1
    if pd.notna(patient_row['L1_Pred']):
        l1_pred = "PROGRESSION" if patient_row['L1_Pred'] == 1 else "NON-PROGRESSION"
        l1_conf = patient_row['L1_Conf']
        print(f"\n🔹 Level 1 (Volume-based):")
        print(f"   Prediction: {l1_pred}")
        print(f"   Confidence: {l1_conf:.1%}")
    else:
        print(f"\n🔹 Level 1: No data available")
    
    # Level 2
    if pd.notna(patient_row['L2_Pred']):
        l2_pred = "PROGRESSION" if patient_row['L2_Pred'] == 1 else "NON-PROGRESSION"
        l2_conf = patient_row['L2_Conf']
        print(f"\n🔹 Level 2 (Region-aware):")
        print(f"   Prediction: {l2_pred}")
        print(f"   Confidence: {l2_conf:.1%}")
        
        if patient_row['L2_Overrode_L1']:
            print(f"   ⚡ OVERRODE Level 1!")
    else:
        print(f"\n🔹 Level 2: No data available")
    
    # Level 3
    if pd.notna(patient_row['L3_Pred']):
        l3_pred = "PROGRESSION" if patient_row['L3_Pred'] == 1 else "NON-PROGRESSION"
        l3_conf = patient_row['L3_Conf']
        print(f"\n🔹 Level 3 (Temporal consistency):")
        print(f"   Prediction: {l3_pred}")
        print(f"   Confidence: {l3_conf:.1%}")
        print(f"   Agreement with L2: {patient_row['L3_Agreement']}")
        
        if patient_row['L3_Agreement'] == 'agree':
            print(f"   ✓ Confidence BOOSTED")
        elif patient_row['L3_Agreement'] == 'conflict':
            print(f"   ⚠ Confidence REDUCED (but decision unchanged)")
    else:
        print(f"\n🔹 Level 3: No temporal data (<3 scans)")
    
    # Final decision
    print("\n" + "─"*70)
    print("FINAL INTEGRATED DECISION")
    print("─"*70)
    
    final_pred = patient_row['Final_Label']
    final_conf = patient_row['Final_Confidence']
    source = patient_row['Source']
    correct = patient_row['Correct']
    
    print(f"\n🔮 Final Prediction: {final_pred}")
    print(f"📊 Final Confidence: {final_conf:.1%}")
    print(f"📍 Decision Source: {source}")
    
    if correct:
        print(f"\n✅ CORRECT - Matches ground truth!")
    else:
        print(f"\n❌ INCORRECT - Does not match ground truth")
    
    # Show explanation
    print(f"\n💡 Explanation:")
    print(f"   {patient_row['Explanation']}")
    
    # Show full trace
    print("\n" + "─"*70)
    print("COMPLETE DECISION TRACE")
    print("─"*70)
    print(trace_row['Trace'])
    
    # Load additional temporal features if available
    if L3_FEATURES.exists() and pd.notna(patient_row['L3_Pred']):
        l3_df = pd.read_csv(L3_FEATURES)
        patient_l3 = l3_df[l3_df['Patient_ID'] == patient_id]
        
        if len(patient_l3) > 0:
            feat = patient_l3.iloc[0]
            print("\n" + "─"*70)
            print("TEMPORAL FEATURES (Level 3)")
            print("─"*70)
            print(f"\n📈 Scan Information:")
            print(f"   Number of scans: {feat['N_Scans']}")
            print(f"   Time span: {feat['Time_Span_Weeks']:.0f} weeks")
            print(f"   Trajectory: {feat.get('Trajectory', 'N/A')}")
            
            print(f"\n📊 Trend Analysis:")
            print(f"   WT slope: {feat['WT_slope']:.4f} ml/week")
            print(f"   TC slope: {feat['TC_slope']:.4f} ml/week")
            print(f"   WT fraction increasing: {feat['WT_frac_increasing']*100:.0f}%")
            print(f"   TC fraction increasing: {feat['TC_frac_increasing']*100:.0f}%")
            print(f"   TC consecutive increases: {feat['TC_max_consecutive_inc']}")
    
    # Summary box
    print("\n" + "═"*70)
    print("SUMMARY")
    print("═"*70)
    print(f"""
    Patient:        {patient_id}
    Ground Truth:   {true_label}
    Final Decision: {final_pred} ({final_conf:.0%})
    Result:         {'✅ CORRECT' if correct else '❌ INCORRECT'}
    
    Decision Flow:
    L1 ({'✓' if pd.notna(patient_row['L1_Pred']) else '✗'}) → L2 ({'✓' if pd.notna(patient_row['L2_Pred']) else '✗'}) → L3 ({'✓' if pd.notna(patient_row['L3_Pred']) else '✗'})
    
    Key Events:
    {'• L2 overrode L1' if patient_row['L2_Overrode_L1'] else '• L2 confirmed L1'}
    {'• L3 boosted confidence (agreement)' if patient_row['L3_Agreement'] == 'agree' else '• L3 reduced confidence (conflict)' if patient_row['L3_Agreement'] == 'conflict' else '• No L3 adjustment'}
    """)


if __name__ == "__main__":
    main()
