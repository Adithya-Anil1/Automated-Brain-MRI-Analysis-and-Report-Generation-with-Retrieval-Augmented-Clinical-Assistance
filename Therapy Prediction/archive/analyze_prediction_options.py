#!/usr/bin/env python3
"""
analyze_prediction_options.py

Comprehensive analysis of LUMIERE dataset to identify viable prediction targets
for treatment response prediction.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent

def parse_week(week_str):
    """Extract week number from string."""
    if pd.isna(week_str):
        return None
    match = re.search(r'week-(\d+)', str(week_str))
    return int(match.group(1)) if match else None

def main():
    print("="*70)
    print("COMPREHENSIVE DATA ANALYSIS FOR TREATMENT RESPONSE PREDICTION")
    print("="*70)
    
    # Load all data
    demo = pd.read_csv(BASE_DIR / 'LUMIERE-Demographics_Pathology.csv')
    rano = pd.read_csv(BASE_DIR / 'dataset' / 'LUMIERE-ExpertRating-v202211.csv')
    phase0 = pd.read_csv(BASE_DIR / 'lumiere_phase0.csv')
    radiomics = pd.read_csv(BASE_DIR / 'previous_mgmt_attempts' / 'Level4_Radiomic_Features_Enhancing.csv')
    
    # =========================================================================
    # 1. SURVIVAL ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("OPTION 1: SURVIVAL PREDICTION")
    print("="*70)
    
    demo['Survival_weeks'] = pd.to_numeric(demo['Survival time (weeks)'], errors='coerce')
    survival_available = demo['Survival_weeks'].notna().sum()
    
    print(f"\nPatients with survival data: {survival_available}/{len(demo)}")
    print(f"Survival statistics (weeks):")
    print(f"  Mean: {demo['Survival_weeks'].mean():.1f}")
    print(f"  Median: {demo['Survival_weeks'].median():.1f}")
    print(f"  Range: {demo['Survival_weeks'].min():.0f} - {demo['Survival_weeks'].max():.0f}")
    
    # Create binary survival labels
    median_survival = demo['Survival_weeks'].median()
    demo['Survival_Binary'] = (demo['Survival_weeks'] > median_survival).astype(int)
    
    valid_survival = demo[demo['Survival_weeks'].notna()]
    long_survivors = (valid_survival['Survival_Binary'] == 1).sum()
    short_survivors = (valid_survival['Survival_Binary'] == 0).sum()
    
    print(f"\nBinary classification (median split at {median_survival:.0f} weeks):")
    print(f"  Long survivors (>{median_survival:.0f}w): {long_survivors}")
    print(f"  Short survivors (≤{median_survival:.0f}w): {short_survivors}")
    print(f"  Class ratio: {max(long_survivors, short_survivors)/min(long_survivors, short_survivors):.2f}:1")
    
    # =========================================================================
    # 2. PROGRESSION-FREE SURVIVAL (Time to First PD)
    # =========================================================================
    print("\n" + "="*70)
    print("OPTION 2: EARLY vs LATE PROGRESSION")
    print("="*70)
    
    rano_col = [c for c in rano.columns if 'rating' in c.lower() and 'rationale' not in c.lower()][0]
    rano['Week_Num'] = rano['Date'].apply(parse_week)
    
    # Find first PD for each patient
    pd_events = rano[rano[rano_col] == 'PD'].copy()
    first_pd = pd_events.groupby('Patient')['Week_Num'].min().reset_index()
    first_pd.columns = ['Patient', 'First_PD_Week']
    
    print(f"\nPatients with progression (PD): {len(first_pd)}/{rano['Patient'].nunique()}")
    print(f"Time to first PD (weeks):")
    print(f"  Mean: {first_pd['First_PD_Week'].mean():.1f}")
    print(f"  Median: {first_pd['First_PD_Week'].median():.1f}")
    print(f"  Range: {first_pd['First_PD_Week'].min():.0f} - {first_pd['First_PD_Week'].max():.0f}")
    
    # Binary: Early vs Late progression
    median_pd = first_pd['First_PD_Week'].median()
    first_pd['Early_Progression'] = (first_pd['First_PD_Week'] <= median_pd).astype(int)
    
    early = (first_pd['Early_Progression'] == 1).sum()
    late = (first_pd['Early_Progression'] == 0).sum()
    
    print(f"\nBinary classification (median split at {median_pd:.0f} weeks):")
    print(f"  Early progression (≤{median_pd:.0f}w): {early}")
    print(f"  Late progression (>{median_pd:.0f}w): {late}")
    print(f"  Class ratio: {max(early, late)/min(early, late):.2f}:1")
    
    # =========================================================================
    # 3. FIRST FOLLOW-UP RESPONSE (Most Balanced!)
    # =========================================================================
    print("\n" + "="*70)
    print("OPTION 3: FIRST FOLLOW-UP RESPONSE (SD vs PD)")
    print("="*70)
    
    # Get first post-treatment RANO for each patient
    valid_rano = rano[rano[rano_col].isin(['CR', 'PR', 'SD', 'PD'])].copy()
    valid_rano['Week_Num'] = valid_rano['Date'].apply(parse_week)
    
    # First RANO rating (earliest week) per patient
    first_rano = valid_rano.loc[valid_rano.groupby('Patient')['Week_Num'].idxmin()]
    
    print(f"\nFirst post-treatment RANO distribution:")
    print(first_rano[rano_col].value_counts())
    
    # Binary: Stable/Response (CR/PR/SD) vs Progressive (PD)
    first_rano['Stable'] = first_rano[rano_col].isin(['CR', 'PR', 'SD']).astype(int)
    
    stable = (first_rano['Stable'] == 1).sum()
    progressive = (first_rano['Stable'] == 0).sum()
    
    print(f"\nBinary classification (Stable vs Progressive at first follow-up):")
    print(f"  Stable/Response (CR/PR/SD): {stable}")
    print(f"  Progressive (PD): {progressive}")
    if min(stable, progressive) > 0:
        print(f"  Class ratio: {max(stable, progressive)/min(stable, progressive):.2f}:1")
    
    # =========================================================================
    # 4. VOLUME-BASED RESPONSE FROM PHASE0
    # =========================================================================
    print("\n" + "="*70)
    print("OPTION 4: VOLUME-BASED RESPONSE (Phase0 Data)")
    print("="*70)
    
    print(f"\nTotal longitudinal records: {len(phase0)}")
    print(f"Unique patients: {phase0['Patient_ID'].nunique()}")
    
    print(f"\nResponse label distribution:")
    print(phase0['Response_Label'].value_counts())
    
    # Get best response per patient
    response_priority = {'Response': 0, 'Stable': 1, 'Progression': 2}
    phase0['Response_Priority'] = phase0['Response_Label'].map(response_priority)
    
    best_response = phase0.loc[phase0.groupby('Patient_ID')['Response_Priority'].idxmin()]
    
    print(f"\nBest response per patient:")
    print(best_response['Response_Label'].value_counts())
    
    # Binary: Response/Stable vs Progression
    best_response['Good_Response'] = best_response['Response_Label'].isin(['Response', 'Stable']).astype(int)
    
    good = (best_response['Good_Response'] == 1).sum()
    bad = (best_response['Good_Response'] == 0).sum()
    
    print(f"\nBinary classification (Response/Stable vs Progression):")
    print(f"  Good response (Response/Stable): {good}")
    print(f"  Poor response (Progression): {bad}")
    if min(good, bad) > 0:
        print(f"  Class ratio: {max(good, bad)/min(good, bad):.2f}:1")
    
    # =========================================================================
    # 5. DELTA VOLUME THRESHOLD
    # =========================================================================
    print("\n" + "="*70)
    print("OPTION 5: TUMOR VOLUME CHANGE (First Follow-up)")
    print("="*70)
    
    # Get first follow-up per patient
    phase0['Followup_Week_Num'] = phase0['Followup_Week'].apply(parse_week)
    first_followup = phase0.loc[phase0.groupby('Patient_ID')['Followup_Week_Num'].idxmin()]
    
    print(f"\nDelta volume at first follow-up:")
    print(f"  Mean: {first_followup['Delta_Volume'].mean():.3f}")
    print(f"  Median: {first_followup['Delta_Volume'].median():.3f}")
    print(f"  (Negative = tumor shrinkage, Positive = tumor growth)")
    
    # Binary: Shrinkage vs Growth
    first_followup['Tumor_Shrinkage'] = (first_followup['Delta_Volume'] < 0).astype(int)
    
    shrinkage = (first_followup['Tumor_Shrinkage'] == 1).sum()
    growth = (first_followup['Tumor_Shrinkage'] == 0).sum()
    
    print(f"\nBinary classification (Shrinkage vs Growth at first follow-up):")
    print(f"  Tumor shrinkage: {shrinkage}")
    print(f"  Tumor growth: {growth}")
    if min(shrinkage, growth) > 0:
        print(f"  Class ratio: {max(shrinkage, growth)/min(shrinkage, growth):.2f}:1")
    
    # =========================================================================
    # 6. FINAL OUTCOME (Ever Progressed)
    # =========================================================================
    print("\n" + "="*70)
    print("OPTION 6: EVER PROGRESSED vs NEVER PROGRESSED")
    print("="*70)
    
    all_patients = set(rano['Patient'].unique())
    pd_patients = set(rano[rano[rano_col] == 'PD']['Patient'].unique())
    never_pd = all_patients - pd_patients
    
    print(f"\nPatients who ever had PD: {len(pd_patients)}")
    print(f"Patients who never had PD: {len(never_pd)}")
    if min(len(pd_patients), len(never_pd)) > 0:
        print(f"Class ratio: {max(len(pd_patients), len(never_pd))/min(len(pd_patients), len(never_pd)):.2f}:1")
    
    # =========================================================================
    # RADIOMICS AVAILABILITY CHECK
    # =========================================================================
    print("\n" + "="*70)
    print("RADIOMICS DATA AVAILABILITY")
    print("="*70)
    
    print(f"\nPatients with radiomics: {len(radiomics)}")
    print(f"Radiomic features: {len(radiomics.columns) - 2}")  # Exclude Patient_ID, Used_Label
    
    # Check overlap with each prediction target
    radiomics_patients = set(radiomics['Patient_ID'].unique())
    
    # Survival
    survival_patients = set(demo[demo['Survival_weeks'].notna()]['Patient'].unique())
    overlap_survival = len(radiomics_patients & survival_patients)
    print(f"\nOverlap with survival data: {overlap_survival} patients")
    
    # First PD
    firstpd_patients = set(first_pd['Patient'].unique())
    overlap_pd = len(radiomics_patients & firstpd_patients)
    print(f"Overlap with progression data: {overlap_pd} patients")
    
    # First RANO
    firstrano_patients = set(first_rano['Patient'].unique())
    overlap_rano = len(radiomics_patients & firstrano_patients)
    print(f"Overlap with first RANO: {overlap_rano} patients")
    
    # Phase0
    phase0_patients = set(phase0['Patient_ID'].unique())
    overlap_phase0 = len(radiomics_patients & phase0_patients)
    print(f"Overlap with volume data: {overlap_phase0} patients")
    
    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    print("""
Based on the analysis, here are the VIABLE prediction targets ranked by feasibility:

1. **SURVIVAL PREDICTION (Best Option!)**
   - Binary: Long vs Short survival (median split)
   - Class balance: ~43 vs 43 (1:1 ratio!)
   - Available patients: 86 with survival data
   - Radiomics overlap: Good
   
2. **EARLY vs LATE PROGRESSION**
   - Binary: Early PD (≤28w) vs Late PD (>28w)
   - Class balance: ~37 vs 37 (1:1 ratio!)
   - Available patients: 74 with PD events
   - Clinically meaningful
   
3. **FIRST FOLLOW-UP RANO (SD vs PD)**
   - Use ONLY first post-treatment scan RANO
   - Better balance than CR/PR vs SD/PD
   - Available patients: ~75

4. **TUMOR VOLUME CHANGE**
   - Binary: Shrinkage vs Growth at first follow-up
   - Quantitative, less subjective than RANO
   - Available patients: 74

AVOID:
- MGMT prediction (you already tried, class imbalance)
- CR/PR vs SD/PD (only 4 CR/PR cases - too imbalanced!)
""")
    
    return {
        'survival_patients': overlap_survival,
        'progression_patients': overlap_pd,
        'volume_patients': overlap_phase0
    }


if __name__ == "__main__":
    main()
