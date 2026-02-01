#!/usr/bin/env python3
"""
check_longitudinal_data.py

Check how many patients have ≥3 scans for longitudinal progression analysis.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent

# Check both data sources
LUMIERE_DATA_PATH = BASE_DIR / "lumiere_phase0.csv"
RANO_PATH = BASE_DIR / "dataset" / "LUMIERE-ExpertRating-v202211.csv"
IMAGING_DIR = BASE_DIR / "dataset" / "Imaging"


def analyze_lumiere_data():
    """Analyze lumiere_phase0.csv for patient scan counts."""
    print("\n" + "="*70)
    print("ANALYZING LUMIERE PHASE0 DATA")
    print("="*70)
    
    df = pd.read_csv(LUMIERE_DATA_PATH)
    
    # Count unique timepoints per patient
    patient_scans = defaultdict(set)
    
    for _, row in df.iterrows():
        patient_id = row['Patient_ID']
        baseline_week = row['Baseline_Week']
        followup_week = row['Followup_Week']
        
        patient_scans[patient_id].add(baseline_week)
        patient_scans[patient_id].add(followup_week)
    
    # Count patients by number of scans
    scan_counts = defaultdict(int)
    for patient, scans in patient_scans.items():
        scan_counts[len(scans)] += 1
    
    print(f"\nTotal patients: {len(patient_scans)}")
    print(f"\nDistribution of scan counts per patient:")
    for n_scans in sorted(scan_counts.keys(), reverse=True):
        count = scan_counts[n_scans]
        print(f"  {n_scans} scans: {count} patients ({100*count/len(patient_scans):.1f}%)")
    
    # Find patients with ≥3 scans
    patients_with_3plus = {p: scans for p, scans in patient_scans.items() if len(scans) >= 3}
    
    print(f"\n✓ Patients with ≥3 scans: {len(patients_with_3plus)} ({100*len(patients_with_3plus)/len(patient_scans):.1f}%)")
    
    # Show top patients by scan count
    print(f"\nTop 10 patients by scan count:")
    sorted_patients = sorted(patient_scans.items(), key=lambda x: len(x[1]), reverse=True)
    for patient, scans in sorted_patients[:10]:
        print(f"  {patient}: {len(scans)} scans - {sorted(scans)}")
    
    return patients_with_3plus


def analyze_rano_data():
    """Analyze RANO data for patient scan counts."""
    print("\n" + "="*70)
    print("ANALYZING RANO EXPERT RATING DATA")
    print("="*70)
    
    df = pd.read_csv(RANO_PATH)
    df.columns = df.columns.str.strip()
    
    rating_col = [c for c in df.columns if 'Rating' in c and 'rationale' not in c.lower()][0]
    df = df.rename(columns={rating_col: 'RANO_Rating'})
    
    # Filter out Pre-Op and Post-Op
    df = df[~df['RANO_Rating'].isin(['Pre-Op', 'Post-Op'])].copy()
    
    # Count timepoints per patient
    patient_timepoints = df.groupby('Patient')['Date'].nunique().to_dict()
    
    # Count patients by number of timepoints
    scan_counts = defaultdict(int)
    for patient, n_timepoints in patient_timepoints.items():
        scan_counts[n_timepoints] += 1
    
    print(f"\nTotal patients with RANO assessments: {len(patient_timepoints)}")
    print(f"\nDistribution of RANO assessment counts per patient:")
    for n_scans in sorted(scan_counts.keys(), reverse=True):
        count = scan_counts[n_scans]
        print(f"  {n_scans} assessments: {count} patients ({100*count/len(patient_timepoints):.1f}%)")
    
    # Find patients with ≥3 assessments
    patients_with_3plus = {p: n for p, n in patient_timepoints.items() if n >= 3}
    
    print(f"\n✓ Patients with ≥3 RANO assessments: {len(patients_with_3plus)} ({100*len(patients_with_3plus)/len(patient_timepoints):.1f}%)")
    
    # Show top patients
    print(f"\nTop 10 patients by RANO assessment count:")
    sorted_patients = sorted(patient_timepoints.items(), key=lambda x: x[1], reverse=True)
    for patient, n_assessments in sorted_patients[:10]:
        patient_data = df[df['Patient'] == patient]['Date'].tolist()
        print(f"  {patient}: {n_assessments} assessments - {patient_data[:5]}{'...' if len(patient_data) > 5 else ''}")
    
    return patients_with_3plus


def analyze_imaging_data():
    """Check actual imaging directory for scan counts."""
    print("\n" + "="*70)
    print("ANALYZING IMAGING DIRECTORY")
    print("="*70)
    
    if not IMAGING_DIR.exists():
        print(f"\n⚠️  Imaging directory not found: {IMAGING_DIR}")
        return {}
    
    patient_scans = {}
    
    for patient_dir in sorted(IMAGING_DIR.iterdir()):
        if not patient_dir.is_dir():
            continue
        
        patient_id = patient_dir.name
        
        # Count week directories
        week_dirs = [d.name for d in patient_dir.iterdir() if d.is_dir() and d.name.startswith('week-')]
        
        patient_scans[patient_id] = len(week_dirs)
    
    # Count distribution
    scan_counts = defaultdict(int)
    for patient, n_scans in patient_scans.items():
        scan_counts[n_scans] += 1
    
    print(f"\nTotal patients with imaging: {len(patient_scans)}")
    print(f"\nDistribution of imaging timepoints per patient:")
    for n_scans in sorted(scan_counts.keys(), reverse=True):
        count = scan_counts[n_scans]
        print(f"  {n_scans} timepoints: {count} patients ({100*count/len(patient_scans):.1f}%)")
    
    # Find patients with ≥3 scans
    patients_with_3plus = {p: n for p, n in patient_scans.items() if n >= 3}
    
    print(f"\n✓ Patients with ≥3 imaging timepoints: {len(patients_with_3plus)} ({100*len(patients_with_3plus)/len(patient_scans):.1f}%)")
    
    # Show top patients
    print(f"\nTop 10 patients by imaging timepoint count:")
    sorted_patients = sorted(patient_scans.items(), key=lambda x: x[1], reverse=True)
    for patient, n_scans in sorted_patients[:10]:
        patient_dir = IMAGING_DIR / patient
        week_dirs = sorted([d.name for d in patient_dir.iterdir() if d.is_dir() and d.name.startswith('week-')])
        print(f"  {patient}: {n_scans} timepoints - {week_dirs[:5]}{'...' if len(week_dirs) > 5 else ''}")
    
    return patients_with_3plus


def cross_reference_data():
    """Cross-reference all three data sources."""
    print("\n" + "="*70)
    print("CROSS-REFERENCING DATA SOURCES")
    print("="*70)
    
    # Get patients with ≥3 scans from each source
    lumiere_3plus = set(analyze_lumiere_data().keys())
    rano_3plus = set(analyze_rano_data().keys())
    imaging_3plus = set(analyze_imaging_data().keys())
    
    # Find intersection
    all_sources = lumiere_3plus & rano_3plus & imaging_3plus
    
    print("\n" + "="*70)
    print("SUMMARY: PATIENTS WITH ≥3 SCANS IN ALL DATA SOURCES")
    print("="*70)
    
    print(f"\nPatients with ≥3 in lumiere_phase0.csv:  {len(lumiere_3plus)}")
    print(f"Patients with ≥3 in RANO assessments:    {len(rano_3plus)}")
    print(f"Patients with ≥3 in imaging directory:   {len(imaging_3plus)}")
    print(f"\n{'─'*70}")
    print(f"✓ Patients with ≥3 in ALL sources:       {len(all_sources)}")
    print(f"{'─'*70}")
    
    if len(all_sources) > 0:
        print(f"\nThese {len(all_sources)} patients have complete longitudinal data:")
        for patient in sorted(all_sources)[:20]:  # Show first 20
            print(f"  • {patient}")
        if len(all_sources) > 20:
            print(f"  ... and {len(all_sources) - 20} more")
        
        print(f"\n✅ SUFFICIENT DATA FOR LONGITUDINAL ANALYSIS")
        print(f"   Can build Level 3+ models with temporal progression patterns")
    else:
        print(f"\n⚠️  INSUFFICIENT DATA FOR LONGITUDINAL ANALYSIS")
        print(f"   Need patients with ≥3 scans in all data sources")
    
    # Also show patients with ≥3 in at least 2 sources
    two_sources = (lumiere_3plus & rano_3plus) | (lumiere_3plus & imaging_3plus) | (rano_3plus & imaging_3plus)
    print(f"\n📊 Patients with ≥3 in at least 2 sources: {len(two_sources)}")


def analyze_progression_trajectories():
    """Analyze progression trajectories for patients with ≥3 scans."""
    print("\n" + "="*70)
    print("ANALYZING PROGRESSION TRAJECTORIES")
    print("="*70)
    
    df = pd.read_csv(RANO_PATH)
    df.columns = df.columns.str.strip()
    
    rating_col = [c for c in df.columns if 'Rating' in c and 'rationale' not in c.lower()][0]
    df = df.rename(columns={rating_col: 'RANO_Rating'})
    
    # Filter out Pre-Op and Post-Op
    df = df[~df['RANO_Rating'].isin(['Pre-Op', 'Post-Op'])].copy()
    
    # Find patients with ≥3 assessments
    patient_counts = df.groupby('Patient')['Date'].nunique()
    patients_3plus = patient_counts[patient_counts >= 3].index.tolist()
    
    print(f"\nAnalyzing {len(patients_3plus)} patients with ≥3 RANO assessments...")
    
    # Analyze trajectories
    trajectories = defaultdict(int)
    
    for patient in patients_3plus[:20]:  # Sample first 20
        patient_data = df[df['Patient'] == patient].sort_values('Date')
        trajectory = ' → '.join(patient_data['RANO_Rating'].tolist())
        
        if len(patient_data) <= 5:  # Only show shorter trajectories
            print(f"\n{patient}:")
            print(f"  {trajectory}")
    
    print("\n" + "="*70)


def main():
    """Main analysis."""
    print("\n" + "="*70)
    print("LONGITUDINAL DATA AVAILABILITY CHECK")
    print("="*70)
    print("\nChecking database for patients with ≥3 scans for temporal analysis...")
    
    # Analyze each data source
    cross_reference_data()
    
    # Analyze progression trajectories
    analyze_progression_trajectories()


if __name__ == "__main__":
    main()
