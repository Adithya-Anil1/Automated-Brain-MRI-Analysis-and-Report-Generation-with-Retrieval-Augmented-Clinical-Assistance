"""
03_create_mgmt_dataset.py
Level 4: MGMT Prediction Dataset Creator

This script creates the master training list for MGMT methylation prediction
by linking patient demographics/pathology data with their baseline MRI scans.

Input:
    - LUMIERE_dataset_-_Demographics_and_pathology_information.csv (MGMT labels)
    - Imaging-v202211/ folder (MRI scans)

Output:
    - Level4_MGMT_Dataset.csv (Patient_ID, Baseline_Scan_Path, MGMT_Label)
"""

import os
import pandas as pd
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent
DEMOGRAPHICS_CSV = SCRIPT_DIR / "LUMIERE-Demographics_Pathology.csv"
IMAGING_FOLDER = SCRIPT_DIR / "dataset" / "Imaging"  # Extracted imaging data
OUTPUT_CSV = SCRIPT_DIR / "Level4_MGMT_Dataset.csv"


def load_mgmt_labels(csv_path: Path) -> pd.DataFrame:
    """
    Load demographics CSV and filter for valid MGMT labels.
    
    Returns DataFrame with Patient_ID and binary MGMT_Label.
    """
    print(f"Loading demographics from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"  Total patients in CSV: {len(df)}")
    
    # Check column names
    print(f"  Columns: {df.columns.tolist()}")
    
    # Get MGMT qualitative column
    mgmt_col = 'MGMT qualitative'
    if mgmt_col not in df.columns:
        raise ValueError(f"Column '{mgmt_col}' not found in CSV")
    
    # Show original distribution
    print(f"\n  Original MGMT distribution:")
    print(df[mgmt_col].value_counts(dropna=False).to_string().replace('\n', '\n    '))
    
    # Filter for valid MGMT status (exclude N/A, Unknown, na, blanks)
    invalid_values = ['na', 'n/a', 'unknown', '', 'nan']
    df_valid = df[~df[mgmt_col].str.lower().str.strip().isin(invalid_values)]
    df_valid = df_valid[df_valid[mgmt_col].notna()]
    
    print(f"\n  Valid MGMT patients: {len(df_valid)}")
    
    # Convert to binary: Methylated = 1, Unmethylated = 0
    def convert_mgmt(value):
        value_lower = str(value).lower().strip()
        if 'methylated' in value_lower and 'not' not in value_lower and 'un' not in value_lower:
            return 1  # Methylated
        elif 'unmethylated' in value_lower or 'not methylated' in value_lower:
            return 0  # Unmethylated
        else:
            return None  # Unknown format
    
    df_valid = df_valid.copy()
    df_valid['MGMT_Label'] = df_valid[mgmt_col].apply(convert_mgmt)
    
    # Remove any that couldn't be converted
    df_valid = df_valid[df_valid['MGMT_Label'].notna()]
    
    # Create clean output
    result = pd.DataFrame({
        'Patient_ID': df_valid['Patient'],
        'MGMT_Label': df_valid['MGMT_Label'].astype(int)
    })
    
    print(f"\n  Final binary distribution:")
    print(f"    Methylated (1): {(result['MGMT_Label'] == 1).sum()}")
    print(f"    Unmethylated (0): {(result['MGMT_Label'] == 0).sum()}")
    
    return result


def find_baseline_scans(df: pd.DataFrame, imaging_folder: Path) -> pd.DataFrame:
    """
    Find the baseline (week-000) scan path for each patient.
    
    Returns DataFrame with Baseline_Scan_Path column added.
    """
    print(f"\nSearching for baseline scans in: {imaging_folder}")
    
    if not imaging_folder.exists():
        print(f"  WARNING: Imaging folder not found!")
        print(f"  Creating placeholder paths (you'll need to update these)")
        df = df.copy()
        df['Baseline_Scan_Path'] = df['Patient_ID'].apply(
            lambda x: str(imaging_folder / x / "week-000-1")
        )
        return df
    
    baseline_paths = []
    patients_found = 0
    patients_missing = []
    
    for _, row in df.iterrows():
        patient_id = row['Patient_ID']
        patient_folder = imaging_folder / patient_id
        
        baseline_path = None
        
        if patient_folder.exists():
            # Look for week-000 folders (could be week-000, week-000-1, week-000-2)
            week_folders = sorted([
                f for f in patient_folder.iterdir() 
                if f.is_dir() and f.name.startswith('week-000')
            ])
            
            if week_folders:
                # Prefer week-000-1 (pre-op) or just week-000
                for preferred in ['week-000-1', 'week-000']:
                    for folder in week_folders:
                        if folder.name == preferred:
                            baseline_path = str(folder)
                            break
                    if baseline_path:
                        break
                
                # If no preferred found, take the first one
                if not baseline_path:
                    baseline_path = str(week_folders[0])
                
                patients_found += 1
        
        if baseline_path is None:
            patients_missing.append(patient_id)
            # Create placeholder path
            baseline_path = str(patient_folder / "week-000-1")
        
        baseline_paths.append(baseline_path)
    
    df = df.copy()
    df['Baseline_Scan_Path'] = baseline_paths
    
    print(f"  Patients with scans found: {patients_found}")
    print(f"  Patients missing scans: {len(patients_missing)}")
    
    if patients_missing and len(patients_missing) <= 10:
        print(f"  Missing: {patients_missing}")
    elif patients_missing:
        print(f"  Missing (first 10): {patients_missing[:10]}...")
    
    return df


def main():
    """Main pipeline to create MGMT dataset."""
    print("=" * 60)
    print("Level 4: MGMT Prediction Dataset Creator")
    print("=" * 60)
    
    # Step 1: Load and filter MGMT labels
    df_mgmt = load_mgmt_labels(DEMOGRAPHICS_CSV)
    
    # Step 2: Find baseline scan paths
    df_dataset = find_baseline_scans(df_mgmt, IMAGING_FOLDER)
    
    # Step 3: Reorder columns
    df_dataset = df_dataset[['Patient_ID', 'Baseline_Scan_Path', 'MGMT_Label']]
    
    # Step 4: Save to CSV
    df_dataset.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDataset saved to: {OUTPUT_CSV}")
    
    # Step 5: Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total valid patients: {len(df_dataset)}")
    print(f"  - Methylated (Chemo-Sensitive): {(df_dataset['MGMT_Label'] == 1).sum()}")
    print(f"  - Unmethylated (Chemo-Resistant): {(df_dataset['MGMT_Label'] == 0).sum()}")
    
    # Check if we meet the >50 threshold
    if len(df_dataset) > 50:
        print(f"\n✓ SUCCESS: Dataset has {len(df_dataset)} patients (>50 threshold)")
        print("  → Proceed with Option A (Radiogenomics Virtual Biopsy)")
    else:
        print(f"\n✗ WARNING: Dataset has only {len(df_dataset)} patients (<50 threshold)")
        print("  → Consider Option B or data augmentation")
    
    print("\nSample rows:")
    print(df_dataset.head(10).to_string(index=False))
    
    return df_dataset


if __name__ == "__main__":
    main()
