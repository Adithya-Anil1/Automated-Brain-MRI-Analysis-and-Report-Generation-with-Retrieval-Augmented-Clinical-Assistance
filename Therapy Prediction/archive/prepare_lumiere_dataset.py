#!/usr/bin/env python3
"""
LUMIERE Dataset Preparation Script (Phase 0 - No ML)

This script processes the LUMIERE longitudinal glioma MRI dataset to create
a model-ready CSV for therapy response prediction.

Workflow:
1. Parse patient folder structure (Patient-0XX/week-*)
2. Identify baseline scans (week-000-*) and follow-up scans (week > 0)
3. Load segmentation masks and compute whole tumor volumes
4. Calculate delta volume between baseline and follow-up
5. Map RANO ratings to simplified labels:
   - CR or PR → Response
   - SD → Stable
   - PD → Progression
   - Pre-Op, Post-Op → Ignored
6. Output CSV with patient-followup pairs

Usage:
    python prepare_lumiere_dataset.py
    python prepare_lumiere_dataset.py --data_dir /path/to/extracted/data
    python prepare_lumiere_dataset.py --extract  # Extract zip first

Output:
    lumiere_phase0.csv - Ready for XGBoost training
"""

import os
import re
import sys
import argparse
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import nibabel as nib

# Constants
SCRIPT_DIR = Path(__file__).parent.absolute()
DATASET_DIR = SCRIPT_DIR / "dataset"
# After extraction, data is in dataset/Imaging/Patient-XXX/
DEFAULT_DATA_DIR = DATASET_DIR / "Imaging"
CSV_FILE = DATASET_DIR / "LUMIERE-ExpertRating-v202211.csv"
OUTPUT_FILE = SCRIPT_DIR / "lumiere_phase0.csv"

# RANO rating to simplified label mapping
RANO_TO_LABEL = {
    'CR': 'Response',      # Complete Response
    'PR': 'Response',      # Partial Response
    'SD': 'Stable',        # Stable Disease
    'PD': 'Progression',   # Progressive Disease
}

# Ratings to ignore
IGNORE_RATINGS = {'Pre-Op', 'Post-Op', 'Post-Op ', 'Post-Op/PD'}

# Week pattern for parsing folder names
WEEK_PATTERN = re.compile(r'^week-(\d{3})(?:-(\d+))?$')


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def print_step(step_num: int, text: str):
    """Print a step indicator."""
    print(f"\n{'─' * 60}")
    print(f"STEP {step_num}: {text}")
    print("─" * 60)


def extract_imaging_data(zip_path: Path, extract_to: Path) -> bool:
    """
    Extract the imaging data from zip file.
    
    Args:
        zip_path: Path to Imaging-v202211.zip
        extract_to: Directory to extract to
        
    Returns:
        True if extraction successful
    """
    if not zip_path.exists():
        print(f"  ❌ Zip file not found: {zip_path}")
        return False
    
    print(f"  📦 Extracting {zip_path.name}...")
    print(f"     This may take a while (31 GB)...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to.parent)
        print(f"  ✅ Extraction complete: {extract_to}")
        return True
    except Exception as e:
        print(f"  ❌ Extraction failed: {e}")
        return False


def parse_week_folder(folder_name: str) -> Optional[Tuple[int, int]]:
    """
    Parse a week folder name to extract week number and scan index.
    
    Args:
        folder_name: e.g., "week-000-1", "week-044", "week-000-2"
        
    Returns:
        Tuple of (week_number, scan_index) or None if invalid
        For week-044, returns (44, 1)
        For week-000-2, returns (0, 2)
    """
    match = WEEK_PATTERN.match(folder_name)
    if not match:
        return None
    
    week_num = int(match.group(1))
    scan_idx = int(match.group(2)) if match.group(2) else 1
    
    return (week_num, scan_idx)


def find_segmentation_file(week_folder: Path) -> Optional[Path]:
    """
    Find the segmentation file in a week folder.
    
    The LUMIERE dataset has automatic segmentations from HD-GLIO or DeepBraTumIA.
    Primary location: HD-GLIO-AUTO-segmentation/registered/segmentation.nii.gz
    
    Args:
        week_folder: Path to the week-XXX folder
        
    Returns:
        Path to segmentation file, or None if not found
    """
    # Priority 1: HD-GLIO-AUTO registered segmentation (aligned, preferred)
    hd_glio_seg = week_folder / "HD-GLIO-AUTO-segmentation" / "registered" / "segmentation.nii.gz"
    if hd_glio_seg.exists():
        return hd_glio_seg
    
    # Priority 2: HD-GLIO-AUTO native segmentations (pick one, preferring FLAIR)
    hd_glio_native = week_folder / "HD-GLIO-AUTO-segmentation" / "native"
    if hd_glio_native.exists():
        native_prefs = [
            "segmentation_FLAIR_origspace.nii.gz",
            "segmentation_T2_origspace.nii.gz",
            "segmentation_CT1_origspace.nii.gz",
            "segmentation_T1_origspace.nii.gz",
        ]
        for pref in native_prefs:
            seg_path = hd_glio_native / pref
            if seg_path.exists():
                return seg_path
    
    # Priority 3: DeepBraTumIA segmentations
    deep_seg_dir = week_folder / "DeepBraTumIA-segmentation" / "native" / "segmentation"
    if deep_seg_dir.exists():
        deep_prefs = [
            "flair_seg_mask.nii.gz",
            "t2_seg_mask.nii.gz",
            "ct1_seg_mask.nii.gz",
            "t1_seg_mask.nii.gz",
        ]
        for pref in deep_prefs:
            seg_path = deep_seg_dir / pref
            if seg_path.exists():
                return seg_path
    
    # Fallback: search for any segmentation file
    seg_patterns = [
        '**/segmentation.nii.gz',
        '**/segmentation*.nii.gz',
        '**/seg*.nii.gz',
    ]
    
    for pattern in seg_patterns:
        matches = list(week_folder.glob(pattern))
        if matches:
            return matches[0]
    
    return None


def compute_tumor_volume(segmentation_path: Path) -> Optional[float]:
    """
    Compute the whole tumor volume from a segmentation mask.
    
    Assumes any non-zero voxel is part of the tumor (whole tumor).
    Volume is computed in milliliters (ml).
    
    Args:
        segmentation_path: Path to the segmentation NIfTI file
        
    Returns:
        Tumor volume in ml, or None if computation fails
    """
    try:
        # Load the segmentation
        img = nib.load(str(segmentation_path))
        data = img.get_fdata()
        
        # Get voxel dimensions in mm
        voxel_dims = img.header.get_zooms()[:3]
        voxel_volume_mm3 = np.prod(voxel_dims)
        
        # Count non-zero voxels (whole tumor)
        tumor_voxels = np.sum(data > 0)
        
        # Convert to ml (1 ml = 1000 mm³)
        tumor_volume_ml = (tumor_voxels * voxel_volume_mm3) / 1000.0
        
        return tumor_volume_ml
        
    except Exception as e:
        print(f"    ⚠ Error computing volume for {segmentation_path}: {e}")
        return None


def load_rano_ratings(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load RANO ratings from the expert annotation CSV.
    
    Args:
        csv_path: Path to LUMIERE-ExpertRating-v202211.csv
        
    Returns:
        Nested dict: {patient_id: {week_folder: simplified_label}}
    """
    df = pd.read_csv(csv_path)
    
    # The rating column has a long name
    rating_col = [c for c in df.columns if 'Rating' in c and 'rationale' not in c.lower()][0]
    
    ratings = defaultdict(dict)
    skipped_ratings = defaultdict(int)
    
    for _, row in df.iterrows():
        patient = row['Patient']
        date = row['Date']  # This is actually the week folder name
        rating = str(row[rating_col]).strip() if pd.notna(row[rating_col]) else None
        
        if rating is None:
            skipped_ratings['NaN'] += 1
            continue
            
        if rating in IGNORE_RATINGS:
            skipped_ratings[rating] += 1
            continue
        
        # Map to simplified label
        simplified = RANO_TO_LABEL.get(rating)
        if simplified:
            ratings[patient][date] = simplified
        else:
            skipped_ratings[f'Unknown: {rating}'] += 1
    
    print(f"  📊 Loaded ratings for {len(ratings)} patients")
    print(f"  📊 Skipped ratings breakdown:")
    for rating, count in sorted(skipped_ratings.items()):
        print(f"      - {rating}: {count}")
    
    return dict(ratings)


def get_patient_scans(patient_folder: Path) -> Tuple[List[Path], List[Path]]:
    """
    Identify baseline and follow-up scans for a patient.
    
    Args:
        patient_folder: Path to Patient-XXX folder
        
    Returns:
        Tuple of (baseline_folders, followup_folders)
        Baseline folders are sorted by scan index (week-000-1 before week-000-2)
        Follow-up folders are sorted by week number
    """
    baselines = []
    followups = []
    
    for item in patient_folder.iterdir():
        if not item.is_dir():
            continue
            
        parsed = parse_week_folder(item.name)
        if parsed is None:
            continue
            
        week_num, scan_idx = parsed
        
        if week_num == 0:
            baselines.append((scan_idx, item))
        else:
            followups.append((week_num, scan_idx, item))
    
    # Sort baselines by scan index and followups by week number
    baselines.sort(key=lambda x: x[0])
    followups.sort(key=lambda x: (x[0], x[1]))
    
    baseline_folders = [b[1] for b in baselines]
    followup_folders = [f[2] for f in followups]
    
    return baseline_folders, followup_folders


def process_patient(
    patient_folder: Path, 
    ratings: Dict[str, str],
    baseline_cache: Dict[str, Tuple[Path, float]]
) -> List[Dict]:
    """
    Process a single patient to generate baseline-followup pairs.
    
    Args:
        patient_folder: Path to Patient-XXX folder
        ratings: Dict mapping week folder names to simplified labels
        baseline_cache: Cache of baseline (seg_path, volume) to avoid recomputation
        
    Returns:
        List of dicts, each representing one row in output CSV
    """
    patient_id = patient_folder.name
    rows = []
    
    # Get baseline and follow-up folders
    baseline_folders, followup_folders = get_patient_scans(patient_folder)
    
    if not baseline_folders:
        print(f"    ⚠ No baseline scans found for {patient_id}")
        return rows
    
    if not followup_folders:
        print(f"    ⚠ No follow-up scans found for {patient_id}")
        return rows
    
    # Use the earliest baseline (first week-000-* with valid segmentation)
    baseline_folder = None
    baseline_volume = None
    baseline_seg = None
    
    for bf in baseline_folders:
        seg_file = find_segmentation_file(bf)
        if seg_file:
            volume = compute_tumor_volume(seg_file)
            if volume is not None:
                baseline_folder = bf
                baseline_volume = volume
                baseline_seg = seg_file
                break
    
    if baseline_folder is None:
        print(f"    ⚠ No valid baseline segmentation for {patient_id}")
        return rows
    
    baseline_week = baseline_folder.name
    print(f"    📍 Baseline: {baseline_week} (Volume: {baseline_volume:.2f} ml)")
    
    # Process each follow-up
    for followup_folder in followup_folders:
        followup_week = followup_folder.name
        
        # Check if we have a rating for this follow-up
        if followup_week not in ratings:
            continue
        
        label = ratings[followup_week]
        
        # Find and load segmentation
        followup_seg = find_segmentation_file(followup_folder)
        if followup_seg is None:
            print(f"      ⚠ No segmentation for {followup_week}, skipping")
            continue
        
        followup_volume = compute_tumor_volume(followup_seg)
        if followup_volume is None:
            continue
        
        # Compute delta volume
        if baseline_volume > 0:
            delta_volume = (followup_volume - baseline_volume) / baseline_volume
        else:
            # Avoid division by zero - use a large value if tumor appeared
            delta_volume = float('inf') if followup_volume > 0 else 0.0
        
        # Create row
        row = {
            'Patient_ID': patient_id,
            'Baseline_Week': baseline_week,
            'Followup_Week': followup_week,
            'Baseline_Volume_ml': round(baseline_volume, 3),
            'Followup_Volume_ml': round(followup_volume, 3),
            'Delta_Volume': round(delta_volume, 4) if delta_volume != float('inf') else 999.0,
            'Response_Label': label
        }
        rows.append(row)
        
        print(f"      ✓ {followup_week}: {followup_volume:.2f} ml, Δ={delta_volume:+.2%}, {label}")
    
    return rows


def process_all_patients(data_dir: Path, ratings: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """
    Process all patients in the dataset.
    
    Args:
        data_dir: Path to extracted imaging data
        ratings: Nested dict of ratings
        
    Returns:
        DataFrame with all patient-followup pairs
    """
    all_rows = []
    patients_processed = 0
    patients_skipped = 0
    
    # Find all patient folders
    patient_folders = sorted([
        d for d in data_dir.iterdir() 
        if d.is_dir() and d.name.startswith('Patient-')
    ])
    
    print(f"  📂 Found {len(patient_folders)} patient folders")
    
    for patient_folder in patient_folders:
        patient_id = patient_folder.name
        print(f"\n  Processing {patient_id}...")
        
        # Get ratings for this patient
        patient_ratings = ratings.get(patient_id, {})
        
        if not patient_ratings:
            print(f"    ⚠ No valid ratings for {patient_id}, skipping")
            patients_skipped += 1
            continue
        
        # Process patient
        baseline_cache = {}
        rows = process_patient(patient_folder, patient_ratings, baseline_cache)
        
        if rows:
            all_rows.extend(rows)
            patients_processed += 1
        else:
            patients_skipped += 1
    
    print(f"\n  📊 Summary:")
    print(f"      Patients processed: {patients_processed}")
    print(f"      Patients skipped: {patients_skipped}")
    print(f"      Total pairs generated: {len(all_rows)}")
    
    return pd.DataFrame(all_rows)


def validate_output(df: pd.DataFrame):
    """
    Validate and display statistics about the output DataFrame.
    """
    print_step(4, "VALIDATING OUTPUT")
    
    print(f"  📊 Dataset Statistics:")
    print(f"      Total samples: {len(df)}")
    print(f"      Unique patients: {df['Patient_ID'].nunique()}")
    
    print(f"\n  📊 Label Distribution:")
    label_counts = df['Response_Label'].value_counts()
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        print(f"      {label}: {count} ({pct:.1f}%)")
    
    print(f"\n  📊 Volume Statistics:")
    print(f"      Baseline Volume (ml):")
    print(f"        Mean: {df['Baseline_Volume_ml'].mean():.2f}")
    print(f"        Median: {df['Baseline_Volume_ml'].median():.2f}")
    print(f"        Range: [{df['Baseline_Volume_ml'].min():.2f}, {df['Baseline_Volume_ml'].max():.2f}]")
    
    print(f"      Follow-up Volume (ml):")
    print(f"        Mean: {df['Followup_Volume_ml'].mean():.2f}")
    print(f"        Median: {df['Followup_Volume_ml'].median():.2f}")
    print(f"        Range: [{df['Followup_Volume_ml'].min():.2f}, {df['Followup_Volume_ml'].max():.2f}]")
    
    print(f"      Delta Volume:")
    print(f"        Mean: {df['Delta_Volume'].mean():+.2%}")
    print(f"        Median: {df['Delta_Volume'].median():+.2%}")
    
    # Check for any issues
    issues = []
    if df['Delta_Volume'].isna().any():
        issues.append("Some Delta_Volume values are NaN")
    if (df['Delta_Volume'] == 999.0).any():
        issues.append("Some Delta_Volume values indicate division by zero")
    if len(df) == 0:
        issues.append("No samples generated!")
    
    if issues:
        print(f"\n  ⚠ Issues found:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print(f"\n  ✅ No issues found")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Prepare LUMIERE dataset for therapy response prediction'
    )
    parser.add_argument(
        '--data_dir', 
        type=Path, 
        default=DEFAULT_DATA_DIR,
        help='Path to extracted imaging data (default: dataset/Imaging-v202211)'
    )
    parser.add_argument(
        '--csv_file',
        type=Path,
        default=CSV_FILE,
        help='Path to RANO ratings CSV file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=OUTPUT_FILE,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--extract',
        action='store_true',
        help='Extract imaging data from zip file first'
    )
    
    args = parser.parse_args()
    
    print_header("LUMIERE DATASET PREPARATION (Phase 0)")
    print(f"\n📋 Configuration:")
    print(f"   Data directory: {args.data_dir}")
    print(f"   Ratings CSV: {args.csv_file}")
    print(f"   Output file: {args.output}")
    
    # Step 1: Extract data if requested
    if args.extract:
        print_step(1, "EXTRACTING IMAGING DATA")
        zip_path = DATASET_DIR / "Imaging-v202211.zip"
        if not extract_imaging_data(zip_path, args.data_dir):
            sys.exit(1)
    
    # Verify data directory exists
    if not args.data_dir.exists():
        print(f"\n❌ Data directory not found: {args.data_dir}")
        print(f"   Run with --extract flag to extract from zip, or specify correct --data_dir")
        
        # Check if zip exists
        zip_path = DATASET_DIR / "Imaging-v202211.zip"
        if zip_path.exists():
            print(f"   Found zip file: {zip_path}")
            print(f"   Run: python prepare_lumiere_dataset.py --extract")
        sys.exit(1)
    
    # Verify CSV exists
    if not args.csv_file.exists():
        print(f"\n❌ Ratings CSV not found: {args.csv_file}")
        sys.exit(1)
    
    # Step 2: Load RANO ratings
    print_step(2, "LOADING RANO RATINGS")
    ratings = load_rano_ratings(args.csv_file)
    
    if not ratings:
        print("❌ No valid ratings loaded")
        sys.exit(1)
    
    # Step 3: Process all patients
    print_step(3, "PROCESSING PATIENTS")
    df = process_all_patients(args.data_dir, ratings)
    
    if len(df) == 0:
        print("\n❌ No samples generated. Check data structure.")
        sys.exit(1)
    
    # Step 4: Validate output
    validate_output(df)
    
    # Step 5: Save output
    print_step(5, "SAVING OUTPUT")
    df.to_csv(args.output, index=False)
    print(f"  ✅ Saved to: {args.output}")
    
    # Display sample rows
    print(f"\n  📋 Sample rows:")
    print(df.head(10).to_string(index=False))
    
    print_header("COMPLETE")
    print(f"\n✅ Created {args.output.name} with {len(df)} samples")
    print(f"   Ready for XGBoost training!")


if __name__ == "__main__":
    main()
