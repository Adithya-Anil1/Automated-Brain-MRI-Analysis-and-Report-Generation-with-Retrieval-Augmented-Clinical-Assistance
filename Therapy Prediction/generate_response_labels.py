#!/usr/bin/env python3
"""
Phase 0: Generate Therapy Response Labels

This script generates ground-truth response labels for therapy response prediction
by comparing PRE and POST treatment tumor volumes.

Workflow:
1. Match PRE and POST cases by patient ID (folder name)
2. Compute total tumor volume from segmentation masks
3. Calculate percentage volume change
4. Assign response labels based on volume change thresholds

Response Label Criteria:
- Regression:  Volume decrease > 25% (change < -25%)
- Stable:      Volume change between -25% and +25%
- Progression: Volume increase > 25% (change > +25%)

Output:
- CSV file with columns: Patient_ID, Pre_Volume, Post_Volume, Volume_Change_Percent, Response_Label

Usage:
    python generate_response_labels.py
    python generate_response_labels.py --pre_dir <path> --post_dir <path>
    python generate_response_labels.py --output labels.csv
"""

import os
import sys
import argparse
import re
from pathlib import Path
from datetime import datetime
import csv

try:
    import nibabel as nib
    import numpy as np
except ImportError:
    print("Error: Required packages not installed.")
    print("Please run: pip install nibabel numpy")
    sys.exit(1)

# Get the script directory
SCRIPT_DIR = Path(__file__).parent.absolute()

# Default directories
DEFAULT_PRE_DIR = SCRIPT_DIR / "data" / "pre_treatment"
DEFAULT_POST_DIR = SCRIPT_DIR / "data" / "post_treatment"
DEFAULT_OUTPUT = SCRIPT_DIR / "response_labels.csv"

# BraTS label values for whole tumor
# NCR (Necrotic Core) = 1
# ED (Edema) = 2  
# ET (Enhancing Tumor) = 4 (in BraTS 2021) or 3 (in some versions)
TUMOR_LABELS = [1, 2, 3, 4]  # Include all possible tumor labels


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def print_step(step_num, text):
    """Print a step indicator."""
    print(f"\n{'─' * 70}")
    print(f"STEP {step_num}: {text}")
    print("─" * 70)


def extract_patient_id(folder_name):
    """
    Extract the core patient ID from a folder name.
    
    Handles various naming conventions:
    - BraTS-GLI-00001-000
    - BraTS-GLI-00001-001
    - BraTS2024-GLI-00001-000
    
    Returns the numeric patient ID (e.g., "00001")
    """
    # Try to extract 5-digit patient ID
    match = re.search(r'(\d{5})', folder_name)
    if match:
        return match.group(1)
    return None


def find_segmentation_file(case_folder):
    """
    Find the segmentation file in a case folder.
    
    Looks for files with 'seg' in the name.
    """
    case_folder = Path(case_folder)
    
    # Common segmentation file patterns
    patterns = [
        "*seg*.nii.gz",
        "*seg*.nii",
        "*_seg.nii.gz",
        "*-seg.nii.gz",
    ]
    
    for pattern in patterns:
        matches = list(case_folder.glob(pattern))
        if matches:
            return matches[0]
    
    return None


def compute_tumor_volume(segmentation_path, voxel_volume_mm3=None):
    """
    Compute the whole tumor volume from a segmentation mask.
    
    Whole tumor = NCR + ED + ET (all non-zero tumor labels)
    
    Args:
        segmentation_path: Path to the segmentation NIfTI file
        voxel_volume_mm3: Optional voxel volume in mm³. If None, computed from header.
    
    Returns:
        Tuple of (volume_mm3, volume_ml, voxel_count)
    """
    try:
        # Load the segmentation
        nii = nib.load(str(segmentation_path))
        seg_data = nii.get_fdata().astype(np.int32)
        
        # Get voxel dimensions from header
        if voxel_volume_mm3 is None:
            voxel_dims = nii.header.get_zooms()[:3]  # Get x, y, z dimensions
            voxel_volume_mm3 = np.prod(voxel_dims)
        
        # Count voxels that are part of the tumor (any non-zero label)
        tumor_mask = np.isin(seg_data, TUMOR_LABELS)
        voxel_count = np.sum(tumor_mask)
        
        # Calculate volume
        volume_mm3 = voxel_count * voxel_volume_mm3
        volume_ml = volume_mm3 / 1000.0  # Convert to ml (cm³)
        
        return volume_mm3, volume_ml, int(voxel_count)
        
    except Exception as e:
        print(f"  ⚠ Error computing volume: {e}")
        return None, None, None


def calculate_volume_change(pre_volume, post_volume):
    """
    Calculate percentage volume change.
    
    Formula: (POST - PRE) / PRE × 100
    
    Negative = tumor shrinkage (regression)
    Positive = tumor growth (progression)
    """
    if pre_volume is None or post_volume is None:
        return None
    
    if pre_volume == 0:
        if post_volume == 0:
            return 0.0  # No tumor in either
        else:
            return float('inf')  # New tumor appeared
    
    return ((post_volume - pre_volume) / pre_volume) * 100


def assign_response_label(volume_change_percent):
    """
    Assign therapy response label based on volume change.
    
    Criteria:
    - Regression:  Volume decrease > 25% (change < -25%)
    - Stable:      Volume change between -25% and +25%
    - Progression: Volume increase > 25% (change > +25%)
    """
    if volume_change_percent is None:
        return "Unknown"
    
    if volume_change_percent < -25:
        return "Regression"
    elif volume_change_percent > 25:
        return "Progression"
    else:
        return "Stable"


def find_matching_cases(pre_dir, post_dir):
    """
    Find matching PRE and POST cases by patient ID.
    
    Returns:
        List of tuples: (patient_id, pre_folder, post_folder)
    """
    pre_dir = Path(pre_dir)
    post_dir = Path(post_dir)
    
    # Build dictionaries of patient ID → folder path
    pre_cases = {}
    post_cases = {}
    
    # Scan PRE directory
    if pre_dir.exists():
        for folder in pre_dir.iterdir():
            if folder.is_dir():
                patient_id = extract_patient_id(folder.name)
                if patient_id:
                    pre_cases[patient_id] = folder
    
    # Scan POST directory
    if post_dir.exists():
        for folder in post_dir.iterdir():
            if folder.is_dir():
                patient_id = extract_patient_id(folder.name)
                if patient_id:
                    post_cases[patient_id] = folder
    
    # Find matches
    matches = []
    common_ids = set(pre_cases.keys()) & set(post_cases.keys())
    
    for patient_id in sorted(common_ids):
        matches.append((patient_id, pre_cases[patient_id], post_cases[patient_id]))
    
    return matches, pre_cases, post_cases


def generate_labels(pre_dir, post_dir, output_path, verbose=True):
    """
    Generate therapy response labels for all matching cases.
    
    Args:
        pre_dir: Directory containing PRE-treatment cases
        post_dir: Directory containing POST-treatment cases
        output_path: Path for output CSV file
        verbose: Print progress information
    
    Returns:
        List of result dictionaries
    """
    pre_dir = Path(pre_dir)
    post_dir = Path(post_dir)
    output_path = Path(output_path)
    
    results = []
    
    # =========================================================================
    # STEP 1: Find matching cases
    # =========================================================================
    if verbose:
        print_step(1, "FINDING MATCHING CASES")
    
    matches, pre_cases, post_cases = find_matching_cases(pre_dir, post_dir)
    
    if verbose:
        print(f"  📂 PRE-treatment cases found:  {len(pre_cases)}")
        print(f"  📂 POST-treatment cases found: {len(post_cases)}")
        print(f"  🔗 Matched cases: {len(matches)}")
        
        # Show unmatched cases
        pre_only = set(pre_cases.keys()) - set(post_cases.keys())
        post_only = set(post_cases.keys()) - set(pre_cases.keys())
        
        if pre_only:
            print(f"  ⚠ PRE-only cases (no POST match): {len(pre_only)}")
        if post_only:
            print(f"  ⚠ POST-only cases (no PRE match): {len(post_only)}")
    
    if not matches:
        print("\n❌ No matching cases found!")
        print("   Make sure PRE and POST directories contain patient data.")
        return results
    
    # =========================================================================
    # STEP 2: Compute volumes and generate labels
    # =========================================================================
    if verbose:
        print_step(2, "COMPUTING TUMOR VOLUMES")
    
    for i, (patient_id, pre_folder, post_folder) in enumerate(matches):
        if verbose:
            print(f"\n  [{i+1}/{len(matches)}] Patient {patient_id}")
        
        # Find segmentation files
        pre_seg = find_segmentation_file(pre_folder)
        post_seg = find_segmentation_file(post_folder)
        
        if pre_seg is None:
            if verbose:
                print(f"     ⚠ No PRE segmentation found in {pre_folder.name}")
            continue
        
        if post_seg is None:
            if verbose:
                print(f"     ⚠ No POST segmentation found in {post_folder.name}")
            continue
        
        # Compute volumes
        pre_vol_mm3, pre_vol_ml, pre_voxels = compute_tumor_volume(pre_seg)
        post_vol_mm3, post_vol_ml, post_voxels = compute_tumor_volume(post_seg)
        
        if pre_vol_ml is None or post_vol_ml is None:
            if verbose:
                print(f"     ⚠ Could not compute volumes")
            continue
        
        # Calculate change and assign label
        volume_change = calculate_volume_change(pre_vol_ml, post_vol_ml)
        response_label = assign_response_label(volume_change)
        
        # Store result
        result = {
            "Patient_ID": patient_id,
            "Pre_Folder": pre_folder.name,
            "Post_Folder": post_folder.name,
            "Pre_Volume_ml": round(pre_vol_ml, 2),
            "Post_Volume_ml": round(post_vol_ml, 2),
            "Volume_Change_Percent": round(volume_change, 2) if volume_change is not None else None,
            "Response_Label": response_label,
            "Pre_Voxel_Count": pre_voxels,
            "Post_Voxel_Count": post_voxels,
        }
        results.append(result)
        
        if verbose:
            arrow = "↓" if volume_change < 0 else "↑" if volume_change > 0 else "→"
            print(f"     PRE:  {pre_vol_ml:.2f} ml ({pre_voxels} voxels)")
            print(f"     POST: {post_vol_ml:.2f} ml ({post_voxels} voxels)")
            print(f"     Change: {volume_change:+.1f}% {arrow} → {response_label}")
    
    # =========================================================================
    # STEP 3: Save results to CSV
    # =========================================================================
    if verbose:
        print_step(3, "SAVING RESULTS")
    
    if results:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write CSV with standard columns first, then additional columns
        fieldnames = [
            "Patient_ID",
            "Pre_Volume_ml",
            "Post_Volume_ml", 
            "Volume_Change_Percent",
            "Response_Label",
            "Pre_Folder",
            "Post_Folder",
            "Pre_Voxel_Count",
            "Post_Voxel_Count",
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        if verbose:
            print(f"  ✅ Saved {len(results)} records to: {output_path}")
    else:
        if verbose:
            print(f"  ⚠ No results to save")
    
    return results


def print_summary(results):
    """Print a summary of the generated labels."""
    if not results:
        return
    
    print_header("LABEL GENERATION SUMMARY")
    
    # Count labels
    label_counts = {"Regression": 0, "Stable": 0, "Progression": 0, "Unknown": 0}
    for r in results:
        label = r.get("Response_Label", "Unknown")
        label_counts[label] = label_counts.get(label, 0) + 1
    
    total = len(results)
    
    print(f"\n📊 Response Label Distribution:")
    print(f"   ├─ Regression:  {label_counts['Regression']:3d} ({100*label_counts['Regression']/total:.1f}%)")
    print(f"   ├─ Stable:      {label_counts['Stable']:3d} ({100*label_counts['Stable']/total:.1f}%)")
    print(f"   ├─ Progression: {label_counts['Progression']:3d} ({100*label_counts['Progression']/total:.1f}%)")
    if label_counts['Unknown'] > 0:
        print(f"   └─ Unknown:     {label_counts['Unknown']:3d} ({100*label_counts['Unknown']/total:.1f}%)")
    print(f"\n   Total: {total} patients")
    
    # Volume statistics
    pre_vols = [r["Pre_Volume_ml"] for r in results if r["Pre_Volume_ml"] is not None]
    post_vols = [r["Post_Volume_ml"] for r in results if r["Post_Volume_ml"] is not None]
    changes = [r["Volume_Change_Percent"] for r in results if r["Volume_Change_Percent"] is not None]
    
    if pre_vols:
        print(f"\n📏 Volume Statistics (ml):")
        print(f"   PRE:  mean={np.mean(pre_vols):.1f}, min={np.min(pre_vols):.1f}, max={np.max(pre_vols):.1f}")
        print(f"   POST: mean={np.mean(post_vols):.1f}, min={np.min(post_vols):.1f}, max={np.max(post_vols):.1f}")
    
    if changes:
        print(f"\n📈 Volume Change (%):")
        print(f"   Mean:   {np.mean(changes):+.1f}%")
        print(f"   Median: {np.median(changes):+.1f}%")
        print(f"   Range:  {np.min(changes):+.1f}% to {np.max(changes):+.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Generate therapy response labels from PRE/POST tumor volumes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
      Use default directories (data/pre_treatment and data/post_treatment)
      
  %(prog)s --pre_dir path/to/pre --post_dir path/to/post
      Specify custom directories
      
  %(prog)s --output my_labels.csv
      Custom output filename

Response Labels:
  Regression:  Volume decrease > 25%%
  Stable:      Volume change between -25%% and +25%%
  Progression: Volume increase > 25%%
        """
    )
    
    parser.add_argument(
        '--pre_dir',
        type=str,
        default=str(DEFAULT_PRE_DIR),
        help=f'Directory containing PRE-treatment cases (default: {DEFAULT_PRE_DIR})'
    )
    
    parser.add_argument(
        '--post_dir',
        type=str,
        default=str(DEFAULT_POST_DIR),
        help=f'Directory containing POST-treatment cases (default: {DEFAULT_POST_DIR})'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f'Output CSV file path (default: {DEFAULT_OUTPUT})'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    print_header("PHASE 0: THERAPY RESPONSE LABEL GENERATION")
    print(f"\n🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📂 PRE-treatment directory:  {args.pre_dir}")
    print(f"📂 POST-treatment directory: {args.post_dir}")
    print(f"📄 Output file: {args.output}")
    
    # Check directories exist
    pre_dir = Path(args.pre_dir)
    post_dir = Path(args.post_dir)
    
    if not pre_dir.exists():
        print(f"\n❌ Error: PRE-treatment directory not found: {pre_dir}")
        print("   Run 'python extract_datasets.py' first to extract the data.")
        return 1
    
    if not post_dir.exists():
        print(f"\n❌ Error: POST-treatment directory not found: {post_dir}")
        print("   Run 'python extract_datasets.py' first to extract the data.")
        return 1
    
    # Generate labels
    results = generate_labels(
        pre_dir=args.pre_dir,
        post_dir=args.post_dir,
        output_path=args.output,
        verbose=not args.quiet
    )
    
    # Print summary
    if results:
        print_summary(results)
        print(f"\n✅ Labels saved to: {args.output}")
        print(f"🕐 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 0
    else:
        print("\n❌ No labels generated. Check your data directories.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
