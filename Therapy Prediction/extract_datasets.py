#!/usr/bin/env python3
"""
Dataset Extraction Script for Therapy Response Analysis

Extracts and organizes BraTS datasets into PRE and POST treatment folders.

PRE-treatment: BraTS2025-GLI-PRE-* (input features)
POST-treatment: BraTS2024-BraTS-GLI-* (outcome measurement)

Usage:
    python extract_datasets.py
"""

import os
import sys
import zipfile
from pathlib import Path
import shutil

# Get the script directory
SCRIPT_DIR = Path(__file__).parent.absolute()
DATASET_DIR = SCRIPT_DIR / "dataset"

# Output directories
PRE_TREATMENT_DIR = SCRIPT_DIR / "data" / "pre_treatment"
POST_TREATMENT_DIR = SCRIPT_DIR / "data" / "post_treatment"

# Dataset zip files
DATASETS = {
    "pre_training": "BraTS2025-GLI-PRE-Challenge-TrainingData (1).zip",
    "pre_validation": "BraTS2025-GLI-PRE-Challenge-ValidationData.zip",
    "post_training": "BraTS2024-BraTS-GLI-TrainingData.zip",
    "post_validation": "BraTS2024-BraTS-GLI-ValidationData.zip",
}


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


def extract_zip(zip_path, output_dir, description):
    """
    Extract a zip file to the specified directory.
    
    Args:
        zip_path: Path to the zip file
        output_dir: Directory to extract to
        description: Description for logging
    
    Returns:
        Number of folders extracted
    """
    if not zip_path.exists():
        print(f"  ⚠ Warning: {zip_path.name} not found")
        return 0
    
    print(f"  📦 Extracting: {zip_path.name}")
    print(f"     → {output_dir}")
    
    # Create temp extraction directory
    temp_dir = output_dir.parent / f"temp_extract_{zip_path.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_dir)
        
        # Find patient folders (BraTS-GLI-XXXXX-XXX pattern)
        folders_moved = 0
        
        for root, dirs, files in os.walk(temp_dir):
            for d in dirs:
                # Match BraTS folder pattern
                if d.startswith("BraTS-GLI-") or d.startswith("BraTS2024") or d.startswith("BraTS2025"):
                    src = Path(root) / d
                    
                    # Check if this is a patient folder (contains .nii.gz files)
                    nii_files = list(src.glob("*.nii.gz"))
                    if nii_files:
                        dst = output_dir / d
                        if not dst.exists():
                            shutil.move(str(src), str(dst))
                            folders_moved += 1
                        else:
                            print(f"     ⚠ Already exists: {d}")
        
        # Also check for nested structures
        for item in temp_dir.iterdir():
            if item.is_dir():
                # Check for patient folders inside
                for sub in item.rglob("BraTS-GLI-*"):
                    if sub.is_dir():
                        nii_files = list(sub.glob("*.nii.gz"))
                        if nii_files:
                            dst = output_dir / sub.name
                            if not dst.exists():
                                shutil.move(str(sub), str(dst))
                                folders_moved += 1
        
        print(f"     ✅ Extracted {folders_moved} patient folders")
        return folders_moved
        
    except zipfile.BadZipFile:
        print(f"  ❌ Error: {zip_path.name} is corrupted")
        return 0
    finally:
        # Clean up temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def count_patient_folders(directory):
    """Count patient folders in a directory."""
    if not directory.exists():
        return 0
    return len([d for d in directory.iterdir() if d.is_dir() and d.name.startswith("BraTS")])


def main():
    print_header("DATASET EXTRACTION FOR THERAPY RESPONSE ANALYSIS")
    
    print(f"\n📂 Dataset source: {DATASET_DIR}")
    print(f"📂 PRE-treatment output: {PRE_TREATMENT_DIR}")
    print(f"📂 POST-treatment output: {POST_TREATMENT_DIR}")
    
    # Check if datasets exist
    print("\n📋 Checking dataset files...")
    for name, filename in DATASETS.items():
        path = DATASET_DIR / filename
        status = "✅" if path.exists() else "❌"
        size = f"({path.stat().st_size / 1e9:.2f} GB)" if path.exists() else "(missing)"
        print(f"   {status} {filename} {size}")
    
    # Create output directories
    PRE_TREATMENT_DIR.mkdir(parents=True, exist_ok=True)
    POST_TREATMENT_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # STEP 1: Extract PRE-treatment data (BraTS 2025)
    # =========================================================================
    print_step(1, "EXTRACTING PRE-TREATMENT DATA (BraTS 2025)")
    
    pre_count = 0
    for key in ["pre_training", "pre_validation"]:
        zip_path = DATASET_DIR / DATASETS[key]
        pre_count += extract_zip(zip_path, PRE_TREATMENT_DIR, key)
    
    # =========================================================================
    # STEP 2: Extract POST-treatment data (BraTS 2024)
    # =========================================================================
    print_step(2, "EXTRACTING POST-TREATMENT DATA (BraTS 2024)")
    
    post_count = 0
    for key in ["post_training", "post_validation"]:
        zip_path = DATASET_DIR / DATASETS[key]
        post_count += extract_zip(zip_path, POST_TREATMENT_DIR, key)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("EXTRACTION COMPLETE")
    
    # Recount to be accurate
    pre_total = count_patient_folders(PRE_TREATMENT_DIR)
    post_total = count_patient_folders(POST_TREATMENT_DIR)
    
    print(f"\n📊 Summary:")
    print(f"   PRE-treatment cases:  {pre_total}")
    print(f"   POST-treatment cases: {post_total}")
    print(f"\n📂 Data locations:")
    print(f"   PRE:  {PRE_TREATMENT_DIR}")
    print(f"   POST: {POST_TREATMENT_DIR}")
    
    print(f"\n💡 Next step: Run 'python generate_response_labels.py' to create labels")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
