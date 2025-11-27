#!/usr/bin/env python3
"""
Convert BraTS 2025 file naming convention to BraTS 2021 format.

BraTS 2025 format:
    BraTS-GLI-XXXXX-XXX-t1n.nii.gz  → T1 native
    BraTS-GLI-XXXXX-XXX-t1c.nii.gz  → T1 contrast-enhanced
    BraTS-GLI-XXXXX-XXX-t2w.nii.gz  → T2 weighted
    BraTS-GLI-XXXXX-XXX-t2f.nii.gz  → T2 FLAIR
    BraTS-GLI-XXXXX-XXX-seg.nii.gz  → Segmentation

BraTS 2021 format:
    BraTS-GLI-XXXXX-XXX_t1.nii.gz
    BraTS-GLI-XXXXX-XXX_t1ce.nii.gz
    BraTS-GLI-XXXXX-XXX_t2.nii.gz
    BraTS-GLI-XXXXX-XXX_flair.nii.gz
    BraTS-GLI-XXXXX-XXX_seg.nii.gz

Usage:
    python convert_brats2025_naming.py <folder_path>
    python convert_brats2025_naming.py <folder_path> --dry-run
    python convert_brats2025_naming.py <folder_path> --recursive
    
Examples:
    python convert_brats2025_naming.py BraTS-GLI-00003-000
    python convert_brats2025_naming.py BraTS_2025_Test_1 --recursive
    python convert_brats2025_naming.py . --recursive --dry-run
"""

import os
import sys
import argparse
import re
import gzip
import shutil
from pathlib import Path


# Mapping from BraTS 2025 suffixes to BraTS 2021 suffixes
SUFFIX_MAPPING = {
    't1n': 't1',      # T1 native → t1
    't1c': 't1ce',    # T1 contrast → t1ce (contrast-enhanced)
    't2w': 't2',      # T2 weighted → t2
    't2f': 'flair',   # T2 FLAIR → flair
    'seg': 'seg',     # Segmentation stays the same
}

# Regex pattern to match BraTS 2025 filenames
# Matches: BraTS-GLI-XXXXX-XXX-suffix.nii or BraTS-GLI-XXXXX-XXX-suffix.nii.gz
BRATS2025_PATTERN = re.compile(
    r'^(BraTS-GLI-\d{5}-\d{3})-(t1n|t1c|t2w|t2f|seg)\.(nii(?:\.gz)?)$'
)


def convert_filename(filename):
    """
    Convert a BraTS 2025 filename to BraTS 2021 format.
    
    Args:
        filename: Original filename (e.g., 'BraTS-GLI-00003-000-t1n.nii.gz')
        
    Returns:
        Tuple of (new_filename, needs_compression) or (None, False) if not a BraTS 2025 file
    """
    match = BRATS2025_PATTERN.match(filename)
    if not match:
        return None, False
    
    case_id = match.group(1)      # e.g., 'BraTS-GLI-00003-000'
    old_suffix = match.group(2)   # e.g., 't1n'
    extension = match.group(3)    # e.g., 'nii' or 'nii.gz'
    
    new_suffix = SUFFIX_MAPPING.get(old_suffix)
    if new_suffix is None:
        return None, False
    
    # Determine if we need to compress (if original is .nii without .gz)
    needs_compression = extension == 'nii'
    new_extension = 'nii.gz'  # Always use .nii.gz for output
    
    # Build new filename with underscore separator
    new_filename = f"{case_id}_{new_suffix}.{new_extension}"
    
    return new_filename, needs_compression


def compress_nifti(input_path, output_path):
    """Compress a .nii file to .nii.gz"""
    with open(input_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def process_folder(folder_path, dry_run=False, verbose=True):
    """
    Process a single folder containing BraTS 2025 files.
    
    Args:
        folder_path: Path to the folder
        dry_run: If True, only show what would be renamed without doing it
        verbose: If True, print detailed information
        
    Returns:
        Tuple of (files_renamed, files_skipped, errors)
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder does not exist: {folder}")
        return 0, 0, 1
    
    if not folder.is_dir():
        print(f"Error: Not a directory: {folder}")
        return 0, 0, 1
    
    files_renamed = 0
    files_skipped = 0
    errors = 0
    
    # Get all files in the folder
    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue
            
        filename = file_path.name
        new_filename, needs_compression = convert_filename(filename)
        
        if new_filename is None:
            # Not a BraTS 2025 file or already converted
            continue
        
        # Check if already in BraTS 2021 format
        if filename == new_filename:
            if verbose:
                print(f"  Already correct: {filename}")
            files_skipped += 1
            continue
        
        new_path = folder / new_filename
        
        # Check if target already exists
        if new_path.exists():
            print(f"  Warning: Target already exists, skipping: {new_filename}")
            files_skipped += 1
            continue
        
        if dry_run:
            if needs_compression:
                print(f"  [DRY RUN] Would rename and compress: {filename} → {new_filename}")
            else:
                print(f"  [DRY RUN] Would rename: {filename} → {new_filename}")
            files_renamed += 1
        else:
            try:
                if needs_compression:
                    # Compress .nii to .nii.gz
                    if verbose:
                        print(f"  Compressing and renaming: {filename} → {new_filename}")
                    compress_nifti(file_path, new_path)
                    # Remove original uncompressed file
                    file_path.unlink()
                else:
                    # Just rename
                    if verbose:
                        print(f"  Renaming: {filename} → {new_filename}")
                    file_path.rename(new_path)
                files_renamed += 1
            except Exception as e:
                print(f"  Error renaming {filename}: {e}")
                errors += 1
    
    return files_renamed, files_skipped, errors


def find_brats_folders(root_path, recursive=False):
    """
    Find all folders that contain BraTS 2025 files.
    
    Args:
        root_path: Root path to search
        recursive: If True, search subdirectories
        
    Returns:
        List of folder paths containing BraTS 2025 files
    """
    root = Path(root_path)
    folders_to_process = []
    
    if recursive:
        # Walk through all subdirectories
        for folder in root.rglob('*'):
            if folder.is_dir():
                # Check if this folder contains BraTS 2025 files
                has_brats2025 = any(
                    BRATS2025_PATTERN.match(f.name) 
                    for f in folder.iterdir() 
                    if f.is_file()
                )
                if has_brats2025:
                    folders_to_process.append(folder)
        
        # Also check the root folder itself
        has_brats2025 = any(
            BRATS2025_PATTERN.match(f.name) 
            for f in root.iterdir() 
            if f.is_file()
        )
        if has_brats2025:
            folders_to_process.insert(0, root)
    else:
        # Just process the given folder
        folders_to_process.append(root)
    
    return folders_to_process


def main():
    parser = argparse.ArgumentParser(
        description='Convert BraTS 2025 file naming to BraTS 2021 format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s BraTS-GLI-00003-000
      Convert files in a single case folder
      
  %(prog)s BraTS_2025_Test_1 --recursive
      Recursively convert all case folders under BraTS_2025_Test_1
      
  %(prog)s . --recursive --dry-run
      Preview what would be renamed in current directory and subdirectories

Naming Conversion:
  BraTS 2025          →  BraTS 2021
  ---------              ---------
  *-t1n.nii(.gz)      →  *_t1.nii.gz
  *-t1c.nii(.gz)      →  *_t1ce.nii.gz
  *-t2w.nii(.gz)      →  *_t2.nii.gz
  *-t2f.nii(.gz)      →  *_flair.nii.gz
  *-seg.nii(.gz)      →  *_seg.nii.gz
        """
    )
    
    parser.add_argument(
        'folder',
        help='Folder containing BraTS 2025 files (or parent folder with --recursive)'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Recursively process all subdirectories'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be renamed without actually renaming'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Only show summary, not individual file operations'
    )
    
    args = parser.parse_args()
    
    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"Error: Path does not exist: {folder_path}")
        sys.exit(1)
    
    # Find folders to process
    folders = find_brats_folders(folder_path, recursive=args.recursive)
    
    if not folders:
        print(f"No BraTS 2025 files found in: {folder_path}")
        if not args.recursive:
            print("Tip: Use --recursive to search subdirectories")
        sys.exit(0)
    
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Converting BraTS 2025 → BraTS 2021 naming convention")
    print(f"Found {len(folders)} folder(s) with BraTS 2025 files\n")
    
    total_renamed = 0
    total_skipped = 0
    total_errors = 0
    
    for folder in folders:
        print(f"Processing: {folder}")
        renamed, skipped, errors = process_folder(
            folder, 
            dry_run=args.dry_run, 
            verbose=not args.quiet
        )
        total_renamed += renamed
        total_skipped += skipped
        total_errors += errors
        print()
    
    # Summary
    print("=" * 50)
    print("Summary:")
    action = "Would rename" if args.dry_run else "Renamed"
    print(f"  {action}: {total_renamed} files")
    if total_skipped > 0:
        print(f"  Skipped: {total_skipped} files")
    if total_errors > 0:
        print(f"  Errors: {total_errors}")
    
    if args.dry_run and total_renamed > 0:
        print("\nRun without --dry-run to apply changes.")
    
    sys.exit(0 if total_errors == 0 else 1)


if __name__ == '__main__':
    main()
