"""
Extract only first N cases from downloaded BraTS 2024 zip file.
Run this AFTER you manually download the zip.
"""

import zipfile
from pathlib import Path
import sys


def extract_limited_cases(zip_path, num_cases=10):
    """
    Extract only first N cases from BraTS 2024 zip file.
    
    Args:
        zip_path: Path to the downloaded zip file
        num_cases: Number of cases to extract (default: 10)
    """
    
    print("=" * 80)
    print(f"EXTRACTING {num_cases} BRATS 2024 CASES")
    print("=" * 80)
    
    zip_path = Path(zip_path)
    
    if not zip_path.exists():
        print(f"❌ File not found: {zip_path}")
        return False
    
    print(f"\n📦 Zip file: {zip_path.name}")
    print(f"   Size: {zip_path.stat().st_size / (1024**3):.2f} GB")
    
    # Output directory
    project_dir = Path(__file__).parent.parent.absolute()
    extract_dir = project_dir / "data" / "BraTS2024_GLI"
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Extracting to: {extract_dir}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get all files
            all_files = zip_ref.namelist()
            print(f"   Total files in zip: {len(all_files)}")
            
            # Find case folders
            case_folders = set()
            for file in all_files:
                parts = Path(file).parts
                if len(parts) > 0:
                    # Look for case folders (BraTS-GLI-xxxxx-xxx format)
                    if 'BraTS-GLI-' in parts[0]:
                        case_folders.add(parts[0])
                    elif len(parts) > 1 and 'BraTS-GLI-' in parts[1]:
                        case_folders.add(parts[1])
            
            # Sort and limit
            case_folders = sorted(list(case_folders))
            print(f"\n   Found {len(case_folders)} total cases")
            
            if num_cases > len(case_folders):
                num_cases = len(case_folders)
            
            selected_cases = case_folders[:num_cases]
            
            print(f"\n   Extracting first {num_cases} cases:")
            for i, case in enumerate(selected_cases, 1):
                print(f"      {i}. {case}")
            
            # Extract files
            print(f"\n   Extracting files...")
            extracted_count = 0
            
            for file in all_files:
                # Check if file belongs to one of our selected cases
                if any(case in file for case in selected_cases):
                    zip_ref.extract(file, extract_dir)
                    extracted_count += 1
                    
                    if extracted_count % 50 == 0:
                        print(f"      {extracted_count} files extracted...")
            
            print(f"\n   ✅ Extracted {extracted_count} files")
            
            # Calculate size
            total_size = sum(f.stat().st_size for f in extract_dir.rglob("*") if f.is_file())
            print(f"   Total size: {total_size / (1024**2):.1f} MB")
            
            # List extracted cases
            print(f"\n   📁 Extracted cases in: {extract_dir}")
            for case in sorted(extract_dir.iterdir()):
                if case.is_dir():
                    nii_files = list(case.glob("*.nii.gz"))
                    print(f"      • {case.name} ({len(nii_files)} files)")
            
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 80)
    print("BRATS 2024 - MANUAL EXTRACTION")
    print("=" * 80)
    
    print("\n📥 STEP 1: Download the zip file")
    print("-" * 80)
    print("Go to: https://www.synapse.org/#!Synapse:syn53708249/files/")
    print("Navigate to: Files > Data > BraTS-GLI")
    print("Download: BraTS2024-BraTS-GLI-TrainingData.zip (4.9 GB)")
    
    project_dir = Path(__file__).parent.parent.absolute()
    downloads_dir = project_dir / "downloads"
    
    print(f"\nSave it to: {downloads_dir}")
    
    print("\n📂 STEP 2: Extract specific cases")
    print("-" * 80)
    
    # Look for zip file
    possible_paths = [
        downloads_dir / "BraTS2024-BraTS-GLI-TrainingData.zip",
        Path.home() / "Downloads" / "BraTS2024-BraTS-GLI-TrainingData.zip",
    ]
    
    zip_path = None
    for path in possible_paths:
        if path.exists():
            zip_path = path
            break
    
    if zip_path:
        print(f"✅ Found zip file: {zip_path}")
        num_cases = input("\nHow many cases to extract? (default: 10): ").strip()
        num_cases = int(num_cases) if num_cases else 10
        
        extract_limited_cases(zip_path, num_cases)
    else:
        print("❌ Zip file not found!")
        print("\nPlease download it first, then run:")
        print("   python scripts/extract_brats2024.py <path-to-zip>")
        print("\nOr provide the path:")
        
        custom_path = input("\nZip file path (or press Enter to skip): ").strip()
        if custom_path:
            extract_limited_cases(custom_path, 10)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Path provided as argument
        zip_path = sys.argv[1]
        num_cases = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        extract_limited_cases(zip_path, num_cases)
    else:
        main()
