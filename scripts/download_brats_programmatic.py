"""
Programmatic download of BraTS 2024 data using Synapse Python client.
Downloads only the first N cases to save bandwidth and storage.
"""

import synapseclient
from synapseclient import Synapse
import synapseutils
from pathlib import Path
import zipfile
import shutil


def download_brats2024_programmatic(username, password, num_cases=10):
    """
    Download BraTS 2024 GLI training data programmatically.
    
    Args:
        username: Synapse username/email
        password: Synapse password
        num_cases: Number of cases to extract (default: 10)
    """
    
    print("=" * 80)
    print("BRATS 2024 PROGRAMMATIC DOWNLOAD")
    print("=" * 80)
    
    project_dir = Path(__file__).parent.parent.absolute()
    download_dir = project_dir / "downloads"
    extract_dir = project_dir / "data" / "BraTS2024_GLI"
    
    download_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    # Login to Synapse
    print("\n🔐 Logging in to Synapse...")
    syn = Synapse()
    
    try:
        syn.login(username, password)
        print("✅ Login successful!")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        print("\nMake sure:")
        print("  1. You have a Synapse account")
        print("  2. You've accepted BraTS 2024 terms at: https://www.synapse.org/#!Synapse:syn53708249")
        print("  3. Credentials are correct")
        return False
    
    # BraTS 2024 GLI Training Data
    # Main project: syn53708249
    # You need to find the specific file ID for TrainingData.zip
    
    print("\n📋 Browsing BraTS 2024 project structure...")
    
    try:
        # Get project details
        project = syn.get("syn53708249")
        print(f"✓ Found project: {project.name}")
        
        # List files in the project
        print("\n📁 Searching for training data...")
        
        # The training data zip file has a specific Synapse ID
        # We need to browse to find it - let me query the project structure
        
        # Method 1: Try to sync entire folder (will get manifest)
        print("\n⚠️  This will download the training data file...")
        print(f"   Destination: {download_dir}")
        
        response = input("\nProceed with download? (y/n): ").lower()
        if response != 'y':
            print("Download cancelled.")
            return False
        
        # Download using syncFromSynapse (gets manifest too)
        files = synapseutils.syncFromSynapse(
            syn, 
            entity="syn53708249",
            path=str(download_dir),
            downloadFile=True
        )
        
        print(f"\n✅ Downloaded {len(files)} files")
        
        # Find the training data zip
        zip_files = list(download_dir.rglob("*TrainingData.zip"))
        
        if not zip_files:
            print("❌ TrainingData.zip not found!")
            print("Files downloaded:")
            for f in download_dir.rglob("*"):
                if f.is_file():
                    print(f"   {f.name}")
            return False
        
        zip_file = zip_files[0]
        print(f"\n📦 Found: {zip_file.name}")
        print(f"   Size: {zip_file.stat().st_size / (1024**3):.2f} GB")
        
        # Extract only first N cases
        print(f"\n📂 Extracting first {num_cases} cases...")
        extract_selective(zip_file, extract_dir, num_cases)
        
        # Clean up
        print(f"\n🧹 Cleaning up...")
        if input("Delete downloaded zip file to save space? (y/n): ").lower() == 'y':
            zip_file.unlink()
            print("   ✓ Zip file deleted")
        
        print(f"\n✅ Complete! {num_cases} cases ready at: {extract_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def extract_selective(zip_path, extract_dir, num_cases=10):
    """Extract only first N cases from zip file"""
    
    print(f"   Opening {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get all files
        all_files = zip_ref.namelist()
        
        # Find case folders
        case_folders = set()
        for file in all_files:
            parts = Path(file).parts
            if len(parts) > 0 and parts[0].startswith('BraTS-GLI-'):
                case_folders.add(parts[0])
        
        # Sort and take first N
        case_folders = sorted(list(case_folders))[:num_cases]
        
        print(f"   Found {len(case_folders)} cases, extracting first {num_cases}:")
        
        for case in case_folders:
            print(f"      • {case}")
        
        # Extract files for selected cases
        extracted = 0
        for file in all_files:
            if any(file.startswith(case) for case in case_folders):
                zip_ref.extract(file, extract_dir)
                extracted += 1
        
        print(f"   ✓ Extracted {extracted} files")
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in extract_dir.rglob("*") if f.is_file())
    print(f"   Total size: {total_size / (1024**2):.1f} MB")


def quick_setup():
    """Quick setup and download"""
    
    print("=" * 80)
    print("BRATS 2024 QUICK SETUP")
    print("=" * 80)
    
    print("\n📥 This will:")
    print("   1. Connect to Synapse")
    print("   2. Download BraTS 2024 GLI Training Data")
    print("   3. Extract only first 10 cases (~100 MB)")
    print("   4. Delete the full download to save space")
    
    print("\n⚠️  You need:")
    print("   • Synapse account (register at https://www.synapse.org)")
    print("   • Accepted BraTS 2024 terms")
    print("   • ~5 GB temporary space (deleted after extraction)")
    
    proceed = input("\nReady to proceed? (y/n): ").lower()
    if proceed != 'y':
        print("Setup cancelled.")
        return
    
    # Get credentials
    print("\n🔐 Enter Synapse credentials:")
    username = input("Username/Email: ").strip()
    password = input("Password: ").strip()
    
    if not username or not password:
        print("❌ Credentials required!")
        return
    
    num_cases = input("\nHow many cases to download? (default: 10): ").strip()
    num_cases = int(num_cases) if num_cases else 10
    
    # Download
    success = download_brats2024_programmatic(username, password, num_cases)
    
    if success:
        print("\n" + "=" * 80)
        print("✅ SETUP COMPLETE!")
        print("=" * 80)
        print("\nNext steps:")
        print("   1. Convert to BraTS 2021 format")
        print("   2. Run batch inference")
        print("   3. Evaluate accuracy")
        print("\nRun: python scripts/convert_brats2024.py")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download BraTS 2024 data programmatically')
    parser.add_argument('--username', '-u', type=str, help='Synapse username/email')
    parser.add_argument('--password', '-p', type=str, help='Synapse password')
    parser.add_argument('--num-cases', '-n', type=int, default=10, 
                        help='Number of cases to extract (default: 10)')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive mode with prompts')
    
    args = parser.parse_args()
    
    if args.interactive or (not args.username or not args.password):
        quick_setup()
    else:
        download_brats2024_programmatic(args.username, args.password, args.num_cases)
