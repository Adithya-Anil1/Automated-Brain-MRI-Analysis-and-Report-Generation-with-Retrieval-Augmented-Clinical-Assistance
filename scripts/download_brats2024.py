"""
Download specific BraTS 2024 cases from Synapse.
This allows you to download only selected cases instead of the entire dataset.
"""

import os
import sys
from pathlib import Path

def setup_synapse():
    """Install and setup Synapse client"""
    print("=" * 80)
    print("BRATS 2024 DATA DOWNLOADER")
    print("=" * 80)
    
    print("\nStep 1: Installing Synapse client...")
    print("Run this command in your terminal:")
    print("   pip install synapseclient")
    print("\nThen run this script again.")
    
    try:
        import synapseclient
        from synapseclient import Synapse
        print("✓ Synapse client is installed!")
        return True
    except ImportError:
        print("\n⚠️  Synapse client not installed.")
        print("\nPlease run:")
        print("   pip install synapseclient")
        return False


def download_brats_cases():
    """Download specific BraTS 2024 GLI cases"""
    
    try:
        import synapseclient
        from synapseclient import Synapse
    except ImportError:
        print("Please install synapseclient first:")
        print("   pip install synapseclient")
        return
    
    print("\n" + "=" * 80)
    print("DOWNLOADING BRATS 2024 GLI CASES")
    print("=" * 80)
    
    # Get credentials
    print("\nPlease enter your Synapse credentials:")
    username = input("Username/Email: ").strip()
    password = input("Password: ").strip()
    
    if not username or not password:
        print("❌ Username and password required!")
        return
    
    # Login to Synapse
    print("\n🔐 Logging in to Synapse...")
    syn = Synapse()
    
    try:
        syn.login(username, password)
        print("✅ Login successful!")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        print("\nMake sure you:")
        print("1. Have a Synapse account (register at https://www.synapse.org/)")
        print("2. Have accepted the BraTS 2024 terms and conditions")
        print("3. Are using correct credentials")
        return
    
    # BraTS 2024 GLI Training Data
    # Synapse ID: syn53708249 (main project)
    # Training data folder ID - you'll need to browse to find the exact ID
    
    print("\n" + "=" * 80)
    print("IMPORTANT: MANUAL DOWNLOAD RECOMMENDED")
    print("=" * 80)
    
    print("\nThe Synapse API can be complex for large datasets.")
    print("Here's the EASIEST way to download specific cases:")
    
    print("\n📥 OPTION 1: Web Interface (Recommended)")
    print("-" * 80)
    print("1. Go to: https://www.synapse.org/#!Synapse:syn53708249")
    print("2. Click 'Files' tab")
    print("3. Navigate to: Data > BraTS-GLI > BraTS2024-BraTS-GLI-TrainingData.zip")
    print("4. Click the three dots (...) next to the file")
    print("5. Select 'Download Options' > 'Download File'")
    print("   OR")
    print("6. If the zip is too large, browse INSIDE the zip to download individual cases")
    
    print("\n📥 OPTION 2: Download Sample Cases Only")
    print("-" * 80)
    print("1. After downloading the full zip, extract it")
    print("2. Copy only the first 10 cases to your data folder")
    print("3. Delete the rest to save space")
    
    print("\n📥 OPTION 3: Use Synapse Command Line")
    print("-" * 80)
    print("Install Synapse client and run:")
    print("   synapse login -u <username> -p <password>")
    print("   synapse get syn53708249 --downloadLocation ./downloads")
    
    print("\n" + "=" * 80)
    print("AFTER DOWNLOADING")
    print("=" * 80)
    print("\nPlace the data in:")
    project_dir = Path(__file__).parent.absolute()
    data_dir = project_dir / "downloads" / "BraTS2024_GLI"
    print(f"   {data_dir}")
    
    print("\nExpected structure:")
    print("   downloads/BraTS2024_GLI/")
    print("   ├── BraTS-GLI-00000-000/")
    print("   │   ├── BraTS-GLI-00000-000-t1n.nii.gz")
    print("   │   ├── BraTS-GLI-00000-000-t1c.nii.gz")
    print("   │   ├── BraTS-GLI-00000-000-t2w.nii.gz")
    print("   │   ├── BraTS-GLI-00000-000-t2f.nii.gz")
    print("   │   └── BraTS-GLI-00000-000-seg.nii.gz")
    print("   ├── BraTS-GLI-00001-000/")
    print("   └── ...")
    
    print("\nThen I'll create a script to:")
    print("   ✓ Convert to BraTS 2021 format")
    print("   ✓ Run batch inference")
    print("   ✓ Evaluate accuracy")


def quick_download_guide():
    """Show quick download guide without requiring login"""
    
    print("=" * 80)
    print("QUICK GUIDE: Download BraTS 2024 Cases")
    print("=" * 80)
    
    print("\n🌐 EASIEST METHOD: Web Browser")
    print("-" * 80)
    print("\n1. Go to BraTS 2024 on Synapse:")
    print("   https://www.synapse.org/#!Synapse:syn53708249/files/")
    
    print("\n2. Navigate to:")
    print("   Files > Data > BraTS-GLI")
    
    print("\n3. Download the Training Data:")
    print("   • Click: BraTS2024-BraTS-GLI-TrainingData.zip")
    print("   • Size: ~4.9 GB")
    print("   • Contains: ~1,200 cases with ground truth")
    
    print("\n4. Save to:")
    project_dir = Path(__file__).parent.absolute()
    print(f"   {project_dir / 'downloads'}")
    
    print("\n5. Extract only 5-10 cases:")
    print("   • Extract the zip file")
    print("   • Copy first 10 folders (e.g., BraTS-GLI-00000-000 to BraTS-GLI-00009-000)")
    print("   • Place in: downloads/BraTS2024_GLI/")
    print("   • Delete the rest to save space")
    
    print("\n⚡ ALTERNATIVE: Download via Python (if you want)")
    print("-" * 80)
    print("If you want to use the Synapse API:")
    print("   1. pip install synapseclient")
    print("   2. Run this script with --download flag")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print("\n✅ For testing: Download 5-10 cases only (~50-100 MB)")
    print("   This is enough to evaluate model accuracy")
    
    print("\n📊 For comprehensive evaluation: Download all training cases")
    print("   Better statistics, but requires 4.9 GB")
    
    print("\n💡 TIP: Start small!")
    print("   Download 10 cases first, test the pipeline, then get more if needed.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download BraTS 2024 data')
    parser.add_argument('--download', action='store_true', 
                        help='Attempt to download using Synapse API')
    parser.add_argument('--setup', action='store_true',
                        help='Setup Synapse client')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_synapse()
    elif args.download:
        if setup_synapse():
            download_brats_cases()
    else:
        quick_download_guide()
