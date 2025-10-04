"""
Simple BraTS 2024 downloader using Synapse authentication token.
More reliable than username/password login.
"""

import synapseclient
from synapseclient import Synapse
import synapseutils
from pathlib import Path
import zipfile


def download_with_token(num_cases=10):
    """Download using Synapse personal access token"""
    
    print("=" * 80)
    print("BRATS 2024 DOWNLOAD - TOKEN AUTHENTICATION")
    print("=" * 80)
    
    print("\n📝 To get your Synapse Personal Access Token:")
    print("   1. Go to: https://www.synapse.org/#!PersonalAccessTokens:")
    print("   2. Click 'Create New Token'")
    print("   3. Give it a name (e.g., 'BraTS Download')")
    print("   4. Select scopes: 'View' and 'Download'")
    print("   5. Click 'Create Token'")
    print("   6. Copy the token (it starts with 'eyJ...')")
    
    print("\n" + "=" * 80)
    
    token = input("\nEnter your Synapse Personal Access Token: ").strip()
    
    if not token:
        print("❌ Token required!")
        return False
    
    project_dir = Path(__file__).parent.parent.absolute()
    download_dir = project_dir / "downloads"
    extract_dir = project_dir / "data" / "BraTS2024_GLI"
    
    download_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    # Login with token
    print("\n🔐 Logging in with token...")
    syn = Synapse()
    
    try:
        syn.login(authToken=token)
        print("✅ Login successful!")
        
        # Get user info
        profile = syn.getUserProfile()
        print(f"   Logged in as: {profile['userName']}")
        
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return False
    
    # Download the training data
    print("\n📥 Downloading BraTS 2024 GLI Training Data...")
    print("   This may take a while (4.9 GB)...")
    
    try:
        # The specific file ID for BraTS2024-BraTS-GLI-TrainingData.zip
        # You need to browse the Synapse project to find this ID
        # For now, let's try to sync the whole folder
        
        print("\n   Syncing BraTS 2024 project files...")
        files = synapseutils.syncFromSynapse(
            syn,
            entity="syn53708249",  # BraTS 2024 main project
            path=str(download_dir),
            followLink=True
        )
        
        print(f"\n✅ Downloaded {len(files)} files")
        
        # Find the training data zip
        print("\n🔍 Looking for TrainingData.zip...")
        zip_files = list(download_dir.rglob("*TrainingData*.zip"))
        
        if not zip_files:
            print("❌ TrainingData.zip not found!")
            print("\nDownloaded files:")
            for f in download_dir.rglob("*.zip"):
                print(f"   {f.relative_to(download_dir)}")
            
            print("\n💡 You may need to:")
            print("   1. Accept BraTS 2024 terms at: https://www.synapse.org/#!Synapse:syn53708249")
            print("   2. Download manually from the web interface")
            return False
        
        zip_file = zip_files[0]
        print(f"✅ Found: {zip_file.name}")
        print(f"   Size: {zip_file.stat().st_size / (1024**3):.2f} GB")
        
        # Extract selectively
        print(f"\n📂 Extracting first {num_cases} cases...")
        extract_selective(zip_file, extract_dir, num_cases)
        
        # Cleanup
        print(f"\n🧹 Cleaning up...")
        if input("Delete zip file to save space? (y/n): ").lower() == 'y':
            zip_file.unlink()
            print("   ✓ Deleted zip file")
        
        print(f"\n✅ Complete! Data ready at: {extract_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def extract_selective(zip_path, extract_dir, num_cases=10):
    """Extract only first N cases"""
    
    print(f"   Opening {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        all_files = zip_ref.namelist()
        
        # Find unique case folders
        case_folders = set()
        for file in all_files:
            parts = Path(file).parts
            if len(parts) > 0 and 'BraTS' in parts[0]:
                case_folders.add(parts[0])
        
        # Sort and limit
        case_folders = sorted(list(case_folders))[:num_cases]
        
        print(f"   Extracting {len(case_folders)} cases:")
        for case in case_folders:
            print(f"      • {case}")
        
        # Extract
        extracted = 0
        for file in all_files:
            if any(file.startswith(case) for case in case_folders):
                zip_ref.extract(file, extract_dir)
                extracted += 1
        
        print(f"   ✓ Extracted {extracted} files")
        
        total_size = sum(f.stat().st_size for f in extract_dir.rglob("*") if f.is_file())
        print(f"   Total size: {total_size / (1024**2):.1f} MB")


def alternative_manual_download():
    """Guide for manual download if API fails"""
    
    print("\n" + "=" * 80)
    print("ALTERNATIVE: MANUAL DOWNLOAD")
    print("=" * 80)
    
    print("\nIf the API download doesn't work:")
    
    print("\n1. Go to Synapse project:")
    print("   https://www.synapse.org/#!Synapse:syn53708249/files/")
    
    print("\n2. Navigate to: Files > Data > BraTS-GLI")
    
    print("\n3. Click on: BraTS2024-BraTS-GLI-TrainingData.zip")
    
    print("\n4. Click the download button")
    
    print("\n5. Save to:")
    project_dir = Path(__file__).parent.parent.absolute()
    download_path = project_dir / "downloads" / "BraTS2024-BraTS-GLI-TrainingData.zip"
    print(f"   {download_path}")
    
    print("\n6. After download, run:")
    print("   python scripts/extract_brats2024.py")
    
    print("\nThis will extract only the first 10 cases automatically.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-cases', '-n', type=int, default=10)
    parser.add_argument('--manual', action='store_true', 
                        help='Show manual download instructions')
    
    args = parser.parse_args()
    
    if args.manual:
        alternative_manual_download()
    else:
        success = download_with_token(args.num_cases)
        
        if not success:
            print("\n" + "=" * 80)
            alternative_manual_download()
