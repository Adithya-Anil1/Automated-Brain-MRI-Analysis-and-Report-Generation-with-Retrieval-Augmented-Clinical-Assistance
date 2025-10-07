"""
Download and setup BraTS 2021 KAIST winning model
GitHub: https://github.com/rixez/Brats21_KAIST_MRI_Lab
"""
import os
import sys
import subprocess
from pathlib import Path
import urllib.request
import zipfile
import shutil


def setup_environment():
    """Setup nnU-Net environment variables"""
    project_dir = Path(__file__).parent.absolute()
    
    nnunet_raw = project_dir / "nnUNet_raw"
    nnunet_preprocessed = project_dir / "nnUNet_preprocessed"
    nnunet_results = project_dir / "nnUNet_results"
    
    os.environ['nnUNet_raw'] = str(nnunet_raw)
    os.environ['nnUNet_preprocessed'] = str(nnunet_preprocessed)
    os.environ['nnUNet_results'] = str(nnunet_results)
    
    print("[OK] Environment variables set")
    return project_dir


def download_brats2021_model():
    """
    Download BraTS 2021 KAIST winning model
    """
    print("\n" + "=" * 70)
    print("DOWNLOADING BraTS 2021 KAIST WINNING MODEL")
    print("=" * 70)
    
    project_dir = setup_environment()
    
    # Model URLs from the GitHub repository
    # The pretrained weights are hosted on Google Drive or GitHub releases
    model_url = "https://github.com/rixez/Brats21_KAIST_MRI_Lab/archive/refs/heads/main.zip"
    
    download_dir = project_dir / "downloads"
    download_dir.mkdir(exist_ok=True)
    
    zip_path = download_dir / "brats2021_model.zip"
    
    print("\nStep 1: Downloading model from GitHub...")
    print(f"URL: {model_url}")
    
    try:
        print("Downloading... (this may take a few minutes)")
        urllib.request.urlretrieve(model_url, zip_path)
        print("[OK] Download complete!")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("\nMANUAL DOWNLOAD REQUIRED:")
        print("1. Visit: https://github.com/rixez/Brats21_KAIST_MRI_Lab")
        print("2. Click 'Code' -> 'Download ZIP'")
        print("3. OR clone: git clone https://github.com/rixez/Brats21_KAIST_MRI_Lab.git")
        return None
    
    print("\nStep 2: Extracting files...")
    extract_dir = download_dir / "brats2021_extracted"
    extract_dir.mkdir(exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print("[OK] Extraction complete!")
    
    # Find the pretrained weights
    print("\nStep 3: Setting up model weights...")
    
    # The model weights should be in the extracted folder
    # We need to organize them in nnU-Net format
    model_results_dir = project_dir / "nnUNet_results" / "Dataset001_BraTS2021"
    model_results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[OK] Model directory created: {model_results_dir}")
    
    print("\n" + "=" * 70)
    print("SETUP INSTRUCTIONS")
    print("=" * 70)
    print("\nThe repository has been downloaded to:")
    print(f"  {extract_dir}")
    print("\nPlease follow these steps to complete setup:")
    print("\n1. Look for pretrained weights in the downloaded folder")
    print("   (usually .pth or .model files)")
    print("\n2. Check the repository README for specific instructions")
    print("\n3. The model weights should be organized as:")
    print(f"   {model_results_dir}/")
    print("     └── nnUNetTrainer__nnUNetPlans__3d_fullres/")
    print("         ├── fold_0/")
    print("         │   └── checkpoint_final.pth")
    print("         └── plans.json")
    
    return extract_dir


def check_git_installed():
    """Check if git is installed"""
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def clone_with_git():
    """Clone the repository using git"""
    print("\n" + "=" * 70)
    print("CLONING BraTS 2021 KAIST MODEL REPOSITORY")
    print("=" * 70)
    
    project_dir = setup_environment()
    
    repo_url = "https://github.com/rixez/Brats21_KAIST_MRI_Lab.git"
    clone_dir = project_dir / "Brats21_KAIST_MRI_Lab"
    
    if clone_dir.exists():
        print(f"[OK] Repository already exists at: {clone_dir}")
        return clone_dir
    
    print(f"\nCloning from: {repo_url}")
    print("This may take a few minutes...")
    
    try:
        subprocess.run(
            ["git", "clone", repo_url, str(clone_dir)],
            check=True,
            capture_output=False
        )
        print("[OK] Repository cloned successfully!")
        return clone_dir
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Git clone failed: {e}")
        return None


if __name__ == "__main__":
    print("=" * 70)
    print("BraTS 2021 KAIST WINNING MODEL SETUP")
    print("=" * 70)
    print("\nThis will download the BraTS 2021 winning submission model")
    print("GitHub: https://github.com/rixez/Brats21_KAIST_MRI_Lab")
    
    # Check if git is available
    if check_git_installed():
        print("\n[OK] Git is installed - using git clone (recommended)")
        repo_dir = clone_with_git()
    else:
        print("\n[INFO] Git not found - using ZIP download")
        repo_dir = download_brats2021_model()
    
    if repo_dir and repo_dir.exists():
        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print("\n1. Navigate to the downloaded repository:")
        print(f"   cd {repo_dir}")
        print("\n2. Read the README.md for specific setup instructions")
        print("\n3. Copy the pretrained model weights to nnUNet_results/")
        print("\n4. Run inference with: python inference_nnunet.py")
        
        # Try to display README if it exists
        readme_path = repo_dir / "README.md"
        if readme_path.exists():
            print("\n" + "=" * 70)
            print("REPOSITORY README (First 50 lines)")
            print("=" * 70)
            with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[:50]
                print(''.join(lines))
    else:
        print("\n[ERROR] Setup failed. Please download manually from:")
        print("https://github.com/rixez/Brats21_KAIST_MRI_Lab")
