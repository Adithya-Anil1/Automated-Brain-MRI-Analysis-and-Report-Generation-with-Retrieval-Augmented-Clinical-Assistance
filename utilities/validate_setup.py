"""
Quick validation script to check if nnU-Net setup is correct
"""
import sys
from pathlib import Path


def check_directories():
    """Check if required directories exist"""
    print("Checking directories...")
    project_dir = Path(__file__).parent
    
    required_dirs = [
        project_dir / "nnUNet_raw" / "Dataset001_BraTS2024",
        project_dir / "nnUNet_preprocessed",
        project_dir / "nnUNet_results"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"  ✓ {dir_path.name}")
        else:
            print(f"  ✗ {dir_path.name} - NOT FOUND")
            all_exist = False
    
    return all_exist


def check_dataset_json():
    """Check if dataset.json exists and is valid"""
    print("\nChecking dataset.json...")
    project_dir = Path(__file__).parent
    json_path = project_dir / "nnUNet_raw" / "Dataset001_BraTS2024" / "dataset.json"
    
    if not json_path.exists():
        print("  ✗ dataset.json not found")
        return False
    
    try:
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        required_keys = ['channel_names', 'labels', 'file_ending']
        missing_keys = [k for k in required_keys if k not in data]
        
        if missing_keys:
            print(f"  ✗ Missing keys: {missing_keys}")
            return False
        
        print("  ✓ dataset.json is valid")
        print(f"    Modalities: {list(data['channel_names'].values())}")
        print(f"    Labels: {list(data['labels'].keys())}")
        print(f"    Training cases: {data.get('numTraining', 0)}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error reading dataset.json: {e}")
        return False


def check_python_packages():
    """Check if required Python packages are installed"""
    print("\nChecking Python packages...")
    
    packages = [
        ('torch', 'PyTorch'),
        ('nnunetv2', 'nnU-Net v2'),
        ('SimpleITK', 'SimpleITK'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm')
    ]
    
    all_installed = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            all_installed = False
    
    return all_installed


def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA version: {torch.version.cuda}")
            print(f"    Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("  ✗ CUDA not available")
            print("    Training will be very slow on CPU")
            return False
    except ImportError:
        print("  ✗ PyTorch not installed")
        return False


def check_data():
    """Check if training data exists"""
    print("\nChecking training data...")
    project_dir = Path(__file__).parent
    
    images_dir = project_dir / "nnUNet_raw" / "Dataset001_BraTS2024" / "imagesTr"
    labels_dir = project_dir / "nnUNet_raw" / "Dataset001_BraTS2024" / "labelsTr"
    
    if not images_dir.exists() or not labels_dir.exists():
        print("  ⚠ Data directories not found")
        return False
    
    image_files = list(images_dir.glob("*.nii.gz"))
    label_files = list(labels_dir.glob("*.nii.gz"))
    
    if len(image_files) == 0:
        print("  ⚠ No training images found")
        print("    Run convert_brats_data.py to prepare your data")
        return False
    
    expected_images = len(label_files) * 4  # 4 modalities per case
    
    print(f"  Image files: {len(image_files)}")
    print(f"  Label files: {len(label_files)}")
    print(f"  Expected: {expected_images} images for {len(label_files)} cases")
    
    if len(image_files) == expected_images:
        print("  ✓ Data looks correct")
        return True
    else:
        print("  ⚠ Number of images doesn't match expected count")
        return False


def main():
    print("=" * 60)
    print("nnU-Net Setup Validation")
    print("=" * 60)
    
    results = {
        'Directories': check_directories(),
        'Dataset JSON': check_dataset_json(),
        'Python Packages': check_python_packages(),
        'CUDA': check_cuda(),
        'Training Data': check_data()
    }
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All checks passed! You're ready to train.")
        print("\nNext steps:")
        print("1. If data not prepared: python convert_brats_data.py --brats_path <path>")
        print("2. Start training: python train_nnunet.py --mode all --dataset_id 1 --config 3d_fullres --fold 0")
    else:
        print("\n⚠ Some checks failed. Please address the issues above.")
        
        if not results['Python Packages']:
            print("\nTo install packages: pip install -r requirements.txt")
        
        if not results['Directories'] or not results['Dataset JSON']:
            print("To setup directories: python setup_nnunet.py")
        
        if not results['Training Data']:
            print("To prepare data: python convert_brats_data.py --brats_path <path>")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
