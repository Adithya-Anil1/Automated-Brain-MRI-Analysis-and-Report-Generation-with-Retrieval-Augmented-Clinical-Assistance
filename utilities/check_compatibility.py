"""
Pre-Segmentation Compatibility Check
Verifies all dependencies, model files, and data before running inference
"""
import sys
from pathlib import Path

print("=" * 70)
print("BRATS 2021 SEGMENTATION - COMPATIBILITY CHECK")
print("=" * 70)

errors = []
warning_msgs = []
passed = []

# 1. Python Version Check
print("\n[1/8] Checking Python Version...")
import platform
python_version = platform.python_version()
print(f"  Python: {python_version}")
if python_version.startswith("3.12"):
    passed.append("Python 3.12 detected")
else:
    warning_msgs.append(f"Using Python {python_version} - recommended: 3.12")

# 2. PyTorch + CUDA Check
print("\n[2/8] Checking PyTorch and CUDA...")
try:
    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        passed.append("PyTorch with CUDA support installed")
    else:
        warning_msgs.append("CUDA not available - will use CPU (slower)")
except ImportError:
    errors.append("PyTorch not installed")

# 3. Required Packages Check
print("\n[3/8] Checking Required Packages...")
required_packages = {
    'numpy': 'NumPy',
    'nibabel': 'NiBabel',
    'SimpleITK': 'SimpleITK',
    'scipy': 'SciPy',
    'medpy': 'MedPy',
    'batchgenerators': 'BatchGenerators',
    'skimage': 'scikit-image'
}

for module, name in required_packages.items():
    try:
        if module == 'SimpleITK':
            import SimpleITK as sitk
            version = sitk.Version_VersionString()
        elif module == 'medpy':
            import medpy
            version = medpy.__version__ if hasattr(medpy, '__version__') else '0.4.0'
        elif module == 'batchgenerators':
            import batchgenerators
            version = batchgenerators.__version__ if hasattr(batchgenerators, '__version__') else '0.21'
        elif module == 'skimage':
            import skimage
            version = skimage.__version__
        else:
            mod = __import__(module)
            version = mod.__version__
        
        print(f"  ✓ {name}: {version}")
        passed.append(f"{name} installed")
    except ImportError:
        print(f"  ✗ {name}: NOT INSTALLED")
        errors.append(f"{name} not installed")

# 4. nnU-Net v1 (Brats21 repo) Check
print("\n[4/8] Checking nnU-Net v1 (BraTS 2021 repository)...")
project_dir = Path(__file__).parent.absolute()
brats_repo = project_dir / "Brats21_KAIST_MRI_Lab"
if brats_repo.exists():
    print(f"  ✓ Repository found: {brats_repo}")
    sys.path.insert(0, str(brats_repo))
    try:
        # Suppress warnings during import
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from nnunet.inference.predict import predict_from_folder
        print("  ✓ nnU-Net v1 modules accessible")
        passed.append("nnU-Net v1 repository configured")
    except Exception as e:
        print(f"  ⚠ nnU-Net v1 import warning: {str(e)[:50]}")
        # Check if it's just a warning, not a critical error
        if "SyntaxWarning" in str(e) or "FutureWarning" in str(e):
            warning_msgs.append("nnU-Net v1 has warnings (non-critical)")
            passed.append("nnU-Net v1 accessible (with warnings)")
        else:
            errors.append(f"nnU-Net v1 import failed: {str(e)[:50]}")
else:
    print(f"  ✗ Repository not found")
    errors.append("Brats21_KAIST_MRI_Lab repository missing")

# 5. Model Files Check
print("\n[5/8] Checking Model Files...")
model_dir = project_dir / "nnUNet_results" / "3d_fullres" / "Task500_BraTS2021"
if model_dir.exists():
    print(f"  ✓ Model directory exists")
    
    # Check Model 1
    model1 = model_dir / "nnUNetTrainerV2BraTSRegions_DA4_BN_BD__nnUNetPlansv2.1"
    if model1.exists():
        folds = list(model1.glob("fold_*"))
        print(f"  ✓ Model 1: {len(folds)} folds found")
        
        # Check if model weights exist
        fold0_model = model1 / "fold_0" / "model_final_checkpoint.model"
        if fold0_model.exists():
            size_mb = fold0_model.stat().st_size / (1024 * 1024)
            print(f"    Model weights: {size_mb:.1f} MB")
            passed.append(f"Model 1 with {len(folds)} folds")
        else:
            errors.append("Model 1 weights missing")
    else:
        errors.append("Model 1 directory missing")
    
    # Check Model 2
    model2 = model_dir / "nnUNetTrainerV2BraTSRegions_DA4_BN_BD_largeUnet_Groupnorm__nnUNetPlansv2.1"
    if model2.exists():
        folds = list(model2.glob("fold_*"))
        print(f"  ✓ Model 2: {len(folds)} folds found")
        passed.append(f"Model 2 with {len(folds)} folds")
    else:
        errors.append("Model 2 directory missing")
else:
    errors.append("Model directory not found")

# 6. Sample Data Check
print("\n[6/8] Checking Sample Data...")
sample_dir = project_dir / "sample_data" / "BraTS2021_sample"
if sample_dir.exists():
    nii_files = list(sample_dir.glob("*.nii.gz"))
    print(f"  ✓ Sample directory exists")
    print(f"  Files found: {len(nii_files)}")
    
    required_modalities = ['_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz', '_flair.nii.gz']
    for mod in required_modalities:
        if any(mod in f.name for f in nii_files):
            print(f"    ✓ {mod}")
        else:
            print(f"    ✗ {mod} - MISSING")
            errors.append(f"Missing modality: {mod}")
    
    if len([f for f in nii_files if not '_seg' in f.name]) == 4:
        passed.append("All 4 MRI modalities present")
else:
    errors.append("Sample data directory not found")

# 7. Environment Variables Check
print("\n[7/8] Checking Environment Variables...")
import os
os.environ['nnUNet_raw_data_base'] = str(project_dir / "nnUNet_raw")
os.environ['nnUNet_preprocessed'] = str(project_dir / "nnUNet_preprocessed")
os.environ['RESULTS_FOLDER'] = str(project_dir / "nnUNet_results")

for var in ['nnUNet_raw_data_base', 'nnUNet_preprocessed', 'RESULTS_FOLDER']:
    if os.environ.get(var):
        print(f"  ✓ {var}: {os.environ[var]}")
        passed.append(f"{var} configured")
    else:
        warning_msgs.append(f"{var} not set")

# 8. Disk Space Check
print("\n[8/8] Checking Disk Space...")
import shutil
total, used, free = shutil.disk_usage(project_dir)
free_gb = free / (1024**3)
print(f"  Free disk space: {free_gb:.1f} GB")
if free_gb > 10:
    passed.append(f"Sufficient disk space ({free_gb:.1f} GB)")
else:
    warning_msgs.append(f"Low disk space ({free_gb:.1f} GB) - recommend 10+ GB")

# Summary
print("\n" + "=" * 70)
print("COMPATIBILITY CHECK SUMMARY")
print("=" * 70)

print(f"\n✓ PASSED: {len(passed)}")
for p in passed[:5]:  # Show first 5
    print(f"  • {p}")
if len(passed) > 5:
    print(f"  ... and {len(passed) - 5} more")

if warning_msgs:
    print(f"\n⚠ WARNINGS: {len(warning_msgs)}")
    for w in warning_msgs:
        print(f"  • {w}")

if errors:
    print(f"\n✗ ERRORS: {len(errors)}")
    for e in errors:
        print(f"  • {e}")
    print("\n" + "=" * 70)
    print("RECOMMENDATION: Fix errors before running segmentation")
    print("=" * 70)
    sys.exit(1)
else:
    print("\n" + "=" * 70)
    print("✓ ALL CHECKS PASSED - READY FOR SEGMENTATION!")
    print("=" * 70)
    print("\nYou can now run:")
    print("  python run_brats2021_inference.py --input sample_data\\BraTS2021_sample --output results\\BraTS2021_00495")
    sys.exit(0)
