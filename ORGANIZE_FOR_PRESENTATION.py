"""
Organize workspace for presentation - Group main files into logical folders
Creates a clean structure with a 'main' folder containing all important presentation files
"""

import os
import shutil
from pathlib import Path


def organize_for_presentation():
    """Organize workspace with presentation-friendly structure"""
    
    project_dir = Path(__file__).parent.absolute()
    
    print("=" * 80)
    print("ORGANIZING WORKSPACE FOR PRESENTATION")
    print("=" * 80)
    
    # Create new folder structure
    folders_to_create = {
        'main_files': 'Main inference scripts and important code',
        'model_architecture': 'U-Net model architecture files',
        'documentation': 'All documentation and guides',
        'utilities': 'Helper scripts and tools',
        'archived': 'Old/deprecated files'
    }
    
    print("\n📁 Creating folder structure...")
    for folder, desc in folders_to_create.items():
        folder_path = project_dir / folder
        folder_path.mkdir(exist_ok=True)
        print(f"   ✓ {folder}/ - {desc}")
    
    # ===== MAIN FILES (For Presentation) =====
    print("\n⭐ Moving MAIN FILES (for presentation)...")
    main_files_mapping = {
        # Main inference script (single-threaded working version)
        'run_brats2021_inference_singlethread.py': 'main_files/run_inference.py',
        
        # Visualization script
        'scripts/visualize_segmentation.py': 'main_files/visualize_segmentation.py',
        
        # KAIST original inference
        'Brats21_KAIST_MRI_Lab/inference_v2.py': 'main_files/kaist_original_inference.py',
    }
    
    for src, dest in main_files_mapping.items():
        src_path = project_dir / src
        dest_path = project_dir / dest
        if src_path.exists() and not dest_path.exists():
            shutil.copy2(src_path, dest_path)  # Use copy2 to preserve metadata
            print(f"   ✓ Copied: {src} → {dest}")
    
    # ===== MODEL ARCHITECTURE =====
    print("\n🧠 Copying MODEL ARCHITECTURE files...")
    arch_files = [
        'Brats21_KAIST_MRI_Lab/nnunet/network_architecture/generic_UNet.py',
        'Brats21_KAIST_MRI_Lab/nnunet/network_architecture/generic_modular_UNet.py',
    ]
    
    for src in arch_files:
        src_path = project_dir / src
        if src_path.exists():
            dest_path = project_dir / 'model_architecture' / src_path.name
            if not dest_path.exists():
                shutil.copy2(src_path, dest_path)
                print(f"   ✓ Copied: {src_path.name}")
    
    # ===== DOCUMENTATION =====
    print("\n📄 Moving DOCUMENTATION files...")
    doc_files = [
        'README.md',
        'WHY_LOW_ACCURACY.md',
        'docs/ACCURACY_EVALUATION_GUIDE.md',
        'docs/MULTIPROCESSING_FIX.md',
        'docs/PYTHON_310_SETUP.md',
        'docs/instructions.md',
    ]
    
    for src in doc_files:
        src_path = project_dir / src
        if src_path.exists():
            dest_path = project_dir / 'documentation' / src_path.name
            if not dest_path.exists():
                shutil.copy2(src_path, dest_path)
                print(f"   ✓ Copied: {src_path.name}")
    
    # ===== UTILITIES (Helper Scripts) =====
    print("\n🔧 Moving UTILITY scripts...")
    utility_scripts = [
        'check_labels.py',
        'compare_segmentations.py',
        'evaluate_segmentation.py',
        'download_more_brats_data.py',
        'scripts/check_compatibility.py',
        'scripts/setup_brats2021_model.py',
        'scripts/setup_nnunet.py',
        'scripts/validate_setup.py',
        'scripts/run_simple_inference.py',
    ]
    
    for src in utility_scripts:
        src_path = project_dir / src
        if src_path.exists():
            dest_path = project_dir / 'utilities' / src_path.name
            if not dest_path.exists():
                shutil.copy2(src_path, dest_path)
                print(f"   ✓ Copied: {src_path.name}")
    
    # ===== ARCHIVED (Old versions) =====
    print("\n📦 Archiving OLD/deprecated files...")
    # Note: We DON'T move, just mark what should be archived
    archived_note = """
Files that should be moved to 'archived/' folder manually:
- run_brats2021_inference.py (old multi-threaded version that doesn't work)
- scripts/run_inference.py (duplicate/old version)
- Any other experimental scripts
"""
    
    with open(project_dir / 'archived' / 'README.txt', 'w') as f:
        f.write(archived_note)
    print(f"   ✓ Created archived/README.txt with notes")
    
    # ===== CREATE PRESENTATION GUIDE =====
    print("\n📋 Creating PRESENTATION_GUIDE.md...")
    guide_content = """# 📊 Presentation Guide - Main Files Overview

## 🎯 Files to Present (in `main_files/` folder)

### 1. **run_inference.py** (Main Execution Script)
- **What**: The primary script that runs brain tumor segmentation
- **Location**: `main_files/run_inference.py`
- **Original**: `run_brats2021_inference_singlethread.py`
- **Show**: How it loads models, processes MRI data, and generates predictions

### 2. **visualize_segmentation.py** (Visualization Generator)
- **What**: Creates visual overlays of tumor segmentation on MRI images
- **Location**: `main_files/visualize_segmentation.py`
- **Show**: How results are visualized for doctors (color-coded tumor regions)

### 3. **kaist_original_inference.py** (KAIST Team's Code)
- **What**: Original winning solution from BraTS 2021 competition
- **Location**: `main_files/kaist_original_inference.py`
- **Original**: `Brats21_KAIST_MRI_Lab/inference_v2.py`
- **Show**: Professional implementation from the winning team

---

## 🧠 Model Architecture (in `model_architecture/` folder)

### 4. **generic_UNet.py** (3D U-Net Neural Network)
- **What**: The actual deep learning model architecture
- **Location**: `model_architecture/generic_UNet.py`
- **Show**: The encoder-decoder structure, convolutional blocks

---

## 📦 Model Weights Location

### 5. **Trained Model Weights**
```
nnUNet_results/3d_fullres/Task500_BraTS2021/
├── nnUNetTrainerV2BraTSRegions_DA4_BN_BD__nnUNetPlansv2.1/
│   └── fold_0/model_final_checkpoint.model
└── nnUNetTrainerV2BraTSRegions_DA4_BN_BD_largeUnet_Groupnorm__nnUNetPlansv2.1/
    └── fold_0/model_final_checkpoint.model
```
- **What**: Pre-trained weights from BraTS 2021 winning team
- **Size**: ~500MB each variant
- **Show**: These are the actual trained parameters

---

## 📚 Documentation (in `documentation/` folder)

- **README.md** - Project overview
- **MULTIPROCESSING_FIX.md** - Why single-threaded version was needed
- **ACCURACY_EVALUATION_GUIDE.md** - How to evaluate results
- **WHY_LOW_ACCURACY.md** - Analysis of model performance

---

## 🎬 Presentation Flow

1. **Start with**: `README.md` - Explain what the project does
2. **Show**: `main_files/run_inference.py` - Main execution logic
3. **Demonstrate**: `main_files/visualize_segmentation.py` - How visualizations are created
4. **Technical deep-dive**: `model_architecture/generic_UNet.py` - The neural network
5. **Evidence of quality**: `main_files/kaist_original_inference.py` - Professional winning solution
6. **Results**: Show sample outputs in `visualizations/` folder

---

## 📂 Clean Folder Structure

```
AI-Powered Brain MRI Assistant/
├── main_files/               ⭐ START HERE - All important code
│   ├── run_inference.py
│   ├── visualize_segmentation.py
│   └── kaist_original_inference.py
├── model_architecture/       🧠 U-Net architecture
│   └── generic_UNet.py
├── documentation/            📚 All guides and docs
├── nnUNet_results/          💾 Trained model weights
├── visualizations/          🖼️  Output examples
├── data/                    📁 Sample MRI data
└── utilities/               🔧 Helper scripts
```

---

## ✅ Quick Checklist for Presentation

- [ ] Show `main_files/run_inference.py` - Main script
- [ ] Show `model_architecture/generic_UNet.py` - Model architecture  
- [ ] Show `main_files/visualize_segmentation.py` - Visualization code
- [ ] Show sample output from `visualizations/` folder
- [ ] Explain model weights in `nnUNet_results/`
- [ ] Reference `documentation/README.md` for project overview

---

**Note**: All files in `main_files/` are copies - originals are preserved in their original locations.
"""
    
    with open(project_dir / 'PRESENTATION_GUIDE.md', 'w') as f:
        f.write(guide_content)
    print(f"   ✓ Created PRESENTATION_GUIDE.md")
    
    # ===== SUMMARY =====
    print("\n" + "=" * 80)
    print("✅ ORGANIZATION COMPLETE!")
    print("=" * 80)
    print("\n📁 NEW FOLDER STRUCTURE:")
    print("   main_files/          ⭐ All important files for presentation")
    print("   model_architecture/  🧠 U-Net model code")
    print("   documentation/       📚 All guides and docs")
    print("   utilities/           🔧 Helper scripts")
    print("   archived/            📦 Old/deprecated files")
    print("\n📋 Next Steps:")
    print("   1. Check the 'main_files/' folder - these are your presentation files")
    print("   2. Read 'PRESENTATION_GUIDE.md' for a complete overview")
    print("   3. Manually move old files to 'archived/' if needed")
    print("\n⭐ TIP: Start your presentation by showing files in 'main_files/' folder!")
    print("=" * 80)


if __name__ == '__main__':
    try:
        organize_for_presentation()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
