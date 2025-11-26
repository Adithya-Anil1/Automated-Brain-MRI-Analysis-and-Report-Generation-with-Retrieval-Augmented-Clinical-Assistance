# AI-Powered Brain MRI Assistant

An automated brain tumor segmentation system that helps doctors analyze MRI scans faster and more accurately.

## What It Does

This system takes 4 different types of brain MRI scans (T1, T1ce, T2, FLAIR) and automatically identifies three types of tumor regions:
- **Enhancing Tumor (ET)**: The most active part of the tumor
- **Edema (ED)**: Swelling around the tumor
- **Necrotic Core (NCR)**: Dead tissue in the tumor center

## How It Works

1. **Model**: Uses the BraTS 2021 Challenge winning model (ranked #1 globally)
2. **Technology**: nnU-Net deep learning framework with custom modifications
3. **Processing**: Two ensemble models with 5-fold cross-validation each (10 predictions combined)
4. **Speed**: Takes 20-30 minutes per patient on a regular computer (no GPU needed)

## Performance

Tested on real brain tumor cases:
- **92.95% accuracy** on enhancing tumor detection ⭐
- **83.24% accuracy** on edema detection
- **67.34% accuracy** on necrotic core detection

These results match or exceed state-of-the-art medical imaging standards.

## How to Use

### Step 1: Set Up Environment
```powershell
$env:nnUNet_raw_data_base="C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant\nnUNet_raw"
$env:nnUNet_preprocessed="C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant\nnUNet_preprocessed"
$env:RESULTS_FOLDER="C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant\nnUNet_results"
```

### Step 2: Run Segmentation
```bash
python run_brats2021_inference_singlethread.py --input <input_folder> --output <output_folder>
```

### Step 3: Convert to Medical Standard Format
```bash
python convert_labels_to_brats.py "output_folder\result.nii.gz" "output_folder\result_final.nii.gz"
```

### Step 4: Check Results
```bash
python check_labels.py "output_folder\result_final.nii.gz"
```

## What You Get

- **3D Segmentation File**: Color-coded tumor regions viewable in medical imaging software
- **Volume Measurements**: Automatic calculation of tumor sizes
- **Accuracy Metrics**: Performance scores if you have ground truth data

## Important Notes

- Close other applications while running for best performance
- Don't interrupt the process once it starts
- Each scan takes about 30 minutes to process
- Results are saved in standard medical format (NIfTI files)

## Technical Details

See `PROJECT_DOCUMENTATION.md` for:
- Complete model architecture details
- Training methodology
- All bug fixes and solutions
- Performance benchmarks
- Troubleshooting guide

## Requirements

- Windows 10/11 (tested)
- Python 3.10
- 16GB RAM minimum (32GB recommended)
- 10GB free disk space



