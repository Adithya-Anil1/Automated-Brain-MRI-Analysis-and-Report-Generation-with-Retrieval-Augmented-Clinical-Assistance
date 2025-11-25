"""
Run only Model 2 (largeUnet_Groupnorm) with memory optimizations
This runs ONE fold at a time to avoid memory issues
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch


# Add the Brats21_KAIST_MRI_Lab to Python path
script_dir = Path(__file__).parent
kaist_dir = script_dir / "Brats21_KAIST_MRI_Lab"
sys.path.insert(0, str(kaist_dir))

from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax

# Set results folder
results_folder = Path(r"C:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant\nnUNet_results")
os.environ['RESULTS_FOLDER'] = str(results_folder)

print("="*70)
print("MODEL 2 INFERENCE (Memory Optimized)")
print("="*70)

# Model 2 path
model2_path = results_folder / "3d_fullres" / "Task500_BraTS2021" / "nnUNetTrainerV2BraTSRegions_DA4_BN_BD_largeUnet_Groupnorm__nnUNetPlansv2.1"

print(f"Model path: {model2_path}")

# Input/output
input_folder = Path(r"data\sample_data\BraTS2021_sample")
output_folder = Path(r"results\model2_only")
output_folder.mkdir(parents=True, exist_ok=True)

# Find input case
case_name = "BraTS2021_00495"
input_files = [
    input_folder / f"{case_name}_t1.nii.gz",
    input_folder / f"{case_name}_t1ce.nii.gz", 
    input_folder / f"{case_name}_t2.nii.gz",
    input_folder / f"{case_name}_flair.nii.gz"
]

print(f"\nProcessing case: {case_name}")
print(f"Input files:")
for f in input_files:
    print(f"  - {f.name}: {'✓' if f.exists() else '✗ MISSING'}")

# Load trainer (lighter memory footprint - load one fold at a time)
print("\nLoading model...")
folds_to_use = (0, 1, 2, 3, 4)  # Try all 5 folds, but process ONE at a time

# Get checkpoint files
checkpoint_files = []
for fold in folds_to_use:
    checkpoint = model2_path / f"fold_{fold}" / "model_final_checkpoint.model"
    if checkpoint.exists():
        checkpoint_files.append(str(checkpoint))
        print(f"  Found fold {fold}: {checkpoint}")
    else:
        print(f"  ✗ Missing fold {fold}")

if not checkpoint_files:
    print("❌ No checkpoint files found!")
    sys.exit(1)

print(f"\nUsing {len(checkpoint_files)} folds")

# Load trainer from first fold - this loads ALL folds
print("\nInitializing trainer...")
trainer, params = load_model_and_checkpoint_files(
    str(model2_path),
    folds=folds_to_use,  # Load all folds
    mixed_precision=False,  # Disable mixed precision to save memory
    checkpoint_name='model_final_checkpoint'
)

print(f"Trainer loaded with {len(params)} checkpoints")

# Preprocess patient data
print(f"\nPreprocessing {case_name}...")
list_of_files = [[str(f) for f in input_files]]
d, _, dct = trainer.preprocess_patient(list_of_files[0])
print(f"Data shape: {d.shape}")

# Process each fold separately to save memory
all_softmax = []

for fold_idx, param in enumerate(params):
    print(f"\n{'='*70}")
    print(f"Processing Fold {folds_to_use[fold_idx]} ({fold_idx+1}/{len(params)})")
    print(f"{'='*70}")
    
    # Load this fold's checkpoint
    trainer.load_checkpoint_ram(param, False)
    
    # Predict with minimal memory usage
    print("Predicting...")
    try:
        softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(
            d,
            do_mirroring=True,  # Keep TTA for accuracy
            mirror_axes=trainer.data_aug_params['mirror_axes'],
            use_sliding_window=True,
            step_size=0.5,
            use_gaussian=True,
            all_in_gpu=False,  # Force CPU to avoid GPU memory issues
            mixed_precision=False  # Disable for stability
        )[1]
        
        all_softmax.append(softmax)
        print(f"✓ Fold {folds_to_use[fold_idx]} completed")
        
        # Clear some memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"✗ Fold {folds_to_use[fold_idx]} failed: {e}")
        print("Continuing with remaining folds...")
        continue

if not all_softmax:
    print("\n❌ All folds failed! Cannot generate prediction.")
    sys.exit(1)

# Ensemble predictions
print(f"\n{'='*70}")
print(f"Ensembling {len(all_softmax)} folds")
print(f"{'='*70}")
softmax_mean = np.mean(all_softmax, axis=0)

# Check labels in prediction
seg = np.argmax(softmax_mean, axis=0)
unique_labels = np.unique(seg)
print(f"\nPredicted labels: {unique_labels}")
print("Expected: [0, 1, 2, 3] (where 3 = Enhancing Tumor)")

# Save segmentation
output_file = output_folder / f"{case_name}.nii.gz"
print(f"\nSaving to: {output_file}")

# Get export parameters
if 'segmentation_export_params' in trainer.plans.keys():
    force_separate_z = trainer.plans['segmentation_export_params']['force_separate_z']
    interpolation_order = trainer.plans['segmentation_export_params']['interpolation_order']
    interpolation_order_z = trainer.plans['segmentation_export_params']['interpolation_order_z']
else:
    force_separate_z = None
    interpolation_order = 1
    interpolation_order_z = 0

save_segmentation_nifti_from_softmax(
    softmax_mean,
    str(output_file),
    dct,
    interpolation_order,
    None, None, None, None, None,
    force_separate_z=force_separate_z,
    interpolation_order_z=interpolation_order_z
)

print(f"\n✅ Segmentation saved to: {output_file}")
print("\nNext step: Check labels and convert to BraTS format (0,1,2,4)")
print(f"  python check_labels.py \"{output_file}\"")
print(f"  python convert_labels_to_brats.py \"{output_file}\" \"{output_folder / f'{case_name}_brats.nii.gz'}\"")
