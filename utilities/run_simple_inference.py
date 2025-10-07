"""
Simplified BraTS 2021 Inference using nnUNet v2
This avoids the multiprocessing issues with nnUNet v1 on Windows/Python 3.12
"""
import os
import sys
from pathlib import Path
import shutil
import torch

# Setup environment
project_dir = Path(__file__).parent.absolute()
os.environ['nnUNet_raw'] = str(project_dir / "nnUNet_raw")
os.environ['nnUNet_preprocessed'] = str(project_dir / "nnUNet_preprocessed")
os.environ['nnUNet_results'] = str(project_dir / "nnUNet_results")

print("=" * 70)
print("BraTS 2021 SIMPLIFIED INFERENCE (nnUNet v2 API)")
print("=" * 70)
print(f"Using GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

def prepare_input(input_folder, temp_folder):
    """Convert BraTS naming to nnU-Net naming"""
    temp_folder = Path(temp_folder)
    temp_folder.mkdir(parents=True, exist_ok=True)
    
    input_folder = Path(input_folder)
    files = list(input_folder.glob("*.nii.gz"))
    
    # Find patient ID
    t1_file = [f for f in files if '_t1.nii.gz' in f.name and '_t1ce' not in f.name][0]
    patient_id = t1_file.name.replace('_t1.nii.gz', '')
    
    print(f"Patient ID: {patient_id}")
    print(f"Preparing input files...")
    
    # Mapping: BraTS modality -> nnU-Net channel
    modality_map = {
        '_t1.nii.gz': '_0000.nii.gz',
        '_t1ce.nii.gz': '_0001.nii.gz',
        '_t2.nii.gz': '_0002.nii.gz',
        '_flair.nii.gz': '_0003.nii.gz'
    }
    
    for brats_suffix, nnunet_suffix in modality_map.items():
        src = input_folder / f"{patient_id}{brats_suffix}"
        dst = temp_folder / f"{patient_id}{nnunet_suffix}"
        if src.exists():
            shutil.copy(src, dst)
            print(f"  ✓ {src.name}")
        else:
            print(f"  ✗ Missing: {src.name}")
            sys.exit(1)
    
    return patient_id

def run_single_fold_inference(model_folder, input_folder, output_folder, fold=0):
    """Run inference using a single fold to avoid multiprocessing"""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    
    print(f"\nLoading model from fold {fold}...")
    
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    
    # Initialize from trained model
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(fold,),
        checkpoint_name='model_final_checkpoint'
    )
    
    # Run prediction
    predictor.predict_from_files(
        [[str(f) for f in sorted(Path(input_folder).glob(f"*_000{i}.nii.gz")) for i in range(4)][0]],
        [str(output_folder / f"prediction_fold{fold}.nii.gz")],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1
    )
    
    print(f"  ✓ Fold {fold} complete")
    return output_folder / f"prediction_fold{fold}.nii.gz"

def ensemble_folds(fold_predictions, output_file):
    """Ensemble predictions from multiple folds"""
    import nibabel as nib
    import numpy as np
    
    print("\nEnsembling fold predictions...")
    
    # Load all predictions
    predictions = []
    for pred_file in fold_predictions:
        nii = nib.load(pred_file)
        predictions.append(nii.get_fdata().astype(np.uint8))
    
    # Simple majority voting
    stacked = np.stack(predictions, axis=0)
    ensemble = np.zeros_like(predictions[0])
    
    for i in range(stacked.shape[1]):
        for j in range(stacked.shape[2]):
            for k in range(stacked.shape[3]):
                values = stacked[:, i, j, k]
                ensemble[i, j, k] = np.bincount(values).argmax()
    
    # Save ensemble result
    nii = nib.load(fold_predictions[0])
    ensemble_nii = nib.Nifti1Image(ensemble, nii.affine, nii.header)
    nib.save(ensemble_nii, output_file)
    
    print(f"  ✓ Ensemble saved: {output_file.name}")
    return output_file

def calculate_volumes(seg_file):
    """Calculate tumor volumes"""
    import nibabel as nib
    import numpy as np
    
    print("\n" + "=" * 70)
    print("TUMOR VOLUME ANALYSIS")
    print("=" * 70)
    
    seg_nii = nib.load(seg_file)
    seg_data = seg_nii.get_fdata()
    
    voxel_dims = seg_nii.header.get_zooms()
    voxel_volume_cm3 = np.prod(voxel_dims) / 1000
    
    ncr_volume = np.sum(seg_data == 1) * voxel_volume_cm3
    ed_volume = np.sum(seg_data == 2) * voxel_volume_cm3
    et_volume = np.sum(seg_data == 4) * voxel_volume_cm3
    
    tc_volume = ncr_volume + et_volume
    wt_volume = ncr_volume + ed_volume + et_volume
    
    print(f"\nSegmentation: {seg_file.name}")
    print("-" * 70)
    print(f"  Necrotic Core (NCR):         {ncr_volume:>10.2f} cm³")
    print(f"  Peritumoral Edema (ED):      {ed_volume:>10.2f} cm³")
    print(f"  Enhancing Tumor (ET):        {et_volume:>10.2f} cm³")
    print("-" * 70)
    print(f"  Tumor Core (TC):             {tc_volume:>10.2f} cm³")
    print(f"  Whole Tumor (WT):            {wt_volume:>10.2f} cm³")
    print("=" * 70)

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--folds', type=int, default=1, help='Number of folds to use (1-5)')
    args = parser.parse_args()
    
    # Prepare input
    temp_input = project_dir / "temp_nnunet_input"
    patient_id = prepare_input(args.input, temp_input)
    
    # Output folder
    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Model folder (using nnUNet v2 compatible path)
    # Note: We'll use just one model for simplicity
    model_folder = project_dir / "nnUNet_results" / "3d_fullres" / "Task500_BraTS2021" / "nnUNetTrainerV2BraTSRegions_DA4_BN_BD__nnUNetPlansv2.1"
    
    if not model_folder.exists():
        print(f"[ERROR] Model not found: {model_folder}")
        sys.exit(1)
    
    print(f"\nModel: {model_folder.name}")
    print(f"Using {args.folds} fold(s) for inference")
    
    # Run inference for each fold
    fold_predictions = []
    for fold in range(min(args.folds, 5)):
        print(f"\n[Fold {fold}/{ min(args.folds, 5)-1}]")
        pred_file = run_single_fold_inference(model_folder, temp_input, output_folder, fold=fold)
        fold_predictions.append(pred_file)
    
    # Ensemble if multiple folds
    if len(fold_predictions) > 1:
        final_seg = ensemble_folds(fold_predictions, output_folder / f"{patient_id}.nii.gz")
    else:
        final_seg = fold_predictions[0]
        final_seg.rename(output_folder / f"{patient_id}.nii.gz")
        final_seg = output_folder / f"{patient_id}.nii.gz"
    
    # Calculate volumes
    calculate_volumes(final_seg)
    
    # Cleanup
    print("\nCleaning up temporary files...")
    shutil.rmtree(temp_input, ignore_errors=True)
    for pred in fold_predictions:
        if pred.exists() and pred != final_seg:
            pred.unlink()
    
    print(f"\n✓ Segmentation complete!")
    print(f"Results saved to: {output_folder}")

if __name__ == "__main__":
    main()
