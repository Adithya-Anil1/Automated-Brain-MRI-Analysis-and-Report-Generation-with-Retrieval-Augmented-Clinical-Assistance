#Script used to run the 2 UNet models, comibines the results and generates segmentation masks.


"""
BraTS 2021 Brain Tumor Segmentation - Single-threaded Version
Uses the BraTS 2021 KAIST MRI Lab winning model for inference.
Modified to avoid Windows multiprocessing issues by running in single-threaded mode.
"""

import os
import sys
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import SimpleITK as sitk

# Add the Brats21_KAIST_MRI_Lab to Python path
script_dir = Path(__file__).parent
kaist_dir = script_dir / "Brats21_KAIST_MRI_Lab"
sys.path.insert(0, str(kaist_dir))

from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from batchgenerators.utilities.file_and_folder_operations import *


def prepare_input(sample_dir, output_dir):
    """
    Prepare BraTS sample data for nnU-Net inference.
    BraTS format: casename_t1.nii.gz, casename_t1ce.nii.gz, casename_t2.nii.gz, casename_flair.nii.gz
    nnU-Net format: casename_0000.nii.gz (T1), _0001.nii.gz (T1CE), _0002.nii.gz (T2), _0003.nii.gz (FLAIR)
    """
    sample_dir = Path(sample_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all cases in the sample directory
    cases = set()
    for file in sample_dir.glob("*.nii.gz"):
        # Extract case name (everything before the modality)
        parts = file.stem.replace('.nii', '').split('_')
        if parts[-1] in ['t1', 't1ce', 't2', 'flair', 'seg']:
            case_name = '_'.join(parts[:-1])
            cases.add(case_name)
    
    print(f"Found {len(cases)} cases: {cases}")
    
    # Map modalities
    modality_map = {
        't1': '0000',
        't1ce': '0001', 
        't2': '0002',
        'flair': '0003'
    }
    
    prepared_cases = []
    for case in cases:
        case_files = []
        all_found = True
        
        for mod, idx in modality_map.items():
            src = sample_dir / f"{case}_{mod}.nii.gz"
            dst = output_dir / f"{case}_{idx}.nii.gz"
            
            if src.exists():
                # Copy/symlink the file
                if not dst.exists():
                    import shutil
                    shutil.copy(src, dst)
                case_files.append(str(dst))
            else:
                print(f"[WARNING] Missing {mod} for {case}")
                all_found = False
                break
        
        if all_found:
            prepared_cases.append((case, case_files))
    
    return prepared_cases


def predict_case_single_threaded(trainer, list_of_files, output_file, params, do_tta=True, 
                                 mixed_precision=True, step_size=0.5, all_in_gpu=False):
    """
    Run prediction for a single case without multiprocessing.
    """
    print(f"Preprocessing {output_file}")
    
    # Preprocess the patient data
    d, _, dct = trainer.preprocess_patient(list_of_files)
    
    print(f"Data shape after preprocessing: {d.shape}")
    print(f"Predicting {output_file}")
    
    # Load checkpoint and predict
    trainer.load_checkpoint_ram(params[0], False)
    
    softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(
        d, 
        do_mirroring=do_tta, 
        mirror_axes=trainer.data_aug_params['mirror_axes'], 
        use_sliding_window=True,
        step_size=step_size, 
        use_gaussian=True, 
        all_in_gpu=all_in_gpu,
        mixed_precision=mixed_precision
    )[1]
    
    # Handle multiple folds ensemble
    all_softmax = [softmax]
    
    # Load other folds
    for p in params[1:]:
        trainer.load_checkpoint_ram(p, False)
        softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(
            d,
            do_mirroring=do_tta,
            mirror_axes=trainer.data_aug_params['mirror_axes'],
            use_sliding_window=True,
            step_size=step_size,
            use_gaussian=True,
            all_in_gpu=all_in_gpu,
            mixed_precision=mixed_precision
        )[1]
        all_softmax.append(softmax)
    
    # Average predictions from all folds
    print(f"Ensembling {len(all_softmax)} folds")
    softmax_mean = np.mean(all_softmax, axis=0)
    
    # Get segmentation export parameters
    if 'segmentation_export_params' in trainer.plans.keys():
        force_separate_z = trainer.plans['segmentation_export_params']['force_separate_z']
        interpolation_order = trainer.plans['segmentation_export_params']['interpolation_order']
        interpolation_order_z = trainer.plans['segmentation_export_params']['interpolation_order_z']
    else:
        force_separate_z = None
        interpolation_order = 1
        interpolation_order_z = 0
    
    # Save segmentation
    print(f"Saving segmentation to {output_file}")
    save_segmentation_nifti_from_softmax(
        softmax_mean, 
        output_file,
        dct, 
        interpolation_order, 
        None,
        None, 
        None,
        None, 
        None,
        force_separate_z=force_separate_z, 
        interpolation_order_z=interpolation_order_z
    )
    
    return output_file


def run_model_single_threaded(model_folder, input_folder, output_folder, folds=(0, 1, 2, 3, 4)):
    """
    Run inference using a single model without multiprocessing.
    """
    model_folder = Path(model_folder)
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    
    if not model_folder.exists():
        print(f"[ERROR] Model not found: {model_folder}")
        sys.exit(1)
    
    print(f"Model path: {model_folder}")
    print(f"Loading model with folds: {folds}")
    
    # Load model and checkpoints
    torch.cuda.empty_cache()
    trainer, params = load_model_and_checkpoint_files(
        str(model_folder), 
        folds, 
        mixed_precision=True,
        checkpoint_name="model_final_checkpoint"
    )
    
    print(f"Loaded {len(params)} fold checkpoints")
    
    # Prepare input cases
    prepared_cases = prepare_input(input_folder, output_folder / "temp_input")
    
    if not prepared_cases:
        print("[ERROR] No valid cases found!")
        return
    
    # Process each case
    for case_name, case_files in prepared_cases:
        output_file = output_folder / f"{case_name}.nii.gz"
        output_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"Processing case: {case_name}")
        print(f"{'='*70}")
        
        predict_case_single_threaded(
            trainer=trainer,
            list_of_files=case_files,
            output_file=str(output_file),
            params=params,
            do_tta=True,
            mixed_precision=True,
            step_size=0.5,
            all_in_gpu=False
        )
        
        print(f"✓ Completed: {output_file}")


def calculate_volumes(seg_path):
    """Calculate tumor volumes from segmentation."""
    seg = nib.load(seg_path).get_fdata()
    
    # Get voxel dimensions (in mm)
    img = nib.load(seg_path)
    voxel_dims = img.header.get_zooms()
    voxel_volume_mm3 = np.prod(voxel_dims)
    voxel_volume_cm3 = voxel_volume_mm3 / 1000.0
    
    # BraTS labels: 0=background, 1=NCR, 2=ED, 4=ET
    # Regions: TC (Tumor Core) = NCR + ET, WT (Whole Tumor) = NCR + ED + ET
    ncr_voxels = np.sum(seg == 1)
    ed_voxels = np.sum(seg == 2)
    et_voxels = np.sum(seg == 4)
    tc_voxels = ncr_voxels + et_voxels
    wt_voxels = ncr_voxels + ed_voxels + et_voxels
    
    volumes = {
        'NCR': ncr_voxels * voxel_volume_cm3,
        'ED': ed_voxels * voxel_volume_cm3,
        'ET': et_voxels * voxel_volume_cm3,
        'TC': tc_voxels * voxel_volume_cm3,
        'WT': wt_voxels * voxel_volume_cm3
    }
    
    return volumes


def main():
    parser = argparse.ArgumentParser(description='BraTS 2021 Brain Tumor Segmentation (Single-threaded)')
    parser.add_argument('--input', type=str, required=True, help='Input directory with BraTS sample data')
    parser.add_argument('--output', type=str, required=True, help='Output directory for segmentation results')
    args = parser.parse_args()
    
    # Set up paths
    results_folder = Path(__file__).parent / "nnUNet_results"
    os.environ['RESULTS_FOLDER'] = str(results_folder)
    
    print("="*70)
    print("BraTS 2021 TUMOR SEGMENTATION (SINGLE-THREADED)")
    print("="*70)
    print(f"RESULTS_FOLDER: {results_folder}")
    print()
    
    # Model paths
    model1_path = results_folder / "3d_fullres" / "Task500_BraTS2021" / "nnUNetTrainerV2BraTSRegions_DA4_BN_BD__nnUNetPlansv2.1"
    model2_path = results_folder / "3d_fullres" / "Task500_BraTS2021" / "nnUNetTrainerV2BraTSRegions_DA4_BN_BD_largeUnet_Groupnorm__nnUNetPlansv2.1"
    
    output_folder = Path(args.output)
    
    # Run both models
    print("\n" + "="*70)
    print("MODEL 1: nnUNetTrainerV2BraTSRegions_DA4_BN_BD")
    print("="*70)
    model1_output = output_folder / "temp_model1"
    run_model_single_threaded(model1_path, args.input, model1_output)
    
    print("\n" + "="*70)
    print("MODEL 2: nnUNetTrainerV2BraTSRegions_DA4_BN_BD_largeUnet_Groupnorm")
    print("="*70)
    model2_output = output_folder / "temp_model2"
    run_model_single_threaded(model2_path, args.input, model2_output)
    
    # Ensemble the two model predictions
    print("\n" + "="*70)
    print("ENSEMBLING MODEL PREDICTIONS")
    print("="*70)
    
    model1_segs = list(model1_output.glob("*.nii.gz"))
    
    for seg1_path in model1_segs:
        case_name = seg1_path.stem.replace('.nii', '')
        seg2_path = model2_output / seg1_path.name
        
        if not seg2_path.exists():
            print(f"[WARNING] Missing model2 prediction for {case_name}")
            continue
        
        print(f"Ensembling {case_name}")
        
        # Load both segmentations
        seg1_img = nib.load(seg1_path)
        seg1 = seg1_img.get_fdata()
        seg2 = nib.load(seg2_path).get_fdata()
        
        # Simple voting ensemble (majority vote per voxel)
        # For 2 models, we just average and round
        ensemble_seg = np.round((seg1 + seg2) / 2.0).astype(np.uint8)
        
        # Save ensemble result
        final_output = output_folder / f"{case_name}.nii.gz"
        ensemble_img = nib.Nifti1Image(ensemble_seg, seg1_img.affine, seg1_img.header)
        nib.save(ensemble_img, final_output)
        
        print(f"✓ Saved: {final_output}")
        
        # Calculate volumes
        volumes = calculate_volumes(final_output)
        
        print(f"\nTumor Volume Analysis for {case_name}:")
        print(f"  NCR (Necrotic Core):        {volumes['NCR']:.2f} cm³")
        print(f"  ED (Peritumoral Edema):     {volumes['ED']:.2f} cm³")
        print(f"  ET (Enhancing Tumor):       {volumes['ET']:.2f} cm³")
        print(f"  TC (Tumor Core):            {volumes['TC']:.2f} cm³")
        print(f"  WT (Whole Tumor):           {volumes['WT']:.2f} cm³")
    
    print("\n" + "="*70)
    print("SEGMENTATION COMPLETE!")
    print("="*70)
    print(f"Results saved to: {output_folder}")


if __name__ == "__main__":
    main()
