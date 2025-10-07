"""
Setup script for nnU-Net with BraTS 2024 dataset
This script configures the environment and prepares the dataset structure
"""
import os
import sys
from pathlib import Path


def setup_nnunet_paths():
    """
    Setup the required nnU-Net environment variables
    """
    # Get the current project directory
    project_dir = Path(__file__).parent.absolute()
    
    # Create nnU-Net directories
    nnunet_raw = project_dir / "nnUNet_raw"
    nnunet_preprocessed = project_dir / "nnUNet_preprocessed"
    nnunet_results = project_dir / "nnUNet_results"
    
    # Create directories if they don't exist
    nnunet_raw.mkdir(exist_ok=True)
    nnunet_preprocessed.mkdir(exist_ok=True)
    nnunet_results.mkdir(exist_ok=True)
    
    # Set environment variables
    os.environ['nnUNet_raw'] = str(nnunet_raw)
    os.environ['nnUNet_preprocessed'] = str(nnunet_preprocessed)
    os.environ['nnUNet_results'] = str(nnunet_results)
    
    print("nnU-Net directories created and environment variables set:")
    print(f"nnUNet_raw: {nnunet_raw}")
    print(f"nnUNet_preprocessed: {nnunet_preprocessed}")
    print(f"nnUNet_results: {nnunet_results}")
    
    # Create dataset structure for BraTS 2024
    dataset_name = "Dataset001_BraTS2024"
    dataset_path = nnunet_raw / dataset_name
    
    # Create train and test image/label directories
    (dataset_path / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dataset_path / "labelsTr").mkdir(parents=True, exist_ok=True)
    (dataset_path / "imagesTs").mkdir(parents=True, exist_ok=True)
    
    print(f"\nDataset structure created at: {dataset_path}")
    print("  - imagesTr/ (for training images)")
    print("  - labelsTr/ (for training labels)")
    print("  - imagesTs/ (for test images)")
    
    return str(nnunet_raw), str(nnunet_preprocessed), str(nnunet_results)


def create_dataset_json(nnunet_raw_path):
    """
    Create dataset.json file for BraTS 2024
    """
    import json
    
    dataset_path = Path(nnunet_raw_path) / "Dataset001_BraTS2024"
    
    dataset_json = {
        "channel_names": {
            "0": "T1",
            "1": "T1CE",
            "2": "T2",
            "3": "FLAIR"
        },
        "labels": {
            "background": 0,
            "NCR": 1,  # Necrotic tumor core
            "ED": 2,   # Peritumoral edematous/invaded tissue
            "ET": 3    # GD-enhancing tumor
        },
        "numTraining": 0,  # Will be updated after data conversion
        "file_ending": ".nii.gz",
        "name": "BraTS2024",
        "description": "Brain Tumor Segmentation Challenge 2024",
        "reference": "https://www.synapse.org/#!Synapse:syn53708249/wiki/",
        "tensorImageSize": "4D"
    }
    
    json_path = dataset_path / "dataset.json"
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"\ndataset.json created at: {json_path}")
    return str(json_path)


if __name__ == "__main__":
    print("Setting up nnU-Net for BraTS 2024 segmentation...")
    print("=" * 60)
    
    # Setup paths and create directories
    raw_path, preprocessed_path, results_path = setup_nnunet_paths()
    
    # Create dataset.json
    json_path = create_dataset_json(raw_path)
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("\nNext steps for PRETRAINED MODEL setup:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download pretrained model: python download_pretrained_model.py")
    print("3. Run inference on patient MRI: python inference_nnunet.py --help")
    print("\nYour segmentation MVP is ready to use!")
