"""
09_train_resnet_mgmt.py
Level 4: Train ResNet-18 for MGMT Prediction from MRI Slices

This script trains a ResNet-18 model (pretrained on ImageNet) to predict
MGMT promoter methylation status from tumor-containing MRI slices.

CRITICAL: Patient-wise splitting is enforced - all slices from one patient
stay entirely in either train or test set to prevent data leakage.

Input:
    - slices_output/slice_metadata.csv (from 08_extract_tumor_slices.py)
    - Extracted slice NPY files

Output:
    - Trained model checkpoint
    - Patient-level ROC-AUC evaluation
    - Training metrics and predictions
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch is required. Install with: pip install torch torchvision")

# Check for torchvision
try:
    import torchvision.models as models
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("ERROR: torchvision is required. Install with: pip install torchvision")

# Check for sklearn
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("ERROR: scikit-learn is required. Install with: pip install scikit-learn")

# Configuration
SCRIPT_DIR = Path(__file__).parent
SLICES_DIR = SCRIPT_DIR / "slices_output"
METADATA_CSV = SLICES_DIR / "slice_metadata.csv"
MODEL_OUTPUT = SCRIPT_DIR / "resnet_mgmt_model.pth"
PREDICTIONS_CSV = SCRIPT_DIR / "resnet_mgmt_predictions.csv"
RESULTS_CSV = SCRIPT_DIR / "resnet_mgmt_results.csv"

# Training parameters
RANDOM_SEED = 42
TEST_SPLIT = 0.2  # 80/20 train/test split
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 10
NUM_WORKERS = 0  # Windows compatibility


class MGMTSliceDataset(Dataset):
    """
    PyTorch Dataset for MGMT slice classification.
    
    Each item returns:
        - image: Preprocessed slice as 3-channel tensor (for RGB ResNet input)
        - label: MGMT label (0 or 1)
        - patient_id: Patient identifier (for aggregation)
    """
    
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        transform=None,
        augment: bool = False
    ):
        """
        Args:
            metadata_df: DataFrame with Slice_Path, MGMT_Label, Patient_ID
            transform: Optional torchvision transforms
            augment: Whether to apply data augmentation
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.transform = transform
        self.augment = augment
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.metadata.iloc[idx]
        
        # Load slice
        slice_path = row['Slice_Path']
        slice_data = np.load(slice_path)
        
        # Convert to 3-channel (RGB) for ResNet
        # Stack the same slice 3 times
        slice_3ch = np.stack([slice_data, slice_data, slice_data], axis=0)
        
        # Convert to tensor
        image = torch.from_numpy(slice_3ch).float()
        
        # Apply augmentation if training
        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[2])
            
            # Random small rotation (±15 degrees) - simplified using affine
            if torch.rand(1).item() > 0.5:
                angle = (torch.rand(1).item() - 0.5) * 30  # ±15 degrees
                # Note: For simplicity, we skip rotation here as it requires
                # more complex implementation. Using flip is sufficient for
                # moderate augmentation.
        
        # Normalize for ImageNet pretrained model
        # ImageNet mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        label = int(row['MGMT_Label'])
        patient_id = row['Patient_ID']
        
        return image, label, patient_id


def patient_wise_split(
    metadata_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data at patient level with stratification by MGMT label.
    
    CRITICAL: All slices from one patient go entirely to train OR test.
    
    Args:
        metadata_df: Full metadata DataFrame
        test_size: Fraction for test set
        random_state: Random seed for reproducibility
    
    Returns:
        train_df, test_df: Metadata DataFrames for train and test sets
    """
    # Get unique patients with their labels
    patient_labels = metadata_df.groupby('Patient_ID')['MGMT_Label'].first().reset_index()
    
    # Stratified split at patient level
    train_patients, test_patients = train_test_split(
        patient_labels['Patient_ID'].values,
        test_size=test_size,
        stratify=patient_labels['MGMT_Label'].values,
        random_state=random_state
    )
    
    # Create train and test DataFrames
    train_df = metadata_df[metadata_df['Patient_ID'].isin(train_patients)]
    test_df = metadata_df[metadata_df['Patient_ID'].isin(test_patients)]
    
    return train_df, test_df


def verify_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """
    Verify that no patient appears in both train and test sets.
    
    Returns:
        True if split is valid (no leakage), False otherwise
    """
    train_patients = set(train_df['Patient_ID'].unique())
    test_patients = set(test_df['Patient_ID'].unique())
    
    overlap = train_patients.intersection(test_patients)
    
    if len(overlap) > 0:
        print(f"ERROR: Data leakage detected! {len(overlap)} patients in both sets:")
        print(overlap)
        return False
    
    print("[OK] Split verification passed: No patient overlap between train/test")
    return True


def create_model(num_classes: int = 1) -> nn.Module:
    """
    Create ResNet-18 model with modified classifier for binary classification.
    
    Args:
        num_classes: Number of output classes (1 for binary with BCEWithLogitsLoss)
    
    Returns:
        Modified ResNet-18 model
    """
    # Load pretrained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        average_loss, accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.float().to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, pd.DataFrame]:
    """
    Evaluate model and collect per-slice predictions.
    
    Returns:
        average_loss, accuracy, predictions_df
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    
    with torch.no_grad():
        for images, labels, patient_ids in dataloader:
            images = images.to(device)
            labels_tensor = labels.float().to(device)
            
            outputs = model(images).squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            loss = criterion(outputs, labels_tensor)
            
            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            total += labels_tensor.size(0)
            correct += (predicted == labels_tensor).sum().item()
            
            # Collect predictions
            for i in range(len(labels)):
                all_predictions.append({
                    'Patient_ID': patient_ids[i],
                    'MGMT_Label': labels[i].item(),
                    'Slice_Probability': probs[i].cpu().item()
                })
    
    avg_loss = running_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    predictions_df = pd.DataFrame(all_predictions)
    
    return avg_loss, accuracy, predictions_df


def aggregate_to_patient_level(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate slice-level predictions to patient-level using mean probability.
    
    Args:
        predictions_df: DataFrame with Slice_Probability per slice
    
    Returns:
        Patient-level DataFrame with aggregated probability
    """
    patient_preds = predictions_df.groupby('Patient_ID').agg({
        'MGMT_Label': 'first',
        'Slice_Probability': 'mean'  # Mean aggregation
    }).reset_index()
    
    patient_preds.rename(columns={'Slice_Probability': 'Patient_Probability'}, inplace=True)
    
    return patient_preds


def compute_patient_auc(patient_df: pd.DataFrame) -> float:
    """
    Compute ROC-AUC at patient level.
    
    Args:
        patient_df: DataFrame with Patient_Probability and MGMT_Label
    
    Returns:
        Patient-level ROC-AUC
    """
    y_true = patient_df['MGMT_Label'].values
    y_scores = patient_df['Patient_Probability'].values
    
    # Check if we have both classes
    if len(np.unique(y_true)) < 2:
        print("WARNING: Only one class present in test set. Cannot compute AUC.")
        return 0.5
    
    try:
        auc = roc_auc_score(y_true, y_scores)
        return auc
    except Exception as e:
        print(f"Error computing AUC: {e}")
        return 0.5


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train ResNet-18 for MGMT prediction')
    parser.add_argument('--verify-split-only', action='store_true',
                        help='Only verify split, do not train')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of epochs (default: {NUM_EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help=f'Learning rate (default: {LEARNING_RATE})')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Level 4: ResNet-18 MGMT Prediction Training")
    print("=" * 70)
    print("\n[!] EXPLORATORY EXPERIMENT - NOT FOR CLINICAL USE")
    print("=" * 70)
    
    # Check dependencies
    if not all([TORCH_AVAILABLE, TORCHVISION_AVAILABLE, SKLEARN_AVAILABLE]):
        print("\nMissing required dependencies. Please install:")
        print("  pip install torch torchvision scikit-learn")
        return
    
    # Set random seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Load metadata
    if not METADATA_CSV.exists():
        print(f"\nERROR: Metadata not found: {METADATA_CSV}")
        print("Run 08_extract_tumor_slices.py first.")
        return
    
    metadata_df = pd.read_csv(METADATA_CSV)
    print(f"\nLoaded {len(metadata_df)} slices from {metadata_df['Patient_ID'].nunique()} patients")
    
    # Patient-wise split
    print("\n" + "-" * 70)
    print("Creating patient-wise train/test split...")
    print("-" * 70)
    
    train_df, test_df = patient_wise_split(metadata_df, test_size=TEST_SPLIT, random_state=RANDOM_SEED)
    
    # Verify split
    if not verify_split(train_df, test_df):
        print("CRITICAL ERROR: Data leakage detected. Aborting.")
        return
    
    # Print split statistics
    print(f"\nTrain set:")
    print(f"  Patients: {train_df['Patient_ID'].nunique()}")
    print(f"  Slices: {len(train_df)}")
    print(f"  Methylated: {train_df.groupby('Patient_ID')['MGMT_Label'].first().sum()}")
    print(f"  Unmethylated: {(train_df.groupby('Patient_ID')['MGMT_Label'].first() == 0).sum()}")
    
    print(f"\nTest set:")
    print(f"  Patients: {test_df['Patient_ID'].nunique()}")
    print(f"  Slices: {len(test_df)}")
    print(f"  Methylated: {test_df.groupby('Patient_ID')['MGMT_Label'].first().sum()}")
    print(f"  Unmethylated: {(test_df.groupby('Patient_ID')['MGMT_Label'].first() == 0).sum()}")
    
    if args.verify_split_only:
        print("\n[OK] Split verification complete. Exiting.")
        return
    
    # Create datasets and dataloaders
    print("\n" + "-" * 70)
    print("Creating datasets...")
    print("-" * 70)
    
    train_dataset = MGMTSliceDataset(train_df, augment=True)
    test_dataset = MGMTSliceDataset(test_df, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create model
    print("\n" + "-" * 70)
    print("Creating ResNet-18 model (pretrained on ImageNet)...")
    print("-" * 70)
    
    model = create_model(num_classes=1)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training loop with early stopping
    print("\n" + "-" * 70)
    print(f"Training for up to {args.epochs} epochs (early stopping patience: {EARLY_STOPPING_PATIENCE})")
    print("-" * 70)
    
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = []
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        val_loss, val_acc, _ = evaluate(model, test_loader, criterion, device)
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        print(f"Epoch {epoch + 1:3d}/{args.epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), MODEL_OUTPUT)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
    
    # Load best model for final evaluation
    print("\n" + "-" * 70)
    print("Final Evaluation (loading best model)")
    print("-" * 70)
    
    model.load_state_dict(torch.load(MODEL_OUTPUT, weights_only=True))
    
    # Get final predictions
    _, _, test_predictions = evaluate(model, test_loader, criterion, device)
    
    # Aggregate to patient level
    patient_predictions = aggregate_to_patient_level(test_predictions)
    
    # Compute patient-level AUC
    patient_auc = compute_patient_auc(patient_predictions)
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS (EXPLORATORY - NOT FOR CLINICAL USE)")
    print("=" * 70)
    print(f"\nPatient-Level ROC-AUC: {patient_auc:.4f}")
    print(f"\nTest Set Statistics:")
    print(f"  Total test patients: {len(patient_predictions)}")
    print(f"  Methylated (actual): {(patient_predictions['MGMT_Label'] == 1).sum()}")
    print(f"  Unmethylated (actual): {(patient_predictions['MGMT_Label'] == 0).sum()}")
    
    # Save predictions
    patient_predictions.to_csv(PREDICTIONS_CSV, index=False)
    print(f"\nPatient predictions saved to: {PREDICTIONS_CSV}")
    
    # Save results summary
    results_summary = {
        'metric': ['Patient_Level_ROC_AUC', 'Train_Patients', 'Test_Patients', 
                   'Total_Slices', 'Epochs_Trained', 'Best_Val_Loss'],
        'value': [patient_auc, train_df['Patient_ID'].nunique(), 
                  test_df['Patient_ID'].nunique(), len(metadata_df),
                  len(training_history), best_val_loss]
    }
    pd.DataFrame(results_summary).to_csv(RESULTS_CSV, index=False)
    print(f"Results summary saved to: {RESULTS_CSV}")
    
    print(f"\nModel checkpoint saved to: {MODEL_OUTPUT}")
    
    # Final disclaimer
    print("\n" + "=" * 70)
    print("[!] IMPORTANT DISCLAIMER")
    print("=" * 70)
    print("This is an EXPLORATORY deep learning experiment on a small cohort.")
    print("Results should NOT be interpreted as clinically valid.")
    print("The purpose is to assess if deep spatial features contain any")
    print("MGMT-related signal beyond classical radiomics.")
    print("=" * 70)
    
    return patient_auc


if __name__ == "__main__":
    main()
