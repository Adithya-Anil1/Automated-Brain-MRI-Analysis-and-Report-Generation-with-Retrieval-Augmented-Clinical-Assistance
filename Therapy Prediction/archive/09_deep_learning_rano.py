#!/usr/bin/env python3
"""
09_deep_learning_rano.py

Deep Learning-based Treatment Response Prediction from Pre-treatment MRI.

CRITICAL: Uses ONLY pre-treatment scans (week-000-*) as INPUT.
          Labels come from post-treatment RANO ratings (week >= 12).

Pipeline:
1. Phase 1: Data Preparation - Extract tumor slices from pre-treatment T1ce scans
2. Phase 2: Training - Train ResNet18 for binary classification (Responder vs Non-Responder)
3. Evaluation at PATIENT-LEVEL (aggregate slice predictions)

Author: AI-Powered Brain MRI Assistant
Date: 2026-02-01
"""

import os
import re
import json
import random
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Paths
BASE_DIR = Path(__file__).parent
IMAGING_DIR = BASE_DIR / "dataset" / "Imaging"
RANO_CSV = BASE_DIR / "dataset" / "LUMIERE-ExpertRating-v202211.csv"
OUTPUT_DIR = BASE_DIR / "Dataset_RANO"
RESULTS_DIR = BASE_DIR / "rano_dl_results"

# Data parameters
# LUMIERE uses HD-GLIO segmentation labels: 0=background, 1=necrosis/non-enhancing, 2=enhancing
TUMOR_LABELS = [1, 2]  # Use both tumor labels (enhancing preferred)
ENHANCING_TUMOR_LABEL = 2  # Enhancing tumor label (prioritized for slice selection)
MIN_TUMOR_PIXELS = 50  # Minimum tumor pixels to include a slice
MIN_WEEK_FOR_LABEL = 12  # Minimum week for RANO rating (post-treatment)
IMAGE_SIZE = 224  # Target image size
PADDING = 10  # Padding around tumor bbox

# Training parameters
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
PATIENCE = 5  # Early stopping patience
NUM_WORKERS = 0  # DataLoader workers (0 for Windows compatibility)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =============================================================================
# PHASE 1: DATA PREPARATION
# =============================================================================

def parse_week_number(week_str):
    """Extract numeric week from string like 'week-044' or 'week-000-1'."""
    match = re.search(r'week-(\d+)', week_str)
    if match:
        return int(match.group(1))
    return None


def load_rano_labels():
    """
    Load RANO ratings and extract labels for each patient.
    Uses earliest RANO rating where Week >= 12.
    
    Returns:
        dict: {patient_id: {'label': 0/1, 'rano': 'CR/PR/SD/PD', 'week': int}}
    """
    print("\n" + "="*60)
    print("LOADING RANO LABELS")
    print("="*60)
    
    # Load RANO ratings
    rano_df = pd.read_csv(RANO_CSV)
    
    # Find the RANO column (it has a long name)
    rano_col = None
    for col in rano_df.columns:
        if "rating" in col.lower() and "rationale" not in col.lower():
            rano_col = col
            break
    
    if rano_col is None:
        raise ValueError("Could not find RANO rating column!")
    
    print(f"Using RANO column: {rano_col[:50]}...")
    
    # Filter for valid RANO responses (exclude Pre-Op, Post-Op)
    valid_ratings = ['CR', 'PR', 'SD', 'PD']
    rano_df = rano_df[rano_df[rano_col].isin(valid_ratings)].copy()
    
    # Parse week numbers
    rano_df['week_num'] = rano_df['Date'].apply(parse_week_number)
    
    # Filter for week >= MIN_WEEK_FOR_LABEL
    rano_df = rano_df[rano_df['week_num'] >= MIN_WEEK_FOR_LABEL]
    
    # For each patient, get the earliest RANO rating after week 12
    patient_labels = {}
    for patient_id in rano_df['Patient'].unique():
        patient_data = rano_df[rano_df['Patient'] == patient_id]
        earliest = patient_data.loc[patient_data['week_num'].idxmin()]
        
        rano_rating = earliest[rano_col]
        week = earliest['week_num']
        
        # Binary label: CR/PR = 1 (Responder), SD/PD = 0 (Non-Responder)
        if rano_rating in ['CR', 'PR']:
            label = 1
        else:  # SD, PD
            label = 0
        
        patient_labels[patient_id] = {
            'label': label,
            'rano': rano_rating,
            'week': week
        }
    
    # Summary statistics
    responders = sum(1 for p in patient_labels.values() if p['label'] == 1)
    non_responders = sum(1 for p in patient_labels.values() if p['label'] == 0)
    
    print(f"\nPatients with valid RANO labels: {len(patient_labels)}")
    print(f"  Responders (CR/PR): {responders}")
    print(f"  Non-Responders (SD/PD): {non_responders}")
    
    # Detailed breakdown
    rano_counts = defaultdict(int)
    for p in patient_labels.values():
        rano_counts[p['rano']] += 1
    print("\nRANO breakdown:")
    for rating in ['CR', 'PR', 'SD', 'PD']:
        print(f"  {rating}: {rano_counts[rating]}")
    
    return patient_labels


def find_pretreatment_scan(patient_dir):
    """
    Find pre-treatment scan directory (week-000-1 or week-000-2).
    
    Returns:
        Path or None: Path to pre-treatment directory
    """
    # Priority: week-000-1 (pre-op), then week-000-2 (immediate post-op)
    for week in ['week-000-1', 'week-000']:
        week_dir = patient_dir / week
        if week_dir.exists():
            return week_dir
    
    # Fallback: week-000-2
    week_dir = patient_dir / 'week-000-2'
    if week_dir.exists():
        return week_dir
    
    return None


def load_nifti_data(patient_dir):
    """
    Load T1ce and segmentation from a patient directory.
    
    Returns:
        tuple: (t1ce_data, seg_data) as numpy arrays, or (None, None)
    """
    # T1ce file (CT1 in LUMIERE naming)
    t1ce_path = patient_dir / "CT1.nii.gz"
    if not t1ce_path.exists():
        # Try alternative name
        t1ce_path = patient_dir / "T1ce.nii.gz"
    
    if not t1ce_path.exists():
        return None, None
    
    # Segmentation file
    seg_path = patient_dir / "HD-GLIO-AUTO-segmentation" / "native" / "segmentation_CT1_origspace.nii.gz"
    if not seg_path.exists():
        # Try alternative segmentation path
        seg_path = patient_dir / "DeepBraTumIA-segmentation" / "segmentation.nii.gz"
    
    if not seg_path.exists():
        return None, None
    
    # Load NIfTI files
    try:
        t1ce_nii = nib.load(str(t1ce_path))
        seg_nii = nib.load(str(seg_path))
        
        t1ce_data = t1ce_nii.get_fdata()
        seg_data = seg_nii.get_fdata()
        
        return t1ce_data, seg_data
    except Exception as e:
        print(f"  Error loading NIfTI: {e}")
        return None, None


def extract_tumor_slices(t1ce_data, seg_data, patient_id, label, output_dir):
    """
    Extract axial slices containing enhancing tumor.
    
    Args:
        t1ce_data: 3D T1ce volume
        seg_data: 3D segmentation volume
        patient_id: Patient identifier
        label: 0 or 1
        output_dir: Directory to save slices
        
    Returns:
        list: Metadata for each extracted slice
    """
    metadata = []
    label_name = "1_Responder" if label == 1 else "0_NonResponder"
    save_dir = output_dir / label_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get dimensions
    n_slices = t1ce_data.shape[2]  # Axial slices along z-axis
    
    slice_count = 0
    for slice_idx in range(n_slices):
        # Extract slice
        t1ce_slice = t1ce_data[:, :, slice_idx]
        seg_slice = seg_data[:, :, slice_idx]
        
        # Check for any tumor (labels 1 or 2)
        tumor_mask = np.isin(seg_slice, TUMOR_LABELS)
        tumor_pixels = np.sum(tumor_mask)
        
        if tumor_pixels < MIN_TUMOR_PIXELS:
            continue
        
        # Find bounding box of tumor
        rows = np.any(tumor_mask, axis=1)
        cols = np.any(tumor_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Add padding
        rmin = max(0, rmin - PADDING)
        rmax = min(t1ce_slice.shape[0], rmax + PADDING)
        cmin = max(0, cmin - PADDING)
        cmax = min(t1ce_slice.shape[1], cmax + PADDING)
        
        # Crop to bounding box
        cropped = t1ce_slice[rmin:rmax, cmin:cmax]
        
        # Normalize to 0-255
        cropped = cropped.astype(np.float32)
        if cropped.max() > cropped.min():
            cropped = (cropped - cropped.min()) / (cropped.max() - cropped.min())
        cropped = (cropped * 255).astype(np.uint8)
        
        # Resize to target size
        img = Image.fromarray(cropped)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        
        # Convert to RGB (3 channels by repeating grayscale)
        img_rgb = Image.merge('RGB', (img, img, img))
        
        # Save
        filename = f"{patient_id}_slice{slice_idx:03d}.png"
        filepath = save_dir / filename
        img_rgb.save(filepath)
        
        # Store metadata
        metadata.append({
            'filepath': str(filepath),
            'patient_id': patient_id,
            'slice_idx': slice_idx,
            'label': label,
            'tumor_pixels': tumor_pixels
        })
        
        slice_count += 1
    
    return metadata


def prepare_dataset(patient_labels):
    """
    Prepare the dataset by extracting tumor slices from all patients.
    Splits at PATIENT level (not slice level).
    
    Returns:
        tuple: (train_metadata, val_metadata)
    """
    print("\n" + "="*60)
    print("PREPARING DATASET")
    print("="*60)
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_dir = OUTPUT_DIR / "train"
    val_dir = OUTPUT_DIR / "val"
    
    # Clear existing data
    for split_dir in [train_dir, val_dir]:
        if split_dir.exists():
            import shutil
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)
    
    # Split patients (stratified by label)
    patient_ids = list(patient_labels.keys())
    labels = [patient_labels[p]['label'] for p in patient_ids]
    
    train_patients, val_patients = train_test_split(
        patient_ids, 
        test_size=0.2, 
        stratify=labels, 
        random_state=SEED
    )
    
    print(f"\nPatient split:")
    print(f"  Train: {len(train_patients)} patients")
    print(f"  Val: {len(val_patients)} patients")
    
    # Process each split
    all_metadata = {'train': [], 'val': []}
    
    for split_name, patients, output_dir in [
        ('train', train_patients, train_dir),
        ('val', val_patients, val_dir)
    ]:
        print(f"\nProcessing {split_name} set...")
        
        for patient_id in tqdm(patients, desc=f"Extracting {split_name} slices"):
            patient_dir = IMAGING_DIR / patient_id
            
            if not patient_dir.exists():
                continue
            
            # Find pre-treatment scan
            pretreat_dir = find_pretreatment_scan(patient_dir)
            if pretreat_dir is None:
                continue
            
            # Load T1ce and segmentation
            t1ce_data, seg_data = load_nifti_data(pretreat_dir)
            if t1ce_data is None:
                continue
            
            # Extract slices
            label = patient_labels[patient_id]['label']
            slice_metadata = extract_tumor_slices(
                t1ce_data, seg_data, patient_id, label, output_dir
            )
            
            all_metadata[split_name].extend(slice_metadata)
    
    # Save metadata CSVs
    train_df = pd.DataFrame(all_metadata['train'])
    val_df = pd.DataFrame(all_metadata['val'])
    
    train_df.to_csv(OUTPUT_DIR / "train_metadata.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val_metadata.csv", index=False)
    
    # Summary
    print("\n" + "-"*40)
    print("Dataset Summary:")
    
    if len(train_df) > 0:
        print(f"  Train: {len(train_df)} slices from {train_df['patient_id'].nunique()} patients")
        train_labels = train_df['label'].value_counts()
        print(f"  Train class distribution:")
        print(f"    Non-Responders (0): {train_labels.get(0, 0)} slices")
        print(f"    Responders (1): {train_labels.get(1, 0)} slices")
    else:
        print("  Train: 0 slices")
    
    if len(val_df) > 0:
        print(f"  Val: {len(val_df)} slices from {val_df['patient_id'].nunique()} patients")
        val_labels = val_df['label'].value_counts()
        print(f"  Val class distribution:")
        print(f"    Non-Responders (0): {val_labels.get(0, 0)} slices")
        print(f"    Responders (1): {val_labels.get(1, 0)} slices")
    else:
        print("  Val: 0 slices")
    
    return all_metadata['train'], all_metadata['val']


# =============================================================================
# PHASE 2: TRAINING
# =============================================================================

class RANODataset(Dataset):
    """PyTorch Dataset for RANO prediction."""
    
    def __init__(self, metadata, transform=None):
        """
        Args:
            metadata: List of dicts with 'filepath', 'patient_id', 'label'
            transform: Image transforms
        """
        self.metadata = metadata
        self.transform = transform
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load image
        img = Image.open(item['filepath']).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        label = torch.tensor(item['label'], dtype=torch.float32)
        patient_id = item['patient_id']
        
        return img, label, patient_id


def get_transforms(train=True):
    """Get image transforms for training or validation."""
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])


def build_model():
    """Build ResNet18 model for binary classification."""
    
    model = models.resnet18(pretrained=True)
    
    # Replace final fully connected layer
    num_features = model.fc.in_features  # 512
    model.fc = nn.Linear(num_features, 1)
    
    return model.to(DEVICE)


def calculate_pos_weight(train_metadata):
    """Calculate positive class weight for imbalanced data."""
    
    labels = [m['label'] for m in train_metadata]
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos
    
    pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    print(f"\nClass weight: pos_weight = {pos_weight:.2f}")
    
    return torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)


def train_epoch(model, dataloader, criterion, optimizer):
    """Train for one epoch."""
    
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels, _ in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    
    try:
        epoch_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        epoch_auc = 0.5
    
    return epoch_loss, epoch_auc


def validate_epoch(model, dataloader, criterion):
    """Validate for one epoch."""
    
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_patient_ids = []
    
    with torch.no_grad():
        for images, labels, patient_ids in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_patient_ids.extend(patient_ids)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    
    # Slice-level AUC
    try:
        slice_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        slice_auc = 0.5
    
    # Patient-level AUC
    patient_scores = defaultdict(list)
    patient_labels = {}
    for pred, label, patient_id in zip(all_preds, all_labels, all_patient_ids):
        patient_scores[patient_id].append(pred)
        patient_labels[patient_id] = label
    
    patient_preds = [np.mean(patient_scores[p]) for p in patient_scores]
    patient_true = [patient_labels[p] for p in patient_scores]
    
    try:
        patient_auc = roc_auc_score(patient_true, patient_preds)
    except ValueError:
        patient_auc = 0.5
    
    return epoch_loss, slice_auc, patient_auc, patient_preds, patient_true


def train_model(train_metadata, val_metadata):
    """
    Full training loop with early stopping.
    
    Returns:
        dict: Training history and final results
    """
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create datasets and dataloaders
    train_dataset = RANODataset(train_metadata, transform=get_transforms(train=True))
    val_dataset = RANODataset(val_metadata, transform=get_transforms(train=False))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS
    )
    
    # Build model
    model = build_model()
    print(f"\nModel: ResNet18 (pretrained)")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function with class balancing
    pos_weight = calculate_pos_weight(train_metadata)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training history
    history = {
        'train_loss': [],
        'train_auc': [],
        'val_loss': [],
        'val_slice_auc': [],
        'val_patient_auc': []
    }
    
    # Early stopping
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None
    
    print("\nStarting training...")
    print("-"*60)
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer)
        
        # Validate
        val_loss, val_slice_auc, val_patient_auc, _, _ = validate_epoch(
            model, val_loader, criterion
        )
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_slice_auc'].append(val_slice_auc)
        history['val_patient_auc'].append(val_patient_auc)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Patient AUC: {val_patient_auc:.4f}")
        
        # Early stopping check (using patient-level AUC)
        if val_patient_auc > best_val_auc:
            best_val_auc = val_patient_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} (patience={PATIENCE})")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save best model
    model_path = RESULTS_DIR / "best_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nBest model saved to: {model_path}")
    
    return model, history, val_loader


def evaluate_model(model, val_loader):
    """
    Final evaluation at patient level.
    
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*60)
    print("FINAL EVALUATION (PATIENT-LEVEL)")
    print("="*60)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_patient_ids = []
    
    with torch.no_grad():
        for images, labels, patient_ids in val_loader:
            images = images.to(DEVICE)
            
            outputs = model(images).squeeze()
            preds = torch.sigmoid(outputs).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_patient_ids.extend(patient_ids)
    
    # Aggregate to patient level
    patient_scores = defaultdict(list)
    patient_labels = {}
    
    for pred, label, patient_id in zip(all_preds, all_labels, all_patient_ids):
        patient_scores[patient_id].append(pred)
        patient_labels[patient_id] = int(label)
    
    # Calculate patient-level predictions
    patient_preds = {p: np.mean(scores) for p, scores in patient_scores.items()}
    patient_preds_binary = {p: 1 if score >= 0.5 else 0 for p, score in patient_preds.items()}
    
    # Prepare arrays for metrics
    y_true = [patient_labels[p] for p in patient_preds]
    y_score = [patient_preds[p] for p in patient_preds]
    y_pred = [patient_preds_binary[p] for p in patient_preds]
    
    # Calculate metrics
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = 0.5
    
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\nPatient-level Results ({len(patient_preds)} patients):")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
    print(f"  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, 
          target_names=['Non-Responder', 'Responder']))
    
    results = {
        'patient_auc': float(auc),
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'n_patients': len(patient_preds),
        'patient_predictions': {p: {'score': float(patient_preds[p]), 
                                    'predicted': int(patient_preds_binary[p]),
                                    'true': int(patient_labels[p])}
                               for p in patient_preds}
    }
    
    return results, y_true, y_score


def plot_training_curves(history):
    """Plot and save training curves."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # AUC curves
    axes[1].plot(history['train_auc'], label='Train AUC', marker='o')
    axes[1].plot(history['val_patient_auc'], label='Val Patient AUC', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].set_title('Training and Validation AUC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {RESULTS_DIR / 'training_curves.png'}")


def plot_confusion_matrix(y_true, y_pred):
    """Plot and save confusion matrix."""
    
    cm = confusion_matrix(y_true, [1 if p >= 0.5 else 0 for p in y_pred])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['Non-Responder', 'Responder']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix (Patient-Level)',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {RESULTS_DIR / 'confusion_matrix.png'}")


def save_results(results, history):
    """Save results to JSON file."""
    
    output = {
        'config': {
            'seed': SEED,
            'image_size': IMAGE_SIZE,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'min_tumor_pixels': MIN_TUMOR_PIXELS,
            'tumor_labels': TUMOR_LABELS,
            'enhancing_tumor_label': ENHANCING_TUMOR_LABEL,
            'device': str(DEVICE)
        },
        'results': results,
        'history': {k: [float(v) for v in vals] for k, vals in history.items()}
    }
    
    with open(RESULTS_DIR / "results.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {RESULTS_DIR / 'results.json'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("DEEP LEARNING RANO PREDICTION")
    print("Pre-treatment MRI → Treatment Response (Responder vs Non-Responder)")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Random seed: {SEED}")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Early stopping patience: {PATIENCE}")
    
    # =========================================================================
    # PHASE 1: DATA PREPARATION
    # =========================================================================
    
    # Load RANO labels
    patient_labels = load_rano_labels()
    
    if len(patient_labels) == 0:
        print("\nERROR: No valid patient labels found!")
        return
    
    # Prepare dataset (extract slices)
    train_metadata, val_metadata = prepare_dataset(patient_labels)
    
    if len(train_metadata) == 0 or len(val_metadata) == 0:
        print("\nERROR: Not enough data for training!")
        return
    
    # =========================================================================
    # PHASE 2: TRAINING
    # =========================================================================
    
    # Train model
    model, history, val_loader = train_model(train_metadata, val_metadata)
    
    # Evaluate
    results, y_true, y_score = evaluate_model(model, val_loader)
    
    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    
    plot_training_curves(history)
    plot_confusion_matrix(y_true, y_score)
    save_results(results, history)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nFinal Patient-Level AUC: {results['patient_auc']:.4f}")
    print(f"\nOutputs saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
