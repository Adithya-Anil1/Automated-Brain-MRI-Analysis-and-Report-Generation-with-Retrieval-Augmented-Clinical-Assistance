"""
10_mgmt_dl_results.py
Level 4: Generate Results Report for Deep Learning MGMT Experiment

This script generates visualizations and a summary report from the
ResNet-18 MGMT prediction experiment results.

Input:
    - resnet_mgmt_predictions.csv (patient-level predictions)
    - resnet_mgmt_results.csv (training metrics)

Output:
    - ROC curve plot
    - Summary statistics
    - Exploratory disclaimer report
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not available. Plots will be skipped.")

# Check for sklearn
try:
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn not available.")

# Configuration
SCRIPT_DIR = Path(__file__).parent
PREDICTIONS_CSV = SCRIPT_DIR / "resnet_mgmt_predictions.csv"
RESULTS_CSV = SCRIPT_DIR / "resnet_mgmt_results.csv"
ROC_PLOT = SCRIPT_DIR / "resnet_mgmt_roc_curve.png"
REPORT_TXT = SCRIPT_DIR / "resnet_mgmt_report.txt"


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, output_path: Path) -> float:
    """
    Plot and save ROC curve.
    
    Returns:
        ROC-AUC score
    """
    if not MATPLOTLIB_AVAILABLE or not SKLEARN_AVAILABLE:
        return roc_auc_score(y_true, y_scores) if SKLEARN_AVAILABLE else 0.5
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    # Create figure
    plt.figure(figsize=(8, 8))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='#2563eb', lw=2, 
             label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
             label='Random classifier')
    
    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    
    plt.scatter([optimal_fpr], [optimal_tpr], c='red', s=100, zorder=5,
                label=f'Optimal threshold ({optimal_threshold:.2f})')
    
    # Styling
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Patient-Level ROC Curve: MGMT Prediction\n(EXPLORATORY - NOT FOR CLINICAL USE)',
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add disclaimer text box
    disclaimer = "⚠️ Exploratory experiment only\nSmall cohort (n<100)"
    plt.text(0.55, 0.15, disclaimer, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to: {output_path}")
    
    return auc


def compute_confusion_metrics(y_true: np.ndarray, y_scores: np.ndarray, 
                              threshold: float = 0.5) -> dict:
    """
    Compute confusion matrix metrics at given threshold.
    """
    y_pred = (y_scores >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'threshold': threshold,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'accuracy': accuracy
    }


def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Find optimal threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]


def generate_report(patient_df: pd.DataFrame, auc: float, 
                    metrics: dict, output_path: Path):
    """Generate text report with results and disclaimers."""
    
    report = []
    report.append("=" * 70)
    report.append("DEEP LEARNING MGMT PREDICTION - EXPLORATORY EXPERIMENT REPORT")
    report.append("=" * 70)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    report.append("\n" + "-" * 70)
    report.append("⚠️  CRITICAL DISCLAIMER")
    report.append("-" * 70)
    report.append("This is an EXPLORATORY deep learning experiment performed on a small")
    report.append("cohort to test whether deep spatial features provide any improvement")
    report.append("over classical radiomics for MGMT methylation prediction.")
    report.append("")
    report.append("THESE RESULTS SHOULD NOT BE INTERPRETED AS CLINICALLY VALID.")
    report.append("No claims of diagnostic accuracy can be made from this experiment.")
    
    report.append("\n" + "-" * 70)
    report.append("COHORT SUMMARY")
    report.append("-" * 70)
    total_patients = len(patient_df)
    methylated = (patient_df['MGMT_Label'] == 1).sum()
    unmethylated = (patient_df['MGMT_Label'] == 0).sum()
    report.append(f"Total test patients: {total_patients}")
    report.append(f"  Methylated (MGMT+): {methylated} ({100*methylated/total_patients:.1f}%)")
    report.append(f"  Unmethylated (MGMT-): {unmethylated} ({100*unmethylated/total_patients:.1f}%)")
    
    report.append("\n" + "-" * 70)
    report.append("PATIENT-LEVEL PERFORMANCE")
    report.append("-" * 70)
    report.append(f"ROC-AUC: {auc:.4f}")
    report.append("")
    report.append(f"At optimal threshold ({metrics['threshold']:.3f}):")
    report.append(f"  Sensitivity: {metrics['sensitivity']:.3f}")
    report.append(f"  Specificity: {metrics['specificity']:.3f}")
    report.append(f"  PPV: {metrics['ppv']:.3f}")
    report.append(f"  NPV: {metrics['npv']:.3f}")
    report.append(f"  Accuracy: {metrics['accuracy']:.3f}")
    
    report.append("\n" + "-" * 70)
    report.append("CONFUSION MATRIX (at optimal threshold)")
    report.append("-" * 70)
    report.append(f"                    Predicted MGMT-    Predicted MGMT+")
    report.append(f"  Actual MGMT-      {metrics['true_negatives']:^15}    {metrics['false_positives']:^15}")
    report.append(f"  Actual MGMT+      {metrics['false_negatives']:^15}    {metrics['true_positives']:^15}")
    
    report.append("\n" + "-" * 70)
    report.append("INTERPRETATION GUIDANCE")
    report.append("-" * 70)
    if auc >= 0.7:
        report.append("The AUC suggests potential signal exists that warrants further")
        report.append("investigation with larger, external validation cohorts.")
    elif auc >= 0.6:
        report.append("The AUC shows weak signal. Results are inconclusive and may")
        report.append("be due to chance given the small sample size.")
    else:
        report.append("The AUC does not show meaningful discriminative ability.")
        report.append("Deep features may not capture MGMT signal in this cohort.")
    
    report.append("\n" + "-" * 70)
    report.append("METHODOLOGY")
    report.append("-" * 70)
    report.append("Model: ResNet-18 pretrained on ImageNet")
    report.append("Input: Tumor-intersecting axial FLAIR slices (224x224)")
    report.append("Splitting: Patient-wise stratified train/test (80/20)")
    report.append("Aggregation: Mean slice probability per patient")
    report.append("Loss: Binary Cross-Entropy")
    report.append("Augmentation: Horizontal flip only (preserving texture statistics)")
    
    report.append("\n" + "=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Report saved to: {output_path}")
    
    # Also print to console
    print("\n" + "\n".join(report))


def main():
    """Main pipeline to generate results report."""
    print("=" * 70)
    print("Level 4: MGMT Deep Learning Results Report Generator")
    print("=" * 70)
    
    # Check dependencies
    if not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn is required.")
        return
    
    # Load predictions
    if not PREDICTIONS_CSV.exists():
        print(f"ERROR: Predictions not found: {PREDICTIONS_CSV}")
        print("Run 09_train_resnet_mgmt.py first.")
        return
    
    patient_df = pd.read_csv(PREDICTIONS_CSV)
    print(f"\nLoaded predictions for {len(patient_df)} test patients")
    
    # Extract arrays
    y_true = patient_df['MGMT_Label'].values
    y_scores = patient_df['Patient_Probability'].values
    
    # Check class distribution
    if len(np.unique(y_true)) < 2:
        print("ERROR: Need both classes in test set for evaluation.")
        return
    
    # Compute AUC
    auc = roc_auc_score(y_true, y_scores)
    print(f"\nPatient-Level ROC-AUC: {auc:.4f}")
    
    # Plot ROC curve
    if MATPLOTLIB_AVAILABLE:
        plot_roc_curve(y_true, y_scores, ROC_PLOT)
    
    # Find optimal threshold and compute metrics
    optimal_threshold = find_optimal_threshold(y_true, y_scores)
    metrics = compute_confusion_metrics(y_true, y_scores, optimal_threshold)
    
    # Generate text report
    generate_report(patient_df, auc, metrics, REPORT_TXT)
    
    return auc


if __name__ == "__main__":
    main()
