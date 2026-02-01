"""
05_train_wholetumor_models.py

Leave-One-Out evaluation for MGMT prediction using WholeTumor radiomic features.

Inputs: `Level4_Radiomic_Features_WholeTumor.csv`
Outputs: prints AUCs, prints confusion matrix for winner, saves `Level4_WholeTumor_ROC.png`
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

SCRIPT_DIR = Path(__file__).parent
INPUT_CSV = SCRIPT_DIR / "Level4_Radiomic_Features_WholeTumor.csv"
ROC_PNG = SCRIPT_DIR / "Level4_WholeTumor_ROC.png"


def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    if 'MGMT_Label' not in df.columns:
        raise KeyError('MGMT_Label column not found in CSV')

    y = df['MGMT_Label'].astype(int).values
    numeric = df.select_dtypes(include=[np.number]).copy()
    if 'MGMT_Label' in numeric.columns:
        numeric = numeric.drop(columns=['MGMT_Label'])
    X = numeric.values
    feature_names = list(numeric.columns)
    return X, y, feature_names


def main():
    X, y, feature_names = load_data(INPUT_CSV)
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} numeric features")

    loo = LeaveOneOut()

    lr = LogisticRegression(penalty='l1', solver='saga', C=1.0, max_iter=5000)
    rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    if XGBClassifier is None:
        print('Warning: xgboost not available; skipping XGBoost')
        xgb = None
    else:
        xgb = XGBClassifier(max_depth=2, n_estimators=50, use_label_encoder=False, eval_metric='logloss', random_state=42)

    probs = {'LogisticRegression': [], 'RandomForest': [], 'XGBoost': [] if xgb is not None else None}
    truths = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        k = min(5, X_train_s.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_sel = selector.fit_transform(X_train_s, y_train)
        X_test_sel = selector.transform(X_test_s)

        # Train and predict probabilities
        try:
            lr.fit(X_train_sel, y_train)
            prob_lr = lr.predict_proba(X_test_sel)[0, 1]
        except Exception:
            prob_lr = float(lr.predict(X_test_sel)[0])

        rf.fit(X_train_sel, y_train)
        prob_rf = rf.predict_proba(X_test_sel)[0, 1]

        if xgb is not None:
            xgb.fit(X_train_sel, y_train)
            prob_xgb = xgb.predict_proba(X_test_sel)[0, 1]
        else:
            prob_xgb = None

        probs['LogisticRegression'].append(prob_lr)
        probs['RandomForest'].append(prob_rf)
        if xgb is not None:
            probs['XGBoost'].append(prob_xgb)

        truths.append(int(y_test[0]))

    truths = np.array(truths)

    results = {}
    try:
        results['LogisticRegression'] = roc_auc_score(truths, np.array(probs['LogisticRegression']))
    except Exception:
        results['LogisticRegression'] = float('nan')
    try:
        results['RandomForest'] = roc_auc_score(truths, np.array(probs['RandomForest']))
    except Exception:
        results['RandomForest'] = float('nan')
    if xgb is not None:
        try:
            results['XGBoost'] = roc_auc_score(truths, np.array(probs['XGBoost']))
        except Exception:
            results['XGBoost'] = float('nan')

    for name, aucv in results.items():
        print(f"{name} AUC: {aucv:.4f}")

    # Determine winner
    winner = max(results.items(), key=lambda kv: (kv[1] if not np.isnan(kv[1]) else -1e9))[0]
    print(f"\nWinner: {winner}")

    # Confusion matrix for winner (threshold 0.5)
    winner_probs = np.array(probs[winner])
    winner_preds = (winner_probs >= 0.5).astype(int)
    cm = confusion_matrix(truths, winner_preds)
    print(f"\nConfusion Matrix for {winner}:\n", cm)

    # ROC plot for winner
    fpr, tpr, _ = roc_curve(truths, winner_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{winner} ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Whole Tumor Features')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ROC_PNG, dpi=150)
    print(f"Saved ROC plot to: {ROC_PNG}")


if __name__ == '__main__':
    main()
