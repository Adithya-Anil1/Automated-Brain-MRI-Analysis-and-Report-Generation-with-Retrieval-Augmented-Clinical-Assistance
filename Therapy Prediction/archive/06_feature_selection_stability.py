"""
06_feature_selection_stability.py

Estimate feature-selection stability and downstream RF performance using nested
selection without leakage.

Procedure:
- Input: Level4_Radiomic_Features_WholeTumor.csv (expects `MGMT_Label` or `MGMT_Status`)
- StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- For each fold:
    - Fit StandardScaler on training data only
    - Fit L1 LogisticRegression on training data only
    - Select non-zero coef features (limit to max 8 by abs(coef))
    - Train RandomForestClassifier on selected features (training data only)
    - Evaluate ROC-AUC on held-out fold
- Report mean ROC-AUC and selection counts per feature

No preprocessing or selection uses test-fold information.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


SCRIPT_DIR = Path(__file__).parent
INPUT_CSV = SCRIPT_DIR / "Level4_Radiomic_Features_WholeTumor.csv"


def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    # Support both possible label names
    if 'MGMT_Label' in df.columns:
        label_col = 'MGMT_Label'
    elif 'MGMT_Status' in df.columns:
        label_col = 'MGMT_Status'
    else:
        raise KeyError('MGMT_Label or MGMT_Status column not found in CSV')

    y = df[label_col].astype(int).values
    # Ensure binary 0/1
    unique = np.unique(y)
    if set(unique) - {0, 1}:
        # try mapping common strings
        mapping = {}
        if set(unique) <= set(['Methylated', 'Unmethylated', 'methylated', 'unmethylated']):
            mapping = {k: 1 if 'methyl' in str(k).lower() else 0 for k in unique}
            y = np.array([mapping[v] for v in df[label_col]])

    numeric = df.select_dtypes(include=[np.number]).copy()
    if label_col in numeric.columns:
        numeric = numeric.drop(columns=[label_col])
    feature_names = list(numeric.columns)
    X = numeric.values
    return X, y, feature_names


def main():
    X, y, feature_names = load_data(INPUT_CSV)
    n_samples, n_features = X.shape
    print(f"Data: {n_samples} samples, {n_features} numeric features")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    aucs = []
    selection_counts = {fn: 0 for fn in feature_names}

    fold_id = 0
    for train_idx, test_idx in skf.split(X, y):
        fold_id += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # scaler fit on train only
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # L1 logistic on train only
        lr = LogisticRegression(penalty='l1', solver='saga', C=1.0, max_iter=5000)
        lr.fit(X_train_s, y_train)

        coefs = lr.coef_.ravel()
        # Non-zero indices
        nz_idx = np.where(np.abs(coefs) > 1e-8)[0]
        if nz_idx.size == 0:
            # pick top features by absolute coef magnitude (even if zeros)
            ranked = np.argsort(-np.abs(coefs))
            selected_idx = ranked[:min(8, len(ranked))]
        else:
            # limit to max 8 by magnitude
            selected_idx = nz_idx
            if selected_idx.size > 8:
                ordered = np.argsort(-np.abs(coefs[selected_idx]))
                selected_idx = selected_idx[ordered][:8]

        selected_names = [feature_names[i] for i in selected_idx]
        for fn in selected_names:
            selection_counts[fn] += 1

        # train RF on selected features (train only)
        rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        X_train_sel = X_train_s[:, selected_idx]
        X_test_sel = X_test_s[:, selected_idx]
        rf.fit(X_train_sel, y_train)
        probs = rf.predict_proba(X_test_sel)[:, 1]
        try:
            auc = roc_auc_score(y_test, probs)
        except Exception:
            auc = float('nan')
        aucs.append(auc)
        print(f"Fold {fold_id}: selected {len(selected_idx)} features, AUC={auc:.4f}")

    mean_auc = np.nanmean(aucs)
    print(f"\nMean ROC-AUC (5-fold Stratified): {mean_auc:.4f}")

    # Feature stability: how many folds selected each feature
    stability = pd.Series(selection_counts).sort_values(ascending=False)
    print('\nFeature selection counts across folds (stability):')
    print(stability[stability > 0])


if __name__ == '__main__':
    main()
