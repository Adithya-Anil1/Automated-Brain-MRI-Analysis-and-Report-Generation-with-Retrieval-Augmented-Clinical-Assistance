"""
05_train_flair_models.py

Train and evaluate MGMT prediction using LOOCV on FLAIR radiomics.

Requirements implemented:
- Loads Level4_Radiomic_Features_FLAIR.csv
- Drops non-numeric columns (keeps `MGMT_Label` as y)
- Runs Leave-One-Out CV
- Inside-loop: StandardScaler -> SelectKBest(k=5) fitted on training only
- Models: LogisticRegression (L1), RandomForest (100, max_depth=3), XGBoost (50, max_depth=2)
- Prints AUC for each model and the winner
"""
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

SCRIPT_DIR = Path(__file__).parent
INPUT_CSV = SCRIPT_DIR / "Level4_Radiomic_Features_FLAIR.csv"


def load_data(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if 'MGMT_Label' not in df.columns:
        raise KeyError('MGMT_Label column not found in CSV')

    # y is MGMT_Label
    y = df['MGMT_Label'].astype(int).values

    # Keep only numeric columns for X (exclude the target)
    numeric = df.select_dtypes(include=[np.number]).copy()
    if 'MGMT_Label' in numeric.columns:
        numeric = numeric.drop(columns=['MGMT_Label'])

    X = numeric.values

    feature_names = list(numeric.columns)
    return X, y, feature_names


def main():
    X, y, feature_names = load_data(INPUT_CSV)
    n_samples, n_features = X.shape
    print(f"Loaded {n_samples} samples, {n_features} numeric features")

    loo = LeaveOneOut()

    # Prepare models
    lr = LogisticRegression(penalty='l1', C=1.0, solver='saga', max_iter=5000)
    rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)

    if XGBClassifier is None:
        print('Warning: xgboost not available; XGBoost will be skipped')
        xgb = None
    else:
        xgb = XGBClassifier(max_depth=2, n_estimators=50, use_label_encoder=False, eval_metric='logloss', random_state=42)

    preds = {
        'LogisticRegression': [],
        'RandomForest': [],
        'XGBoost': [] if xgb is not None else None,
    }
    truths = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scaling
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # SelectKBest inside loop, adapt k if features fewer than 5
        k = min(5, X_train_s.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_sel = selector.fit_transform(X_train_s, y_train)
        X_test_sel = selector.transform(X_test_s)

        # Fit models
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

        preds['LogisticRegression'].append(prob_lr)
        preds['RandomForest'].append(prob_rf)
        if xgb is not None:
            preds['XGBoost'].append(prob_xgb)

        truths.append(int(y_test[0]))

    truths = np.array(truths)

    # Compute AUCs
    results = {}
    try:
        auc_lr = roc_auc_score(truths, np.array(preds['LogisticRegression']))
    except Exception:
        auc_lr = float('nan')
    results['LogisticRegression'] = auc_lr

    try:
        auc_rf = roc_auc_score(truths, np.array(preds['RandomForest']))
    except Exception:
        auc_rf = float('nan')
    results['RandomForest'] = auc_rf

    if xgb is not None:
        try:
            auc_xgb = roc_auc_score(truths, np.array(preds['XGBoost']))
        except Exception:
            auc_xgb = float('nan')
        results['XGBoost'] = auc_xgb

    # Print results
    for name, auc in results.items():
        print(f"{name} AUC: {auc:.4f}")

    # Identify winner
    winner = max(results.items(), key=lambda kv: (kv[1] if not np.isnan(kv[1]) else -1e9))[0]
    print(f"\nWinner: {winner} (highest AUC)")


if __name__ == '__main__':
    main()
