"""
06_merge_age_and_evaluate.py

Merge Age into radiomics and evaluate MGMT prediction using StratifiedKFold
with leakage-safe processing and L1-based feature selection.

Defaults:
- Demographics: `LUMIERE-Demographics_Pathology.csv`
- Radiomics: prefer `Level4_Radiomic_Features_WholeTumor.csv`, then `Level4_Radiomic_Features_FLAIR.csv`,
  then `Level4_Radiomic_Features.csv`, then `Level4_MGMT_Dataset.csv`.

Outputs: prints mean ROC-AUC and feature selection stability (counts), including Age.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


SCRIPT_DIR = Path(__file__).parent
DEMOG_CSV = SCRIPT_DIR / 'LUMIERE-Demographics_Pathology.csv'
PREFERRED_RAD_FILES = [
    SCRIPT_DIR / 'Level4_Radiomic_Features_WholeTumor.csv',
    SCRIPT_DIR / 'Level4_Radiomic_Features_FLAIR.csv',
    SCRIPT_DIR / 'Level4_Radiomic_Features.csv',
    SCRIPT_DIR / 'Level4_MGMT_Dataset.csv'
]


def find_radiomics_file():
    for p in PREFERRED_RAD_FILES:
        if p.exists():
            return p
    raise FileNotFoundError('No radiomics CSV found in Therapy Prediction folder')


def load_and_merge(radiomics_path: Path, demog_path: Path):
    df_rad = pd.read_csv(radiomics_path)
    df_demo = pd.read_csv(demog_path)

    # Find ID columns in demographics: try common names
    demo_id_col = None
    for c in df_demo.columns:
        if c.lower() in ('patient', 'patient_id', 'subject_id', 'subject'):
            demo_id_col = c
            break
    if demo_id_col is None:
        # fallback: first column
        demo_id_col = df_demo.columns[0]

    # Rename demographics id to Patient_ID
    df_demo = df_demo.rename(columns={demo_id_col: 'Patient_ID'})

    # Clean age column
    age_col = None
    for c in df_demo.columns:
        if 'age' in c.lower():
            age_col = c
            break
    if age_col is None:
        raise KeyError('No age column found in demographics file')

    df_demo['Age'] = pd.to_numeric(df_demo[age_col], errors='coerce')

    # Identify target column in radiomics: prefer MGMT_Label or MGMT_Status
    target_col = None
    for name in ('MGMT_Label', 'MGMT_Status', 'MGMT'):
        if name in df_rad.columns:
            target_col = name
            break
    if target_col is None:
        # try lower-case match
        for c in df_rad.columns:
            if 'mgmt' in c.lower():
                target_col = c
                break
    if target_col is None:
        raise KeyError('No MGMT label column found in radiomics CSV')

    # Ensure patient id present in radiomics
    if 'Patient_ID' not in df_rad.columns:
        # try to find a column that looks like patient id
        for c in df_rad.columns:
            if c.lower().startswith('patient') or c.lower().startswith('subject'):
                df_rad = df_rad.rename(columns={c: 'Patient_ID'})
                break

    if 'Patient_ID' not in df_rad.columns:
        raise KeyError('No Patient_ID column found or inferred in radiomics CSV')

    # Inner join on Patient_ID
    df_merged = pd.merge(df_rad, df_demo[['Patient_ID', 'Age']], on='Patient_ID', how='inner')
    if df_merged.empty:
        raise ValueError('No overlapping patients between radiomics and demographics')

    # Drop rows with missing Age or missing target
    df_merged = df_merged.dropna(subset=['Age', target_col])

    # Standardize target name
    df_merged = df_merged.rename(columns={target_col: 'MGMT_Target'})

    return df_merged


def evaluate(df: pd.DataFrame):
    # Prepare X and y
    y = df['MGMT_Target'].astype(int).values

    numeric = df.select_dtypes(include=[np.number]).copy()
    if 'MGMT_Target' in numeric.columns:
        numeric = numeric.drop(columns=['MGMT_Target'])
    # Keep Patient_ID out
    if 'Patient_ID' in numeric.columns:
        numeric = numeric.drop(columns=['Patient_ID'])

    feature_names = list(numeric.columns)
    X = numeric.values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    selection_counter = Counter()

    fold = 0
    for train_idx, test_idx in skf.split(X, y):
        fold += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit scaler on train only
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Fit L1 LogisticRegressionCV on training data only
        l1 = LogisticRegressionCV(penalty='l1', solver='liblinear', scoring='roc_auc', cv=3, max_iter=5000)
        l1.fit(X_train_s, y_train)

        coefs = np.mean(l1.coefs_paths_[1].values() if hasattr(l1, 'coefs_paths_') else l1.coef_, axis=0) if False else l1.coef_.ravel()
        # select non-zero coefficients
        nz = np.where(np.abs(coefs) > 1e-8)[0]
        if nz.size == 0:
            # fallback: pick top 5 by abs coef
            ranked = np.argsort(-np.abs(coefs))
            selected_idx = ranked[:min(5, len(ranked))]
        else:
            selected_idx = nz

        selected_names = [feature_names[i] for i in selected_idx]
        for fn in selected_names:
            selection_counter[fn] += 1

        # Train RF on selected features only
        rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        Xtr_sel = X_train_s[:, selected_idx]
        Xte_sel = X_test_s[:, selected_idx]
        rf.fit(Xtr_sel, y_train)
        probs = rf.predict_proba(Xte_sel)[:, 1]
        try:
            auc = roc_auc_score(y_test, probs)
        except Exception:
            auc = float('nan')
        aucs.append(auc)
        print(f'Fold {fold}: selected {len(selected_idx)} features -> AUC={auc:.4f}')

    mean_auc = np.nanmean(aucs)
    print(f'\nMean ROC-AUC (5-fold): {mean_auc:.4f}')

    # Stability report
    stability = pd.Series(selection_counter)
    stability = stability.reindex(feature_names).fillna(0).astype(int)
    stability = stability[stability > 0].sort_values(ascending=False)
    print('\nFeature selection counts (stability across folds):')
    print(stability)

    # Specifically report Age
    age_count = stability.get('Age', 0)
    print(f"\n'Age' selected in {age_count} out of 5 folds")


def main():
    rad = find_radiomics_file()
    print('Using radiomics file:', rad.name)
    print('Loading demographics:', DEMOG_CSV.name)
    df = load_and_merge(rad, DEMOG_CSV)
    print(f'Merged dataset: {len(df)} patients')
    evaluate(df)


if __name__ == '__main__':
    main()
