import os
import sys
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def find_id_col(df):
    candidates = [
        "Patient ID", "PatientID", "Patient_Id", "Patient_ID",
        "patient_id", "patientID", "patient", "Patient", "ID", "id"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: return the first column
    return df.columns[0]


def load_and_merge(radiomics_path, demo_path):
    r = pd.read_csv(radiomics_path)
    d = pd.read_csv(demo_path)

    # find id columns
    id_r = find_id_col(r)
    id_d = find_id_col(d)

    # Clean age column in demographics
    age_cols = [c for c in d.columns if 'age' in c.lower()]
    if 'Age at surgery (years)' in d.columns:
        age_col = 'Age at surgery (years)'
    elif 'Age' in d.columns:
        age_col = 'Age'
    elif age_cols:
        age_col = age_cols[0]
    else:
        age_col = None

    if age_col is None:
        raise ValueError('No age column found in demographics')

    d['Age'] = pd.to_numeric(d[age_col], errors='coerce')
    # merge
    merged = pd.merge(r, d, left_on=id_r, right_on=id_d, how='inner')

    # Ensure MGMT_Status column exists (detect common variants)
    if 'MGMT_Status' not in merged.columns:
        mgmt_cols = [c for c in merged.columns if 'mgmt' in c.lower()]
        if mgmt_cols:
            merged = merged.rename(columns={mgmt_cols[0]: 'MGMT_Status'})

    return merged, id_r, id_d


def prepare_X_y(df, id_cols, drop_labels=True):
    df2 = df.copy()
    # Drop Used_Label if present
    if 'Used_Label' in df2.columns and drop_labels:
        df2 = df2.drop(columns=['Used_Label'])

    # Drop patient id columns
    for c in id_cols:
        if c in df2.columns:
            df2 = df2.drop(columns=[c])

    if 'MGMT_Status' not in df2.columns:
        raise ValueError('MGMT_Status column missing from merged data')

    y = df2['MGMT_Status'].copy()
    X = df2.drop(columns=['MGMT_Status'])

    # Keep numeric columns only
    X = X.select_dtypes(include=[np.number])

    # Drop shape features (tumor size) to focus on texture
    shape_cols = [c for c in X.columns if c.startswith('original_shape_')]
    if shape_cols:
        X = X.drop(columns=shape_cols)

    # Encode y if non-numeric
    if not np.issubdtype(y.dtype, np.number):
        y = LabelEncoder().fit_transform(y)

    # Drop rows with NaNs in X or y
    mask = (~X.isna().any(axis=1)) & (~pd.isna(y))
    X = X.loc[mask].reset_index(drop=True)
    y = pd.Series(y).loc[mask].reset_index(drop=True)

    return X, y


def evaluate(X, y, max_features=8):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    selections = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

        # Separate Age (force inclusion) from radiomics features
        if 'Age' in X_train.columns:
            age_train = X_train['Age'].reset_index(drop=True)
            age_test = X_test['Age'].reset_index(drop=True)
            X_train_rad = X_train.drop(columns=['Age']).reset_index(drop=True)
            X_test_rad = X_test.drop(columns=['Age']).reset_index(drop=True)
        else:
            age_train = None
            age_test = None
            X_train_rad = X_train.reset_index(drop=True)
            X_test_rad = X_test.reset_index(drop=True)

        # Scale radiomics for L1 selection only
        scaler_rad = StandardScaler()
        X_train_rad_s = scaler_rad.fit_transform(X_train_rad)
        X_test_rad_s = scaler_rad.transform(X_test_rad)

        # Logistic L1 selection on radiomics only
        logcv = LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear', scoring='roc_auc', max_iter=2000)
        try:
            logcv.fit(X_train_rad_s, y_train)
            coef_arr = logcv.coef_
            if coef_arr.ndim > 1:
                coefs = np.max(np.abs(coef_arr), axis=0)
            else:
                coefs = np.abs(coef_arr).ravel()
            selected_idx = np.where(coefs != 0)[0].tolist()
        except Exception:
            selected_idx = []

        if len(selected_idx) == 0:
            k = min(max_features, X_train_rad.shape[1])
            skb = SelectKBest(f_classif, k=k).fit(X_train_rad_s, y_train)
            selected_idx = np.where(skb.get_support())[0].tolist()

        # cap number of features
        if len(selected_idx) > max_features:
            if 'coefs' in locals() and len(coefs) == X_train_rad.shape[1] and coefs.sum() > 0:
                sorted_idx = sorted(selected_idx, key=lambda i: -coefs[i])
                selected_idx = sorted_idx[:max_features]
            else:
                selected_idx = selected_idx[:max_features]

        selected_rad_features = list(X_train_rad.columns[selected_idx])

        # Build final training/test sets: selected radiomics + Age (Age always included if present)
        if age_train is not None:
            X_train_final = pd.concat([X_train[selected_rad_features].reset_index(drop=True), age_train.reset_index(drop=True)], axis=1)
            X_test_final = pd.concat([X_test[selected_rad_features].reset_index(drop=True), age_test.reset_index(drop=True)], axis=1)
            sel_list = selected_rad_features + ['Age']
        else:
            X_train_final = X_train[selected_rad_features]
            X_test_final = X_test[selected_rad_features]
            sel_list = selected_rad_features

        selections.append(sel_list)

        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train_final, y_train)
        probs_all = rf.predict_proba(X_test_final)
        if len(np.unique(y_train)) > 2:
            auc = roc_auc_score(y_test, probs_all, multi_class='ovr', average='macro')
        else:
            auc = roc_auc_score(y_test, probs_all[:, 1])

        aucs.append(auc)
        print(f'[Fold {fold}] AUC={auc:.4f} selected={sel_list}')

    return aucs, selections


def report(aucs, selections):
    mean_auc = np.mean(aucs)
    print(f'\nMean ROC-AUC (5 folds): {mean_auc:.4f}\n')

    # stability
    flat = [f for sel in selections for f in sel]
    counts = Counter(flat)
    print('Feature selection counts across folds:')
    for feat, cnt in counts.most_common():
        print(f'- {feat}: {cnt} folds')

    age_count = counts.get('Age', 0)
    print(f'\nAge selected in {age_count} / 5 folds')


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    rad_path = os.path.join(base, 'Level4_Radiomic_Features_Enhancing.csv')
    demo_path = os.path.join(base, 'LUMIERE-Demographics_Pathology.csv')

    if not os.path.exists(rad_path):
        print('Radiomics file not found:', rad_path)
        sys.exit(1)
    if not os.path.exists(demo_path):
        print('Demographics file not found:', demo_path)
        sys.exit(1)

    merged, id_r, id_d = load_and_merge(rad_path, demo_path)
    X, y = prepare_X_y(merged, id_cols=[id_r, id_d])

    print(f'Input rows: {len(X)} features: {X.shape[1]}')

    aucs, selections = evaluate(X, y, max_features=8)
    report(aucs, selections)


if __name__ == '__main__':
    main()

