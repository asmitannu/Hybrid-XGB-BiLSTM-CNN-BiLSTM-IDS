# src/preprocess.py
import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from datetime import datetime

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

COMMON_LABEL_NAMES = ["Attack Type","attack_type","Label","label","attack","target","class","Class","category"]

def is_float_str(s):
    try:
        float(s)
        return True
    except:
        return False

def _find_label_column(df):
    for name in COMMON_LABEL_NAMES:
        if name in df.columns:
            return name
    last_col = df.columns[-1]
    sample_vals = df[last_col].astype(str).head(10).tolist()
    numeric_count = sum(1 for v in sample_vals if is_float_str(v))
    if numeric_count < len(sample_vals):
        return last_col
    for c in df.columns[::-1]:
        vals = df[c].astype(str).head(10).tolist()
        numeric_count = sum(1 for v in vals if is_float_str(v))
        if numeric_count < len(vals):
            return c
    raise ValueError("Could not detect label column. Provide label_col_hint.")

def _read_csv_safe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def _pre_generic_df(df, label_col_hint=None, sample_frac=None):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna(axis=0)
    if sample_frac and 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=SEED)

    if label_col_hint and label_col_hint in df.columns:
        label_col = label_col_hint
    else:
        label_col = _find_label_column(df)

    df.rename(columns={label_col: "Attack Type"}, inplace=True)
    df["Attack Type"] = df["Attack Type"].astype(str).str.strip()

    # drop obvious index/time cols heuristics
    drop_list = [c for c in df.columns if c.lower() in ("flowid","timestamp","id","time","date")]
    if drop_list:
        df = df.drop(columns=[c for c in drop_list if c in df.columns])

    return df

def load_and_preprocess(csv_path, label_col_hint=None, test_size=0.2, sample_frac=None):
    """
    Generic CSV preprocessing (used for CICIDS, UNSW).
    Returns: X_train, X_test, y_train, y_test, scaler, label_encoder, feature_names
    """
    df = _read_csv_safe(csv_path)
    df = _pre_generic_df(df, label_col_hint=label_col_hint, sample_frac=sample_frac)

    X_df = df.drop(columns=["Attack Type"])
    y_raw = df["Attack Type"].values

    # encode object columns
    for col in X_df.select_dtypes(include=['object']).columns:
        X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

    # drop constant columns
    nunique = X_df.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X_df = X_df.drop(columns=const_cols)

    feature_names = X_df.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=SEED, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler, le, feature_names

def align_and_merge(csv_paths, label_col_hint=None, test_size=0.2, sample_frac=None, merge_strategy="intersection"):
    """
    Attempt to align multiple dataset CSVs and merge. Two strategies:
      - intersection: use only features common to all (default)
      - union: take union of features and fill missing with zeros (fallback)
    Returns same format as load_and_preprocess.
    """
    dfs = []
    feature_sets = []
    for p in csv_paths:
        if p is None:
            raise ValueError("Invalid dataset path provided to align_and_merge: None")
        df = _read_csv_safe(p)
        df = _pre_generic_df(df, label_col_hint=label_col_hint, sample_frac=sample_frac)
        dfs.append(df)
        feature_sets.append(set(df.columns) - {"Attack Type"})

    # try intersection first if requested
    common_features = set.intersection(*feature_sets) if merge_strategy == "intersection" else set.union(*feature_sets)
    if not common_features:
        # fallback to union strategy
        common_features = set.union(*feature_sets)
        if not common_features:
            raise ValueError("No features found to merge across datasets.")
        fallback_msg = "[WARN] No shared features across datasets — falling back to UNION strategy and filling missing columns with 0s."
        print(fallback_msg)

    common_features = sorted(list(common_features))

    X_list = []
    y_list = []
    for df in dfs:
        # ensure all common_features present; if missing, fill with zeros
        missing = [c for c in common_features if c not in df.columns]
        df_local = df.copy()
        if missing:
            for m in missing:
                df_local[m] = 0
        # keep only the common feature order
        df_sub = df_local[common_features + ["Attack Type"]].copy()
        # encode object features if any
        for c in df_sub.select_dtypes(include=['object']).columns:
            if c != "Attack Type":
                df_sub[c] = LabelEncoder().fit_transform(df_sub[c].astype(str))
        X_list.append(df_sub[common_features].values)
        y_list.append(df_sub["Attack Type"].astype(str).values)

    X = np.vstack(X_list)
    y_raw = np.concatenate(y_list)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=SEED, stratify=y)
    return X_train, X_test, y_train, y_test, scaler, le, common_features

# NSL-KDD loader wrapper (calls user-provided prepare_nsl_kdd module)
def load_nslkdd_preprocessed(train_path, test_path):
    """
    Expect prepare_nsl_kdd.load_nsl_kdd to return:
      X_train, y_train, X_test, y_test, scaler, y_enc, encoders
    This wrapper converts into the train.py expected return order.
    """
    from .prepare_nsl_kdd import load_nsl_kdd
    X_train, y_train, X_test, y_test, scaler, y_enc, encoders = load_nsl_kdd(train_path, test_path)
    feature_names = [f"f{i}" for i in range(X_train.shape[1])]
    return X_train, X_test, y_train, y_test, scaler, y_enc, feature_names

# WSN-DS loader wrapper (calls prepare_wsnds)
def load_wsnds_preprocessed(csv_path, test_size=0.2):
    from .prepare_wsnds import load_wsnds
    # load_wsnds should return X_train, X_test, y_train, y_test, scaler, le, feature_names
    return load_wsnds(csv_path, test_size=test_size)
