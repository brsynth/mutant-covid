import os
import re
import json
from copy import deepcopy
from itertools import product
from functools import reduce
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
)
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier


# =========================================================
# CONFIG
# =========================================================
RUN_ALL_STRAINS = True
STRAINS = ["A0"]

MODE = "M_vs_S"   # "M_vs_S" or "N_vs_P" or "all"

# -------- Feature selection / K tuning --------
USE_FEATURE_SELECTION = True
TUNE_K = True

NB_PARAMS = 7
NB_STRAINS = len(STRAINS)
DEFAULT_FIXED_K = NB_PARAMS * NB_STRAINS
K_RANGE = list(range(2, NB_PARAMS * NB_STRAINS + 1))

# If not tuning K:
FIXED_K = DEFAULT_FIXED_K

# -------- Ensemble --------
USE_ENSEMBLE = True
VOTING = "soft"

# candidate weight values for the VotingClassifier weight search
ENSEMBLE_WEIGHT_VALUES = [1, 2, 3]

# -------- CV --------
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 3
RANDOM_STATE = 2026

# Stable per-model seed offsets
MODEL_SEED_OFFSET = {
    "SVM": 11,
    "LogReg": 23,
    "XGBoost": 37,
    "VotingEnsemble": 53,
}

# =========================================================
# PATHS
# =========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input folders next to this script
BASE_PATH = os.path.join(SCRIPT_DIR, "growth_parameters")
SPLITS_DIR = os.path.join(SCRIPT_DIR, "splits")

# Case-specific output folders
CASE_OUTPUT_DIRS = {
    "M_vs_S": "Mild_vs_severe",
    "N_vs_P": "Negative_vs_positive",
    "all": "Mild_vs_severe_vs_negative",
}

case_output_dirname = CASE_OUTPUT_DIRS[MODE]
CASE_DIR = os.path.join(SCRIPT_DIR, case_output_dirname)

# Feature-selection-specific parent folder
FS_DIRNAME = "With_feature_selection" if USE_FEATURE_SELECTION else "Without_feature_selection"
OUTPUT_ROOT = os.path.join(CASE_DIR, FS_DIRNAME)

RESULTS_DIR = os.path.join(OUTPUT_ROOT, "Results")
RAW_FOLDS_DIR = os.path.join(OUTPUT_ROOT, "Raw_folds")
ROC_DIR = os.path.join(OUTPUT_ROOT, "ROC_curves")
PARAMS_DIR = os.path.join(OUTPUT_ROOT, "Hyperparameters")
PRED_DIR = os.path.join(OUTPUT_ROOT, "Patient_predictions")
CM_DIR = os.path.join(OUTPUT_ROOT, "Confusion_matrices")
SELECTED_FEATURES_DIR = os.path.join(OUTPUT_ROOT, "Selected_features")


# -------- Shared outer split file --------
OUTER_SPLITS_FILE = os.path.join(
    SPLITS_DIR,
    f"kfold_shared_{MODE}_{N_OUTER_SPLITS}folds.json"
)

CLASS_NAME_MAP = {
    "M_vs_S": {0: "Severe", 1: "Mild"},
    "N_vs_P": {0: "Negative", 1: "Positive"},
    "all":    {0: "Negative", 1: "Mild", 2: "Severe"},
}

# =========================================================
# STRAIN DISCOVERY
# =========================================================
def discover_strains(base_path):
    """
    Discover strain names from files like:
        A0 - parameters_v3.xlsx
        A1 - parameters_v3.xlsx
        ...
    """
    pattern = re.compile(r"^(.*?)\s*-\s*parameters_v3\.xlsx$", re.IGNORECASE)

    strains = []
    for fname in os.listdir(base_path):
        m = pattern.match(fname)
        if m:
            strains.append(m.group(1).strip())

    if not strains:
        raise FileNotFoundError(
            f"No files matching '* - parameters_v3.xlsx' found in: {base_path}"
        )

    def natural_key(s):
        parts = re.split(r"(\d+)", s)
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    return sorted(set(strains), key=natural_key)


# =========================================================
# DATA LOADING
# =========================================================
def load_data(strains, mode="M_vs_S", base_path="parameter"):
    if isinstance(strains, str):
        strains = [strains]

    strain_dfs = []

    for strain in strains:
        path = os.path.join(base_path, f"{strain} - parameters_v3.xlsx")

        df1 = pd.read_excel(path, sheet_name="Replicate 1").copy()
        df1["Replicate"] = 1

        df2 = pd.read_excel(path, sheet_name="Replicate 2").copy()
        df2["Replicate"] = 2

        df = pd.concat([df1, df2], ignore_index=True)

        first_col = df.columns[0]
        df[first_col] = df[first_col].astype(str)

        df["Patient"] = df[first_col].str.extract(r"^([SMN]\d+)")[0]
        df["Group"] = df[first_col].str[0]

        if df["Patient"].isna().any():
            bad = df.loc[df["Patient"].isna(), first_col].head(10).tolist()
            raise ValueError(f"Could not parse patient IDs from first column. Examples: {bad}")

        exclude = ["Patient", "Group", "Replicate", first_col]
        feature_cols = [c for c in df.columns if c not in exclude]
        rename_dict = {c: f"{strain}_{c}" for c in feature_cols}

        df = df[["Patient", "Group", "Replicate"] + feature_cols].rename(columns=rename_dict)
        strain_dfs.append(df)

    df_final = reduce(
        lambda left, right: pd.merge(left, right, on=["Patient", "Group", "Replicate"], how="inner"),
        strain_dfs,
    )

    if mode == "M_vs_S":
        df_final = df_final[df_final["Group"].isin(["M", "S"])].copy()
        df_final["GroupCode"] = df_final["Group"].map({"M": 1, "S": 0})

    elif mode == "N_vs_P":
        df_final = df_final[df_final["Group"].isin(["N", "M", "S"])].copy()
        df_final["GroupCode"] = df_final["Group"].map({"N": 0, "M": 1, "S": 1})

    elif mode == "all":
        df_final = df_final[df_final["Group"].isin(["N", "M", "S"])].copy()
        df_final["GroupCode"] = df_final["Group"].map({"N": 0, "M": 1, "S": 2})

    else:
        raise ValueError(f"Unsupported MODE: {mode}")

    # strict patient-label consistency
    patient_label_counts = df_final.groupby("Patient")["GroupCode"].nunique()
    bad = patient_label_counts[patient_label_counts != 1]
    if len(bad) > 0:
        ex = bad.head(10).to_dict()
        raise ValueError(f"Patients with inconsistent labels across replicates found. Examples: {ex}")

    X = df_final.drop(columns=["Patient", "Group", "Replicate", "GroupCode"])
    y = df_final["GroupCode"].astype(int)
    groups = df_final["Patient"].astype(str)

    # strict replicate check: exactly 2 rows per patient
    rep_counts = groups.value_counts()
    bad_rep = rep_counts[rep_counts != 2]
    if len(bad_rep) > 0:
        ex = bad_rep.head(10).to_dict()
        raise ValueError(f"Expected exactly 2 replicates per patient. Examples: {ex}")

    return X, y, groups


# =========================================================
# HELPERS
# =========================================================
def _sanitize_excel_sheet_name(name: str) -> str:
    invalid = ['\\', '/', '*', '[', ']', ':', '?']
    for ch in invalid:
        name = name.replace(ch, "_")
    return name[:31]


def summarize_metric(x):
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=0)),
        "median": float(np.median(x)),
        "p2.5": float(np.percentile(x, 2.5)),
        "p97.5": float(np.percentile(x, 97.5)),
    }


def macro_specificity(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specs = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        den = tn + fp
        specs.append(tn / den if den > 0 else 0.0)
    return float(np.mean(specs))


def get_patient_labels(groups, y):
    """
    Strict rule:
    each patient must have exactly one label across replicates.
    """
    groups = pd.Series(groups).astype(str).reset_index(drop=True)
    y = pd.Series(y).astype(int).reset_index(drop=True)

    out_patients = []
    out_labels = []

    for pid, idx in groups.groupby(groups).groups.items():
        labels = y.iloc[list(idx)].unique()
        if len(labels) != 1:
            raise ValueError(f"Patient {pid} has inconsistent labels across replicates: {labels.tolist()}")
        out_patients.append(pid)
        out_labels.append(int(labels[0]))

    return np.array(out_patients, dtype=str), np.array(out_labels, dtype=int)


def get_patient_class_counts(groups, y):
    patient_ids, patient_labels = get_patient_labels(groups, y)
    return pd.Series(patient_labels).value_counts()


def validate_numeric_features(X):
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if len(non_numeric) > 0:
        raise ValueError(f"Non-numeric feature columns found. Examples: {non_numeric[:10]}")

    if X.isna().any().any():
        na_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Missing values found in features. Examples of columns with NaN: {na_cols[:10]}")


def choose_feasible_n_splits(groups, y, requested_n_splits, context):
    class_counts = get_patient_class_counts(groups, y)
    min_count = int(class_counts.min())

    if min_count < 2:
        raise ValueError(
            f"{context}: not enough patients in at least one class for CV. "
            f"Patient-level class counts: {class_counts.to_dict()}"
        )

    feasible = int(min(requested_n_splits, min_count))
    if feasible < 2:
        raise ValueError(
            f"{context}: feasible n_splits became < 2. "
            f"Patient-level class counts: {class_counts.to_dict()}"
        )
    return feasible


# =========================================================
# SHARED OUTER SPLITS (JSON, patient IDs)
# Compatible with the deep-learning code
# =========================================================
def generate_patient_stratkfold_split_file(path, groups, y, n_splits, random_state):
    """
    Save outer CV splits as patient IDs in a JSON file:
      - each split stores train_patients and test_patients
      - format matches the deep-learning code
    """
    patient_ids, patient_labels = get_patient_labels(groups, y)

    class_counts = pd.Series(patient_labels).value_counts()
    if class_counts.min() < 2:
        raise ValueError(
            "Not enough patients in at least one class for StratifiedKFold. "
            f"Patient-level class counts: {class_counts.to_dict()}"
        )

    if n_splits > int(class_counts.min()):
        raise ValueError(
            f"n_splits={n_splits} is too large for the smallest class "
            f"(min patients in a class={int(class_counts.min())})."
        )

    skf = StratifiedKFold(
        n_splits=int(n_splits),
        shuffle=True,
        random_state=int(random_state),
    )

    splits = []
    for split_id, (tr_pat_idx, te_pat_idx) in enumerate(skf.split(patient_ids, patient_labels)):
        tr_patients = patient_ids[tr_pat_idx]
        te_patients = patient_ids[te_pat_idx]

        overlap = set(tr_patients).intersection(set(te_patients))
        if overlap:
            raise RuntimeError(f"Patient leakage detected in split {split_id}: {sorted(list(overlap))[:10]}")

        splits.append({
            "split_id": int(split_id),
            "seed": int(random_state),
            "test_size": float(len(te_patients) / len(patient_ids)),
            "train_patients": [str(p) for p in tr_patients.tolist()],
            "test_patients": [str(p) for p in te_patients.tolist()],
        })

    payload = {
        "format_version": 1,
        "kind": "StratifiedKFold_patient",
        "n_splits": int(n_splits),
        "base_seed": int(random_state),
        "splits": splits,
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


def load_patient_stratkfold_split_file(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if payload.get("format_version") != 1:
        raise ValueError("Unsupported split file format_version.")
    if payload.get("kind") != "StratifiedKFold_patient":
        raise ValueError(f"Unsupported split kind: {payload.get('kind')}")

    return payload


def _rows_from_patient_split(groups, train_patients, test_patients):
    groups_arr = np.asarray(groups).astype(str)
    train_patients = np.asarray(train_patients).astype(str)
    test_patients = np.asarray(test_patients).astype(str)

    overlap = set(train_patients).intersection(set(test_patients))
    if overlap:
        raise RuntimeError(f"Patient leakage in split file. Overlap examples: {sorted(list(overlap))[:10]}")

    all_dataset_patients = set(np.unique(groups_arr))
    split_patients = set(train_patients).union(set(test_patients))

    missing_from_dataset = split_patients - all_dataset_patients
    if missing_from_dataset:
        raise ValueError(
            "Split file is incompatible with this dataset: some split patients are missing. "
            f"Examples: {sorted(list(missing_from_dataset))[:10]}"
        )

    uncovered_in_split = all_dataset_patients - split_patients
    if uncovered_in_split:
        raise ValueError(
            "Split file does not cover all patients in this dataset. "
            f"Examples: {sorted(list(uncovered_in_split))[:10]}"
        )

    if len(train_patients) != len(set(train_patients)):
        raise ValueError("Duplicate patient IDs found in train_patients.")
    if len(test_patients) != len(set(test_patients)):
        raise ValueError("Duplicate patient IDs found in test_patients.")

    tr_mask = np.isin(groups_arr, train_patients)
    te_mask = np.isin(groups_arr, test_patients)

    tr_rows = np.where(tr_mask)[0]
    te_rows = np.where(te_mask)[0]

    if len(tr_rows) == 0 or len(te_rows) == 0:
        raise ValueError("Split file produced empty train or test rows.")

    if len(set(groups_arr[tr_rows]).intersection(set(groups_arr[te_rows]))) > 0:
        raise RuntimeError("Patient leakage detected between train and test folds.")

    return tr_rows, te_rows


def payload_to_outer_splits(payload, groups):
    """
    Convert JSON patient-level split payload to the row-index format expected by this tabular code:
        [(train_row_idx, test_row_idx), ...]
    """
    outer_splits = []
    for s in payload["splits"]:
        tr_rows, te_rows = _rows_from_patient_split(
            groups=groups,
            train_patients=s["train_patients"],
            test_patients=s["test_patients"],
        )
        outer_splits.append((tr_rows, te_rows))
    return outer_splits


def save_or_load_outer_splits(groups, y, path, n_splits, random_state):
    """
    Shared split loader/creator:
      - creates JSON with train_patients / test_patients if missing
      - loads JSON if present
      - returns row-index splits for this tabular script
    """
    if os.path.exists(path):
        payload = load_patient_stratkfold_split_file(path)
        print(f"[LOADED] Shared outer splits file: {path}")
    else:
        payload = generate_patient_stratkfold_split_file(
            path=path,
            groups=groups,
            y=y,
            n_splits=n_splits,
            random_state=random_state,
        )
        print(f"[SAVED] Shared outer splits file: {path}")

    if int(payload["n_splits"]) != int(n_splits):
        raise ValueError(
            f"Split file n_splits={payload['n_splits']} does not match requested n_splits={n_splits}."
        )

    return payload_to_outer_splits(payload, groups)


# =========================================================
# INNER SPLITS
# =========================================================
def make_patient_stratified_folds(groups, y, n_splits, random_state):
    """
    Returns a list of (train_row_idx, test_row_idx) where the splitting unit is the patient.
    Used for INNER CV.
    """
    patient_ids, patient_labels = get_patient_labels(groups, y)

    class_counts = pd.Series(patient_labels).value_counts()
    if class_counts.min() < n_splits:
        raise ValueError(
            f"Not enough patients in the smallest class for StratifiedKFold with n_splits={n_splits}. "
            f"Patient-level class counts: {class_counts.to_dict()}"
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []

    groups_arr = np.asarray(groups).astype(str)

    for tr_pat_idx, te_pat_idx in skf.split(patient_ids, patient_labels):
        tr_patients = set(patient_ids[tr_pat_idx])
        te_patients = set(patient_ids[te_pat_idx])

        tr_mask = np.isin(groups_arr, list(tr_patients))
        te_mask = np.isin(groups_arr, list(te_patients))

        tr_rows = np.where(tr_mask)[0]
        te_rows = np.where(te_mask)[0]

        if len(set(groups_arr[tr_rows]).intersection(set(groups_arr[te_rows]))) > 0:
            raise RuntimeError("Patient leakage detected between inner train and validation folds.")

        splits.append((tr_rows, te_rows))

    return splits


# =========================================================
# PREDICTION / METRICS
# =========================================================
def aggregate_patient_predictions(y_true_rows, y_proba_rows, patient_rows):
    """
    Average replicate probabilities per patient.
    Returns:
        patient_ids, y_true_patient, y_pred_patient, y_proba_patient
    """
    df = pd.DataFrame({
        "Patient": np.asarray(patient_rows).astype(str),
        "y_true": np.asarray(y_true_rows).astype(int),
    })

    y_proba_rows = np.asarray(y_proba_rows, dtype=float)
    proba_cols = [f"proba_{i}" for i in range(y_proba_rows.shape[1])]
    proba_df = pd.DataFrame(y_proba_rows, columns=proba_cols)

    tmp = pd.concat([df, proba_df], axis=1)

    patient_ids = []
    y_true_patient = []
    y_proba_patient = []

    for pid, sub in tmp.groupby("Patient", sort=True):
        labels = sub["y_true"].unique()
        if len(labels) != 1:
            raise ValueError(f"Patient {pid} has inconsistent labels across replicates: {labels.tolist()}")

        patient_ids.append(pid)
        y_true_patient.append(int(labels[0]))
        y_proba_patient.append(sub[proba_cols].mean(axis=0).to_numpy(dtype=float))

    patient_ids = np.asarray(patient_ids, dtype=str)
    y_true_patient = np.asarray(y_true_patient, dtype=int)
    y_proba_patient = np.vstack(y_proba_patient)
    y_pred_patient = np.argmax(y_proba_patient, axis=1).astype(int)

    return patient_ids, y_true_patient, y_pred_patient, y_proba_patient


def compute_patient_metrics(y_true_patient, y_pred_patient, y_proba_patient, labels):
    out = {}
    out["BalancedAcc"] = float(balanced_accuracy_score(y_true_patient, y_pred_patient))
    out["MacroPrecision"] = float(precision_score(y_true_patient, y_pred_patient, average="macro", zero_division=0))
    out["MacroRecall"] = float(recall_score(y_true_patient, y_pred_patient, average="macro", zero_division=0))
    out["MacroF1"] = float(f1_score(y_true_patient, y_pred_patient, average="macro", zero_division=0))
    out["MacroSpecificity"] = float(macro_specificity(y_true_patient, y_pred_patient, labels=labels))

    n_classes = len(labels)
    if n_classes == 2:
        try:
            out["AUC"] = float(roc_auc_score(y_true_patient, y_proba_patient[:, 1]))
        except Exception:
            out["AUC"] = float("nan")
        out["AUC_per_class"] = {
            0: float(roc_auc_score((y_true_patient == 0).astype(int), y_proba_patient[:, 0]))
               if len(np.unique(y_true_patient == 0)) > 1 else float("nan"),
            1: float(roc_auc_score((y_true_patient == 1).astype(int), y_proba_patient[:, 1]))
               if len(np.unique(y_true_patient == 1)) > 1 else float("nan"),
        }
    else:
        y_bin = label_binarize(y_true_patient, classes=labels)
        try:
            out["AUC"] = float(roc_auc_score(y_bin, y_proba_patient, average="macro", multi_class="ovr"))
        except Exception:
            out["AUC"] = float("nan")

        auc_per_class = {}
        for i, cls in enumerate(labels):
            try:
                auc_per_class[int(cls)] = float(roc_auc_score(y_bin[:, i], y_proba_patient[:, i]))
            except Exception:
                auc_per_class[int(cls)] = float("nan")
        out["AUC_per_class"] = auc_per_class

    return out


def evaluate_estimator_patient_level(estimator, X_eval, y_eval, groups_eval, labels):
    y_proba_rows = estimator.predict_proba(X_eval)
    patient_ids, y_true_p, y_pred_p, y_proba_p = aggregate_patient_predictions(
        y_true_rows=y_eval,
        y_proba_rows=y_proba_rows,
        patient_rows=groups_eval,
    )
    metrics = compute_patient_metrics(y_true_p, y_pred_p, y_proba_p, labels)
    return patient_ids, metrics, y_true_p, y_pred_p, y_proba_p


# =========================================================
# MODELS
# =========================================================
def build_models(n_classes, random_state=RANDOM_STATE):
    is_multiclass = n_classes > 2

    xgb_params = dict(
        eval_metric="mlogloss" if is_multiclass else "logloss",
        random_state=random_state,
        n_estimators=300,
        n_jobs=-1,
    )
    if is_multiclass:
        xgb_params.update(objective="multi:softprob", num_class=n_classes)
    else:
        xgb_params.update(objective="binary:logistic")

    models = {
        "SVM": (
            SVC(class_weight="balanced", probability=True, random_state=random_state),
            {
                "clf__C": [0.1, 1, 10],
                "clf__kernel": ["rbf"],
            },
        ),
        "LogReg": (
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear" if n_classes == 2 else "lbfgs",
                multi_class="auto",
                random_state=random_state,
            ),
            {
                "clf__C": [0.01, 0.1, 1, 10],
            },
        ),
        "XGBoost": (
            XGBClassifier(**xgb_params),
            {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [2, 3, 5],
                "clf__learning_rate": [0.03, 0.1],
            },
        ),
    }
    return models


def build_pipeline(base_estimator, use_feature_selection):
    steps = [("scaler", StandardScaler())]
    if use_feature_selection:
        steps.append(("selector", SelectKBest(score_func=f_classif, k=10)))  # dummy k; replaced in tuning
    steps.append(("clf", base_estimator))
    return Pipeline(steps)


def build_param_grid(base_grid, X_train, use_feature_selection, tune_k, fixed_k):
    grid = deepcopy(base_grid)

    if use_feature_selection:
        n_features = X_train.shape[1]
        if tune_k:
            valid_k = [int(k) for k in K_RANGE if 1 <= int(k) <= n_features]
            if len(valid_k) == 0:
                raise ValueError("No valid K values for SelectKBest.")
            grid["selector__k"] = valid_k
        else:
            kk = int(min(max(1, fixed_k), n_features))
            grid["selector__k"] = [kk]

    return list(ParameterGrid(grid))


# =========================================================
# INNER TUNING
# =========================================================
def inner_cv_score_for_params(
    estimator_template,
    params,
    X_train,
    y_train,
    groups_train,
    labels,
    n_inner_splits,
    seed,
):
    inner_splits = make_patient_stratified_folds(
        groups=groups_train,
        y=y_train,
        n_splits=n_inner_splits,
        random_state=seed,
    )

    scores = []
    for inner_tr_idx, inner_va_idx in inner_splits:
        est = clone(estimator_template)
        est.set_params(**params)
        est.fit(X_train.iloc[inner_tr_idx], y_train.iloc[inner_tr_idx])

        _, metrics, _, _, _ = evaluate_estimator_patient_level(
            estimator=est,
            X_eval=X_train.iloc[inner_va_idx],
            y_eval=y_train.iloc[inner_va_idx].to_numpy(),
            groups_eval=groups_train.iloc[inner_va_idx].to_numpy(),
            labels=labels,
        )
        scores.append(metrics["BalancedAcc"])

    return float(np.mean(scores))


def tune_single_model(
    model_name,
    base_estimator,
    base_grid,
    X_train,
    y_train,
    groups_train,
    labels,
    use_feature_selection,
    tune_k,
    fixed_k,
    n_inner_splits,
    seed,
):
    pipeline = build_pipeline(base_estimator, use_feature_selection=use_feature_selection)
    param_grid = build_param_grid(
        base_grid=base_grid,
        X_train=X_train,
        use_feature_selection=use_feature_selection,
        tune_k=tune_k,
        fixed_k=fixed_k,
    )

    best_score = -np.inf
    best_params = None

    for params in param_grid:
        score = inner_cv_score_for_params(
            estimator_template=pipeline,
            params=params,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            labels=labels,
            n_inner_splits=n_inner_splits,
            seed=seed,
        )
        if score > best_score:
            best_score = score
            best_params = deepcopy(params)

    if best_params is None:
        raise RuntimeError(f"No best params found for model {model_name}")

    final_model = clone(pipeline)
    final_model.set_params(**best_params)
    final_model.fit(X_train, y_train)

    return final_model, best_params, float(best_score)


def generate_ensemble_weight_grid(base_model_names):
    grids = []
    for weights in product(ENSEMBLE_WEIGHT_VALUES, repeat=len(base_model_names)):
        grids.append(tuple(int(w) for w in weights))
    grids = list(dict.fromkeys(grids))
    return grids


def generate_oof_patient_predictions_for_ensemble(
    models,
    X_train,
    y_train,
    groups_train,
    labels,
    use_feature_selection,
    tune_k,
    fixed_k,
    n_inner_splits,
    seed,
):
    """
    Fully leakage-free OOF prediction generation for ensemble weight tuning.

    For each inner fold of the OUTER-train set:
      - tune each base model using only that fold's inner-train partition
      - fit best model on that partition
      - predict probabilities on the held-out inner-valid partition
      - aggregate replicate probabilities to patient-level
      - store one patient-level probability vector per patient and per model

    Returns
    -------
    patient_ids : np.ndarray[str]
    y_true_oof  : np.ndarray[int]
    oof_pred_store : dict[model_name -> np.ndarray of shape (n_patients_outer_train, n_classes)]
    base_fold_param_rows : list[dict]
        bookkeeping of the per-fold params used to generate OOF predictions.
    """
    inner_splits = make_patient_stratified_folds(
        groups=groups_train,
        y=y_train,
        n_splits=n_inner_splits,
        random_state=seed,
    )

    model_names = list(models.keys())
    patient_oof_store = {}
    base_fold_param_rows = []

    for inner_fold, (inner_tr_idx, inner_va_idx) in enumerate(inner_splits):
        X_itr = X_train.iloc[inner_tr_idx].reset_index(drop=True)
        y_itr = y_train.iloc[inner_tr_idx].reset_index(drop=True)
        g_itr = groups_train.iloc[inner_tr_idx].reset_index(drop=True)

        X_iva = X_train.iloc[inner_va_idx].reset_index(drop=True)
        y_iva = y_train.iloc[inner_va_idx].reset_index(drop=True)
        g_iva = groups_train.iloc[inner_va_idx].reset_index(drop=True)

        # This is the extra nesting needed for fully leakage-free OOF generation.
        # We may need fewer splits here if the current inner-train partition is small.
        sub_tune_splits = choose_feasible_n_splits(
            groups=g_itr,
            y=y_itr,
            requested_n_splits=n_inner_splits,
            context=f"OOF ensemble tuning, inner fold {inner_fold}",
        )

        for model_name, (base_estimator, base_grid) in models.items():
            fold_seed = int(seed + 10000 * inner_fold + MODEL_SEED_OFFSET[model_name])

            tuned_model, best_params, best_inner_score = tune_single_model(
                model_name=model_name,
                base_estimator=base_estimator,
                base_grid=base_grid,
                X_train=X_itr,
                y_train=y_itr,
                groups_train=g_itr,
                labels=labels,
                use_feature_selection=use_feature_selection,
                tune_k=tune_k,
                fixed_k=fixed_k,
                n_inner_splits=sub_tune_splits,
                seed=fold_seed,
            )

            base_fold_param_rows.append({
                "Stage": "OOF_for_ensemble",
                "InnerFold": int(inner_fold),
                "Model": model_name,
                "BestScore_InnerCV_BalAcc": float(best_inner_score),
                "SelectedK": best_params.get("selector__k", np.nan),
                **best_params,
            })

            y_proba_rows = tuned_model.predict_proba(X_iva)

            patient_ids, y_true_p, _, y_proba_p = aggregate_patient_predictions(
                y_true_rows=y_iva.to_numpy(),
                y_proba_rows=y_proba_rows,
                patient_rows=g_iva.to_numpy(),
            )

            for pid, yt, yp in zip(patient_ids, y_true_p, y_proba_p):
                pid = str(pid)

                if pid not in patient_oof_store:
                    patient_oof_store[pid] = {"y_true": int(yt)}
                else:
                    if int(patient_oof_store[pid]["y_true"]) != int(yt):
                        raise ValueError(
                            f"Inconsistent OOF patient label for patient {pid}: "
                            f"{patient_oof_store[pid]['y_true']} vs {yt}"
                        )

                if model_name in patient_oof_store[pid]:
                    raise RuntimeError(
                        f"Duplicate OOF prediction encountered for patient {pid}, model {model_name}. "
                        "Each patient should appear in validation exactly once per model."
                    )

                patient_oof_store[pid][model_name] = np.asarray(yp, dtype=float)

    patient_ids_sorted = np.array(sorted(patient_oof_store.keys()), dtype=str)
    y_true_oof = []
    oof_pred_store = {m: [] for m in model_names}

    for pid in patient_ids_sorted:
        row = patient_oof_store[pid]

        if "y_true" not in row:
            raise RuntimeError(f"Missing y_true for patient {pid} in OOF store.")

        missing_models = [m for m in model_names if m not in row]
        if len(missing_models) > 0:
            raise RuntimeError(
                f"Missing OOF predictions for patient {pid} from models: {missing_models}"
            )

        y_true_oof.append(int(row["y_true"]))
        for m in model_names:
            oof_pred_store[m].append(np.asarray(row[m], dtype=float))

    y_true_oof = np.asarray(y_true_oof, dtype=int)
    for m in model_names:
        oof_pred_store[m] = np.vstack(oof_pred_store[m])

    expected_patients = set(np.unique(groups_train.astype(str)))
    got_patients = set(patient_ids_sorted.tolist())
    if expected_patients != got_patients:
        missing = sorted(list(expected_patients - got_patients))
        extra = sorted(list(got_patients - expected_patients))
        raise RuntimeError(
            f"OOF patient coverage mismatch. Missing: {missing[:10]}, Extra: {extra[:10]}"
        )

    return patient_ids_sorted, y_true_oof, oof_pred_store, base_fold_param_rows


def tune_ensemble_weights_from_oof(
    y_true_oof,
    oof_pred_store,
    labels,
):
    """
    Tune ensemble weights using only patient-level OOF predictions.
    This is leakage-free because every OOF prediction is out-of-sample.
    """
    model_names = list(oof_pred_store.keys())
    candidate_weights = generate_ensemble_weight_grid(model_names)

    best_score = -np.inf
    best_weights = None

    template = next(iter(oof_pred_store.values()))

    for weights in candidate_weights:
        weights = np.asarray(weights, dtype=float)
        if np.sum(weights) <= 0:
            continue

        combined = np.zeros_like(template, dtype=float)
        for w, m in zip(weights, model_names):
            combined += float(w) * oof_pred_store[m]
        combined /= float(np.sum(weights))

        y_pred_oof = np.argmax(combined, axis=1).astype(int)
        score = float(balanced_accuracy_score(y_true_oof, y_pred_oof))

        if score > best_score:
            best_score = score
            best_weights = tuple(int(w) for w in weights.tolist())

    if best_weights is None:
        raise RuntimeError("No best ensemble weights found from OOF predictions.")

    return best_weights, float(best_score)


def fit_final_voting_ensemble(tuned_base_models, best_weights):
    """
    Build final soft-voting ensemble using base models already tuned/refit on full outer-train.
    """
    final_ensemble = VotingClassifier(
        estimators=[(name, clone(model)) for name, model in tuned_base_models.items()],
        voting=VOTING,
        weights=list(best_weights),
        n_jobs=None,
    )
    return final_ensemble


# =========================================================
# OUTPUT HELPERS
# =========================================================
def save_pooled_confusion(cm_output_file, pooled_store, labels, label_names):
    with pd.ExcelWriter(cm_output_file, engine="openpyxl") as writer:
        for model_name, item in pooled_store.items():
            y_true_all = np.asarray(item["y_true"], dtype=int)
            y_pred_all = np.asarray(item["y_pred"], dtype=int)

            cm = confusion_matrix(y_true_all, y_pred_all, labels=labels).astype(int)
            cm_df = pd.DataFrame(
                cm,
                index=[f"True_{label_names.get(int(c), c)}" for c in labels],
                columns=[f"Pred_{label_names.get(int(c), c)}" for c in labels],
            )
            cm_df.to_excel(writer, sheet_name=_sanitize_excel_sheet_name(f"{model_name}_counts"), index=True)

            row_sum = cm.sum(axis=1).astype(float)
            row_sum[row_sum == 0] = 1.0
            cm_norm = cm.astype(float) / row_sum[:, None]
            cm_norm_df = pd.DataFrame(
                cm_norm,
                index=[f"True_{label_names.get(int(c), c)}" for c in labels],
                columns=[f"Pred_{label_names.get(int(c), c)}" for c in labels],
            )
            cm_norm_df.to_excel(writer, sheet_name=_sanitize_excel_sheet_name(f"{model_name}_norm"), index=True)


def save_pooled_roc(roc_plot_file, pooled_store, labels, label_names, mode):
    plt.figure(figsize=(10, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", alpha=0.5, label="Chance")

    n_classes = len(labels)

    for model_name, item in pooled_store.items():
        y_true_all = np.asarray(item["y_true"], dtype=int)
        y_proba_all = np.asarray(item["y_proba"], dtype=float)

        if y_proba_all.size == 0:
            continue

        if n_classes == 2:
            try:
                fpr, tpr, _ = roc_curve(y_true_all, y_proba_all[:, 1], pos_label=labels[1])
                auc_val = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{model_name} (pooled AUC={auc_val:.3f})")
            except Exception:
                pass
        else:
            y_bin = label_binarize(y_true_all, classes=labels)
            try:
                pooled_macro_auc = roc_auc_score(y_bin, y_proba_all, average="macro", multi_class="ovr")
            except Exception:
                pooled_macro_auc = np.nan

            for i, cls in enumerate(labels):
                try:
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba_all[:, i])
                    auc_i = auc(fpr, tpr)

                    if i == 0:
                        curve_label = (
                            f"{model_name} (pooled macro AUC={pooled_macro_auc:.3f}) | "
                            f"{label_names.get(int(cls), str(cls))} (AUC={auc_i:.3f})"
                        )
                    else:
                        curve_label = f"{model_name} {label_names.get(int(cls), str(cls))} (AUC={auc_i:.3f})"

                    plt.plot(fpr, tpr, alpha=0.6, label=curve_label)
                except Exception:
                    continue

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Pooled outer-test ROC curves - {mode}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(roc_plot_file, dpi=300)
    plt.close()


def write_best_params_workbook(best_params_rows, output_path):
    bp_df = pd.DataFrame(best_params_rows)

    if bp_df.empty:
        return

    param_cols = [
        c for c in bp_df.columns
        if c not in {
            "OuterFold",
            "InnerFold",
            "Stage",
            "Model",
            "BestScore_InnerCV_BalAcc",
            "SelectedK",
        }
    ]

    summary_rows = []
    for model_name in bp_df["Model"].unique():
        sub = bp_df[bp_df["Model"] == model_name].copy()
        n = len(sub)
        for col in param_cols:
            counts = sub[col].astype(str).value_counts(dropna=False)
            for val, cnt in counts.items():
                summary_rows.append({
                    "Model": model_name,
                    "Param": col,
                    "Value": val,
                    "Count": int(cnt),
                    "Rate": float(cnt / n) if n > 0 else 0.0,
                })

    summary_df = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        bp_df.to_excel(writer, sheet_name="All", index=False)
        if not summary_df.empty:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        for model_name in bp_df["Model"].unique():
            bp_df[bp_df["Model"] == model_name].to_excel(
                writer,
                sheet_name=_sanitize_excel_sheet_name(model_name),
                index=False
            )


def extract_selected_features_info(estimator, original_feature_names):
    """
    Extract selected feature names and, when available, their univariate selection
    scores/p-values from a fitted Pipeline.

    Returns a list of dicts:
        [
            {
                "FeatureRank": 1,
                "FeatureName": "...",
                "FeatureScore": ...,
                "FeaturePValue": ...,
                "Selected": True,
            },
            ...
        ]

    If no selector is present, all original features are returned with NaN score/p-value.
    """
    original_feature_names = list(original_feature_names)

    # No selector -> keep all features
    if not (hasattr(estimator, "named_steps") and "selector" in estimator.named_steps):
        rows = []
        for rank, feat in enumerate(original_feature_names, start=1):
            rows.append({
                "FeatureRank": int(rank),
                "FeatureName": str(feat),
                "FeatureScore": np.nan,
                "FeaturePValue": np.nan,
                "Selected": True,
            })
        return rows

    selector = estimator.named_steps["selector"]
    support = selector.get_support()

    scores = getattr(selector, "scores_", None)
    pvalues = getattr(selector, "pvalues_", None)

    selected_rows = []
    for feat, keep, idx in zip(original_feature_names, support, range(len(original_feature_names))):
        if not keep:
            continue

        score_val = np.nan
        pval_val = np.nan

        if scores is not None and idx < len(scores):
            score_val = float(scores[idx]) if pd.notna(scores[idx]) else np.nan

        if pvalues is not None and idx < len(pvalues):
            pval_val = float(pvalues[idx]) if pd.notna(pvalues[idx]) else np.nan

        selected_rows.append({
            "FeatureName": str(feat),
            "FeatureScore": score_val,
            "FeaturePValue": pval_val,
            "Selected": True,
        })

    # Sort selected features by descending score when available
    def _sort_key(d):
        score = d["FeatureScore"]
        if pd.isna(score):
            return -np.inf
        return score

    selected_rows = sorted(selected_rows, key=_sort_key, reverse=True)

    for rank, row in enumerate(selected_rows, start=1):
        row["FeatureRank"] = int(rank)

    return selected_rows


def make_selected_features_rows(
    outer_fold,
    model_name,
    estimator,
    original_feature_names,
    applied_to="outer_test",
):
    """
    Build long-format rows for selected features of one fitted estimator.
    """
    feat_info = extract_selected_features_info(estimator, original_feature_names)
    n_selected = len(feat_info)

    rows = []
    for row in feat_info:
        rows.append({
            "OuterFold": int(outer_fold),
            "Model": str(model_name),
            "AppliedTo": str(applied_to),
            "n_selected_features": int(n_selected),
            "FeatureRank": int(row["FeatureRank"]),
            "FeatureName": str(row["FeatureName"]),
            "FeatureScore": row["FeatureScore"],
            "FeaturePValue": row["FeaturePValue"],
        })
    return rows


def save_selected_features_workbook(selected_feature_rows, output_path):
    sf_df = pd.DataFrame(selected_feature_rows)

    if sf_df.empty:
        return

    # Summary: one row per fold/model/applied_to with concatenated feature names
    summary_df = (
        sf_df.sort_values(["OuterFold", "Model", "AppliedTo", "FeatureRank"])
        .groupby(["OuterFold", "Model", "AppliedTo", "n_selected_features"], as_index=False)
        .agg(
            SelectedFeatures=("FeatureName", lambda x: " | ".join(map(str, x))),
            MeanFeatureScore=("FeatureScore", lambda x: float(np.nanmean(x)) if len(x) > 0 else np.nan),
            MinFeaturePValue=("FeaturePValue", lambda x: float(np.nanmin(x)) if len(x) > 0 else np.nan),
        )
    )

    # Frequency across folds for each model-feature
    freq_df = (
        sf_df.groupby(["Model", "FeatureName"], as_index=False)
        .agg(
            TimesSelected=("FeatureName", "count"),
            MeanRank=("FeatureRank", "mean"),
            MeanScore=("FeatureScore", "mean"),
            MeanPValue=("FeaturePValue", "mean"),
        )
        .sort_values(["Model", "TimesSelected", "MeanRank"], ascending=[True, False, True])
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        sf_df.to_excel(writer, sheet_name="Long", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        freq_df.to_excel(writer, sheet_name="Frequency", index=False)

        for model_name in sf_df["Model"].unique():
            sub = sf_df[sf_df["Model"] == model_name].sort_values(
                ["OuterFold", "AppliedTo", "FeatureRank"]
            )
            sub.to_excel(
                writer,
                sheet_name=_sanitize_excel_sheet_name(model_name),
                index=False
            )


def make_patient_prediction_rows(
    outer_fold,
    model_name,
    patient_ids,
    y_true_patient,
    y_pred_patient,
    y_proba_patient,
    label_names,
):
    rows = []
    n_classes = y_proba_patient.shape[1]

    for i, pid in enumerate(patient_ids):
        row = {
            "OuterFold": int(outer_fold),
            "Model": str(model_name),
            "Patient": str(pid),
            "y_true_code": int(y_true_patient[i]),
            "y_true_label": label_names.get(int(y_true_patient[i]), str(y_true_patient[i])),
            "y_pred_code": int(y_pred_patient[i]),
            "y_pred_label": label_names.get(int(y_pred_patient[i]), str(y_pred_patient[i])),
            "Correct": int(y_true_patient[i] == y_pred_patient[i]),
        }

        for cls_idx in range(n_classes):
            row[f"proba_{cls_idx}"] = float(y_proba_patient[i, cls_idx])
            row[f"proba_{label_names.get(int(cls_idx), str(cls_idx))}"] = float(y_proba_patient[i, cls_idx])

        rows.append(row)

    return rows


def save_patient_predictions_workbook(patient_prediction_rows, output_path):
    pp_df = pd.DataFrame(patient_prediction_rows)

    if pp_df.empty:
        return

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pp_df.to_excel(writer, sheet_name="All", index=False)

        for model_name in pp_df["Model"].unique():
            sub = pp_df[pp_df["Model"] == model_name].sort_values(["OuterFold", "Patient"])
            sub.to_excel(
                writer,
                sheet_name=_sanitize_excel_sheet_name(model_name),
                index=False
            )


# =========================================================
# MAIN EXPERIMENT
# =========================================================
def run_nested_experiment(
    X,
    y,
    groups,
    models,
    outer_splits,
    use_feature_selection,
    tune_k,
    fixed_k,
    use_ensemble,
    n_inner_splits,
    random_state,
):
    labels = np.sort(np.unique(y))
    label_names = CLASS_NAME_MAP.get(MODE, {int(l): str(l) for l in labels})

    base_model_names = list(models.keys())
    all_model_names = base_model_names + (["VotingEnsemble"] if use_ensemble else [])

    fold_metrics_store = defaultdict(list)
    pooled_store = {
        name: {"y_true": [], "y_pred": [], "y_proba": []}
        for name in all_model_names
    }

    best_params_rows = []
    selected_feature_rows = []
    patient_prediction_rows = []

    for outer_fold, (tr_idx, te_idx) in enumerate(outer_splits):
        print(f"\n[OUTER FOLD {outer_fold+1}/{len(outer_splits)}]")

        X_train = X.iloc[tr_idx].reset_index(drop=True)
        y_train = y.iloc[tr_idx].reset_index(drop=True)
        groups_train = groups.iloc[tr_idx].reset_index(drop=True)

        X_test = X.iloc[te_idx].reset_index(drop=True)
        y_test = y.iloc[te_idx].reset_index(drop=True)
        groups_test = groups.iloc[te_idx].reset_index(drop=True)

        tuned_base_models = {}
        tuned_base_params = {}
        tuned_base_scores = {}

        # -------------------------
        # Tune each base model on full outer-train only
        # -------------------------
        full_outer_train_inner_splits = choose_feasible_n_splits(
            groups=groups_train,
            y=y_train,
            requested_n_splits=n_inner_splits,
            context=f"Outer fold {outer_fold}: base-model tuning on full outer-train",
        )

        for model_name, (base_estimator, base_grid) in models.items():
            print(f"  - Tuning {model_name} ...")
            model_seed = random_state + outer_fold * 1000 + MODEL_SEED_OFFSET[model_name]

            best_model, best_params, best_inner_score = tune_single_model(
                model_name=model_name,
                base_estimator=base_estimator,
                base_grid=base_grid,
                X_train=X_train,
                y_train=y_train,
                groups_train=groups_train,
                labels=labels,
                use_feature_selection=use_feature_selection,
                tune_k=tune_k,
                fixed_k=fixed_k,
                n_inner_splits=full_outer_train_inner_splits,
                seed=model_seed,
            )

            tuned_base_models[model_name] = best_model
            tuned_base_params[model_name] = best_params
            tuned_base_scores[model_name] = best_inner_score

            # Selected features learned on outer-train and applied to outer-test
            if use_feature_selection:
                selected_feature_rows.extend(
                    make_selected_features_rows(
                        outer_fold=outer_fold,
                        model_name=model_name,
                        estimator=best_model,
                        original_feature_names=X_train.columns,
                        applied_to="outer_test",
                    )
                )

            selected_k = best_params.get("selector__k", np.nan)

            row = {
                "Stage": "Final_outer_train_refit",
                "OuterFold": int(outer_fold),
                "Model": model_name,
                "BestScore_InnerCV_BalAcc": float(best_inner_score),
                "SelectedK": selected_k,
            }
            row.update(best_params)
            best_params_rows.append(row)

            patient_ids_p, metrics, y_true_p, y_pred_p, y_proba_p = evaluate_estimator_patient_level(
                estimator=best_model,
                X_eval=X_test,
                y_eval=y_test.to_numpy(),
                groups_eval=groups_test.to_numpy(),
                labels=labels,
            )

            patient_prediction_rows.extend(
                make_patient_prediction_rows(
                    outer_fold=outer_fold,
                    model_name=model_name,
                    patient_ids=patient_ids_p,
                    y_true_patient=y_true_p,
                    y_pred_patient=y_pred_p,
                    y_proba_patient=y_proba_p,
                    label_names=label_names,
                )
            )

            fold_row = {
                "OuterFold": int(outer_fold),
                "Model": model_name,
                "BestScore_InnerCV_BalAcc": float(best_inner_score),
                "SelectedK": selected_k,
                "Balanced accuracy": float(metrics["BalancedAcc"]),
                "Macro precision": float(metrics["MacroPrecision"]),
                "Macro recall": float(metrics["MacroRecall"]),
                "Macro F1": float(metrics["MacroF1"]),
                "Macro specificity": float(metrics["MacroSpecificity"]),
                "AUC": float(metrics["AUC"]),
                "n_patients_train_outer": int(len(np.unique(groups_train))),
                "n_patients_test_outer": int(len(np.unique(groups_test))),
            }

            for cls_idx, aucv in metrics["AUC_per_class"].items():
                fold_row[f"AUC_{label_names.get(int(cls_idx), f'Class{cls_idx}')}"] = aucv

            fold_metrics_store[model_name].append(fold_row)

            pooled_store[model_name]["y_true"].append(y_true_p)
            pooled_store[model_name]["y_pred"].append(y_pred_p)
            pooled_store[model_name]["y_proba"].append(y_proba_p)

        # -------------------------
        # Leakage-free ensemble tuning with OOF predictions
        # -------------------------
        if use_ensemble:
            print("  - Generating OOF predictions for leakage-free VotingEnsemble tuning ...")
            ens_seed = random_state + outer_fold * 1000 + MODEL_SEED_OFFSET["VotingEnsemble"]

            oof_splits_for_weights = choose_feasible_n_splits(
                groups=groups_train,
                y=y_train,
                requested_n_splits=n_inner_splits,
                context=f"Outer fold {outer_fold}: OOF ensemble-weight tuning",
            )

            (
                oof_patient_ids,
                y_true_oof,
                oof_pred_store,
                oof_param_rows,
            ) = generate_oof_patient_predictions_for_ensemble(
                models=models,
                X_train=X_train,
                y_train=y_train,
                groups_train=groups_train,
                labels=labels,
                use_feature_selection=use_feature_selection,
                tune_k=tune_k,
                fixed_k=fixed_k,
                n_inner_splits=oof_splits_for_weights,
                seed=ens_seed,
            )

            for row in oof_param_rows:
                row["OuterFold"] = int(outer_fold)
                best_params_rows.append(row)

            best_weights, best_ens_score = tune_ensemble_weights_from_oof(
                y_true_oof=y_true_oof,
                oof_pred_store=oof_pred_store,
                labels=labels,
            )

            ens_row = {
                "Stage": "OOF_weight_tuning",
                "OuterFold": int(outer_fold),
                "Model": "VotingEnsemble",
                "BestScore_InnerCV_BalAcc": float(best_ens_score),
                "SelectedK": np.nan,
                "weights": str(best_weights),
                "n_patients_oof": int(len(oof_patient_ids)),
            }
            for base_name, base_params in tuned_base_params.items():
                for pk, pv in base_params.items():
                    ens_row[f"{base_name}:{pk}"] = pv
            best_params_rows.append(ens_row)

            ensemble_model = fit_final_voting_ensemble(
                tuned_base_models=tuned_base_models,
                best_weights=best_weights,
            )
            ensemble_model.fit(X_train, y_train)

            # Extract selected features for each fitted sub-estimator inside the ensemble
            if use_feature_selection and hasattr(ensemble_model, "named_estimators_"):
                for sub_name, sub_est in ensemble_model.named_estimators_.items():
                    selected_feature_rows.extend(
                        make_selected_features_rows(
                            outer_fold=outer_fold,
                            model_name=f"VotingEnsemble_{sub_name}",
                            estimator=sub_est,
                            original_feature_names=X_train.columns,
                            applied_to="outer_test",
                        )
                    )

            patient_ids_p, metrics, y_true_p, y_pred_p, y_proba_p = evaluate_estimator_patient_level(
                estimator=ensemble_model,
                X_eval=X_test,
                y_eval=y_test.to_numpy(),
                groups_eval=groups_test.to_numpy(),
                labels=labels,
            )

            patient_prediction_rows.extend(
                make_patient_prediction_rows(
                    outer_fold=outer_fold,
                    model_name="VotingEnsemble",
                    patient_ids=patient_ids_p,
                    y_true_patient=y_true_p,
                    y_pred_patient=y_pred_p,
                    y_proba_patient=y_proba_p,
                    label_names=label_names,
                )
            )

            fold_row = {
                "OuterFold": int(outer_fold),
                "Model": "VotingEnsemble",
                "BestScore_InnerCV_BalAcc": float(best_ens_score),
                "SelectedK": np.nan,
                "weights": str(best_weights),
                "Balanced accuracy": float(metrics["BalancedAcc"]),
                "Macro precision": float(metrics["MacroPrecision"]),
                "Macro recall": float(metrics["MacroRecall"]),
                "Macro F1": float(metrics["MacroF1"]),
                "Macro specificity": float(metrics["MacroSpecificity"]),
                "AUC": float(metrics["AUC"]),
                "n_patients_train_outer": int(len(np.unique(groups_train))),
                "n_patients_test_outer": int(len(np.unique(groups_test))),
            }

            for cls_idx, aucv in metrics["AUC_per_class"].items():
                fold_row[f"AUC_{label_names.get(int(cls_idx), f'Class{cls_idx}')}"] = aucv

            fold_metrics_store["VotingEnsemble"].append(fold_row)

            pooled_store["VotingEnsemble"]["y_true"].append(y_true_p)
            pooled_store["VotingEnsemble"]["y_pred"].append(y_pred_p)
            pooled_store["VotingEnsemble"]["y_proba"].append(y_proba_p)

    # -----------------------------------------
    # Concatenate pooled patient predictions
    # -----------------------------------------
    for model_name in pooled_store:
        pooled_store[model_name]["y_true"] = (
            np.concatenate(pooled_store[model_name]["y_true"], axis=0)
            if len(pooled_store[model_name]["y_true"]) > 0 else np.array([], dtype=int)
        )
        pooled_store[model_name]["y_pred"] = (
            np.concatenate(pooled_store[model_name]["y_pred"], axis=0)
            if len(pooled_store[model_name]["y_pred"]) > 0 else np.array([], dtype=int)
        )
        pooled_store[model_name]["y_proba"] = (
            np.concatenate(pooled_store[model_name]["y_proba"], axis=0)
            if len(pooled_store[model_name]["y_proba"]) > 0 else np.zeros((0, len(labels)))
        )

    # -----------------------------------------
    # Build raw folds dataframe
    # -----------------------------------------
    raw_rows = []
    for model_name, rows in fold_metrics_store.items():
        raw_rows.extend(rows)
    raw_folds_df = pd.DataFrame(raw_rows)

    # -----------------------------------------
    # Build summary dataframe
    # -----------------------------------------
    summary_rows = []
    metric_map = {
        "Balanced accuracy": "Balanced accuracy",
        "Macro precision": "Macro precision",
        "Macro recall": "Macro recall",
        "Macro F1": "Macro F1",
        "Macro specificity": "Macro specificity",
        "AUC": "AUC",
    }

    for model_name in raw_folds_df["Model"].unique():
        sub = raw_folds_df[raw_folds_df["Model"] == model_name].copy()

        row = {
            "Model": model_name,
            "Outer folds": int(len(sub)),
        }

        for col, pretty in metric_map.items():
            stats = summarize_metric(sub[col].to_numpy(dtype=float))
            row[f"{pretty} mean (across folds)"] = stats["mean"]
            row[f"{pretty} SD (across folds)"] = stats["sd"]
            row[f"{pretty} median (across folds)"] = stats["median"]
            row[f"{pretty} p2.5 (across folds)"] = stats["p2.5"]
            row[f"{pretty} p97.5 (across folds)"] = stats["p97.5"]

        y_true_all = pooled_store[model_name]["y_true"]
        y_proba_all = pooled_store[model_name]["y_proba"]

        if len(labels) == 2:
            try:
                pooled_auc = float(roc_auc_score(y_true_all, y_proba_all[:, 1]))
            except Exception:
                pooled_auc = float("nan")
        else:
            try:
                y_bin = label_binarize(y_true_all, classes=labels)
                pooled_auc = float(roc_auc_score(y_bin, y_proba_all, average="macro", multi_class="ovr"))
            except Exception:
                pooled_auc = float("nan")

        row["AUC pooled outer-test"] = pooled_auc
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    return (
        summary_df,
        raw_folds_df,
        pooled_store,
        best_params_rows,
        selected_feature_rows,
        patient_prediction_rows,
    )


# =========================================================
# SAVE OUTPUTS
# =========================================================
def save_results_workbook(results_xlsx, summary_df, raw_folds_df):
    mean_sd_cols = [
        "Model",
        "Balanced accuracy mean (across folds)",
        "Balanced accuracy SD (across folds)",
        "Macro precision mean (across folds)",
        "Macro precision SD (across folds)",
        "Macro recall mean (across folds)",
        "Macro recall SD (across folds)",
        "Macro F1 mean (across folds)",
        "Macro F1 SD (across folds)",
        "Macro specificity mean (across folds)",
        "Macro specificity SD (across folds)",
        "AUC mean (across folds)",
        "AUC SD (across folds)",
        "AUC pooled outer-test",
    ]
    mean_sd_cols = [c for c in mean_sd_cols if c in summary_df.columns]
    mean_sd_df = summary_df[mean_sd_cols].copy()

    with pd.ExcelWriter(results_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        mean_sd_df.to_excel(writer, sheet_name="Mean_SD_only", index=False)
        raw_folds_df.to_excel(writer, sheet_name="RawFolds", index=False)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    strain_runs = [[s] for s in discover_strains(BASE_PATH)] if RUN_ALL_STRAINS else [STRAINS]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RAW_FOLDS_DIR, exist_ok=True)
    os.makedirs(ROC_DIR, exist_ok=True)
    os.makedirs(PARAMS_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(CM_DIR, exist_ok=True)
    if USE_FEATURE_SELECTION:
        os.makedirs(SELECTED_FEATURES_DIR, exist_ok=True)

    for STRAINS in strain_runs:
        TAG = f"{MODE}_{'_'.join(STRAINS)}"

        NB_STRAINS = len(STRAINS)
        DEFAULT_FIXED_K = NB_PARAMS * NB_STRAINS
        K_RANGE = list(range(2, NB_PARAMS * NB_STRAINS + 1))
        FIXED_K = DEFAULT_FIXED_K

        RESULTS_XLSX = os.path.join(RESULTS_DIR, f"results_{TAG}.xlsx")
        RAW_FOLDS_XLSX = os.path.join(RAW_FOLDS_DIR, f"raw_folds_{TAG}.xlsx")
        BEST_PARAMS_XLSX = os.path.join(PARAMS_DIR, f"best_params_{TAG}.xlsx")
        CONFUSION_XLSX = os.path.join(CM_DIR, f"confusion_{TAG}.xlsx")
        ROC_PNG = os.path.join(ROC_DIR, f"roc_{TAG}.png")
        PATIENT_PREDICTIONS_XLSX = os.path.join(PRED_DIR, f"patient_predictions_{TAG}.xlsx")
        SELECTED_FEATURES_XLSX = os.path.join(SELECTED_FEATURES_DIR, f"selected_features_{TAG}.xlsx")

        print("\n" + "=" * 70)
        print(f"RUNNING STRAIN(S): {STRAINS}")
        print("=" * 70)

        X, y, groups = load_data(STRAINS, mode=MODE, base_path=BASE_PATH)
        validate_numeric_features(X)

        labels = np.sort(np.unique(y))
        n_classes = len(labels)

        print(f"Samples (replicates): {len(X)}")
        print(f"Patients:             {groups.nunique()}")
        print(f"Features:             {X.shape[1]}")
        print(f"Classes:              {labels.tolist()}")

        models = build_models(n_classes=n_classes, random_state=RANDOM_STATE)

        outer_splits = save_or_load_outer_splits(
            groups=groups,
            y=y,
            path=OUTER_SPLITS_FILE,
            n_splits=N_OUTER_SPLITS,
            random_state=RANDOM_STATE,
        )

        (
            summary_df,
            raw_folds_df,
            pooled_store,
            best_params_rows,
            selected_feature_rows,
            patient_prediction_rows,
        ) = run_nested_experiment(
            X=X,
            y=y,
            groups=groups,
            models=models,
            outer_splits=outer_splits,
            use_feature_selection=USE_FEATURE_SELECTION,
            tune_k=TUNE_K,
            fixed_k=FIXED_K,
            use_ensemble=USE_ENSEMBLE,
            n_inner_splits=N_INNER_SPLITS,
            random_state=RANDOM_STATE,
        )

        label_names = CLASS_NAME_MAP.get(MODE, {int(l): str(l) for l in labels})

        save_results_workbook(
            results_xlsx=RESULTS_XLSX,
            summary_df=summary_df,
            raw_folds_df=raw_folds_df,
        )
        print(f"[SAVED] Results workbook: {RESULTS_XLSX}")

        raw_folds_df.to_excel(RAW_FOLDS_XLSX, index=False)
        print(f"[SAVED] Raw folds: {RAW_FOLDS_XLSX}")

        write_best_params_workbook(best_params_rows, BEST_PARAMS_XLSX)
        print(f"[SAVED] Best params workbook: {BEST_PARAMS_XLSX}")

        save_patient_predictions_workbook(patient_prediction_rows, PATIENT_PREDICTIONS_XLSX)
        print(f"[SAVED] Patient predictions workbook: {PATIENT_PREDICTIONS_XLSX}")

        if USE_FEATURE_SELECTION:
            save_selected_features_workbook(selected_feature_rows, SELECTED_FEATURES_XLSX)
            print(f"[SAVED] Selected features workbook: {SELECTED_FEATURES_XLSX}")

        save_pooled_confusion(
            cm_output_file=CONFUSION_XLSX,
            pooled_store=pooled_store,
            labels=labels,
            label_names=label_names,
        )
        print(f"[SAVED] Pooled confusion matrices: {CONFUSION_XLSX}")

        save_pooled_roc(
            roc_plot_file=ROC_PNG,
            pooled_store=pooled_store,
            labels=labels,
            label_names=label_names,
            mode=MODE,
        )
        print(f"[SAVED] Pooled ROC plot: {ROC_PNG}")

        print("\nFINAL SUMMARY:")
        print(summary_df)

        print("\nOUTPUT FILES:")
        print(f"  Outer splits:    {OUTER_SPLITS_FILE}")
        print(f"  Results:         {RESULTS_XLSX}")
        print(f"  Raw folds:       {RAW_FOLDS_XLSX}")
        print(f"  Best params:     {BEST_PARAMS_XLSX}")
        print(f"  Patient preds:   {PATIENT_PREDICTIONS_XLSX}")
        print(f"  Confusions:      {CONFUSION_XLSX}")
        print(f"  ROC plot:        {ROC_PNG}")
        if USE_FEATURE_SELECTION:
            print(f"  Selected feats:  {SELECTED_FEATURES_XLSX}")