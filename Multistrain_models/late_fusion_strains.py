import os
import json
import sys
import textwrap
from itertools import combinations
from collections import defaultdict

# Force UTF-8 everywhere on Windows
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["RAY_DISABLE_METRICS"] = "1"
os.environ["RAY_DISABLE_USAGE_STATS"] = "1"
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import numpy as np
import optuna
import pandas as pd
import ray

from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold

from late_fusion_strain_model import (
    CNN1D,
    TCN,
    FiLMCNN,
    FiLMTCN,
    generate_kfold_split_file,
    run_nested_split_ray_ts as run_ts_split_ray,
    run_nested_split_ray_mm as run_mm_split_ray,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# -----------------------
# CONFIG
# -----------------------
CASE = "M_vs_S"  # "all", "N_vs_P", "M_vs_S"

RUNS = 5
EPOCHS = 100
N_TRIALS = 60
OPTUNA_MAX_EPOCHS = 75
N_INNER_SPLITS = 3

FORCE_CPU = True
RAY_NUM_CPUS = None
RAY_LOG_LEVEL = "ERROR"

# Weighted fusion settings
FUSION_WEIGHT_TRIALS = 300
FUSION_WEIGHT_OBJECTIVE = "balanced accuracy"   # "logloss" or "balanced_accuracy"
FUSION_INNER_SPLITS = 3               # Inner CV over outer-train OOF table

# -----------------------
# PATH CONFIG
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input folders 
CURVE_FOLDER_PATH = os.path.join(BASE_DIR, "time_series")
PARAMETER_FOLDER_PATH = os.path.join(BASE_DIR, "growth_parameters")

# Shared split folder 
SPLIT_DIR = os.path.join(BASE_DIR, "splits")

# Output folder
RESULT_DIR = os.path.join(BASE_DIR, "Late_fusion")
RAW_PROBA_DIR = os.path.join(RESULT_DIR, "outer_fold_patient_probabilities")
OOF_PROBA_DIR = os.path.join(RESULT_DIR, "outer_train_oof_patient_probabilities")
ROC_DIR = os.path.join(RESULT_DIR, "roc_curves")
CM_DIR = os.path.join(RESULT_DIR, "confusion_matrices")

SAVE_ROC = True
SAVE_CONFUSION = True
SAVE_OOF_PROBA = True

# ------------------------------------------------------------------
# Edit here if a strain uses a different backbone or settings.
# ------------------------------------------------------------------
STRAIN_CONFIGS = {
    "A0": {
        "mode": "film",
        "model_class": FiLMTCN,
        "use_first_derivative": True,
        "use_second_derivative": False,
        "normalize_raw": True,
        "normalize_d1": True,
        "normalize_d2": False,
        "normalize_tabular": True,
    },
    "A1": {
        "mode": "ts",
        "model_class": TCN,
        "use_first_derivative": True,
        "use_second_derivative": False,
        "normalize_raw": True,
        "normalize_d1": True,
        "normalize_d2": False,
    },
    "A5": {
        "mode": "ts",
        "model_class": CNN1D,
        "use_first_derivative": False,
        "use_second_derivative": False,
        "normalize_raw": False,
        "normalize_d1": False,
        "normalize_d2": False,
    },
    "A15": {
        "mode": "ts",
        "model_class": TCN,
        "use_first_derivative": True,
        "use_second_derivative": False,
        "normalize_raw": False,
        "normalize_d1": False,
        "normalize_d2": False,
    },
    "A19": {
        "mode": "ts",
        "model_class": TCN,
        "use_first_derivative": True,
        "use_second_derivative": False,
        "normalize_raw": True,
        "normalize_d1": True,
        "normalize_d2": False,
    },
}

STRAINS_TO_USE = ["A0", "A1", "A5", "A15", "A19"]


# -----------------------
# Case mapping
# -----------------------
case_dict = {
    "all": {"group_map": {"N": 0, "M": 1, "S": 2}, "filter_groups": None},
    "N_vs_P": {"group_map": {"N": 0, "M": 1, "S": 1}, "filter_groups": None},
    "M_vs_S": {"group_map": {"N": 2, "M": 1, "S": 0}, "filter_groups": 2},
}
group_map = case_dict[CASE]["group_map"]
filter_groups = case_dict[CASE]["filter_groups"]

CASE_LABELS = {
    "all": {0: "N : Negative", 1: "M : Mild", 2: "S : Severe"},
    "N_vs_P": {0: "N : Negative", 1: "P : Positive"},
    "M_vs_S": {0: "S : Severe", 1: "M : Mild"},
}

# -----------------------
# Shared helpers
# -----------------------
def maybe_init_ray():
    if ray.is_initialized():
        return

    init_kwargs = dict(
        include_dashboard=False,
        logging_level=RAY_LOG_LEVEL,
        local_mode=False,
    )
    if RAY_NUM_CPUS is not None:
        init_kwargs["num_cpus"] = int(RAY_NUM_CPUS)

    ray.init(**init_kwargs)


def build_input_channels(
    X_raw_2d: np.ndarray,
    time_values: np.ndarray,
    *,
    use_first_derivative: bool = False,
    use_second_derivative: bool = False,
) -> np.ndarray:
    X_raw_2d = np.asarray(X_raw_2d, dtype=np.float32)
    time_values = np.asarray(time_values, dtype=np.float32)

    if X_raw_2d.ndim != 2:
        raise ValueError(f"X_raw_2d must be [N, T], got shape {X_raw_2d.shape}")
    if time_values.ndim != 1:
        raise ValueError(f"time_values must be 1D, got shape {time_values.shape}")
    if X_raw_2d.shape[1] != len(time_values):
        raise ValueError(
            f"Mismatch: X has T={X_raw_2d.shape[1]} timepoints but time_values has {len(time_values)} values."
        )

    channels = [X_raw_2d]

    d1 = None
    if use_first_derivative or use_second_derivative:
        d1 = np.gradient(X_raw_2d, time_values, axis=1).astype(np.float32)
        if use_first_derivative:
            channels.append(d1)

    if use_second_derivative:
        d2 = np.gradient(d1, time_values, axis=1).astype(np.float32)
        channels.append(d2)

    X = np.stack(channels, axis=1)
    return X.astype(np.float32)


def build_normalize_mask(
    *,
    use_first_derivative: bool,
    use_second_derivative: bool,
    normalize_raw: bool,
    normalize_d1: bool,
    normalize_d2: bool,
):
    mask = [bool(normalize_raw)]
    if use_first_derivative:
        mask.append(bool(normalize_d1))
    if use_second_derivative:
        mask.append(bool(normalize_d2))
    return mask


def get_feature_tag(use_first_derivative: bool, use_second_derivative: bool) -> str:
    if use_first_derivative and use_second_derivative:
        return "raw_d1_d2"
    if use_first_derivative:
        return "raw_d1"
    if use_second_derivative:
        return "raw_d2"
    return "raw"


def normalize_mask_to_tag(mask):
    names = ["raw", "d1", "d2"]
    chosen = [names[i] for i, flag in enumerate(mask) if flag]
    if not chosen:
        return "norm_none"
    return "norm_" + "_".join(chosen)


def summarize_metric(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x)),
        "p2.5": float(np.percentile(x, 2.5)),
        "p97.5": float(np.percentile(x, 97.5)),
        "median": float(np.median(x)),
    }


# -----------------------
# Metric helpers
# -----------------------
def _macro_specificity(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities = []
    for i in range(num_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        specificities.append(float(spec))
    return float(np.mean(specificities)) if specificities else 0.0


def compute_patient_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    proba = np.asarray(proba, dtype=float)

    num_classes = int(proba.shape[1])

    out = {}
    out["BalancedAcc"] = float(balanced_accuracy_score(y_true, y_pred))
    out["MacroPrecision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    out["MacroRecall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    out["MacroF1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["MacroSpecificity"] = float(_macro_specificity(y_true, y_pred, num_classes))

    if num_classes == 2:
        try:
            auc_pos1 = float(roc_auc_score(y_true, proba[:, 1]))
        except Exception:
            auc_pos1 = float("nan")

        try:
            auc_pos0 = float(roc_auc_score(1 - y_true, proba[:, 0]))
        except Exception:
            auc_pos0 = float("nan")

        out["AUC"] = auc_pos1
        out["AUC_per_class"] = {0: auc_pos0, 1: auc_pos1}
    else:
        try:
            out["AUC"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
        except Exception:
            out["AUC"] = float("nan")

        per_class = {}
        for c in range(num_classes):
            y_bin = (y_true == c).astype(int)
            try:
                per_class[c] = float(roc_auc_score(y_bin, proba[:, c]))
            except Exception:
                per_class[c] = float("nan")
        out["AUC_per_class"] = per_class

    return out


# -----------------------
# Plotting helpers
# -----------------------
def _wrap_title(title: str, width: int = 90) -> str:
    return "\n".join(textwrap.wrap(title, width=width))


def save_pooled_roc(out_png: str, pooled: dict, label_map: dict, title: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    y_true = np.asarray(pooled["y_true"], dtype=int)
    proba = np.asarray(pooled["proba"], dtype=float)

    if len(np.unique(y_true)) < 2 or proba.shape[0] == 0:
        return

    n_classes = proba.shape[1]
    plt.figure(figsize=(8, 6))

    if n_classes == 2:
        y_bin_0 = (y_true == 0).astype(int)
        score_0 = proba[:, 0]
        fpr_0, tpr_0, _ = roc_curve(y_bin_0, score_0)
        auc_0 = auc(fpr_0, tpr_0)

        y_bin_1 = (y_true == 1).astype(int)
        score_1 = proba[:, 1]
        fpr_1, tpr_1, _ = roc_curve(y_bin_1, score_1)
        auc_1 = auc(fpr_1, tpr_1)

        plt.plot(fpr_0, tpr_0, label=f"{label_map[0]} | AUC={auc_0:.3f}")
        plt.plot(fpr_1, tpr_1, label=f"{label_map[1]} | AUC={auc_1:.3f}")
    else:
        for c in range(n_classes):
            y_bin = (y_true == c).astype(int)
            if len(np.unique(y_bin)) < 2:
                continue
            score_c = proba[:, c]
            fpr_c, tpr_c, _ = roc_curve(y_bin, score_c)
            auc_c = auc(fpr_c, tpr_c)
            plt.plot(fpr_c, tpr_c, label=f"{label_map.get(c, f'Class {c}')} | AUC={auc_c:.3f}")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(_wrap_title(title))
    plt.legend(loc="lower right")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_pooled_confusion(out_prefix: str, pooled: dict, label_map: dict, title: str):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    y_true = np.asarray(pooled["y_true"], dtype=int)
    y_pred = np.asarray(pooled["y_pred"], dtype=int)

    labels = sorted(list(set(y_true.tolist()) | set(y_pred.tolist())))
    names = [label_map.get(i, f"Class {i}") for i in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(int)
    df_cm = pd.DataFrame(
        cm,
        index=[f"True: {n}" for n in names],
        columns=[f"Pred: {n}" for n in names],
    )
    df_cm.to_excel(out_prefix + "_counts_pooled.xlsx", index=True)

    row_sum = cm.sum(axis=1).astype(float)
    row_sum[row_sum == 0] = 1.0
    cmn = cm.astype(float) / row_sum[:, None]

    def _plot_cm(matrix, out_png, is_normalized: bool, counts=None):
        vmax = 1.0 if is_normalized else (np.max(matrix) if np.max(matrix) > 0 else 1)

        plt.figure(figsize=(8, 6))
        im = plt.imshow(matrix, cmap="Blues", interpolation="nearest", vmin=0, vmax=vmax)
        plt.gca().set_aspect("equal")
        cbar = plt.colorbar(im)
        if is_normalized:
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
            cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

        t = _wrap_title(title)
        t += "\n(Pooled row-normalized)" if is_normalized else "\n(Pooled counts)"
        plt.title(t)

        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(range(len(names)), names, rotation=30, ha="right")
        plt.yticks(range(len(names)), names)

        thresh = 0.5 if is_normalized else ((np.max(matrix) / 2.0) if np.max(matrix) > 0 else 0.5)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                color = "white" if val > thresh else "black"
                txt = f"{counts[i, j]}\n({val*100:.1f}%)" if is_normalized else str(int(val))
                plt.text(j, i, txt, ha="center", va="center", color=color)

        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.savefig(out_png, dpi=200)
        plt.close()

    _plot_cm(cm.astype(float), out_prefix + "_counts_pooled.png", is_normalized=False, counts=cm)
    _plot_cm(cmn, out_prefix + "_normalized_pooled.png", is_normalized=True, counts=cm)


# -----------------------
# Data loading: time-series
# -----------------------
def load_and_process_od_data(filepath, sheet_name=0, group_map=None, filter_groups=None):
    if group_map is None:
        raise ValueError("group_map must be provided")

    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df = df.rename(columns={df.columns[0]: "time"})
    df = df.set_index("time")

    df_long = df.reset_index().melt(id_vars="time", var_name="sample", value_name="OD")

    df_long["group_letter"] = df_long["sample"].str.extract(r"^([NMS])")
    df_long["patient"] = df_long["sample"].str.extract(r"^([NMS]\d+)")
    rep = df_long["sample"].str.extract(r"Replicate\s*(\d+)")
    df_long["repetition"] = pd.to_numeric(rep[0], errors="coerce")

    if df_long["repetition"].isna().any():
        bad = df_long.loc[df_long["repetition"].isna(), "sample"].head(10).tolist()
        raise ValueError(
            "Could not parse 'Replicate <n>' from some column names. "
            f"Examples: {bad}"
        )
    df_long["repetition"] = df_long["repetition"].astype(int)

    df_long["group"] = df_long["group_letter"].map(group_map)
    if df_long["group"].isna().any():
        bad = df_long.loc[df_long["group"].isna(), "sample"].head(10).tolist()
        raise ValueError(f"Some samples have unknown group_letter. Examples: {bad}")

    df_cnn = (
        df_long.pivot_table(
            index=["patient", "repetition", "group"],
            columns="time",
            values="OD",
        )
        .reset_index()
    )

    meta_cols = ["patient", "repetition", "group"]
    time_cols = sorted([c for c in df_cnn.columns if c not in meta_cols], key=lambda x: float(x))
    df_cnn = df_cnn[meta_cols + time_cols]

    out = df_cnn if filter_groups is None else df_cnn[df_cnn["group"] != filter_groups].copy()

    rep_counts = out.groupby("patient")["repetition"].nunique()
    bad = rep_counts[rep_counts != 2]
    if len(bad) > 0:
        raise ValueError(f"Expected exactly 2 replicates per patient; found deviations. Examples: {bad.head(10).to_dict()}")

    y_by_p = out.groupby("patient")["group"].nunique()
    bad2 = y_by_p[y_by_p != 1]
    if len(bad2) > 0:
        raise ValueError(f"Patients with inconsistent labels across replicates found. Examples: {bad2.head(10).to_dict()}")

    return out


# -----------------------
# Data loading: multimodal
# -----------------------
def load_parameter_data(filepath, group_map=None, filter_groups=None):
    if group_map is None:
        raise ValueError("group_map must be provided")

    df1 = pd.read_excel(filepath, sheet_name="Replicate 1").copy()
    df1["repetition"] = 1

    df2 = pd.read_excel(filepath, sheet_name="Replicate 2").copy()
    df2["repetition"] = 2

    df = pd.concat([df1, df2], ignore_index=True)

    first_col = df.columns[0]
    df[first_col] = df[first_col].astype(str)

    df["patient"] = df[first_col].str.extract(r"^([NMS]\d+)")[0]
    df["group_letter"] = df[first_col].str[0]

    if df["patient"].isna().any():
        bad = df.loc[df["patient"].isna(), first_col].head(10).tolist()
        raise ValueError(f"Could not parse patient IDs from first column. Examples: {bad}")

    df["group"] = df["group_letter"].map(group_map)
    if df["group"].isna().any():
        bad = df.loc[df["group"].isna(), first_col].head(10).tolist()
        raise ValueError(f"Some parameter rows have unknown group_letter. Examples: {bad}")

    if filter_groups is not None:
        df = df[df["group"] != filter_groups].copy()

    exclude = ["patient", "group_letter", "group", "repetition", first_col]
    feature_cols = [c for c in df.columns if c not in exclude]

    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if len(non_numeric) > 0:
        raise ValueError(f"Non-numeric parameter columns found. Examples: {non_numeric[:10]}")

    if df[feature_cols].isna().any().any():
        bad_cols = df[feature_cols].columns[df[feature_cols].isna().any()].tolist()
        raise ValueError(f"NaN found in parameter features. Examples: {bad_cols[:10]}")

    out = df[["patient", "repetition", "group"] + feature_cols].copy()

    rep_counts = out.groupby("patient")["repetition"].nunique()
    bad = rep_counts[rep_counts != 2]
    if len(bad) > 0:
        raise ValueError(f"Expected exactly 2 parameter replicates per patient; found deviations. Examples: {bad.head(10).to_dict()}")

    y_by_p = out.groupby("patient")["group"].nunique()
    bad2 = y_by_p[y_by_p != 1]
    if len(bad2) > 0:
        raise ValueError(f"Patients with inconsistent parameter labels across replicates found. Examples: {bad2.head(10).to_dict()}")

    return out


def load_and_merge_multimodal_data(curve_file, parameter_file, group_map, filter_groups):
    df_curve = load_and_process_od_data(curve_file, group_map=group_map, filter_groups=filter_groups)
    df_param = load_parameter_data(parameter_file, group_map=group_map, filter_groups=filter_groups)

    curve_time_cols = [c for c in df_curve.columns if c not in ["patient", "repetition", "group"]]
    param_cols = [c for c in df_param.columns if c not in ["patient", "repetition", "group"]]

    merged = pd.merge(
        df_curve,
        df_param,
        on=["patient", "repetition", "group"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(df_curve) or len(merged) != len(df_param):
        curve_keys = set(zip(df_curve["patient"], df_curve["repetition"]))
        param_keys = set(zip(df_param["patient"], df_param["repetition"]))
        missing_in_param = sorted(list(curve_keys - param_keys))[:10]
        missing_in_curve = sorted(list(param_keys - curve_keys))[:10]
        raise ValueError(
            "Curve and parameter files do not align perfectly on (patient, repetition).\n"
            f"Missing in parameter examples: {missing_in_param}\n"
            f"Missing in curve examples: {missing_in_curve}"
        )

    dup = merged.duplicated(subset=["patient", "repetition"])
    if dup.any():
        bad = merged.loc[dup, ["patient", "repetition"]].head(10).values.tolist()
        raise ValueError(f"Duplicate merged (patient, repetition) rows found. Examples: {bad}")

    return merged, curve_time_cols, param_cols


# -----------------------
# Preparation per strain
# -----------------------
def get_curve_file(base_curve_dir: str, strain: str) -> str:
    path = os.path.join(base_curve_dir, f"{strain} strain.xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing curve file: {path}")
    return path


def get_parameter_file(base_param_dir: str, strain: str) -> str:
    path = os.path.join(base_param_dir, f"{strain} - parameters_v3.xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing parameter file: {path}")
    return path


def patient_label_map_from_replicates(patients: np.ndarray, y: np.ndarray):
    patients = np.asarray(patients).astype(str)
    y = np.asarray(y).astype(int)
    out = {}
    for p in np.unique(patients):
        yp = np.unique(y[patients == p])
        if len(yp) != 1:
            raise ValueError(f"Patient {p} has inconsistent labels: {yp.tolist()}")
        out[str(p)] = int(yp[0])
    return out


def prepare_one_strain(base_curve_dir: str, base_param_dir: str, strain: str, cfg: dict):
    if cfg["mode"] == "ts":
        curve_file = get_curve_file(base_curve_dir, strain)
        df = load_and_process_od_data(curve_file, group_map=group_map, filter_groups=filter_groups)

        patients = df["patient"].astype(str).values
        y = df["group"].astype(int).values.astype(np.int64)

        time_cols = [c for c in df.columns if c not in ["patient", "repetition", "group"]]
        X_raw = df[time_cols].values.astype(np.float32)
        time_values = np.asarray(time_cols, dtype=np.float32)

        X = build_input_channels(
            X_raw,
            time_values,
            use_first_derivative=cfg["use_first_derivative"],
            use_second_derivative=cfg["use_second_derivative"],
        )

        normalize_channels = build_normalize_mask(
            use_first_derivative=cfg["use_first_derivative"],
            use_second_derivative=cfg["use_second_derivative"],
            normalize_raw=cfg["normalize_raw"],
            normalize_d1=cfg["normalize_d1"],
            normalize_d2=cfg["normalize_d2"],
        )

        if len(normalize_channels) != X.shape[1]:
            raise ValueError(
                f"{strain}: normalize_channels length ({len(normalize_channels)}) "
                f"does not match X.shape[1] ({X.shape[1]})."
            )

        return {
            "strain": strain,
            "mode": "ts",
            "model_class": cfg["model_class"],
            "patients": patients,
            "y": y,
            "X": X,
            "normalize_channels": normalize_channels,
            "feature_tag": get_feature_tag(cfg["use_first_derivative"], cfg["use_second_derivative"]),
            "normalization_tag": normalize_mask_to_tag(normalize_channels),
            "patient_label_map": patient_label_map_from_replicates(patients, y),
        }

    if cfg["mode"] == "film":
        curve_file = get_curve_file(base_curve_dir, strain)
        param_file = get_parameter_file(base_param_dir, strain)

        merged, curve_time_cols, param_cols = load_and_merge_multimodal_data(
            curve_file=curve_file,
            parameter_file=param_file,
            group_map=group_map,
            filter_groups=filter_groups,
        )

        patients = merged["patient"].astype(str).values
        y = merged["group"].astype(int).values.astype(np.int64)

        X_raw = merged[curve_time_cols].values.astype(np.float32)
        time_values = np.asarray(curve_time_cols, dtype=np.float32)
        X_ts = build_input_channels(
            X_raw,
            time_values,
            use_first_derivative=cfg["use_first_derivative"],
            use_second_derivative=cfg["use_second_derivative"],
        )
        X_tab = merged[param_cols].values.astype(np.float32)

        normalize_channels = build_normalize_mask(
            use_first_derivative=cfg["use_first_derivative"],
            use_second_derivative=cfg["use_second_derivative"],
            normalize_raw=cfg["normalize_raw"],
            normalize_d1=cfg["normalize_d1"],
            normalize_d2=cfg["normalize_d2"],
        )

        if len(normalize_channels) != X_ts.shape[1]:
            raise ValueError(
                f"{strain}: normalize_channels length ({len(normalize_channels)}) "
                f"does not match X_ts.shape[1] ({X_ts.shape[1]})."
            )

        return {
            "strain": strain,
            "mode": "film",
            "model_class": cfg["model_class"],
            "patients": patients,
            "y": y,
            "X_ts": X_ts,
            "X_tab": X_tab,
            "normalize_channels": normalize_channels,
            "normalize_tabular": bool(cfg["normalize_tabular"]),
            "feature_tag": get_feature_tag(cfg["use_first_derivative"], cfg["use_second_derivative"]),
            "normalization_tag": normalize_mask_to_tag(normalize_channels),
            "tab_tag": "tabnorm_on" if cfg["normalize_tabular"] else "tabnorm_off",
            "patient_label_map": patient_label_map_from_replicates(patients, y),
        }

    raise ValueError(f"Unknown mode for strain {strain}: {cfg['mode']}")


def assert_all_strains_share_same_patients(prepared_by_strain: dict):
    strains = list(prepared_by_strain.keys())
    ref = strains[0]
    ref_map = prepared_by_strain[ref]["patient_label_map"]

    for s in strains[1:]:
        cur_map = prepared_by_strain[s]["patient_label_map"]
        if set(cur_map.keys()) != set(ref_map.keys()):
            missing_in_s = sorted(list(set(ref_map.keys()) - set(cur_map.keys())))[:10]
            extra_in_s = sorted(list(set(cur_map.keys()) - set(ref_map.keys())))[:10]
            raise ValueError(
                f"Patient sets differ between {ref} and {s}.\n"
                f"Missing in {s}: {missing_in_s}\n"
                f"Extra in {s}: {extra_in_s}\n"
                "For exact shared-split late fusion, all strains must have the same patient set."
            )

        mismatched = [p for p in ref_map if ref_map[p] != cur_map[p]]
        if mismatched:
            raise ValueError(
                f"Patient labels differ between {ref} and {s}. "
                f"Examples: {[(p, ref_map[p], cur_map[p]) for p in mismatched[:10]]}"
            )


# -----------------------
# Running one strain across all outer folds
# -----------------------
def run_all_outer_folds_for_one_strain(prep: dict, split_path: str):
    strain = prep["strain"]
    model_name = prep["model_class"].__name__
    print(f"\n=== Running strain {strain} with {model_name} ===")

    futures = []
    split_ids = list(range(RUNS))

    if prep["mode"] == "ts":
        X_ref = ray.put(prep["X"])
        y_ref = ray.put(prep["y"])
        patients_ref = ray.put(prep["patients"])

        for split_id in split_ids:
            futures.append(
                run_ts_split_ray.remote(
                    prep["model_class"],
                    X_ref,
                    y_ref,
                    patients_ref,
                    split_file=split_path,
                    split_id=split_id,
                    test_size=0.0,
                    epochs=EPOCHS,
                    use_optuna=True,
                    n_trials=N_TRIALS,
                    optuna_max_epochs=OPTUNA_MAX_EPOCHS,
                    n_inner_splits=N_INNER_SPLITS,
                    normalize_channels=prep["normalize_channels"],
                    seed=2026 + split_id,
                    force_cpu=FORCE_CPU,
                )
            )
    elif prep["mode"] == "film":
        X_ts_ref = ray.put(prep["X_ts"])
        X_tab_ref = ray.put(prep["X_tab"])
        y_ref = ray.put(prep["y"])
        patients_ref = ray.put(prep["patients"])

        for split_id in split_ids:
            futures.append(
                run_mm_split_ray.remote(
                    prep["model_class"],
                    X_ts_ref,
                    X_tab_ref,
                    y_ref,
                    patients_ref,
                    split_file=split_path,
                    split_id=split_id,
                    test_size=0.0,
                    epochs=EPOCHS,
                    use_optuna=True,
                    n_trials=N_TRIALS,
                    optuna_max_epochs=OPTUNA_MAX_EPOCHS,
                    n_inner_splits=N_INNER_SPLITS,
                    normalize_channels=prep["normalize_channels"],
                    normalize_tabular=prep["normalize_tabular"],
                    seed=2026 + split_id,
                    force_cpu=FORCE_CPU,
                )
            )
    else:
        raise ValueError(f"Unknown mode for strain {strain}: {prep['mode']}")

    results = ray.get(futures)

    per_fold = {}
    for split_id, (metrics, artifacts) in zip(split_ids, results):
        patient_ids = np.asarray(artifacts["patient_ids"]).astype(str)
        y_true = np.asarray(artifacts["y_true"], dtype=int)
        y_pred = np.asarray(artifacts["y_pred"], dtype=int)
        proba = np.asarray(artifacts["proba"], dtype=float)

        oof_train_patient_ids = np.asarray(artifacts["oof_train_patient_ids"]).astype(str)
        oof_train_y_true = np.asarray(artifacts["oof_train_y_true"], dtype=int)
        oof_train_y_pred = np.asarray(artifacts["oof_train_y_pred"], dtype=int)
        oof_train_proba = np.asarray(artifacts["oof_train_proba"], dtype=float)

        per_fold[split_id] = {
            "strain": strain,
            "split_id": int(split_id),
            "model_name": model_name,
            "metrics": metrics,
            "patient_ids": patient_ids,
            "y_true": y_true,
            "y_pred": y_pred,
            "proba": proba,
            "best_params": artifacts["best_params"],
            "oof_train_patient_ids": oof_train_patient_ids,
            "oof_train_y_true": oof_train_y_true,
            "oof_train_y_pred": oof_train_y_pred,
            "oof_train_proba": oof_train_proba,
        }

        os.makedirs(RAW_PROBA_DIR, exist_ok=True)
        out_npz = os.path.join(
            RAW_PROBA_DIR,
            f"{strain}_{model_name}_{CASE}_fold{split_id:02d}_patient_probs.npz",
        )
        np.savez_compressed(
            out_npz,
            patient_ids=patient_ids,
            y_true=y_true,
            y_pred=y_pred,
            proba=proba,
        )

    return per_fold


# -----------------------
# Weighted late fusion
# -----------------------
def align_outputs(
    strain_fold_outputs: list,
    combo_name: str,
    split_id: int,
    *,
    patient_key: str,
    y_key: str,
    proba_key: str,
):
    ref = strain_fold_outputs[0]
    ref_patients = np.asarray(ref[patient_key]).astype(str)
    ref_y_true = np.asarray(ref[y_key], dtype=int)

    aligned_probabilities = [np.asarray(ref[proba_key], dtype=float)]

    for other in strain_fold_outputs[1:]:
        other_patients = np.asarray(other[patient_key]).astype(str)
        other_y_true = np.asarray(other[y_key], dtype=int)

        if set(other_patients.tolist()) != set(ref_patients.tolist()):
            missing = sorted(list(set(ref_patients.tolist()) - set(other_patients.tolist())))[:10]
            extra = sorted(list(set(other_patients.tolist()) - set(ref_patients.tolist())))[:10]
            raise ValueError(
                f"[{combo_name} | fold {split_id}] patient sets do not match.\n"
                f"Missing: {missing}\n"
                f"Extra: {extra}"
            )

        idx = {p: i for i, p in enumerate(other_patients.tolist())}
        order = [idx[p] for p in ref_patients.tolist()]

        other_y_true_aligned = other_y_true[order]
        if not np.array_equal(other_y_true_aligned, ref_y_true):
            raise ValueError(f"[{combo_name} | fold {split_id}] y_true mismatch after alignment.")

        aligned_probabilities.append(np.asarray(other[proba_key], dtype=float)[order])

    n_classes_set = {p.shape[1] for p in aligned_probabilities}
    if len(n_classes_set) != 1:
        raise ValueError(
            f"[{combo_name} | fold {split_id}] Models disagree on number of classes: {n_classes_set}"
        )

    return ref_patients, ref_y_true, aligned_probabilities


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def weighted_average_probabilities(proba_list: list, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)

    stacked = np.stack(proba_list, axis=0)   # [M, N, C]
    fused = np.tensordot(weights, stacked, axes=(0, 0))   # [N, C]

    row_sum = fused.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return fused / row_sum


def fit_simplex_weights_on_aligned_proba(
    proba_list: list,
    y_true: np.ndarray,
    *,
    n_trials: int,
    objective_name: str,
    seed: int,
):
    y_true = np.asarray(y_true, dtype=int)
    n_models = len(proba_list)
    num_classes = int(proba_list[0].shape[1])

    if n_models == 1:
        return np.array([1.0], dtype=float)

    sampler = optuna.samplers.TPESampler(seed=int(seed))
    direction = "minimize" if objective_name == "logloss" else "maximize"
    study = optuna.create_study(direction=direction, sampler=sampler)

    labels = list(range(num_classes))

    def objective(trial: optuna.Trial) -> float:
        logits = np.array(
            [trial.suggest_float(f"logit_{i}", -6.0, 6.0) for i in range(n_models)],
            dtype=float,
        )
        weights_local = _softmax(logits)
        fused = weighted_average_probabilities(proba_list, weights_local)

        if objective_name == "logloss":
            fused = np.clip(fused, 1e-12, 1.0)
            return float(log_loss(y_true, fused, labels=labels))

        pred = np.argmax(fused, axis=1).astype(int)
        return float(balanced_accuracy_score(y_true, pred))

    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)

    best_logits = np.array(
        [study.best_params[f"logit_{i}"] for i in range(n_models)],
        dtype=float,
    )
    return _softmax(best_logits)


def cross_validated_fusion_assessment_on_oof(
    strain_fold_outputs: list,
    combo_name: str,
    split_id: int,
    *,
    n_trials: int,
    objective_name: str,
    seed: int,
    n_splits: int,
):
    patient_ids, y_true, proba_list = align_outputs(
        strain_fold_outputs,
        combo_name=combo_name,
        split_id=split_id,
        patient_key="oof_train_patient_ids",
        y_key="oof_train_y_true",
        proba_key="oof_train_proba",
    )

    patient_ids = np.asarray(patient_ids).astype(str)
    y_true = np.asarray(y_true, dtype=int)
    proba_list = [np.asarray(p, dtype=float) for p in proba_list]

    classes, counts = np.unique(y_true, return_counts=True)
    if len(classes) < 2:
        raise ValueError(f"[{combo_name} | fold {split_id}] Need at least 2 classes in OOF fusion table.")
    if counts.min() < 2:
        raise ValueError(
            f"[{combo_name} | fold {split_id}] Not enough samples in at least one class "
            f"for fusion-inner CV. Class counts: {dict(zip(classes.tolist(), counts.tolist()))}"
        )

    K = min(int(n_splits), int(counts.min()))
    if K < 2:
        raise ValueError(
            f"[{combo_name} | fold {split_id}] fusion-inner CV requires at least 2 folds."
        )

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=int(seed))

    fused_pred_oof = np.empty_like(y_true, dtype=int)
    fused_proba_oof = np.zeros_like(proba_list[0], dtype=float)

    for inner_fold_idx, (tr_idx, va_idx) in enumerate(skf.split(patient_ids, y_true)):
        y_tr = y_true[tr_idx]
        proba_tr = [p[tr_idx] for p in proba_list]
        proba_va = [p[va_idx] for p in proba_list]

        weights = fit_simplex_weights_on_aligned_proba(
            proba_tr,
            y_tr,
            n_trials=int(n_trials),
            objective_name=objective_name,
            seed=int(seed + inner_fold_idx),
        )

        fused_va = weighted_average_probabilities(proba_va, weights)
        fused_proba_oof[va_idx] = fused_va
        fused_pred_oof[va_idx] = np.argmax(fused_va, axis=1).astype(int)

    metrics = compute_patient_metrics(y_true, fused_pred_oof, fused_proba_oof)

    return {
        "cv_patient_ids": patient_ids,
        "cv_y_true": y_true,
        "cv_y_pred": fused_pred_oof,
        "cv_proba": fused_proba_oof,
        "cv_metrics": metrics,
    }


def fit_final_fusion_weights_from_oof(
    strain_fold_outputs: list,
    combo_name: str,
    split_id: int,
    *,
    n_trials: int,
    objective_name: str,
    seed: int,
):
    patient_ids, y_true, proba_list = align_outputs(
        strain_fold_outputs,
        combo_name=combo_name,
        split_id=split_id,
        patient_key="oof_train_patient_ids",
        y_key="oof_train_y_true",
        proba_key="oof_train_proba",
    )

    weights = fit_simplex_weights_on_aligned_proba(
        proba_list,
        y_true,
        n_trials=int(n_trials),
        objective_name=objective_name,
        seed=int(seed),
    )

    return {
        "weights": weights,
        "fit_patient_ids": np.asarray(patient_ids).astype(str),
        "fit_y_true": np.asarray(y_true, dtype=int),
    }


def late_fuse_weighted_probabilities(
    strain_fold_outputs: list,
    combo_name: str,
    split_id: int,
    *,
    weights: np.ndarray,
):
    patient_ids, y_true, proba_list = align_outputs(
        strain_fold_outputs,
        combo_name=combo_name,
        split_id=split_id,
        patient_key="patient_ids",
        y_key="y_true",
        proba_key="proba",
    )

    fused_proba = weighted_average_probabilities(proba_list, weights)
    fused_pred = np.argmax(fused_proba, axis=1).astype(int)
    metrics = compute_patient_metrics(y_true, fused_pred, fused_proba)

    return {
        "combo_name": combo_name,
        "split_id": int(split_id),
        "patient_ids": patient_ids,
        "y_true": y_true,
        "y_pred": fused_pred,
        "proba": fused_proba,
        "weights": np.asarray(weights, dtype=float),
        "metrics": metrics,
    }


# -----------------------
# Main
# -----------------------
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(SPLIT_DIR, exist_ok=True)
    os.makedirs(ROC_DIR, exist_ok=True)
    os.makedirs(CM_DIR, exist_ok=True)
    os.makedirs(RAW_PROBA_DIR, exist_ok=True)
    os.makedirs(OOF_PROBA_DIR, exist_ok=True)

    curve_dir = CURVE_FOLDER_PATH
    param_dir = PARAMETER_FOLDER_PATH

    maybe_init_ray()

    # 1) Prepare each strain once
    prepared_by_strain = {}
    for strain in STRAINS_TO_USE:
        cfg = STRAIN_CONFIGS[strain]
        prep = prepare_one_strain(curve_dir, param_dir, strain, cfg)
        prepared_by_strain[strain] = prep

        if prep["mode"] == "ts":
            print(
                f"[Prepared] {strain} | mode=ts | model={prep['model_class'].__name__} | "
                f"X shape={prep['X'].shape} | feature_tag={prep['feature_tag']} | "
                f"normalization_tag={prep['normalization_tag']}"
            )
        else:
            print(
                f"[Prepared] {strain} | mode=film | model={prep['model_class'].__name__} | "
                f"X_ts shape={prep['X_ts'].shape} | X_tab shape={prep['X_tab'].shape} | "
                f"feature_tag={prep['feature_tag']} | normalization_tag={prep['normalization_tag']} | "
                f"tab_tag={prep['tab_tag']}"
            )

    # 2) Enforce identical patient sets/labels across strains
    assert_all_strains_share_same_patients(prepared_by_strain)

    # 3) Create/load one shared split file for every strain
    reference_strain = STRAINS_TO_USE[0]
    ref_patients = prepared_by_strain[reference_strain]["patients"]
    ref_y = prepared_by_strain[reference_strain]["y"]

    split_path = os.path.join(SPLIT_DIR, f"kfold_shared_{CASE}_{RUNS}folds.json")
    if not os.path.exists(split_path):
        print(f"[Split] Creating shared StratifiedKFold split file: {split_path}")
        generate_kfold_split_file(
            split_path,
            patients=ref_patients,
            y=ref_y,
            n_splits=RUNS,
            shuffle=True,
            base_seed=2026,
        )
        print(f"[Split] CREATED shared splits ({RUNS} folds).")
    else:
        print(f"[Split] LOADED existing shared split file: {split_path}")

    # 4) Run each strain once across all outer folds
    all_single_results = {}
    for strain in STRAINS_TO_USE:
        all_single_results[strain] = run_all_outer_folds_for_one_strain(prepared_by_strain[strain], split_path)

    # 5) Build all pairs, triplets, 4-way, 5-way
    combos = []
    combos.extend(list(combinations(STRAINS_TO_USE, 2)))
    combos.extend(list(combinations(STRAINS_TO_USE, 3)))
    combos.extend(list(combinations(STRAINS_TO_USE, 4)))
    combos.extend(list(combinations(STRAINS_TO_USE, 5)))

    rows_raw = []
    rows_summary = []

    label_map = CASE_LABELS[CASE]

    # 6) Weighted late fusion per combo and per fold
    for combo in combos:
        combo = tuple(combo)
        combo_name = "+".join(combo)
        combo_size = len(combo)

        print(f"\n=== Weighted late fusion: {combo_name} ===")

        metric_lists = defaultdict(list)
        pooled_true = []
        pooled_pred = []
        pooled_proba = []

        component_desc = []
        for s in combo:
            prep = prepared_by_strain[s]
            if prep["mode"] == "ts":
                component_desc.append(
                    f"{s}:{prep['model_class'].__name__}:{prep['feature_tag']}:{prep['normalization_tag']}"
                )
            else:
                component_desc.append(
                    f"{s}:{prep['model_class'].__name__}:{prep['feature_tag']}:{prep['normalization_tag']}:{prep['tab_tag']}"
                )

        for split_id in range(RUNS):
            fold_outputs = [all_single_results[s][split_id] for s in combo]

            fusion_cv = cross_validated_fusion_assessment_on_oof(
                fold_outputs,
                combo_name=combo_name,
                split_id=split_id,
                n_trials=FUSION_WEIGHT_TRIALS,
                objective_name=FUSION_WEIGHT_OBJECTIVE,
                seed=90_000 + split_id,
                n_splits=FUSION_INNER_SPLITS,
            )

            final_weight_fit = fit_final_fusion_weights_from_oof(
                fold_outputs,
                combo_name=combo_name,
                split_id=split_id,
                n_trials=FUSION_WEIGHT_TRIALS,
                objective_name=FUSION_WEIGHT_OBJECTIVE,
                seed=95_000 + split_id,
            )

            fused = late_fuse_weighted_probabilities(
                fold_outputs,
                combo_name=combo_name,
                split_id=split_id,
                weights=final_weight_fit["weights"],
            )

            metrics = fused["metrics"]

            raw_row = {
                "Combo": combo_name,
                "ComboSize": int(combo_size),
                "Strains": json.dumps(list(combo)),
                "Case": CASE,
                "FusionRule": "weighted_probabilities",
                "WeightObjective": FUSION_WEIGHT_OBJECTIVE,
                "Split_id": int(split_id),
                "Balanced accuracy": float(metrics["BalancedAcc"]),
                "Macro Precision": float(metrics["MacroPrecision"]),
                "Macro Recall": float(metrics["MacroRecall"]),
                "Macro F1": float(metrics["MacroF1"]),
                "Macro Specificity": float(metrics["MacroSpecificity"]),
                "AUC": float(metrics["AUC"]),
                "Components": " | ".join(component_desc),
                "runs": int(RUNS),
                "epochs": int(EPOCHS),
                "n_trials": int(N_TRIALS),
                "optuna_max_epochs": int(OPTUNA_MAX_EPOCHS),
                "n_inner_splits": int(N_INNER_SPLITS),
                "fusion_inner_splits": int(FUSION_INNER_SPLITS),
                "fusion_weight_trials": int(FUSION_WEIGHT_TRIALS),
                "LearnedWeights": json.dumps({s: float(w) for s, w in zip(combo, final_weight_fit["weights"])}),
                "Fusion-CV BalancedAcc (outer train, OOF-based)": float(fusion_cv["cv_metrics"]["BalancedAcc"]),
                "Fusion-CV Macro F1 (outer train, OOF-based)": float(fusion_cv["cv_metrics"]["MacroF1"]),
                "Fusion-CV AUC (outer train, OOF-based)": float(fusion_cv["cv_metrics"]["AUC"]),
            }

            for s, w in zip(combo, final_weight_fit["weights"]):
                raw_row[f"weight_{s}"] = float(w)
                metric_lists[f"weight::{s}"].append(float(w))

            auc_per_class = metrics.get("AUC_per_class", {})
            for cls_idx, aucv in auc_per_class.items():
                cls_idx = int(cls_idx)
                raw_row[f"AUC_{label_map.get(cls_idx, f'Class{cls_idx}')}"] = float(aucv)

            rows_raw.append(raw_row)

            metric_lists["BalancedAcc"].append(float(metrics["BalancedAcc"]))
            metric_lists["MacroPrecision"].append(float(metrics["MacroPrecision"]))
            metric_lists["MacroRecall"].append(float(metrics["MacroRecall"]))
            metric_lists["MacroF1"].append(float(metrics["MacroF1"]))
            metric_lists["MacroSpecificity"].append(float(metrics["MacroSpecificity"]))
            metric_lists["AUC"].append(float(metrics["AUC"]))
            metric_lists["FusionCV_BalancedAcc"].append(float(fusion_cv["cv_metrics"]["BalancedAcc"]))
            metric_lists["FusionCV_AUC"].append(float(fusion_cv["cv_metrics"]["AUC"]))

            pooled_true.append(np.asarray(fused["y_true"], dtype=int))
            pooled_pred.append(np.asarray(fused["y_pred"], dtype=int))
            pooled_proba.append(np.asarray(fused["proba"], dtype=float))

            out_npz = os.path.join(
                RAW_PROBA_DIR,
                f"{combo_name}_{CASE}_fold{split_id:02d}_late_fusion_weighted_probs.npz",
            )
            np.savez_compressed(
                out_npz,
                patient_ids=np.asarray(fused["patient_ids"]).astype(str),
                y_true=np.asarray(fused["y_true"], dtype=int),
                y_pred=np.asarray(fused["y_pred"], dtype=int),
                proba=np.asarray(fused["proba"], dtype=float),
                weights=np.asarray(fused["weights"], dtype=float),
            )

            if SAVE_OOF_PROBA:
                out_oof_npz = os.path.join(
                    OOF_PROBA_DIR,
                    f"{combo_name}_{CASE}_fold{split_id:02d}_weighted_fusion_outer_train_honest_cv_probs.npz",
                )
                np.savez_compressed(
                    out_oof_npz,
                    patient_ids=np.asarray(fusion_cv["cv_patient_ids"]).astype(str),
                    y_true=np.asarray(fusion_cv["cv_y_true"], dtype=int),
                    y_pred=np.asarray(fusion_cv["cv_y_pred"], dtype=int),
                    proba=np.asarray(fusion_cv["cv_proba"], dtype=float),
                    weights=np.asarray(final_weight_fit["weights"], dtype=float),
                )

        pooled = {
            "y_true": np.concatenate(pooled_true, axis=0) if pooled_true else np.array([], dtype=int),
            "y_pred": np.concatenate(pooled_pred, axis=0) if pooled_pred else np.array([], dtype=int),
            "proba": np.concatenate(pooled_proba, axis=0) if pooled_proba else np.zeros((0, len(label_map)), dtype=float),
        }

        if SAVE_ROC:
            out_png = os.path.join(
                ROC_DIR,
                f"{combo_name}_{CASE}_weighted_latefusion_ROC_POOLED.png"
            )
            roc_title = f"Pooled ROC | {combo_name} | {CASE} | weighted late fusion"
            save_pooled_roc(out_png=out_png, pooled=pooled, label_map=label_map, title=roc_title)

        if SAVE_CONFUSION:
            out_prefix = os.path.join(
                CM_DIR,
                f"{combo_name}_{CASE}_weighted_latefusion_CM_POOLED"
            )
            save_pooled_confusion(
                out_prefix=out_prefix,
                pooled=pooled,
                label_map=label_map,
                title=f"Pooled confusion matrix | {combo_name} | {CASE} | weighted late fusion",
            )

        summary_row = {
            "Combo": combo_name,
            "ComboSize": int(combo_size),
            "Strains": json.dumps(list(combo)),
            "Case": CASE,
            "FusionRule": "weighted_probabilities",
            "WeightObjective": FUSION_WEIGHT_OBJECTIVE,
            "Folds": int(RUNS),
            "OuterKind": "StratifiedKFold_patient",
            "Components": " | ".join(component_desc),
            "epochs": int(EPOCHS),
            "n_trials": int(N_TRIALS),
            "optuna_max_epochs": int(OPTUNA_MAX_EPOCHS),
            "n_inner_splits": int(N_INNER_SPLITS),
            "fusion_inner_splits": int(FUSION_INNER_SPLITS),
            "fusion_weight_trials": int(FUSION_WEIGHT_TRIALS),
        }

        pretty = {
            "BalancedAcc": "Balanced accuracy",
            "MacroPrecision": "Macro precision",
            "MacroRecall": "Macro recall",
            "MacroF1": "Macro F1",
            "MacroSpecificity": "Macro specificity",
            "AUC": "AUC",
            "FusionCV_BalancedAcc": "Fusion-CV Balanced accuracy",
            "FusionCV_AUC": "Fusion-CV AUC",
        }

        for k, label in pretty.items():
            stats = summarize_metric(np.array(metric_lists[k], dtype=float))
            summary_row[f"{label} mean (across folds)"] = stats["mean"]
            summary_row[f"{label} SD (across folds)"] = stats["sd"]
            summary_row[f"{label} median (across folds)"] = stats["median"]
            summary_row[f"{label} p2.5 (across folds)"] = stats["p2.5"]
            summary_row[f"{label} p97.5 (across folds)"] = stats["p97.5"]

        for s in combo:
            w_arr = np.asarray(metric_lists[f"weight::{s}"], dtype=float)
            summary_row[f"weight_{s} mean (across folds)"] = float(np.mean(w_arr))
            summary_row[f"weight_{s} SD (across folds)"] = float(np.std(w_arr))
            summary_row[f"weight_{s} median (across folds)"] = float(np.median(w_arr))
            summary_row[f"weight_{s} p2.5 (across folds)"] = float(np.percentile(w_arr, 2.5))
            summary_row[f"weight_{s} p97.5 (across folds)"] = float(np.percentile(w_arr, 97.5))

        rows_summary.append(summary_row)

    # 7) Save workbook
    out_df = pd.DataFrame(rows_summary)
    raw_df = pd.DataFrame(rows_raw)

    mean_sd_cols = [
        "Combo",
        "ComboSize",
        "Case",
        "FusionRule",
        "WeightObjective",
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
        "Fusion-CV Balanced accuracy mean (across folds, outer train OOF)",
        "Fusion-CV Balanced accuracy SD (across folds, outer train OOF)",
        "Fusion-CV AUC mean (across folds, outer train OOF)",
        "Fusion-CV AUC SD (across folds, outer train OOF)",
    ]
    mean_sd_cols = [c for c in mean_sd_cols if c in out_df.columns]
    mean_sd_df = out_df[mean_sd_cols]

    excel_name = os.path.join(
        RESULT_DIR,
        f"results_late_fusion_{CASE}_KFOLD{RUNS}_trials{N_TRIALS}_optMaxE{OPTUNA_MAX_EPOCHS}_inner{N_INNER_SPLITS}_fusionInner{FUSION_INNER_SPLITS}_wtrials{FUSION_WEIGHT_TRIALS}.xlsx"
    )

    with pd.ExcelWriter(excel_name, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="Summary", index=False)
        mean_sd_df.to_excel(writer, sheet_name="Mean_SD_only", index=False)
        raw_df.to_excel(writer, sheet_name="RawFolds", index=False)

    print(f"\nSaved Excel workbook to: {excel_name}")
    print(f"ROC curves saved in: {ROC_DIR}")
    print(f"Confusion matrices saved in: {CM_DIR}")
    print(f"Outer-test raw fold probabilities saved in: {RAW_PROBA_DIR}")
    print(f"Fusion-CV outer-train probabilities saved in: {OOF_PROBA_DIR}")

    ray.shutdown()


if __name__ == "__main__":
    main()