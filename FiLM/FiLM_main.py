import os
import glob
from collections import defaultdict
import sys
import textwrap
import json

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
import pandas as pd
import ray

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
)

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from FiLM_model import (
    FusionTCN,
    FusionCNN,
    FiLMTCN,
    FiLMCNN,
    run_nested_split_ray,
    generate_kfold_split_file,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# -----------------------
# CONFIG
# -----------------------
strain = None  # e.g. "A28" or None for all strains found in folder
case = "all"  # "all", "N_vs_P", "M_vs_S"

runs = 5
epochs = 100
n_trials = 60
optuna_max_epochs = 75
n_inner_splits = 3

# Optional derivative channels
use_first_derivative = False
use_second_derivative = False

# Channel order is always:
#   [raw] or [raw, d1] or [raw, d1, d2]
normalize_raw = False
normalize_d1 = False
normalize_d2 = False

# Tabular normalization
normalize_tabular = True

force_cpu = True
ray_num_cpus = None
ray_log_level = "ERROR"

# Input folders (all relative to this script)
curve_folder_path = "time_series"
parameter_folder_path = "growth_parameters"

# Output folders (all relative to this script)
RESULT_DIR = "Results"
SPLIT_DIR = "splits"  # shared split filenames stay unchanged
PARAMS_DIR = "Hyperparameters"
ROC_DIR = "ROC_curves"
CM_DIR = "Confusion_matrices"
PRED_DIR = "Patient_predictions"

SAVE_BEST_PARAMS = True
SAVE_ROC = True
SAVE_CONFUSION = True
SAVE_PATIENT_PREDICTIONS = True


# -----------------------
# Case mapping
# -----------------------
case_dict = {
    "all": {"group_map": {"N": 0, "M": 1, "S": 2}, "filter_groups": None},
    "N_vs_P": {"group_map": {"N": 0, "M": 1, "S": 1}, "filter_groups": None},
    "M_vs_S": {"group_map": {"N": 2, "M": 1, "S": 0}, "filter_groups": 2},
}
group_map = case_dict[case]["group_map"]
filter_groups = case_dict[case]["filter_groups"]

CASE_LABELS = {
    "all": {0: "N : Negative", 1: "M : Mild", 2: "S : Severe"},
    "N_vs_P": {0: "N : Negative", 1: "P : Positive"},
    "M_vs_S": {0: "S : Severe", 1: "M : Mild"},
}


# -----------------------
# Curve loader
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
        ex = bad.head(10).to_dict()
        raise ValueError(f"Expected exactly 2 replicates per patient; found deviations. Examples: {ex}")

    y_by_p = out.groupby("patient")["group"].nunique()
    bad2 = y_by_p[y_by_p != 1]
    if len(bad2) > 0:
        ex = bad2.head(10).to_dict()
        raise ValueError(f"Patients with inconsistent labels across replicates found. Examples: {ex}")

    return out


# -----------------------
# Parameter loader
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
        ex = bad.head(10).to_dict()
        raise ValueError(f"Expected exactly 2 parameter replicates per patient; found deviations. Examples: {ex}")

    y_by_p = out.groupby("patient")["group"].nunique()
    bad2 = y_by_p[y_by_p != 1]
    if len(bad2) > 0:
        ex = bad2.head(10).to_dict()
        raise ValueError(f"Patients with inconsistent parameter labels across replicates found. Examples: {ex}")

    return out


# -----------------------
# Merge both modalities
# -----------------------
def load_and_merge_multimodal_data(curve_file, parameter_file, group_map, filter_groups):
    df_curve = load_and_process_od_data(
        curve_file,
        group_map=group_map,
        filter_groups=filter_groups,
    )
    df_param = load_parameter_data(
        parameter_file,
        group_map=group_map,
        filter_groups=filter_groups,
    )

    curve_time_cols = [c for c in df_curve.columns if c not in ["patient", "repetition", "group"]]
    param_cols = [c for c in df_param.columns if c not in ["patient", "repetition", "group"]]

    merged = pd.merge(
        df_curve,
        df_param,
        on=["patient", "repetition", "group"],
        how="inner",
        validate="one_to_one",
        suffixes=("", ""),
    )

    expected_rows_curve = len(df_curve)
    expected_rows_param = len(df_param)
    expected_rows_merged = len(merged)

    if expected_rows_merged != expected_rows_curve or expected_rows_merged != expected_rows_param:
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

    rep_counts = merged.groupby("patient")["repetition"].nunique()
    bad = rep_counts[rep_counts != 2]
    if len(bad) > 0:
        ex = bad.head(10).to_dict()
        raise ValueError(f"Expected exactly 2 merged replicates per patient. Examples: {ex}")

    y_by_p = merged.groupby("patient")["group"].nunique()
    bad2 = y_by_p[y_by_p != 1]
    if len(bad2) > 0:
        ex = bad2.head(10).to_dict()
        raise ValueError(f"Merged patients with inconsistent labels found. Examples: {ex}")

    return merged, curve_time_cols, param_cols


def maybe_init_ray():
    if ray.is_initialized():
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))

    ray.init(
        include_dashboard=False,
        logging_level=ray_log_level,
        local_mode=False,
        runtime_env={"working_dir": base_dir},
    )


# -----------------------
# Input channels
# -----------------------
def build_input_channels(
    X_raw_2d: np.ndarray,
    time_values: np.ndarray,
    *,
    use_first_derivative: bool = False,
    use_second_derivative: bool = False,
) -> np.ndarray:
    """
    Build input tensor from raw curve and optional derivatives.

    Returns
    -------
    X : np.ndarray
        Shape [N, C, T]
    """
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

    X = np.stack(channels, axis=1)  # [N, C, T]
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


def normalize_mask_to_tag(mask):
    names = ["raw", "d1", "d2"]
    chosen = [names[i] for i, flag in enumerate(mask) if flag]
    if not chosen:
        return "norm_none"
    return "norm_" + "_".join(chosen)


def get_feature_tag(use_first_derivative: bool, use_second_derivative: bool) -> str:
    if use_first_derivative and use_second_derivative:
        return "raw_d1_d2"
    if use_first_derivative:
        return "raw_d1"
    if use_second_derivative:
        return "raw_d2"
    return "raw"


# -----------------------
# Plotting helpers
# -----------------------
def _wrap_title(title: str, width: int = 90) -> str:
    return "\n".join(textwrap.wrap(title, width=width))


def save_pooled_roc(
    out_png: str,
    pooled: dict,
    label_map: dict,
    title: str,
):
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


def save_pooled_confusion(
    out_prefix: str,
    pooled: dict,
    label_map: dict,
    title: str,
):
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
        im = plt.imshow(
            matrix,
            cmap="Blues",
            interpolation="nearest",
            vmin=0,
            vmax=vmax,
        )
        plt.gca().set_aspect("equal")
        cbar = plt.colorbar(im)
        if is_normalized:
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
            cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

        t = _wrap_title(title)
        if is_normalized:
            t += "\n(Pooled row-normalized)"
        else:
            t += "\n(Pooled counts)"
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
                if is_normalized:
                    txt = f"{counts[i, j]}\n({val*100:.1f}%)"
                else:
                    txt = str(int(val))
                plt.text(j, i, txt, ha="center", va="center", color=color)

        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.savefig(out_png, dpi=200)
        plt.close()

    _plot_cm(cm.astype(float), out_prefix + "_counts_pooled.png", is_normalized=False, counts=cm)
    _plot_cm(cmn, out_prefix + "_normalized_pooled.png", is_normalized=True, counts=cm)


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
# Main
# -----------------------
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    result_dir = os.path.join(base_dir, RESULT_DIR)
    split_dir = os.path.join(base_dir, SPLIT_DIR)
    params_dir = os.path.join(base_dir, PARAMS_DIR)
    roc_dir = os.path.join(base_dir, ROC_DIR)
    cm_dir = os.path.join(base_dir, CM_DIR)
    pred_dir = os.path.join(base_dir, PRED_DIR)

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(roc_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)
    if SAVE_BEST_PARAMS:
        os.makedirs(params_dir, exist_ok=True)
    if SAVE_PATIENT_PREDICTIONS:
        os.makedirs(pred_dir, exist_ok=True)

    curve_dir = os.path.join(base_dir, curve_folder_path)
    param_dir = os.path.join(base_dir, parameter_folder_path)

    if strain is None:
        curve_files = [
            f for f in glob.glob(os.path.join(curve_dir, "*.xlsx"))
            if not os.path.basename(f).startswith("~$")
        ]
        if not curve_files:
            raise FileNotFoundError(f"No .xlsx files found in: {curve_dir}")
    else:
        curve_files = [os.path.join(curve_dir, f"{strain} strain.xlsx")]
        if not os.path.exists(curve_files[0]):
            raise FileNotFoundError(f"Missing file: {curve_files[0]}")

    maybe_init_ray()

    rows_summary = []
    rows_raw = []
    rows_patient_predictions = []

    models = [FiLMTCN, FiLMCNN]
    label_map = CASE_LABELS[case]

    feature_tag = get_feature_tag(use_first_derivative, use_second_derivative)
    normalize_channels = build_normalize_mask(
        use_first_derivative=use_first_derivative,
        use_second_derivative=use_second_derivative,
        normalize_raw=normalize_raw,
        normalize_d1=normalize_d1,
        normalize_d2=normalize_d2,
    )
    normalization_tag = normalize_mask_to_tag(normalize_channels)
    tab_tag = "tabnorm_on" if normalize_tabular else "tabnorm_off"

    for curve_file in curve_files:
        file_stem = os.path.splitext(os.path.basename(curve_file))[0]
        inferred_strain = file_stem.replace(" strain", "").strip()

        parameter_file = os.path.join(param_dir, f"{inferred_strain} - parameters_v3.xlsx")
        if not os.path.exists(parameter_file):
            raise FileNotFoundError(f"Missing parameter file for strain '{inferred_strain}': {parameter_file}")

        print(f"\n=== Processing multimodal strain: {inferred_strain} ===")

        merged, curve_time_cols, param_cols = load_and_merge_multimodal_data(
            curve_file=curve_file,
            parameter_file=parameter_file,
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
            use_first_derivative=use_first_derivative,
            use_second_derivative=use_second_derivative,
        )

        X_tab = merged[param_cols].values.astype(np.float32)

        if len(normalize_channels) != X_ts.shape[1]:
            raise ValueError(
                f"normalize_channels length ({len(normalize_channels)}) does not match "
                f"number of input channels ({X_ts.shape[1]})."
            )

        print(
            f"[Features] feature_tag={feature_tag} | "
            f"X_ts shape={X_ts.shape} | X_tab shape={X_tab.shape} | "
            f"normalize_channels={normalize_channels} | normalize_tabular={normalize_tabular}"
        )

        split_path = os.path.join(split_dir, f"kfold_shared_{case}_{runs}folds.json")
        if not os.path.exists(split_path):
            print(f"[Split] Creating patient-level StratifiedKFold splits at: {split_path}")
            generate_kfold_split_file(
                split_path,
                patients=patients,
                y=y,
                n_splits=runs,
                shuffle=True,
                base_seed=2026,
            )
            print(f"[Split] CREATED shared StratifiedKFold splits ({runs} folds).")
        else:
            print(f"[Split] LOADED existing shared K-fold splits: {split_path}")

        X_ts_ref = ray.put(X_ts)
        X_tab_ref = ray.put(X_tab)
        y_ref = ray.put(y)
        patients_ref = ray.put(patients)

        for ModelClass in models:
            mname = ModelClass.__name__
            print(f"\n--- Model: {mname} ---")

            futures = []
            job_meta = []
            for split_id in range(runs):
                futures.append(
                    run_nested_split_ray.remote(
                        ModelClass,
                        X_ts_ref,
                        X_tab_ref,
                        y_ref,
                        patients_ref,
                        split_file=split_path,
                        split_id=split_id,
                        test_size=0.0,
                        epochs=epochs,
                        use_optuna=True,
                        n_trials=n_trials,
                        optuna_max_epochs=optuna_max_epochs,
                        n_inner_splits=n_inner_splits,
                        normalize_channels=normalize_channels,
                        normalize_tabular=normalize_tabular,
                        seed=2026 + split_id,
                        force_cpu=force_cpu,
                    )
                )
                job_meta.append(split_id)

            results = ray.get(futures)

            metric_lists = defaultdict(list)
            pooled_true = []
            pooled_pred = []
            pooled_proba = []

            for split_id, res in zip(job_meta, results):
                metrics_dict, artifacts = res

                raw_row = {
                    "File": inferred_strain,
                    "Case": case,
                    "Model": mname,
                    "FeatureTag": feature_tag,
                    "NormalizationTag": normalization_tag,
                    "TabularNormTag": tab_tag,
                    "Split_id": int(split_id),
                    "MaxEpochs": int(epochs),
                    "EffectiveEpochs": int(metrics_dict["EffectiveEpochs"]),
                    "Balanced accuracy": float(metrics_dict["BalancedAcc"]),
                    "Macro Precision": float(metrics_dict["MacroPrecision"]),
                    "Macro Recall": float(metrics_dict["MacroRecall"]),
                    "Macro F1": float(metrics_dict["MacroF1"]),
                    "Macro Specificity": float(metrics_dict["MacroSpecificity"]),
                    "AUC": float(metrics_dict["AUC"]),
                    "n_trials": int(n_trials),
                    "optuna_max_epochs": int(optuna_max_epochs),
                    "n_inner_splits": int(n_inner_splits),
                    "normalize_channels": json.dumps(normalize_channels),
                    "normalize_raw": bool(normalize_raw),
                    "normalize_d1": bool(normalize_d1) if use_first_derivative else False,
                    "normalize_d2": bool(normalize_d2) if use_second_derivative else False,
                    "normalize_tabular": bool(normalize_tabular),
                    "use_first_derivative": bool(use_first_derivative),
                    "use_second_derivative": bool(use_second_derivative),
                    "n_input_channels": int(X_ts.shape[1]),
                    "n_tabular_features": int(X_tab.shape[1]),
                    "n_patients_train_outer": int(metrics_dict["n_patients_train_outer"]),
                    "n_patients_test_outer": int(metrics_dict["n_patients_test_outer"]),
                }

                auc_per_class = metrics_dict.get("AUC_per_class", {})
                for cls_idx, aucv in auc_per_class.items():
                    cls_idx = int(cls_idx)
                    raw_row[f"AUC_{label_map.get(cls_idx, f'Class{cls_idx}') }"] = float(aucv)

                rows_raw.append(raw_row)

                metric_lists["BalancedAcc"].append(float(metrics_dict["BalancedAcc"]))
                metric_lists["MacroPrecision"].append(float(metrics_dict["MacroPrecision"]))
                metric_lists["MacroRecall"].append(float(metrics_dict["MacroRecall"]))
                metric_lists["MacroF1"].append(float(metrics_dict["MacroF1"]))
                metric_lists["MacroSpecificity"].append(float(metrics_dict["MacroSpecificity"]))
                metric_lists["AUC"].append(float(metrics_dict["AUC"]))
                metric_lists["EffectiveEpochs"].append(float(metrics_dict["EffectiveEpochs"]))

                pooled_true.append(np.asarray(artifacts["y_true"], dtype=int))
                pooled_pred.append(np.asarray(artifacts["y_pred"], dtype=int))
                pooled_proba.append(np.asarray(artifacts["proba"], dtype=float))

                if SAVE_PATIENT_PREDICTIONS:
                    patient_ids_fold = np.asarray(artifacts["patient_ids"]).astype(str)
                    y_true_fold = np.asarray(artifacts["y_true"], dtype=int)
                    y_pred_fold = np.asarray(artifacts["y_pred"], dtype=int)
                    proba_fold = np.asarray(artifacts["proba"], dtype=float)

                    if not (
                        len(patient_ids_fold) == len(y_true_fold) == len(y_pred_fold) == proba_fold.shape[0]
                    ):
                        raise ValueError(
                            f"Patient prediction length mismatch in split {split_id}: "
                            f"{len(patient_ids_fold)=}, {len(y_true_fold)=}, "
                            f"{len(y_pred_fold)=}, {proba_fold.shape[0]=}"
                        )

                    for i in range(len(patient_ids_fold)):
                        row_pred = {
                            "File": inferred_strain,
                            "Case": case,
                            "Model": mname,
                            "FeatureTag": feature_tag,
                            "NormalizationTag": normalization_tag,
                            "TabularNormTag": tab_tag,
                            "Split_id": int(split_id),
                            "Patient": patient_ids_fold[i],
                            "TrueLabel": int(y_true_fold[i]),
                            "PredLabel": int(y_pred_fold[i]),
                            "TrueLabelName": label_map.get(int(y_true_fold[i]), f"Class {int(y_true_fold[i])}"),
                            "PredLabelName": label_map.get(int(y_pred_fold[i]), f"Class {int(y_pred_fold[i])}"),
                        }

                        for cls_idx in range(proba_fold.shape[1]):
                            cls_name = label_map.get(cls_idx, f"Class {cls_idx}")
                            row_pred[f"Proba_{cls_name}"] = float(proba_fold[i, cls_idx])

                        rows_patient_predictions.append(row_pred)

                if SAVE_BEST_PARAMS:
                    outp = os.path.join(
                        params_dir,
                        f"{inferred_strain}_{feature_tag}_{normalization_tag}_{tab_tag}_{case}_{mname}_fold{split_id:02d}_best_params.json"
                    )
                    with open(outp, "w", encoding="utf-8") as f:
                        json.dump(artifacts["best_params"], f, indent=2)

            pooled = {
                "y_true": np.concatenate(pooled_true, axis=0) if pooled_true else np.array([], dtype=int),
                "y_pred": np.concatenate(pooled_pred, axis=0) if pooled_pred else np.array([], dtype=int),
                "proba": np.concatenate(pooled_proba, axis=0) if pooled_proba else np.zeros((0, len(label_map)), dtype=float),
            }

            if SAVE_ROC:
                out_png = os.path.join(
                    roc_dir,
                    f"{inferred_strain}_{feature_tag}_{normalization_tag}_{tab_tag}_{case}_{mname}_ROC_POOLED.png"
                )
                roc_title = (
                    f"Pooled ROC | {inferred_strain} | {feature_tag} | "
                    f"{normalization_tag} | {tab_tag} | {case} | {mname}"
                )
                save_pooled_roc(
                    out_png=out_png,
                    pooled=pooled,
                    label_map=label_map,
                    title=roc_title,
                )

            if SAVE_CONFUSION:
                out_prefix = os.path.join(
                    cm_dir,
                    f"{inferred_strain}_{feature_tag}_{normalization_tag}_{tab_tag}_{case}_{mname}_CM_POOLED"
                )
                save_pooled_confusion(
                    out_prefix=out_prefix,
                    pooled=pooled,
                    label_map=label_map,
                    title=(
                        f"Pooled confusion matrix | {inferred_strain} | {feature_tag} | "
                        f"{normalization_tag} | {tab_tag} | {case} | {mname}"
                    ),
                )

            summary_row = {
                "File": inferred_strain,
                "Case": case,
                "Model": mname,
                "FeatureTag": feature_tag,
                "NormalizationTag": normalization_tag,
                "TabularNormTag": tab_tag,
                "normalize_channels": json.dumps(normalize_channels),
                "Folds": int(runs),
                "OuterKind": "StratifiedKFold_patient",
                "MaxEpochs": int(epochs),
                "n_trials": int(n_trials),
                "optuna_max_epochs": int(optuna_max_epochs),
                "n_inner_splits": int(n_inner_splits),
                "Use_first_derivative": bool(use_first_derivative),
                "Use_second_derivative": bool(use_second_derivative),
                "Normalize_raw": bool(normalize_raw),
                "Normalize_d1": bool(normalize_d1) if use_first_derivative else False,
                "Normalize_d2": bool(normalize_d2) if use_second_derivative else False,
                "Normalize_tabular": bool(normalize_tabular),
                "Input_channels": int(X_ts.shape[1]),
                "Tabular_features": int(X_tab.shape[1]),
            }

            pretty = {
                "BalancedAcc": "Balanced accuracy",
                "MacroPrecision": "Macro precision",
                "MacroRecall": "Macro recall",
                "MacroF1": "Macro F1",
                "MacroSpecificity": "Macro specificity",
                "AUC": "AUC",
                "EffectiveEpochs": "Effective epochs",
            }

            for k, label in pretty.items():
                stats = summarize_metric(np.array(metric_lists[k], dtype=float))
                summary_row[f"{label} mean (across folds)"] = stats["mean"]
                summary_row[f"{label} SD (across folds)"] = stats["sd"]
                summary_row[f"{label} median (across folds)"] = stats["median"]
                summary_row[f"{label} p2.5 (across folds)"] = stats["p2.5"]
                summary_row[f"{label} p97.5 (across folds)"] = stats["p97.5"]

            rows_summary.append(summary_row)

    out_df = pd.DataFrame(rows_summary)
    raw_df = pd.DataFrame(rows_raw)
    pred_df = pd.DataFrame(rows_patient_predictions)

    mean_sd_cols = [
        "File",
        "Case",
        "Model",
        "FeatureTag",
        "NormalizationTag",
        "TabularNormTag",
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
        "Effective epochs mean (across folds)",
        "Effective epochs SD (across folds)",
    ]

    mean_sd_df = out_df[mean_sd_cols]

    if strain is None:
        strain_names = []
        for f in curve_files:
            stem = os.path.splitext(os.path.basename(f))[0]
            inferred = stem.replace(" strain", "").strip()
            strain_names.append(inferred)
        strain_part = "_".join(sorted(strain_names))
    else:
        strain_part = strain

    mode_part = "M_vs_S_vs_N" if case == "all" else case

    excel_name = os.path.join(
        result_dir,
        f"{strain_part}_{mode_part}_FiLM.xlsx"
    )

    with pd.ExcelWriter(excel_name, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="Summary", index=False)
        mean_sd_df.to_excel(writer, sheet_name="Mean_SD_only", index=False)
        raw_df.to_excel(writer, sheet_name="RawFolds", index=False)

    if SAVE_PATIENT_PREDICTIONS:
        pred_name = os.path.join(
            pred_dir,
            f"{strain}_{feature_tag}_{normalization_tag}_{tab_tag}_multimodal_{case}_{epochs}_nestedCV_KFOLD{runs}_patient_predictions.xlsx"
        )
        with pd.ExcelWriter(pred_name, engine="openpyxl") as writer:
            pred_df.to_excel(writer, sheet_name="PatientPredictions", index=False)

    print(f"\nSaved Excel workbook to: {excel_name}")
    if SAVE_PATIENT_PREDICTIONS:
        print(f"Patient predictions saved in: {pred_dir}")
    print(f"ROC curves saved in: {roc_dir}")
    print(f"Confusion matrices saved in: {cm_dir}")

    ray.shutdown()


if __name__ == "__main__":
    main()