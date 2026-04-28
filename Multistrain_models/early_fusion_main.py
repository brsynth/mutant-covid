import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import glob
from collections import defaultdict
import sys
import textwrap
import json
from itertools import combinations

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

from sklearn.metrics import roc_curve, auc, confusion_matrix

from early_fusion_model import (
    CNN1D,
    TCN,
    FiLMCNN_A0Only,
    FiLMTCN_A0Only,
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
case = "M_vs_S"   # "all", "N_vs_P", "M_vs_S"

runs = 5
epochs = 100
n_trials = 60
optuna_max_epochs = 75
n_inner_splits = 3

force_cpu = True
ray_num_cpus = None
ray_log_level = "ERROR"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TIME_SERIES_DIR = os.path.join(BASE_DIR, "time_series")
GROWTH_PARAMETERS_DIR = os.path.join(BASE_DIR, "growth_parameters")
SPLIT_DIR = os.path.join(BASE_DIR, "splits")

RESULT_DIR = os.path.join(BASE_DIR, "Early_fusion")
PARAMS_DIR = os.path.join(RESULT_DIR, "nested_best_params")
ROC_DIR = os.path.join(RESULT_DIR, "roc_curves")
CM_DIR = os.path.join(RESULT_DIR, "confusion_matrices")
PRED_DIR = os.path.join(RESULT_DIR, "outer_fold_predictions")

SAVE_BEST_PARAMS = True
SAVE_ROC = True
SAVE_CONFUSION = True
SAVE_OUTER_PREDICTIONS = True

ALL_STRAIN_SETTINGS = {
    "A0":  {"use_raw": True, "use_d1": True,  "normalize_raw": True,  "normalize_d1": True},
    "A1":  {"use_raw": True, "use_d1": True,  "normalize_raw": True,  "normalize_d1": True},
    "A5":  {"use_raw": True, "use_d1": False, "normalize_raw": False, "normalize_d1": False},
    "A15": {"use_raw": True, "use_d1": True,  "normalize_raw": False, "normalize_d1": False},
    "A19": {"use_raw": True, "use_d1": True,  "normalize_raw": True,  "normalize_d1": True},
}

RUN_MODE = "all_2_to_5" # "manual", "pairs", "triplets", "groups_of_4", "groups_of_5", "groups_of_4_and_5", "pairs_and_triplets", "all_2_to_5"

MANUAL_STRAIN_SETS = [
    ["A0", "A1"],
    ["A0", "A5", "A19"],
    ["A0", "A1", "A5", "A19"],
    ["A0", "A1", "A5", "A15", "A19"],
]

NORMALIZE_TABULAR_A0 = True

BASE_MODELS = [CNN1D, TCN]
FILM_MODELS = [FiLMCNN_A0Only, FiLMTCN_A0Only]


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
# Original extraction logic preserved
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

    out = df_cnn if filter_groups is None else df_cnn[df_cnn["group"] != filter_groups]

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

    return out, feature_cols


def maybe_init_ray():
    if ray.is_initialized():
        return

    init_kwargs = dict(
        include_dashboard=False,
        logging_level=ray_log_level,
        local_mode=False,
        runtime_env={"working_dir": BASE_DIR},
    )
    if ray_num_cpus is not None:
        init_kwargs["num_cpus"] = int(ray_num_cpus)

    ray.init(**init_kwargs)

def strain_filename(strain_name: str) -> str:
    return f"{strain_name} strain.xlsx"


def a0_parameter_filename() -> str:
    return "A0 - parameters_v3.xlsx"


def df_to_sample_dict(df: pd.DataFrame):
    sample_dict = {}

    patients = df.iloc[:, 0].astype(str).values
    repetitions = df.iloc[:, 1].astype(int).values
    labels = df.iloc[:, 2].astype(int).values
    X_raw = df.iloc[:, 3:].values.astype(np.float32)
    time_values = np.asarray(df.columns[3:], dtype=np.float32)

    for i in range(len(df)):
        key = (str(patients[i]), int(repetitions[i]))
        sample_dict[key] = {
            "label": int(labels[i]),
            "time": time_values.copy(),
            "raw": X_raw[i].astype(np.float32).copy(),
        }

    return sample_dict


def param_df_to_sample_dict(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in ["patient", "repetition", "group"]]
    out = {}
    for _, row in df.iterrows():
        key = (str(row["patient"]), int(row["repetition"]))
        out[key] = {
            "label": int(row["group"]),
            "tab": row[feature_cols].to_numpy(dtype=np.float32),
        }
    return out, feature_cols


def build_channel_tag_list(strain_config: dict):
    channel_names = []
    normalize_channels = []

    for strain_name, cfg in strain_config.items():
        if cfg.get("use_raw", False):
            channel_names.append(f"{strain_name}_raw")
            normalize_channels.append(bool(cfg.get("normalize_raw", False)))

        if cfg.get("use_d1", False):
            channel_names.append(f"{strain_name}_d1")
            normalize_channels.append(bool(cfg.get("normalize_d1", False)))

    return channel_names, normalize_channels


def build_feature_tag_from_strain_config(strain_config: dict) -> str:
    parts = []
    for strain_name, cfg in strain_config.items():
        this = [strain_name]
        if cfg.get("use_raw", False):
            this.append("raw")
        if cfg.get("use_d1", False):
            this.append("d1")
        if cfg.get("normalize_raw", False):
            this.append("nr")
        if cfg.get("normalize_d1", False):
            this.append("nd1")
        parts.append("_".join(this))
    return "__".join(parts)


def get_global_common_keys(strain_data_dict: dict, strain_names: list[str]):
    key_sets = []
    for s in strain_names:
        if s not in strain_data_dict:
            raise KeyError(f"Missing loaded data for strain {s}")
        key_sets.append(set(strain_data_dict[s].keys()))

    global_common_keys = sorted(list(set.intersection(*key_sets)))
    if len(global_common_keys) == 0:
        raise ValueError("No global common (patient, repetition) tuples across all configured strains.")

    return global_common_keys


def build_early_fusion_tensor(strain_data_dict: dict, strain_config: dict, common_keys: list):
    strain_names = list(strain_config.keys())
    if len(strain_names) == 0:
        raise ValueError("No strains in strain_config.")
    if len(common_keys) == 0:
        raise ValueError("common_keys is empty.")

    labels = []
    patients = []
    global_tmax = 0

    for key in common_keys:
        patient, repetition = key
        y_values = []
        for s in strain_names:
            if key not in strain_data_dict[s]:
                raise ValueError(f"Key {key} missing for selected strain {s}")
            y_values.append(int(strain_data_dict[s][key]["label"]))

        if len(set(y_values)) != 1:
            raise ValueError(f"Label mismatch across strains for key={key}: {y_values}")

        labels.append(y_values[0])
        patients.append(patient)

        for s in strain_names:
            T = len(strain_data_dict[s][key]["raw"])
            global_tmax = max(global_tmax, int(T))

    channel_names, normalize_channels = build_channel_tag_list(strain_config)
    C_total = len(channel_names)
    N = len(common_keys)
    T_max = int(global_tmax)

    X = np.zeros((N, C_total, T_max), dtype=np.float32)
    mask = np.zeros((N, C_total, T_max), dtype=np.float32)

    for i, key in enumerate(common_keys):
        c = 0
        for s in strain_names:
            cfg = strain_config[s]
            entry = strain_data_dict[s][key]

            raw = np.asarray(entry["raw"], dtype=np.float32)
            time_values = np.asarray(entry["time"], dtype=np.float32)

            if raw.ndim != 1:
                raise ValueError(f"raw must be 1D for key={key}, strain={s}")
            if time_values.ndim != 1:
                raise ValueError(f"time must be 1D for key={key}, strain={s}")
            if len(raw) != len(time_values):
                raise ValueError(f"Length mismatch for key={key}, strain={s}")

            d1 = None
            if cfg.get("use_d1", False):
                d1 = np.gradient(raw, time_values).astype(np.float32)

            if cfg.get("use_raw", False):
                t = len(raw)
                X[i, c, :t] = raw
                mask[i, c, :t] = 1.0
                c += 1

            if cfg.get("use_d1", False):
                t = len(d1)
                X[i, c, :t] = d1
                mask[i, c, :t] = 1.0
                c += 1

        if c != C_total:
            raise RuntimeError(f"Internal channel count mismatch for sample i={i}")

    y = np.asarray(labels, dtype=np.int64)
    patients = np.asarray(patients).astype(str)

    unique_p = np.unique(patients)
    for p in unique_p:
        yp = np.unique(y[patients == p])
        if len(yp) != 1:
            raise ValueError(f"Patient {p} has inconsistent labels after fusion: {yp.tolist()}")

    return X, y, mask, patients, channel_names, normalize_channels


def build_a0_tabular_for_keys(a0_param_dict: dict, common_keys: list):
    missing = [k for k in common_keys if k not in a0_param_dict]
    if missing:
        raise ValueError(
            "A0 parameter file does not align with fused keys.\n"
            f"Examples missing in A0 parameters: {missing[:10]}"
        )

    X_tab = []
    labels = []
    for k in common_keys:
        X_tab.append(np.asarray(a0_param_dict[k]["tab"], dtype=np.float32))
        labels.append(int(a0_param_dict[k]["label"]))

    return np.stack(X_tab, axis=0).astype(np.float32), np.asarray(labels, dtype=int)


def normalize_mask_to_tag(channel_names, normalize_channels):
    chosen = [name for name, flag in zip(channel_names, normalize_channels) if flag]
    if not chosen:
        return "norm_none"
    return "norm_" + "__".join(chosen)


def canonicalize_subset(subset):
    return tuple(sorted([str(x) for x in subset]))


def get_strain_sets_to_run(all_settings, run_mode, manual_sets=None):
    names = sorted(list(all_settings.keys()))

    if run_mode == "manual":
        if not manual_sets:
            raise ValueError("RUN_MODE='manual' but MANUAL_STRAIN_SETS is empty.")
        out = [canonicalize_subset(s) for s in manual_sets]
    elif run_mode == "pairs":
        out = [tuple(c) for c in combinations(names, 2)]
    elif run_mode == "triplets":
        out = [tuple(c) for c in combinations(names, 3)]
    elif run_mode == "groups_of_4":
        out = [tuple(c) for c in combinations(names, 4)]
    elif run_mode == "groups_of_5":
        out = [tuple(c) for c in combinations(names, 5)]
    elif run_mode == "pairs_and_triplets":
        out = [tuple(c) for c in combinations(names, 2)] + [tuple(c) for c in combinations(names, 3)]
    elif run_mode == "groups_of_4_and_5":
        out = [tuple(c) for c in combinations(names, 4)] + [tuple(c) for c in combinations(names, 5)]
    elif run_mode == "all_2_to_5":
        out = []
        for k in [2, 3, 4, 5]:
            if k <= len(names):
                out += [tuple(c) for c in combinations(names, k)]
    elif run_mode == "all_1_to_5":
        out = []
        for k in [1, 2, 3, 4, 5]:
            if k <= len(names):
                out += [tuple(c) for c in combinations(names, k)]
    else:
        raise ValueError(f"Unknown RUN_MODE={run_mode}")

    out = sorted(set(out))
    return [list(x) for x in out]


def get_a0_channel_indices(channel_names):
    a0_idx = [i for i, name in enumerate(channel_names) if str(name).startswith("A0_")]
    other_idx = [i for i, name in enumerate(channel_names) if not str(name).startswith("A0_")]
    return a0_idx, other_idx


def _wrap_title(title: str, width: int = 90) -> str:
    return "\n".join(textwrap.wrap(title, width=width))


def save_pooled_roc(out_png: str, pooled: dict, label_map: dict, title: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    y_true = np.asarray(pooled["y_true"], dtype=int)
    proba = np.asarray(pooled["proba"], dtype=float)
    if proba.shape[1] != 2:
        raise ValueError("ROC plotting currently expects binary (2 classes).")

    plt.figure(figsize=(8, 6))

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


def summarize_metric(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x)),
        "p2.5": float(np.percentile(x, 2.5)),
        "p97.5": float(np.percentile(x, 97.5)),
        "median": float(np.median(x)),
    }


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(SPLIT_DIR, exist_ok=True)
    os.makedirs(ROC_DIR, exist_ok=True)
    os.makedirs(CM_DIR, exist_ok=True)
    if SAVE_BEST_PARAMS:
        os.makedirs(PARAMS_DIR, exist_ok=True)
    if SAVE_OUTER_PREDICTIONS:
        os.makedirs(PRED_DIR, exist_ok=True)

    curve_dir = TIME_SERIES_DIR
    param_dir = GROWTH_PARAMETERS_DIR

    maybe_init_ray()

    rows_summary = []
    rows_raw = []
    label_map = CASE_LABELS[case]

    # load all strain curves once
    strain_data_dict = {}
    for strain_name in sorted(ALL_STRAIN_SETTINGS.keys()):
        path = os.path.join(curve_dir, strain_filename(strain_name))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file for strain {strain_name}: {path}")

        print(f"[LoadCurve] {strain_name}: {path}")
        df = load_and_process_od_data(path, group_map=group_map, filter_groups=filter_groups)
        strain_data_dict[strain_name] = df_to_sample_dict(df)
        print(f"[LoadCurve] {strain_name} rows={len(strain_data_dict[strain_name])}")

    # single global cohort across all configured strains
    all_strain_names = sorted(list(ALL_STRAIN_SETTINGS.keys()))
    global_common_keys = get_global_common_keys(
        strain_data_dict=strain_data_dict,
        strain_names=all_strain_names,
    )

    global_patients = np.asarray([k[0] for k in global_common_keys]).astype(str)
    ref_strain = all_strain_names[0]
    global_labels = np.asarray(
        [int(strain_data_dict[ref_strain][k]["label"]) for k in global_common_keys],
        dtype=np.int64,
    )

    print(f"[GlobalCohort] common keys across all strains = {len(global_common_keys)}")
    print(f"[GlobalCohort] unique patients = {len(np.unique(global_patients))}")

    # one shared split file reused for all subsets
    shared_split_path = os.path.join(SPLIT_DIR, f"kfold_shared_{case}_{runs}folds.json")
    if not os.path.exists(shared_split_path):
        print(f"[Split] Creating shared patient-level StratifiedKFold splits at: {shared_split_path}")
        generate_kfold_split_file(
            shared_split_path,
            patients=global_patients,
            y=global_labels,
            n_splits=runs,
            shuffle=True,
            base_seed=2026,
        )
        print(f"[Split] CREATED shared StratifiedKFold splits ({runs} folds).")
    else:
        print(f"[Split] LOADED existing shared K-fold splits: {shared_split_path}")

    strain_sets_to_run = get_strain_sets_to_run(
        all_settings=ALL_STRAIN_SETTINGS,
        run_mode=RUN_MODE,
        manual_sets=MANUAL_STRAIN_SETS,
    )

    print(f"\n[RunMode] {RUN_MODE}")
    print(f"[RunMode] Number of strain subsets to run: {len(strain_sets_to_run)}")
    for idx, ss in enumerate(strain_sets_to_run, start=1):
        print(f"  {idx:02d}. {ss}")

    for strain_subset in strain_sets_to_run:
        active_strain_config = {s: ALL_STRAIN_SETTINGS[s] for s in strain_subset}

        print("\n====================================================")
        print(f"[Subset] Running subset: {strain_subset}")
        print("====================================================")

        X, y, mask, patients, channel_names, normalize_channels = build_early_fusion_tensor(
            strain_data_dict=strain_data_dict,
            strain_config=active_strain_config,
            common_keys=global_common_keys,
        )
        common_keys = global_common_keys

        feature_tag = build_feature_tag_from_strain_config(active_strain_config)
        normalization_tag = normalize_mask_to_tag(channel_names, normalize_channels)
        subset_tag = "__".join(strain_subset)

        print(f"[Fusion] subset_tag={subset_tag}")
        print(f"[Fusion] Samples={X.shape[0]} | Channels={X.shape[1]} | T_max={X.shape[2]}")
        print(f"[Fusion] channel_names={channel_names}")
        print(f"[Fusion] normalize_channels={normalize_channels}")
        print(f"[Fusion] unique patients={len(np.unique(patients))}")

        a0_idx, other_idx = get_a0_channel_indices(channel_names)
        has_a0 = len(a0_idx) > 0

        if has_a0:
            a0_param_path = os.path.join(param_dir, a0_parameter_filename())
            if not os.path.exists(a0_param_path):
                raise FileNotFoundError(f"Missing A0 parameter file: {a0_param_path}")

            print(f"[LoadParam] A0: {a0_param_path}")
            a0_param_df, _ = load_parameter_data(a0_param_path, group_map=group_map, filter_groups=filter_groups)
            a0_param_dict, _ = param_df_to_sample_dict(a0_param_df)

            X_tab_A0, a0_tab_labels = build_a0_tabular_for_keys(a0_param_dict, common_keys)
            if not np.array_equal(a0_tab_labels.astype(int), y.astype(int)):
                raise ValueError("A0 parameter labels do not align with fused labels.")

            tab_tag = "a0tab_norm_on" if NORMALIZE_TABULAR_A0 else "a0tab_norm_off"
            print(f"[A0Tab] shape={X_tab_A0.shape}")
        else:
            X_tab_A0 = np.zeros((len(y), 0), dtype=np.float32)
            tab_tag = "a0tab_none"

        X_ref = ray.put(X)
        Xtab_ref = ray.put(X_tab_A0)
        y_ref = ray.put(y)
        mask_ref = ray.put(mask)
        patients_ref = ray.put(patients)

        if has_a0:
            model_list = list(FILM_MODELS)
        else:
            model_list = list(BASE_MODELS)

        for ModelClass in model_list:
            mname = ModelClass.__name__
            print(f"\n--- Model: {mname} | Subset: {subset_tag} ---")

            futures = []
            job_meta = []
            for split_id in range(runs):
                futures.append(
                    run_nested_split_ray.remote(
                        ModelClass,
                        X_ref,
                        Xtab_ref,
                        y_ref,
                        mask_ref,
                        patients_ref,
                        split_file=shared_split_path,
                        split_id=split_id,
                        epochs=epochs,
                        use_optuna=True,
                        n_trials=n_trials,
                        optuna_max_epochs=optuna_max_epochs,
                        n_inner_splits=n_inner_splits,
                        normalize_channels=normalize_channels,
                        normalize_tabular=NORMALIZE_TABULAR_A0,
                        seed=2026 + split_id,
                        force_cpu=force_cpu,
                        a0_channel_indices=a0_idx,
                        other_channel_indices=other_idx,
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
                    "Case": case,
                    "StrainSubset": json.dumps(strain_subset),
                    "SubsetTag": subset_tag,
                    "Model": mname,
                    "FeatureTag": feature_tag,
                    "NormalizationTag": normalization_tag,
                    "A0TabTag": tab_tag,
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
                    "channel_names": json.dumps(channel_names),
                    "n_input_channels": int(X.shape[1]),
                    "n_patients_train_outer": int(metrics_dict["n_patients_train_outer"]),
                    "n_patients_test_outer": int(metrics_dict["n_patients_test_outer"]),
                    "N_samples_subset": int(X.shape[0]),
                    "N_patients_subset": int(len(np.unique(patients))),
                    "T_max_subset": int(X.shape[2]),
                    "has_A0": bool(has_a0),
                    "A0_tab_features": int(X_tab_A0.shape[1]),
                    "normalize_tabular_A0": bool(NORMALIZE_TABULAR_A0) if has_a0 else False,
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

                if SAVE_BEST_PARAMS:
                    outp = os.path.join(
                        PARAMS_DIR,
                        f"{subset_tag}_{feature_tag}_{normalization_tag}_{tab_tag}_{case}_{mname}_fold{split_id:02d}_best_params.json"
                    )
                    with open(outp, "w", encoding="utf-8") as f:
                        json.dump(artifacts["best_params"], f, indent=2)
                
                if SAVE_OUTER_PREDICTIONS:
                    pred_path = os.path.join(
                        PRED_DIR,
                        f"{subset_tag}_{feature_tag}_{normalization_tag}_{tab_tag}_{case}_{mname}_fold{split_id:02d}_outer_test_predictions.npz"
                    )
                    np.savez_compressed(
                        pred_path,
                        patient_ids=np.asarray(artifacts["patient_ids"]).astype(str),
                        y_true=np.asarray(artifacts["y_true"], dtype=np.int64),
                        y_pred=np.asarray(artifacts["y_pred"], dtype=np.int64),
                        proba=np.asarray(artifacts["proba"], dtype=np.float32),
                        class_indices=np.arange(artifacts["proba"].shape[1], dtype=np.int64),
                        class_labels=np.asarray(
                            [label_map[i] for i in range(artifacts["proba"].shape[1])],
                            dtype=str,
                        ),
                        split_id=np.int64(split_id),
                        case=np.array(case),
                        model=np.array(mname),
                        subset_tag=np.array(subset_tag),
                        feature_tag=np.array(feature_tag),
                        normalization_tag=np.array(normalization_tag),
                        a0_tab_tag=np.array(tab_tag),
                        channel_names=np.asarray(channel_names).astype(str),
                        normalize_channels=np.asarray(normalize_channels, dtype=bool)
                    )

            pooled = {
                "y_true": np.concatenate(pooled_true, axis=0) if pooled_true else np.array([], dtype=int),
                "y_pred": np.concatenate(pooled_pred, axis=0) if pooled_pred else np.array([], dtype=int),
                "proba": np.concatenate(pooled_proba, axis=0) if pooled_proba else np.zeros((0, 2), dtype=float),
            }

            if SAVE_ROC and case != "all":
                out_png = os.path.join(
                    ROC_DIR,
                    f"{subset_tag}_{feature_tag}_{normalization_tag}_{tab_tag}_{case}_{mname}_ROC_POOLED.png"
                )
                roc_title = f"Pooled ROC | {subset_tag} | {feature_tag} | {normalization_tag} | {tab_tag} | {case} | {mname}"
                save_pooled_roc(out_png=out_png, pooled=pooled, label_map=label_map, title=roc_title)

            if SAVE_CONFUSION:
                out_prefix = os.path.join(
                    CM_DIR,
                    f"{subset_tag}_{feature_tag}_{normalization_tag}_{tab_tag}_{case}_{mname}_CM_POOLED"
                )
                save_pooled_confusion(
                    out_prefix=out_prefix,
                    pooled=pooled,
                    label_map=label_map,
                    title=f"Pooled confusion matrix | {subset_tag} | {feature_tag} | {normalization_tag} | {tab_tag} | {case} | {mname}",
                )

            summary_row = {
                "Case": case,
                "StrainSubset": json.dumps(strain_subset),
                "SubsetTag": subset_tag,
                "Model": mname,
                "FeatureTag": feature_tag,
                "NormalizationTag": normalization_tag,
                "A0TabTag": tab_tag,
                "normalize_channels": json.dumps(normalize_channels),
                "channel_names": json.dumps(channel_names),
                "Folds": int(runs),
                "OuterKind": "StratifiedKFold_patient",
                "MaxEpochs": int(epochs),
                "n_trials": int(n_trials),
                "optuna_max_epochs": int(optuna_max_epochs),
                "n_inner_splits": int(n_inner_splits),
                "Input_channels": int(X.shape[1]),
                "N_samples": int(X.shape[0]),
                "N_patients": int(len(np.unique(patients))),
                "T_max": int(X.shape[2]),
                "RunMode": RUN_MODE,
                "has_A0": bool(has_a0),
                "A0_tab_features": int(X_tab_A0.shape[1]),
                "normalize_tabular_A0": bool(NORMALIZE_TABULAR_A0) if has_a0 else False,
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

    mean_sd_cols = [
        "Case",
        "StrainSubset",
        "SubsetTag",
        "Model",
        "FeatureTag",
        "NormalizationTag",
        "A0TabTag",
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

    excel_name = os.path.join(
        RESULT_DIR,
        f"early_fusion_{RUN_MODE}_{case}.xlsx"
    )

    with pd.ExcelWriter(excel_name, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="Summary", index=False)
        mean_sd_df.to_excel(writer, sheet_name="Mean_SD_only", index=False)
        raw_df.to_excel(writer, sheet_name="RawFolds", index=False)

    print(f"\nSaved Excel workbook to: {excel_name}")
    print(f"ROC curves saved in: {ROC_DIR}")
    print(f"Confusion matrices saved in: {CM_DIR}")

    ray.shutdown()


if __name__ == "__main__":
    main()