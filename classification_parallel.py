import pandas as pd
import os
os.environ["RAY_DISABLE_METRICS"] = "1"
os.environ["RAY_DISABLE_USAGE_STATS"] = "1"
import ray
ray.init(include_dashboard=False, logging_level="ERROR")
import numpy as np

from model import *

strain = None
case = "N_vs_P"  # Options: "all", "N_vs_P", "M_vs_S"
epochs = 160
batch = 128
kernal_size= "optuna"
runs = 50
result_name = f"classification_result/{strain}_CNN_TCN_{case}_{epochs}_{batch}_{kernal_size}.csv"

case_dict = {
    "all": {
        "group_map": {"N": 0, "M": 1, "S": 2},
        "filter_groups": None
    },
    "N_vs_P": {
        "group_map": {"N": 0, "M": 1, "S": 1},
        "filter_groups": None
    },
    "M_vs_S": {
        "group_map": {"N": 2, "M": 1, "S": 0},
        "filter_groups": 2
    }
}

group_map = case_dict[case]["group_map"]
filter_groups = case_dict[case]["filter_groups"]

def load_and_process_od_data(
    filepath,
    sheet_name=0,
    group_map=group_map
):
    """
    Load OD data from Excel and reshape it for CNN-style modeling.

    Parameters
    ----------
    filepath : str
        Path to the Excel file.
    sheet_name : str or int, default 0
        Sheet name or index to read.
    group_map : dict, default {"N": 0, "M": 1, "S": 2}
        Mapping from group letter to numeric label.

    Returns
    -------
    pd.DataFrame
        Processed dataframe with metadata columns followed by time columns.
    """

    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df = df.rename(columns={df.columns[0]: "time"})
    df = df.set_index("time")

    df_long = (
        df
        .reset_index()
        .melt(
            id_vars="time",
            var_name="sample",
            value_name="OD"
        )
    )

    df_long["group_letter"] = df_long["sample"].str.extract(r'^([NMS])')

    df_long["patient"] = df_long["sample"].str.extract(r'^([NMS]\d+)')

    df_long["repetition"] = (
        df_long["sample"]
        .str.extract(r'Replicate\s*(\d)')
        .astype(int)
    )

    df_long["group"] = df_long["group_letter"].map(group_map)

    df_cnn = (
        df_long
        .pivot_table(
            index=["patient", "repetition", "group"],
            columns="time",
            values="OD"
        )
        .reset_index()
    )

    meta_cols = ["patient", "repetition", "group"]
    time_cols = sorted(
        [c for c in df_cnn.columns if c not in meta_cols],
        key=lambda x: float(x)
    )
    df_cnn = df_cnn[meta_cols + time_cols]
    return df_cnn[df_cnn["group"] != filter_groups]


    
import glob
from collections import defaultdict

excel_files = [
    f for f in glob.glob("time serie/*.xlsx") if not f.startswith("~$")
]

if strain == None:
    pass
else:
    excel_files = [f"time serie/{strain} strain.xlsx"]

X_all, y_all, patients_all = [], [], []
rows = []
# lengths = []

# n_cols = None

# for file in excel_files:
#     df = load_and_process_od_data(file)
#     X = df.iloc[:, 3:].values
#     n_cols = X.shape[1] if n_cols is None else min(n_cols, X.shape[1])

# print("Using common columns:", n_cols)

# 2️⃣ Load again and keep ONLY common columns
for file in excel_files:
    df = load_and_process_od_data(file)

    y = df.iloc[:, 2].values
    # X = df.iloc[:, 3:].values[:, :n_cols]   # ✔ column intersection
    X = df.iloc[:, 3:].values
    patients = df.iloc[:, 1].values

    # X_all.append(X)
    # y_all.append(y)
    # patients_all.append(patients)

# # 3️⃣ Stack rows
# X = np.concatenate(X_all, axis=0)
# y = np.concatenate(y_all, axis=0)
# patients = np.concatenate(patients_all, axis=0)

# # ✅ Remove columns with ANY NaN (after concatenation)
# valid_cols = ~np.isnan(X).any(axis=0)
# X = X[:, valid_cols]

    # # ✅ Your original normalization
    # normalize_value = X.max()
    # X = X / normalize_value

    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True)

    # Add a small epsilon to avoid division by zero if a signal is perfectly flat
    X = (X - X_mean) / (X_std + 1e-8)

    # 4. Convert types and add channel dimension
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    X = X[:, np.newaxis, :]

    ray.shutdown()
    ray.init(include_dashboard=False, logging_level="ERROR")

    X_ref = ray.put(X)
    y_ref = ray.put(y)
    patients_ref = ray.put(patients)

    models = [CNN1D, TCN]  # LSTMNet, TCN

    futures = []

    for model in models:
        for _ in range(runs):
            futures.append(
                run_single_run_ray.remote(
                    model,
                    X_ref,
                    y_ref,
                    patients_ref,
                    epochs=epochs,
                    batch=batch,
                )
            )

    results = ray.get(futures)


    metrics = defaultdict(list)

    for model, (acc, prec, spec) in zip(
            [m for m in models for _ in range(runs)],
            results):

        metrics[model.__name__ + "_acc"].append(acc)
        metrics[model.__name__ + "_prec"].append(prec)
        metrics[model.__name__ + "_spec"].append(spec)


    cnn_acc = np.array(metrics["CNN1D_acc"])
    cnn_prec = np.array(metrics["CNN1D_prec"])
    cnn_spec = np.array(metrics["CNN1D_spec"])


    tcn_acc = np.array(metrics["TCN_acc"])
    tcn_prec = np.array(metrics["TCN_prec"])
    tcn_spec = np.array(metrics["TCN_spec"])

    ray.shutdown()

    print(f"Results for file: {file}")
    print("CNN:"
            f" Acc: {cnn_acc.mean():.3f} ± {cnn_acc.std():.3f},"
            f" Prec: {cnn_prec.mean():.3f} ± {cnn_prec.std():.3f},"
            f" Spec: {cnn_spec.mean():.3f} ± {cnn_spec.std():.3f}"  )

    print("TCN:"           
            f" Acc: {tcn_acc.mean():.3f} ± {tcn_acc.std():.3f},"
            f" Prec: {tcn_prec.mean():.3f} ± {tcn_prec.std():.3f},"
            f" Spec: {tcn_spec.mean():.3f} ± {tcn_spec.std():.3f}"  )

    rows.append({
    "case": case,

    "File": os.path.splitext(os.path.basename(file))[0],

    "CNN acc": cnn_acc.mean(),
    "CNN std acc": cnn_acc.std(),

    # "LSTM acc": lstm_acc.mean(),
    # "LSTM std acc": lstm_acc.std(),

    "TCN acc": tcn_acc.mean(),
    "TCN std acc": tcn_acc.std(),

    "CNN prec": cnn_prec.mean(),
    "CNN std prec": cnn_prec.std(),

    # "LSTM prec": lstm_prec.mean(),
    # "LSTM std prec": lstm_prec.std(),

    "TCN prec": tcn_prec.mean(),
    "TCN std prec": tcn_prec.std(),


    "CNN spec": cnn_spec.mean(),
    "CNN std spec": cnn_spec.std(),     

    # "LSTM spec": lstm_spec.mean(),
    # "LSTM std spec": lstm_spec.std(),

    "TCN spec": tcn_spec.mean(),            
    "TCN std spec": tcn_spec.std(),
    })


df = pd.DataFrame(rows)

df.to_csv(result_name, index=False, float_format="%.3f")



