import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(os.cpu_count())

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, precision_score, confusion_matrix

import ray
ray.init(ignore_reinit_error=True)

case = "all"  # Options: "all", "N_vs_P", "M_vs_S"
result_name = f"classification_result/baseline_{case}.csv"

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

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class LSTMNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            batch_first=True
        )
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class TCN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 3, padding=2, dilation=1)
        self.conv2 = nn.Conv1d(32, 32, 3, padding=4, dilation=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=8, dilation=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()

def eval_accuracy(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model(Xb).argmax(1)
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    balanced_accuaracy = balanced_accuracy_score(y_true, y_pred)
    macro_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]

    specificities = []
    for i in range(num_classes):
        TN = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        FP = cm[:, i].sum() - cm[i, i]
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificities.append(specificity)
    macro_specificity = np.mean(specificities)  

    return balanced_accuaracy, macro_precision, macro_specificity

def run_experiment(
    ModelClass,
    X,
    y,
    patients,
    runs=100,
    epochs=30,
    test_size=0.2
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = len(np.unique(y))
    accs, precs, specs = [], [], []

    patients = np.array(patients)
    unique_patients = np.unique(patients)

    for _ in range(runs):

        train_patients, test_patients = train_test_split(
            unique_patients,
            test_size=test_size,
            random_state=None,
            stratify=None,  
        )

        train_mask = np.isin(patients, train_patients)
        test_mask  = np.isin(patients, test_patients)

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask],  y[test_mask]

        train_ds = TimeSeriesDataset(X_tr, y_tr)
        test_ds  = TimeSeriesDataset(X_te, y_te)

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        test_loader  = DataLoader(test_ds, batch_size=32)

        model = ModelClass(n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            train_epoch(model, train_loader, optimizer, criterion, device)

        balanced_accuaracy, macro_precision, macro_specificity = eval_accuracy(model, test_loader, device)
        accs.append(balanced_accuaracy)
        precs.append(macro_precision)
        specs.append(macro_specificity)

    return np.array(accs), np.array(precs), np.array(specs)

def run_single_run(
    ModelClass,
    X,
    y,
    patients,
    epochs=30,
    test_size=0.2,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = len(np.unique(y))

    patients = np.array(patients)
    unique_patients = np.unique(patients)

    train_patients, test_patients = train_test_split(
        unique_patients,
        test_size=test_size,
        random_state=None,
        stratify=None,
    )

    train_mask = np.isin(patients, train_patients)
    test_mask  = np.isin(patients, test_patients)

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask],  y[test_mask]

    train_ds = TimeSeriesDataset(X_tr, y_tr)
    test_ds  = TimeSeriesDataset(X_te, y_te)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=32)

    model = ModelClass(n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, device)

    return eval_accuracy(model, test_loader, device)

@ray.remote(num_cpus=1, num_gpus=0)  # set num_gpus=1 if needed
def run_single_run_ray(
    ModelClass,
    X,
    y,
    patients,
    epochs=30,
    test_size=0.2,
):
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    import random, numpy as np, torch
    seed = np.random.randint(0, 1_000_000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return run_single_run(
        ModelClass,
        X,
        y,
        patients,
        epochs,
        test_size,
    )


    
import glob
from collections import defaultdict

excel_files = [
    f for f in glob.glob("*.xlsx") if not f.startswith("~$")
]

rows = []

for file in excel_files:

    df = load_and_process_od_data(file)
    # Extract from DataFrame
    y = df.iloc[:, 2].values          
    X = df.iloc[:, 3:].values         
    patients = df.iloc[:, 1].values   

    # Convert to float32
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    # Add channel dimension: (samples, channels, time)
    X = X[:, np.newaxis, :]

    # cnn_acc, cnn_prec, cnn_spec = run_experiment(CNN1D, X, y, patients)
    # lstm_acc, lstm_prec, lstm_spec = run_experiment(LSTMNet, X, y, patients)
    # tcn_acc, tcn_prec, tcn_spec = run_experiment(TCN, X, y, patients)

    # print(f"Results for file: {file}")
    # print("CNN:"
    #       f" Acc: {cnn_acc.mean():.3f} ± {cnn_acc.std():.3f},"
    #       f" Prec: {cnn_prec.mean():.3f} ± {cnn_prec.std():.3f},"
    #       f" Spec: {cnn_spec.mean():.3f} ± {cnn_spec.std():.3f}"  )
    # print("LSTM:"
    #       f" Acc: {lstm_acc.mean():.3f} ± {lstm_acc.std():.3f},"
    #       f" Prec: {lstm_prec.mean():.3f} ± {lstm_prec.std():.3f},"
    #       f" Spec: {lstm_spec.mean():.3f} ± {lstm_spec.std():.3f}"  )
    # print("TCN:"           
    #       f" Acc: {tcn_acc.mean():.3f} ± {tcn_acc.std():.3f},"
    #       f" Prec: {tcn_prec.mean():.3f} ± {tcn_prec.std():.3f},"
    #       f" Spec: {tcn_spec.mean():.3f} ± {tcn_spec.std():.3f}"  )
    
    # rows.append({
    # "File": os.path.splitext(os.path.basename(file))[0],

    # "CNN acc": cnn_acc.mean(),
    # "CNN std acc": cnn_acc.std(),

    # "LSTM acc": lstm_acc.mean(),
    # "LSTM std acc": lstm_acc.std(),

    # "TCN acc": tcn_acc.mean(),
    # "TCN std acc": tcn_acc.std(),
    # })
    ray.shutdown()
    ray.init()

    X_ref = ray.put(X)
    y_ref = ray.put(y)
    patients_ref = ray.put(patients)

    models = [CNN1D, LSTMNet, TCN]
    runs = 100

    futures = []

    for model in models:
        for _ in range(runs):
            futures.append(
                run_single_run_ray.remote(
                    model,
                    X_ref,
                    y_ref,
                    patients_ref,
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

    lstm_acc = np.array(metrics["LSTMNet_acc"])
    lstm_prec = np.array(metrics["LSTMNet_prec"])
    lstm_spec = np.array(metrics["LSTMNet_spec"])

    tcn_acc = np.array(metrics["TCN_acc"])
    tcn_prec = np.array(metrics["TCN_prec"])
    tcn_spec = np.array(metrics["TCN_spec"])

    ray.shutdown()

    print(f"Results for file: {file}")
    print("CNN:"
          f" Acc: {cnn_acc.mean():.3f} ± {cnn_acc.std():.3f},"
          f" Prec: {cnn_prec.mean():.3f} ± {cnn_prec.std():.3f},"
          f" Spec: {cnn_spec.mean():.3f} ± {cnn_spec.std():.3f}"  )
    print("LSTM:"
          f" Acc: {lstm_acc.mean():.3f} ± {lstm_acc.std():.3f},"
          f" Prec: {lstm_prec.mean():.3f} ± {lstm_prec.std():.3f},"
          f" Spec: {lstm_spec.mean():.3f} ± {lstm_spec.std():.3f}"  )
    print("TCN:"           
          f" Acc: {tcn_acc.mean():.3f} ± {tcn_acc.std():.3f},"
          f" Prec: {tcn_prec.mean():.3f} ± {tcn_prec.std():.3f},"
          f" Spec: {tcn_spec.mean():.3f} ± {tcn_spec.std():.3f}"  )
    
    rows.append({
    "case": case,
    
    "File": os.path.splitext(os.path.basename(file))[0],

    "CNN acc": cnn_acc.mean(),
    "CNN std acc": cnn_acc.std(),

    "LSTM acc": lstm_acc.mean(),
    "LSTM std acc": lstm_acc.std(),

    "TCN acc": tcn_acc.mean(),
    "TCN std acc": tcn_acc.std(),

    "CNN prec": cnn_prec.mean(),
    "CNN std prec": cnn_prec.std(),

    "LSTM prec": lstm_prec.mean(),
    "LSTM std prec": lstm_prec.std(),

    "TCN prec": tcn_prec.mean(),
    "TCN std prec": tcn_prec.std(),

    "CNN spec": cnn_spec.mean(),
    "CNN std spec": cnn_spec.std(),     

    "LSTM spec": lstm_spec.mean(),
    "LSTM std spec": lstm_spec.std(),

    "TCN spec": tcn_spec.mean(),            
    "TCN std spec": tcn_spec.std(),
    })


df = pd.DataFrame(rows)

df.to_csv(result_name, index=False, float_format="%.3f")



