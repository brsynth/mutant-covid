import pandas as pd

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def load_and_process_od_data(
    filepath,
    sheet_name=0,
    group_map={"N": 0, "M": 1, "S": 2}
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

    # Read Excel
    df = pd.read_excel(filepath, sheet_name=sheet_name)

    # Ensure first column is time
    df = df.rename(columns={df.columns[0]: "time"})
    df = df.set_index("time")

    # Long format
    df_long = (
        df
        .reset_index()
        .melt(
            id_vars="time",
            var_name="sample",
            value_name="OD"
        )
    )

    # Group letter (N, M, S)
    df_long["group_letter"] = df_long["sample"].str.extract(r'^([NMS])')

    # Patient number (e.g. N1, M3, S2)
    df_long["patient"] = df_long["sample"].str.extract(r'^([NMS]\d+)')

    # Replicate number
    df_long["repetition"] = (
        df_long["sample"]
        .str.extract(r'Replicate\s*(\d)')
        .astype(int)
    )

    # Numeric group label
    df_long["group"] = df_long["group_letter"].map(group_map)

    # Pivot to wide format (time series per sample)
    df_cnn = (
        df_long
        .pivot_table(
            index=["patient", "repetition", "group"],
            columns="time",
            values="OD"
        )
        .reset_index()
    )

    # Order columns: metadata first, then sorted time points
    meta_cols = ["patient", "repetition", "group"]
    time_cols = sorted(
        [c for c in df_cnn.columns if c not in meta_cols],
        key=lambda x: float(x)
    )

    return df_cnn[meta_cols + time_cols]





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
        # (B, C, T) → (B, T, C)
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
    correct, total = 0, 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model(Xb).argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total

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
    accs = []

    patients = np.array(patients)
    unique_patients = np.unique(patients)

    for _ in range(runs):

        # Split by patient (not by repetition)
        train_patients, test_patients = train_test_split(
            unique_patients,
            test_size=test_size,
            random_state=42,
            stratify=None,   # optional: see note below
        )

        # Mask samples belonging to selected patients
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

        acc = eval_accuracy(model, test_loader, device)
        accs.append(acc)

    return np.array(accs)


import glob

excel_files = [
    f for f in glob.glob("*.xlsx") if not f.startswith("~$")
]

for file in excel_files:

    df = load_and_process_od_data(file)
    # Extract from DataFrame
    y = df.iloc[:, 2].values          # labels
    X = df.iloc[:, 3:].values         # time series
    patients = df.iloc[:, 1].values   # patient IDs

    # Convert to float32
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    # Add channel dimension: (samples, channels, time)
    X = X[:, np.newaxis, :]

    cnn_acc  = run_experiment(CNN1D, X, y, patients)
    lstm_acc = run_experiment(LSTMNet, X, y, patients)
    tcn_acc  = run_experiment(TCN, X, y, patients)

    print(f"CNN  : {cnn_acc.mean():.3f} ± {cnn_acc.std():.3f}")
    print(f"LSTM : {lstm_acc.mean():.3f} ± {lstm_acc.std():.3f}")
    print(f"TCN  : {tcn_acc.mean():.3f} ± {tcn_acc.std():.3f}")


