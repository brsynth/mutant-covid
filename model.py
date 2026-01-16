import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(os.cpu_count())

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

import ray
import optuna

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CNN1D(nn.Module):
    def __init__(self, in_channels, num_classes, k1, k2, dropout):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, 16,
            kernel_size=k1,   # very smooth bias
            padding="same"
        )
        self.norm1 = nn.GroupNorm(4, 16)

        self.conv2 = nn.Conv1d(
            16, 32,
            kernel_size=k2,
            padding="same"
        )
        self.norm2 = nn.GroupNorm(4, 32)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = F.gelu(self.norm2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout=0.0,
    ):
        super().__init__()

        # causal padding
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

        self.norm = nn.GroupNorm(
            num_groups=4 if out_channels >= 32 else 2,
            num_channels=out_channels
        )

        self.dropout = nn.Dropout(dropout)

        # match channels for residual if needed
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        # main path
        out = self.conv(x)
        out = out[:, :, :-self.padding]  # remove future leakage
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)

        # residual path
        res = self.residual(x)

        return F.gelu(out + res)

class TCN(nn.Module):
    def __init__(self, in_channels, num_classes, k1, k2, dropout):
        super().__init__()

        self.block1 = TCNBlock(
            in_channels, 16,
            kernel_size=k1,
            dilation=1
        )

        self.block2 = TCNBlock(
            16, 32,
            kernel_size=k2,
            dilation=2,
            dropout=dropout
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def train_epoch(model, loader, optimizer, criterion):
    model.train()

    for Xb, yb in loader:
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

def eval_accuracy(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            logits = model(Xb)
            preds = logits.argmax(dim=1)

            y_true.extend(yb.numpy())
            y_pred.extend(preds.numpy())
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
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

    return balanced_accuracy, macro_precision, macro_specificity


##### OPTUNA
def inner_objective(trial, X_train, y_train, patients):

    model_type = trial.suggest_categorical(
        "model_type", ["cnn", "tcn"]
    )

    # ---- shared hyperparameters ----
    lr = trial.suggest_float("lr", 5e-1, 1e-3, log=True)
    weight_decay = trial.suggest_float("wd", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    k1 = trial.suggest_int("k1", 7, 15, step=2)
    k2 = trial.suggest_int("k2", 7, 15, step=2)

    # ---- model-specific branches ----
    if model_type == "cnn":
        ModelClass = lambda in_ch, num_cl: CNN1D(in_ch, num_cl, k1, k2, dropout)
    elif model_type == "tcn":  # ---- TCN ----
        ModelClass = lambda in_ch, num_cl: TCN(in_ch, num_cl, k1, k2, dropout)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        train_ds = TimeSeriesDataset(X_tr, y_tr)
        val_ds = TimeSeriesDataset(X_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        in_channels = X_tr.shape[1]
        num_classes = len(np.unique(y_tr))

        model = ModelClass(in_channels, num_classes)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

        for _ in range(30):
            train_epoch(model, train_loader, optimizer, criterion)

        balanced_accuracy, _, _ = eval_accuracy(model, val_loader)
        scores.append(balanced_accuracy)

        trial.report(balanced_accuracy, step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))

def build_model_from_params(
    params,
    in_channels,
    num_classes,
):
    model_type = params["model_type"]
    k1 = params["k1"]
    k2 = params["k2"]
    dropout = params["dropout"]

    if model_type == "cnn":
        model = CNN1D(in_channels, num_classes, k1, k2, dropout)
    elif model_type == "tcn":
        model = TCN(in_channels, num_classes, k1, k2, dropout)
    return model

def run_single_run(
    ModelClass,
    X,
    y,
    patients,
    epochs= 50,
    lr = 0.001, batch = 64,
    test_size=0.2,
):
    in_channels = X.shape[1]
    num_classes = len(np.unique(y))

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

    study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=1,
    ),
)

    study.optimize(
        lambda t: inner_objective(
            t,
            X_tr,
            y_tr,
            patients_tr,
            in_channels=X_tr.shape[1],
        ),
        n_trials=30,
    )

    best_params = study.best_params

    model = build_model_from_params(
        best_params,        
        in_channels,
        num_classes,
    )

    train_ds = TimeSeriesDataset(X_tr, y_tr)
    test_ds  = TimeSeriesDataset(X_te, y_te)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=best_params["lr"],
                                 weight_decay=best_params["wd"])
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

    for _ in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion)

    return eval_accuracy(model, test_loader)

@ray.remote(num_cpus=1)  
def run_single_run_ray(
    ModelClass,
    X,
    y,
    patients,
    epochs=50,
    lr=0.001,
    batch=64,
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
        epochs = epochs,
        lr = lr,
        batch = batch,
        test_size=test_size,
    )

######## OPTUNA



