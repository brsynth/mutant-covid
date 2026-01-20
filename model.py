import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
torch.set_num_threads(1)
torch.set_num_interop_threads(os.cpu_count())

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

import ray
import optuna

#TO do: add patient when split for validation
# tune model with hyperparameter?

device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CNN1D(nn.Module):
    def __init__(self, in_channels, num_classes, k1, k2, out_channels=32, spatial_resolution=12, dropout=0.5):
        super().__init__()
        
        # Branch 1: Large Kernel (The 8-degree Spline shape)
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//2, kernel_size=k1, padding="same"),
            nn.GroupNorm(1, out_channels//2), 
            nn.GELU()
        )
        
        # Branch 2: Small Kernel (The Local Variation/Jitter)
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//2, kernel_size=k2, padding="same"),
            nn.GroupNorm(1, out_channels//2),
            nn.GELU()
        )

        self.pool = nn.AdaptiveAvgPool1d(spatial_resolution)
        
        # The Classifier: High dropout is the cure for High Std Dev
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout), # Heavy dropout to stop memorization
            nn.Linear(out_channels * spatial_resolution, 32),
            nn.GELU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # Parallel processing prevents noise accumulation
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        
        # Merge the "Global" and "Local" views
        x = torch.cat([x1, x2], dim=1)
        
        x = self.pool(x)
        return self.classifier(x)



class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation = 1, dropout=0.2):
        super().__init__()
        self.padding_size = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              dilation=dilation, padding=0)
        
        # GroupNorm is good for your small batch/small sample size
        self.norm = nn.GroupNorm(num_groups=2, num_channels=out_channels)
        
        # 1. Feature Dropout: Use Dropout1d for time-series.
        # It drops entire channels, forcing the model to learn 
        # multiple independent features of the 8-degree spline.
        self.dropout = nn.Dropout1d(dropout) 
        
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x_padded = F.pad(x, (self.padding_size, 0))
        
        out = self.conv(x_padded)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out) # Dropout applied to features
        
        # Adding residual before final GELU to preserve signal flow
        return F.gelu(out + res)

class TCN(nn.Module):
    def __init__(self, in_channels, num_classes, k1, k2, dilation = 1, out_channels=32, spatial_resolution=12, dropout=0.3):
        super().__init__()
        self.block1 = TCNBlock(in_channels, out_channels, k1, dilation, dropout=dropout*0.5)
        self.block2 = TCNBlock(out_channels, out_channels, kernel_size=k2, dilation=1, dropout=dropout*0.5)
        
        self.spatial_resolution = spatial_resolution
        self.pool_avg = nn.AdaptiveAvgPool1d(spatial_resolution)
        self.pool_max = nn.AdaptiveMaxPool1d(spatial_resolution)
        
        # 2. Classification Dropout: 
        # Since we flatten (Channels * Res), this layer has many weights.
        # We need a stronger Dropout here to prevent memorizing "Point X = Class Y".
        self.dropout_final = nn.Dropout(dropout)
        
        # Input to FC is out_channels * 2 (because of Avg + Max pool) * spatial_resolution
        self.fc = nn.Linear(out_channels * 2 * spatial_resolution, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        
        # Hybrid Pooling: Captures the 'energy' and the 'peak' of the spline
        avg_p = self.pool_avg(x)
        max_p = self.pool_max(x)
        x = torch.cat([avg_p, max_p], dim=1) # [Batch, out_channels * 2, spatial_resolution]
        
        x = torch.flatten(x, 1)
        x = self.dropout_final(x) # Protecting the linear layer
        return self.fc(x)


def train_epoch(model, loader, optimizer, criterion):
    model.train()

    for Xb, yb in loader:
        # scaling_factor = torch.FloatTensor(1).uniform_(0.9, 1.1).to(Xb.device)
        # Xb = Xb * scaling_factor
        # Xb = Xb + torch.randn_like(Xb) * 0.01
        Xb, yb = Xb.to(device), yb.to(device)
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
def inner_objective(trial, X_train, y_train, ModelClass = None):
    # Force single threading for Ray/CPU stability, pls don't remove
    torch.set_num_threads(1)
    device = torch.device("cpu")

    in_channels = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    lr = trial.suggest_float("lr", 5e-3, 1e-1, log=True)

    # if ModelClass == TCN:
    k1 = trial.suggest_int("k1", 5, 20, step=2)
    k2 = trial.suggest_int("k2", 3, 15, step=2)
    spatial_resolution = trial.suggest_int("res", 10, 26, step=2)
    dropout = trial.suggest_float("do", 0.1, 0.3)
    out_channels = trial.suggest_categorical("out_channels", [16, 32, 64])
    # dilation = trial.suggest_categorical("dilation", [1, 2])
    model = ModelClass(in_channels, num_classes, 
                        k1 = k1,
                        k2 = k2,
                        out_channels = out_channels, 
                        spatial_resolution = spatial_resolution,
                        dropout = dropout).to(device)
    # elif ModelClass == CNN1D:
    #     k1 = trial.suggest_int("k1", 7, 15, step=2)
    #     k2 = trial.suggest_int("k2", 5, 10, step=2)

    #     model = ModelClass(in_channels, num_classes, k1, k2).to(device)
    # else:
    #     raise ValueError(f"Unknown model_type: {ModelClass}")

    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

    train_ds = TimeSeriesDataset(X_tr, y_tr)
    val_ds = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # model = ModelClass(in_channels, num_classes, k1, k2).to(device)
    epochs = 20
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=5e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0
    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()
        balanced_accuracy, _, _ = eval_accuracy(model, val_loader)

        trial.report(balanced_accuracy, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        best_val_acc = max(balanced_accuracy, best_val_acc)

    return best_val_acc

def build_model_from_params(
    params,
    in_channels,
    num_classes,
    ModelClass = None,
):
    # if ModelClass == TCN:
    k1 = params["k1"]
    k2 = params["k2"]
    out_channels = params["out_channels"]
    spatial_resolution = params["res"]
    dropout = params["do"]
    model = ModelClass(in_channels, num_classes,
                        out_channels = out_channels,
                        spatial_resolution = spatial_resolution, 
                        k1 = k1,
                        k2 = k2,
                        dropout = dropout 
                        ).to(device)

    # elif ModelClass == CNN1D:
    #     k1 = params["k1"]
    #     k2 = params["k2"]
    #     model = ModelClass(in_channels, num_classes, k1, k2)
    return model

def run_single_run(
    ModelClass,
    X,
    y,
    patients,
    epochs= 50,
    batch = 128,
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
    sampler=optuna.samplers.TPESampler(),
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
            ModelClass,
        ),
        n_trials=20,
    )

    best_params = study.best_params
    print(f"Best parameters: {best_params}")

    model = build_model_from_params(
        best_params,        
        in_channels,
        num_classes,
        ModelClass=ModelClass,
    )

    train_ds = TimeSeriesDataset(X_tr, y_tr)
    test_ds  = TimeSeriesDataset(X_te, y_te)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=best_params["lr"],
                                 weight_decay=5e-3)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

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
    batch=128,
    test_size=0.2,
):
    import os
    import torch
    import random
    import numpy as np

    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

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
        batch = batch,
        test_size=test_size,
    )

######## OPTUNA



