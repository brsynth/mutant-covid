from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Threading / performance safety
# -----------------------------
torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


# -----------------------------
# Shared utilities
# -----------------------------
def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int, device: Optional[torch.device] = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def make_torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def validate_normalize_channels(normalize_channels: Optional[Sequence[bool]], n_channels: int) -> np.ndarray:
    if normalize_channels is None:
        return np.zeros(int(n_channels), dtype=bool)

    mask = np.asarray(normalize_channels, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(f"normalize_channels must be 1D, got shape {mask.shape}")
    if len(mask) != int(n_channels):
        raise ValueError(
            f"normalize_channels length ({len(mask)}) does not match number of channels ({n_channels})."
        )
    return mask


def fit_channelwise_normalizer(
    X_train: np.ndarray,
    normalize_channels: Optional[Sequence[bool]] = None,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train = np.asarray(X_train, dtype=np.float32)
    if X_train.ndim != 3:
        raise ValueError(f"X_train must be [N, C, T], got shape {X_train.shape}")

    _, C, _ = X_train.shape
    mask = validate_normalize_channels(normalize_channels, C)

    mean = np.zeros((1, C, 1), dtype=np.float32)
    std = np.ones((1, C, 1), dtype=np.float32)

    if np.any(mask):
        ch_mean = X_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        ch_std = X_train.std(axis=(0, 2), keepdims=True).astype(np.float32)
        ch_std = np.maximum(ch_std, eps)

        mean[:, mask, :] = ch_mean[:, mask, :]
        std[:, mask, :] = ch_std[:, mask, :]

    return mean, std, mask.astype(bool)


def apply_channelwise_normalizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return ((X - mean) / std).astype(np.float32)


def fit_tabular_normalizer(
    X_train: np.ndarray,
    normalize_tabular: bool = True,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    X_train = np.asarray(X_train, dtype=np.float32)
    if X_train.ndim != 2:
        raise ValueError(f"X_train must be [N, P], got shape {X_train.shape}")

    P = X_train.shape[1]
    mean = np.zeros((1, P), dtype=np.float32)
    std = np.ones((1, P), dtype=np.float32)

    if normalize_tabular:
        mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
        std = X_train.std(axis=0, keepdims=True).astype(np.float32)
        std = np.maximum(std, eps)

    return mean, std, bool(normalize_tabular)


def apply_tabular_normalizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return ((X - mean) / std).astype(np.float32)


def _macro_specificity(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities: List[float] = []
    for i in range(num_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        specificities.append(float(spec))
    return float(np.mean(specificities)) if specificities else 0.0


def compute_patient_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    proba = np.asarray(proba, dtype=float)

    num_classes = int(proba.shape[1])

    out: Dict[str, Any] = {}
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
        per_class: Dict[int, float] = {}
        for c in range(num_classes):
            y_bin = (y_true == c).astype(int)
            try:
                per_class[c] = float(roc_auc_score(y_bin, proba[:, c]))
            except Exception:
                per_class[c] = float("nan")
        out["AUC_per_class"] = per_class

    return out


def _patient_labels(patients: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    patients = np.asarray(patients).astype(str)
    y = np.asarray(y).astype(int)

    unique_patients = np.unique(patients)
    labels = []
    for p in unique_patients:
        yp = y[patients == p]
        if yp.size == 0:
            raise ValueError("Empty patient group found unexpectedly.")
        u = np.unique(yp)
        if u.size != 1:
            raise ValueError(f"Patient {p} has inconsistent labels across replicates: {u.tolist()}")
        labels.append(int(u[0]))
    return unique_patients, np.asarray(labels, dtype=int)


def stratified_patient_split(
    patients: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    unique_patients, labels = _patient_labels(patients, y)
    if unique_patients.shape[0] < 2:
        raise ValueError("Not enough unique patients to split.")
    _, counts = np.unique(labels, return_counts=True)
    if counts.min() < 2:
        raise ValueError(
            "Not enough patients in at least one class for stratified split. "
            f"Patient-level class counts: {dict(zip(*np.unique(labels, return_counts=True)))}"
        )
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss.split(unique_patients, labels))
    return unique_patients[train_idx], unique_patients[test_idx]


def masks_from_patients(
    patients: np.ndarray,
    train_patients: np.ndarray,
    test_patients: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    train_mask = np.isin(patients, train_patients)
    test_mask = np.isin(patients, test_patients)
    return train_mask, test_mask


@dataclass
class SplitSpec:
    split_id: int
    seed: int
    test_size: float
    train_patients: List[Any]
    test_patients: List[Any]


def generate_kfold_split_file(
    split_file: str,
    patients: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    base_seed: int = 2026,
) -> str:
    patients = np.asarray(patients).astype(str)
    y = np.asarray(y).astype(int)

    unique_patients, labels = _patient_labels(patients, y)

    _, counts = np.unique(labels, return_counts=True)
    if counts.min() < 2:
        raise ValueError(
            "Not enough patients in at least one class for StratifiedKFold. "
            f"Patient-level class counts: {dict(zip(*np.unique(labels, return_counts=True)))}"
        )

    if int(n_splits) > int(counts.min()):
        raise ValueError(
            f"n_splits={n_splits} is too large for the smallest class (min patients in a class={counts.min()}). "
            "Reduce n_splits."
        )

    skf = StratifiedKFold(n_splits=int(n_splits), shuffle=bool(shuffle), random_state=int(base_seed))

    splits: List[Dict[str, Any]] = []
    for split_id, (train_idx, test_idx) in enumerate(skf.split(unique_patients, labels)):
        tr_p = unique_patients[train_idx]
        te_p = unique_patients[test_idx]
        splits.append(
            {
                "split_id": int(split_id),
                "seed": int(base_seed),
                "test_size": float(len(te_p) / len(unique_patients)),
                "train_patients": tr_p.tolist(),
                "test_patients": te_p.tolist(),
            }
        )

    payload = {
        "format_version": 1,
        "kind": "StratifiedKFold_patient",
        "n_splits": int(n_splits),
        "base_seed": int(base_seed),
        "splits": splits,
    }

    os.makedirs(os.path.dirname(split_file) or ".", exist_ok=True)
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return split_file


def load_split_file(split_file: str) -> Dict[str, Any]:
    with open(split_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported split file format_version.")
    return payload


def get_split_from_file(split_payload: Dict[str, Any], split_id: int) -> SplitSpec:
    match = None
    for s in split_payload["splits"]:
        if int(s["split_id"]) == int(split_id):
            match = s
            break
    if match is None:
        raise KeyError(f"split_id={split_id} not found in split file.")

    return SplitSpec(
        split_id=int(match["split_id"]),
        seed=int(match["seed"]),
        test_size=float(match.get("test_size", 0.0)),
        train_patients=list(match["train_patients"]),
        test_patients=list(match["test_patients"]),
    )


def assert_split_compatible(patients: np.ndarray, spec: SplitSpec) -> None:
    pats = set(map(str, np.asarray(patients).astype(str).tolist()))
    tr = set(map(str, spec.train_patients))
    te = set(map(str, spec.test_patients))
    missing = (tr | te) - pats
    if missing:
        ex = list(sorted(missing))[:10]
        raise ValueError(
            "Split file is incompatible with this dataset: some split patients are missing.\n"
            f"Examples missing patients: {ex}\n"
            "If you insist on shared splits across strains, ensure patient IDs are identical across files."
        )


RES_UNIVERSE = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 44, 50, 64]


def tcn_receptive_field(k1: int, k2: int, d1: int, d2: int = 1) -> int:
    return 1 + (k1 - 1) * d1 + (k2 - 1) * d2


# -----------------------------
# Time-series pipeline
# -----------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, patients: Optional[np.ndarray] = None):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays.")
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.patients = None if patients is None else np.asarray(patients)

        if self.patients is not None and len(self.patients) != len(self.y):
            raise ValueError("patients must have the same length as y.")

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        if self.patients is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx], self.patients[idx]


def _unpack_ts_batch(batch):
    if len(batch) == 2:
        Xb, yb = batch
        pb = None
    else:
        Xb, yb, pb = batch
    return Xb, yb, pb


class CNN1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        k1: int,
        k2: int,
        out_channels: int = 32,
        spatial_resolution: int = 12,
        dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 2, kernel_size=k1, padding="same"),
            nn.GroupNorm(1, out_channels // 2),
            nn.GELU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 2, kernel_size=k2, padding="same"),
            nn.GroupNorm(1, out_channels // 2),
            nn.GELU(),
        )
        self.pool_avg = nn.AdaptiveAvgPool1d(spatial_resolution)
        self.pool_max = nn.AdaptiveMaxPool1d(spatial_resolution)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 2 * spatial_resolution, 32),
            nn.GELU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat([x1, x2], dim=1)
        avg_p = self.pool_avg(x)
        max_p = self.pool_max(x)
        x = torch.cat([avg_p, max_p], dim=1)
        x = F.gelu(x)
        return self.classifier(x)


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.padding_size = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)

        num_groups = 2 if out_channels % 2 == 0 else 1
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        self.dropout = nn.Dropout1d(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x_padded = F.pad(x, (self.padding_size, 0))
        out = self.conv(x_padded)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        return F.gelu(out + res)


class TCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        k1: int,
        k2: int,
        dilation: int = 1,
        out_channels: int = 32,
        spatial_resolution: int = 12,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.block1 = TCNBlock(in_channels, out_channels, k1, dilation=dilation, dropout=dropout * 0.5)
        self.block2 = TCNBlock(out_channels, out_channels, k2, dilation=1, dropout=dropout * 0.5)

        self.pool_avg = nn.AdaptiveAvgPool1d(spatial_resolution)
        self.pool_max = nn.AdaptiveMaxPool1d(spatial_resolution)
        self.dropout_final = nn.Dropout(dropout)
        self.fc = nn.Linear(out_channels * 2 * spatial_resolution, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        avg_p = self.pool_avg(x)
        max_p = self.pool_max(x)
        x = torch.cat([avg_p, max_p], dim=1)
        x = torch.flatten(x, 1)
        x = self.dropout_final(x)
        return self.fc(x)


def train_epoch_ts(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    accum_steps: int = 1,
) -> None:
    model.train()
    accum_steps = max(1, int(accum_steps))

    optimizer.zero_grad(set_to_none=True)

    step = 0
    acc_count = 0

    for step, batch in enumerate(loader, start=1):
        Xb, yb, _ = _unpack_ts_batch(batch)
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(Xb)
        loss = criterion(logits, yb)

        acc_count += 1
        loss = loss / accum_steps
        loss.backward()

        if acc_count == accum_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            acc_count = 0

    if step > 0 and acc_count > 0:
        scale = float(accum_steps) / float(acc_count)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


@torch.no_grad()
def eval_patient_outputs_ts(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    verify_patient_labels: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    probs_by_pat: Dict[str, List[np.ndarray]] = {}
    y_by_pat: Dict[str, int] = {}

    for batch in loader:
        Xb, yb, pb = _unpack_ts_batch(batch)
        if pb is None:
            raise ValueError("eval_patient_outputs_ts requires patient ids in the dataset.")

        Xb = Xb.to(device, non_blocking=True)
        logits = model(Xb)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

        y_np = yb.detach().cpu().numpy()
        pb_np = np.asarray(pb)

        for i in range(len(y_np)):
            pid = str(pb_np[i])
            probs_by_pat.setdefault(pid, []).append(probs[i])
            yi = int(y_np[i])
            if pid not in y_by_pat:
                y_by_pat[pid] = yi
            elif verify_patient_labels and y_by_pat[pid] != yi:
                raise ValueError(f"Inconsistent labels for patient_id={pid}: {y_by_pat[pid]} vs {yi}")

    patient_ids: List[str] = []
    y_true: List[int] = []
    mean_probs_list: List[np.ndarray] = []

    for pid in sorted(probs_by_pat.keys()):
        patient_ids.append(pid)
        y_true.append(int(y_by_pat[pid]))
        mean_probs_list.append(np.mean(np.stack(probs_by_pat[pid], axis=0), axis=0))

    proba = np.stack(mean_probs_list, axis=0).astype(np.float32)
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.argmax(proba, axis=1).astype(int)

    return np.asarray(patient_ids), y_true_arr, y_pred_arr, proba


def train_fixed_epochs_ts(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    epochs: int,
    effective_batch_size: int,
    device: torch.device,
    lr: float,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.0,
    seed: int = 0,
    max_physical_batch: int = 32,
) -> nn.Module:
    train_ds = TimeSeriesDataset(X_tr, y_tr.astype(np.int64))
    g = make_torch_generator(seed)

    n_tr = int(len(train_ds))
    eff_bs = max(1, int(effective_batch_size))

    phys_bs = min(eff_bs, int(max_physical_batch), n_tr)
    phys_bs = max(1, int(phys_bs))

    accum_steps = int(np.ceil(eff_bs / phys_bs))
    accum_steps = max(1, accum_steps)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(phys_bs),
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=g,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)))
    criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))

    for _ in range(int(epochs)):
        train_epoch_ts(
            model,
            train_loader,
            optimizer,
            criterion,
            device=device,
            accum_steps=accum_steps,
        )
        scheduler.step()

    return model


def suggest_params_ts(trial: optuna.Trial, ModelClass: Any, *, max_train_epochs: int) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    lo = max(5, int(max_train_epochs // 4))
    hi = max(5, int(max_train_epochs))
    step = max(1, int(max_train_epochs // 10))
    params["train_epochs"] = int(trial.suggest_int("train_epochs", lo, hi, step=step))
    params["batch"] = trial.suggest_categorical("batch", [4, 8, 16, 32, 64])
    params["lr"] = trial.suggest_float("lr", 3e-4, 1e-2, log=True)
    params["k1"] = trial.suggest_int("k1", 5, 21, step=2)
    params["k2"] = trial.suggest_int("k2", 3, 15, step=2)
    params["do"] = trial.suggest_float("do", 0.05, 0.35)
    params["out_channels"] = trial.suggest_categorical("out_channels", [16, 32, 64])
    params["res"] = int(trial.suggest_categorical("res", RES_UNIVERSE))
    params["dilation"] = int(trial.suggest_categorical("dilation", [1, 2, 4, 8])) if ModelClass is TCN else 1
    return params


def sanitize_params_for_data_ts(params: Dict[str, Any], ModelClass: Any, T: int) -> Dict[str, Any]:
    p = dict(params)

    r = int(p.get("res", 12))
    r = min(r, int(T))
    if int(p.get("out_channels", 32)) == 64:
        r = min(r, 36)
    r = max(8, r)
    if r % 2 == 1:
        r -= 1
    p["res"] = int(r)

    if ModelClass is TCN:
        d = int(p.get("dilation", 1))
        rf = tcn_receptive_field(int(p["k1"]), int(p["k2"]), int(d), d2=1)
        mult = 1.0 if T <= 120 else 1.25
        limit = int(mult * T)
        if rf > limit:
            for dd in [4, 2, 1]:
                rf_dd = tcn_receptive_field(int(p["k1"]), int(p["k2"]), dd, d2=1)
                if rf_dd <= limit:
                    d = dd
                    break
        p["dilation"] = int(d)

    return p


def build_model_from_params_ts(
    params: Dict[str, Any],
    in_channels: int,
    num_classes: int,
    ModelClass: Any,
    device: torch.device,
) -> nn.Module:
    kwargs = dict(
        in_channels=in_channels,
        num_classes=num_classes,
        k1=int(params["k1"]),
        k2=int(params["k2"]),
        out_channels=int(params["out_channels"]),
        spatial_resolution=int(params["res"]),
        dropout=float(params["do"]),
    )
    if ModelClass is TCN:
        kwargs["dilation"] = int(params.get("dilation", 1))
    return ModelClass(**kwargs).to(device)


def inner_objective_nested_ts(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    patients_train: np.ndarray,
    ModelClass: Any,
    device: torch.device,
    *,
    max_train_epochs: int,
    n_inner_splits: int,
    base_seed: int,
    normalize_channels: Optional[Sequence[bool]],
) -> float:
    in_channels = int(X_train.shape[1])
    num_classes = int(len(np.unique(y_train)))
    T = int(X_train.shape[2])

    params = suggest_params_ts(trial, ModelClass, max_train_epochs=int(max_train_epochs))
    params = sanitize_params_for_data_ts(params, ModelClass, T)

    patients_train = np.asarray(patients_train).astype(str)
    y_train = np.asarray(y_train).astype(int)

    unique_pats, pat_labels = _patient_labels(patients_train, y_train)
    _, counts = np.unique(pat_labels, return_counts=True)

    if counts.min() < 2:
        raise ValueError(
            "Inner CV impossible: not enough patients in at least one class. "
            f"Patient-level class counts: {dict(zip(*np.unique(pat_labels, return_counts=True)))}"
        )

    K = int(n_inner_splits)
    if K < 2:
        raise ValueError("n_inner_splits must be >= 2 for StratifiedKFold inner CV.")
    if K > int(counts.min()):
        raise ValueError(f"Inner CV impossible: n_inner_splits={K} > min patients in a class={counts.min()}.")

    fold_seed = int(base_seed + trial.number * 10_000)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=fold_seed)

    scores: List[float] = []

    for fold_idx, (tr_pat_idx, va_pat_idx) in enumerate(skf.split(unique_pats, pat_labels)):
        seed_k = int(base_seed + trial.number * 100 + fold_idx)
        seed_everything(seed_k, device=device)

        tr_p = unique_pats[tr_pat_idx]
        va_p = unique_pats[va_pat_idx]
        tr_mask, va_mask = masks_from_patients(patients_train, tr_p, va_p)

        X_tr, y_tr, pat_tr = X_train[tr_mask], y_train[tr_mask], patients_train[tr_mask]
        X_va, y_va, pat_va = X_train[va_mask], y_train[va_mask], patients_train[va_mask]

        mean, std, _ = fit_channelwise_normalizer(X_tr, normalize_channels=normalize_channels)
        X_tr = apply_channelwise_normalizer(X_tr, mean, std)
        X_va = apply_channelwise_normalizer(X_va, mean, std)

        model = build_model_from_params_ts(params, in_channels, num_classes, ModelClass, device=device)
        model = train_fixed_epochs_ts(
            model,
            X_tr,
            y_tr,
            epochs=int(params["train_epochs"]),
            effective_batch_size=int(params["batch"]),
            device=device,
            lr=float(params["lr"]),
            weight_decay=1e-4,
            label_smoothing=0.0,
            seed=seed_k,
            max_physical_batch=32,
        )

        val_ds = TimeSeriesDataset(X_va, y_va.astype(np.int64), patients=pat_va.astype(str))
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
        _, y_true_pat, y_pred_pat, _ = eval_patient_outputs_ts(model, val_loader, device=device)
        scores.append(float(balanced_accuracy_score(y_true_pat, y_pred_pat)))

    return float(np.mean(scores)) if scores else 0.0


def generate_outer_train_oof_predictions_ts(
    X_train: np.ndarray,
    y_train: np.ndarray,
    patients_train: np.ndarray,
    ModelClass: Any,
    device: torch.device,
    *,
    params: Dict[str, Any],
    n_oof_splits: int,
    seed: int,
    normalize_channels: Optional[Sequence[bool]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    in_channels = int(X_train.shape[1])
    num_classes = int(len(np.unique(y_train)))

    patients_train = np.asarray(patients_train).astype(str)
    y_train = np.asarray(y_train).astype(int)

    unique_pats, pat_labels = _patient_labels(patients_train, y_train)
    _, counts = np.unique(pat_labels, return_counts=True)

    K = min(int(n_oof_splits), int(counts.min()))
    if K < 2:
        raise ValueError(
            "Need at least 2 patient-level folds to generate OOF predictions "
            f"(min class count in outer-train = {counts.min()})."
        )

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=int(seed))

    all_patient_ids = []
    all_y_true = []
    all_y_pred = []
    all_proba = []

    for fold_idx, (fit_idx, hold_idx) in enumerate(skf.split(unique_pats, pat_labels)):
        seed_k = int(seed + fold_idx)
        seed_everything(seed_k, device=device)

        fit_p = unique_pats[fit_idx]
        hold_p = unique_pats[hold_idx]
        fit_mask, hold_mask = masks_from_patients(patients_train, fit_p, hold_p)

        X_fit, y_fit = X_train[fit_mask], y_train[fit_mask]
        X_hold, y_hold, pat_hold = X_train[hold_mask], y_train[hold_mask], patients_train[hold_mask]

        mean, std, _ = fit_channelwise_normalizer(X_fit, normalize_channels=normalize_channels)
        X_fit = apply_channelwise_normalizer(X_fit, mean, std)
        X_hold = apply_channelwise_normalizer(X_hold, mean, std)

        model = build_model_from_params_ts(params, in_channels, num_classes, ModelClass, device=device)
        model = train_fixed_epochs_ts(
            model,
            X_fit,
            y_fit,
            epochs=int(params["train_epochs"]),
            effective_batch_size=int(params["batch"]),
            device=device,
            lr=float(params["lr"]),
            weight_decay=1e-4,
            label_smoothing=0.0,
            seed=seed_k,
            max_physical_batch=32,
        )

        hold_ds = TimeSeriesDataset(X_hold, y_hold.astype(np.int64), patients=pat_hold.astype(str))
        hold_loader = DataLoader(hold_ds, batch_size=64, shuffle=False, num_workers=0)
        patient_ids_k, y_true_k, y_pred_k, proba_k = eval_patient_outputs_ts(model, hold_loader, device=device)

        all_patient_ids.append(patient_ids_k.astype(str))
        all_y_true.append(y_true_k.astype(int))
        all_y_pred.append(y_pred_k.astype(int))
        all_proba.append(proba_k.astype(np.float32))

    patient_ids = np.concatenate(all_patient_ids, axis=0)
    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    proba = np.concatenate(all_proba, axis=0)

    if len(np.unique(patient_ids)) != len(unique_pats):
        raise ValueError("OOF generation failed: some outer-train patients were predicted more than once or missed.")

    order = np.argsort(patient_ids)
    return patient_ids[order], y_true[order], y_pred[order], proba[order]


def generate_crossfit_outer_train_oof_predictions_ts(
    X_train: np.ndarray,
    y_train: np.ndarray,
    patients_train: np.ndarray,
    ModelClass: Any,
    device: torch.device,
    *,
    n_oof_splits: int,
    seed: int,
    normalize_channels: Optional[Sequence[bool]],
    use_optuna: bool,
    n_trials: int,
    optuna_max_epochs: int,
    epochs: int,
    n_inner_splits: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    in_channels = int(X_train.shape[1])
    num_classes = int(len(np.unique(y_train)))
    T = int(X_train.shape[2])

    patients_train = np.asarray(patients_train).astype(str)
    y_train = np.asarray(y_train).astype(int)

    unique_pats, pat_labels = _patient_labels(patients_train, y_train)
    _, counts = np.unique(pat_labels, return_counts=True)

    K = min(int(n_oof_splits), int(counts.min()))
    if K < 2:
        raise ValueError(
            "Need at least 2 patient-level folds to generate cross-fitted OOF predictions "
            f"(min class count in outer-train = {counts.min()})."
        )

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=int(seed))

    all_patient_ids = []
    all_y_true = []
    all_y_pred = []
    all_proba = []

    for fold_idx, (fit_idx, hold_idx) in enumerate(skf.split(unique_pats, pat_labels)):
        seed_k = int(seed + fold_idx)
        seed_everything(seed_k, device=device)

        fit_p = unique_pats[fit_idx]
        hold_p = unique_pats[hold_idx]
        fit_mask, hold_mask = masks_from_patients(patients_train, fit_p, hold_p)

        X_fit, y_fit, pat_fit = X_train[fit_mask], y_train[fit_mask], patients_train[fit_mask]
        X_hold, y_hold, pat_hold = X_train[hold_mask], y_train[hold_mask], patients_train[hold_mask]

        if use_optuna:
            sampler = optuna.samplers.TPESampler(seed=int(seed_k + 1_000))
            pruner = optuna.pruners.NopPruner()
            study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

            base_seed = int(seed_k + 2_000)
            max_train_epochs = int(max(5, min(optuna_max_epochs, epochs)))

            study.optimize(
                lambda t: inner_objective_nested_ts(
                    t,
                    X_fit,
                    y_fit,
                    pat_fit,
                    ModelClass=ModelClass,
                    device=device,
                    max_train_epochs=max_train_epochs,
                    n_inner_splits=int(n_inner_splits),
                    base_seed=base_seed,
                    normalize_channels=normalize_channels,
                ),
                n_trials=int(n_trials),
            )
            best_params_fold = dict(study.best_params)
        else:
            best_params_fold = {
                "lr": 1e-3,
                "batch": 32,
                "k1": 11,
                "k2": 5,
                "res": 12,
                "do": 0.25,
                "out_channels": 32,
                "dilation": 1,
                "train_epochs": min(int(epochs), 50),
            }

        best_params_fold = sanitize_params_for_data_ts(best_params_fold, ModelClass, T)

        mean, std, _ = fit_channelwise_normalizer(X_fit, normalize_channels=normalize_channels)
        X_fit = apply_channelwise_normalizer(X_fit, mean, std)
        X_hold = apply_channelwise_normalizer(X_hold, mean, std)

        model = build_model_from_params_ts(best_params_fold, in_channels, num_classes, ModelClass, device=device)
        model = train_fixed_epochs_ts(
            model,
            X_fit,
            y_fit,
            epochs=int(best_params_fold["train_epochs"]),
            effective_batch_size=int(best_params_fold["batch"]),
            device=device,
            lr=float(best_params_fold["lr"]),
            weight_decay=1e-4,
            label_smoothing=0.0,
            seed=seed_k,
            max_physical_batch=32,
        )

        hold_ds = TimeSeriesDataset(X_hold, y_hold.astype(np.int64), patients=pat_hold.astype(str))
        hold_loader = DataLoader(hold_ds, batch_size=64, shuffle=False, num_workers=0)
        patient_ids_k, y_true_k, y_pred_k, proba_k = eval_patient_outputs_ts(model, hold_loader, device=device)

        all_patient_ids.append(patient_ids_k.astype(str))
        all_y_true.append(y_true_k.astype(int))
        all_y_pred.append(y_pred_k.astype(int))
        all_proba.append(proba_k.astype(np.float32))

    patient_ids = np.concatenate(all_patient_ids, axis=0)
    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    proba = np.concatenate(all_proba, axis=0)

    if len(np.unique(patient_ids)) != len(unique_pats):
        raise ValueError(
            "Cross-fitted OOF generation failed: some outer-train patients were predicted more than once or missed."
        )

    order = np.argsort(patient_ids)
    return patient_ids[order], y_true[order], y_pred[order], proba[order]


def run_nested_split_ts(
    ModelClass: Any,
    X: np.ndarray,
    y: np.ndarray,
    patients: np.ndarray,
    *,
    split_file: Optional[str],
    split_id: int,
    test_size: float = 0.2,
    split_seed: int = 12345,
    epochs: int = 160,
    use_optuna: bool = True,
    n_trials: int = 30,
    optuna_max_epochs: int = 30,
    n_inner_splits: int = 3,
    normalize_channels: Optional[Sequence[bool]] = None,
    seed: int = 2026,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int)
    patients = np.asarray(patients).astype(str)

    _patient_labels(patients, y)

    if device is None:
        device = get_default_device()

    seed_everything(int(seed), device=device)

    in_channels = int(X.shape[1])
    num_classes = int(len(np.unique(y)))
    if num_classes < 2:
        raise ValueError("Need at least 2 classes.")

    normalize_channels = validate_normalize_channels(normalize_channels, in_channels)

    if split_file is not None:
        payload = load_split_file(split_file)
        spec = get_split_from_file(payload, split_id=split_id)
        assert_split_compatible(patients, spec)
        train_patients = np.asarray(spec.train_patients).astype(str)
        test_patients_ = np.asarray(spec.test_patients).astype(str)
    else:
        train_patients, test_patients_ = stratified_patient_split(
            patients, y, test_size=test_size, seed=split_seed
        )

    train_mask, test_mask = masks_from_patients(patients, train_patients, test_patients_)
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Outer split produced empty train or test set (check split compatibility).")

    X_tr, y_tr, pat_tr = X[train_mask], y[train_mask], patients[train_mask]
    X_te, y_te, pat_te = X[test_mask], y[test_mask], patients[test_mask]

    T = int(X.shape[2])

    if use_optuna:
        sampler = optuna.samplers.TPESampler(seed=int(seed + 10_000 + split_id))
        pruner = optuna.pruners.NopPruner()
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        base_seed = int(seed + 20_000 + split_id * 1000)
        max_train_epochs = int(max(5, min(optuna_max_epochs, epochs)))

        study.optimize(
            lambda t: inner_objective_nested_ts(
                t,
                X_tr,
                y_tr,
                pat_tr,
                ModelClass=ModelClass,
                device=device,
                max_train_epochs=max_train_epochs,
                n_inner_splits=int(n_inner_splits),
                base_seed=base_seed,
                normalize_channels=normalize_channels,
            ),
            n_trials=int(n_trials),
        )
        best_params = dict(study.best_params)
    else:
        best_params = {
            "lr": 1e-3,
            "batch": 32,
            "k1": 11,
            "k2": 5,
            "res": 12,
            "do": 0.25,
            "out_channels": 32,
            "dilation": 1,
            "train_epochs": min(int(epochs), 50),
        }

    best_params = sanitize_params_for_data_ts(best_params, ModelClass, T)

    oof_train_patient_ids, oof_train_y_true, oof_train_y_pred, oof_train_proba = (
        generate_crossfit_outer_train_oof_predictions_ts(
            X_tr,
            y_tr,
            pat_tr,
            ModelClass,
            device=device,
            n_oof_splits=int(n_inner_splits),
            seed=int(seed + 25_000 + split_id),
            normalize_channels=normalize_channels,
            use_optuna=bool(use_optuna),
            n_trials=int(n_trials),
            optuna_max_epochs=int(optuna_max_epochs),
            epochs=int(epochs),
            n_inner_splits=int(n_inner_splits),
        )
    )

    mean, std, norm_mask = fit_channelwise_normalizer(X_tr, normalize_channels=normalize_channels)
    X_tr_final = apply_channelwise_normalizer(X_tr, mean, std)
    X_te_final = apply_channelwise_normalizer(X_te, mean, std)

    final_seed = int(seed + 30_000 + split_id)
    seed_everything(final_seed, device=device)

    model = build_model_from_params_ts(best_params, in_channels, num_classes, ModelClass, device=device)
    model = train_fixed_epochs_ts(
        model,
        X_tr_final,
        y_tr,
        epochs=int(best_params.get("train_epochs", min(int(epochs), 50))),
        effective_batch_size=int(best_params.get("batch", 32)),
        device=device,
        lr=float(best_params["lr"]),
        weight_decay=1e-4,
        label_smoothing=0.0,
        seed=final_seed + 777,
        max_physical_batch=32,
    )

    test_ds = TimeSeriesDataset(X_te_final, y_te.astype(np.int64), patients=pat_te)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    patient_ids, y_true_pat, y_pred_pat, proba_pat = eval_patient_outputs_ts(model, test_loader, device=device)

    metrics = compute_patient_metrics(y_true_pat, y_pred_pat, proba_pat)
    metrics.update(
        {
            "EffectiveEpochs": int(best_params.get("train_epochs", 0)),
            "n_patients_train_outer": int(len(np.unique(pat_tr))),
            "n_patients_test_outer": int(len(np.unique(pat_te))),
            "normalize_channels": norm_mask.astype(int).tolist(),
        }
    )

    artifacts = {
        "best_params": best_params,
        "patient_ids": patient_ids,
        "y_true": y_true_pat,
        "y_pred": y_pred_pat,
        "proba": proba_pat,
        "normalize_channels": norm_mask.astype(int).tolist(),
        "oof_train_patient_ids": oof_train_patient_ids,
        "oof_train_y_true": oof_train_y_true,
        "oof_train_y_pred": oof_train_y_pred,
        "oof_train_proba": oof_train_proba,
    }

    return metrics, artifacts


@ray.remote(num_cpus=1)
def run_nested_split_ray_ts(
    ModelClass: Any,
    X: np.ndarray,
    y: np.ndarray,
    patients: np.ndarray,
    *,
    split_file: Optional[str],
    split_id: int,
    test_size: float = 0.2,
    split_seed: int = 12345,
    epochs: int = 160,
    use_optuna: bool = True,
    n_trials: int = 30,
    optuna_max_epochs: int = 30,
    n_inner_splits: int = 3,
    normalize_channels: Optional[Sequence[bool]] = None,
    seed: int = 2026,
    force_cpu: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    device = torch.device("cpu") if force_cpu else get_default_device()
    return run_nested_split_ts(
        ModelClass,
        X,
        y,
        patients,
        split_file=split_file,
        split_id=int(split_id),
        test_size=float(test_size),
        split_seed=int(split_seed),
        epochs=int(epochs),
        use_optuna=bool(use_optuna),
        n_trials=int(n_trials),
        optuna_max_epochs=int(optuna_max_epochs),
        n_inner_splits=int(n_inner_splits),
        normalize_channels=normalize_channels,
        seed=int(seed),
        device=device,
    )


# -----------------------------
# Multimodal pipeline (FiLM only)
# -----------------------------
class MultiModalDataset(Dataset):
    def __init__(
        self,
        X_ts: np.ndarray,
        X_tab: np.ndarray,
        y: np.ndarray,
        patients: Optional[np.ndarray] = None,
    ):
        if not isinstance(X_ts, np.ndarray) or not isinstance(X_tab, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X_ts, X_tab, and y must be numpy arrays.")

        if X_ts.shape[0] != X_tab.shape[0] or X_ts.shape[0] != y.shape[0]:
            raise ValueError("X_ts, X_tab, and y must have the same number of rows.")

        self.X_ts = torch.from_numpy(X_ts)
        self.X_tab = torch.from_numpy(X_tab)
        self.y = torch.from_numpy(y)
        self.patients = None if patients is None else np.asarray(patients)

        if self.patients is not None and len(self.patients) != len(self.y):
            raise ValueError("patients must have the same length as y.")

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        if self.patients is None:
            return self.X_ts[idx], self.X_tab[idx], self.y[idx]
        return self.X_ts[idx], self.X_tab[idx], self.y[idx], self.patients[idx]


def _unpack_mm_batch(batch):
    if len(batch) == 3:
        Xts, Xtab, yb = batch
        pb = None
    else:
        Xts, Xtab, yb, pb = batch
    return Xts, Xtab, yb, pb


class FiLMGenerator(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_channels: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        use_identity_init: bool = True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * n_channels),
        )
        self.n_channels = int(n_channels)

        if use_identity_init:
            last = self.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x_tab: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x_tab)
        gamma_raw, beta = torch.chunk(h, 2, dim=1)
        gamma = 1.0 + gamma_raw
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return gamma, beta


class FiLMLayer(nn.Module):
    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return gamma * x + beta


class FiLMCNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        tabular_dim: int,
        k1: int,
        k2: int,
        out_channels: int = 32,
        spatial_resolution: int = 12,
        dropout: float = 0.3,
        film_hidden: int = 64,
    ):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 2, kernel_size=k1, padding="same"),
            nn.GroupNorm(1, out_channels // 2),
            nn.GELU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 2, kernel_size=k2, padding="same"),
            nn.GroupNorm(1, out_channels // 2),
            nn.GELU(),
        )

        self.film_gen = FiLMGenerator(
            in_features=tabular_dim,
            n_channels=out_channels,
            hidden_dim=film_hidden,
            dropout=dropout,
        )
        self.film = FiLMLayer()
        self.pool_avg = nn.AdaptiveAvgPool1d(spatial_resolution)
        self.pool_max = nn.AdaptiveMaxPool1d(spatial_resolution)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = int(out_channels * 2 * spatial_resolution)

    def forward(self, x_ts: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        x1 = self.branch1(x_ts)
        x2 = self.branch2(x_ts)
        x = torch.cat([x1, x2], dim=1)

        gamma, beta = self.film_gen(x_tab)
        x = self.film(x, gamma, beta)
        x = F.gelu(x)

        avg_p = self.pool_avg(x)
        max_p = self.pool_max(x)
        x = torch.cat([avg_p, max_p], dim=1)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return x


class FiLMTCNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        tabular_dim: int,
        k1: int,
        k2: int,
        dilation: int = 1,
        out_channels: int = 32,
        spatial_resolution: int = 12,
        dropout: float = 0.3,
        film_hidden: int = 64,
    ):
        super().__init__()
        self.block1 = TCNBlock(in_channels, out_channels, k1, dilation=dilation, dropout=dropout * 0.5)
        self.block2 = TCNBlock(out_channels, out_channels, k2, dilation=1, dropout=dropout * 0.5)

        self.film_gen = FiLMGenerator(
            in_features=tabular_dim,
            n_channels=out_channels,
            hidden_dim=film_hidden,
            dropout=dropout,
        )
        self.film = FiLMLayer()
        self.pool_avg = nn.AdaptiveAvgPool1d(spatial_resolution)
        self.pool_max = nn.AdaptiveMaxPool1d(spatial_resolution)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = int(out_channels * 2 * spatial_resolution)

    def forward(self, x_ts: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        x = self.block1(x_ts)
        x = self.block2(x)

        gamma, beta = self.film_gen(x_tab)
        x = self.film(x, gamma, beta)
        x = F.gelu(x)

        avg_p = self.pool_avg(x)
        max_p = self.pool_max(x)
        x = torch.cat([avg_p, max_p], dim=1)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return x


class FiLMCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        tabular_dim: int,
        num_classes: int,
        k1: int,
        k2: int,
        out_channels: int = 32,
        spatial_resolution: int = 12,
        dropout: float = 0.3,
        tab_hidden: int = 64,
        fusion_hidden: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.ts_encoder = FiLMCNNEncoder(
            in_channels=in_channels,
            tabular_dim=tabular_dim,
            k1=k1,
            k2=k2,
            out_channels=out_channels,
            spatial_resolution=spatial_resolution,
            dropout=dropout,
            film_hidden=tab_hidden,
        )
        self.head = nn.Sequential(
            nn.Linear(self.ts_encoder.out_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, x_ts: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        z_ts = self.ts_encoder(x_ts, x_tab)
        return self.head(z_ts)


class FiLMTCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        tabular_dim: int,
        num_classes: int,
        k1: int,
        k2: int,
        dilation: int = 1,
        out_channels: int = 32,
        spatial_resolution: int = 12,
        dropout: float = 0.3,
        tab_hidden: int = 64,
        fusion_hidden: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.ts_encoder = FiLMTCNEncoder(
            in_channels=in_channels,
            tabular_dim=tabular_dim,
            k1=k1,
            k2=k2,
            dilation=dilation,
            out_channels=out_channels,
            spatial_resolution=spatial_resolution,
            dropout=dropout,
            film_hidden=tab_hidden,
        )
        self.head = nn.Sequential(
            nn.Linear(self.ts_encoder.out_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, x_ts: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        z_ts = self.ts_encoder(x_ts, x_tab)
        return self.head(z_ts)


def train_epoch_mm(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    accum_steps: int = 1,
) -> None:
    model.train()
    accum_steps = max(1, int(accum_steps))

    optimizer.zero_grad(set_to_none=True)

    step = 0
    acc_count = 0

    for step, batch in enumerate(loader, start=1):
        Xts, Xtab, yb, _ = _unpack_mm_batch(batch)
        Xts = Xts.to(device, non_blocking=True)
        Xtab = Xtab.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(Xts, Xtab)
        loss = criterion(logits, yb)

        acc_count += 1
        loss = loss / accum_steps
        loss.backward()

        if acc_count == accum_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            acc_count = 0

    if step > 0 and acc_count > 0:
        scale = float(accum_steps) / float(acc_count)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


@torch.no_grad()
def eval_patient_outputs_mm(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    verify_patient_labels: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    probs_by_pat: Dict[str, List[np.ndarray]] = {}
    y_by_pat: Dict[str, int] = {}

    for batch in loader:
        Xts, Xtab, yb, pb = _unpack_mm_batch(batch)
        if pb is None:
            raise ValueError("eval_patient_outputs_mm requires patient ids in the dataset.")

        Xts = Xts.to(device, non_blocking=True)
        Xtab = Xtab.to(device, non_blocking=True)
        logits = model(Xts, Xtab)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

        y_np = yb.detach().cpu().numpy()
        pb_np = np.asarray(pb)

        for i in range(len(y_np)):
            pid = str(pb_np[i])
            probs_by_pat.setdefault(pid, []).append(probs[i])
            yi = int(y_np[i])
            if pid not in y_by_pat:
                y_by_pat[pid] = yi
            elif verify_patient_labels and y_by_pat[pid] != yi:
                raise ValueError(f"Inconsistent labels for patient_id={pid}: {y_by_pat[pid]} vs {yi}")

    patient_ids: List[str] = []
    y_true: List[int] = []
    mean_probs_list: List[np.ndarray] = []

    for pid in sorted(probs_by_pat.keys()):
        patient_ids.append(pid)
        y_true.append(int(y_by_pat[pid]))
        mean_probs_list.append(np.mean(np.stack(probs_by_pat[pid], axis=0), axis=0))

    proba = np.stack(mean_probs_list, axis=0).astype(np.float32)
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.argmax(proba, axis=1).astype(int)

    return np.asarray(patient_ids), y_true_arr, y_pred_arr, proba


def train_fixed_epochs_mm(
    model: nn.Module,
    X_ts_tr: np.ndarray,
    X_tab_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    epochs: int,
    effective_batch_size: int,
    device: torch.device,
    lr: float,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.0,
    seed: int = 0,
    max_physical_batch: int = 32,
) -> nn.Module:
    train_ds = MultiModalDataset(X_ts_tr, X_tab_tr, y_tr.astype(np.int64))
    g = make_torch_generator(seed)

    n_tr = int(len(train_ds))
    eff_bs = max(1, int(effective_batch_size))

    phys_bs = min(eff_bs, int(max_physical_batch), n_tr)
    phys_bs = max(1, int(phys_bs))

    accum_steps = int(np.ceil(eff_bs / phys_bs))
    accum_steps = max(1, accum_steps)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(phys_bs),
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=g,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)))
    criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))

    for _ in range(int(epochs)):
        train_epoch_mm(
            model,
            train_loader,
            optimizer,
            criterion,
            device=device,
            accum_steps=accum_steps,
        )
        scheduler.step()

    return model


def suggest_params_mm(trial: optuna.Trial, ModelClass: Any, *, max_train_epochs: int) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    lo = max(5, int(max_train_epochs // 4))
    hi = max(5, int(max_train_epochs))
    step = max(1, int(max_train_epochs // 10))
    params["train_epochs"] = int(trial.suggest_int("train_epochs", lo, hi, step=step))
    params["batch"] = trial.suggest_categorical("batch", [4, 8, 16, 32, 64])
    params["lr"] = trial.suggest_float("lr", 3e-4, 1e-2, log=True)
    params["k1"] = trial.suggest_int("k1", 5, 21, step=2)
    params["k2"] = trial.suggest_int("k2", 3, 15, step=2)
    params["do"] = trial.suggest_float("do", 0.05, 0.35)
    params["out_channels"] = trial.suggest_categorical("out_channels", [16, 32, 64])
    params["res"] = int(trial.suggest_categorical("res", RES_UNIVERSE))
    params["tab_hidden"] = trial.suggest_categorical("tab_hidden", [32, 64, 128])
    params["fusion_hidden"] = trial.suggest_categorical("fusion_hidden", [32, 64, 128])
    params["dilation"] = int(trial.suggest_categorical("dilation", [1, 2, 4, 8])) if ModelClass is FiLMTCN else 1
    return params


def sanitize_params_for_data_mm(params: Dict[str, Any], ModelClass: Any, T: int) -> Dict[str, Any]:
    p = dict(params)

    r = int(p.get("res", 12))
    r = min(r, int(T))
    if int(p.get("out_channels", 32)) == 64:
        r = min(r, 36)
    r = max(8, r)
    if r % 2 == 1:
        r -= 1
    p["res"] = int(r)

    if ModelClass is FiLMTCN:
        d = int(p.get("dilation", 1))
        rf = tcn_receptive_field(int(p["k1"]), int(p["k2"]), int(d), d2=1)
        mult = 1.0 if T <= 120 else 1.25
        limit = int(mult * T)
        if rf > limit:
            for dd in [4, 2, 1]:
                rf_dd = tcn_receptive_field(int(p["k1"]), int(p["k2"]), dd, d2=1)
                if rf_dd <= limit:
                    d = dd
                    break
        p["dilation"] = int(d)

    return p


def build_model_from_params_mm(
    params: Dict[str, Any],
    in_channels: int,
    tabular_dim: int,
    num_classes: int,
    ModelClass: Any,
    device: torch.device,
) -> nn.Module:
    kwargs = dict(
        in_channels=in_channels,
        tabular_dim=tabular_dim,
        num_classes=num_classes,
        k1=int(params["k1"]),
        k2=int(params["k2"]),
        out_channels=int(params["out_channels"]),
        spatial_resolution=int(params["res"]),
        dropout=float(params["do"]),
        tab_hidden=int(params["tab_hidden"]),
        fusion_hidden=int(params["fusion_hidden"]),
    )
    if ModelClass is FiLMTCN:
        kwargs["dilation"] = int(params.get("dilation", 1))
    return ModelClass(**kwargs).to(device)


def inner_objective_nested_mm(
    trial: optuna.Trial,
    X_ts_train: np.ndarray,
    X_tab_train: np.ndarray,
    y_train: np.ndarray,
    patients_train: np.ndarray,
    ModelClass: Any,
    device: torch.device,
    *,
    max_train_epochs: int,
    n_inner_splits: int,
    base_seed: int,
    normalize_channels: Optional[Sequence[bool]],
    normalize_tabular: bool,
) -> float:
    in_channels = int(X_ts_train.shape[1])
    tabular_dim = int(X_tab_train.shape[1])
    num_classes = int(len(np.unique(y_train)))
    T = int(X_ts_train.shape[2])

    params = suggest_params_mm(trial, ModelClass, max_train_epochs=int(max_train_epochs))
    params = sanitize_params_for_data_mm(params, ModelClass, T)

    patients_train = np.asarray(patients_train).astype(str)
    y_train = np.asarray(y_train).astype(int)

    unique_pats, pat_labels = _patient_labels(patients_train, y_train)
    _, counts = np.unique(pat_labels, return_counts=True)

    if counts.min() < 2:
        raise ValueError(
            "Inner CV impossible: not enough patients in at least one class. "
            f"Patient-level class counts: {dict(zip(*np.unique(pat_labels, return_counts=True)))}"
        )

    K = int(n_inner_splits)
    if K < 2:
        raise ValueError("n_inner_splits must be >= 2 for StratifiedKFold inner CV.")
    if K > int(counts.min()):
        raise ValueError(f"Inner CV impossible: n_inner_splits={K} > min patients in a class={counts.min()}.")

    fold_seed = int(base_seed + trial.number * 10_000)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=fold_seed)

    scores: List[float] = []

    for fold_idx, (tr_pat_idx, va_pat_idx) in enumerate(skf.split(unique_pats, pat_labels)):
        seed_k = int(base_seed + trial.number * 100 + fold_idx)
        seed_everything(seed_k, device=device)

        tr_p = unique_pats[tr_pat_idx]
        va_p = unique_pats[va_pat_idx]
        tr_mask, va_mask = masks_from_patients(patients_train, tr_p, va_p)

        X_ts_tr, X_tab_tr, y_tr = X_ts_train[tr_mask], X_tab_train[tr_mask], y_train[tr_mask]
        X_ts_va, X_tab_va, y_va, pat_va = (
            X_ts_train[va_mask],
            X_tab_train[va_mask],
            y_train[va_mask],
            patients_train[va_mask],
        )

        ts_mean, ts_std, _ = fit_channelwise_normalizer(X_ts_tr, normalize_channels=normalize_channels)
        X_ts_tr = apply_channelwise_normalizer(X_ts_tr, ts_mean, ts_std)
        X_ts_va = apply_channelwise_normalizer(X_ts_va, ts_mean, ts_std)

        tab_mean, tab_std, _ = fit_tabular_normalizer(X_tab_tr, normalize_tabular=normalize_tabular)
        X_tab_tr = apply_tabular_normalizer(X_tab_tr, tab_mean, tab_std)
        X_tab_va = apply_tabular_normalizer(X_tab_va, tab_mean, tab_std)

        model = build_model_from_params_mm(
            params,
            in_channels=in_channels,
            tabular_dim=tabular_dim,
            num_classes=num_classes,
            ModelClass=ModelClass,
            device=device,
        )
        model = train_fixed_epochs_mm(
            model,
            X_ts_tr,
            X_tab_tr,
            y_tr,
            epochs=int(params["train_epochs"]),
            effective_batch_size=int(params["batch"]),
            device=device,
            lr=float(params["lr"]),
            weight_decay=1e-4,
            label_smoothing=0.0,
            seed=seed_k,
            max_physical_batch=32,
        )

        val_ds = MultiModalDataset(X_ts_va, X_tab_va, y_va.astype(np.int64), patients=pat_va.astype(str))
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
        _, y_true_pat, y_pred_pat, _ = eval_patient_outputs_mm(model, val_loader, device=device)
        scores.append(float(balanced_accuracy_score(y_true_pat, y_pred_pat)))

    return float(np.mean(scores)) if scores else 0.0


def generate_outer_train_oof_predictions_mm(
    X_ts_train: np.ndarray,
    X_tab_train: np.ndarray,
    y_train: np.ndarray,
    patients_train: np.ndarray,
    ModelClass: Any,
    device: torch.device,
    *,
    params: Dict[str, Any],
    n_oof_splits: int,
    seed: int,
    normalize_channels: Optional[Sequence[bool]],
    normalize_tabular: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    in_channels = int(X_ts_train.shape[1])
    tabular_dim = int(X_tab_train.shape[1])
    num_classes = int(len(np.unique(y_train)))

    patients_train = np.asarray(patients_train).astype(str)
    y_train = np.asarray(y_train).astype(int)

    unique_pats, pat_labels = _patient_labels(patients_train, y_train)
    _, counts = np.unique(pat_labels, return_counts=True)

    K = min(int(n_oof_splits), int(counts.min()))
    if K < 2:
        raise ValueError(
            "Need at least 2 patient-level folds to generate multimodal OOF predictions "
            f"(min class count in outer-train = {counts.min()})."
        )

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=int(seed))

    all_patient_ids = []
    all_y_true = []
    all_y_pred = []
    all_proba = []

    for fold_idx, (fit_idx, hold_idx) in enumerate(skf.split(unique_pats, pat_labels)):
        seed_k = int(seed + fold_idx)
        seed_everything(seed_k, device=device)

        fit_p = unique_pats[fit_idx]
        hold_p = unique_pats[hold_idx]
        fit_mask, hold_mask = masks_from_patients(patients_train, fit_p, hold_p)

        X_ts_fit, X_tab_fit, y_fit = X_ts_train[fit_mask], X_tab_train[fit_mask], y_train[fit_mask]
        X_ts_hold = X_ts_train[hold_mask]
        X_tab_hold = X_tab_train[hold_mask]
        y_hold = y_train[hold_mask]
        pat_hold = patients_train[hold_mask]

        ts_mean, ts_std, _ = fit_channelwise_normalizer(X_ts_fit, normalize_channels=normalize_channels)
        X_ts_fit = apply_channelwise_normalizer(X_ts_fit, ts_mean, ts_std)
        X_ts_hold = apply_channelwise_normalizer(X_ts_hold, ts_mean, ts_std)
        tab_mean, tab_std, _ = fit_tabular_normalizer(X_tab_fit, normalize_tabular=normalize_tabular)
        X_tab_fit = apply_tabular_normalizer(X_tab_fit, tab_mean, tab_std)
        X_tab_hold = apply_tabular_normalizer(X_tab_hold, tab_mean, tab_std)

        model = build_model_from_params_mm(
            params,
            in_channels=in_channels,
            tabular_dim=tabular_dim,
            num_classes=num_classes,
            ModelClass=ModelClass,
            device=device,
        )
        model = train_fixed_epochs_mm(
            model,
            X_ts_fit,
            X_tab_fit,
            y_fit,
            epochs=int(params["train_epochs"]),
            effective_batch_size=int(params["batch"]),
            device=device,
            lr=float(params["lr"]),
            weight_decay=1e-4,
            label_smoothing=0.0,
            seed=seed_k,
            max_physical_batch=32,
        )

        hold_ds = MultiModalDataset(X_ts_hold, X_tab_hold, y_hold.astype(np.int64), patients=pat_hold.astype(str))
        hold_loader = DataLoader(hold_ds, batch_size=64, shuffle=False, num_workers=0)
        patient_ids_k, y_true_k, y_pred_k, proba_k = eval_patient_outputs_mm(model, hold_loader, device=device)

        all_patient_ids.append(patient_ids_k.astype(str))
        all_y_true.append(y_true_k.astype(int))
        all_y_pred.append(y_pred_k.astype(int))
        all_proba.append(proba_k.astype(np.float32))

    patient_ids = np.concatenate(all_patient_ids, axis=0)
    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    proba = np.concatenate(all_proba, axis=0)

    if len(np.unique(patient_ids)) != len(unique_pats):
        raise ValueError(
            "Multimodal OOF generation failed: some outer-train patients were predicted more than once or missed."
        )

    order = np.argsort(patient_ids)
    return patient_ids[order], y_true[order], y_pred[order], proba[order]


def generate_crossfit_outer_train_oof_predictions_mm(
    X_ts_train: np.ndarray,
    X_tab_train: np.ndarray,
    y_train: np.ndarray,
    patients_train: np.ndarray,
    ModelClass: Any,
    device: torch.device,
    *,
    n_oof_splits: int,
    seed: int,
    normalize_channels: Optional[Sequence[bool]],
    normalize_tabular: bool,
    use_optuna: bool,
    n_trials: int,
    optuna_max_epochs: int,
    epochs: int,
    n_inner_splits: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    in_channels = int(X_ts_train.shape[1])
    tabular_dim = int(X_tab_train.shape[1])
    num_classes = int(len(np.unique(y_train)))
    T = int(X_ts_train.shape[2])

    patients_train = np.asarray(patients_train).astype(str)
    y_train = np.asarray(y_train).astype(int)

    unique_pats, pat_labels = _patient_labels(patients_train, y_train)
    _, counts = np.unique(pat_labels, return_counts=True)

    K = min(int(n_oof_splits), int(counts.min()))
    if K < 2:
        raise ValueError(
            "Need at least 2 patient-level folds to generate cross-fitted multimodal OOF predictions "
            f"(min class count in outer-train = {counts.min()})."
        )

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=int(seed))

    all_patient_ids = []
    all_y_true = []
    all_y_pred = []
    all_proba = []

    for fold_idx, (fit_idx, hold_idx) in enumerate(skf.split(unique_pats, pat_labels)):
        seed_k = int(seed + fold_idx)
        seed_everything(seed_k, device=device)

        fit_p = unique_pats[fit_idx]
        hold_p = unique_pats[hold_idx]
        fit_mask, hold_mask = masks_from_patients(patients_train, fit_p, hold_p)

        X_ts_fit, X_tab_fit, y_fit, pat_fit = (
            X_ts_train[fit_mask],
            X_tab_train[fit_mask],
            y_train[fit_mask],
            patients_train[fit_mask],
        )
        X_ts_hold, X_tab_hold, y_hold, pat_hold = (
            X_ts_train[hold_mask],
            X_tab_train[hold_mask],
            y_train[hold_mask],
            patients_train[hold_mask],
        )

        if use_optuna:
            sampler = optuna.samplers.TPESampler(seed=int(seed_k + 1_000))
            pruner = optuna.pruners.NopPruner()
            study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

            base_seed = int(seed_k + 2_000)
            max_train_epochs = int(max(5, min(optuna_max_epochs, epochs)))

            study.optimize(
                lambda t: inner_objective_nested_mm(
                    t,
                    X_ts_fit,
                    X_tab_fit,
                    y_fit,
                    pat_fit,
                    ModelClass=ModelClass,
                    device=device,
                    max_train_epochs=max_train_epochs,
                    n_inner_splits=int(n_inner_splits),
                    base_seed=base_seed,
                    normalize_channels=normalize_channels,
                    normalize_tabular=bool(normalize_tabular),
                ),
                n_trials=int(n_trials),
            )
            best_params_fold = dict(study.best_params)
        else:
            best_params_fold = {
                "lr": 1e-3,
                "batch": 32,
                "k1": 11,
                "k2": 5,
                "res": 12,
                "do": 0.25,
                "out_channels": 32,
                "dilation": 1,
                "tab_hidden": 64,
                "fusion_hidden": 64,
                "train_epochs": min(int(epochs), 50),
            }

        best_params_fold = sanitize_params_for_data_mm(best_params_fold, ModelClass, T)

        ts_mean, ts_std, _ = fit_channelwise_normalizer(X_ts_fit, normalize_channels=normalize_channels)
        X_ts_fit = apply_channelwise_normalizer(X_ts_fit, ts_mean, ts_std)
        X_ts_hold = apply_channelwise_normalizer(X_ts_hold, ts_mean, ts_std)

        tab_mean, tab_std, _ = fit_tabular_normalizer(X_tab_fit, normalize_tabular=normalize_tabular)
        X_tab_fit = apply_tabular_normalizer(X_tab_fit, tab_mean, tab_std)
        X_tab_hold = apply_tabular_normalizer(X_tab_hold, tab_mean, tab_std)

        model = build_model_from_params_mm(
            best_params_fold,
            in_channels=in_channels,
            tabular_dim=tabular_dim,
            num_classes=num_classes,
            ModelClass=ModelClass,
            device=device,
        )
        model = train_fixed_epochs_mm(
            model,
            X_ts_fit,
            X_tab_fit,
            y_fit,
            epochs=int(best_params_fold["train_epochs"]),
            effective_batch_size=int(best_params_fold["batch"]),
            device=device,
            lr=float(best_params_fold["lr"]),
            weight_decay=1e-4,
            label_smoothing=0.0,
            seed=seed_k,
            max_physical_batch=32,
        )

        hold_ds = MultiModalDataset(X_ts_hold, X_tab_hold, y_hold.astype(np.int64), patients=pat_hold.astype(str))
        hold_loader = DataLoader(hold_ds, batch_size=64, shuffle=False, num_workers=0)
        patient_ids_k, y_true_k, y_pred_k, proba_k = eval_patient_outputs_mm(model, hold_loader, device=device)

        all_patient_ids.append(patient_ids_k.astype(str))
        all_y_true.append(y_true_k.astype(int))
        all_y_pred.append(y_pred_k.astype(int))
        all_proba.append(proba_k.astype(np.float32))

    patient_ids = np.concatenate(all_patient_ids, axis=0)
    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    proba = np.concatenate(all_proba, axis=0)

    if len(np.unique(patient_ids)) != len(unique_pats):
        raise ValueError(
            "Cross-fitted multimodal OOF generation failed: some outer-train patients were predicted more than once or missed."
        )

    order = np.argsort(patient_ids)
    return patient_ids[order], y_true[order], y_pred[order], proba[order]


def run_nested_split_mm(
    ModelClass: Any,
    X_ts: np.ndarray,
    X_tab: np.ndarray,
    y: np.ndarray,
    patients: np.ndarray,
    *,
    split_file: Optional[str],
    split_id: int,
    test_size: float = 0.2,
    split_seed: int = 12345,
    epochs: int = 160,
    use_optuna: bool = True,
    n_trials: int = 30,
    optuna_max_epochs: int = 30,
    n_inner_splits: int = 3,
    normalize_channels: Optional[Sequence[bool]] = None,
    normalize_tabular: bool = True,
    seed: int = 2026,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    X_ts = np.asarray(X_ts, dtype=np.float32)
    X_tab = np.asarray(X_tab, dtype=np.float32)
    y = np.asarray(y).astype(int)
    patients = np.asarray(patients).astype(str)

    if X_ts.shape[0] != X_tab.shape[0] or X_ts.shape[0] != y.shape[0]:
        raise ValueError("X_ts, X_tab, and y must have the same number of rows.")

    _patient_labels(patients, y)

    if device is None:
        device = get_default_device()

    seed_everything(int(seed), device=device)

    in_channels = int(X_ts.shape[1])
    tabular_dim = int(X_tab.shape[1])
    num_classes = int(len(np.unique(y)))
    if num_classes < 2:
        raise ValueError("Need at least 2 classes.")

    normalize_channels = validate_normalize_channels(normalize_channels, in_channels)

    if split_file is not None:
        payload = load_split_file(split_file)
        spec = get_split_from_file(payload, split_id=split_id)
        assert_split_compatible(patients, spec)
        train_patients = np.asarray(spec.train_patients).astype(str)
        test_patients_ = np.asarray(spec.test_patients).astype(str)
    else:
        train_patients, test_patients_ = stratified_patient_split(
            patients, y, test_size=test_size, seed=split_seed
        )

    train_mask, test_mask = masks_from_patients(patients, train_patients, test_patients_)
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Outer split produced empty train or test set (check split compatibility).")

    X_ts_tr, X_tab_tr, y_tr, pat_tr = X_ts[train_mask], X_tab[train_mask], y[train_mask], patients[train_mask]
    X_ts_te, X_tab_te, y_te, pat_te = X_ts[test_mask], X_tab[test_mask], y[test_mask], patients[test_mask]

    T = int(X_ts.shape[2])

    if use_optuna:
        sampler = optuna.samplers.TPESampler(seed=int(seed + 10_000 + split_id))
        pruner = optuna.pruners.NopPruner()
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        base_seed = int(seed + 20_000 + split_id * 1000)
        max_train_epochs = int(max(5, min(optuna_max_epochs, epochs)))

        study.optimize(
            lambda t: inner_objective_nested_mm(
                t,
                X_ts_tr,
                X_tab_tr,
                y_tr,
                pat_tr,
                ModelClass=ModelClass,
                device=device,
                max_train_epochs=max_train_epochs,
                n_inner_splits=int(n_inner_splits),
                base_seed=base_seed,
                normalize_channels=normalize_channels,
                normalize_tabular=bool(normalize_tabular),
            ),
            n_trials=int(n_trials),
        )
        best_params = dict(study.best_params)
    else:
        best_params = {
            "lr": 1e-3,
            "batch": 32,
            "k1": 11,
            "k2": 5,
            "res": 12,
            "do": 0.25,
            "out_channels": 32,
            "dilation": 1,
            "tab_hidden": 64,
            "fusion_hidden": 64,
            "train_epochs": min(int(epochs), 50),
        }

    best_params = sanitize_params_for_data_mm(best_params, ModelClass, T)

    oof_train_patient_ids, oof_train_y_true, oof_train_y_pred, oof_train_proba = (
        generate_crossfit_outer_train_oof_predictions_mm(
            X_ts_tr,
            X_tab_tr,
            y_tr,
            pat_tr,
            ModelClass,
            device=device,
            n_oof_splits=int(n_inner_splits),
            seed=int(seed + 25_000 + split_id),
            normalize_channels=normalize_channels,
            normalize_tabular=bool(normalize_tabular),
            use_optuna=bool(use_optuna),
            n_trials=int(n_trials),
            optuna_max_epochs=int(optuna_max_epochs),
            epochs=int(epochs),
            n_inner_splits=int(n_inner_splits),
        )
    )

    ts_mean, ts_std, norm_mask = fit_channelwise_normalizer(X_ts_tr, normalize_channels=normalize_channels)
    X_ts_tr_final = apply_channelwise_normalizer(X_ts_tr, ts_mean, ts_std)
    X_ts_te_final = apply_channelwise_normalizer(X_ts_te, ts_mean, ts_std)

    tab_mean, tab_std, tab_norm_flag = fit_tabular_normalizer(X_tab_tr, normalize_tabular=normalize_tabular)
    X_tab_tr_final = apply_tabular_normalizer(X_tab_tr, tab_mean, tab_std)
    X_tab_te_final = apply_tabular_normalizer(X_tab_te, tab_mean, tab_std)

    final_seed = int(seed + 30_000 + split_id)
    seed_everything(final_seed, device=device)

    model = build_model_from_params_mm(
        best_params,
        in_channels=in_channels,
        tabular_dim=tabular_dim,
        num_classes=num_classes,
        ModelClass=ModelClass,
        device=device,
    )
    model = train_fixed_epochs_mm(
        model,
        X_ts_tr_final,
        X_tab_tr_final,
        y_tr,
        epochs=int(best_params.get("train_epochs", min(int(epochs), 50))),
        effective_batch_size=int(best_params.get("batch", 32)),
        device=device,
        lr=float(best_params["lr"]),
        weight_decay=1e-4,
        label_smoothing=0.0,
        seed=final_seed + 777,
        max_physical_batch=32,
    )

    test_ds = MultiModalDataset(X_ts_te_final, X_tab_te_final, y_te.astype(np.int64), patients=pat_te)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    patient_ids, y_true_pat, y_pred_pat, proba_pat = eval_patient_outputs_mm(model, test_loader, device=device)

    metrics = compute_patient_metrics(y_true_pat, y_pred_pat, proba_pat)
    metrics.update(
        {
            "EffectiveEpochs": int(best_params.get("train_epochs", 0)),
            "n_patients_train_outer": int(len(np.unique(pat_tr))),
            "n_patients_test_outer": int(len(np.unique(pat_te))),
            "normalize_channels": norm_mask.astype(int).tolist(),
            "normalize_tabular": bool(tab_norm_flag),
        }
    )

    artifacts = {
        "best_params": best_params,
        "patient_ids": patient_ids,
        "y_true": y_true_pat,
        "y_pred": y_pred_pat,
        "proba": proba_pat,
        "normalize_channels": norm_mask.astype(int).tolist(),
        "normalize_tabular": bool(tab_norm_flag),
        "oof_train_patient_ids": oof_train_patient_ids,
        "oof_train_y_true": oof_train_y_true,
        "oof_train_y_pred": oof_train_y_pred,
        "oof_train_proba": oof_train_proba,
    }

    return metrics, artifacts


@ray.remote(num_cpus=1)
def run_nested_split_ray_mm(
    ModelClass: Any,
    X_ts: np.ndarray,
    X_tab: np.ndarray,
    y: np.ndarray,
    patients: np.ndarray,
    *,
    split_file: Optional[str],
    split_id: int,
    test_size: float = 0.2,
    split_seed: int = 12345,
    epochs: int = 160,
    use_optuna: bool = True,
    n_trials: int = 30,
    optuna_max_epochs: int = 30,
    n_inner_splits: int = 3,
    normalize_channels: Optional[Sequence[bool]] = None,
    normalize_tabular: bool = True,
    seed: int = 2026,
    force_cpu: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    device = torch.device("cpu") if force_cpu else get_default_device()
    return run_nested_split_mm(
        ModelClass,
        X_ts,
        X_tab,
        y,
        patients,
        split_file=split_file,
        split_id=int(split_id),
        test_size=float(test_size),
        split_seed=int(split_seed),
        epochs=int(epochs),
        use_optuna=bool(use_optuna),
        n_trials=int(n_trials),
        optuna_max_epochs=int(optuna_max_epochs),
        n_inner_splits=int(n_inner_splits),
        normalize_channels=normalize_channels,
        normalize_tabular=bool(normalize_tabular),
        seed=int(seed),
        device=device,
    )


__all__ = [
    "CNN1D",
    "TCN",
    "FiLMCNN",
    "FiLMTCN",
    "generate_kfold_split_file",
    "run_nested_split_ts",
    "run_nested_split_ray_ts",
    "run_nested_split_mm",
    "run_nested_split_ray_mm",
]