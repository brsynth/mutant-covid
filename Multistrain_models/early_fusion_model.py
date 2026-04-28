from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

import optuna
import ray


torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


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


# -----------------------------
# Validation / normalization
# -----------------------------
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


def validate_X_mask(X: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)

    if X.ndim != 3:
        raise ValueError(f"X must be [N, C, T], got shape {X.shape}")
    if mask.ndim != 3:
        raise ValueError(f"mask must be [N, C, T], got shape {mask.shape}")
    if X.shape != mask.shape:
        raise ValueError(f"X and mask must have identical shape, got {X.shape} vs {mask.shape}")

    mask = (mask > 0).astype(np.float32)
    return X, mask


def fit_masked_channelwise_normalizer(
    X_train: np.ndarray,
    mask_train: np.ndarray,
    normalize_channels: Optional[Sequence[bool]] = None,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train, mask_train = validate_X_mask(X_train, mask_train)
    _, C, _ = X_train.shape
    norm_mask = validate_normalize_channels(normalize_channels, C)

    mean = np.zeros((1, C, 1), dtype=np.float32)
    std = np.ones((1, C, 1), dtype=np.float32)

    if np.any(norm_mask):
        valid_count = mask_train.sum(axis=(0, 2), keepdims=True).astype(np.float32)
        valid_count = np.maximum(valid_count, 1.0)

        ch_mean = (X_train * mask_train).sum(axis=(0, 2), keepdims=True) / valid_count
        sq = ((X_train - ch_mean) ** 2) * mask_train
        ch_var = sq.sum(axis=(0, 2), keepdims=True) / valid_count
        ch_std = np.sqrt(np.maximum(ch_var, eps)).astype(np.float32)

        mean[:, norm_mask, :] = ch_mean[:, norm_mask, :]
        std[:, norm_mask, :] = ch_std[:, norm_mask, :]

    return mean.astype(np.float32), std.astype(np.float32), norm_mask.astype(bool)


def apply_masked_channelwise_normalizer(
    X: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    X, mask = validate_X_mask(X, mask)
    Xn = ((X - mean) / std).astype(np.float32)
    Xn = Xn * mask
    return Xn.astype(np.float32)


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

    if normalize_tabular and P > 0:
        mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
        std = X_train.std(axis=0, keepdims=True).astype(np.float32)
        std = np.maximum(std, eps)

    return mean, std, bool(normalize_tabular and P > 0)


def apply_tabular_normalizer(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.shape[1] == 0:
        return X.astype(np.float32)
    return ((X - mean) / std).astype(np.float32)


# -----------------------------
# Dataset
# -----------------------------
class EarlyFusionDataset(Dataset):
    """
    Always returns:
        X, X_tab, y, mask, patient
    where X_tab may be shape [P=0].
    """
    def __init__(
        self,
        X: np.ndarray,
        X_tab: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
        patients: np.ndarray,
    ):
        X, mask = validate_X_mask(X, mask)
        X_tab = np.asarray(X_tab, dtype=np.float32)
        y = np.asarray(y).astype(np.int64)
        patients = np.asarray(patients).astype(str)

        if X_tab.ndim != 2:
            raise ValueError(f"X_tab must be [N, P], got shape {X_tab.shape}")

        n = len(y)
        if len(X) != n or len(mask) != n or len(X_tab) != n or len(patients) != n:
            raise ValueError("X, X_tab, y, mask, and patients must have same number of rows.")

        self.X = torch.from_numpy(X)
        self.X_tab = torch.from_numpy(X_tab)
        self.y = torch.from_numpy(y)
        self.mask = torch.from_numpy(mask.astype(np.float32))
        self.patients = patients

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.X_tab[idx], self.y[idx], self.mask[idx], self.patients[idx]


# -----------------------------
# Mask-aware pooling
# -----------------------------
def masked_adaptive_avg_pool1d(x: torch.Tensor, mask: torch.Tensor, out_len: int, eps: float = 1e-8) -> torch.Tensor:
    x_sum_like = F.adaptive_avg_pool1d(x * mask, out_len)
    m_avg = F.adaptive_avg_pool1d(mask, out_len)
    return x_sum_like / torch.clamp(m_avg, min=eps)


def masked_adaptive_max_pool1d(x: torch.Tensor, mask: torch.Tensor, out_len: int) -> torch.Tensor:
    neg_inf = torch.finfo(x.dtype).min
    x_masked = torch.where(mask > 0, x, torch.full_like(x, neg_inf))
    out = F.adaptive_max_pool1d(x_masked, out_len)
    pooled_valid = F.adaptive_max_pool1d(mask, out_len)
    out = torch.where(pooled_valid > 0, out, torch.zeros_like(out))
    return out


def pooled_valid_mask(mask: torch.Tensor, out_len: int) -> torch.Tensor:
    return (F.adaptive_max_pool1d(mask, out_len) > 0).to(mask.dtype)

# -----------------------------
# True masked convolution (partial-conv style)
# -----------------------------
class MaskedConv1d(nn.Module):
    """
    1D partial-convolution style layer.

    This keeps all learned quantities inside the training fold: the convolution weights
    are learned only from training data, while the mask path uses a fixed all-ones kernel.
    No information from validation/test folds is used to fit preprocessing or masks.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        padding: Union[int, str] = 0,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(padding, str) and padding != "same":
            raise ValueError("padding must be int or 'same'")
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.padding = padding
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=0,
            bias=bias,
        )
        self.register_buffer("mask_kernel", torch.ones(1, 1, self.kernel_size), persistent=False)

    def _padding_tuple(self) -> Tuple[int, int]:
        if self.padding == "same":
            total = self.dilation * (self.kernel_size - 1)
            left = total // 2
            right = total - left
            return int(left), int(right)
        p = int(self.padding)
        return int(p), int(p)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or mask.ndim != 3:
            raise ValueError("x and mask must both be 3D [B, C, T]")
        if x.shape != mask.shape:
            raise ValueError(f"x and mask must have identical shape, got {x.shape} vs {mask.shape}")

        mask = (mask > 0).to(x.dtype)
        x_masked = x * mask

        pad_left, pad_right = self._padding_tuple()
        if pad_left or pad_right:
            x_masked = F.pad(x_masked, (pad_left, pad_right))

        raw_out = self.conv(x_masked)

        # Temporal validity uses the union across channels. This is the right mask notion
        # for mixed-channel convolutions and avoids letting padded tails create valid outputs.
        mask_any = (mask.amax(dim=1, keepdim=True) > 0).to(x.dtype)
        if pad_left or pad_right:
            mask_any = F.pad(mask_any, (pad_left, pad_right))

        valid_count = F.conv1d(
            mask_any,
            self.mask_kernel.to(dtype=x.dtype, device=x.device),
            bias=None,
            stride=1,
            padding=0,
            dilation=self.dilation,
        )
        full_count = float(self.kernel_size)
        updated_valid = (valid_count > 0).to(x.dtype)
        mask_ratio = full_count / torch.clamp(valid_count, min=1.0)

        if self.conv.bias is not None:
            bias = self.conv.bias.view(1, -1, 1)
            out = (raw_out - bias) * mask_ratio + bias
        else:
            out = raw_out * mask_ratio

        out = out * updated_valid
        out_mask = updated_valid.expand(-1, out.shape[1], -1)
        return out, out_mask

class MaskedGroupNorm1d(nn.Module):
    """
    GroupNorm for tensors shaped [B, C, T], but statistics are computed
    only over valid entries indicated by mask.

    The mask must have shape [B, C, T] and be 1 for valid positions, 0 for invalid.
    Invalid positions are zeroed again after normalization.
    """
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()

        if num_channels <= 0:
            raise ValueError("num_channels must be > 0")
        if num_groups <= 0:
            raise ValueError("num_groups must be > 0")
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
            )

        self.num_groups = int(num_groups)
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.affine = bool(affine)

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or mask.ndim != 3:
            raise ValueError("x and mask must both be 3D [B, C, T]")
        if x.shape != mask.shape:
            raise ValueError(f"x and mask must have identical shape, got {x.shape} vs {mask.shape}")
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected x to have {self.num_channels} channels, got {x.shape[1]}"
            )

        B, C, T = x.shape
        G = self.num_groups
        CpG = C // G

        mask = (mask > 0).to(dtype=x.dtype)
        x = x * mask

        # reshape into groups: [B, G, CpG, T]
        xg = x.view(B, G, CpG, T)
        mg = mask.view(B, G, CpG, T)

        # count valid elements per sample/group over channels+time
        count = mg.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)

        mean = (xg * mg).sum(dim=(2, 3), keepdim=True) / count
        var = (((xg - mean) ** 2) * mg).sum(dim=(2, 3), keepdim=True) / count

        xg = (xg - mean) / torch.sqrt(var + self.eps)

        # zero invalid positions again
        xg = xg * mg

        out = xg.view(B, C, T)

        if self.affine:
            out = out * self.weight + self.bias

        out = out * mask
        return out

# -----------------------------
# Building blocks
# -----------------------------
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
        self.conv = MaskedConv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)

        num_groups = 2 if out_channels % 2 == 0 else 1
        self.norm = MaskedGroupNorm1d(num_groups=num_groups, num_channels=out_channels)
        self.dropout = nn.Dropout1d(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x * mask
        res = self.residual(x)
        x_padded = F.pad(x, (self.padding_size, 0))
        m_padded = F.pad(mask, (self.padding_size, 0))
        out, out_mask = self.conv(x_padded, m_padded)
        out = self.norm(out, out_mask)
        out = F.gelu(out)
        out = self.dropout(out)
        out = F.gelu(out + res * out_mask)
        out = out * out_mask
        return out, out_mask


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


# -----------------------------
# Encoders
# -----------------------------
class CNNEncoderMasked(nn.Module):
    def __init__(
        self,
        in_channels: int,
        k1: int,
        k2: int,
        out_channels: int = 32,
        spatial_resolution: int = 12,
        dropout: float = 0.3,
    ):
        super().__init__()
        half = out_channels // 2
        self.branch1_conv = MaskedConv1d(in_channels, half, kernel_size=k1, padding="same")
        self.branch1_norm = MaskedGroupNorm1d(1, half)
        self.branch2_conv = MaskedConv1d(in_channels, half, kernel_size=k2, padding="same")
        self.branch2_norm = MaskedGroupNorm1d(1, half)
        self.spatial_resolution = int(spatial_resolution)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = int(out_channels * 2 * spatial_resolution)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x * mask
        x1, m1 = self.branch1_conv(x, mask)
        x1 = F.gelu(self.branch1_norm(x1, m1)) * m1
        x2, m2 = self.branch2_conv(x, mask)
        x2 = F.gelu(self.branch2_norm(x2, m2)) * m2
        x = torch.cat([x1, x2], dim=1)
        feat_mask = torch.cat([m1, m2], dim=1)

        avg_p = masked_adaptive_avg_pool1d(x, feat_mask, self.spatial_resolution)
        max_p = masked_adaptive_max_pool1d(x, feat_mask, self.spatial_resolution)

        pooled_mask = pooled_valid_mask(feat_mask, self.spatial_resolution)
        avg_p = avg_p * pooled_mask
        max_p = max_p * pooled_mask

        z = torch.cat([avg_p, max_p], dim=1)
        z = F.gelu(z)
        z = torch.flatten(z, 1)
        z = self.dropout(z)
        return z


class TCNEncoderMasked(nn.Module):
    def __init__(
        self,
        in_channels: int,
        k1: int,
        k2: int,
        dilation: int = 1,
        out_channels: int = 32,
        spatial_resolution: int = 12,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.block1 = TCNBlock(in_channels, out_channels, k1, dilation=dilation, dropout=dropout * 0.5)
        self.block2 = TCNBlock(out_channels, out_channels, k2, dilation=1, dropout=dropout * 0.5)
        self.spatial_resolution = int(spatial_resolution)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = int(out_channels * 2 * spatial_resolution)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x * mask
        x, m = self.block1(x, mask)
        x, m = self.block2(x, m)

        avg_p = masked_adaptive_avg_pool1d(x, m, self.spatial_resolution)
        max_p = masked_adaptive_max_pool1d(x, m, self.spatial_resolution)

        pooled_mask = pooled_valid_mask(m, self.spatial_resolution)
        avg_p = avg_p * pooled_mask
        max_p = max_p * pooled_mask

        z = torch.cat([avg_p, max_p], dim=1)
        z = torch.flatten(z, 1)
        z = self.dropout(z)
        return z


class FiLMCNNEncoderMasked(nn.Module):
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
        half = out_channels // 2
        self.branch1_conv = MaskedConv1d(in_channels, half, kernel_size=k1, padding="same")
        self.branch1_norm = MaskedGroupNorm1d(1, half)
        self.branch2_conv = MaskedConv1d(in_channels, half, kernel_size=k2, padding="same")
        self.branch2_norm = MaskedGroupNorm1d(1, half)
        self.film_gen = FiLMGenerator(tabular_dim, out_channels, hidden_dim=film_hidden, dropout=dropout)
        self.film = FiLMLayer()
        self.spatial_resolution = int(spatial_resolution)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = int(out_channels * 2 * spatial_resolution)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        x = x * mask
        x1, m1 = self.branch1_conv(x, mask)
        x1 = F.gelu(self.branch1_norm(x1, m1)) * m1
        x2, m2 = self.branch2_conv(x, mask)
        x2 = F.gelu(self.branch2_norm(x2, m2)) * m2
        x = torch.cat([x1, x2], dim=1)
        feat_mask = torch.cat([m1, m2], dim=1)

        gamma, beta = self.film_gen(x_tab)
        x = self.film(x, gamma, beta)
        x = F.gelu(x) * feat_mask

        avg_p = masked_adaptive_avg_pool1d(x, feat_mask, self.spatial_resolution)
        max_p = masked_adaptive_max_pool1d(x, feat_mask, self.spatial_resolution)

        pooled_mask = pooled_valid_mask(feat_mask, self.spatial_resolution)
        avg_p = avg_p * pooled_mask
        max_p = max_p * pooled_mask

        z = torch.cat([avg_p, max_p], dim=1)
        z = torch.flatten(z, 1)
        z = self.dropout(z)
        return z


class FiLMTCNEncoderMasked(nn.Module):
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
        self.film_gen = FiLMGenerator(tabular_dim, out_channels, hidden_dim=film_hidden, dropout=dropout)
        self.film = FiLMLayer()
        self.spatial_resolution = int(spatial_resolution)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = int(out_channels * 2 * spatial_resolution)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        x = x * mask
        x, m = self.block1(x, mask)
        x, m = self.block2(x, m)

        gamma, beta = self.film_gen(x_tab)
        x = self.film(x, gamma, beta)
        x = F.gelu(x) * m

        avg_p = masked_adaptive_avg_pool1d(x, m, self.spatial_resolution)
        max_p = masked_adaptive_max_pool1d(x, m, self.spatial_resolution)

        pooled_mask = pooled_valid_mask(m, self.spatial_resolution)
        avg_p = avg_p * pooled_mask
        max_p = max_p * pooled_mask

        z = torch.cat([avg_p, max_p], dim=1)
        z = torch.flatten(z, 1)
        z = self.dropout(z)
        return z


# -----------------------------
# Models
# -----------------------------
class CNN1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        k1: int,
        k2: int,
        out_channels: int = 32,
        spatial_resolution: int = 12,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.encoder = CNNEncoderMasked(in_channels, k1, k2, out_channels, spatial_resolution, dropout)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x, mask)
        return self.head(z)


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
        self.encoder = TCNEncoderMasked(in_channels, k1, k2, dilation, out_channels, spatial_resolution, dropout)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x, mask)
        return self.head(z)


class FiLMCNN_A0Only(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        a0_channel_indices: Sequence[int],
        other_channel_indices: Sequence[int],
        tabular_dim: int,
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
        if tabular_dim <= 0:
            raise ValueError("FiLMCNN_A0Only requires tabular_dim > 0.")
        if len(a0_channel_indices) == 0:
            raise ValueError("FiLMCNN_A0Only requires at least one A0 channel.")

        self.a0_idx = list(map(int, a0_channel_indices))
        self.other_idx = list(map(int, other_channel_indices))

        self.a0_encoder = FiLMCNNEncoderMasked(
            len(self.a0_idx), tabular_dim, k1, k2, out_channels, spatial_resolution, dropout, tab_hidden
        )

        if len(self.other_idx) > 0:
            self.other_encoder = CNNEncoderMasked(
                len(self.other_idx), k1, k2, out_channels, spatial_resolution, dropout
            )
            fusion_in = self.a0_encoder.out_dim + self.other_encoder.out_dim
        else:
            self.other_encoder = None
            fusion_in = self.a0_encoder.out_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        x_a0 = x[:, self.a0_idx, :]
        m_a0 = mask[:, self.a0_idx, :]
        z_a0 = self.a0_encoder(x_a0, m_a0, x_tab)

        if self.other_encoder is not None:
            x_other = x[:, self.other_idx, :]
            m_other = mask[:, self.other_idx, :]
            z_other = self.other_encoder(x_other, m_other)
            z = torch.cat([z_a0, z_other], dim=1)
        else:
            z = z_a0

        return self.head(z)


class FiLMTCN_A0Only(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        a0_channel_indices: Sequence[int],
        other_channel_indices: Sequence[int],
        tabular_dim: int,
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
        if tabular_dim <= 0:
            raise ValueError("FiLMTCN_A0Only requires tabular_dim > 0.")
        if len(a0_channel_indices) == 0:
            raise ValueError("FiLMTCN_A0Only requires at least one A0 channel.")

        self.a0_idx = list(map(int, a0_channel_indices))
        self.other_idx = list(map(int, other_channel_indices))

        self.a0_encoder = FiLMTCNEncoderMasked(
            len(self.a0_idx), tabular_dim, k1, k2, dilation, out_channels, spatial_resolution, dropout, tab_hidden
        )

        if len(self.other_idx) > 0:
            self.other_encoder = TCNEncoderMasked(
                len(self.other_idx), k1, k2, dilation, out_channels, spatial_resolution, dropout
            )
            fusion_in = self.a0_encoder.out_dim + self.other_encoder.out_dim
        else:
            self.other_encoder = None
            fusion_in = self.a0_encoder.out_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        x_a0 = x[:, self.a0_idx, :]
        m_a0 = mask[:, self.a0_idx, :]
        z_a0 = self.a0_encoder(x_a0, m_a0, x_tab)

        if self.other_encoder is not None:
            x_other = x[:, self.other_idx, :]
            m_other = mask[:, self.other_idx, :]
            z_other = self.other_encoder(x_other, m_other)
            z = torch.cat([z_a0, z_other], dim=1)
        else:
            z = z_a0

        return self.head(z)


# -----------------------------
# Training / evaluation
# -----------------------------
def train_epoch(
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

    for step, (Xb, Xtab, yb, mb, _) in enumerate(loader, start=1):
        Xb = Xb.to(device, non_blocking=True)
        Xtab = Xtab.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)

        logits = model(Xb, mb, Xtab)
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


def _macro_specificity(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities: List[float] = []
    for i in range(num_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        specificities.append(float(spec))
    return float(np.mean(specificities)) if specificities else 0.0


@torch.no_grad()
def eval_patient_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    verify_patient_labels: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    probs_by_pat: Dict[str, List[np.ndarray]] = {}
    y_by_pat: Dict[str, int] = {}

    for Xb, Xtab, yb, mb, pb in loader:
        Xb = Xb.to(device, non_blocking=True)
        Xtab = Xtab.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)

        logits = model(Xb, mb, Xtab)
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


# -----------------------------
# Fixed-epoch training
# -----------------------------
def train_fixed_epochs(
    model,
    X_tr,
    X_tab_tr,
    y_tr,
    mask_tr,
    *,
    epochs,
    effective_batch_size: int,
    device,
    lr,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.0,
    seed: int = 0,
    max_physical_batch: int = 32,
):
    dummy_patients = np.asarray([f"train_{i}" for i in range(len(y_tr))], dtype=str)
    train_ds = EarlyFusionDataset(X_tr, X_tab_tr, y_tr.astype(np.int64), mask_tr, dummy_patients)
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
        train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device=device,
            accum_steps=accum_steps,
        )
        scheduler.step()

    return model


# -----------------------------
# Patient split helpers
# -----------------------------
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


def masks_from_patients(
    patients: np.ndarray,
    train_patients: np.ndarray,
    test_patients: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    train_mask = np.isin(patients, train_patients)
    test_mask = np.isin(patients, test_patients)
    return train_mask, test_mask


# -----------------------------
# Reusable split file
# -----------------------------
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
        raise ValueError(f"n_splits={n_splits} is too large for smallest class={counts.min()}.")

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
            "Split file is incompatible with this dataset.\n"
            f"Examples missing patients: {ex}"
        )


# -----------------------------
# Optuna spaces
# -----------------------------
RES_UNIVERSE = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 44, 50, 64]


def tcn_receptive_field(k1: int, k2: int, d1: int, d2: int = 1) -> int:
    return 1 + (k1 - 1) * d1 + (k2 - 1) * d2


def suggest_params(trial: optuna.Trial, ModelClass: Any, *, max_train_epochs: int) -> Dict[str, Any]:
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

    if ModelClass in [TCN, FiLMTCN_A0Only]:
        params["dilation"] = int(trial.suggest_categorical("dilation", [1, 2, 4, 8]))
    else:
        params["dilation"] = 1

    if ModelClass in [FiLMCNN_A0Only, FiLMTCN_A0Only]:
        params["tab_hidden"] = trial.suggest_categorical("tab_hidden", [32, 64, 128])
        params["fusion_hidden"] = trial.suggest_categorical("fusion_hidden", [32, 64, 128])

    return params


def sanitize_params_for_data(params: Dict[str, Any], ModelClass: Any, T: int) -> Dict[str, Any]:
    p = dict(params)

    r = int(p.get("res", 12))
    r = min(r, int(T))
    if int(p.get("out_channels", 32)) == 64:
        r = min(r, 36)
    r = max(8, r)
    if r % 2 == 1:
        r -= 1
    p["res"] = int(r)

    if ModelClass in [TCN, FiLMTCN_A0Only]:
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


def build_model_from_params(
    params: Dict[str, Any],
    in_channels: int,
    num_classes: int,
    ModelClass: Any,
    device: torch.device,
    *,
    tabular_dim: int = 0,
    a0_channel_indices: Optional[Sequence[int]] = None,
    other_channel_indices: Optional[Sequence[int]] = None,
) -> nn.Module:
    common = dict(
        in_channels=in_channels,
        num_classes=num_classes,
        k1=int(params["k1"]),
        k2=int(params["k2"]),
        out_channels=int(params["out_channels"]),
        spatial_resolution=int(params["res"]),
        dropout=float(params["do"]),
    )

    if ModelClass is CNN1D:
        return CNN1D(**common).to(device)

    if ModelClass is TCN:
        return TCN(dilation=int(params.get("dilation", 1)), **common).to(device)

    if ModelClass is FiLMCNN_A0Only:
        return FiLMCNN_A0Only(
            **common,
            a0_channel_indices=list(a0_channel_indices or []),
            other_channel_indices=list(other_channel_indices or []),
            tabular_dim=int(tabular_dim),
            tab_hidden=int(params["tab_hidden"]),
            fusion_hidden=int(params["fusion_hidden"]),
        ).to(device)

    if ModelClass is FiLMTCN_A0Only:
        return FiLMTCN_A0Only(
            **common,
            a0_channel_indices=list(a0_channel_indices or []),
            other_channel_indices=list(other_channel_indices or []),
            tabular_dim=int(tabular_dim),
            dilation=int(params.get("dilation", 1)),
            tab_hidden=int(params["tab_hidden"]),
            fusion_hidden=int(params["fusion_hidden"]),
        ).to(device)

    raise ValueError(f"Unknown ModelClass: {ModelClass}")


# -----------------------------
# INNER objective
# -----------------------------
def inner_objective_nested(
    trial: optuna.Trial,
    X_train: np.ndarray,
    X_tab_train: np.ndarray,
    y_train: np.ndarray,
    mask_train: np.ndarray,
    patients_train: np.ndarray,
    ModelClass: Any,
    device: torch.device,
    *,
    max_train_epochs: int,
    n_inner_splits: int,
    base_seed: int,
    normalize_channels: Optional[Sequence[bool]],
    normalize_tabular: bool,
    a0_channel_indices: Optional[Sequence[int]],
    other_channel_indices: Optional[Sequence[int]],
) -> float:
    in_channels = int(X_train.shape[1])
    num_classes = int(len(np.unique(y_train)))
    T = int(X_train.shape[2])
    tabular_dim = int(X_tab_train.shape[1])

    params = suggest_params(trial, ModelClass, max_train_epochs=int(max_train_epochs))
    params = sanitize_params_for_data(params, ModelClass, T)

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
        raise ValueError("n_inner_splits must be >= 2.")
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

        tr_mask_bool, va_mask_bool = masks_from_patients(patients_train, tr_p, va_p)

        X_tr = X_train[tr_mask_bool]
        Xtab_tr = X_tab_train[tr_mask_bool]
        y_tr = y_train[tr_mask_bool]
        m_tr = mask_train[tr_mask_bool]
        pat_tr = patients_train[tr_mask_bool]

        X_va = X_train[va_mask_bool]
        Xtab_va = X_tab_train[va_mask_bool]
        y_va = y_train[va_mask_bool]
        m_va = mask_train[va_mask_bool]
        pat_va = patients_train[va_mask_bool]

        mean, std, _ = fit_masked_channelwise_normalizer(X_tr, m_tr, normalize_channels=normalize_channels)
        X_tr = apply_masked_channelwise_normalizer(X_tr, m_tr, mean, std)
        X_va = apply_masked_channelwise_normalizer(X_va, m_va, mean, std)

        tab_mean, tab_std, _ = fit_tabular_normalizer(Xtab_tr, normalize_tabular=normalize_tabular)
        Xtab_tr = apply_tabular_normalizer(Xtab_tr, tab_mean, tab_std)
        Xtab_va = apply_tabular_normalizer(Xtab_va, tab_mean, tab_std)

        model = build_model_from_params(
            params,
            in_channels=in_channels,
            num_classes=num_classes,
            ModelClass=ModelClass,
            device=device,
            tabular_dim=tabular_dim,
            a0_channel_indices=a0_channel_indices,
            other_channel_indices=other_channel_indices,
        )

        model = train_fixed_epochs(
            model,
            X_tr,
            Xtab_tr,
            y_tr,
            m_tr,
            epochs=int(params["train_epochs"]),
            effective_batch_size=int(params["batch"]),
            device=device,
            lr=float(params["lr"]),
            weight_decay=1e-4,
            label_smoothing=0.0,
            seed=seed_k,
            max_physical_batch=32,
        )

        val_ds = EarlyFusionDataset(X_va, Xtab_va, y_va.astype(np.int64), m_va, pat_va.astype(str))
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

        _, y_true_pat, y_pred_pat, _ = eval_patient_outputs(model, val_loader, device=device)
        scores.append(float(balanced_accuracy_score(y_true_pat, y_pred_pat)))

    return float(np.mean(scores)) if scores else 0.0


# -----------------------------
# TRUE nested CV: one OUTER split
# -----------------------------
def run_nested_split(
    ModelClass: Any,
    X: np.ndarray,
    X_tab: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    patients: np.ndarray,
    *,
    split_file: Optional[str],
    split_id: int,
    epochs: int = 160,
    use_optuna: bool = True,
    n_trials: int = 30,
    optuna_max_epochs: int = 30,
    n_inner_splits: int = 3,
    normalize_channels: Optional[Sequence[bool]] = None,
    normalize_tabular: bool = True,
    seed: int = 2026,
    device: Optional[torch.device] = None,
    a0_channel_indices: Optional[Sequence[int]] = None,
    other_channel_indices: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    X = np.asarray(X, dtype=np.float32)
    X_tab = np.asarray(X_tab, dtype=np.float32)
    y = np.asarray(y).astype(int)
    mask = np.asarray(mask, dtype=np.float32)
    patients = np.asarray(patients).astype(str)

    X, mask = validate_X_mask(X, mask)

    if X_tab.ndim != 2 or len(X_tab) != len(y):
        raise ValueError("X_tab must be [N, P] and aligned with X/y.")

    _patient_labels(patients, y)

    if device is None:
        device = get_default_device()

    seed_everything(int(seed), device=device)

    in_channels = int(X.shape[1])
    num_classes = int(len(np.unique(y)))
    tabular_dim = int(X_tab.shape[1])

    if num_classes < 2:
        raise ValueError("Need at least 2 classes.")

    normalize_channels = validate_normalize_channels(normalize_channels, in_channels)

    if split_file is None:
        raise ValueError("This early-fusion pipeline expects a shared split_file for outer CV.")

    payload = load_split_file(split_file)
    spec = get_split_from_file(payload, split_id=split_id)
    assert_split_compatible(patients, spec)

    train_patients = np.asarray(spec.train_patients).astype(str)
    test_patients_ = np.asarray(spec.test_patients).astype(str)

    train_mask_bool, test_mask_bool = masks_from_patients(patients, train_patients, test_patients_)
    if train_mask_bool.sum() == 0 or test_mask_bool.sum() == 0:
        raise ValueError("Outer split produced empty train or test set.")

    X_tr, Xtab_tr, y_tr, m_tr, pat_tr = (
        X[train_mask_bool], X_tab[train_mask_bool], y[train_mask_bool], mask[train_mask_bool], patients[train_mask_bool]
    )
    X_te, Xtab_te, y_te, m_te, pat_te = (
        X[test_mask_bool], X_tab[test_mask_bool], y[test_mask_bool], mask[test_mask_bool], patients[test_mask_bool]
    )

    T = int(X.shape[2])

    if use_optuna:
        sampler = optuna.samplers.TPESampler(seed=int(seed + 10_000 + split_id))
        pruner = optuna.pruners.NopPruner()
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        base_seed = int(seed + 20_000 + split_id * 1000)
        max_train_epochs = int(max(5, min(optuna_max_epochs, epochs)))

        study.optimize(
            lambda t: inner_objective_nested(
                t,
                X_tr, Xtab_tr, y_tr, m_tr, pat_tr,
                ModelClass=ModelClass,
                device=device,
                max_train_epochs=max_train_epochs,
                n_inner_splits=int(n_inner_splits),
                base_seed=base_seed,
                normalize_channels=normalize_channels,
                normalize_tabular=normalize_tabular,
                a0_channel_indices=a0_channel_indices,
                other_channel_indices=other_channel_indices,
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
            "tab_hidden": 64,
            "fusion_hidden": 64,
        }

    best_params = sanitize_params_for_data(best_params, ModelClass, T)

    mean, std, norm_mask = fit_masked_channelwise_normalizer(X_tr, m_tr, normalize_channels=normalize_channels)
    X_tr_final = apply_masked_channelwise_normalizer(X_tr, m_tr, mean, std)
    X_te_final = apply_masked_channelwise_normalizer(X_te, m_te, mean, std)

    tab_mean, tab_std, tab_norm_flag = fit_tabular_normalizer(Xtab_tr, normalize_tabular=normalize_tabular)
    Xtab_tr_final = apply_tabular_normalizer(Xtab_tr, tab_mean, tab_std)
    Xtab_te_final = apply_tabular_normalizer(Xtab_te, tab_mean, tab_std)

    final_seed = int(seed + 30_000 + split_id)
    seed_everything(final_seed, device=device)

    model = build_model_from_params(
        best_params,
        in_channels=in_channels,
        num_classes=num_classes,
        ModelClass=ModelClass,
        device=device,
        tabular_dim=tabular_dim,
        a0_channel_indices=a0_channel_indices,
        other_channel_indices=other_channel_indices,
    )

    model = train_fixed_epochs(
        model,
        X_tr_final,
        Xtab_tr_final,
        y_tr,
        m_tr,
        epochs=int(best_params.get("train_epochs", min(int(epochs), 50))),
        effective_batch_size=int(best_params.get("batch", 32)),
        device=device,
        lr=float(best_params["lr"]),
        weight_decay=1e-4,
        label_smoothing=0.0,
        seed=final_seed + 777,
        max_physical_batch=32,
    )

    test_ds = EarlyFusionDataset(X_te_final, Xtab_te_final, y_te.astype(np.int64), m_te, pat_te)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    patient_ids, y_true_pat, y_pred_pat, proba_pat = eval_patient_outputs(model, test_loader, device=device)

    metrics = compute_patient_metrics(y_true_pat, y_pred_pat, proba_pat)
    metrics.update({
        "EffectiveEpochs": int(best_params.get("train_epochs", 0)),
        "n_patients_train_outer": int(len(np.unique(pat_tr))),
        "n_patients_test_outer": int(len(np.unique(pat_te))),
        "normalize_channels": norm_mask.astype(int).tolist(),
        "normalize_tabular": bool(tab_norm_flag),
    })

    artifacts = {
        "best_params": best_params,
        "patient_ids": patient_ids,
        "y_true": y_true_pat,
        "y_pred": y_pred_pat,
        "proba": proba_pat,
        "normalize_channels": norm_mask.astype(int).tolist(),
        "normalize_tabular": bool(tab_norm_flag),
    }

    return metrics, artifacts


@ray.remote(num_cpus=1)
def run_nested_split_ray(
    ModelClass: Any,
    X: np.ndarray,
    X_tab: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    patients: np.ndarray,
    *,
    split_file: Optional[str],
    split_id: int,
    epochs: int = 160,
    use_optuna: bool = True,
    n_trials: int = 30,
    optuna_max_epochs: int = 30,
    n_inner_splits: int = 3,
    normalize_channels: Optional[Sequence[bool]] = None,
    normalize_tabular: bool = True,
    seed: int = 2026,
    force_cpu: bool = True,
    a0_channel_indices: Optional[Sequence[int]] = None,
    other_channel_indices: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    device = torch.device("cpu") if force_cpu else get_default_device()

    return run_nested_split(
        ModelClass,
        X,
        X_tab,
        y,
        mask,
        patients,
        split_file=split_file,
        split_id=split_id,
        epochs=int(epochs),
        use_optuna=bool(use_optuna),
        n_trials=int(n_trials),
        optuna_max_epochs=int(optuna_max_epochs),
        n_inner_splits=int(n_inner_splits),
        normalize_channels=normalize_channels,
        normalize_tabular=bool(normalize_tabular),
        seed=int(seed),
        device=device,
        a0_channel_indices=a0_channel_indices,
        other_channel_indices=other_channel_indices,
    )