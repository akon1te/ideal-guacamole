"""Dataset preparation shared by all training entry points."""
from __future__ import annotations

import numpy as np
import torch


def normalise_trajectory(y: np.ndarray):
    """Return z-scored trajectory plus statistics required for inversion."""
    y = y.astype(np.float32, copy=False)
    mean, std = y.mean(axis=0), y.std(axis=0)
    if np.any(std == 0):
        raise ValueError("Cannot normalise a trajectory with a constant component.")
    return (y - mean) / std, mean, std


def normalise_train_validation(y_train: np.ndarray, y_validation: np.ndarray):
    """Normalise both splits using statistics fitted only on training data.

    Fitting on the entire trajectory leaks information from the held-out future
    region.  The returned statistics must be saved with a checkpoint and used
    unchanged for evaluation.
    """
    y_train_norm, mean, std = normalise_trajectory(y_train)
    y_validation_norm = (y_validation.astype(np.float32, copy=False) - mean) / std
    return y_train_norm, y_validation_norm, mean, std


def split_trajectory(y: np.ndarray, train_fraction: float = 0.8):
    """Chronologically split a trajectory; never shuffle time-series data."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie strictly between 0 and 1.")
    split = int(len(y) * train_fraction)
    return y[:split], y[split:], split


def apply_training_constraints(
    y_train: np.ndarray,
    seq_len: int,
    data_fraction: float = 1.0,
    data_noise: float = 0.0,
    seed: int = 0,
):
    """Apply low-data and measurement-noise experiments to train data only."""
    if not 0 < data_fraction <= 1:
        raise ValueError("data_fraction must lie in (0, 1].")
    n_keep = len(y_train)
    if data_fraction < 1:
        n_keep = max(seq_len + 2, int(round(len(y_train) * data_fraction)))
        n_keep = min(n_keep, len(y_train))
        y_train = y_train[:n_keep]
    if data_noise > 0:
        rng = np.random.RandomState(seed)
        y_train = y_train + data_noise * rng.randn(*y_train.shape).astype(np.float32)
    return y_train, n_keep


def make_windows(values: np.ndarray, seq_len: int):
    """Create ``state_t -> states_[t+1:t+seq_len]`` supervised pairs."""
    if len(values) <= seq_len:
        raise ValueError("Trajectory must be longer than seq_len.")
    x = np.asarray(values[:-seq_len], dtype=np.float32)
    y = np.stack([values[i + 1 : i + 1 + seq_len] for i in range(len(values) - seq_len)])
    return torch.from_numpy(x), torch.from_numpy(y.astype(np.float32, copy=False))
