"""Shared latent-factor runtime utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np

TaskType = Literal["regression", "classification"]


def validate_panel_shapes(
    chars_train: np.ndarray,
    returns_train: np.ndarray,
    chars_val: np.ndarray | None = None,
    returns_val: np.ndarray | None = None,
) -> None:
    """Validate `(T, N, L)` / `(T, N)` panel shapes."""
    if chars_train.ndim != 3:
        raise ValueError("chars_train must be 3D")
    if returns_train.ndim != 2:
        raise ValueError("returns_train must be 2D")
    if chars_train.shape[:2] != returns_train.shape:
        raise ValueError("chars_train and returns_train disagree on (T, N)")
    if chars_val is not None:
        if chars_val.ndim != 3:
            raise ValueError("chars_val must be 3D")
        if chars_train.shape[2] != chars_val.shape[2]:
            raise ValueError("chars_train and chars_val must share the feature dimension")
    if returns_val is not None:
        if returns_val.ndim != 2:
            raise ValueError("returns_val must be 2D")
        if chars_val is None:
            raise ValueError("chars_val is required when returns_val is provided")
        if chars_val.shape[:2] != returns_val.shape:
            raise ValueError("chars_val and returns_val disagree on (T, N)")


def compute_managed_portfolios(chars: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Compute diagonal characteristic-managed portfolios for each date."""
    validate_panel_shapes(chars, returns)
    n_dates, n_slots, n_features = chars.shape
    ones = np.ones((n_dates, n_slots, 1), dtype=np.float32)
    chars_aug = np.concatenate([chars.astype(np.float32, copy=False), ones], axis=2)
    portfolios = np.zeros((n_dates, n_slots, n_features + 1), dtype=np.float32)
    eps = 1e-8

    for date_idx in range(n_dates):
        z_t = chars_aug[date_idx]
        r_t = returns[date_idx]
        valid = np.isfinite(r_t) & np.isfinite(z_t).all(axis=1)
        if not valid.any():
            continue
        z_valid = z_t[valid].astype(np.float64)
        r_valid = r_t[valid].astype(np.float64)
        numerator = (z_valid * r_valid[:, None]).sum(axis=0)
        denominator = (z_valid**2).sum(axis=0)
        x_t = numerator / np.maximum(denominator, eps)
        portfolios[date_idx] = np.broadcast_to(
            x_t.astype(np.float32)[None, :],
            (n_slots, n_features + 1),
        )

    return portfolios


def resolve_checkpoint_epochs(
    max_epoch: int,
    *,
    checkpoint_interval: int | None = 5,
    checkpoint_epochs: list[int] | None = None,
    include_final: bool = True,
) -> list[int]:
    """Resolve a training checkpoint grid."""
    if max_epoch < 1:
        raise ValueError(f"max_epoch must be positive; got {max_epoch}")

    if checkpoint_epochs is not None:
        epochs = sorted({int(epoch) for epoch in checkpoint_epochs if 1 <= int(epoch) <= max_epoch})
        if not epochs:
            raise ValueError("checkpoint_epochs did not contain a valid epoch")
    elif checkpoint_interval is None or checkpoint_interval <= 0:
        epochs = [max_epoch]
    else:
        epochs = list(range(int(checkpoint_interval), max_epoch + 1, int(checkpoint_interval)))
        if not epochs:
            epochs = [max_epoch]

    if include_final and max_epoch not in epochs:
        epochs.append(max_epoch)
    return sorted(set(epochs))


def select_checkpoint_epoch(
    *,
    checkpoint: int | None,
    configured_default: int | None,
    available: tuple[int, ...],
) -> int:
    """Select a checkpoint from the available extraction grid."""
    if checkpoint is not None:
        if checkpoint not in available:
            raise ValueError(f"checkpoint={checkpoint} is not in available_checkpoints={available}")
        return checkpoint
    if configured_default is not None:
        if configured_default not in available:
            raise ValueError(
                f"default_checkpoint={configured_default} is not in available_checkpoints={available}"
            )
        return configured_default
    return available[-1]


def summarize_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    task_type: TaskType,
) -> dict[str, float | int | None]:
    """Summarize validation predictions with task-appropriate metrics."""
    valid = np.isfinite(y_true) & np.isfinite(y_score)
    if not valid.any():
        if task_type == "classification":
            return {
                "n_validation_obs": 0,
                "validation_auc": None,
                "validation_log_loss": None,
            }
        return {
            "n_validation_obs": 0,
            "validation_mean_cs_ic": None,
        }

    if task_type == "classification":
        y_true_valid = y_true[valid]
        y_score_valid = y_score[valid]
        return {
            "n_validation_obs": int(valid.sum()),
            "validation_auc": _binary_auc(y_true_valid, y_score_valid),
            "validation_log_loss": _binary_log_loss(y_true_valid, y_score_valid),
        }

    return {
        "n_validation_obs": int(valid.sum()),
        "validation_mean_cs_ic": mean_cross_sectional_spearman(y_true, y_score),
    }


def mean_cross_sectional_spearman(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Average cross-sectional Spearman correlation across dates."""
    correlations: list[float] = []
    for date_idx in range(y_true.shape[0]):
        valid = np.isfinite(y_true[date_idx]) & np.isfinite(y_score[date_idx])
        if valid.sum() < 3:
            continue
        y_rank = average_ranks(y_true[date_idx, valid])
        score_rank = average_ranks(y_score[date_idx, valid])
        correlation = np.corrcoef(y_rank, score_rank)[0, 1]
        if np.isfinite(correlation):
            correlations.append(float(correlation))
    if not correlations:
        return None
    return float(np.mean(correlations))


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks with stable tie handling."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y = (np.asarray(y_true) > 0.5).astype(np.int8)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = average_ranks(np.asarray(y_score, dtype=np.float64))
    rank_sum_pos = float(ranks[y == 1].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _binary_log_loss(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y = (np.asarray(y_true) > 0.5).astype(np.float64)
    p = np.clip(np.asarray(y_score, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
