"""Shared PyTorch runtime helpers for neural ML4T models."""

from __future__ import annotations

from typing import Any


def import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This model requires PyTorch. Install torch to use neural ML4T models."
        ) from exc
    return torch


def resolve_device(torch: Any, requested: str) -> Any:
    requested_device = requested.strip().lower()
    if requested_device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    mps_backend = getattr(torch.backends, "mps", None)
    if requested_device == "mps" and mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_torch(torch: Any, seed: int, device: Any) -> None:
    torch.manual_seed(seed)
    device_type = getattr(device, "type", "cpu")
    if device_type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device_type == "mps":
        mps_manual_seed = getattr(getattr(torch, "mps", None), "manual_seed", None)
        if callable(mps_manual_seed):
            mps_manual_seed(seed)
