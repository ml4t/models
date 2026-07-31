"""Shared PyTorch runtime helpers for neural ML4T models."""

from __future__ import annotations

import os
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
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device={requested!r} requires CUDA, but torch.cuda.is_available() is False"
            )
        if ":" in requested_device:
            index = int(requested_device.split(":", maxsplit=1)[1])
            device_count = int(torch.cuda.device_count())
            if index >= device_count:
                raise RuntimeError(
                    f"device={requested!r} requests CUDA index {index}, but {device_count} devices exist"
                )
        return torch.device(requested_device)
    mps_backend = getattr(torch.backends, "mps", None)
    if requested_device == "mps":
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError(f"device={requested!r} requires an available PyTorch MPS backend")
        return torch.device("mps")
    raise ValueError(
        f"requested device must be 'cpu', 'mps', 'cuda', or 'cuda:<index>'; got {requested!r}"
    )


def resolve_dtype(torch: Any, requested: str) -> Any:
    """Resolve a validated public precision name to its PyTorch dtype."""

    if requested == "float32":
        return torch.float32
    if requested == "float64":
        return torch.float64
    raise ValueError(f"requested dtype must be 'float32' or 'float64'; got {requested!r}")


def seed_torch(torch: Any, seed: int, device: Any) -> None:
    torch.manual_seed(seed)
    device_type = getattr(device, "type", "cpu")
    if device_type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        use_deterministic_algorithms = getattr(torch, "use_deterministic_algorithms", None)
        if callable(use_deterministic_algorithms):
            use_deterministic_algorithms(True)
        cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
        if cudnn is not None:
            cudnn.benchmark = False
            cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
    elif device_type == "mps":
        mps_manual_seed = getattr(getattr(torch, "mps", None), "manual_seed", None)
        if callable(mps_manual_seed):
            mps_manual_seed(seed)
