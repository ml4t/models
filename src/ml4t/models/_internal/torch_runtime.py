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
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def seed_torch(torch: Any, seed: int, device: Any) -> None:
    torch.manual_seed(seed)
    if getattr(device, "type", "cpu") == "cuda":
        torch.cuda.manual_seed_all(seed)
