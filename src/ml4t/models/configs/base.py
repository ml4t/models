"""Base config types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BaseModelConfig:
    """Common configuration for ML4T models.

    The ``device`` field selects the PyTorch accelerator where applicable:
    ``cpu`` (default), ``cuda``/``cuda:N`` when CUDA is available, or ``mps`` when the
    PyTorch MPS backend is available; otherwise training falls back to CPU.
    """

    seed: int = 42
    device: str = "cpu"
    dtype: str = "float64"
