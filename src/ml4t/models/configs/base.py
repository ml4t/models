"""Base config types."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isfinite

__all__ = ["BaseModelConfig"]


def _require_int(name: str, value: int, *, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}")


def _require_real(name: str, value: float, *, minimum: float, inclusive: bool = True) -> None:
    if not isfinite(value) or (value < minimum if inclusive else value <= minimum):
        operator = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be finite and {operator} {minimum}; got {value!r}")


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite; got {value!r}")


def _require_probability(name: str, value: float) -> None:
    if not isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0, 1); got {value!r}")


def _validate_checkpoint_schedule(
    *,
    total: int,
    interval: int | None,
    checkpoints: tuple[int, ...],
    default: int | None,
    prefix: str = "",
) -> None:
    if interval is not None:
        _require_int(f"{prefix}checkpoint_interval", interval)
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError(f"{prefix}checkpoint_epochs must not contain duplicates")
    for checkpoint in checkpoints:
        _require_int(f"{prefix}checkpoint_epochs entry", checkpoint)
        if checkpoint > total:
            raise ValueError(
                f"{prefix}checkpoint_epochs entries must be <= {total}; got {checkpoint}"
            )
    if default is not None:
        _require_int(f"{prefix}default_checkpoint", default)
        if default > total:
            raise ValueError(f"{prefix}default_checkpoint must be <= {total}; got {default}")


@dataclass(frozen=True, slots=True)
class BaseModelConfig:
    """Common configuration for ML4T models."""

    seed: int = 42
    device: str = "cpu"
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError(f"seed must be an integer; got {self.seed!r}")
        if re.fullmatch(r"(?:cpu|mps|cuda(?::[0-9]+)?)", self.device) is None:
            raise ValueError(
                f"device must be 'cpu', 'mps', 'cuda', or 'cuda:<index>'; got {self.device!r}"
            )
        if self.dtype not in {"float32", "float64"}:
            raise ValueError(f"dtype must be 'float32' or 'float64'; got {self.dtype!r}")
