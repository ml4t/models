"""Config dataclasses for factor forecasters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ml4t.models.configs.base import BaseModelConfig, _require_real

__all__ = ["AR1ForecasterConfig", "EWMABaseForecasterConfig", "ExpandingMeanForecasterConfig"]


@dataclass(frozen=True, slots=True)
class ExpandingMeanForecasterConfig(BaseModelConfig):
    """Config for the historical-mean factor-premium baseline."""

    model_name: ClassVar[str] = "expanding_mean"


@dataclass(frozen=True, slots=True)
class AR1ForecasterConfig(BaseModelConfig):
    """Config for per-factor AR(1) forecasts."""

    model_name: ClassVar[str] = "ar1"


@dataclass(frozen=True, slots=True)
class EWMABaseForecasterConfig(BaseModelConfig):
    """Config for EWMA factor-premium forecasts."""

    model_name: ClassVar[str] = "ewma"
    half_life: float = 12.0

    def __post_init__(self) -> None:
        BaseModelConfig.__post_init__(self)
        _require_real("half_life", self.half_life, minimum=0.0, inclusive=False)
