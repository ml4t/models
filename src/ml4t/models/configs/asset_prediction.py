"""Config dataclasses for direct asset-prediction models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ml4t.models.configs.base import (
    BaseModelConfig,
    _require_int,
    _require_real,
    _validate_checkpoint_schedule,
)

__all__ = ["AssetPredictionConfig", "SAEConfig"]


@dataclass(frozen=True, slots=True)
class AssetPredictionConfig(BaseModelConfig):
    """Shared configuration for direct asset-prediction models."""

    model_name: ClassVar[str] = "asset_prediction"
    task_type: str = "regression"

    def __post_init__(self) -> None:
        BaseModelConfig.__post_init__(self)
        if self.task_type not in {"regression", "classification"}:
            raise ValueError(
                f"task_type must be 'regression' or 'classification'; got {self.task_type!r}"
            )


@dataclass(frozen=True, slots=True)
class SAEConfig(AssetPredictionConfig):
    """Config for supervised autoencoder predictors."""

    model_name: ClassVar[str] = "sae"
    bottleneck_dim: int = 96
    aux_hidden_dim: int = 96
    main_hidden_units: tuple[int, ...] = (896, 448, 448, 256)
    dropout_rates: tuple[float, ...] | None = None
    noise_std: float = 0.035
    alpha: float = 1.0
    aux_weight: float = 1.0
    n_epochs: int = 50
    batch_size: int | None = None
    checkpoint_interval: int | None = 5
    checkpoint_epochs: tuple[int, ...] = ()
    default_checkpoint: int | None = None
    lr: float = 1e-4

    def __post_init__(self) -> None:
        AssetPredictionConfig.__post_init__(self)
        _require_int("bottleneck_dim", self.bottleneck_dim)
        _require_int("aux_hidden_dim", self.aux_hidden_dim)
        if len(self.main_hidden_units) != 4 or any(
            type(unit) is not int or unit < 1 for unit in self.main_hidden_units
        ):
            raise ValueError(
                "main_hidden_units must contain four positive integers; "
                f"got {self.main_hidden_units!r}"
            )
        if self.dropout_rates is not None and (
            len(self.dropout_rates) != 8
            or any(not 0.0 <= rate < 1.0 for rate in self.dropout_rates)
        ):
            raise ValueError(
                f"dropout_rates must contain eight values in [0, 1); got {self.dropout_rates!r}"
            )
        _require_real("noise_std", self.noise_std, minimum=0.0)
        _require_real("alpha", self.alpha, minimum=0.0)
        _require_real("aux_weight", self.aux_weight, minimum=0.0)
        _require_int("n_epochs", self.n_epochs)
        if self.batch_size is not None:
            _require_int("batch_size", self.batch_size)
        _require_real("lr", self.lr, minimum=0.0, inclusive=False)
        _validate_checkpoint_schedule(
            total=self.n_epochs,
            interval=self.checkpoint_interval,
            checkpoints=self.checkpoint_epochs,
            default=self.default_checkpoint,
        )
