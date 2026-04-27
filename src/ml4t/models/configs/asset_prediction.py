"""Config dataclasses for direct asset-prediction models."""

from __future__ import annotations

from dataclasses import dataclass

from ml4t.models.configs.base import BaseModelConfig


@dataclass(frozen=True, slots=True)
class AssetPredictionConfig(BaseModelConfig):
    """Shared configuration for direct asset-prediction models."""

    model_name: str = "asset_prediction"
    task_type: str = "regression"


@dataclass(frozen=True, slots=True)
class SAEConfig(AssetPredictionConfig):
    """Config for supervised autoencoder predictors."""

    model_name: str = "sae"
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
