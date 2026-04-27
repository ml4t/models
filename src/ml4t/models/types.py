"""Typed batches and result objects for ml4t-models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array1D = NDArray[np.object_]
FloatArray1D = NDArray[np.float64]
Array2D = NDArray[np.float64]
Array3D = NDArray[np.float64]
Array4D = NDArray[np.float64]


def _coerce_float_array(
    values: NDArray[Any] | None, ndim: int, name: str
) -> NDArray[np.float64] | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D; got shape {array.shape}")
    return array


def _resolve_panel_shape(
    returns: NDArray[np.float64] | None,
    characteristics: NDArray[np.float64] | None,
    timestamps: tuple[Any, ...],
    asset_ids: tuple[str, ...],
) -> tuple[int, int]:
    if returns is not None:
        return int(returns.shape[0]), int(returns.shape[1])
    if characteristics is not None:
        return int(characteristics.shape[0]), int(characteristics.shape[1])
    if timestamps and asset_ids:
        return len(timestamps), len(asset_ids)
    raise ValueError(
        "Need returns, characteristics, or both timestamps and asset_ids to determine panel shape."
    )


@dataclass(slots=True, frozen=True)
class PersistentPanelBatch:
    """Stable-entity panel for models such as PCA and RP-PCA."""

    returns: Array2D | None = None
    characteristics: Array3D | None = None
    timestamps: tuple[Any, ...] = ()
    asset_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        returns = _coerce_float_array(self.returns, ndim=2, name="returns")
        characteristics = _coerce_float_array(
            self.characteristics,
            ndim=3,
            name="characteristics",
        )
        object.__setattr__(self, "returns", returns)
        object.__setattr__(self, "characteristics", characteristics)

        n_periods, n_assets = _resolve_panel_shape(
            returns=returns,
            characteristics=characteristics,
            timestamps=self.timestamps,
            asset_ids=self.asset_ids,
        )
        if characteristics is not None and characteristics.shape[:2] != (n_periods, n_assets):
            raise ValueError("characteristics and returns disagree on (T, N)")
        if self.timestamps and len(self.timestamps) != n_periods:
            raise ValueError("timestamps length does not match number of periods")
        if self.asset_ids and len(self.asset_ids) != n_assets:
            raise ValueError("asset_ids length does not match number of assets")

    @property
    def n_periods(self) -> int:
        return _resolve_panel_shape(
            returns=self.returns,
            characteristics=self.characteristics,
            timestamps=self.timestamps,
            asset_ids=self.asset_ids,
        )[0]

    @property
    def n_assets(self) -> int:
        return _resolve_panel_shape(
            returns=self.returns,
            characteristics=self.characteristics,
            timestamps=self.timestamps,
            asset_ids=self.asset_ids,
        )[1]


@dataclass(slots=True, frozen=True)
class CrossSectionBatch:
    """Dated observed cross-sections with a date-local slot axis."""

    characteristics: Array3D
    returns: Array2D | None = None
    factor_returns: Array2D | None = None
    context_features: Array2D | None = None
    timestamps: tuple[Any, ...] = ()
    asset_ids: tuple[str, ...] = ()
    mask: NDArray[np.bool_] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        characteristics = _coerce_float_array(self.characteristics, ndim=3, name="characteristics")
        returns = _coerce_float_array(self.returns, ndim=2, name="returns")
        factor_returns = _coerce_float_array(self.factor_returns, ndim=2, name="factor_returns")
        context_features = _coerce_float_array(
            self.context_features,
            ndim=2,
            name="context_features",
        )
        assert characteristics is not None
        object.__setattr__(self, "characteristics", characteristics)
        object.__setattr__(self, "returns", returns)
        object.__setattr__(self, "factor_returns", factor_returns)
        object.__setattr__(self, "context_features", context_features)

        n_periods, n_slots = characteristics.shape[:2]
        if returns is not None and returns.shape != (n_periods, n_slots):
            raise ValueError("returns and characteristics disagree on (T, N)")
        if factor_returns is not None and factor_returns.shape != (n_periods, n_slots):
            raise ValueError("factor_returns and characteristics disagree on (T, N)")
        if context_features is not None and context_features.shape[0] != n_periods:
            raise ValueError("context_features and characteristics disagree on T")
        if self.timestamps and len(self.timestamps) != n_periods:
            raise ValueError("timestamps length does not match number of periods")
        if self.asset_ids and len(self.asset_ids) != n_slots:
            raise ValueError("asset_ids length does not match number of slots")
        if self.mask is not None and np.asarray(self.mask).shape != (n_periods, n_slots):
            raise ValueError("mask must match (T, N)")

    @property
    def n_periods(self) -> int:
        return self.characteristics.shape[0]

    @property
    def n_assets(self) -> int:
        return self.characteristics.shape[1]


@dataclass(slots=True, frozen=True)
class PortfolioSequenceBatch:
    """Sequence batch for end-to-end portfolio learners."""

    features: Array4D
    returns: Array3D | None = None
    vol_scale: Array3D | None = None
    prev_weights: Array2D | None = None
    mask: NDArray[np.bool_] | None = None
    group_ids: NDArray[np.int64] | None = None
    costs: Array2D | None = None
    adjacency_mask: NDArray[np.bool_] | None = None
    timestamps: tuple[Any, ...] = ()
    asset_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        features = _coerce_float_array(self.features, ndim=4, name="features")
        returns = _coerce_float_array(self.returns, ndim=3, name="returns")
        vol_scale = _coerce_float_array(self.vol_scale, ndim=3, name="vol_scale")
        prev_weights = _coerce_float_array(self.prev_weights, ndim=2, name="prev_weights")
        assert features is not None
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "returns", returns)
        object.__setattr__(self, "vol_scale", vol_scale)
        object.__setattr__(self, "prev_weights", prev_weights)

        batch_size, _, n_assets, _ = features.shape
        if returns is not None and returns.shape[:2] != features.shape[:2]:
            raise ValueError("returns and features disagree on (B, T)")
        if returns is not None and returns.shape[2] != n_assets:
            raise ValueError("returns and features disagree on N")
        if vol_scale is not None and vol_scale.shape != features.shape[:3]:
            raise ValueError("vol_scale and features disagree on (B, T, N)")
        if prev_weights is not None and prev_weights.shape != (batch_size, n_assets):
            raise ValueError("prev_weights must have shape (B, N)")
        if self.mask is not None and np.asarray(self.mask).shape != features.shape[:3]:
            raise ValueError("mask must match (B, T, N)")
        if self.group_ids is not None and np.asarray(self.group_ids).shape != (n_assets,):
            raise ValueError("group_ids must have shape (N,)")
        costs = None
        if self.costs is not None:
            costs = np.asarray(self.costs, dtype=np.float64)
            if costs.ndim == 1:
                costs = costs[:, None]
            if costs.ndim != 2 or costs.shape != (n_assets, 1):
                raise ValueError("costs must have shape (N,) or (N, 1)")
        object.__setattr__(self, "costs", costs)
        if self.adjacency_mask is not None and np.asarray(self.adjacency_mask).shape != (
            n_assets,
            n_assets,
        ):
            raise ValueError("adjacency_mask must have shape (N, N)")
        if self.asset_ids and len(self.asset_ids) != n_assets:
            raise ValueError("asset_ids length does not match N")

    @property
    def batch_size(self) -> int:
        return self.features.shape[0]

    @property
    def n_periods(self) -> int:
        return self.features.shape[1]

    @property
    def n_assets(self) -> int:
        return self.features.shape[2]


@dataclass(slots=True, frozen=True)
class FitSummary:
    """Fit outcome for a model or forecaster."""

    converged: bool
    train_metrics: dict[str, float] = field(default_factory=dict)
    val_metrics: dict[str, float] = field(default_factory=dict)
    best_epoch: int | None = None
    history: tuple[dict[str, float | str], ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class LatentFactorState:
    """Structural latent-factor state extracted from a batch."""

    asset_betas: Array3D
    factor_returns: Array2D | None = None
    checkpoint_epoch: int | None = None
    timestamps: tuple[Any, ...] = ()
    asset_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        asset_betas = _coerce_float_array(self.asset_betas, ndim=3, name="asset_betas")
        factor_returns = _coerce_float_array(self.factor_returns, ndim=2, name="factor_returns")
        assert asset_betas is not None
        object.__setattr__(self, "asset_betas", asset_betas)
        object.__setattr__(self, "factor_returns", factor_returns)

        n_periods, n_assets, n_factors = asset_betas.shape
        if factor_returns is not None and factor_returns.shape != (n_periods, n_factors):
            raise ValueError("factor_returns must have shape (T, K)")
        if self.timestamps and len(self.timestamps) != n_periods:
            raise ValueError("timestamps length does not match T")
        if self.asset_ids and len(self.asset_ids) != n_assets:
            raise ValueError("asset_ids length does not match N")

    @property
    def n_periods(self) -> int:
        return self.asset_betas.shape[0]

    @property
    def n_assets(self) -> int:
        return self.asset_betas.shape[1]

    @property
    def n_factors(self) -> int:
        return self.asset_betas.shape[2]


@dataclass(slots=True, frozen=True)
class FactorForecastResult:
    """Forecast of latent factor premia."""

    factor_premia: Array2D
    timestamps: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        factor_premia = _coerce_float_array(self.factor_premia, ndim=2, name="factor_premia")
        assert factor_premia is not None
        object.__setattr__(self, "factor_premia", factor_premia)
        if self.timestamps and len(self.timestamps) != factor_premia.shape[0]:
            raise ValueError("timestamps length does not match T")


@dataclass(slots=True, frozen=True)
class AssetForecastResult:
    """Asset-level expected-return forecasts."""

    expected_returns: Array2D
    timestamps: tuple[Any, ...] = ()
    asset_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected_returns = _coerce_float_array(
            self.expected_returns,
            ndim=2,
            name="expected_returns",
        )
        assert expected_returns is not None
        object.__setattr__(self, "expected_returns", expected_returns)
        n_periods, n_assets = expected_returns.shape
        if self.timestamps and len(self.timestamps) != n_periods:
            raise ValueError("timestamps length does not match T")
        if self.asset_ids and len(self.asset_ids) != n_assets:
            raise ValueError("asset_ids length does not match N")


@dataclass(slots=True, frozen=True)
class AssetSignalResult:
    """Asset-level predictive signals."""

    signal_values: Array2D
    timestamps: tuple[Any, ...] = ()
    asset_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        signal_values = _coerce_float_array(
            self.signal_values,
            ndim=2,
            name="signal_values",
        )
        assert signal_values is not None
        object.__setattr__(self, "signal_values", signal_values)
        n_periods, n_assets = signal_values.shape
        if self.timestamps and len(self.timestamps) != n_periods:
            raise ValueError("timestamps length does not match T")
        if self.asset_ids and len(self.asset_ids) != n_assets:
            raise ValueError("asset_ids length does not match N")


@dataclass(slots=True, frozen=True)
class AssetWeightsResult:
    """Cross-sectional asset-weight output indexed by date and asset."""

    weights: Array2D
    timestamps: tuple[Any, ...] = ()
    asset_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        weights = _coerce_float_array(self.weights, ndim=2, name="weights")
        assert weights is not None
        object.__setattr__(self, "weights", weights)
        n_periods, n_assets = weights.shape
        if self.timestamps and len(self.timestamps) != n_periods:
            raise ValueError("timestamps length does not match T")
        if self.asset_ids and len(self.asset_ids) != n_assets:
            raise ValueError("asset_ids length does not match N")


@dataclass(slots=True, frozen=True)
class StochasticDiscountFactorState:
    """Structural state extracted from a stochastic discount factor model."""

    asset_weights: Array2D
    sdf_values: FloatArray1D | None = None
    checkpoint_epoch: int | None = None
    timestamps: tuple[Any, ...] = ()
    asset_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        asset_weights = _coerce_float_array(self.asset_weights, ndim=2, name="asset_weights")
        sdf_values = _coerce_float_array(self.sdf_values, ndim=1, name="sdf_values")
        assert asset_weights is not None
        object.__setattr__(self, "asset_weights", asset_weights)
        object.__setattr__(self, "sdf_values", sdf_values)
        n_periods, n_assets = asset_weights.shape
        if sdf_values is not None and sdf_values.shape != (n_periods,):
            raise ValueError("sdf_values must have shape (T,)")
        if self.timestamps and len(self.timestamps) != n_periods:
            raise ValueError("timestamps length does not match T")
        if self.asset_ids and len(self.asset_ids) != n_assets:
            raise ValueError("asset_ids length does not match N")

    @property
    def n_periods(self) -> int:
        return self.asset_weights.shape[0]

    @property
    def n_assets(self) -> int:
        return self.asset_weights.shape[1]


@dataclass(slots=True, frozen=True)
class PortfolioWeightsResult:
    """Portfolio-weight output for end-to-end allocators."""

    weights: Array3D
    checkpoint_step: int | None = None
    timestamps: tuple[Any, ...] = ()
    asset_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        weights = _coerce_float_array(self.weights, ndim=3, name="weights")
        assert weights is not None
        object.__setattr__(self, "weights", weights)
        _, n_periods, n_assets = weights.shape
        if self.timestamps and len(self.timestamps) != n_periods:
            raise ValueError("timestamps length does not match T")
        if self.asset_ids and len(self.asset_ids) != n_assets:
            raise ValueError("asset_ids length does not match N")


@dataclass(slots=True, frozen=True)
class LatentFactorPrediction:
    """Full prediction bundle from a latent-factor pipeline."""

    state: LatentFactorState
    factor_forecast: FactorForecastResult
    asset_forecast: AssetForecastResult


@dataclass(slots=True, frozen=True)
class PortfolioPrediction:
    """Full prediction bundle from a portfolio-allocation pipeline."""

    raw_weights: PortfolioWeightsResult
    processed_weights: PortfolioWeightsResult
