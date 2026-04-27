"""Standardized prediction and signal surfaces for downstream ML4T libraries."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from ml4t.models.types import (
    AssetForecastResult,
    AssetSignalResult,
    AssetWeightsResult,
    PortfolioWeightsResult,
)


@dataclass(frozen=True, slots=True)
class SurfaceFrame:
    """Long-format tabular surface with optional export helpers."""

    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dicts(self) -> list[dict[str, Any]]:
        """Return the surface as a list of row dictionaries."""

        return [dict(zip(self.columns, row, strict=True)) for row in self.rows]

    def to_columnar(self) -> dict[str, list[Any]]:
        """Return the surface as columnar Python lists."""

        data = {column: [] for column in self.columns}
        for row in self.rows:
            for column, value in zip(self.columns, row, strict=True):
                data[column].append(value)
        return data

    def to_polars(self) -> Any:
        """Return the surface as a Polars DataFrame when Polars is installed."""

        pl = _import_polars()
        return pl.DataFrame(self.to_dicts())

    def write_parquet(self, path: str | Path, *, compression: str = "zstd") -> Path:
        """Write the surface to parquet when Polars is installed."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_polars().write_parquet(output_path, compression=compression)
        return output_path


def prediction_surface_from_asset_forecast(
    forecast: AssetForecastResult,
    *,
    constants: dict[str, Any] | None = None,
) -> SurfaceFrame:
    """Convert asset expected returns to a diagnostic-ready prediction surface."""

    expected_returns = np.asarray(forecast.expected_returns, dtype=np.float64)
    timestamps = _resolve_timestamps(expected_returns.shape[0], forecast.timestamps)
    assets = _resolve_assets(expected_returns.shape[1], forecast.asset_ids)
    constant_columns = tuple((constants or {}).keys())
    rows: list[tuple[Any, ...]] = []

    for t_idx, timestamp in enumerate(timestamps):
        for a_idx, asset in enumerate(assets):
            value = expected_returns[t_idx, a_idx]
            if not np.isfinite(value):
                continue
            rows.append(
                (
                    timestamp,
                    asset,
                    float(value),
                    *tuple((constants or {}).values()),
                )
            )

    return SurfaceFrame(
        columns=("timestamp", "asset", "prediction_value", *constant_columns),
        rows=tuple(rows),
        metadata={"surface_type": "prediction", **forecast.metadata},
    )


def prediction_surface_from_asset_signal(
    signal: AssetSignalResult,
    *,
    constants: dict[str, Any] | None = None,
) -> SurfaceFrame:
    """Convert asset-level signals to a diagnostic-ready prediction surface."""

    signal_values = np.asarray(signal.signal_values, dtype=np.float64)
    timestamps = _resolve_timestamps(signal_values.shape[0], signal.timestamps)
    assets = _resolve_assets(signal_values.shape[1], signal.asset_ids)
    constant_columns = tuple((constants or {}).keys())
    rows: list[tuple[Any, ...]] = []

    for t_idx, timestamp in enumerate(timestamps):
        for a_idx, asset in enumerate(assets):
            value = signal_values[t_idx, a_idx]
            if not np.isfinite(value):
                continue
            rows.append(
                (
                    timestamp,
                    asset,
                    float(value),
                    *tuple((constants or {}).values()),
                )
            )

    return SurfaceFrame(
        columns=("timestamp", "asset", "prediction_value", *constant_columns),
        rows=tuple(rows),
        metadata={"surface_type": "prediction", **signal.metadata},
    )


def signal_surface_from_portfolio_weights(
    weights: PortfolioWeightsResult,
    *,
    constants: dict[str, Any] | None = None,
    selected_threshold: float = 1e-9,
) -> SurfaceFrame:
    """Convert portfolio weights to a diagnostic-ready signal surface."""

    return _surface_from_portfolio_weights(
        weights,
        value_column="signal_value",
        include_selected=True,
        selected_threshold=selected_threshold,
        constants=constants,
        surface_type="signal",
    )


def signal_surface_from_asset_weights(
    weights: AssetWeightsResult,
    *,
    constants: dict[str, Any] | None = None,
    selected_threshold: float = 1e-9,
) -> SurfaceFrame:
    """Convert cross-sectional asset weights to a diagnostic-ready signal surface."""

    return _surface_from_asset_weights(
        weights,
        value_column="signal_value",
        include_selected=True,
        selected_threshold=selected_threshold,
        constants=constants,
        surface_type="signal",
    )


def weight_surface_from_portfolio_weights(
    weights: PortfolioWeightsResult,
    *,
    constants: dict[str, Any] | None = None,
    selected_threshold: float = 1e-9,
) -> SurfaceFrame:
    """Convert portfolio weights to a backtest-ready target-weight surface."""

    return _surface_from_portfolio_weights(
        weights,
        value_column="weight",
        include_selected=True,
        selected_threshold=selected_threshold,
        constants=constants,
        surface_type="weight",
    )


def weight_surface_from_asset_weights(
    weights: AssetWeightsResult,
    *,
    constants: dict[str, Any] | None = None,
    selected_threshold: float = 1e-9,
) -> SurfaceFrame:
    """Convert cross-sectional asset weights to a backtest-ready target-weight surface."""

    return _surface_from_asset_weights(
        weights,
        value_column="weight",
        include_selected=True,
        selected_threshold=selected_threshold,
        constants=constants,
        surface_type="weight",
    )


def context_surface_from_weights(
    weights: AssetWeightsResult | PortfolioWeightsResult,
    *,
    prefix: str = "w_",
    constants: dict[str, Any] | None = None,
) -> SurfaceFrame:
    """Convert asset weights to a wide context frame for backtest strategies."""

    weight_matrix, timestamps, assets = _resolve_weight_matrix(weights)
    constant_columns = tuple((constants or {}).keys())
    columns = ("timestamp", *(f"{prefix}{asset}" for asset in assets), *constant_columns)
    rows: list[tuple[Any, ...]] = []

    for t_idx, timestamp in enumerate(timestamps):
        values = [
            float(weight_matrix[t_idx, a_idx]) if np.isfinite(weight_matrix[t_idx, a_idx]) else 0.0
            for a_idx in range(len(assets))
        ]
        rows.append((timestamp, *values, *tuple((constants or {}).values())))

    metadata = {"surface_type": "context", **weights.metadata}
    if isinstance(weights, PortfolioWeightsResult) and weights.checkpoint_step is not None:
        metadata["checkpoint_step"] = weights.checkpoint_step

    return SurfaceFrame(columns=columns, rows=tuple(rows), metadata=metadata)


def write_backtest_surfaces(
    artifact_dir: str | Path,
    *,
    predictions: SurfaceFrame | None = None,
    weights: SurfaceFrame | None = None,
    compression: str = "zstd",
) -> dict[str, Path]:
    """Write standardized prediction and weight artifacts for downstream ML4T tooling."""

    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    if predictions is not None:
        written["predictions"] = predictions.write_parquet(
            output_dir / "predictions.parquet",
            compression=compression,
        )
    if weights is not None:
        written["weights"] = weights.write_parquet(
            output_dir / "weights.parquet",
            compression=compression,
        )
    return written


def _surface_from_portfolio_weights(
    weights: PortfolioWeightsResult,
    *,
    value_column: str,
    include_selected: bool,
    selected_threshold: float,
    constants: dict[str, Any] | None,
    surface_type: str,
) -> SurfaceFrame:
    weight_array = np.asarray(weights.weights, dtype=np.float64)
    batch_size, n_periods, n_assets = weight_array.shape
    timestamps = _resolve_timestamps(n_periods, weights.timestamps)
    assets = _resolve_assets(n_assets, weights.asset_ids)
    constant_columns = tuple((constants or {}).keys())

    columns = ["timestamp", "asset"]
    if batch_size > 1:
        columns.append("batch_id")
    columns.append(value_column)
    if include_selected:
        columns.append("selected")
    columns.extend(constant_columns)

    rows: list[tuple[Any, ...]] = []
    for batch_idx in range(batch_size):
        for t_idx, timestamp in enumerate(timestamps):
            for a_idx, asset in enumerate(assets):
                value = weight_array[batch_idx, t_idx, a_idx]
                if not np.isfinite(value):
                    continue
                row: list[Any] = [timestamp, asset]
                if batch_size > 1:
                    row.append(batch_idx)
                row.append(float(value))
                if include_selected:
                    row.append(bool(abs(value) > selected_threshold))
                row.extend((constants or {}).values())
                rows.append(tuple(row))

    metadata = {
        "surface_type": surface_type,
        **weights.metadata,
    }
    if weights.checkpoint_step is not None:
        metadata["checkpoint_step"] = weights.checkpoint_step

    return SurfaceFrame(
        columns=tuple(columns),
        rows=tuple(rows),
        metadata=metadata,
    )


def _surface_from_asset_weights(
    weights: AssetWeightsResult,
    *,
    value_column: str,
    include_selected: bool,
    selected_threshold: float,
    constants: dict[str, Any] | None,
    surface_type: str,
) -> SurfaceFrame:
    weight_array = np.asarray(weights.weights, dtype=np.float64)
    n_periods, n_assets = weight_array.shape
    timestamps = _resolve_timestamps(n_periods, weights.timestamps)
    assets = _resolve_assets(n_assets, weights.asset_ids)
    constant_columns = tuple((constants or {}).keys())

    columns = ["timestamp", "asset", value_column]
    if include_selected:
        columns.append("selected")
    columns.extend(constant_columns)

    rows: list[tuple[Any, ...]] = []
    for t_idx, timestamp in enumerate(timestamps):
        for a_idx, asset in enumerate(assets):
            value = weight_array[t_idx, a_idx]
            if not np.isfinite(value):
                continue
            row: list[Any] = [timestamp, asset, float(value)]
            if include_selected:
                row.append(bool(abs(value) > selected_threshold))
            row.extend((constants or {}).values())
            rows.append(tuple(row))

    return SurfaceFrame(
        columns=tuple(columns),
        rows=tuple(rows),
        metadata={"surface_type": surface_type, **weights.metadata},
    )


def _resolve_weight_matrix(
    weights: AssetWeightsResult | PortfolioWeightsResult,
) -> tuple[ArrayLike2D, tuple[Any, ...], tuple[str, ...]]:
    if isinstance(weights, AssetWeightsResult):
        weight_matrix = np.asarray(weights.weights, dtype=np.float64)
        timestamps = _resolve_timestamps(weight_matrix.shape[0], weights.timestamps)
        assets = _resolve_assets(weight_matrix.shape[1], weights.asset_ids)
        return weight_matrix, timestamps, assets

    weight_array = np.asarray(weights.weights, dtype=np.float64)
    if weight_array.shape[0] != 1:
        raise ValueError("Wide context export requires a single portfolio-weight batch")
    weight_matrix = weight_array[0]
    timestamps = _resolve_timestamps(weight_matrix.shape[0], weights.timestamps)
    assets = _resolve_assets(weight_matrix.shape[1], weights.asset_ids)
    return weight_matrix, timestamps, assets


ArrayLike2D = np.ndarray


def _resolve_timestamps(n_periods: int, timestamps: tuple[Any, ...]) -> tuple[Any, ...]:
    if timestamps:
        return timestamps
    return tuple(range(n_periods))


def _resolve_assets(n_assets: int, asset_ids: tuple[str, ...]) -> tuple[str, ...]:
    if asset_ids:
        return asset_ids
    return tuple(f"asset_{idx}" for idx in range(n_assets))


def _import_polars() -> Any:
    try:
        return import_module("polars")
    except ImportError as exc:
        raise ImportError(
            "Polars is required for DataFrame/parquet export. Install with: "
            "pip install 'ml4t-models[integration]'"
        ) from exc
