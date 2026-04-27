"""Standardized prediction and signal surfaces for downstream ML4T libraries."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from ml4t.models.types import AssetForecastResult, PortfolioWeightsResult


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
