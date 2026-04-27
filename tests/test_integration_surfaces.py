from __future__ import annotations

import numpy as np
import pytest

from ml4t.models import (
    AssetForecastResult,
    AssetWeightsResult,
    PortfolioWeightsResult,
    context_surface_from_weights,
    prediction_surface_from_asset_forecast,
    signal_surface_from_asset_weights,
    signal_surface_from_portfolio_weights,
    weight_surface_from_asset_weights,
    weight_surface_from_portfolio_weights,
    write_backtest_surfaces,
)


def test_prediction_surface_uses_diagnostic_column_names() -> None:
    forecast = AssetForecastResult(
        expected_returns=np.array([[0.1, np.nan], [0.2, 0.3]], dtype=np.float64),
        timestamps=("2024-01-01", "2024-01-02"),
        asset_ids=("AAPL", "MSFT"),
        metadata={"model_name": "ipca"},
    )

    surface = prediction_surface_from_asset_forecast(
        forecast,
        constants={"config_name": "baseline"},
    )

    assert surface.columns == ("timestamp", "asset", "prediction_value", "config_name")
    assert surface.metadata["surface_type"] == "prediction"
    assert surface.to_dicts()[0]["asset"] == "AAPL"
    assert surface.to_dicts()[0]["config_name"] == "baseline"
    assert len(surface.rows) == 3


def test_signal_surface_uses_signal_value_and_selected() -> None:
    weights = PortfolioWeightsResult(
        weights=np.array([[[0.1, 0.0], [-0.2, 0.3]]], dtype=np.float64),
        checkpoint_step=5,
        timestamps=("2024-01-01", "2024-01-02"),
        asset_ids=("AAPL", "MSFT"),
    )

    surface = signal_surface_from_portfolio_weights(weights)

    assert surface.columns == ("timestamp", "asset", "signal_value", "selected")
    rows = surface.to_dicts()
    assert rows[0]["signal_value"] == 0.1
    assert rows[1]["selected"] is False
    assert surface.metadata["checkpoint_step"] == 5


def test_weight_surface_adds_batch_id_for_multi_batch_outputs() -> None:
    weights = PortfolioWeightsResult(
        weights=np.array(
            [
                [[0.1, -0.1]],
                [[0.2, -0.2]],
            ],
            dtype=np.float64,
        ),
        timestamps=("2024-01-01",),
        asset_ids=("AAPL", "MSFT"),
    )

    surface = weight_surface_from_portfolio_weights(weights, constants={"run_id": "r1"})

    assert surface.columns == ("timestamp", "asset", "batch_id", "weight", "selected", "run_id")
    rows = surface.to_dicts()
    assert rows[0]["batch_id"] == 0
    assert rows[-1]["batch_id"] == 1
    assert rows[-1]["run_id"] == "r1"


def test_asset_weight_surfaces_support_sdf_style_outputs() -> None:
    weights = AssetWeightsResult(
        weights=np.array([[0.4, -0.1], [0.0, 0.2]], dtype=np.float64),
        timestamps=("2024-01-01", "2024-01-02"),
        asset_ids=("AAPL", "MSFT"),
    )

    signal_surface = signal_surface_from_asset_weights(weights)
    weight_surface = weight_surface_from_asset_weights(weights)

    assert signal_surface.columns == ("timestamp", "asset", "signal_value", "selected")
    assert weight_surface.columns == ("timestamp", "asset", "weight", "selected")
    assert signal_surface.to_dicts()[1]["selected"] is True
    assert weight_surface.to_dicts()[2]["selected"] is False


def test_context_surface_from_weights_builds_wide_context_frame() -> None:
    weights = AssetWeightsResult(
        weights=np.array([[0.4, -0.1], [0.0, 0.2]], dtype=np.float64),
        timestamps=("2024-01-01", "2024-01-02"),
        asset_ids=("AAPL", "MSFT"),
        metadata={"family": "sdf"},
    )

    surface = context_surface_from_weights(weights, prefix="tw_", constants={"run_id": "r1"})

    assert surface.columns == ("timestamp", "tw_AAPL", "tw_MSFT", "run_id")
    rows = surface.to_dicts()
    assert rows[0]["tw_AAPL"] == 0.4
    assert rows[1]["tw_AAPL"] == 0.0
    assert rows[1]["run_id"] == "r1"
    assert surface.metadata["surface_type"] == "context"


def test_surface_to_polars_requires_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    surface = weight_surface_from_portfolio_weights(
        PortfolioWeightsResult(weights=np.array([[[0.1]]], dtype=np.float64))
    )

    def _raise_import_error(name: str) -> None:
        raise ImportError(name)

    monkeypatch.setattr("ml4t.models.integration.surfaces.import_module", _raise_import_error)
    with pytest.raises(ImportError, match="ml4t-models\\[integration\\]"):
        surface.to_polars()


def test_write_backtest_surfaces_uses_standard_artifact_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    prediction_surface = prediction_surface_from_asset_forecast(
        AssetForecastResult(expected_returns=np.array([[0.1]], dtype=np.float64))
    )
    weight_surface = weight_surface_from_portfolio_weights(
        PortfolioWeightsResult(weights=np.array([[[0.2]]], dtype=np.float64))
    )
    written_paths: list[str] = []

    def _write_parquet(self, path, *, compression="zstd"):
        written_paths.append(f"{Path(path).name}:{compression}")
        return Path(path)

    from pathlib import Path

    monkeypatch.setattr(
        "ml4t.models.integration.surfaces.SurfaceFrame.write_parquet",
        _write_parquet,
    )
    written = write_backtest_surfaces(
        tmp_path,
        predictions=prediction_surface,
        weights=weight_surface,
    )

    assert set(written) == {"predictions", "weights"}
    assert "predictions.parquet:zstd" in written_paths
    assert "weights.parquet:zstd" in written_paths
