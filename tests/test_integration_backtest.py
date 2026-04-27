from __future__ import annotations

from pathlib import Path

import numpy as np

from ml4t.models import (
    AssetForecastResult,
    BacktestDataFeedInputs,
    PortfolioWeightsResult,
    backtest_datafeed_inputs,
    prediction_surface_from_asset_forecast,
    resolve_feed_spec_mapping,
    weight_surface_from_portfolio_weights,
)


def test_resolve_feed_spec_mapping_uses_nested_schema_metadata() -> None:
    frame = {
        "date": np.array(["2024-01-01", "2024-01-01"], dtype=object),
        "ticker": np.array(["AAPL", "MSFT"], dtype=object),
        "settle": np.array([100.0, 200.0], dtype=np.float64),
    }

    feed_spec = resolve_feed_spec_mapping(
        frame,
        schema={
            "schema": {
                "timestamp_col": "date",
                "entity_col": "ticker",
                "close_col": "settle",
            },
            "semantics": {
                "calendar": "NYSE",
                "timezone": "America/New_York",
                "data_frequency": "daily",
            },
        },
    )

    assert feed_spec["timestamp_col"] == "date"
    assert feed_spec["entity_col"] == "ticker"
    assert feed_spec["close_col"] == "settle"
    assert feed_spec["price_col"] == "settle"
    assert feed_spec["calendar"] == "NYSE"
    assert feed_spec["timezone"] == "America/New_York"


def test_backtest_datafeed_inputs_exports_datafeed_kwargs(
    monkeypatch,
) -> None:
    prices_frame = object()
    predictions = prediction_surface_from_asset_forecast(
        AssetForecastResult(
            expected_returns=np.array([[0.1]], dtype=np.float64),
            timestamps=("2024-01-01",),
            asset_ids=("AAPL",),
        )
    )
    weights = weight_surface_from_portfolio_weights(
        PortfolioWeightsResult(
            weights=np.array([[[0.25]]], dtype=np.float64),
            timestamps=("2024-01-01",),
            asset_ids=("AAPL",),
        )
    )
    converted_frames: list[str] = []

    def _to_polars_predictions(self) -> str:
        converted_frames.append(self.metadata["surface_type"])
        return f"{self.metadata['surface_type']}_df"

    monkeypatch.setattr(
        "ml4t.models.integration.surfaces.SurfaceFrame.to_polars",
        _to_polars_predictions,
    )

    inputs = backtest_datafeed_inputs(
        prices_frame=prices_frame,
        signals=predictions,
        context=weights,
        schema={"timestamp_col": "timestamp", "entity_col": "asset", "close_col": "close"},
    )

    kwargs = inputs.to_datafeed_kwargs()

    assert kwargs["prices_df"] is prices_frame
    assert kwargs["signals_df"] == "prediction_df"
    assert kwargs["context_df"] == "weight_df"
    assert kwargs["feed_spec"]["timestamp_col"] == "timestamp"
    assert kwargs["feed_spec"]["entity_col"] == "asset"
    assert inputs.metadata["signal_surface_type"] == "prediction"
    assert converted_frames == ["prediction", "weight"]


def test_backtest_datafeed_inputs_supports_prices_path_only() -> None:
    inputs = backtest_datafeed_inputs(
        prices_path=Path("/tmp/prices.parquet"),
        signals=None,
        schema={"schema": {"timestamp_col": "date", "entity_col": "symbol"}},
        close_col="settle",
    )

    kwargs = inputs.to_datafeed_kwargs()

    assert kwargs["prices_path"] == "/tmp/prices.parquet"
    assert kwargs["feed_spec"]["timestamp_col"] == "date"
    assert kwargs["feed_spec"]["entity_col"] == "symbol"
    assert kwargs["feed_spec"]["close_col"] == "settle"
    assert kwargs["feed_spec"]["price_col"] == "settle"


def test_backtest_datafeed_inputs_require_single_price_source() -> None:
    try:
        BacktestDataFeedInputs(feed_spec={"timestamp_col": "timestamp"})
    except ValueError as exc:
        assert "prices_frame or prices_path" in str(exc)
    else:
        raise AssertionError("expected ValueError when no price source is provided")
