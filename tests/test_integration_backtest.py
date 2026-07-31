from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ml4t.specs import market_data as specs

from ml4t.models import (
    AssetForecastResult,
    AssetSignalResult,
    AssetWeightsResult,
    BacktestDataFeedInputs,
    PortfolioWeightsResult,
    backtest_datafeed_inputs,
    backtest_inputs_from_asset_forecast,
    backtest_inputs_from_asset_signal,
    backtest_inputs_from_weights,
    predictions_frame_from_asset_forecast,
    resolve_feed_spec_mapping,
    weights_frame_from_portfolio_weights,
)
from ml4t.models.integration import backtest as backtest_module


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


def test_resolve_feed_spec_mapping_accepts_ml4t_specs_feedspec() -> None:
    frame = {
        "date": np.array(["2024-01-01", "2024-01-01"], dtype=object),
        "ticker": np.array(["AAPL", "MSFT"], dtype=object),
        "settle": np.array([100.0, 200.0], dtype=np.float64),
    }
    schema = specs.FeedSpec(
        timestamp_col="date",
        entity_col="ticker",
        close_col="settle",
        calendar="NYSE",
        timezone="America/New_York",
    )

    feed_spec = resolve_feed_spec_mapping(frame, schema=schema)

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
    predictions = predictions_frame_from_asset_forecast(
        AssetForecastResult(
            expected_returns=np.array([[0.1]], dtype=np.float64),
            timestamps=("2024-01-01",),
            asset_ids=("AAPL",),
        )
    )
    weights = weights_frame_from_portfolio_weights(
        PortfolioWeightsResult(
            weights=np.array([[[0.25]]], dtype=np.float64),
            timestamps=("2024-01-01",),
            asset_ids=("AAPL",),
        )
    )
    converted_frames: list[str] = []

    def _to_polars_predictions(self) -> str:
        converted_frames.append(self.metadata["frame_type"])
        return f"{self.metadata['frame_type']}_df"

    monkeypatch.setattr(
        "ml4t.models.integration.surfaces.ResultsFrame.to_polars",
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
    assert inputs.metadata["signal_frame_type"] == "prediction"
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


def test_backtest_inputs_from_asset_forecast_builds_predictions_frame(monkeypatch) -> None:
    forecast = AssetForecastResult(
        expected_returns=np.array([[0.1]], dtype=np.float64),
        timestamps=("2024-01-01",),
        asset_ids=("AAPL",),
    )

    monkeypatch.setattr(
        "ml4t.models.integration.surfaces.ResultsFrame.to_polars",
        lambda self: self.to_dicts(),
    )

    inputs = backtest_inputs_from_asset_forecast(
        forecast,
        prices_path=Path("/tmp/prices.parquet"),
        schema={"timestamp_col": "timestamp", "entity_col": "asset"},
    )

    kwargs = inputs.to_datafeed_kwargs()

    assert kwargs["signals_df"][0]["prediction_value"] == 0.1
    assert kwargs["feed_spec"]["entity_col"] == "asset"


def test_backtest_inputs_from_asset_signal_builds_predictions_frame(monkeypatch) -> None:
    signal = AssetSignalResult(
        signal_values=np.array([[0.7]], dtype=np.float64),
        timestamps=("2024-01-01",),
        asset_ids=("AAPL",),
    )

    monkeypatch.setattr(
        "ml4t.models.integration.surfaces.ResultsFrame.to_polars",
        lambda self: self.to_dicts(),
    )

    inputs = backtest_inputs_from_asset_signal(
        signal,
        prices_path=Path("/tmp/prices.parquet"),
        schema={"timestamp_col": "timestamp", "entity_col": "asset"},
    )

    kwargs = inputs.to_datafeed_kwargs()

    assert kwargs["signals_df"][0]["prediction_value"] == 0.7
    assert kwargs["feed_spec"]["entity_col"] == "asset"


def test_backtest_inputs_from_weights_supports_signal_and_context_modes(monkeypatch) -> None:
    weights = AssetWeightsResult(
        weights=np.array([[0.4, -0.1]], dtype=np.float64),
        timestamps=("2024-01-01",),
        asset_ids=("AAPL", "MSFT"),
    )

    monkeypatch.setattr(
        "ml4t.models.integration.surfaces.ResultsFrame.to_polars",
        lambda self: self.to_dicts(),
    )

    signal_inputs = backtest_inputs_from_weights(
        weights,
        prices_path=Path("/tmp/prices.parquet"),
        schema={"timestamp_col": "timestamp", "entity_col": "asset"},
    )
    signal_kwargs = signal_inputs.to_datafeed_kwargs()
    assert signal_kwargs["signals_df"][0]["weight"] == 0.4
    assert "context_df" not in signal_kwargs

    context_inputs = backtest_inputs_from_weights(
        weights,
        prices_path=Path("/tmp/prices.parquet"),
        schema={"timestamp_col": "timestamp", "entity_col": "asset"},
        as_context=True,
    )
    context_kwargs = context_inputs.to_datafeed_kwargs()
    assert "signals_df" not in context_kwargs
    assert context_kwargs["context_df"][0]["w_AAPL"] == 0.4


def test_backtest_datafeed_inputs_require_single_price_source() -> None:
    try:
        BacktestDataFeedInputs(feed_spec={"timestamp_col": "timestamp"})
    except ValueError as exc:
        assert "prices_frame or prices_path" in str(exc)
    else:
        raise AssertionError("expected ValueError when no price source is provided")


def test_backtest_datafeed_inputs_reject_both_price_sources() -> None:
    with pytest.raises(ValueError, match="not both"):
        BacktestDataFeedInputs(
            feed_spec={"timestamp_col": "timestamp"},
            prices_frame=object(),
            prices_path="prices.parquet",
        )
    with pytest.raises(ValueError, match="not both"):
        backtest_datafeed_inputs(prices_frame=object(), prices_path="prices.parquet")


def test_backtest_datafeed_inputs_function_requires_price_source() -> None:
    with pytest.raises(ValueError, match="prices_frame or prices_path"):
        backtest_datafeed_inputs()


def test_resolve_feed_spec_mapping_applies_defaults_and_all_overrides() -> None:
    default = resolve_feed_spec_mapping(object())
    assert default == {
        "timestamp_col": "timestamp",
        "entity_col": "asset",
        "price_col": "close",
        "open_col": "open",
        "high_col": "high",
        "low_col": "low",
        "close_col": "close",
        "volume_col": "volume",
    }

    overridden = resolve_feed_spec_mapping(
        timestamp_col="ts",
        entity_col="symbol",
        price_col="midpoint",
        open_col="o",
        high_col="h",
        low_col="l",
        close_col="c",
        volume_col="v",
        bid_col="bid",
        ask_col="ask",
        mid_col="mid",
        bid_size_col="bid_size",
        ask_size_col="ask_size",
        calendar="XNYS",
        timezone="UTC",
        data_frequency="daily",
        bar_type="time",
        timestamp_semantics="close",
        session_start_time="09:30",
    )
    assert overridden["timestamp_col"] == "ts"
    assert overridden["entity_col"] == "symbol"
    assert overridden["price_col"] == "midpoint"
    assert overridden["session_start_time"] == "09:30"


def test_fallback_feed_spec_supports_flat_object_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backtest_module, "_load_feed_spec_class", lambda: None)

    mapping = resolve_feed_spec_mapping(
        schema=SimpleNamespace(
            datetime_col="event_time",
            symbol_col="symbol",
            ticker_col="ignored_ticker",
            frequency="1d",
            close_col="settle",
        )
    )

    assert mapping["timestamp_col"] == "event_time"
    assert mapping["entity_col"] == "symbol"
    assert mapping["data_frequency"] == "1d"
    assert mapping["price_col"] == "settle"
    assert "datetime_col" not in mapping
    assert "symbol_col" not in mapping
    assert "ticker_col" not in mapping


def test_fallback_feed_spec_supports_nested_sections_and_time_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backtest_module, "_load_feed_spec_class", lambda: None)

    mapping = resolve_feed_spec_mapping(
        schema={
            "schema": {"time_col": "date", "asset_col": "asset_id", "price_col": "px"},
            "semantics": {"calendar": "NYSE", "frequency": "daily"},
        }
    )

    assert mapping["timestamp_col"] == "date"
    assert mapping["entity_col"] == "asset_id"
    assert mapping["price_col"] == "px"
    assert mapping["calendar"] == "NYSE"
    assert mapping["data_frequency"] == "daily"


def test_fallback_feed_spec_handles_empty_nested_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backtest_module, "_load_feed_spec_class", lambda: None)

    mapping = resolve_feed_spec_mapping(schema={"schema": None, "semantics": None})

    assert mapping["timestamp_col"] == "timestamp"
    assert mapping["entity_col"] == "asset"
    assert backtest_module._fallback_feed_spec_mapping(None) == {}
    assert backtest_module._get_nested_field(None, "missing") is backtest_module._MISSING


def test_feed_spec_loader_handles_missing_dependency_and_missing_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_module(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(backtest_module, "import_module", missing_module)
    assert backtest_module._load_feed_spec_class() is None

    monkeypatch.setattr(backtest_module, "import_module", lambda _name: SimpleNamespace())
    assert backtest_module._load_feed_spec_class() is None


def test_backtest_inputs_from_portfolio_weights_uses_portfolio_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = PortfolioWeightsResult(
        weights=np.array([[[0.4, -0.1]]]),
        timestamps=("2024-01-01",),
        asset_ids=("AAPL", "MSFT"),
    )
    monkeypatch.setattr(
        "ml4t.models.integration.surfaces.ResultsFrame.to_polars",
        lambda self: self.to_dicts(),
    )

    inputs = backtest_inputs_from_weights(weights, prices_path="prices.parquet")

    assert inputs.to_datafeed_kwargs()["signals_df"][0]["weight"] == 0.4
