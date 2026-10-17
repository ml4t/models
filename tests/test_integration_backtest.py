from __future__ import annotations

import json
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

    assert Path(kwargs["prices_path"]) == Path("/tmp/prices.parquet")
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


@pytest.fixture(params=["forecast", "signal", "asset_weights", "portfolio_weights"])
def calendar_handoff_result(
    request: pytest.FixtureRequest,
) -> AssetForecastResult | AssetSignalResult | AssetWeightsResult | PortfolioWeightsResult:
    identity = {"timestamps": ("2025-07-07",), "asset_ids": ("A", "B")}
    if request.param == "forecast":
        return AssetForecastResult(expected_returns=np.array([[0.01, -0.01]]), **identity)
    if request.param == "signal":
        return AssetSignalResult(signal_values=np.array([[1.0, -1.0]]), **identity)
    if request.param == "asset_weights":
        return AssetWeightsResult(weights=np.array([[0.5, -0.5]]), **identity)
    return PortfolioWeightsResult(weights=np.array([[[0.5, -0.5]]]), **identity)


def _calendar_handoff(
    result: AssetForecastResult | AssetSignalResult | AssetWeightsResult | PortfolioWeightsResult,
    **kwargs,
) -> BacktestDataFeedInputs:
    if isinstance(result, AssetForecastResult):
        return backtest_inputs_from_asset_forecast(result, prices_path="prices.parquet", **kwargs)
    if isinstance(result, AssetSignalResult):
        return backtest_inputs_from_asset_signal(result, prices_path="prices.parquet", **kwargs)
    return backtest_inputs_from_weights(result, prices_path="prices.parquet", **kwargs)


@pytest.mark.parametrize(
    ("selection", "expected"),
    [({}, "CME_Equity"), ({"calendar": None}, None), ({"calendar": "NYSE"}, "NYSE")],
)
def test_model_handoffs_preserve_calendar_override_through_feed_serialization(
    calendar_handoff_result,
    selection: dict,
    expected: str | None,
) -> None:
    schema = {"calendar": "CME_Equity"}
    inputs = _calendar_handoff(calendar_handoff_result, schema=schema, **selection)

    payload = json.loads(json.dumps(inputs.feed_spec))
    assert payload["calendar"] == expected
    assert specs.FeedSpec.from_mapping(payload).resolve_calendar() == expected
    assert schema == {"calendar": "CME_Equity"}


@pytest.mark.parametrize("as_context", [False, True])
def test_model_weight_context_preserves_explicit_calendar_disable(as_context: bool) -> None:
    result = AssetWeightsResult(
        weights=np.array([[0.5, -0.5]]), timestamps=("2025-07-07",), asset_ids=("A", "B")
    )
    inputs = backtest_inputs_from_weights(
        result,
        prices_path="prices.parquet",
        calendar=None,
        schema={"calendar": "CME_Equity"},
        as_context=as_context,
    )

    payload = inputs.to_datafeed_kwargs()["feed_spec"]
    assert payload["calendar"] is None
    assert payload["time_contract_version"] == 2
    assert specs.FeedSpec.from_mapping(payload).resolve_calendar() is None


@pytest.mark.parametrize("declared", [False, True])
def test_model_handoffs_preserve_versioned_calendar_presence(
    calendar_handoff_result, declared
) -> None:
    schema = specs.FeedSpec(time_contract_version=2, **({"calendar": None} if declared else {}))
    before = schema.to_dict()
    inputs = _calendar_handoff(calendar_handoff_result, schema=schema)

    payload = json.loads(json.dumps(inputs.feed_spec))
    assert ("calendar" in payload) is declared
    assert specs.FeedSpec.from_mapping(payload).resolve_calendar() == (None if declared else "NYSE")
    assert schema.to_dict() == before


def test_model_handoff_retains_legacy_null_inheritance_and_explicit_override() -> None:
    legacy = {"calendar": None}
    inherited = backtest_datafeed_inputs(prices_path="prices.parquet", schema=legacy)
    disabled = backtest_datafeed_inputs(prices_path="prices.parquet", schema=legacy, calendar=None)

    assert specs.FeedSpec.from_mapping(inherited.feed_spec).resolve_calendar() == "NYSE"
    assert specs.FeedSpec.from_mapping(disabled.feed_spec).resolve_calendar() is None
    assert legacy == {"calendar": None}


@pytest.mark.parametrize("calendar", ["", " ", False, 42])
def test_invalid_calendar_override_does_not_mutate_model_feed_metadata(calendar) -> None:
    schema = {"calendar": "CME_Equity"}
    with pytest.raises(ValueError, match="calendar"):
        backtest_datafeed_inputs(prices_path="prices.parquet", schema=schema, calendar=calendar)
    assert schema == {"calendar": "CME_Equity"}


def test_model_handoff_preserves_bar_grid_without_specs_dependency(monkeypatch) -> None:
    monkeypatch.setattr(backtest_module, "_load_feed_spec_class", lambda: None)
    schema = {
        "semantics": {
            "time_contract_version": 2,
            "calendar": None,
            "bar_grid_anchor": "session_open",
            "bar_label": "right",
            "partial_bar_policy": "include",
            "timestamp_semantics": "bar_close",
        }
    }
    inputs = backtest_datafeed_inputs(prices_path="prices.parquet", schema=schema)

    payload = json.loads(json.dumps(inputs.feed_spec))
    for key, value in schema["semantics"].items():
        assert payload[key] == value
    assert specs.FeedSpec.from_mapping(payload).resolve_calendar() is None


@pytest.mark.parametrize("annualization", [252.0, 365.0])
def test_calendar_free_model_handoff_preserves_observation_scale_across_timestamp_gaps(
    annualization: float,
) -> None:
    from ml4t.models import LSTMPortfolioConfig, LSTMPortfolioModel, PortfolioSequenceBatch

    rng = np.random.default_rng(103)
    features = rng.normal(size=(1, 3, 2, 2))
    returns = 0.01 * rng.normal(size=(1, 3, 2))
    date_grids = (
        ("2025-07-01", "2025-07-02", "2025-07-03"),
        ("2025-07-03", "2025-07-07", "2025-07-14"),
        ("2025-01-01", "2025-12-31", "2027-03-08"),
    )
    first_weights = None
    for timestamps in date_grids:
        batch = PortfolioSequenceBatch(
            features=features,
            returns=returns,
            timestamps=timestamps,
            asset_ids=("A", "B"),
        )
        config = LSTMPortfolioConfig(
            hidden_size=4,
            dropout=0.0,
            max_iters=1,
            eval_every=1,
            checkpoint_every=1,
            default_checkpoint=1,
            batch_size=2,
            annualization_factor=annualization,
            seed=17,
            device="cpu",
            dtype="float64",
        )
        model = LSTMPortfolioModel(config)
        fitted = model.fit(batch)
        prediction = model.predict(batch)

        realized = (prediction.weights * returns).mean(axis=-1)
        expected_sharpe = (
            np.sqrt(annualization) * realized.mean() / np.sqrt(realized.var() + config.sharpe_eps)
        )
        assert fitted.val_metrics["best_validation_sharpe"] == pytest.approx(expected_sharpe)
        if first_weights is None:
            first_weights = prediction.weights.copy()
        else:
            np.testing.assert_allclose(prediction.weights, first_weights, rtol=0, atol=1e-12)
        assert prediction.timestamps == timestamps

        handoff = backtest_inputs_from_weights(
            prediction, prices_path="prices.parquet", calendar=None
        )
        assert handoff.feed_spec["calendar"] is None
        assert handoff.signals is not None
        assert handoff.signals.to_dicts()[0]["timestamp"] == timestamps[0]
