"""Integration helpers for cross-library data contracts."""

from ml4t.models.integration.backtest import (
    BacktestDataFeedInputs,
    backtest_datafeed_inputs,
    backtest_inputs_from_asset_forecast,
    backtest_inputs_from_asset_signal,
    backtest_inputs_from_weights,
    resolve_feed_spec_mapping,
)
from ml4t.models.integration.data import (
    ResolvedDatasetSchema,
    cross_section_batch_from_long_frame,
    persistent_panel_batch_from_long_frame,
    resolve_dataset_schema,
)
from ml4t.models.integration.surfaces import (
    SurfaceFrame,
    context_surface_from_weights,
    prediction_surface_from_asset_forecast,
    prediction_surface_from_asset_signal,
    signal_surface_from_asset_weights,
    signal_surface_from_portfolio_weights,
    weight_surface_from_asset_weights,
    weight_surface_from_portfolio_weights,
    write_backtest_surfaces,
)

__all__ = [
    "BacktestDataFeedInputs",
    "ResolvedDatasetSchema",
    "SurfaceFrame",
    "backtest_datafeed_inputs",
    "backtest_inputs_from_asset_forecast",
    "backtest_inputs_from_asset_signal",
    "backtest_inputs_from_weights",
    "context_surface_from_weights",
    "cross_section_batch_from_long_frame",
    "prediction_surface_from_asset_forecast",
    "prediction_surface_from_asset_signal",
    "persistent_panel_batch_from_long_frame",
    "resolve_feed_spec_mapping",
    "resolve_dataset_schema",
    "signal_surface_from_asset_weights",
    "signal_surface_from_portfolio_weights",
    "weight_surface_from_asset_weights",
    "weight_surface_from_portfolio_weights",
    "write_backtest_surfaces",
]
