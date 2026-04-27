"""Integration helpers for cross-library data contracts."""

from ml4t.models.integration.data import (
    ResolvedDatasetSchema,
    cross_section_batch_from_long_frame,
    persistent_panel_batch_from_long_frame,
    resolve_dataset_schema,
)
from ml4t.models.integration.surfaces import (
    SurfaceFrame,
    prediction_surface_from_asset_forecast,
    signal_surface_from_portfolio_weights,
    weight_surface_from_portfolio_weights,
    write_backtest_surfaces,
)

__all__ = [
    "ResolvedDatasetSchema",
    "SurfaceFrame",
    "cross_section_batch_from_long_frame",
    "prediction_surface_from_asset_forecast",
    "persistent_panel_batch_from_long_frame",
    "resolve_dataset_schema",
    "signal_surface_from_portfolio_weights",
    "weight_surface_from_portfolio_weights",
    "write_backtest_surfaces",
]
