# Integration

`ml4t-models` integrates with the rest of the ML4T stack at boundaries. It does not try to absorb execution or evaluation logic.

## Boundary Design

### This Library Owns

- model estimation
- typed input contracts
- typed result objects
- prediction and weight surfaces

### This Library Does Not Own

- execution simulation
- portfolio diagnostics
- statistical validation reports

Those belong in:

- `ml4t-backtest`
- `ml4t-diagnostic`

## Long-Frame To Batch Adapters

Use:

- `persistent_panel_batch_from_long_frame`
- `cross_section_batch_from_long_frame`
- `resolve_dataset_schema`

These help when your data starts as:

- a pandas frame
- a polars frame
- a long-format table with ML4T-style schema metadata

## Surface Adapters

The surface helpers normalize model outputs into standard long-format tables.

### Prediction Surfaces

- `prediction_surface_from_asset_forecast`
- `prediction_surface_from_asset_signal`

Output columns:

- `timestamp`
- `asset`
- `prediction_value`

### Weight And Signal Surfaces

- `signal_surface_from_portfolio_weights`
- `signal_surface_from_asset_weights`
- `weight_surface_from_portfolio_weights`
- `weight_surface_from_asset_weights`
- `context_surface_from_weights`

## Backtest Handoff

Use:

- `backtest_datafeed_inputs`
- `backtest_inputs_from_asset_forecast`
- `backtest_inputs_from_asset_signal`
- `backtest_inputs_from_weights`

These construct:

- standardized signal surfaces
- optional context surfaces
- `FeedSpec`-compatible metadata for `ml4t-backtest`

## Artifact Writing

Use:

```python
from ml4t.models import write_backtest_surfaces
```

to emit:

- `predictions.parquet`
- `weights.parquet`

in the artifact conventions expected downstream.

## Example

```python
from ml4t.models import (
    backtest_inputs_from_asset_forecast,
    prediction_surface_from_asset_forecast,
    write_backtest_surfaces,
)

surface = prediction_surface_from_asset_forecast(asset_forecast)
write_backtest_surfaces("artifacts/run_001", predictions=surface)

inputs = backtest_inputs_from_asset_forecast(
    asset_forecast,
    prices_path="prices.parquet",
    timestamp_col="timestamp",
    entity_col="asset",
    close_col="close",
)
```

## Rule Of Thumb

If you find yourself computing:

- IC summaries
- tearsheets
- execution PnL
- trade analytics

inside `ml4t-models`, you are probably crossing the intended library boundary.

