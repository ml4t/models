# Integration

`ml4t-models` integrates with the rest of the ML4T stack at boundaries. It does not try to absorb execution or evaluation logic.

## Boundary Design

### This Library Owns

- model estimation
- typed input contracts
- typed result objects
- prediction and weight frames

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

If `ml4t-specs` is installed, the adapters accept `FeedSpec` objects directly. Without that
optional dependency, they still accept explicit column names and plain metadata mappings.
The library does not require users to source data through `ml4t-data`.

## Frame Adapters

The frame helpers normalize model outputs into standard long-format tables.

### Predictions Frames

- `predictions_frame_from_asset_forecast`
- `predictions_frame_from_asset_signal`

Output columns:

- `timestamp`
- `asset`
- `prediction_value`

### Weight And Signal Frames

- `signals_frame_from_portfolio_weights`
- `signals_frame_from_asset_weights`
- `weights_frame_from_portfolio_weights`
- `weights_frame_from_asset_weights`
- `context_frame_from_weights`

## Backtest Handoff

Use:

- `backtest_datafeed_inputs`
- `backtest_inputs_from_asset_forecast`
- `backtest_inputs_from_asset_signal`
- `backtest_inputs_from_weights`

These construct:

- standardized signal frames
- optional context frames
- `FeedSpec`-compatible metadata for `ml4t-backtest`

The handoff payload is intentionally shallow: it prepares frames and metadata, then lets
`ml4t-backtest` own execution simulation.

## Diagnostics Handoff

`ml4t-models` emits prediction, signal, and weight frames with standard timestamp and asset
columns. Use those frames as inputs to `ml4t-diagnostic` for cross-sectional IC, portfolio
diagnostics, tearsheets, and validation reports.

This library does not compute IC summaries or diagnostics itself. Keeping diagnostics in
`ml4t-diagnostic` prevents model implementations from carrying a second evaluation stack.

## Artifact Writing

Use:

```python
from ml4t.models import write_backtest_frames
```

to emit:

- `predictions.parquet`
- `weights.parquet`

in the artifact conventions expected downstream.

## Example

```python
from ml4t.models import (
    backtest_inputs_from_asset_forecast,
    predictions_frame_from_asset_forecast,
    write_backtest_frames,
)

frame = predictions_frame_from_asset_forecast(asset_forecast)
write_backtest_frames("artifacts/run_001", predictions=frame)

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
