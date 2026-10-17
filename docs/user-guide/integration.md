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

### Calendar declarations

All handoff helpers use the keyword `calendar`. Omitting it preserves the calendar
declared by the input `FeedSpec` or schema. The consuming feed resolves an omitted
declaration to NYSE. Passing a calendar name overrides that metadata; passing
`calendar=None` explicitly disables calendar handling. Calendar-free operation uses
ordered observations and is different from a 24/7 exchange schedule.

An explicit selection produces `time_contract_version=2` metadata so a disabled
calendar survives JSON serialization and feed construction. Specs 0.2.0 and a
Backtest version supporting that temporal contract are required to consume it.
Version-2 `FeedSpec` metadata also preserves omitted versus null calendars and bar-grid
declarations. Historical version-1 payloads, including unversioned `calendar: null`,
retain their null-as-inheritance behavior; pass `calendar=None` to disable calendars
when adapting such a payload.

```python
inputs = backtest_inputs_from_asset_forecast(
    asset_forecast,
    prices_path="prices.parquet",
    schema={"calendar": "CME_Equity"},
    calendar=None,
)
assert inputs.feed_spec["calendar"] is None
assert inputs.feed_spec["time_contract_version"] == 2
```

The adapter preserves timestamp values and their declared semantics; it does not assign
exchange sessions or resample returns. Timestamp-free estimators retain their observation-based
behavior. Portfolio models use `annualization_factor` as a separate explicit scale, with
a default of 252 observations per year; selecting or disabling a feed calendar does not
change that training convention.

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
