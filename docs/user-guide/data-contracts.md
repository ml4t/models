# Data Contracts

The library uses three primary batch contracts because the underlying finance problems are not all the same.

## PersistentPanelBatch

Used by:

- `PCAModel`
- `RPPCAModel`

This contract is for stable-ID panels where a global entity axis is meaningful across time.

```python
from ml4t.models import PersistentPanelBatch

batch = PersistentPanelBatch(
    returns=returns_tn,
    characteristics=chars_tnp,
    timestamps=timestamps,
    asset_ids=asset_ids,
)
```

### When To Use It

- equities, ETFs, futures, or curves with persistent entity identity
- static-loading or persistent-loading factor models
- panel-wide PCA-style decompositions

### When Not To Use It

- anonymized datasets with unstable identifiers
- cross-sections whose asset axis is ragged date by date

## CrossSectionBatch

Used by:

- `IPCAModel`
- `CAEModel`
- `SAEModel`
- `StochasticDiscountFactorModel`

This contract represents dated observed cross-sections with a date-local slot axis.

```python
from ml4t.models import CrossSectionBatch

batch = CrossSectionBatch(
    characteristics=chars_tnp,
    returns=returns_tn,
    context_features=context_tq,
    timestamps=timestamps,
    mask=mask_tn,
)
```

### Why The Slot Axis Matters

For many finance datasets, a fixed global `T × N × P` panel is the wrong abstraction. What matters is:

- the cross-section observed at each date
- which assets are present on that date
- which returns are valid for training or scoring

`CrossSectionBatch` makes that explicit with:

- `characteristics: (T, N_slots, P)`
- `returns: (T, N_slots)` when available
- `mask: (T, N_slots)` for observed assets
- optional `context_features: (T, Q)` for macro or market state

## PortfolioSequenceBatch

Used by:

- `LinearFeaturePortfolioModel`
- `LSTMPortfolioModel`
- `DeepPortfolioModel`

This contract is for sequence-to-allocation models.

```python
from ml4t.models import PortfolioSequenceBatch

batch = PortfolioSequenceBatch(
    features=features_btnf,
    returns=returns_btn,
    prev_weights=prev_weights_bn,
    costs=costs_n,
    group_ids=group_ids_n,
    adjacency_mask=adjacency_nn,
    timestamps=timestamps,
    asset_ids=asset_ids,
)
```

### Typical Shape Semantics

- `B`: rolling windows or mini-batches
- `T`: time steps inside each window
- `N`: assets
- `F`: features

## Surface Adapters

When you already have tabular long-format data, use the integration helpers:

```python
from ml4t.models import cross_section_batch_from_long_frame, persistent_panel_batch_from_long_frame
```

These resolve:

- timestamp column
- entity column
- long-frame to batch reshaping
- optional ML4T-style schema metadata

## Which Contract Goes With Which Model?

| Contract | Models |
|---|---|
| `PersistentPanelBatch` | `PCAModel`, `RPPCAModel` |
| `CrossSectionBatch` | `IPCAModel`, `CAEModel`, `SAEModel`, `StochasticDiscountFactorModel` |
| `PortfolioSequenceBatch` | `LinearFeaturePortfolioModel`, `LSTMPortfolioModel`, `DeepPortfolioModel` |

## Common Mistake

Do not force ragged cross-sectional models into a fixed panel just because a tensor shape is convenient. In asset pricing, that often changes the economic problem rather than just the implementation.

