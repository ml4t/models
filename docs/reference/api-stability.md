# API Stability

This page defines the public surface that is intended to be stable for the
`0.1` beta series.

## Stable Import Surface

Use these modules in notebooks, examples, and downstream libraries:

| Surface | Purpose |
|---|---|
| `ml4t.models` | convenience imports for reader-facing workflows |
| `ml4t.models.api` | structural protocols for model families |
| `ml4t.models.types` | batch, state, prediction, and frame contracts |
| `ml4t.models.configs` | frozen dataclass configs |
| `ml4t.models.integration` | adapters for long frames, diagnostics, and backtests |

The package root re-exports the core model classes, configs, result contracts,
and integration helpers used throughout the documentation.

## Stable Families

The beta API is organized around four model families:

| Family | Stable models |
|---|---|
| Latent factors | `PCAModel`, `RPPCAModel`, `IPCAModel`, `CAEModel` |
| Stochastic discount factor | `StochasticDiscountFactorModel` |
| Direct asset prediction | `SAEModel` |
| Portfolio learning | `LinearFeaturePortfolioModel`, `LSTMPortfolioModel`, `DeepPortfolioModel` |

Each family keeps its own contract because the native object differs by model:
latent-factor state, SDF weights, direct signals, or portfolio weights.

## Naming Conventions

Reader-facing result names describe the object they carry:

| Name | Object |
|---|---|
| `AssetForecastResult` | expected returns by timestamp and asset |
| `AssetSignalResult` | direct asset-level scores or signals |
| `AssetWeightsResult` | cross-sectional SDF or asset weights |
| `PortfolioWeightsResult` | sequence-model portfolio allocations |
| `PredictionsFrame` | long-format expected-return predictions |
| `SignalsFrame` | long-format direct signals |
| `WeightsFrame` | long-format portfolio or asset weights |
| `ResultsFrame` | common frame protocol implemented by integration outputs |

Avoid names that describe implementation mechanics rather than the financial
object. If a new public result type is added, it should fit this pattern.

## Extension Points

The following extension points are intentionally public:

- `LatentFactorModel`
- `FactorForecaster`
- `AssetMapper`
- `AssetPredictionModel`
- `StochasticDiscountFactorEstimator`
- `PortfolioModel`
- `PortfolioPostprocessor`

New models should implement the relevant protocol and return the existing result
contracts before introducing a new public type.

## Deferred Surface

The beta release does not freeze internals under `ml4t.models._internal`.
Functions and modules prefixed with `_` are implementation details and may change
between `0.1` releases.
