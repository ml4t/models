# Model Families

`ml4t-models` is organized around four families. The split is deliberate: each family answers a different question about a return panel, and forcing them through one common interface would obscure that.

This page is the canonical map. Use it as a lookup; the [user guide](../user-guide/index.md) develops each family in depth and the [academic references](academic-references.md) page collects the papers behind each implementation.

![Model Family Map](../images/ml4t_model_family_map.svg)

## The Four Families At A Glance

| Family | Models | Native object | Predictive object |
|---|---|---|---|
| Latent factors | `PCAModel`, `RPPCAModel`, `IPCAModel`, `CAEModel` | exposures + factor history | $\beta \cdot \lambda$ after a factor-premium forecast |
| Stochastic discount factor | `StochasticDiscountFactorModel` | tradable weight vector + SDF series | optional return projection through a beta head |
| Direct asset prediction | `SAEModel` | asset-level signal | the signal itself |
| End-to-end portfolio learning | `LinearFeaturePortfolioModel`, `LSTMPortfolioModel`, `DeepPortfolioModel` | sequence-to-allocation tensor | target weights |

Two principles run through the table:

1. The native object and the predictive object are different in three of four families. Latent-factor models extract structure first; SDF models learn weights first; portfolio models learn weights from the start. Forecasting and weight extraction are kept separate so that downstream code (forecasters, mappers, postprocessors) can be swapped without retouching the structural estimator.
2. The contract follows the economics. PCA needs a stable entity axis; IPCA tolerates ragged dated cross-sections; portfolio learning needs sequence batches with cost and turnover state. The library represents these distinctions with three batch types (`PersistentPanelBatch`, `CrossSectionBatch`, `PortfolioSequenceBatch`) rather than collapsing them into one tensor.

## Latent Factors

| Class | Contract | Loading structure | Estimator |
|---|---|---|---|
| `PCAModel` | `PersistentPanelBatch` | static loadings | SVD on demeaned panel |
| `RPPCAModel` | `PersistentPanelBatch` | static loadings, risk-premium-aware extraction | eigendecomposition of $\Sigma + \gamma\, \bar{r}\bar{r}^{\top}$ |
| `IPCAModel` | `CrossSectionBatch` | linear $\beta_{i,t} = \Gamma^{\top} z_{i,t}$ | alternating least squares |
| `CAEModel` | `CrossSectionBatch` | nonlinear $\beta_{i,t} = g(z_{i,t}; W)$ | dual-network neural training, ensemble averaging |

Predictive pattern, shared across the family:

```text
extract structural state -> forecast factor premia -> map back to assets
```

The pattern is enforced by [`LatentFactorForecastPipeline`](../user-guide/latent-factor-pipelines.md), which composes a structural model, a [factor-premium forecaster](../user-guide/latent-factor-pipelines.md#factor-forecaster), and a [`BetaLambdaMapper`](../user-guide/latent-factor-pipelines.md#asset-mapper) into a single object exposing `fit` and `predict`.

The split is not cosmetic. For IPCA and CAE, the in-sample fitted return uses *realized* factor returns; the *implementable* forecast must replace those with an ex-ante factor-premium estimate. Conflating the two is the most common chapter-to-production failure mode.

## Stochastic Discount Factor

| Class | Contract | Estimator |
|---|---|---|
| `StochasticDiscountFactorModel` | `CrossSectionBatch` | three-phase adversarial training (unconditional / moment / conditional) |

The native object is a portfolio-weight vector $\hat\omega_{t,i}$ together with its induced SDF series $\hat M_{t+1}$, not a $\beta \cdot \lambda$ decomposition. Expected-return-style projections are handled by optional helpers:

- `LinearStochasticDiscountFactorReturnMapper`
- `StochasticDiscountFactorBetaNetworkHead`

These are downstream transformations. They do not change the fact that the structural estimator is weight-native.

The model is phase-aware (`n_epochs_unc`, `n_epochs_moment`, `n_epochs_cond`) and checkpointed (`checkpoint_epochs`, `default_checkpoint`) so that long training horizons can be evaluated at sparse, named milestones rather than at dense intervals.

## Direct Asset Prediction

| Class | Contract | Architecture |
|---|---|---|
| `SAEModel` | `CrossSectionBatch` | encoder, decoder, auxiliary head, main predictive head |

`SAEModel` is the **supervised autoencoder** of the Jane Street competition lineage. It is treated as a direct cross-sectional predictor: `predict(batch)` returns `AssetSignalResult`, not `AssetForecastResult`. The model lives outside `latent_factors` because it does not factorize returns into $\beta \cdot \lambda$, even though it shares the bottleneck-and-decoder shape with autoencoder factor models.

## End-to-End Portfolio Learning

| Class | Contract | Style |
|---|---|---|
| `LinearFeaturePortfolioModel` | `PortfolioSequenceBatch` | deterministic pooled-linear baseline |
| `LSTMPortfolioModel` | `PortfolioSequenceBatch` | sequence baseline with variable selection and bounded head |
| `DeepPortfolioModel` | `PortfolioSequenceBatch` | DeePM-style allocator with temporal and cross-sectional attention |

All three optimize a portfolio objective directly. The default training loop is differentiable Sharpe with turnover and cost penalties; it is not a "predict returns, then optimize" two-stage workflow. That is the design contract — if you want a forecast first, use the latent-factor pipeline.

## Forecasters

Factor-premium forecasters consume `LatentFactorState` and produce `FactorForecastResult`:

| Class | Idea |
|---|---|
| `ExpandingMeanFactorForecaster` | sample-mean baseline; the standard premium estimate |
| `AR1FactorForecaster` | per-factor AR(1) with shrinkage to mean |
| `EWMABaseFactorForecaster` | EWMA with configurable half-life |

The expanding-mean forecaster is the default for two reasons: it requires no tuning, and it makes the failure modes of a richer forecaster easier to diagnose. Switch to AR(1) or EWMA only when there is direct evidence of premium time-variation in the in-sample factor returns.

## Mappers

| Class | Idea |
|---|---|
| `BetaLambdaMapper` | computes $\hat r_{i,t+1} = \beta_{i,t} \cdot \hat\lambda_{t+1}$ from the structural state and the factor-premium forecast |

This is the only mapper in v0.1. Anything more elaborate (state-conditional premia, regime-switching mappers) is roadmap.

## Configs

Every estimator takes a frozen dataclass config. The conventions are:

- `model_name` — used in metadata and run logs.
- `n_factors` — for latent-factor models.
- `checkpoint_*` — for neural models, controls which epochs are kept.
- `default_checkpoint` — the epoch returned by `extract` / `predict` when called without an explicit `checkpoint=` argument.

Configs are immutable; instantiate a new one to change any field.

## Integration

Boundary helpers live in `ml4t.models.integration` and are documented in the [Integration user guide](../user-guide/integration.md). The role of integration code is to translate model results into the long-format frames `ml4t-backtest` and `ml4t-diagnostic` consume, and to translate long-format frames into the typed batches each model expects. It does not perform evaluation, signal aggregation, or execution simulation.
