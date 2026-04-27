# User Guide

`ml4t-models` is organized around model semantics, not around a generic trainer abstraction.

That means the fastest way to understand the library is to ask four questions:

1. what data contract does the model require?
2. what object does it estimate?
3. what is the native output?
4. what still has to happen before you can trade or backtest it?

## The Four Main Workflows

### 1. Latent-Factor Forecasting

Used by:

- `PCAModel`
- `RPPCAModel`
- `IPCAModel`
- `CAEModel`

Workflow:

```text
batch -> structural model -> latent factor state -> factor forecaster -> asset mapper
```

This is the right abstraction for models where:

- exposures and factor realizations are the structural objects
- expected returns come from a separate premium forecast

### 2. Stochastic Discount Factor Estimation

Used by:

- `StochasticDiscountFactorModel`

Workflow:

```text
cross-section batch -> phase-aware no-arbitrage training -> asset weights + SDF series
```

This family is intentionally separate because the native object is a traded pricing kernel proxy, not a latent factor plus a premium forecast.

### 3. Direct Asset Prediction

Used by:

- `SAEModel`

Workflow:

```text
cross-section batch -> checkpointed predictor -> asset signals
```

This is where the library puts supervised models that predict asset-level signals directly.

### 4. End-To-End Portfolio Learning

Used by:

- `LinearFeaturePortfolioModel`
- `LSTMPortfolioModel`
- `DeepPortfolioModel`

Workflow:

```text
portfolio sequence batch -> allocation model -> target weights -> optional postprocessing
```

These models optimize allocation decisions directly rather than first estimating returns.

## Design Rules

- Stable-ID panel models and ragged cross-sectional models use different contracts.
- Neural checkpoints are configurable rather than hard-coded.
- Forecasting is kept outside structural latent-factor estimation.
- Evaluation belongs in `ml4t-diagnostic`, not in this library.
- Execution belongs in `ml4t-backtest`, not in this library.

## Reading Order

- [Data Contracts](data-contracts.md)
- [Latent-Factor Pipelines](latent-factor-pipelines.md)
- [Latent-Factor Models](latent-factor-models.md)
- [Stochastic Discount Factor](stochastic-discount-factor.md)
- [Direct Asset Prediction](direct-asset-prediction.md)
- [Portfolio Learning](portfolio-learning.md)
- [Integration](integration.md)

