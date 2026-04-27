# Model Families

This page summarizes the current public model families and how they map to the literature and the API.

## Latent Factors

| Class | Contract | Main idea |
|---|---|---|
| `PCAModel` | `PersistentPanelBatch` | covariance-style latent decomposition with static loadings |
| `RPPCAModel` | `PersistentPanelBatch` | latent decomposition that also emphasizes risk-premium information |
| `IPCAModel` | `CrossSectionBatch` | linear characteristic-implied exposures |
| `CAEModel` | `CrossSectionBatch` | nonlinear characteristic-implied exposures |

Predictive pattern:

```text
extract structural state -> forecast factor premia -> map back to assets
```

## Stochastic Discount Factor

| Class | Contract | Main idea |
|---|---|---|
| `StochasticDiscountFactorModel` | `CrossSectionBatch` | phase-aware no-arbitrage estimation of a weight-native SDF |

Optional helpers:

- `LinearStochasticDiscountFactorReturnMapper`
- `StochasticDiscountFactorBetaNetworkHead`

## Direct Asset Prediction

| Class | Contract | Main idea |
|---|---|---|
| `SAEModel` | `CrossSectionBatch` | supervised autoencoder for direct asset-level signals |

## Portfolio Learning

| Class | Contract | Main idea |
|---|---|---|
| `LinearFeaturePortfolioModel` | `PortfolioSequenceBatch` | deterministic linear pooled baseline |
| `LSTMPortfolioModel` | `PortfolioSequenceBatch` | sequence-based allocation learner |
| `DeepPortfolioModel` | `PortfolioSequenceBatch` | structured DeePM-style allocator |

## Forecasters

Current factor-premium forecasters:

- `ExpandingMeanFactorForecaster`
- `AR1FactorForecaster`
- `EWMABaseFactorForecaster`

## Mappers

Current latent-factor mapper:

- `BetaLambdaMapper`

## Integration Surface

Current integration modules:

- `ml4t.models.integration.data`
- `ml4t.models.integration.surfaces`
- `ml4t.models.integration.backtest`

