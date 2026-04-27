# Architecture

The library is organized around finance-native model families and contracts.

## Top-Level Design

![ml4t-models Architecture](../images/ml4t_models_architecture.svg)

## Why The Families Are Separate

### Latent Factors

These families estimate structural state first:

- loadings or conditional betas
- latent factor realizations
- then expected returns through a separate factor-premium forecast

### Stochastic Discount Factor

This family estimates a weight-native no-arbitrage object:

- asset weights
- SDF series
- optional downstream return projections

### Direct Asset Prediction

This family predicts signals directly:

- no latent factor state
- no separate premium forecast

### Portfolio Learning

This family learns allocations directly:

- sequential input windows
- cost-aware or risk-aware objectives
- target-weight outputs

## Latent-Factor Pipeline Diagram

```mermaid
flowchart LR
    A[PersistentPanelBatch or CrossSectionBatch] --> B[Structural Estimator]
    B --> C[LatentFactorState]
    C --> D[Factor Forecaster]
    D --> E[FactorForecastResult]
    C --> F[Asset Mapper]
    E --> F
    F --> G[AssetForecastResult]
    G --> H[Prediction Surface]
    H --> I[Backtest / Diagnostic]
```

## Stochastic Discount Factor Flow

```mermaid
flowchart LR
    A[CrossSectionBatch] --> B[Unconditional SDF Phase]
    B --> C[Moment Network Phase]
    C --> D[Conditional SDF Phase]
    D --> E[StochasticDiscountFactorState]
    E --> F[Weight Surface]
    E --> G[Optional Return Projection]
```

## Portfolio Flow

```mermaid
flowchart LR
    A[PortfolioSequenceBatch] --> B[Portfolio Model]
    B --> C[PortfolioWeightsResult]
    C --> D[Postprocessor]
    D --> E[Target Weight Surface]
    E --> F[ml4t-backtest]
```

## Package Layout

```text
ml4t.models
├── api.py
├── types.py
├── pipelines.py
├── configs/
├── latent_factors/
├── forecasters/
├── mappers/
├── stochastic_discount_factor/
├── asset_prediction/
├── portfolio/
└── integration/
```

## Boundary Rules

### Belongs Here

- model estimation
- batch and result contracts
- checkpoint handling
- surface emission

### Belongs Elsewhere

- feature engineering: `ml4t-engineer`
- execution and order simulation: `ml4t-backtest`
- validation and diagnostics: `ml4t-diagnostic`

