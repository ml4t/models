# Book Guide

`ml4t-models` is the library form of the model families developed manually in the book notebooks.

The goal is not to hide the teaching implementation. The goal is to:

- show the architecture and mathematics clearly in the chapter notebooks
- use the library for repeatable case-study execution and downstream integration

## Chapter Mapping

### Chapter 14: Latent Factors

The latent-factor chapter corresponds most directly to:

- `PCAModel`
- `RPPCAModel`
- `IPCAModel`
- `CAEModel`
- `StochasticDiscountFactorModel`
- `SAEModel` as supervised autoencoder direct prediction

The key conceptual transition from the notebooks to the library is:

- notebook exposition may derive the math and architecture step by step
- library code enforces the clean separation between:
  - structural extraction
  - factor forecasting
  - asset mapping

### Chapter 17: Portfolio Construction

The end-to-end allocation family corresponds to:

- `LinearFeaturePortfolioModel`
- `LSTMPortfolioModel`
- `DeepPortfolioModel`

These models are designed to connect naturally to:

- Chapter 18 cost modeling
- Chapter 19 risk controls
- Chapter 20 strategy analysis

## Why The Library Split Matters

The book often needs to compare multiple modeling ideas side by side:

- latent-factor models
- no-arbitrage SDF models
- direct signal models
- end-to-end allocation models

The library turns those into explicit families instead of treating them as one generic “deep learning model.”

## Case Studies

The case studies are intended to act as:

- integration tests
- realistic pressure tests for the API
- examples of how to hand model outputs into `ml4t-backtest` and `ml4t-diagnostic`

They should not define the public API by accident.

## Recommended Reading Order

If you are moving from the book notebooks to the library:

1. [Data Contracts](../user-guide/data-contracts.md)
2. [Latent-Factor Pipelines](../user-guide/latent-factor-pipelines.md)
3. [Stochastic Discount Factor](../user-guide/stochastic-discount-factor.md)
4. [Portfolio Learning](../user-guide/portfolio-learning.md)
5. [Integration](../user-guide/integration.md)

