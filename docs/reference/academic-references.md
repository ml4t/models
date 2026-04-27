# Academic References

This page explains the conceptual sources behind each model family in the current library.

## PCA

`PCAModel` is the persistent-panel baseline for latent factor extraction.

In the library, PCA is treated as:

- a structural decomposition
- not a direct forecasting model

## RP-PCA

`RPPCAModel` follows the risk-premium-aware PCA idea summarized in the RP-PCA resources:

- covariance structure alone may miss pricing-relevant low-variance factors
- the extraction objective can be tilted toward expected-return information

Library interpretation:

- persistent-panel latent-factor model
- structural state first
- predictive use via downstream premium forecasting and mapping

## IPCA

`IPCAModel` follows the linear conditional factor view:

- characteristics imply exposures
- factor realizations are estimated jointly with those exposures
- predictive forecasts use current implied betas and estimated factor premia

Library interpretation:

- the total fitted return and the ex ante predictive return are different objects
- only the latter is suitable for implementable forecasting

## CAE

`CAEModel` follows the conditional autoencoder interpretation:

- nonlinear characteristic-to-beta mapping
- latent factors estimated from managed portfolios
- predictive use requires replacing realized factors with ex ante factor-premium estimates

The scalable CAE resources further motivate:

- modular factor forecasting
- moving beyond a simple sample-mean premium baseline

That is why the library keeps:

- structural extraction
- factor forecasting
- asset mapping

as separate stages.

## Stochastic Discount Factor

`StochasticDiscountFactorModel` follows the no-arbitrage, weight-native perspective reflected in the SDF resources:

- the model learns a pricing kernel proxy through conditional moment restrictions
- the native object is a traded weight vector and induced SDF series
- expected-return projections are secondary derivatives of that object

That is why the library documents this family separately from latent factors.

## SAE

`SAEModel` refers to **supervised autoencoder**, not sparse autoencoder.

The current library interpretation follows the Jane Street-style supervised architecture lineage:

- encoder
- decoder
- auxiliary prediction head
- main prediction head

This is why the model lives in `asset_prediction` rather than under `latent_factors`.

## End-To-End Portfolio Learning

The portfolio family draws on two related ideas from the resources:

### End-to-end portfolio optimization

- optimize a portfolio objective directly
- do not force the workflow through a two-stage return-forecast-then-optimize pipeline

### DeePM-style structured portfolio learning

- combine temporal encoding, cross-sectional interaction, and structural priors
- train under robust or turnover-aware allocation objectives

The library currently reflects this through:

- `LinearFeaturePortfolioModel`
- `LSTMPortfolioModel`
- `DeepPortfolioModel`

with `DeepPortfolioModel` documented as a DeePM-style allocator rather than as a generic transformer.

