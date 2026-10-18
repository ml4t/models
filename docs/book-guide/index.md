# Book Guide

The public companion repository at revision
[`d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb`](https://github.com/stefan-jansen/machine-learning-for-trading/tree/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb)
provides the teaching files below. Every linked path was checked against that revision's Git tree.
These notebooks explain methods and often use their own data and dependencies. None of the Chapter 14
teaching notebooks below imports `ml4t.models`; run the library's small examples for its API.

## Latent factors and factor forecasts

| Public book file | What it does | Related library task |
|---|---|---|
| [IPCA notebook](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/14_latent_factors/04_ipca.ipynb) | Manually teaches characteristic-dependent betas and factor forecasts. | [Fit a latent-factor pipeline](../user-guide/latent-factor-pipelines.md) with `IPCAModel` and a separate forecaster. |
| [Risk-premium PCA notebook](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/14_latent_factors/05_rp_pca.ipynb) | Manually teaches pricing-aware factor extraction. | [Choose a latent-factor model](../user-guide/latent-factor-models.md) with `RPPCAModel`. |
| [Conditional autoencoder notebook](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/14_latent_factors/06_conditional_autoencoder.ipynb) | Manually builds a neural conditional factor model. | [Choose a latent-factor model](../user-guide/latent-factor-models.md) with `CAEModel`; the library requires `deep`. |
| [Case-study insights notebook](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/14_latent_factors/09_case_study_insights.ipynb) | Illustrates analysis of stored case-study results, not a first API example. | [Hand results downstream](../user-guide/integration.md). |

The book's [case-study library bridge](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/case_studies/utils/latent_factors/library_bridge.py)
*does* import `ml4t.models` for PCA, IPCA, CAE, SDF, and SAE runs. It is case-study integration
code with data and registry prerequisites, not a standalone quickstart.

## SDF and direct prediction

| Public book file | What it does | Related library task |
|---|---|---|
| [Adversarial SDF notebook](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/14_latent_factors/07_stochastic_discount_factor.ipynb) | Manually teaches phase-aware SDF training. | [Estimate SDF weights](../user-guide/stochastic-discount-factor.md) with `StochasticDiscountFactorModel`. |
| [Supervised autoencoder notebook](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/14_latent_factors/08_supervised_autoencoder.ipynb) | Manually teaches a direct supervised predictor. | [Predict asset signals](../user-guide/direct-asset-prediction.md) with `SAEModel`. |

These neural notebooks need PyTorch and book data. Their full training runs are longer than the
small CPU examples in this site's task guides. The book's targets and splits may also differ from
the synthetic examples; results are not directly comparable.

## Portfolio learning

| Public book file | What it does | Related library task |
|---|---|---|
| [Deep portfolio optimization](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/17_portfolio_construction/11_dl_portfolio_allocation.ipynb) | Illustrates a related neural allocation workflow, without calling this library. | [Learn portfolio weights](../user-guide/portfolio-learning.md). |
| [VLSTM portfolio](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/17_portfolio_construction/12_vlstm_portfolio.ipynb) | Manually teaches variable selection and sequence allocation. | [Learn portfolio weights](../user-guide/portfolio-learning.md) with `LSTMPortfolioModel`. |
| [DeePM regime robustness](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/17_portfolio_construction/13_deepm_regime_robust.ipynb) | Manually teaches a related DeePM architecture; it is not an exact library implementation. | [Learn portfolio weights](../user-guide/portfolio-learning.md) with `DeepPortfolioModel`. |

Portfolio notebooks use book-specific data and longer neural training. Start with the library's
[linear CPU example](https://github.com/ml4t/models/blob/main/examples/portfolio_learning.py)
for an observable result.

## Data and downstream boundaries

The library's [data contracts](../user-guide/data-contracts.md) preserve timestamps and asset
identity. [Integration](../user-guide/integration.md) converts predictions and weights to frames for
`ml4t-diagnostic` and `ml4t-backtest`. The Chapter 14 case-study insights notebook illustrates
analysis after model runs, but it does not replace those packages' guides or APIs.

The [Quickstart](../getting-started/quickstart.md) is the first runnable library workflow.
