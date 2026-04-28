# Academic References

This page is the canonical bibliography for the model families implemented in `ml4t-models`. Every claim that touches an estimator, a loss, or an architectural choice in the user guide should resolve here. Citations follow the Chicago author-date convention used throughout *Machine Learning for Trading*; the paper anchors mirror the bibliography in `~/ml4t/book/14_latent_factors/bibliography.json`.

If a model adds a new mechanism, the corresponding paper should be added here first, then referenced from the user guide.

## How To Read This Page

Each entry includes:

- the implementation in `ml4t-models` it supports
- the specific role the paper plays for that model (definition, identification, training procedure, evaluation diagnostic)
- a short qualification when the library departs from the paper

The bibliography is grouped by model family. Cross-cutting references (factor zoo critique, weak-factor theory, replication evidence) live in the final section.

---

## PCA on Asset Returns

Implementation: [`PCAModel`](../user-guide/latent-factor-models.md#pca).

### <a name="ref-connor-korajczyk-2009"></a>Connor and Korajczyk (2009)

*Factor Models of Asset Returns.* Provides the canonical statement of static-loading approximate factor models for return panels and motivates the eigendecomposition of the demeaned return covariance as a structural extraction step (rather than a forecasting step).

### <a name="ref-litterman-scheinkman-1991"></a>Litterman and Scheinkman (1991)

*Common Factors Affecting Bond Returns.* The classical low-dimensional yield-curve PCA result: three components (level, slope, curvature) explain almost all variation in Treasury returns. Used in the chapter to justify component selection by economic interpretation rather than scree-elbow heuristics alone.

### <a name="ref-avellaneda-lee-2010"></a>Avellaneda and Lee (2010)

*Statistical Arbitrage in the US Equities Market.* Establishes the eigenportfolio framing: principal-component vectors are tradable portfolios, and PCA residuals can be modeled with mean-reverting OU processes for stat-arb signals. The library’s `PCAModel` exposes the loadings that this paper interprets as portfolio weights.

### <a name="ref-avellaneda-2019"></a>Avellaneda (2019)

*Hierarchical PCA and Applications to Portfolio Management.* Develops sector-conditional PCA as a robustness fix for global equity decompositions. Out of scope for the v0.1 implementation but flagged in the [Roadmap](https://github.com/ml4t/models/blob/main/ROADMAP.md) as a candidate refinement.

### <a name="ref-baik-ben-arous-peche-2005"></a>Baik, Ben Arous, and Péché (2005)

*Phase Transition of the Largest Eigenvalue for Nonnull Complex Sample Covariance Matrices.* The BBP threshold: a factor is recoverable from a sample covariance matrix only if its population eigenvalue exceeds $1 + \sqrt{n/T}$. This bound governs how many components `PCAModel` can recover when $n/T$ is non-trivial.

### <a name="ref-paleologo-2025"></a>Paleologo (2025)

*The Elements of Quantitative Investing.* Chapter 7 collects the random-matrix-theory machinery (Marchenko–Pastur, BBP, eigenvalue shrinkage) that the chapter relies on for component selection.

---

## RP-PCA

Implementation: [`RPPCAModel`](../user-guide/latent-factor-models.md#rp-pca).

### <a name="ref-lettau-pelger-2020"></a>Lettau and Pelger (2020)

*Estimating Latent Asset-Pricing Factors.* Defines the RP-PCA objective as a weighted sum of unexplained variance and squared cross-sectional pricing errors, controlled by a single parameter $\gamma$. The library exposes `gamma`, `base_moment` (covariance vs. second moment), `scale_by_asset_volatility`, and `normalize_loadings` as direct counterparts of the paper's design knobs. Out-of-sample Sharpe gains over standard PCA are reported there but are not reproduced inside the library — that is the case-study's job.

---

## IPCA

Implementation: [`IPCAModel`](../user-guide/latent-factor-models.md#ipca).

### <a name="ref-kelly-pruitt-su-2019"></a>Kelly, Pruitt, and Su (2019)

*Characteristics Are Covariances: A Unified Model of Risk and Return.* The IPCA model: excess returns are explained by characteristic-implied conditional betas times latent factors,

$$ r^{e}_{i,t+1} = z_{i,t}^{\top} \Gamma f_{t+1} + \varepsilon_{i,t+1}. $$

The library’s alternating-least-squares solver mirrors the paper's identification: $\Gamma$ is updated holding $f_t$ fixed and vice versa, with ridge stabilizers on both sides (`gamma_ridge`, `factor_ridge`).

### <a name="ref-didisheim-2023"></a>Didisheim et al. (2023)

*Complexity in Factor Pricing Models.* Shows theoretically and empirically that factor models with many parameters can outperform sparse ones out of sample, qualifying the "select a single optimal $K$" intuition. The library's IPCA defaults treat $K$ as a swept argument, not a parsimony pick.

---

## Conditional Autoencoder

Implementation: [`CAEModel`](../user-guide/latent-factor-models.md#cae).

### <a name="ref-gu-kelly-xiu-2021"></a>Gu, Kelly, and Xiu (2021)

*Autoencoder Asset Pricing Models.* Defines the conditional autoencoder (CAE) as a non-linear generalization of IPCA: the linear $\beta = \Gamma' z$ map is replaced by a feed-forward network $\beta = g(z; W)$, and the factor-network input is the characteristic-managed-portfolio block

$$ x_t = (Z_{t-1}^{\top} Z_{t-1})^{-1} Z_{t-1}^{\top} r_t. $$

The library's `CAEModel` follows this dual-network structure (`hidden_units` controls the beta-network pyramid, `n_ensemble` averages over random initializations) and trains under MSE reconstruction loss with optional L1 regularization (`lambda_l1`). Reported GKX out-of-sample Sharpe figures (~1.53 value-weighted) are not reproduced inside the library.

---

## Stochastic Discount Factor Network

Implementation: [`StochasticDiscountFactorModel`](../user-guide/stochastic-discount-factor.md).

### <a name="ref-chen-pelger-zhu-2021"></a>Chen, Pelger, and Zhu (2021)

*Deep Learning in Asset Pricing.* The CPZ adversarial SDF: a neural network parameterizes the pricing kernel $M_{t+1}$, a separate moment network learns the worst-case test-asset weighting, and the two networks train against each other so that the SDF is disciplined by the most-mispriced portfolio at every step. The library implements the three-phase training that the paper describes:

1. unconditional pre-training of the SDF network (`n_epochs_unc`),
2. moment-network fitting (`n_epochs_moment`),
3. conditional-SDF refinement under the adversarial signal (`n_epochs_cond`).

The output is weight-native: `StochasticDiscountFactorState.asset_weights` are the SDF portfolio weights; `sdf_values` are the realized $M_{t+1}$ series. Expected-return-style projections require the optional [beta-head network](../user-guide/stochastic-discount-factor.md#optional-return-mapping).

### <a name="ref-gospodinov-kan-robotti-2017"></a>Gospodinov, Kan, and Robotti (2017)

*Spurious Inference in Reduced-Rank Asset-Pricing Models.* The cautionary counterpart to the CPZ adversarial machinery: cross-sectional GMM fit can be made arbitrarily good through weighting-matrix choices when the model is misspecified. The library does not encode this critique mechanically but the [user guide](../user-guide/stochastic-discount-factor.md) calls out that pricing-error minimization alone does not certify economic validity.

---

## Supervised Autoencoder

Implementation: [`SAEModel`](../user-guide/direct-asset-prediction.md).

The SAE in this library is the **supervised autoencoder** of the Jane Street competition lineage — encoder, decoder, auxiliary head, and main predictive head trained jointly under a composite loss. It is **not** a sparse autoencoder and **not** a latent-factor model. The library treats `SAEModel` as a direct supervised cross-sectional predictor; `predict()` returns `AssetSignalResult`, not a beta-times-lambda decomposition.

The library's contract draws on the supervised-autoencoder pattern documented in:

- the Jane Street Market Prediction competition winners' write-ups (encoder, denoising bottleneck, auxiliary classification head, MLP main head trained jointly),
- and the broader literature on auxiliary-task regularization for tabular cross-sectional prediction.

Treat this section as a deliberate departure from chapter conventions: the chapter discusses SAE alongside latent-factor models because the architectures share a bottleneck, but in production they answer different questions and the library reflects that.

---

## End-to-End Portfolio Learning

Implementations: [`LinearFeaturePortfolioModel`](../user-guide/portfolio-learning.md#linearfeatureportfoliomodel), [`LSTMPortfolioModel`](../user-guide/portfolio-learning.md#lstmportfoliomodel), [`DeepPortfolioModel`](../user-guide/portfolio-learning.md#deepportfoliomodel).

The portfolio family draws on two threads.

### End-to-end portfolio learning

The Sharpe-ratio-as-loss training framework: optimize portfolio weights directly under a differentiable Sharpe-style or turnover-aware objective rather than chaining a return forecast into a separate optimizer. This is the design behind the linear and LSTM baselines, and motivates why portfolio costs (`costs`) and previous weights (`prev_weights`) appear in `PortfolioSequenceBatch` rather than only in a downstream backtester.

### DeePM-style structured portfolio learning

`DeepPortfolioModel` implements a DeePM-style allocator: static context encoding, feature modulation, variable selection, an LSTM temporal backbone, temporal self-attention, cross-sectional attention, and optional macro-graph attention. The architecture choices come from the published end-to-end portfolio-learning literature (TFT-style encoders for context; DeePM-style multi-head architectures for cross-sectional structure). The library presents this as one structured allocator, not a generic Transformer wrapper.

### <a name="ref-wood-roberts-zohren-2026"></a>Wood, Roberts, and Zohren (2026)

*DeePM: Regime-Robust Deep Learning for Systematic Macro Portfolio Management.* Provides the structured portfolio-learning blueprint behind `DeepPortfolioModel`: sequence encoding, cross-sectional interaction, graph-aware structure, and direct optimization of a robust risk-adjusted objective under transaction costs.

### <a name="ref-kisiel-gorse-2023"></a>Kisiel et al. (2023)

*Portfolio Transformer for Attention-Based Asset Allocation.* Cited here as motivation for treating attention-based allocators as a separate family rather than a special case of return forecasting.

---

## Cross-Cutting References

These papers do not pin a specific estimator but shape the library's framing.

### <a name="ref-cochrane-2011"></a>Cochrane (2011)

*Discount Rates.* Frames the factor-zoo problem and the variance-vs-pricing distinction that motivates separating PCA / RP-PCA / IPCA / CAE from the SDF family.

### <a name="ref-harvey-liu-zhu-2016"></a>Harvey, Liu, and Zhu (2016)

*…and the Cross-Section of Expected Returns.* The multiple-testing critique that justifies treating factor discovery as a disciplined model-selection problem rather than open-ended specification search. Cited in the user guide only when arguing for the RP-PCA / IPCA / CAE design's structural posture, not as a default citation.

### <a name="ref-feng-giglio-xiu-2020"></a>Feng, Giglio, and Xiu (2020)

*Taming the Factor Zoo: A Test of New Factors.* Double-selection LASSO inference for new-factor evaluation. The library does not bundle this test; it is the recommended diagnostic when a user adds a new latent-factor model and wants robust incremental-significance evidence.

### <a name="ref-jensen-kelly-pedersen-2022"></a>Jensen, Kelly, and Pedersen (2022)

*Is There a Replication Crisis in Finance?* Hierarchical Bayesian replication across 153 factors in 93 countries; finds that factors cluster into roughly 13 themes that hold up out of sample. Used in the user guide to qualify how aggressively to interpret factor-zoo critique when interpreting `RPPCAModel` and `IPCAModel` outputs.

### <a name="ref-bryzgalova-pelger-zhu-2025"></a>Bryzgalova, Pelger, and Zhu (2025)

*Forest through the Trees: Building Cross-Sections of Stock Returns.* Endogenous test-asset construction. Cited as the design rationale for treating test-asset choice as a first-class evaluation knob rather than a neutral backdrop, and listed in the Roadmap as a future model addition.

### <a name="ref-bagnara-2024"></a>Bagnara (2024)

*Asset Pricing and Machine Learning: A Critical Review.* Survey that organizes the IPCA / RP-PCA / CAE / SDF families along the same axes the library uses (variance maximization, variance plus pricing, no-arbitrage). Useful onboarding reading for anyone moving between the chapter and the library.

---

## What This Bibliography Deliberately Excludes

Three classes of citations are kept out on purpose, in line with the user-guide voice rules:

- **Default trading-research citations** (Harvey 2016 standalone, Lopez de Prado on PBO / triple-barrier / fractional differentiation). These are out of scope for `ml4t-models`; they shape `ml4t-diagnostic` and `ml4t-engineer`, not this library.
- **One-off application papers** that do not generalize to a reusable model family. See `ROADMAP.md` for the explicit prioritization rule.
- **Numerical results from the chapter case studies.** Those live with the case-study artifacts and the `run_log/registry.db` they are emitted to. Importing them here would create a maintenance liability the library cannot honor.
