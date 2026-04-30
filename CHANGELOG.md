# Changelog

All notable changes to `ml4t-models` will be documented in this file.

## Unreleased

- Neural configs: `device="mps"` is supported on Apple Silicon when PyTorch MPS is available
  (with CPU fallback), alongside existing `cpu` / `cuda` handling in `resolve_device`.

## 0.1.0a0

- Added finance-native data contracts for persistent panels, ragged cross-sections, and
  portfolio sequences.
- Added latent-factor model families: `PCA`, `RP-PCA`, `IPCA`, and `CAE`.
- Added direct asset prediction with `SAEModel`.
- Added weight-native stochastic discount factor modeling with
  `StochasticDiscountFactorModel`.
- Added portfolio learning baselines and deep models.
- Added factor forecasters, asset mappers, and composable modeling pipelines.
- Added integration adapters for ML4T data schemas, backtest handoff, and prediction or weight
  surfaces.
- Added user guide, architecture documentation, and API reference with MkDocs.
