# Changelog

All notable changes to `ml4t-models` will be documented in this file.

## 0.1.0a5

- Added Apple Silicon MPS device resolution and seeding support for Torch-backed models.
- Added a non-skipping macOS arm64 CI smoke test that proves PyTorch MPS tensor execution.
- Updated the NumPy requirement to allow NumPy 2.5 releases.
- Fixed CUDA device normalization for whitespace or case variants such as `CUDA:0`.

## 0.1.0a4

- Aligned conditional autoencoder training with shuffled mini-batches, BatchNorm hidden layers,
  validation-best checkpoints, and per-member extraction.
- Aligned stochastic discount factor checkpoints with phase-local training, validation-best
  tracking, and legacy checkpoint-label compatibility.
- Kept IPCA factor extraction consistent with the final normalized loading matrix.

## 0.1.0a1

- Declared Python 3.14 support in package metadata.
- Extended the CI test matrix to cover Python 3.14.

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
