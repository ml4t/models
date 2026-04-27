# Portfolio Learning

Portfolio models in `ml4t-models` learn weights directly.

They do not first estimate expected returns and then call a separate optimizer unless you explicitly build that workflow yourself.

## Family Overview

| Model | Style | Native output |
|---|---|---|
| `LinearFeaturePortfolioModel` | deterministic baseline | `PortfolioWeightsResult` |
| `LSTMPortfolioModel` | sequence baseline | `PortfolioWeightsResult` |
| `DeepPortfolioModel` | structured DeePM-style allocator | `PortfolioWeightsResult` |

## Shared Contract

All portfolio models use:

- `PortfolioSequenceBatch`

and implement:

- `fit(batch, validation_batch=None)`
- `predict(batch, checkpoint=None)`

## LinearFeaturePortfolioModel

This is the simplest allocation baseline.

It:

- fits pooled linear feature scores
- maps scores to cross-sectional weights
- normalizes those weights under configurable exposure constraints

Good uses:

- sanity checks
- deterministic regression baselines
- quick integration tests

## LSTMPortfolioModel

This model adds sequence modeling while staying relatively simple.

Architecture elements:

- context encoder
- feature modulation
- variable selection
- LSTM backbone
- bounded output head

Use it when you want:

- a sequence-based baseline
- checkpointed end-to-end portfolio training
- a lighter alternative to the full DeePM-style architecture

## DeepPortfolioModel

`DeepPortfolioModel` is the structured portfolio learner in the library.

Current architecture includes:

- static context encoding
- feature modulation
- variable selection
- LSTM temporal backbone
- temporal self-attention blocks
- cross-sectional attention
- optional macro-graph attention

This is a DeePM-style implementation rather than a generic transformer allocator.

## Shared Training Features

Portfolio models support:

- checkpointed training
- validation-aware selection
- turnover-aware objective terms
- cost inputs
- group IDs
- adjacency masks for graph structure

Common config controls include:

- `turnover_penalty`
- `gamma_cost`
- `checkpoint_every`
- `checkpoint_steps`
- `default_checkpoint`
- `early_stopping_patience`

## Pipeline Layer

`PortfolioAllocationPipeline` wraps:

- one portfolio model
- zero or more `PortfolioPostprocessor` hooks

This is the correct place for:

- exposure clipping
- turnover caps
- normalization tweaks

without mixing those concerns into the model architecture itself.

## Postprocessing

Current helper:

- `WeightConstraintPostprocessor`

Use it when you want to transform raw learned weights into a stricter target-weight surface before handing them to `ml4t-backtest`.

