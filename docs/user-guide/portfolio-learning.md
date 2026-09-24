# Portfolio Learning

## Run and verify weights

Run the [linear CPU example](https://github.com/ml4t/models/blob/main/examples/portfolio_learning.py)
with the base install. It verifies `(windows, periods, assets)` output and that postprocessed
gross exposure stays at or below `0.8`. The sequence batch must retain the same asset IDs and
time order as the features and returns. LSTM and DeepPortfolio need `ml4t-models[deep]`; their
production training may need an accelerator and considerably more time. A small CPU smoke run
can check shapes and constraints, but not investment performance. Consult the
[API reference](../api/index.md) for each config and result contract.

The [neural smoke example](https://github.com/ml4t/models/blob/main/examples/portfolio_neural.py)
checks bounded LSTM and DeepPortfolio fits on CPU with two optimization steps. Use a separate
validation period and more training for any research comparison.

The book's [VLSTM](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/17_portfolio_construction/12_vlstm_portfolio.ipynb)
and [DeePM](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/17_portfolio_construction/13_deepm_regime_robust.ipynb)
notebooks teach related manual implementations. They do not call this library and are not exact
equivalents of these classes.

Portfolio models in `ml4t-models` learn weights directly.

They do not first estimate expected returns and then call a separate optimizer unless you
explicitly build that workflow yourself.

This family covers two related ideas:

- differentiable end-to-end portfolio objectives
- structured deep allocation architectures such as DeePM

## Family Overview

| Model | Input contract | Style | Native output | Main assumption |
|---|---|---|---|---|
| `LinearFeaturePortfolioModel` | `PortfolioSequenceBatch` | deterministic baseline | `PortfolioWeightsResult` | pooled feature scores can rank allocation desirability |
| `LSTMPortfolioModel` | `PortfolioSequenceBatch` | sequence baseline | `PortfolioWeightsResult` | recent feature paths contain allocation-relevant state |
| `DeepPortfolioModel` | `PortfolioSequenceBatch` | structured DeePM-style allocator | `PortfolioWeightsResult` | temporal, cross-sectional, and graph structure can improve direct allocations |

## Shared Contract

All portfolio models use:

- `PortfolioSequenceBatch`

and implement:

- `fit(batch, validation_batch=None)`
- `predict(batch, checkpoint=None)`

That contract is intentionally separate from latent-factor prediction contracts. The target
here is an allocation decision, not an expected-return vector.

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

The design borrows from the recent end-to-end portfolio-learning literature:

- sequence modeling for local path dependence
- attention for long-range or cross-asset interaction
- direct optimization of a risk-adjusted objective
- explicit handling of costs and turnover

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

These are not cosmetic training options. They shape the learned portfolio policy because the
loss is already an allocation objective with cost terms inside the loop.

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

Use it when you want to transform raw learned weights into a stricter target-weights frame
before handing them to `ml4t-backtest`.

Postprocessing is deliberately separate from model fitting. This lets the same trained
allocator be evaluated under different exposure caps, leverage normalization rules, or
turnover controls.
