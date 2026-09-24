# Direct Asset Prediction

## Run and verify

Install `ml4t-models[deep]` and run the
[small CPU SAE example](https://github.com/ml4t/models/blob/main/examples/direct_asset_prediction.py).
It checks that fitting converges and that the signal matrix has one value per date and asset.
The two-epoch run verifies the contract, not predictive skill. Keep validation and test dates
separate for a real study. The [API reference](../api/index.md) describes `SAEConfig`,
`SAEModel`, and `AssetSignalResult`.

The [supervised autoencoder notebook](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/14_latent_factors/08_supervised_autoencoder.ipynb)
teaches a manual implementation with book data and longer PyTorch training; it does not call
`SAEModel`.

This family covers models that predict asset-level signals directly rather than first
estimating latent structure.

## SAEModel

`SAEModel` means **supervised autoencoder** in this library.

It does not mean:

- sparse autoencoder
- unsupervised autoencoder
- latent-factor model

The current implementation follows the supervised autoencoder pattern used in the Jane
Street competition lineage:

- encoder / bottleneck
- decoder
- auxiliary head
- main predictive head

The key architectural idea is not "autoencoding for its own sake." The bottleneck and
decoder regularize the representation that the predictive heads use. The reconstruction task
is there to improve the supervised signal, not to recover a structural latent-factor
decomposition.

## Why It Lives Outside Latent Factors

In the current design:

- `SAEModel` is a direct predictor
- it consumes `CrossSectionBatch`
- it emits `AssetSignalResult`

So the workflow is:

```text
cross-section batch -> supervised autoencoder -> asset signals
```

not:

```text
cross-section batch -> factor state -> factor forecaster -> beta × lambda
```

| Input contract | Native output | Assumption |
|---|---|---|
| `CrossSectionBatch` | `AssetSignalResult` | supervised targets are the object of interest |

## Checkpoints

`SAEModel` supports:

- `checkpoint_interval`
- `checkpoint_epochs`
- `default_checkpoint`

so you can evaluate intermediate training horizons explicitly.

## Example

```python
from ml4t.models import CrossSectionBatch, SAEConfig, SAEModel

model = SAEModel(SAEConfig(n_epochs=50, checkpoint_interval=5))
fit_summary = model.fit(train_batch, validation_batch=val_batch)
signals = model.predict(test_batch)
```

## Outputs

`predict()` returns:

- `AssetSignalResult`

with:

- `signal_values`
- timestamps
- asset IDs
- selected checkpoint metadata

The signal is not automatically calibrated as an expected return. Use the integration
helpers to convert it to long-format signal frames, then evaluate or transform it in the
diagnostic and backtest layers.

## When To Use It

Use `SAEModel` when you want:

- a direct supervised predictor
- a strong tabular deep-learning baseline for cross-sectional signals
- checkpoint-aware asset-level predictions

This is the right family when the modeling question is:

- "Can I learn a useful cross-sectional signal directly?"

Use latent-factor models instead when the question is:

- "Can I explain returns through exposures to a small latent factor system?"
