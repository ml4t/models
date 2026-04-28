# Direct Asset Prediction

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

The library originally explored treating `SAE` as another latent-factor model. That turns
out to be the wrong abstraction.

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

## When To Use It

Use `SAEModel` when you want:

- a direct supervised predictor
- a strong tabular deep-learning baseline for cross-sectional signals
- checkpoint-aware asset-level predictions

This is the right family when the modeling question is:

- "Can I learn a useful cross-sectional signal directly?"

Use latent-factor models instead when the question is:

- "Can I explain returns through exposures to a small latent factor system?"
