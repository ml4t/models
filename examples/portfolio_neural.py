from __future__ import annotations

import numpy as np

from ml4t.models import (
    DeepPortfolioConfig,
    DeepPortfolioModel,
    LSTMPortfolioConfig,
    LSTMPortfolioModel,
    PortfolioSequenceBatch,
)

rng = np.random.default_rng(11)
features = rng.normal(size=(3, 4, 3, 4))
returns = 0.03 * features[..., 0] - 0.01 * features[..., 1]
batch = PortfolioSequenceBatch(
    features=features,
    returns=returns,
    vol_scale=np.ones((3, 4, 3)),
    mask=np.ones((3, 4, 3), dtype=bool),
    asset_ids=("A", "B", "C"),
)

common = {
    "dropout": 0.0,
    "batch_size": 2,
    "max_iters": 2,
    "eval_every": 1,
    "checkpoint_every": 1,
    "default_checkpoint": 2,
    "seed": 7,
    "device": "cpu",
}
for label, model in (
    (
        "LSTM",
        LSTMPortfolioModel(LSTMPortfolioConfig(hidden_size=8, n_layers=1, **common)),
    ),
    (
        "DeepPortfolio",
        DeepPortfolioModel(
            DeepPortfolioConfig(
                d_model=8,
                n_heads=1,
                lstm_layers=1,
                temporal_mha_layers=1,
                cross_attention_heads=1,
                macro_gnn_heads=1,
                **common,
            )
        ),
    ),
):
    fit = model.fit(batch, validation_batch=batch)
    weights = model.predict(batch)
    assert fit.converged
    assert weights.weights.shape == (3, 4, 3)
    assert np.isfinite(weights.weights).all()
    print(f"{label} weights shape: {weights.weights.shape}")
