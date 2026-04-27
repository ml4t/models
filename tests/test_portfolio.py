from __future__ import annotations

import numpy as np
import pytest

from ml4t.models import DeepPortfolioConfig, DeepPortfolioModel, PortfolioSequenceBatch

pytest.importorskip("torch")


def test_deep_portfolio_trains_checkpointed_policy_and_predicts_weights() -> None:
    rng = np.random.default_rng(11)
    batch_size = 4
    n_periods = 5
    n_assets = 3
    n_features = 4

    features = rng.normal(size=(batch_size, n_periods, n_assets, n_features))
    signal = 0.03 * features[..., 0] - 0.01 * features[..., 1]
    returns = signal + 0.005 * rng.normal(size=signal.shape)
    vol_scale = np.ones((batch_size, n_periods, n_assets), dtype=np.float64)
    mask = np.ones((batch_size, n_periods, n_assets), dtype=bool)
    group_ids = np.array([0, 1, 0], dtype=np.int64)
    costs = np.array([0.001, 0.002, 0.0015], dtype=np.float64)
    adjacency_mask = np.zeros((n_assets, n_assets), dtype=bool)

    train = PortfolioSequenceBatch(
        features=features,
        returns=returns,
        vol_scale=vol_scale,
        mask=mask,
        group_ids=group_ids,
        costs=costs,
        adjacency_mask=adjacency_mask,
        asset_ids=("A", "B", "C"),
    )

    model = DeepPortfolioModel(
        DeepPortfolioConfig(
            d_model=8,
            n_heads=1,
            dropout=0.0,
            lstm_layers=1,
            temporal_mha_layers=1,
            cross_attention_heads=1,
            cross_attention_lag=1,
            macro_gnn_heads=1,
            asset_embedding_dim=4,
            group_embedding_dim=2,
            vvsn_hidden_dim=8,
            adapter_hidden_mult=2,
            batch_size=2,
            learning_rate=1e-3,
            max_iters=3,
            eval_every=1,
            checkpoint_every=1,
            early_stopping_patience=10,
            early_stopping_burn_in_iters=3,
            default_checkpoint=2,
            seed=7,
            device="cpu",
        )
    )

    fit_summary = model.fit(train, validation_batch=train)
    weights = model.predict(train)
    checkpoint_weights = model.predict(train, checkpoint=1)

    assert fit_summary.converged
    assert model.available_checkpoints == (1, 2, 3)
    assert weights.checkpoint_step == 2
    assert weights.weights.shape == (batch_size, n_periods, n_assets)
    assert checkpoint_weights.checkpoint_step == 1
    assert checkpoint_weights.weights.shape == (batch_size, n_periods, n_assets)
    assert np.isfinite(weights.weights).all()
