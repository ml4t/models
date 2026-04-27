from __future__ import annotations

import numpy as np
import pytest

from ml4t.models import (
    DeepPortfolioConfig,
    DeepPortfolioModel,
    LinearFeaturePortfolioModel,
    LinearPortfolioConfig,
    LSTMPortfolioConfig,
    LSTMPortfolioModel,
    PortfolioAllocationPipeline,
    PortfolioSequenceBatch,
    WeightConstraintPostprocessor,
)

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


def test_lstm_portfolio_trains_checkpointed_policy_and_predicts_weights() -> None:
    rng = np.random.default_rng(19)
    batch_size = 4
    n_periods = 5
    n_assets = 3
    n_features = 4

    features = rng.normal(size=(batch_size, n_periods, n_assets, n_features))
    signal = 0.025 * features[..., 0] - 0.015 * features[..., 2]
    returns = signal + 0.005 * rng.normal(size=signal.shape)
    vol_scale = np.ones((batch_size, n_periods, n_assets), dtype=np.float64)
    mask = np.ones((batch_size, n_periods, n_assets), dtype=bool)
    group_ids = np.array([0, 1, 0], dtype=np.int64)
    costs = np.array([0.001, 0.0015, 0.002], dtype=np.float64)

    train = PortfolioSequenceBatch(
        features=features,
        returns=returns,
        vol_scale=vol_scale,
        mask=mask,
        group_ids=group_ids,
        costs=costs,
        asset_ids=("A", "B", "C"),
    )

    model = LSTMPortfolioModel(
        LSTMPortfolioConfig(
            hidden_size=8,
            n_layers=1,
            dropout=0.0,
            asset_embedding_dim=4,
            group_embedding_dim=2,
            vvsn_hidden_dim=8,
            batch_size=2,
            learning_rate=1e-3,
            max_iters=3,
            eval_every=1,
            checkpoint_every=1,
            early_stopping_patience=10,
            early_stopping_burn_in_iters=3,
            default_checkpoint=2,
            seed=5,
            device="cpu",
        )
    )

    fit_summary = model.fit(train, validation_batch=train)
    weights = model.predict(train)
    checkpoint_weights = model.predict(train, checkpoint=3)

    assert fit_summary.converged
    assert model.available_checkpoints == (1, 2, 3)
    assert weights.checkpoint_step == 2
    assert weights.weights.shape == (batch_size, n_periods, n_assets)
    assert checkpoint_weights.checkpoint_step == 3
    assert checkpoint_weights.weights.shape == (batch_size, n_periods, n_assets)
    assert np.isfinite(weights.weights).all()


def test_linear_feature_portfolio_fits_and_predicts() -> None:
    rng = np.random.default_rng(31)
    batch_size = 3
    n_periods = 4
    n_assets = 5
    n_features = 3

    features = rng.normal(size=(batch_size, n_periods, n_assets, n_features))
    returns = 0.04 * features[..., 0] - 0.02 * features[..., 1]
    returns += 0.005 * rng.normal(size=(batch_size, n_periods, n_assets))
    vol_scale = np.ones((batch_size, n_periods, n_assets), dtype=np.float64)
    mask = np.ones((batch_size, n_periods, n_assets), dtype=bool)

    batch = PortfolioSequenceBatch(
        features=features,
        returns=returns,
        vol_scale=vol_scale,
        mask=mask,
        timestamps=("t1", "t2", "t3", "t4"),
        asset_ids=tuple(f"A{i}" for i in range(n_assets)),
    )

    model = LinearFeaturePortfolioModel(
        LinearPortfolioConfig(
            ridge_alpha=1e-4,
            fit_intercept=True,
            gross_exposure=1.0,
        )
    )
    fit_summary = model.fit(batch)
    prediction = model.predict(batch)

    assert fit_summary.converged
    assert model.available_checkpoints == (1,)
    assert prediction.checkpoint_step == 1
    assert prediction.weights.shape == (batch_size, n_periods, n_assets)
    gross = np.abs(prediction.weights).sum(axis=-1)
    assert np.allclose(gross, 1.0)


def test_portfolio_allocation_pipeline_applies_weight_constraints() -> None:
    rng = np.random.default_rng(41)
    batch_size = 2
    n_periods = 4
    n_assets = 4
    n_features = 3

    features = rng.normal(size=(batch_size, n_periods, n_assets, n_features))
    returns = 0.05 * features[..., 0] + 0.01 * rng.normal(size=(batch_size, n_periods, n_assets))
    vol_scale = np.ones((batch_size, n_periods, n_assets), dtype=np.float64)
    mask = np.ones((batch_size, n_periods, n_assets), dtype=bool)
    prev_weights = np.zeros((batch_size, n_assets), dtype=np.float64)

    batch = PortfolioSequenceBatch(
        features=features,
        returns=returns,
        vol_scale=vol_scale,
        mask=mask,
        prev_weights=prev_weights,
        asset_ids=tuple(f"A{i}" for i in range(n_assets)),
    )

    pipeline = PortfolioAllocationPipeline(
        LinearFeaturePortfolioModel(
            LinearPortfolioConfig(
                ridge_alpha=1e-4,
                gross_exposure=1.0,
                max_abs_weight=None,
            )
        ),
        postprocessors=(
            WeightConstraintPostprocessor(
                gross_exposure=0.8,
                max_abs_weight=0.35,
                turnover_limit=0.6,
            ),
        ),
    )
    fit_result = pipeline.fit(batch)
    prediction = pipeline.predict(batch)

    assert fit_result.model_fit.converged
    assert prediction.raw_weights.weights.shape == (batch_size, n_periods, n_assets)
    assert prediction.processed_weights.weights.shape == (batch_size, n_periods, n_assets)
    assert not np.allclose(prediction.raw_weights.weights, prediction.processed_weights.weights)
    processed_gross = np.abs(prediction.processed_weights.weights).sum(axis=-1)
    assert np.all(processed_gross <= 0.8 + 1e-8)
    assert np.all(np.abs(prediction.processed_weights.weights) <= 0.35 + 1e-8)
