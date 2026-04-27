from __future__ import annotations

import numpy as np
import pytest

from ml4t.models import AssetSignalResult, CrossSectionBatch, SAEConfig, SAEModel

pytest.importorskip("torch")


def test_sae_predicts_checkpointed_direct_signals() -> None:
    rng = np.random.default_rng(17)
    n_periods = 8
    n_assets = 9
    n_features = 4

    characteristics = rng.normal(size=(n_periods, n_assets, n_features))
    signal = (
        0.6 * characteristics[..., 0]
        - 0.3 * characteristics[..., 1]
        + 0.2 * characteristics[..., 2]
    )
    returns = signal + 0.05 * rng.normal(size=signal.shape)

    train = CrossSectionBatch(
        characteristics=characteristics,
        returns=returns,
        timestamps=tuple(f"2024-{month:02d}" for month in range(1, n_periods + 1)),
    )
    future = CrossSectionBatch(
        characteristics=rng.normal(size=(3, n_assets, n_features)),
        timestamps=("2024-09", "2024-10", "2024-11"),
    )

    model = SAEModel(
        SAEConfig(
            bottleneck_dim=8,
            aux_hidden_dim=8,
            main_hidden_units=(16, 12, 12, 8),
            dropout_rates=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            n_epochs=10,
            checkpoint_interval=5,
            batch_size=16,
            lr=1e-3,
        )
    )
    fit = model.fit(train, validation_batch=train)
    train_signal = model.predict(train, checkpoint=5)
    future_signal = model.predict(future, checkpoint=10)

    assert fit.converged
    assert model.available_checkpoints == (5, 10)
    assert fit.best_epoch is not None
    assert isinstance(train_signal, AssetSignalResult)
    assert train_signal.signal_values.shape == (n_periods, n_assets)
    assert future_signal.signal_values.shape == (3, n_assets)
    assert np.isfinite(train_signal.signal_values).any()


def test_sae_requires_returns_for_training() -> None:
    batch = CrossSectionBatch(characteristics=np.zeros((2, 3, 2), dtype=np.float64))
    model = SAEModel(SAEConfig())

    with pytest.raises(ValueError):
        model.fit(batch)


def test_sae_classification_returns_probabilities() -> None:
    rng = np.random.default_rng(5)
    characteristics = rng.normal(size=(6, 7, 3))
    raw_scores = characteristics[..., 0] - characteristics[..., 1]
    labels = (raw_scores > 0.0).astype(np.float64)

    batch = CrossSectionBatch(
        characteristics=characteristics,
        returns=labels,
        timestamps=tuple(f"2024-{month:02d}" for month in range(1, 7)),
    )
    model = SAEModel(
        SAEConfig(
            task_type="classification",
            bottleneck_dim=6,
            aux_hidden_dim=6,
            main_hidden_units=(12, 12, 8, 8),
            dropout_rates=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            n_epochs=5,
            checkpoint_interval=5,
        )
    )
    model.fit(batch)
    signal = model.predict(batch, checkpoint=5)

    assert np.nanmin(signal.signal_values) >= 0.0
    assert np.nanmax(signal.signal_values) <= 1.0
