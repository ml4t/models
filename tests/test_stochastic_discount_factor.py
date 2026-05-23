from __future__ import annotations

import numpy as np
import pytest

from ml4t.models import (
    AssetSignalResult,
    CrossSectionBatch,
    LinearStochasticDiscountFactorReturnMapper,
    StochasticDiscountFactorBetaNetworkHead,
    StochasticDiscountFactorConfig,
    StochasticDiscountFactorModel,
)

pytest.importorskip("torch")


def test_stochastic_discount_factor_extracts_weight_state_and_linear_mapping() -> None:
    rng = np.random.default_rng(23)
    n_periods = 8
    n_assets = 6
    n_features = 3
    n_context = 2

    characteristics = rng.normal(size=(n_periods, n_assets, n_features))
    context_features = rng.normal(size=(n_periods, n_context))
    signal = 0.4 * characteristics[..., 0] - 0.2 * characteristics[..., 1]
    signal += 0.1 * context_features[:, 0][:, None]
    returns = 0.05 * signal + 0.01 * rng.normal(size=signal.shape)

    train = CrossSectionBatch(
        characteristics=characteristics,
        returns=returns,
        context_features=context_features,
        timestamps=tuple(f"2024-{month:02d}" for month in range(1, n_periods + 1)),
        asset_ids=tuple(f"A{i}" for i in range(n_assets)),
    )
    future = CrossSectionBatch(
        characteristics=rng.normal(size=(3, n_assets, n_features)),
        context_features=rng.normal(size=(3, n_context)),
        timestamps=("2024-09", "2024-10", "2024-11"),
        asset_ids=tuple(f"A{i}" for i in range(n_assets)),
    )

    model = StochasticDiscountFactorModel(
        StochasticDiscountFactorConfig(
            state_dim_sdf=2,
            state_dim_moment=4,
            hidden_dim=8,
            n_instruments=3,
            n_epochs_unc=4,
            n_epochs_moment=2,
            n_epochs_cond=4,
            checkpoint_interval=2,
            lr=1e-3,
            dropout=0.0,
        )
    )
    fit = model.fit(train)
    train_state = model.extract(train, checkpoint=("unconditional", 2))
    future_state = model.extract(future, checkpoint=("conditional", 4))

    mapper = LinearStochasticDiscountFactorReturnMapper()
    mapper_fit = mapper.fit(train_state, train)
    forecast = mapper.predict(future_state)

    assert fit.converged
    assert model.available_checkpoints == (
        ("unconditional", 2),
        ("unconditional", 4),
        ("conditional", 2),
        ("conditional", 4),
    )
    assert train_state.checkpoint_epoch == ("unconditional", 2)
    assert train_state.sdf_values is not None
    assert train_state.asset_weights.shape == (n_periods, n_assets)
    assert future_state.checkpoint_epoch == ("conditional", 4)
    assert future_state.sdf_values is None
    assert future_state.asset_weights.shape == (3, n_assets)
    assert mapper_fit.converged
    assert forecast.expected_returns.shape == (3, n_assets)


def test_stochastic_discount_factor_beta_head_returns_signals() -> None:
    rng = np.random.default_rng(11)
    n_periods = 8
    n_assets = 5
    n_features = 3
    n_context = 2
    characteristics = rng.normal(size=(n_periods, n_assets, n_features))
    context_features = rng.normal(size=(n_periods, n_context))
    returns = (
        0.04 * characteristics[..., 0]
        - 0.03 * characteristics[..., 1]
        + 0.02 * context_features[:, 0][:, None]
    )
    returns += 0.01 * rng.normal(size=returns.shape)

    train = CrossSectionBatch(
        characteristics=characteristics,
        returns=returns,
        context_features=context_features,
        timestamps=tuple(f"2024-{month:02d}" for month in range(1, n_periods + 1)),
        asset_ids=tuple(f"A{i}" for i in range(n_assets)),
    )
    model = StochasticDiscountFactorModel(
        StochasticDiscountFactorConfig(
            state_dim_sdf=2,
            state_dim_moment=4,
            hidden_dim=8,
            n_instruments=3,
            n_epochs_unc=4,
            n_epochs_moment=2,
            n_epochs_cond=4,
            checkpoint_interval=2,
            beta_n_epochs=6,
            beta_checkpoint_interval=3,
            lr=1e-3,
            beta_lr=1e-3,
            dropout=0.0,
        )
    )
    model.fit(train)
    train_state = model.extract(train, checkpoint=("conditional", 4))

    head = StochasticDiscountFactorBetaNetworkHead(model.config)
    fit = head.fit(train_state, train, validation_state=train_state, validation_batch=train)
    signal = head.predict(train, checkpoint=6)

    assert fit.converged
    assert head.available_checkpoints == (3, 6)
    assert isinstance(signal, AssetSignalResult)
    assert signal.signal_values.shape == (n_periods, n_assets)
    assert signal.metadata["signal_type"] == "stochastic_discount_factor_beta"
    assert np.isfinite(signal.signal_values).any()


def _sdf_batch(
    seed: int, n_periods: int = 8, n_assets: int = 6, n_features: int = 3
) -> CrossSectionBatch:
    rng = np.random.default_rng(seed)
    characteristics = rng.normal(size=(n_periods, n_assets, n_features))
    signal = 0.4 * characteristics[..., 0] - 0.2 * characteristics[..., 1]
    returns = 0.05 * signal + 0.01 * rng.normal(size=signal.shape)
    return CrossSectionBatch(
        characteristics=characteristics,
        returns=returns,
        timestamps=tuple(f"2024-{i:02d}" for i in range(1, n_periods + 1)),
        asset_ids=tuple(f"A{i}" for i in range(n_assets)),
    )


def test_phase_local_burn_in_resets_between_phases() -> None:
    """Burn-in is phase-local: early epochs of BOTH phases are skipped (CPZ ignoreEpoch)."""
    model = StochasticDiscountFactorModel(
        StochasticDiscountFactorConfig(
            state_dim_sdf=2,
            state_dim_moment=4,
            hidden_dim=8,
            n_instruments=3,
            n_epochs_unc=5,
            n_epochs_moment=2,
            n_epochs_cond=5,
            checkpoint_interval=1,
            burn_in_epochs=3,
            dropout=0.0,
        )
    )
    model.fit(_sdf_batch(seed=1))
    # Each SDF training phase uses its own epoch index for burn-in and checkpoints.
    assert model.available_checkpoints == (
        ("unconditional", 4),
        ("unconditional", 5),
        ("conditional", 4),
        ("conditional", 5),
    )
    assert ("conditional", 1) not in model.available_checkpoints
    assert ("conditional", 2) not in model.available_checkpoints
    assert ("conditional", 3) not in model.available_checkpoints


def test_validation_best_checkpoints_tracked() -> None:
    """fit(validation_batch=...) stores CPZ best-by-val-loss and best-by-val-sharpe per phase."""
    from ml4t.models.stochastic_discount_factor.model import (
        VAL_BEST_LOSS_CONDITIONAL,
        VAL_BEST_LOSS_UNCONDITIONAL,
        VAL_BEST_SHARPE_CONDITIONAL,
        VAL_BEST_SHARPE_UNCONDITIONAL,
    )

    model = StochasticDiscountFactorModel(
        StochasticDiscountFactorConfig(
            state_dim_sdf=2,
            state_dim_moment=4,
            hidden_dim=8,
            n_instruments=3,
            n_epochs_unc=6,
            n_epochs_moment=2,
            n_epochs_cond=6,
            checkpoint_interval=2,
            dropout=0.0,
        )
    )
    train = _sdf_batch(seed=2)
    val = _sdf_batch(seed=3)
    model.fit(train, validation_batch=val)

    keys = model.available_checkpoints
    for sentinel in (
        VAL_BEST_LOSS_UNCONDITIONAL,
        VAL_BEST_SHARPE_UNCONDITIONAL,
        VAL_BEST_LOSS_CONDITIONAL,
        VAL_BEST_SHARPE_CONDITIONAL,
    ):
        assert sentinel in keys

    # The val-sharpe-best conditional checkpoint is extractable and finite.
    state = model.extract(train, checkpoint=VAL_BEST_SHARPE_CONDITIONAL)
    assert state.checkpoint_epoch == VAL_BEST_SHARPE_CONDITIONAL
    assert np.isfinite(state.asset_weights[np.isfinite(state.asset_weights)]).all()

    # Default extraction (no checkpoint) still selects a positive epoch, not a sentinel.
    default_state = model.extract(train)
    assert default_state.checkpoint_epoch == ("conditional", 6)


def test_validation_batch_requires_returns() -> None:
    model = StochasticDiscountFactorModel(
        StochasticDiscountFactorConfig(n_epochs_unc=2, n_epochs_moment=1, n_epochs_cond=2)
    )
    train = _sdf_batch(seed=4)
    val_no_returns = CrossSectionBatch(
        characteristics=np.random.default_rng(5).normal(size=(8, 6, 3)),
        timestamps=tuple(f"2024-{i:02d}" for i in range(1, 9)),
        asset_ids=tuple(f"A{i}" for i in range(6)),
    )
    with pytest.raises(ValueError, match="validation_batch requires returns"):
        model.fit(train, validation_batch=val_no_returns)


def test_stochastic_discount_factor_rejects_non_weight_native_output_mode() -> None:
    batch = CrossSectionBatch(
        characteristics=np.zeros((2, 3, 2), dtype=np.float64),
        returns=np.zeros((2, 3), dtype=np.float64),
    )
    model = StochasticDiscountFactorModel(
        StochasticDiscountFactorConfig(output_mode="expected_returns")
    )

    with pytest.raises(ValueError):
        model.fit(batch)
