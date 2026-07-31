from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
import torch

from ml4t.models import (
    CAEConfig,
    CAEModel,
    CrossSectionBatch,
    DeepPortfolioConfig,
    DeepPortfolioModel,
    LSTMPortfolioConfig,
    LSTMPortfolioModel,
    PortfolioSequenceBatch,
    SAEConfig,
    SAEModel,
    StochasticDiscountFactorBetaNetworkHead,
    StochasticDiscountFactorConfig,
    StochasticDiscountFactorModel,
)


def _cross_section_batch() -> CrossSectionBatch:
    rng = np.random.default_rng(101)
    return CrossSectionBatch(
        characteristics=rng.normal(size=(5, 4, 3)),
        returns=rng.normal(scale=0.02, size=(5, 4)),
        asset_ids=("A", "B", "C", "D"),
    )


def _portfolio_batch() -> PortfolioSequenceBatch:
    rng = np.random.default_rng(103)
    return PortfolioSequenceBatch(
        features=rng.normal(size=(2, 3, 2, 2)),
        returns=rng.normal(scale=0.02, size=(2, 3, 2)),
        asset_ids=("A", "B"),
    )


def _cae_config() -> CAEConfig:
    return CAEConfig(
        n_factors=1,
        hidden_units=(),
        n_epochs=1,
        checkpoint_interval=1,
        batch_size=32,
    )


def _sae_config() -> SAEConfig:
    return SAEConfig(
        bottleneck_dim=2,
        aux_hidden_dim=2,
        main_hidden_units=(4, 4, 4, 4),
        dropout_rates=(0.0,) * 8,
        n_epochs=1,
        checkpoint_interval=1,
        batch_size=32,
    )


def _sdf_config() -> StochasticDiscountFactorConfig:
    return StochasticDiscountFactorConfig(
        state_dim_sdf=2,
        state_dim_moment=2,
        hidden_dim=4,
        n_instruments=2,
        n_epochs_unc=1,
        n_epochs_moment=1,
        n_epochs_cond=1,
        checkpoint_interval=1,
        beta_state_dim=2,
        beta_hidden_dim=4,
        beta_n_epochs=1,
        beta_checkpoint_interval=1,
        dropout=0.0,
    )


def _raise_checkpoint_failure(*args: object, **kwargs: object) -> None:
    del args, kwargs
    raise RuntimeError("injected checkpoint failure")


def _assert_tree_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_equal(actual[key], expected[key])
    elif torch.is_tensor(expected):
        assert torch.is_tensor(actual)
        assert torch.equal(actual, expected)
    elif isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        assert isinstance(actual, Sequence)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_tree_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _assert_numpy_rng_unchanged(fit: Callable[[], object]) -> None:
    original = np.random.get_state()
    try:
        np.random.seed(90210)
        expected = np.random.get_state()
        fit()
        actual = np.random.get_state()
        assert actual[0] == expected[0]
        assert np.array_equal(actual[1], expected[1])
        assert actual[2:] == expected[2:]
    finally:
        np.random.set_state(original)


def test_cae_failed_refit_restores_prior_state(monkeypatch: pytest.MonkeyPatch) -> None:
    import ml4t.models.latent_factors.cae as cae_module

    batch = _cross_section_batch()
    model = CAEModel(_cae_config())
    model.fit(batch)
    expected_prediction = model.extract(batch).asset_betas.copy()
    expected_checkpoints = deepcopy(model._checkpoint_states)
    expected_metadata = (
        model._asset_ids,
        model._n_characteristics,
        model._n_instruments,
        model._history,
        model._fit_default_checkpoint,
        model.is_fitted,
    )
    monkeypatch.setattr(cae_module, "_cpu_state_dict", _raise_checkpoint_failure)

    with pytest.raises(RuntimeError, match="injected checkpoint failure"):
        model.fit(batch)

    _assert_tree_equal(model._checkpoint_states, expected_checkpoints)
    assert (
        model._asset_ids,
        model._n_characteristics,
        model._n_instruments,
        model._history,
        model._fit_default_checkpoint,
        model.is_fitted,
    ) == expected_metadata
    assert np.array_equal(model.extract(batch).asset_betas, expected_prediction, equal_nan=True)


def test_sae_failed_refit_restores_prior_state(monkeypatch: pytest.MonkeyPatch) -> None:
    import ml4t.models.asset_prediction.sae as sae_module

    batch = _cross_section_batch()
    model = SAEModel(_sae_config())
    model.fit(batch)
    expected_prediction = model.predict(batch).signal_values.copy()
    expected_checkpoints = deepcopy(model._checkpoint_states)
    expected_metadata = (
        model._n_features,
        model._asset_ids,
        model._history,
        model.is_fitted,
    )
    monkeypatch.setattr(sae_module, "_cpu_state_dict", _raise_checkpoint_failure)

    with pytest.raises(RuntimeError, match="injected checkpoint failure"):
        model.fit(batch)

    _assert_tree_equal(model._checkpoint_states, expected_checkpoints)
    assert (
        model._n_features,
        model._asset_ids,
        model._history,
        model.is_fitted,
    ) == expected_metadata
    assert np.array_equal(model.predict(batch).signal_values, expected_prediction, equal_nan=True)


def test_sdf_failed_refit_restores_prior_state(monkeypatch: pytest.MonkeyPatch) -> None:
    import ml4t.models.stochastic_discount_factor.model as sdf_module

    batch = _cross_section_batch()
    model = StochasticDiscountFactorModel(_sdf_config())
    model.fit(batch)
    expected_prediction = model.extract(batch, checkpoint=("conditional", 1)).asset_weights.copy()
    expected_checkpoints = deepcopy(model._checkpoint_states)
    expected_metadata = (
        model._asset_ids,
        model._n_characteristics,
        model._n_context_features,
        model._history,
        model.is_fitted,
    )
    monkeypatch.setattr(sdf_module, "_capture_state", _raise_checkpoint_failure)

    with pytest.raises(RuntimeError, match="injected checkpoint failure"):
        model.fit(batch)

    _assert_tree_equal(model._checkpoint_states, expected_checkpoints)
    assert (
        model._asset_ids,
        model._n_characteristics,
        model._n_context_features,
        model._history,
        model.is_fitted,
    ) == expected_metadata
    assert np.array_equal(
        model.extract(batch, checkpoint=("conditional", 1)).asset_weights,
        expected_prediction,
        equal_nan=True,
    )


def test_beta_head_failed_refit_restores_prior_state(monkeypatch: pytest.MonkeyPatch) -> None:
    import ml4t.models.stochastic_discount_factor.mapper as mapper_module

    batch = _cross_section_batch()
    sdf = StochasticDiscountFactorModel(_sdf_config())
    sdf.fit(batch)
    state = sdf.extract(batch, checkpoint=("conditional", 1))
    model = StochasticDiscountFactorBetaNetworkHead(_sdf_config())
    model.fit(state, batch)
    expected_prediction = model.predict(batch, checkpoint=1).signal_values.copy()
    expected_checkpoints = deepcopy(model._checkpoint_states)
    expected_metadata = (
        model._n_asset_features,
        model._n_context_features,
        model._asset_ids,
        model._f_hat_scale,
        model._history,
    )
    monkeypatch.setattr(mapper_module, "_cpu_state_dict", _raise_checkpoint_failure)

    with pytest.raises(RuntimeError, match="injected checkpoint failure"):
        model.fit(state, batch)

    _assert_tree_equal(model._checkpoint_states, expected_checkpoints)
    assert (
        model._n_asset_features,
        model._n_context_features,
        model._asset_ids,
        model._f_hat_scale,
        model._history,
    ) == expected_metadata
    assert np.array_equal(
        model.predict(batch, checkpoint=1).signal_values,
        expected_prediction,
        equal_nan=True,
    )


def test_neural_fits_do_not_modify_numpy_global_rng() -> None:
    cross_section = _cross_section_batch()
    portfolio = _portfolio_batch()

    _assert_numpy_rng_unchanged(lambda: CAEModel(_cae_config()).fit(cross_section))
    _assert_numpy_rng_unchanged(lambda: SAEModel(_sae_config()).fit(cross_section))

    sdf = StochasticDiscountFactorModel(_sdf_config())
    _assert_numpy_rng_unchanged(lambda: sdf.fit(cross_section))
    state = sdf.extract(cross_section, checkpoint=("conditional", 1))
    _assert_numpy_rng_unchanged(
        lambda: StochasticDiscountFactorBetaNetworkHead(_sdf_config()).fit(state, cross_section)
    )

    _assert_numpy_rng_unchanged(
        lambda: LSTMPortfolioModel(
            LSTMPortfolioConfig(
                hidden_size=4,
                max_iters=1,
                eval_every=1,
                checkpoint_every=1,
            )
        ).fit(portfolio)
    )
    _assert_numpy_rng_unchanged(
        lambda: DeepPortfolioModel(
            DeepPortfolioConfig(
                d_model=4,
                n_heads=1,
                cross_attention_heads=1,
                macro_gnn_heads=1,
                max_iters=1,
                eval_every=1,
                checkpoint_every=1,
            )
        ).fit(portfolio)
    )
