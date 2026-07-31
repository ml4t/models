from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest
import torch

from ml4t.models import (
    AR1FactorForecaster,
    AR1ForecasterConfig,
    CAEConfig,
    CAEModel,
    CrossSectionBatch,
    DeepPortfolioConfig,
    DeepPortfolioModel,
    EWMABaseFactorForecaster,
    EWMABaseForecasterConfig,
    ExpandingMeanFactorForecaster,
    ExpandingMeanForecasterConfig,
    IPCAConfig,
    IPCAModel,
    LatentFactorState,
    LinearPortfolioConfig,
    LSTMPortfolioConfig,
    LSTMPortfolioModel,
    PCAConfig,
    PCAModel,
    PersistentPanelBatch,
    PortfolioSequenceBatch,
    RPPCAConfig,
    RPPCAModel,
    SAEConfig,
    SAEModel,
    StochasticDiscountFactorBetaNetworkHead,
    StochasticDiscountFactorConfig,
    StochasticDiscountFactorModel,
)
from ml4t.models.portfolio.linear import LinearFeaturePortfolioModel


def _cross_section() -> CrossSectionBatch:
    rng = np.random.default_rng(503)
    characteristics = rng.normal(size=(5, 3, 2))
    returns = 0.1 * characteristics[..., 0] + rng.normal(scale=0.01, size=(5, 3))
    return CrossSectionBatch(
        characteristics=characteristics,
        returns=returns,
        asset_ids=("A", "B", "C"),
    )


def _persistent_panel() -> PersistentPanelBatch:
    rng = np.random.default_rng(509)
    return PersistentPanelBatch(
        returns=rng.normal(size=(5, 3)),
        asset_ids=("A", "B", "C"),
    )


def _portfolio_batch() -> PortfolioSequenceBatch:
    rng = np.random.default_rng(521)
    return PortfolioSequenceBatch(
        features=rng.normal(size=(2, 3, 2, 2)),
        returns=rng.normal(scale=0.01, size=(2, 3, 2)),
        asset_ids=("A", "B"),
    )


def _tensor_dtypes(value: Any) -> Iterable[torch.dtype]:
    if torch.is_tensor(value):
        yield value.dtype
    elif isinstance(value, dict):
        for child in value.values():
            yield from _tensor_dtypes(child)
    elif isinstance(value, list | tuple):
        for child in value:
            yield from _tensor_dtypes(child)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_numpy_estimators_use_configured_dtype(dtype: str) -> None:
    expected = np.dtype(dtype)
    panel = _persistent_panel()
    cross_section = _cross_section()
    portfolio = _portfolio_batch()

    pca = PCAModel(PCAConfig(n_factors=1, dtype=dtype))
    pca.fit(panel)
    assert pca._loadings is not None and pca._loadings.dtype == expected

    rp_pca = RPPCAModel(RPPCAConfig(n_factors=1, dtype=dtype))
    rp_pca.fit(panel)
    assert rp_pca._factor_weights is not None and rp_pca._factor_weights.dtype == expected

    ipca = IPCAModel(IPCAConfig(n_factors=1, max_iter=2, dtype=dtype))
    ipca.fit(cross_section)
    assert ipca.gamma.dtype == expected
    assert ipca.train_factor_returns.dtype == expected

    linear = LinearFeaturePortfolioModel(LinearPortfolioConfig(dtype=dtype))
    linear.fit(portfolio)
    assert linear._coefficients is not None and linear._coefficients.dtype == expected


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_forecasters_use_configured_dtype(dtype: str) -> None:
    expected = np.dtype(dtype)
    state = LatentFactorState(
        asset_betas=np.ones((4, 2, 1)),
        factor_returns=np.arange(4, dtype=np.float64)[:, None],
    )

    ar = AR1FactorForecaster(AR1ForecasterConfig(dtype=dtype))
    ar.fit(state)
    assert ar._intercepts is not None and ar._intercepts.dtype == expected

    ewma = EWMABaseFactorForecaster(EWMABaseForecasterConfig(dtype=dtype))
    ewma.fit(state)
    assert ewma._ewma_level is not None and ewma._ewma_level.dtype == expected

    mean = ExpandingMeanFactorForecaster(ExpandingMeanForecasterConfig(dtype=dtype))
    mean.fit(state)
    assert mean._mean_factor_premium is not None
    assert mean._mean_factor_premium.dtype == expected


@pytest.mark.parametrize(
    ("dtype", "expected"), [("float32", torch.float32), ("float64", torch.float64)]
)
def test_neural_estimators_use_configured_dtype(dtype: str, expected: torch.dtype) -> None:
    cross_section = _cross_section()
    portfolio = _portfolio_batch()

    cae = CAEModel(
        CAEConfig(
            n_factors=1,
            hidden_units=(),
            n_epochs=1,
            checkpoint_interval=1,
            batch_size=32,
            dtype=dtype,
        )
    )
    cae.fit(cross_section)

    sae = SAEModel(
        SAEConfig(
            bottleneck_dim=2,
            aux_hidden_dim=2,
            main_hidden_units=(4, 4, 4, 4),
            dropout_rates=(0.0,) * 8,
            n_epochs=1,
            checkpoint_interval=1,
            batch_size=32,
            dtype=dtype,
        )
    )
    sae.fit(cross_section)

    sdf_config = StochasticDiscountFactorConfig(
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
        dtype=dtype,
    )
    sdf = StochasticDiscountFactorModel(sdf_config)
    sdf.fit(cross_section)
    sdf_state = sdf.extract(cross_section)
    beta = StochasticDiscountFactorBetaNetworkHead(sdf_config)
    beta.fit(sdf_state, cross_section)

    lstm = LSTMPortfolioModel(
        LSTMPortfolioConfig(
            hidden_size=4,
            max_iters=1,
            eval_every=1,
            checkpoint_every=1,
            dtype=dtype,
        )
    )
    lstm.fit(portfolio)
    deep = DeepPortfolioModel(
        DeepPortfolioConfig(
            d_model=4,
            n_heads=1,
            cross_attention_heads=1,
            macro_gnn_heads=1,
            max_iters=1,
            eval_every=1,
            checkpoint_every=1,
            dtype=dtype,
        )
    )
    deep.fit(portfolio)

    for checkpoints in (
        cae._checkpoint_states,
        sae._checkpoint_states,
        sdf._checkpoint_states,
        beta._checkpoint_states,
        lstm._checkpoint_states,
        deep._checkpoint_states,
    ):
        tensor_dtypes = tuple(_tensor_dtypes(checkpoints))
        assert tensor_dtypes
        assert set(tensor_dtypes) <= {expected, torch.int64}
