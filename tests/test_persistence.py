from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ml4t.models import (
    AR1FactorForecaster,
    CAEConfig,
    CAEModel,
    CrossSectionBatch,
    DeepPortfolioConfig,
    DeepPortfolioModel,
    EWMABaseFactorForecaster,
    ExpandingMeanFactorForecaster,
    IPCAConfig,
    IPCAModel,
    LatentFactorState,
    LinearFeaturePortfolioModel,
    LinearPortfolioConfig,
    LinearStochasticDiscountFactorReturnMapper,
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


def _assert_run_record_preserved(original: object, recovered: object) -> None:
    original_record = original.last_fit_record
    assert original_record is not None
    assert recovered.last_fit_record == original_record


def test_pca_round_trip_survives_a_fresh_process(tmp_path: Path) -> None:
    batch = PersistentPanelBatch(
        returns=np.array([[1.0, 2.0], [2.0, 4.5], [4.0, 7.0]], dtype=np.float64),
        asset_ids=("A", "B"),
    )
    model = PCAModel(PCAConfig(n_factors=1))
    model.fit(batch)
    artifact = model.save(tmp_path / "pca.ml4t")
    expected = model.extract(batch).asset_betas

    recovered = PCAModel.load(artifact)
    assert np.array_equal(recovered.extract(batch).asset_betas, expected)
    _assert_run_record_preserved(model, recovered)

    command = [
        sys.executable,
        "-c",
        (
            "import sys; from ml4t.models import PCAModel, PersistentPanelBatch; "
            "model=PCAModel.load(sys.argv[1]); "
            "batch=PersistentPanelBatch(timestamps=('t',), asset_ids=('A','B')); "
            "print(model.extract(batch).asset_betas.shape, model.last_fit_record.model_name)"
        ),
        str(artifact),
    ]
    environment = {**os.environ, "PYTHONPATH": str(Path.cwd() / "src")}
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.stdout.strip() == "(1, 2, 1) pca"


def test_numpy_model_artifacts_preserve_predictions(tmp_path: Path) -> None:
    rng = np.random.default_rng(71)
    panel = PersistentPanelBatch(
        returns=rng.normal(size=(8, 4)),
        asset_ids=("A", "B", "C", "D"),
    )
    rp_pca = RPPCAModel(RPPCAConfig(n_factors=2, gamma=-1.0))
    rp_pca.fit(panel)
    expected_rp = rp_pca.extract(panel)
    recovered_rp = RPPCAModel.load(rp_pca.save(tmp_path / "rp-pca.ml4t"))
    actual_rp = recovered_rp.extract(panel)
    assert np.array_equal(actual_rp.asset_betas, expected_rp.asset_betas)
    assert np.array_equal(actual_rp.factor_returns, expected_rp.factor_returns)
    _assert_run_record_preserved(rp_pca, recovered_rp)

    characteristics = rng.normal(size=(8, 4, 3))
    cross_section = CrossSectionBatch(
        characteristics=characteristics,
        returns=rng.normal(size=(8, 4)),
        asset_ids=("A", "B", "C", "D"),
    )
    ipca = IPCAModel(IPCAConfig(n_factors=1, max_iter=10))
    ipca.fit(cross_section)
    expected_ipca = ipca.extract(cross_section)
    recovered_ipca = IPCAModel.load(ipca.save(tmp_path / "ipca.ml4t"))
    actual_ipca = recovered_ipca.extract(cross_section)
    assert np.array_equal(actual_ipca.asset_betas, expected_ipca.asset_betas)
    assert np.array_equal(actual_ipca.factor_returns, expected_ipca.factor_returns)
    _assert_run_record_preserved(ipca, recovered_ipca)


@pytest.mark.parametrize(
    "forecaster",
    [AR1FactorForecaster(), EWMABaseFactorForecaster(), ExpandingMeanFactorForecaster()],
)
def test_factor_forecaster_artifacts_preserve_predictions(
    forecaster: object,
    tmp_path: Path,
) -> None:
    state = LatentFactorState(
        asset_betas=np.ones((5, 2, 2), dtype=np.float64),
        factor_returns=np.arange(10, dtype=np.float64).reshape(5, 2),
    )
    future = LatentFactorState(asset_betas=np.ones((3, 2, 2), dtype=np.float64))
    forecaster.fit(state)
    expected = forecaster.predict(future).factor_premia

    recovered = type(forecaster).load(forecaster.save(tmp_path / "forecaster.ml4t"))

    assert np.array_equal(recovered.predict(future).factor_premia, expected)
    _assert_run_record_preserved(forecaster, recovered)


def test_linear_portfolio_artifact_preserves_predictions(tmp_path: Path) -> None:
    rng = np.random.default_rng(73)
    batch = PortfolioSequenceBatch(
        features=rng.normal(size=(2, 3, 4, 2)),
        returns=rng.normal(size=(2, 3, 4)),
        asset_ids=("A", "B", "C", "D"),
    )
    model = LinearFeaturePortfolioModel(LinearPortfolioConfig())
    model.fit(batch)
    expected = model.predict(batch).weights

    recovered = LinearFeaturePortfolioModel.load(model.save(tmp_path / "linear.ml4t"))

    assert np.array_equal(recovered.predict(batch).weights, expected)
    _assert_run_record_preserved(model, recovered)


def test_artifact_loader_rejects_the_wrong_model_type(tmp_path: Path) -> None:
    batch = PersistentPanelBatch(returns=np.eye(2), asset_ids=("A", "B"))
    model = PCAModel(PCAConfig(n_factors=1))
    model.fit(batch)
    artifact = model.save(tmp_path / "pca.ml4t")

    with pytest.raises(ValueError, match="model_type mismatch"):
        RPPCAModel.load(artifact)


def test_cae_and_sae_artifacts_preserve_predictions(tmp_path: Path) -> None:
    rng = np.random.default_rng(79)
    batch = CrossSectionBatch(
        characteristics=rng.normal(size=(5, 4, 3)),
        returns=rng.normal(size=(5, 4)),
        asset_ids=("A", "B", "C", "D"),
    )
    future = CrossSectionBatch(
        characteristics=rng.normal(size=(2, 4, 3)),
        asset_ids=batch.asset_ids,
    )

    cae = CAEModel(
        CAEConfig(
            n_factors=1,
            hidden_units=(),
            n_epochs=1,
            checkpoint_interval=1,
            batch_size=32,
        )
    )
    cae.fit(batch)
    expected_cae = cae.extract(future).asset_betas
    recovered_cae = CAEModel.load(cae.save(tmp_path / "cae.ml4t"))
    assert np.array_equal(recovered_cae.extract(future).asset_betas, expected_cae)
    _assert_run_record_preserved(cae, recovered_cae)

    sae = SAEModel(
        SAEConfig(
            bottleneck_dim=2,
            aux_hidden_dim=2,
            main_hidden_units=(4, 4, 4, 4),
            dropout_rates=(0.0,) * 8,
            n_epochs=1,
            checkpoint_interval=1,
            batch_size=32,
        )
    )
    sae.fit(batch)
    expected_sae = sae.predict(future).signal_values
    recovered_sae = SAEModel.load(sae.save(tmp_path / "sae.ml4t"))
    assert np.array_equal(recovered_sae.predict(future).signal_values, expected_sae)
    _assert_run_record_preserved(sae, recovered_sae)


@pytest.mark.parametrize(
    "model",
    [
        LSTMPortfolioModel(
            LSTMPortfolioConfig(
                hidden_size=4,
                max_iters=1,
                eval_every=1,
                checkpoint_every=1,
            )
        ),
        DeepPortfolioModel(
            DeepPortfolioConfig(
                d_model=4,
                n_heads=1,
                cross_attention_heads=1,
                macro_gnn_heads=1,
                max_iters=1,
                eval_every=1,
                checkpoint_every=1,
            )
        ),
    ],
)
def test_neural_portfolio_artifacts_preserve_predictions(
    model: object,
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(83)
    train = PortfolioSequenceBatch(
        features=rng.normal(size=(2, 3, 2, 2)),
        returns=rng.normal(size=(2, 3, 2)),
        asset_ids=("A", "B"),
    )
    future = PortfolioSequenceBatch(
        features=rng.normal(size=(1, 2, 2, 2)),
        asset_ids=train.asset_ids,
    )
    model.fit(train)
    expected = model.predict(future).weights

    recovered = type(model).load(model.save(tmp_path / "portfolio.ml4t"))

    assert np.array_equal(recovered.predict(future).weights, expected)
    _assert_run_record_preserved(model, recovered)


def test_sdf_model_and_heads_preserve_predictions(tmp_path: Path) -> None:
    rng = np.random.default_rng(89)
    batch = CrossSectionBatch(
        characteristics=rng.normal(size=(5, 4, 3)),
        returns=rng.normal(scale=0.02, size=(5, 4)),
        asset_ids=("A", "B", "C", "D"),
    )
    config = StochasticDiscountFactorConfig(
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
    sdf = StochasticDiscountFactorModel(config)
    sdf.fit(batch)
    expected_state = sdf.extract(batch, checkpoint=("conditional", 1))
    recovered_sdf = StochasticDiscountFactorModel.load(sdf.save(tmp_path / "sdf.ml4t"))
    recovered_state = recovered_sdf.extract(batch, checkpoint=("conditional", 1))
    assert np.array_equal(recovered_state.asset_weights, expected_state.asset_weights)
    _assert_run_record_preserved(sdf, recovered_sdf)

    mapper = LinearStochasticDiscountFactorReturnMapper()
    mapper.fit(expected_state, batch)
    expected_forecast = mapper.predict(expected_state).expected_returns
    recovered_mapper = LinearStochasticDiscountFactorReturnMapper.load(
        mapper.save(tmp_path / "mapper.ml4t")
    )
    assert np.array_equal(
        recovered_mapper.predict(expected_state).expected_returns, expected_forecast
    )
    _assert_run_record_preserved(mapper, recovered_mapper)

    head = StochasticDiscountFactorBetaNetworkHead(config)
    head.fit(expected_state, batch)
    expected_signal = head.predict(batch, checkpoint=1).signal_values
    recovered_head = StochasticDiscountFactorBetaNetworkHead.load(
        head.save(tmp_path / "beta-head.ml4t")
    )
    assert np.array_equal(
        recovered_head.predict(batch, checkpoint=1).signal_values, expected_signal
    )
    _assert_run_record_preserved(head, recovered_head)
