from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ml4t.models import (
    AR1FactorForecaster,
    CrossSectionBatch,
    EWMABaseFactorForecaster,
    ExpandingMeanFactorForecaster,
    IPCAConfig,
    IPCAModel,
    LatentFactorState,
    LinearFeaturePortfolioModel,
    LinearPortfolioConfig,
    PCAConfig,
    PCAModel,
    PersistentPanelBatch,
    PortfolioSequenceBatch,
    RPPCAConfig,
    RPPCAModel,
)


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

    command = [
        sys.executable,
        "-c",
        (
            "import sys; from ml4t.models import PCAModel, PersistentPanelBatch; "
            "model=PCAModel.load(sys.argv[1]); "
            "batch=PersistentPanelBatch(timestamps=('t',), asset_ids=('A','B')); "
            "print(model.extract(batch).asset_betas.shape)"
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
    assert result.stdout.strip() == "(1, 2, 1)"


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


def test_artifact_loader_rejects_the_wrong_model_type(tmp_path: Path) -> None:
    batch = PersistentPanelBatch(returns=np.eye(2), asset_ids=("A", "B"))
    model = PCAModel(PCAConfig(n_factors=1))
    model.fit(batch)
    artifact = model.save(tmp_path / "pca.ml4t")

    with pytest.raises(ValueError, match="model_type mismatch"):
        RPPCAModel.load(artifact)
