from __future__ import annotations

import ml4t.models as models
from ml4t.models import (
    api,
    asset_prediction,
    configs,
    forecasters,
    latent_factors,
    mappers,
    portfolio,
    stochastic_discount_factor,
    types,
)
from ml4t.models._public_api import STABLE_ROOT_EXPORTS


def test_root_exports_match_authoritative_stable_manifest() -> None:
    assert tuple(models.__all__) == STABLE_ROOT_EXPORTS
    assert tuple(sorted(STABLE_ROOT_EXPORTS)) == STABLE_ROOT_EXPORTS
    assert all(hasattr(models, name) for name in STABLE_ROOT_EXPORTS)
    assert "PipelineConfig" not in STABLE_ROOT_EXPORTS


def test_stable_submodules_have_explicit_supported_exports() -> None:
    assert set(api.__all__) == {
        "AssetMapper",
        "AssetPredictionModel",
        "FactorForecaster",
        "LatentFactorModel",
        "PortfolioModel",
        "PortfolioPostprocessor",
        "StochasticDiscountFactorEstimator",
    }
    assert set(types.__all__) == {
        "AssetForecastResult",
        "AssetSignalResult",
        "AssetWeightsResult",
        "CrossSectionBatch",
        "FactorForecastResult",
        "FitSummary",
        "LatentFactorPrediction",
        "LatentFactorState",
        "PersistentPanelBatch",
        "PortfolioPrediction",
        "PortfolioSequenceBatch",
        "PortfolioWeightsResult",
        "StochasticDiscountFactorState",
    }
    assert "PipelineConfig" not in configs.__all__


def test_family_exports_exclude_implementation_bases_and_loss_internals() -> None:
    family_exports = {
        *asset_prediction.__all__,
        *forecasters.__all__,
        *latent_factors.__all__,
        *mappers.__all__,
        *portfolio.__all__,
        *stochastic_discount_factor.__all__,
    }

    assert not any(name.startswith("Base") for name in family_exports)
    assert "PortfolioLossOutput" not in family_exports
    assert "robust_sharpe_loss" not in family_exports


def test_documented_pipeline_fit_result_is_public() -> None:
    assert models.PipelineFitResult.__module__ == "ml4t.models.pipelines"
    assert "PipelineFitResult" in models.__all__


def test_public_frame_names_use_financial_objects() -> None:
    assert "PredictionsFrame" in models.__all__
    assert "SignalsFrame" in models.__all__
    assert "WeightsFrame" in models.__all__
    assert "ResultsFrame" in models.__all__
    assert "SurfaceFrame" not in models.__all__
