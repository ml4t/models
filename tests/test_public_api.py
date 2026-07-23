from __future__ import annotations

import ml4t.models as models


def test_documented_pipeline_fit_result_is_public() -> None:
    assert models.PipelineFitResult.__module__ == "ml4t.models.pipelines"
    assert "PipelineFitResult" in models.__all__


def test_public_frame_names_use_financial_objects() -> None:
    assert "PredictionsFrame" in models.__all__
    assert "SignalsFrame" in models.__all__
    assert "WeightsFrame" in models.__all__
    assert "ResultsFrame" in models.__all__
    assert "SurfaceFrame" not in models.__all__
