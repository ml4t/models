from __future__ import annotations

import runpy
from pathlib import Path

import pytest

EXAMPLES = (
    "latent_factor_pipeline.py",
    "stochastic_discount_factor.py",
    "direct_asset_prediction.py",
    "portfolio_learning.py",
)


@pytest.mark.parametrize("example", EXAMPLES)
def test_examples_execute(example: str) -> None:
    if example in {"stochastic_discount_factor.py", "direct_asset_prediction.py"}:
        pytest.importorskip("torch")
    runpy.run_path(str(Path(__file__).parents[1] / "examples" / example))
