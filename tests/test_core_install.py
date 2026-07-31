from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_linear_portfolio_workflow_runs_without_torch() -> None:
    script = r"""
import importlib.abc
import sys

import numpy as np


class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        del path, target
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("torch is blocked", name="torch")
        return None


sys.meta_path.insert(0, BlockTorch())

from ml4t.models import (
    LinearFeaturePortfolioModel,
    LinearPortfolioConfig,
    PortfolioSequenceBatch,
)

batch = PortfolioSequenceBatch(
    features=np.array([[[[1.0], [2.0]], [[2.0], [1.0]]]]),
    returns=np.array([[[0.01, 0.02], [0.02, 0.01]]]),
    asset_ids=("A", "B"),
)
model = LinearFeaturePortfolioModel(LinearPortfolioConfig())
model.fit(batch)
weights = model.predict(
    PortfolioSequenceBatch(features=batch.features, asset_ids=batch.asset_ids)
)
assert weights.weights.shape == (1, 2, 2)

try:
    from ml4t.models import LSTMPortfolioModel
except ImportError as exc:
    assert "install ml4t-models[deep]" in str(exc)
else:
    raise AssertionError("neural import unexpectedly succeeded without PyTorch")
"""
    environment = {**os.environ, "PYTHONPATH": str(Path.cwd() / "src")}

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
