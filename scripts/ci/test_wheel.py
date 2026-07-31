from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

CORE_TESTS = (
    "test_configs.py",
    "test_forecasters.py",
    "test_integration_data.py",
    "test_pca_pipeline.py",
    "test_rp_pca.py",
    "test_types.py",
)


def _python_executable(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def _wheel(candidate_dir: Path) -> Path:
    wheels = tuple((candidate_dir / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"candidate must contain one wheel, found {len(wheels)}")
    return wheels[0].resolve()


def _new_venv(root: Path, name: str, python_version: str) -> Path:
    venv = root / name
    _run(["uv", "venv", str(venv), "--python", python_version])
    return _python_executable(venv)


def _core_smoke(python: Path) -> None:
    script = """
import importlib.util

import numpy as np

from ml4t.models import (
    LinearFeaturePortfolioModel,
    LinearPortfolioConfig,
    PCAConfig,
    PCAModel,
    PersistentPanelBatch,
    PortfolioSequenceBatch,
)

assert importlib.util.find_spec("torch") is None
panel = PersistentPanelBatch(returns=np.array([[0.01, 0.02], [0.02, 0.01], [0.03, 0.01]]))
pca = PCAModel(PCAConfig(n_factors=1))
pca.fit(panel)
assert pca.extract(panel).n_periods == 3

portfolio_batch = PortfolioSequenceBatch(
    features=np.array([[[[1.0], [2.0]], [[2.0], [1.0]]]]),
    returns=np.array([[[0.01, 0.02], [0.02, 0.01]]]),
    asset_ids=("A", "B"),
)
portfolio = LinearFeaturePortfolioModel(LinearPortfolioConfig())
portfolio.fit(portfolio_batch)
assert portfolio.predict(
    PortfolioSequenceBatch(features=portfolio_batch.features, asset_ids=portfolio_batch.asset_ids)
).weights.shape == (1, 2, 2)
"""
    with tempfile.TemporaryDirectory() as directory:
        _run([str(python), "-c", script], cwd=Path(directory))


def test_wheel(candidate_dir: Path, python_version: str, mode: str) -> None:
    wheel = _wheel(candidate_dir)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        core_python = _new_venv(root, "core", python_version)
        _run(["uv", "pip", "install", "--python", str(core_python), str(wheel)])
        _core_smoke(core_python)

        test_python = core_python
        if mode == "full":
            test_python = _new_venv(root, "full", python_version)
            _run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(test_python),
                    f"{wheel}[all]",
                    "pytest>=9.0.3",
                    "pytest-cov>=5.0.0",
                ]
            )
        else:
            _run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(test_python),
                    "pytest>=9.0.3",
                ]
            )

        shutil.copytree("tests", root / "tests")
        shutil.copytree("examples", root / "examples")
        if mode == "full":
            selected_tests = [str(root / "tests")]
            ignored = [
                "--ignore",
                str(root / "tests" / "test_release_candidate.py"),
                "--ignore",
                str(root / "tests" / "test_repo_hygiene.py"),
            ]
        else:
            selected_tests = [str(root / "tests" / name) for name in CORE_TESTS]
            ignored = []
        _run(
            [
                str(test_python),
                "-m",
                "pytest",
                *selected_tests,
                *ignored,
                "-q",
                "--tb=short",
            ],
            cwd=root,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--mode", choices=("full", "core"), required=True)
    args = parser.parse_args()
    test_wheel(args.candidate_dir, args.python_version, args.mode)


if __name__ == "__main__":
    main()
