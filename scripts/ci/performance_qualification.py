"""Measure correctness-gated Chapter 14 release workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import tempfile
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from ml4t.models import (
    CAEConfig,
    CAEModel,
    CrossSectionBatch,
    DeepPortfolioConfig,
    DeepPortfolioModel,
    IPCAConfig,
    IPCAModel,
    LinearFeaturePortfolioModel,
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
    StochasticDiscountFactorConfig,
    StochasticDiscountFactorModel,
    __version__,
)

SEED = 20260731
CUDA_RTOL = 1e-5
CUDA_ATOL = 1e-5


@dataclass(frozen=True)
class Profile:
    periods: int
    assets: int
    features: int
    factors: int = 0
    epochs: int = 0
    ensemble: int = 1
    windows: int = 0
    sequence: int = 0
    iterations: int = 0


CANONICAL = {
    "pca": Profile(3_750, 9, 0, factors=5),
    "rp_pca": Profile(4_700, 59, 0, factors=5),
    "ipca": Profile(500, 100, 10, factors=3),
    "cae": Profile(500, 500, 46, factors=5, epochs=200, ensemble=5),
    "sdf": Profile(500, 200, 46),
    "sae": Profile(500, 300, 46, epochs=25),
    "linear_portfolio": Profile(0, 200, 17, windows=16, sequence=63),
    "lstm_portfolio": Profile(0, 200, 17, windows=16, sequence=63, iterations=5),
    "deep_portfolio": Profile(0, 200, 17, windows=16, sequence=63, iterations=5),
}

SMOKE = {
    "pca": Profile(30, 6, 0, factors=2),
    "rp_pca": Profile(30, 6, 0, factors=2),
    "ipca": Profile(12, 6, 3, factors=2),
    "cae": Profile(8, 5, 3, factors=1, epochs=1),
    "sdf": Profile(8, 5, 3),
    "sae": Profile(8, 5, 3, epochs=1),
    "linear_portfolio": Profile(0, 3, 2, windows=2, sequence=4),
    "lstm_portfolio": Profile(0, 3, 2, windows=2, sequence=4, iterations=1),
    "deep_portfolio": Profile(0, 3, 2, windows=2, sequence=4, iterations=1),
}

TIME_LIMITS = {
    "pca": 50.0,
    "rp_pca": 30.0,
    "ipca": 30.0,
    "cae": 360.0,
    "sdf": 1_200.0,
    "sae": 2_400.0,
    "linear_portfolio": 30.0,
    "lstm_portfolio": 300.0,
    "deep_portfolio": 600.0,
}

CUDA_MEMORY_LIMITS = {
    "cae": int(6.5 * 1024**3),
    "sdf": int(9.75 * 1024**3),
    "sae": int(10.625 * 1024**3),
    "lstm_portfolio": int(8.0 * 1024**3),
    "deep_portfolio": int(10.0 * 1024**3),
}


def _digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _persistent_panel(profile: Profile) -> PersistentPanelBatch:
    rng = np.random.default_rng(SEED)
    factors = rng.normal(0.0, 0.01, size=(profile.periods, profile.factors))
    loadings = rng.normal(size=(profile.assets, profile.factors))
    returns = factors @ loadings.T + rng.normal(0.0, 0.005, (profile.periods, profile.assets))
    return PersistentPanelBatch(
        returns=returns,
        timestamps=tuple(range(profile.periods)),
        asset_ids=tuple(f"asset-{index}" for index in range(profile.assets)),
    )


def _cross_section(profile: Profile) -> CrossSectionBatch:
    rng = np.random.default_rng(SEED)
    characteristics = rng.normal(size=(profile.periods, profile.assets, profile.features))
    coefficients = rng.normal(0.0, 0.01, size=profile.features)
    returns = characteristics @ coefficients + rng.normal(
        0.0, 0.01, (profile.periods, profile.assets)
    )
    return CrossSectionBatch(
        characteristics=characteristics,
        returns=returns,
        context_features=rng.normal(size=(profile.periods, 8)),
        timestamps=tuple(range(profile.periods)),
        asset_ids=tuple(f"asset-{index}" for index in range(profile.assets)),
    )


def _portfolio_batch(profile: Profile) -> PortfolioSequenceBatch:
    rng = np.random.default_rng(SEED)
    shape = (profile.windows, profile.sequence, profile.assets, profile.features)
    features = rng.normal(size=shape)
    coefficients = rng.normal(0.0, 0.01, size=profile.features)
    returns = features @ coefficients + rng.normal(
        0.0,
        0.01,
        size=(profile.windows, profile.sequence, profile.assets),
    )
    return PortfolioSequenceBatch(
        features=features,
        returns=returns,
        vol_scale=np.ones_like(returns),
        prev_weights=np.zeros((profile.windows, profile.assets)),
        mask=np.ones_like(returns, dtype=bool),
        group_ids=np.arange(profile.assets, dtype=np.int64) % 10,
        costs=np.full(profile.assets, 0.001),
        adjacency_mask=np.zeros((profile.assets, profile.assets), dtype=bool),
        timestamps=tuple(range(profile.sequence)),
        asset_ids=tuple(f"asset-{index}" for index in range(profile.assets)),
    )


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _timed(call: Callable[[], Any], device: str) -> tuple[Any, float]:
    started = time.perf_counter()
    value = call()
    _synchronize(device)
    return value, time.perf_counter() - started


def _assert_recovered(name: str, expected: np.ndarray, actual: np.ndarray, device: str) -> None:
    rtol = CUDA_RTOL if device == "cuda" else 1e-10
    atol = CUDA_ATOL if device == "cuda" else 1e-10
    if not np.allclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True):
        difference = float(np.nanmax(np.abs(expected - actual)))
        raise AssertionError(
            f"{name} recovered output mismatch: max_abs_difference={difference}, "
            f"rtol={rtol}, atol={atol}"
        )


def _measure(
    *,
    name: str,
    model: Any,
    fit: Callable[[], Any],
    infer: Callable[[Any], np.ndarray],
    load: Callable[[Path], Any],
    device: str,
    directory: Path,
    enforce_limits: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    _, fit_seconds = _timed(fit, device)
    output, inference_seconds = _timed(lambda: infer(model), device)
    if np.isinf(output).any() or not np.isfinite(output).any():
        raise AssertionError(f"{name} produced a non-finite observed output")

    artifact_path = directory / f"{name}.ml4t"
    _, save_seconds = _timed(lambda: model.save(artifact_path), device)
    recovered, load_seconds = _timed(lambda: load(artifact_path), device)
    recovered_output, recovered_inference_seconds = _timed(lambda: infer(recovered), device)
    _assert_recovered(name, output, recovered_output, device)

    peak_cuda_bytes = torch.cuda.max_memory_reserved() if device == "cuda" else None
    if enforce_limits and fit_seconds > TIME_LIMITS[name]:
        raise AssertionError(
            f"{name} fit exceeded time limit: expected <= {TIME_LIMITS[name]}s, got {fit_seconds}s"
        )
    memory_limit = CUDA_MEMORY_LIMITS.get(name)
    if (
        enforce_limits
        and memory_limit is not None
        and peak_cuda_bytes is not None
        and peak_cuda_bytes > memory_limit
    ):
        raise AssertionError(
            f"{name} exceeded CUDA memory limit: expected <= {memory_limit}, got {peak_cuda_bytes}"
        )

    return (
        {
            "artifact_bytes": artifact_path.stat().st_size,
            "digest": _digest(output),
            "fit_seconds": fit_seconds,
            "inference_seconds": inference_seconds,
            "load_seconds": load_seconds,
            "peak_cuda_reserved_bytes": peak_cuda_bytes,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "recovered_inference_seconds": recovered_inference_seconds,
            "save_seconds": save_seconds,
            "shape": list(output.shape),
        },
        output,
    )


def _pca(profile: Profile, directory: Path, enforce: bool) -> tuple[dict[str, Any], np.ndarray]:
    batch = _persistent_panel(profile)
    model = PCAModel(PCAConfig(n_factors=profile.factors))
    return _measure(
        name="pca",
        model=model,
        fit=lambda: model.fit(batch),
        infer=lambda value: value.extract(batch).asset_betas,
        load=lambda path: PCAModel.load(path),
        device="cpu",
        directory=directory,
        enforce_limits=enforce,
    )


def _rp_pca(profile: Profile, directory: Path, enforce: bool) -> tuple[dict[str, Any], np.ndarray]:
    batch = _persistent_panel(profile)
    model = RPPCAModel(RPPCAConfig(n_factors=profile.factors, gamma=10.0))
    return _measure(
        name="rp_pca",
        model=model,
        fit=lambda: model.fit(batch),
        infer=lambda value: value.extract(batch).asset_betas,
        load=lambda path: RPPCAModel.load(path),
        device="cpu",
        directory=directory,
        enforce_limits=enforce,
    )


def _ipca(profile: Profile, directory: Path, enforce: bool) -> tuple[dict[str, Any], np.ndarray]:
    batch = _cross_section(profile)
    model = IPCAModel(IPCAConfig(n_factors=profile.factors, max_iter=100))
    return _measure(
        name="ipca",
        model=model,
        fit=lambda: model.fit(batch),
        infer=lambda value: value.extract(batch).asset_betas,
        load=lambda path: IPCAModel.load(path),
        device="cpu",
        directory=directory,
        enforce_limits=enforce,
    )


def _cae(
    profile: Profile, device: str, directory: Path, enforce: bool
) -> tuple[dict[str, Any], np.ndarray]:
    batch = _cross_section(profile)
    model = CAEModel(
        CAEConfig(
            n_factors=profile.factors,
            n_epochs=profile.epochs,
            n_ensemble=profile.ensemble,
            checkpoint_interval=max(profile.epochs // 10, 1),
            batch_size=10_000,
            dtype="float32",
            seed=SEED,
            device=device,
        )
    )
    return _measure(
        name="cae",
        model=model,
        fit=lambda: model.fit(batch),
        infer=lambda value: value.extract(batch).asset_betas,
        load=lambda path: CAEModel.load(path, device=device),
        device=device,
        directory=directory,
        enforce_limits=enforce,
    )


def _sdf(
    profile: Profile, device: str, directory: Path, enforce: bool
) -> tuple[dict[str, Any], np.ndarray]:
    batch = _cross_section(profile)
    epochs = (512, 64, 2_048) if enforce else (1, 1, 1)
    model = StochasticDiscountFactorModel(
        StochasticDiscountFactorConfig(
            state_dim_sdf=32 if enforce else 2,
            state_dim_moment=16 if enforce else 2,
            hidden_dim=64 if enforce else 4,
            n_instruments=8 if enforce else 2,
            n_epochs_unc=epochs[0],
            n_epochs_moment=epochs[1],
            n_epochs_cond=epochs[2],
            checkpoint_interval=64 if enforce else 1,
            dropout=0.0,
            dtype="float32",
            seed=SEED,
            device=device,
        )
    )
    return _measure(
        name="sdf",
        model=model,
        fit=lambda: model.fit(batch),
        infer=lambda value: value.extract(batch).asset_weights,
        load=lambda path: StochasticDiscountFactorModel.load(path, device=device),
        device=device,
        directory=directory,
        enforce_limits=enforce,
    )


def _sae(
    profile: Profile, device: str, directory: Path, enforce: bool
) -> tuple[dict[str, Any], np.ndarray]:
    batch = _cross_section(profile)
    model = SAEModel(
        SAEConfig(
            bottleneck_dim=8 if enforce else 2,
            aux_hidden_dim=16 if enforce else 2,
            main_hidden_units=(64, 32, 16, 8) if enforce else (4, 4, 4, 4),
            dropout_rates=(0.0,) * 8,
            noise_std=0.0,
            n_epochs=profile.epochs,
            checkpoint_interval=max(profile.epochs // 5, 1),
            batch_size=8_192,
            dtype="float32",
            seed=SEED,
            device=device,
        )
    )
    return _measure(
        name="sae",
        model=model,
        fit=lambda: model.fit(batch),
        infer=lambda value: value.predict(batch).signal_values,
        load=lambda path: SAEModel.load(path, device=device),
        device=device,
        directory=directory,
        enforce_limits=enforce,
    )


def _linear_portfolio(
    profile: Profile, directory: Path, enforce: bool
) -> tuple[dict[str, Any], np.ndarray]:
    batch = _portfolio_batch(profile)
    model = LinearFeaturePortfolioModel(LinearPortfolioConfig(dtype="float64", device="cpu"))
    return _measure(
        name="linear_portfolio",
        model=model,
        fit=lambda: model.fit(batch),
        infer=lambda value: value.predict(batch).weights,
        load=lambda path: LinearFeaturePortfolioModel.load(path, device="cpu"),
        device="cpu",
        directory=directory,
        enforce_limits=enforce,
    )


def _lstm_portfolio(
    profile: Profile, device: str, directory: Path, enforce: bool
) -> tuple[dict[str, Any], np.ndarray]:
    batch = _portfolio_batch(profile)
    model = LSTMPortfolioModel(
        LSTMPortfolioConfig(
            hidden_size=64 if enforce else 4,
            n_layers=1,
            dropout=0.0,
            batch_size=4,
            max_iters=profile.iterations,
            eval_every=1,
            checkpoint_every=profile.iterations,
            default_checkpoint=profile.iterations,
            early_stopping_burn_in_iters=profile.iterations + 1,
            dtype="float32",
            seed=SEED,
            device=device,
        )
    )
    return _measure(
        name="lstm_portfolio",
        model=model,
        fit=lambda: model.fit(batch, validation_batch=batch),
        infer=lambda value: value.predict(batch, checkpoint=profile.iterations).weights,
        load=lambda path: LSTMPortfolioModel.load(path, device=device),
        device=device,
        directory=directory,
        enforce_limits=enforce,
    )


def _deep_portfolio(
    profile: Profile, device: str, directory: Path, enforce: bool
) -> tuple[dict[str, Any], np.ndarray]:
    batch = _portfolio_batch(profile)
    model = DeepPortfolioModel(
        DeepPortfolioConfig(
            d_model=64 if enforce else 4,
            n_heads=2 if enforce else 1,
            cross_attention_heads=2 if enforce else 1,
            macro_gnn_heads=2 if enforce else 1,
            dropout=0.0,
            batch_size=4,
            max_iters=profile.iterations,
            eval_every=1,
            checkpoint_every=profile.iterations,
            default_checkpoint=profile.iterations,
            early_stopping_burn_in_iters=profile.iterations + 1,
            dtype="float32",
            seed=SEED,
            device=device,
        )
    )
    return _measure(
        name="deep_portfolio",
        model=model,
        fit=lambda: model.fit(batch, validation_batch=batch),
        infer=lambda value: value.predict(batch, checkpoint=profile.iterations).weights,
        load=lambda path: DeepPortfolioModel.load(path, device=device),
        device=device,
        directory=directory,
        enforce_limits=enforce,
    )


def qualify(profile_name: str, device: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("canonical CUDA performance qualification requires CUDA")
    profiles = CANONICAL if profile_name == "canonical" else SMOKE
    enforce = profile_name == "canonical"
    with tempfile.TemporaryDirectory(prefix="ml4t-models-performance-") as temp_dir:
        directory = Path(temp_dir)
        measured = {
            "pca": _pca(profiles["pca"], directory, enforce),
            "rp_pca": _rp_pca(profiles["rp_pca"], directory, enforce),
            "ipca": _ipca(profiles["ipca"], directory, enforce),
            "cae": _cae(profiles["cae"], device, directory, enforce),
            "sdf": _sdf(profiles["sdf"], device, directory, enforce),
            "sae": _sae(profiles["sae"], device, directory, enforce),
            "linear_portfolio": _linear_portfolio(profiles["linear_portfolio"], directory, enforce),
            "lstm_portfolio": _lstm_portfolio(
                profiles["lstm_portfolio"], device, directory, enforce
            ),
            "deep_portfolio": _deep_portfolio(
                profiles["deep_portfolio"], device, directory, enforce
            ),
        }
    results = {name: value[0] for name, value in measured.items()}
    arrays = {name: value[1] for name, value in measured.items()}
    return (
        {
            "device": device,
            "environment": {
                "cuda_capability": (
                    list(torch.cuda.get_device_capability()) if device == "cuda" else None
                ),
                "cuda_device": torch.cuda.get_device_name() if device == "cuda" else None,
                "cuda_runtime": torch.version.cuda,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                "package_version": __version__,
                "platform": platform.platform(),
                "python": sys.version,
                "torch": torch.__version__,
            },
            "profile": profile_name,
            "results": results,
        },
        arrays,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "canonical"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    args = parser.parse_args()
    warnings.simplefilter("error", RuntimeWarning)
    result, arrays = qualify(args.profile, args.device)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cast(Any, np.savez_compressed)(args.arrays, **arrays)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
