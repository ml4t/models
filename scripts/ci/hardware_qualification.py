"""Qualify installed neural model workflows on CPU, CUDA, or MPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
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

SEED = 20260731
REPLAY_TOLERANCES = {
    "cpu": (1e-7, 1e-7),
    "cuda": (1e-5, 1e-5),
    "mps": (1e-4, 1e-4),
}

FitOnce = Callable[[], tuple[Any, np.ndarray, Callable[[Any], np.ndarray]]]
Loader = Callable[[Path, str], Any]


def _cross_section() -> CrossSectionBatch:
    rng = np.random.default_rng(SEED)
    periods, assets, features = 8, 5, 3
    characteristics = rng.normal(size=(periods, assets, features))
    coefficients = rng.normal(0.0, 0.01, size=features)
    returns = characteristics @ coefficients + rng.normal(0.0, 0.01, (periods, assets))
    return CrossSectionBatch(
        characteristics=characteristics,
        returns=returns,
        context_features=rng.normal(size=(periods, 2)),
        timestamps=tuple(range(periods)),
        asset_ids=tuple(f"asset-{index}" for index in range(assets)),
    )


def _portfolio_batch() -> PortfolioSequenceBatch:
    rng = np.random.default_rng(SEED)
    windows, periods, assets, features = 2, 4, 3, 2
    values = rng.normal(size=(windows, periods, assets, features))
    coefficients = rng.normal(0.0, 0.01, size=features)
    returns = values @ coefficients + rng.normal(0.0, 0.01, (windows, periods, assets))
    return PortfolioSequenceBatch(
        features=values,
        returns=returns,
        timestamps=tuple(range(periods)),
        asset_ids=tuple(f"asset-{index}" for index in range(assets)),
    )


def _digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _finite(value: np.ndarray) -> bool:
    observed = value[np.isfinite(value)]
    return bool(observed.size and not np.isinf(value).any())


def _max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.nanmax(np.abs(left - right)))


def _assert_close(name: str, left: np.ndarray, right: np.ndarray, device: str) -> float:
    rtol, atol = REPLAY_TOLERANCES[device]
    difference = _max_abs_difference(left, right)
    if not np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True):
        raise AssertionError(
            f"{name} exceeded replay tolerance: max_abs_difference={difference}, "
            f"rtol={rtol}, atol={atol}"
        )
    return difference


def _qualify(
    *,
    name: str,
    device: str,
    directory: Path,
    fit_once: FitOnce,
    loader: Loader,
) -> dict[str, Any]:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    model, first, predict = fit_once()
    fit_seconds = time.perf_counter() - started
    if not _finite(first):
        raise AssertionError(f"{name} produced no finite output")
    record = model.last_fit_record
    if record is None or record.resolved_device != device:
        raise AssertionError(
            f"{name} fit record device mismatch: expected {device!r}, "
            f"got {None if record is None else record.resolved_device!r}"
        )

    artifact = model.save(directory / f"{name}.ml4t")
    loaded_device = loader(artifact, device)
    loaded_cpu = loader(artifact, "cpu")
    if loaded_device.last_fit_record != record or loaded_cpu.last_fit_record != record:
        raise AssertionError(f"{name} did not preserve its fit run record")

    recovered_device = predict(loaded_device)
    recovered_cpu = predict(loaded_cpu)
    recovery_device_difference = _assert_close(name, first, recovered_device, device)
    recovery_cpu_difference = _assert_close(name, first, recovered_cpu, device)

    _, replay, _ = fit_once()
    replay_difference = _assert_close(name, first, replay, device)
    peak_cuda_bytes = torch.cuda.max_memory_allocated() if device.startswith("cuda") else None
    return {
        "artifact_bytes": artifact.stat().st_size,
        "digest": _digest(first),
        "fit_seconds": fit_seconds,
        "finite": True,
        "peak_cuda_allocated_bytes": peak_cuda_bytes,
        "recovery_cpu_max_abs_difference": recovery_cpu_difference,
        "recovery_device_max_abs_difference": recovery_device_difference,
        "replay_max_abs_difference": replay_difference,
    }


def _cae(batch: CrossSectionBatch, device: str) -> FitOnce:
    config = CAEConfig(
        n_factors=1,
        hidden_units=(),
        n_epochs=1,
        checkpoint_interval=1,
        batch_size=64,
        dtype="float32",
        seed=SEED,
        device=device,
    )

    def fit_once() -> tuple[Any, np.ndarray, Callable[[Any], np.ndarray]]:
        model = CAEModel(config)
        model.fit(batch)

        def predict(value: CAEModel) -> np.ndarray:
            return value.extract(batch, checkpoint=1).asset_betas

        return model, predict(model), predict

    return fit_once


def _sae(batch: CrossSectionBatch, device: str) -> FitOnce:
    config = SAEConfig(
        bottleneck_dim=2,
        aux_hidden_dim=2,
        main_hidden_units=(4, 4, 4, 4),
        dropout_rates=(0.0,) * 8,
        noise_std=0.0,
        n_epochs=1,
        checkpoint_interval=1,
        batch_size=64,
        dtype="float32",
        seed=SEED,
        device=device,
    )

    def fit_once() -> tuple[Any, np.ndarray, Callable[[Any], np.ndarray]]:
        model = SAEModel(config)
        model.fit(batch)

        def predict(value: SAEModel) -> np.ndarray:
            return value.predict(batch, checkpoint=1).signal_values

        return model, predict(model), predict

    return fit_once


def _sdf(batch: CrossSectionBatch, device: str) -> FitOnce:
    config = StochasticDiscountFactorConfig(
        state_dim_sdf=2,
        state_dim_moment=2,
        hidden_dim=4,
        n_instruments=2,
        dropout=0.0,
        n_epochs_unc=1,
        n_epochs_moment=1,
        n_epochs_cond=1,
        checkpoint_interval=1,
        dtype="float32",
        seed=SEED,
        device=device,
    )

    def fit_once() -> tuple[Any, np.ndarray, Callable[[Any], np.ndarray]]:
        model = StochasticDiscountFactorModel(config)
        model.fit(batch)

        def predict(value: StochasticDiscountFactorModel) -> np.ndarray:
            return value.extract(batch, checkpoint=("conditional", 1)).asset_weights

        return model, predict(model), predict

    return fit_once


def _beta_head(batch: CrossSectionBatch, state: Any, device: str) -> FitOnce:
    config = StochasticDiscountFactorConfig(
        beta_state_dim=2,
        beta_hidden_dim=4,
        beta_n_epochs=1,
        beta_checkpoint_interval=1,
        dropout=0.0,
        dtype="float32",
        seed=SEED,
        device=device,
    )

    def fit_once() -> tuple[Any, np.ndarray, Callable[[Any], np.ndarray]]:
        model = StochasticDiscountFactorBetaNetworkHead(config)
        model.fit(state, batch)

        def predict(value: StochasticDiscountFactorBetaNetworkHead) -> np.ndarray:
            return value.predict(batch, checkpoint=1).signal_values

        return model, predict(model), predict

    return fit_once


def _lstm(batch: PortfolioSequenceBatch, device: str) -> FitOnce:
    config = LSTMPortfolioConfig(
        hidden_size=4,
        max_iters=1,
        eval_every=1,
        checkpoint_every=1,
        dtype="float32",
        seed=SEED,
        device=device,
    )

    def fit_once() -> tuple[Any, np.ndarray, Callable[[Any], np.ndarray]]:
        model = LSTMPortfolioModel(config)
        model.fit(batch)

        def predict(value: LSTMPortfolioModel) -> np.ndarray:
            return value.predict(batch, checkpoint=1).weights

        return model, predict(model), predict

    return fit_once


def _deep(batch: PortfolioSequenceBatch, device: str) -> FitOnce:
    config = DeepPortfolioConfig(
        d_model=4,
        n_heads=1,
        cross_attention_heads=1,
        macro_gnn_heads=1,
        max_iters=1,
        eval_every=1,
        checkpoint_every=1,
        dtype="float32",
        seed=SEED,
        device=device,
    )

    def fit_once() -> tuple[Any, np.ndarray, Callable[[Any], np.ndarray]]:
        model = DeepPortfolioModel(config)
        model.fit(batch)

        def predict(value: DeepPortfolioModel) -> np.ndarray:
            return value.predict(batch, checkpoint=1).weights

        return model, predict(model), predict

    return fit_once


def _validate_device(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} requires an available CUDA backend")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device='mps' requires an available MPS backend")


def qualify(device: str) -> dict[str, Any]:
    _validate_device(device)
    cross_section = _cross_section()
    portfolio = _portfolio_batch()
    with tempfile.TemporaryDirectory(prefix="ml4t-models-hardware-") as temp_dir:
        directory = Path(temp_dir)
        sdf_model, _, _ = _sdf(cross_section, device)()
        sdf_state = sdf_model.extract(cross_section, checkpoint=("conditional", 1))
        results = {
            "cae": _qualify(
                name="cae",
                device=device,
                directory=directory,
                fit_once=_cae(cross_section, device),
                loader=lambda path, target: CAEModel.load(path, device=target),
            ),
            "sae": _qualify(
                name="sae",
                device=device,
                directory=directory,
                fit_once=_sae(cross_section, device),
                loader=lambda path, target: SAEModel.load(path, device=target),
            ),
            "sdf": _qualify(
                name="sdf",
                device=device,
                directory=directory,
                fit_once=_sdf(cross_section, device),
                loader=lambda path, target: StochasticDiscountFactorModel.load(path, device=target),
            ),
            "sdf_beta_head": _qualify(
                name="sdf-beta-head",
                device=device,
                directory=directory,
                fit_once=_beta_head(cross_section, sdf_state, device),
                loader=lambda path, target: StochasticDiscountFactorBetaNetworkHead.load(
                    path, device=target
                ),
            ),
            "lstm_portfolio": _qualify(
                name="lstm-portfolio",
                device=device,
                directory=directory,
                fit_once=_lstm(portfolio, device),
                loader=lambda path, target: LSTMPortfolioModel.load(path, device=target),
            ),
            "deep_portfolio": _qualify(
                name="deep-portfolio",
                device=device,
                directory=directory,
                fit_once=_deep(portfolio, device),
                loader=lambda path, target: DeepPortfolioModel.load(path, device=target),
            ),
        }
    return {
        "device": device,
        "environment": {
            "cuda_capability": (
                list(torch.cuda.get_device_capability()) if device.startswith("cuda") else None
            ),
            "cuda_device": torch.cuda.get_device_name() if device.startswith("cuda") else None,
            "cuda_runtime": torch.version.cuda,
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
        },
        "replay_tolerance": {
            "atol": REPLAY_TOLERANCES[device][1],
            "rtol": REPLAY_TOLERANCES[device][0],
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), required=True)
    parser.add_argument("--expected-cuda-device")
    parser.add_argument("--expected-cuda-capability")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = qualify(args.device)
    environment = result["environment"]
    if args.expected_cuda_device is not None and environment["cuda_device"] != (
        args.expected_cuda_device
    ):
        raise RuntimeError(
            "CUDA device mismatch: "
            f"expected {args.expected_cuda_device!r}, got {environment['cuda_device']!r}"
        )
    if args.expected_cuda_capability is not None:
        expected_capability = [int(part) for part in args.expected_cuda_capability.split(".")]
        if environment["cuda_capability"] != expected_capability:
            raise RuntimeError(
                "CUDA capability mismatch: "
                f"expected {expected_capability!r}, got {environment['cuda_capability']!r}"
            )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
