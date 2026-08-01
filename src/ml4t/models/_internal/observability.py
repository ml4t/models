"""Common, redacted fit-attempt observability."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, fields, is_dataclass, replace
from functools import wraps
from hashlib import sha256
from time import perf_counter
from typing import Any, cast

import numpy as np

from ml4t.models._version import __version__
from ml4t.models.types import FitRunRecord, FitSummary


class FitObservable:
    """Expose the most recent successful or failed fit-attempt record."""

    _last_fit_record: FitRunRecord | None = None

    @property
    def last_fit_record(self) -> FitRunRecord | None:
        return self._last_fit_record


def state_with_fit_record(model: FitObservable, state: dict[str, Any]) -> dict[str, Any]:
    """Add the required fit record to a persisted model state."""

    if model.last_fit_record is None:
        raise RuntimeError("fitted model has no fit run record")
    return {**state, "fit_run_record": asdict(model.last_fit_record)}


def restore_fit_record(model: FitObservable, state: dict[str, Any]) -> None:
    """Restore and validate a fit record from persisted model state."""

    raw_record = state.get("fit_run_record")
    if not isinstance(raw_record, dict):
        raise ValueError("artifact is missing a valid fit run record")
    try:
        model._last_fit_record = FitRunRecord(**raw_record)
    except (TypeError, ValueError) as exc:
        raise ValueError("artifact contains an invalid fit run record") from exc


def observed_fit(method: Callable[..., FitSummary]) -> Callable[..., FitSummary]:
    """Attach a complete run record to a fit summary and retain failures."""

    @wraps(method)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> FitSummary:
        started = perf_counter()
        inputs = tuple(_fit_inputs(args, kwargs.values()))
        try:
            summary = method(self, *args, **kwargs)
        except Exception as exc:
            record = _build_record(
                self,
                inputs,
                elapsed_seconds=perf_counter() - started,
                stopping_reason="failed",
                skipped_updates=0,
                error_type=type(exc).__name__,
            )
            self._last_fit_record = record
            raise

        record = _build_record(
            self,
            inputs,
            elapsed_seconds=perf_counter() - started,
            stopping_reason=_stopping_reason(self, summary, kwargs),
            skipped_updates=int(summary.train_metrics.get("skipped_updates", 0.0)),
            error_type=None,
        )
        self._last_fit_record = record
        return replace(summary, run_record=record)

    return wrapped


def _fit_inputs(args: Iterable[Any], keyword_values: Iterable[Any]) -> Iterable[Any]:
    for value in (*args, *keyword_values):
        if value is not None and is_dataclass(value) and not isinstance(value, type):
            yield value


def _build_record(
    model: Any,
    inputs: tuple[Any, ...],
    *,
    elapsed_seconds: float,
    stopping_reason: str,
    skipped_updates: int,
    error_type: str | None,
) -> FitRunRecord:
    config = getattr(model, "config", None)
    config_values = asdict(config) if config is not None and is_dataclass(config) else {}
    model_name = str(getattr(config, "model_name", type(model).__name__))
    return FitRunRecord(
        schema_version=1,
        package_version=__version__,
        model_name=model_name,
        config=config_values,
        seed=int(getattr(config, "seed", 0)),
        resolved_device=str(getattr(config, "device", "cpu")).strip().lower(),
        resolved_dtype=str(getattr(config, "dtype", "float64")),
        input_dimensions=_input_dimensions(inputs),
        input_sha256=_input_sha256(inputs),
        stopping_reason=stopping_reason,
        skipped_updates=skipped_updates,
        elapsed_seconds=elapsed_seconds,
        error_type=error_type,
    )


def _input_dimensions(inputs: tuple[Any, ...]) -> dict[str, int]:
    dimensions: dict[str, int] = {}
    for input_index, value in enumerate(inputs):
        prefix = f"input_{input_index}"
        for field in fields(value):
            item = getattr(value, field.name)
            if isinstance(item, np.ndarray):
                for axis, size in enumerate(item.shape):
                    dimensions[f"{prefix}_{field.name}_{axis}"] = cast(int, size)
    return dimensions


def _input_sha256(inputs: tuple[Any, ...]) -> str:
    digest = sha256()
    for input_index, value in enumerate(inputs):
        digest.update(f"{input_index}:{type(value).__module__}.{type(value).__qualname__}".encode())
        for field in fields(value):
            if field.name == "metadata":
                continue
            item = getattr(value, field.name)
            digest.update(field.name.encode())
            if isinstance(item, np.ndarray):
                _update_array_digest(digest, item)
            elif isinstance(item, tuple):
                for child in item:
                    digest.update(type(child).__qualname__.encode())
                    digest.update(str(child).encode("utf-8", errors="backslashreplace"))
            elif item is None or isinstance(item, str | int | float | bool):
                digest.update(repr(item).encode())
    return digest.hexdigest()


def _update_array_digest(digest: Any, array: np.ndarray) -> None:
    digest.update(array.dtype.str.encode())
    digest.update(repr(array.shape).encode())
    iterator = np.nditer(
        array,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="C",
        buffersize=131_072,
    )
    for chunk in iterator:
        digest.update(np.ascontiguousarray(chunk).view(np.uint8))


def _stopping_reason(model: Any, summary: FitSummary, kwargs: dict[str, Any]) -> str:
    if not summary.converged:
        return "iteration_limit"

    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "max_iter"):
        return "converged"
    if config is not None and hasattr(config, "max_iters") and summary.history:
        last_step = max(float(record.get("step", 0.0)) for record in summary.history)
        if last_step < int(config.max_iters):
            return "early_stopping"

    if kwargs.get("validation_batch") is not None and hasattr(config, "n_epochs"):
        positive_checkpoints = [epoch for epoch in model.available_checkpoints if epoch > 0]
        if positive_checkpoints and max(positive_checkpoints) < int(config.n_epochs):
            return "early_stopping"

    if (
        kwargs.get("patience") is not None
        and summary.history
        and config is not None
        and hasattr(config, "n_epochs_unc")
    ):
        phase_maxima: dict[str, float] = {}
        for record in summary.history:
            phase = str(record.get("phase", ""))
            phase_maxima[phase] = max(phase_maxima.get(phase, 0.0), float(record.get("epoch", 0.0)))
        if phase_maxima.get("unconditional", 0.0) < int(config.n_epochs_unc) or phase_maxima.get(
            "conditional", 0.0
        ) < int(config.n_epochs_cond):
            return "early_stopping"

    return "completed"
