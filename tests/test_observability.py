from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from ml4t.models import (
    CAEConfig,
    CAEModel,
    CrossSectionBatch,
    FitRunRecord,
    LSTMPortfolioConfig,
    LSTMPortfolioModel,
    PCAConfig,
    PCAModel,
    PersistentPanelBatch,
    PortfolioSequenceBatch,
)
from ml4t.models._internal.observability import FitObservable, state_with_fit_record
from ml4t.models.portfolio import runtime as portfolio_runtime


def _panel(offset: float = 0.0) -> PersistentPanelBatch:
    return PersistentPanelBatch(
        returns=np.array([[1.0, 0.0], [0.0, 1.0], [1.0 + offset, -1.0]], dtype=np.float64),
        timestamps=("SECRET_DATE_1", "SECRET_DATE_2", "SECRET_DATE_3"),
        asset_ids=("SECRET_ASSET_A", "SECRET_ASSET_B"),
        metadata={"api_key": "SECRET_API_KEY"},
    )


def _valid_record() -> FitRunRecord:
    return FitRunRecord(
        schema_version=1,
        package_version="0.1.0",
        model_name="pca",
        config={"n_factors": 1},
        seed=42,
        resolved_device="cpu",
        resolved_dtype="float64",
        input_dimensions={"input_0_returns_0": 3, "input_0_returns_1": 2},
        input_sha256="a" * 64,
        stopping_reason="completed",
        skipped_updates=0,
        elapsed_seconds=0.1,
    )


def test_fit_record_is_stable_complete_and_excludes_raw_inputs() -> None:
    first = PCAModel(PCAConfig(n_factors=1)).fit(_panel()).run_record
    second = PCAModel(PCAConfig(n_factors=1)).fit(_panel()).run_record
    changed = PCAModel(PCAConfig(n_factors=1)).fit(_panel(offset=0.5)).run_record

    assert first is not None and second is not None and changed is not None
    assert first.input_sha256 == second.input_sha256
    assert first.input_sha256 != changed.input_sha256
    assert first.input_dimensions == {"input_0_returns_0": 3, "input_0_returns_1": 2}
    assert first.config["n_factors"] == 1
    assert first.seed == 42
    assert first.resolved_device == "cpu"
    assert first.resolved_dtype == "float64"
    assert first.elapsed_seconds >= 0.0

    serialized = json.dumps(asdict(first), sort_keys=True)
    assert "SECRET" not in serialized
    assert len(serialized) < 2_000


def test_failed_fit_retains_redacted_attempt_record() -> None:
    model = PCAModel(PCAConfig(n_factors=1))
    assert model.last_fit_record is None
    invalid = PersistentPanelBatch(
        timestamps=("SECRET_DATE",),
        asset_ids=("SECRET_ASSET",),
        metadata={"password": "SECRET_PASSWORD"},
    )

    with pytest.raises(ValueError, match="requires returns"):
        model.fit(invalid)

    record = model.last_fit_record
    assert record is not None
    assert record.stopping_reason == "failed"
    assert record.error_type == "ValueError"
    assert "SECRET" not in json.dumps(asdict(record), sort_keys=True)


def test_persistence_rejects_a_fitted_surface_without_a_run_record() -> None:
    with pytest.raises(RuntimeError, match="has no fit run record"):
        state_with_fit_record(FitObservable(), {})


def test_portfolio_fit_records_early_stopping(monkeypatch: pytest.MonkeyPatch) -> None:
    values = iter((1.0, 0.0, -1.0))
    monkeypatch.setattr(
        portfolio_runtime,
        "evaluate_pooled_sharpe",
        lambda *_args, **_kwargs: next(values),
    )
    rng = np.random.default_rng(601)
    batch = PortfolioSequenceBatch(
        features=rng.normal(size=(1, 3, 2, 2)),
        returns=rng.normal(size=(1, 3, 2)),
    )
    model = LSTMPortfolioModel(
        LSTMPortfolioConfig(
            hidden_size=4,
            max_iters=3,
            eval_every=1,
            checkpoint_every=1,
            early_stopping_burn_in_iters=1,
            early_stopping_patience=1,
            metric_ema_alpha=1.0,
        )
    )

    summary = model.fit(batch)

    assert summary.run_record is not None
    assert summary.run_record.stopping_reason == "early_stopping"


def test_cae_fit_records_skipped_singleton_updates() -> None:
    batch = CrossSectionBatch(
        characteristics=np.array([[[1.0], [2.0], [3.0]]]),
        returns=np.array([[0.1, 0.2, 0.3]]),
    )
    model = CAEModel(
        CAEConfig(
            n_factors=1,
            hidden_units=(2,),
            n_epochs=1,
            checkpoint_interval=1,
            batch_size=2,
        )
    )

    summary = model.fit(batch)

    assert summary.run_record is not None
    assert summary.run_record.skipped_updates == 1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"package_version": ""}, "must be non-empty"),
        ({"model_name": ""}, "must be non-empty"),
        ({"input_sha256": "not-a-digest"}, "input_sha256"),
        ({"skipped_updates": -1}, "skipped_updates"),
        ({"elapsed_seconds": float("nan")}, "elapsed_seconds"),
    ],
)
def test_fit_run_record_rejects_invalid_schema(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_valid_record(), **changes)
