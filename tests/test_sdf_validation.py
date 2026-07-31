from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from ml4t.models import (
    CrossSectionBatch,
    LinearStochasticDiscountFactorReturnMapper,
    StochasticDiscountFactorBetaNetworkHead,
    StochasticDiscountFactorConfig,
    StochasticDiscountFactorModel,
    StochasticDiscountFactorState,
)
from ml4t.models._internal.persistence import LoadedArtifact
from ml4t.models.stochastic_discount_factor import mapper as mapper_module
from ml4t.models.stochastic_discount_factor import model as model_module


def _config() -> StochasticDiscountFactorConfig:
    return StochasticDiscountFactorConfig(
        state_dim_sdf=2,
        state_dim_moment=2,
        hidden_dim=4,
        n_instruments=2,
        n_epochs_unc=1,
        n_epochs_moment=1,
        n_epochs_cond=1,
        checkpoint_interval=1,
        beta_state_dim=2,
        beta_hidden_dim=4,
        beta_n_epochs=1,
        beta_checkpoint_interval=1,
        dropout=0.0,
    )


def _batch(*, context: bool = False, returns: bool = True) -> CrossSectionBatch:
    rng = np.random.default_rng(401)
    return CrossSectionBatch(
        characteristics=rng.normal(size=(4, 3, 2)),
        returns=rng.normal(size=(4, 3)) if returns else None,
        context_features=rng.normal(size=(4, 2)) if context else None,
        mask=np.array(
            [[True, True, False], [True, False, True], [True, True, True], [False, True, True]]
        ),
        asset_ids=("A", "B", "C"),
    )


def _state(*, sdf_values: bool = True) -> StochasticDiscountFactorState:
    return StochasticDiscountFactorState(
        asset_weights=np.ones((4, 3)),
        sdf_values=np.ones(4) if sdf_values else None,
        asset_ids=("A", "B", "C"),
    )


def test_linear_mapper_rejects_unavailable_training_and_inference(tmp_path: Path) -> None:
    mapper = LinearStochasticDiscountFactorReturnMapper()
    assert not mapper.is_fitted

    with pytest.raises(ValueError, match="requires returns"):
        mapper.fit(_state(), _batch(returns=False))
    with pytest.raises(RuntimeError, match="fitted before predict"):
        mapper.predict(_state())
    with pytest.raises(RuntimeError, match="fitted before save"):
        mapper.save(tmp_path / "mapper.ml4t")


def test_linear_mapper_handles_fewer_than_two_finite_observations() -> None:
    batch = _batch()
    returns = np.full_like(batch.returns, np.nan)
    assert returns is not None
    returns[0, 0] = 0.1
    sparse_batch = CrossSectionBatch(
        characteristics=batch.characteristics,
        returns=returns,
        asset_ids=batch.asset_ids,
    )
    state = StochasticDiscountFactorState(
        asset_weights=np.ones((4, 3)),
        sdf_values=np.ones(4),
        asset_ids=batch.asset_ids,
    )

    summary = LinearStochasticDiscountFactorReturnMapper().fit(state, sparse_batch)

    assert summary.train_metrics == {"intercept": 0.0, "slope": 0.0}


def test_sdf_model_rejects_invalid_fit_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="requires returns"):
        StochasticDiscountFactorModel(_config()).fit(_batch(returns=False))

    batch = _batch()
    invalid = CrossSectionBatch(
        characteristics=np.full_like(batch.characteristics, np.nan),
        returns=batch.returns,
        asset_ids=batch.asset_ids,
    )
    with pytest.raises(ValueError, match="no valid observations"):
        StochasticDiscountFactorModel(_config()).fit(invalid)

    monkeypatch.setattr(StochasticDiscountFactorConfig, "output_mode", "expected_returns")
    with pytest.raises(ValueError, match="weight-native"):
        StochasticDiscountFactorModel(_config()).fit(batch)
    monkeypatch.setattr(StochasticDiscountFactorConfig, "output_mode", "weights")
    monkeypatch.setattr(StochasticDiscountFactorConfig, "expected_return_mapper", "nonlinear")
    with pytest.raises(ValueError, match="expected_return_mapper"):
        StochasticDiscountFactorModel(_config()).fit(batch)


def test_sdf_model_rejects_unavailable_and_incompatible_extraction(tmp_path: Path) -> None:
    model = StochasticDiscountFactorModel(_config())
    with pytest.raises(RuntimeError, match="fitted before extract"):
        model.extract(_batch())
    with pytest.raises(RuntimeError, match="fitted before save"):
        model.save(tmp_path / "sdf.ml4t")

    model.fit(_batch(context=True))
    wrong_features = CrossSectionBatch(characteristics=np.ones((2, 3, 3)))
    with pytest.raises(ValueError, match="feature dimension"):
        model.extract(wrong_features)
    wrong_context = CrossSectionBatch(
        characteristics=np.ones((2, 3, 2)), context_features=np.ones((2, 1))
    )
    with pytest.raises(ValueError, match="context feature dimension"):
        model.extract(wrong_context)


@pytest.mark.parametrize(
    ("checkpoint", "message"),
    [
        (("conditional", 9), "not in available_checkpoints"),
        (99, "not in available_checkpoints"),
    ],
)
def test_sdf_model_rejects_unknown_checkpoints(
    checkpoint: tuple[str, int] | int, message: str
) -> None:
    model = StochasticDiscountFactorModel(_config())
    model.fit(_batch())
    with pytest.raises(ValueError, match=message):
        model.extract(_batch(), checkpoint=checkpoint)


def test_sdf_checkpoint_helpers_cover_empty_and_masked_panels() -> None:
    with pytest.raises(ValueError, match="available_checkpoints is empty"):
        model_module._select_checkpoint(
            checkpoint=None, configured_default=None, available=(), n_epochs_unc=1
        )

    panel = model_module._reshape_weight_panel(
        weights=np.array([1.0]),
        mask=np.array([[False], [True]]),
        shape=(2, 1),
    )
    assert np.isnan(panel[0, 0])
    assert panel[1, 0] == 1.0
    assert model_module._select_checkpoint(
        checkpoint=2,
        configured_default=None,
        available=(("unconditional", 1), ("conditional", 1)),
        n_epochs_unc=1,
    ) == ("conditional", 1)


def test_sdf_validation_tensor_preparation_rejects_empty_panel() -> None:
    torch = pytest.importorskip("torch")
    batch = CrossSectionBatch(
        characteristics=np.full((2, 2, 2), np.nan),
        returns=np.ones((2, 2)),
        context_features=np.ones((2, 1)),
    )
    with pytest.raises(ValueError, match="no valid SDF validation observations"):
        model_module._prepare_sdf_tensors(batch, torch, torch.device("cpu"))

    tensors = model_module._prepare_sdf_tensors(_batch(context=True), torch, torch.device("cpu"))
    assert tensors[-1] is not None


def test_beta_head_rejects_invalid_fit_inputs() -> None:
    head = StochasticDiscountFactorBetaNetworkHead(_config())
    with pytest.raises(ValueError, match="requires returns"):
        head.fit(_state(), _batch(returns=False))
    with pytest.raises(ValueError, match="include sdf_values"):
        head.fit(_state(sdf_values=False), _batch())

    batch = _batch()
    state = StochasticDiscountFactorState(
        asset_weights=np.ones((4, 3)),
        sdf_values=np.full(4, np.nan),
        asset_ids=batch.asset_ids,
    )
    with pytest.raises(ValueError, match="no valid observations"):
        head.fit(state, batch)


def test_beta_head_rejects_unavailable_and_incompatible_prediction(tmp_path: Path) -> None:
    head = StochasticDiscountFactorBetaNetworkHead(_config())
    with pytest.raises(RuntimeError, match="fitted before predict"):
        head.predict(_batch())
    with pytest.raises(RuntimeError, match="fitted before save"):
        head.save(tmp_path / "head.ml4t")

    batch = _batch(context=True)
    head.fit(_state(), batch)
    wrong_features = CrossSectionBatch(characteristics=np.ones((2, 3, 3)))
    with pytest.raises(ValueError, match="feature dimension"):
        head.predict(wrong_features)
    wrong_context = CrossSectionBatch(
        characteristics=np.ones((2, 3, 2)), context_features=np.ones((2, 1))
    )
    with pytest.raises(ValueError, match="context feature dimension"):
        head.predict(wrong_context)


def test_beta_training_payload_handles_missing_data_and_constant_scale() -> None:
    assert mapper_module._beta_training_payload(_state(), _batch(returns=False), scale=None) is None
    assert (
        mapper_module._beta_training_payload(_state(sdf_values=False), _batch(), scale=None) is None
    )

    batch = _batch()
    state = StochasticDiscountFactorState(
        asset_weights=np.ones((4, 3)),
        sdf_values=np.ones(4),
        asset_ids=batch.asset_ids,
    )
    payload = mapper_module._beta_training_payload(state, batch, scale=None)
    assert payload is not None
    assert payload["scale"] == 1.0


@pytest.mark.parametrize("module", [model_module, mapper_module])
def test_sdf_load_rejects_invalid_checkpoint_tree(
    module: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = LoadedArtifact(config=asdict(_config()), state={"checkpoint_tree": []}, arrays={})
    monkeypatch.setattr(module, "load_artifact", lambda *_args, **_kwargs: artifact)
    model_type = (
        StochasticDiscountFactorModel
        if module is model_module
        else StochasticDiscountFactorBetaNetworkHead
    )
    with pytest.raises(ValueError, match="checkpoint tree is invalid"):
        model_type.load("unused.ml4t")


@pytest.mark.parametrize("module", [model_module, mapper_module])
def test_sdf_load_rejects_empty_checkpoints(
    module: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {
        "checkpoint_tree": {"kind": "dict", "items": []},
        "asset_ids": (),
        "history": (),
        "n_context_features": 0,
        "n_characteristics": 2,
        "n_asset_features": 2,
        "f_hat_scale": 1.0,
    }
    artifact = LoadedArtifact(config=asdict(_config()), state=state, arrays={})
    monkeypatch.setattr(module, "load_artifact", lambda *_args, **_kwargs: artifact)
    model_type = (
        StochasticDiscountFactorModel
        if module is model_module
        else StochasticDiscountFactorBetaNetworkHead
    )
    with pytest.raises(ValueError, match="no checkpoints"):
        model_type.load("unused.ml4t")
