"""Phase-aware stochastic discount factor model."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

import numpy as np

from ml4t.models._internal.latent_factor_utils import (
    resolve_checkpoint_epochs,
)
from ml4t.models._internal.torch_runtime import import_torch, resolve_device, seed_torch
from ml4t.models.configs import StochasticDiscountFactorConfig
from ml4t.models.stochastic_discount_factor.base import BaseStochasticDiscountFactorModel
from ml4t.models.types import CrossSectionBatch, FitSummary, StochasticDiscountFactorState

SDFCheckpoint = tuple[str, int]

VAL_BEST_LOSS_UNCONDITIONAL: SDFCheckpoint = ("unconditional", 0)
VAL_BEST_SHARPE_UNCONDITIONAL: SDFCheckpoint = ("unconditional", -1)
VAL_BEST_LOSS_CONDITIONAL: SDFCheckpoint = ("conditional", 0)
VAL_BEST_SHARPE_CONDITIONAL: SDFCheckpoint = ("conditional", -1)


class StochasticDiscountFactorModel(BaseStochasticDiscountFactorModel):
    """Stochastic discount factor model with weight-native structural outputs."""

    def __init__(self, config: StochasticDiscountFactorConfig) -> None:
        super().__init__(config)
        self._checkpoint_states: dict[SDFCheckpoint, dict[str, dict[str, Any]]] = {}
        self._asset_ids: tuple[str, ...] = ()
        self._n_characteristics: int | None = None
        self._n_context_features: int = 0
        self._history: tuple[dict[str, float | str], ...] = ()

    @property
    def available_checkpoints(self) -> tuple[SDFCheckpoint, ...]:
        return tuple(sorted(self._checkpoint_states, key=_checkpoint_sort_key))

    def fit(
        self,
        batch: CrossSectionBatch,
        *,
        validation_batch: CrossSectionBatch | None = None,
        patience: int | None = None,
    ) -> FitSummary:
        if batch.returns is None:
            raise ValueError("Stochastic discount factor training requires returns in the batch")
        if validation_batch is not None and validation_batch.returns is None:
            raise ValueError(
                "validation_batch requires returns to compute CPZ best-by-validation checkpoints"
            )
        if self.config.output_mode != "weights":
            raise ValueError(
                "StochasticDiscountFactorModel is weight-native; use a mapper for expected-return output"
            )
        if self.config.expected_return_mapper != "linear":
            raise ValueError("expected_return_mapper must be 'linear'")

        torch = import_torch()
        nn = _import_stochastic_discount_factor_nn()
        device = resolve_device(torch, self.config.device)

        returns_raw = torch.as_tensor(np.asarray(batch.returns, dtype=np.float32), device=device)
        mask = _resolve_mask(batch, torch, device)
        returns = torch.where(mask, returns_raw, torch.zeros_like(returns_raw))
        n_obs_per_asset = mask.float().sum(dim=0)
        if int(mask.sum().item()) == 0:
            raise ValueError("Stochastic discount factor training received no valid observations")

        asset_features = torch.as_tensor(
            np.asarray(batch.characteristics, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        context_features = None
        if batch.context_features is not None:
            context_features = torch.as_tensor(
                np.asarray(batch.context_features, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )

        # Seed before network construction so initialization and training are
        # reproducible for a fixed configuration.
        seed_torch(torch, self.config.seed, device)
        np.random.seed(self.config.seed)

        stochastic_discount_factor_net = nn.StochasticDiscountFactorNetwork(
            n_asset_features=batch.characteristics.shape[2],
            n_context_features=0 if context_features is None else context_features.shape[1],
            state_dim=self.config.state_dim_sdf,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        ).to(device)
        moment_net = nn.MomentNetwork(
            n_asset_features=batch.characteristics.shape[2],
            n_context_features=0 if context_features is None else context_features.shape[1],
            n_instruments=self.config.n_instruments,
            state_dim=self.config.state_dim_moment,
            dropout=self.config.dropout,
        ).to(device)

        sdf_optimizer = torch.optim.Adam(
            stochastic_discount_factor_net.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        moment_optimizer = torch.optim.Adam(
            moment_net.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        checkpoint_epochs = tuple(
            resolve_checkpoint_epochs(
                max(self.config.n_epochs_unc, self.config.n_epochs_cond),
                checkpoint_interval=self.config.checkpoint_interval,
                checkpoint_epochs=list(self.config.checkpoint_epochs) or None,
            )
        )
        self._checkpoint_states = defaultdict(dict)
        history: list[dict[str, float | str]] = []

        val_tensors = (
            _prepare_sdf_tensors(validation_batch, torch, device)
            if validation_batch is not None
            else None
        )
        best_val_loss_unc = float("inf")
        best_val_sharpe_unc = float("-inf")
        best_val_loss_cond = float("inf")
        best_val_sharpe_cond = float("-inf")
        epochs_since_improve = 0

        for phase_epoch in range(1, self.config.n_epochs_unc + 1):
            stochastic_discount_factor_net.train()
            weights, _ = stochastic_discount_factor_net(
                asset_features,
                context_features=context_features,
                mask=mask,
            )
            loss, stochastic_discount_factor_values = nn.unconditional_loss(
                weights, returns, mask, n_obs_per_asset
            )
            sdf_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            sdf_optimizer.step()

            sharpe = nn.compute_sharpe(stochastic_discount_factor_values)
            history.append(
                {
                    "epoch": float(phase_epoch),
                    "phase": "unconditional",
                    "train_loss": float(loss.item()),
                    "train_sharpe": float(sharpe.item()),
                }
            )
            self._maybe_store_checkpoint(
                phase="unconditional",
                phase_epoch=phase_epoch,
                checkpoint_epochs=checkpoint_epochs,
                stochastic_discount_factor_net=stochastic_discount_factor_net,
                moment_net=moment_net,
            )

            if val_tensors is not None and phase_epoch > self.config.burn_in_epochs:
                val_loss, val_sharpe = _validation_metrics(
                    stochastic_discount_factor_net=stochastic_discount_factor_net,
                    moment_net=moment_net,
                    val_tensors=val_tensors,
                    phase="unconditional",
                    nn=nn,
                    torch=torch,
                )
                improved = False
                if val_loss < best_val_loss_unc:
                    best_val_loss_unc = val_loss
                    self._checkpoint_states[VAL_BEST_LOSS_UNCONDITIONAL] = _capture_state(
                        stochastic_discount_factor_net, moment_net
                    )
                    improved = True
                if val_sharpe > best_val_sharpe_unc:
                    best_val_sharpe_unc = val_sharpe
                    self._checkpoint_states[VAL_BEST_SHARPE_UNCONDITIONAL] = _capture_state(
                        stochastic_discount_factor_net, moment_net
                    )
                    improved = True
                epochs_since_improve = 0 if improved else epochs_since_improve + 1
                if patience is not None and epochs_since_improve >= patience:
                    break

        epochs_since_improve = 0
        best_moment_state = deepcopy(moment_net.state_dict())
        best_moment_loss = float("-inf")
        for _ in range(self.config.n_epochs_moment):
            stochastic_discount_factor_net.eval()
            moment_net.train()
            with torch.no_grad():
                weights, _ = stochastic_discount_factor_net(
                    asset_features,
                    context_features=context_features,
                    mask=mask,
                )
            instruments, _ = moment_net(asset_features, context_features=context_features)
            loss, _ = nn.conditional_loss(weights, instruments, returns, mask, n_obs_per_asset)
            moment_optimizer.zero_grad(set_to_none=True)
            (-loss).backward()
            moment_optimizer.step()
            moment_loss = float(loss.item())
            if moment_loss > best_moment_loss:
                best_moment_loss = moment_loss
                best_moment_state = deepcopy(moment_net.state_dict())

        moment_net.load_state_dict(best_moment_state)

        for phase_epoch in range(1, self.config.n_epochs_cond + 1):
            stochastic_discount_factor_net.train()
            moment_net.eval()
            weights, _ = stochastic_discount_factor_net(
                asset_features,
                context_features=context_features,
                mask=mask,
            )
            with torch.no_grad():
                instruments, _ = moment_net(asset_features, context_features=context_features)
            loss, stochastic_discount_factor_values = nn.conditional_loss(
                weights, instruments, returns, mask, n_obs_per_asset
            )
            sdf_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            sdf_optimizer.step()

            sharpe = nn.compute_sharpe(stochastic_discount_factor_values)
            history.append(
                {
                    "epoch": float(phase_epoch),
                    "phase": "conditional",
                    "train_loss": float(loss.item()),
                    "train_sharpe": float(sharpe.item()),
                }
            )
            self._maybe_store_checkpoint(
                phase="conditional",
                phase_epoch=phase_epoch,
                checkpoint_epochs=checkpoint_epochs,
                stochastic_discount_factor_net=stochastic_discount_factor_net,
                moment_net=moment_net,
            )

            if val_tensors is not None and phase_epoch > self.config.burn_in_epochs:
                val_loss, val_sharpe = _validation_metrics(
                    stochastic_discount_factor_net=stochastic_discount_factor_net,
                    moment_net=moment_net,
                    val_tensors=val_tensors,
                    phase="conditional",
                    nn=nn,
                    torch=torch,
                )
                improved = False
                if val_loss < best_val_loss_cond:
                    best_val_loss_cond = val_loss
                    self._checkpoint_states[VAL_BEST_LOSS_CONDITIONAL] = _capture_state(
                        stochastic_discount_factor_net, moment_net
                    )
                    improved = True
                if val_sharpe > best_val_sharpe_cond:
                    best_val_sharpe_cond = val_sharpe
                    self._checkpoint_states[VAL_BEST_SHARPE_CONDITIONAL] = _capture_state(
                        stochastic_discount_factor_net, moment_net
                    )
                    improved = True
                epochs_since_improve = 0 if improved else epochs_since_improve + 1
                if patience is not None and epochs_since_improve >= patience:
                    break

        self._history = tuple(history)
        self._asset_ids = batch.asset_ids
        self._n_characteristics = batch.characteristics.shape[2]
        self._n_context_features = (
            0 if batch.context_features is None else batch.context_features.shape[1]
        )
        self._mark_fitted()

        return FitSummary(
            converged=True,
            train_metrics={
                "n_train_periods": float(batch.n_periods),
                "n_checkpoints": float(len(self.available_checkpoints)),
            },
            best_epoch=_select_checkpoint(
                checkpoint=None,
                configured_default=self.config.default_checkpoint,
                available=self.available_checkpoints,
                n_epochs_unc=self.config.n_epochs_unc,
            ),
            history=self._history,
            notes=(
                "Phase-aware stochastic discount factor training with weight-native checkpoint extraction.",
            ),
        )

    def extract(
        self,
        batch: CrossSectionBatch,
        *,
        checkpoint: SDFCheckpoint | int | None = None,
    ) -> StochasticDiscountFactorState:
        if not self.is_fitted or self._n_characteristics is None or not self._checkpoint_states:
            raise RuntimeError("StochasticDiscountFactorModel must be fitted before extract()")
        if batch.characteristics.shape[2] != self._n_characteristics:
            raise ValueError(
                "characteristics feature dimension does not match the fitted stochastic discount "
                "factor model; "
                f"expected {self._n_characteristics}, got {batch.characteristics.shape[2]}"
            )
        n_context_features = (
            0 if batch.context_features is None else batch.context_features.shape[1]
        )
        if n_context_features != self._n_context_features:
            raise ValueError(
                "context feature dimension does not match the fitted stochastic discount factor "
                "model; "
                f"expected {self._n_context_features}, got {n_context_features}"
            )

        torch = import_torch()
        nn = _import_stochastic_discount_factor_nn()
        device = resolve_device(torch, self.config.device)
        selected_checkpoint = _select_checkpoint(
            checkpoint=checkpoint,
            configured_default=self.config.default_checkpoint,
            available=self.available_checkpoints,
            n_epochs_unc=self.config.n_epochs_unc,
        )
        checkpoint_state = self._checkpoint_states[selected_checkpoint]

        stochastic_discount_factor_net = nn.StochasticDiscountFactorNetwork(
            n_asset_features=self._n_characteristics,
            n_context_features=self._n_context_features,
            state_dim=self.config.state_dim_sdf,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        ).to(device)
        stochastic_discount_factor_net.load_state_dict(
            deepcopy(checkpoint_state["stochastic_discount_factor"])
        )
        stochastic_discount_factor_net.eval()

        asset_features = torch.as_tensor(
            np.asarray(batch.characteristics, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        context_features = None
        if batch.context_features is not None:
            context_features = torch.as_tensor(
                np.asarray(batch.context_features, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )
        mask = _resolve_mask(batch, torch, device)

        with torch.no_grad():
            weights_flat, _ = stochastic_discount_factor_net(
                asset_features,
                context_features=context_features,
                mask=mask,
            )
        asset_weights = _reshape_weight_panel(
            weights=weights_flat.detach().cpu().numpy().astype(np.float64),
            mask=mask.detach().cpu().numpy(),
            shape=(batch.n_periods, batch.n_assets),
        )

        sdf_values = None
        if batch.returns is not None:
            returns = torch.as_tensor(np.asarray(batch.returns, dtype=np.float32), device=device)
            returns = torch.where(mask, returns, torch.zeros_like(returns))
            with torch.no_grad():
                sdf_series = nn.construct_stochastic_discount_factor(returns, weights_flat, mask)
            sdf_values = sdf_series.detach().cpu().numpy().astype(np.float64)

        return StochasticDiscountFactorState(
            asset_weights=asset_weights,
            sdf_values=sdf_values,
            checkpoint_epoch=selected_checkpoint,
            timestamps=batch.timestamps,
            asset_ids=batch.asset_ids or self._asset_ids,
            metadata={
                "model_name": self.config.model_name,
                "available_checkpoints": self.available_checkpoints,
                "native_output": "weights",
            },
        )

    def _maybe_store_checkpoint(
        self,
        *,
        phase: str,
        phase_epoch: int,
        checkpoint_epochs: tuple[int, ...],
        stochastic_discount_factor_net: Any,
        moment_net: Any,
    ) -> None:
        if phase_epoch <= self.config.burn_in_epochs or phase_epoch not in checkpoint_epochs:
            return
        self._checkpoint_states[(phase, phase_epoch)] = _capture_state(
            stochastic_discount_factor_net, moment_net
        )


def _resolve_mask(batch: CrossSectionBatch, torch: Any, device: Any) -> Any:
    base_mask = np.isfinite(batch.characteristics).all(axis=2)
    if batch.mask is not None:
        base_mask = base_mask & np.asarray(batch.mask, dtype=bool)
    if batch.returns is not None:
        base_mask = base_mask & np.isfinite(batch.returns)
    return torch.as_tensor(base_mask, dtype=torch.bool, device=device)


def _reshape_weight_panel(
    *,
    weights: np.ndarray,
    mask: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    panel = np.full(shape, np.nan, dtype=np.float64)
    cursor = 0
    for date_idx in range(shape[0]):
        n_valid = int(mask[date_idx].sum())
        if n_valid > 0:
            panel[date_idx, mask[date_idx]] = weights[cursor : cursor + n_valid]
        cursor += n_valid
    return panel


def _checkpoint_sort_key(checkpoint: SDFCheckpoint) -> tuple[int, int]:
    phase, epoch = checkpoint
    phase_order = 0 if phase == "unconditional" else 1
    return phase_order, epoch


def _select_checkpoint(
    *,
    checkpoint: SDFCheckpoint | int | None,
    configured_default: SDFCheckpoint | int | None,
    available: tuple[SDFCheckpoint, ...],
    n_epochs_unc: int,
) -> SDFCheckpoint:
    if not available:
        raise ValueError("available_checkpoints is empty")
    legacy_lookup = _legacy_checkpoint_lookup(available, n_epochs_unc=n_epochs_unc)
    selected = checkpoint if checkpoint is not None else configured_default
    if selected is not None:
        if isinstance(selected, tuple):
            if selected not in available:
                raise ValueError(
                    f"checkpoint={selected!r} is not in available_checkpoints={available}"
                )
            return selected
        if int(selected) not in legacy_lookup:
            raise ValueError(f"checkpoint={selected!r} is not in available_checkpoints={available}")
        return legacy_lookup[int(selected)]

    positive = [key for key in available if key[1] > 0]
    return positive[-1] if positive else available[-1]


def _legacy_checkpoint_lookup(
    available: tuple[SDFCheckpoint, ...],
    *,
    n_epochs_unc: int,
) -> dict[int, SDFCheckpoint]:
    lookup: dict[int, SDFCheckpoint] = {}
    sentinel_aliases = {
        -4: VAL_BEST_SHARPE_CONDITIONAL,
        -3: VAL_BEST_LOSS_CONDITIONAL,
        -2: VAL_BEST_SHARPE_UNCONDITIONAL,
        -1: VAL_BEST_LOSS_UNCONDITIONAL,
    }
    for legacy, key in sentinel_aliases.items():
        if key in available:
            lookup[legacy] = key
    for key in available:
        phase, phase_epoch = key
        if phase_epoch <= 0:
            continue
        legacy_epoch = phase_epoch if phase == "unconditional" else n_epochs_unc + phase_epoch
        lookup[legacy_epoch] = key
    return lookup


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _capture_state(
    stochastic_discount_factor_net: Any, moment_net: Any
) -> dict[str, dict[str, Any]]:
    return {
        "stochastic_discount_factor": _cpu_state_dict(stochastic_discount_factor_net),
        "moment": _cpu_state_dict(moment_net),
    }


def _prepare_sdf_tensors(batch: CrossSectionBatch, torch: Any, device: Any) -> tuple[Any, ...]:
    returns_raw = torch.as_tensor(np.asarray(batch.returns, dtype=np.float32), device=device)
    mask = _resolve_mask(batch, torch, device)
    returns = torch.where(mask, returns_raw, torch.zeros_like(returns_raw))
    n_obs_per_asset = mask.float().sum(dim=0)
    if int(mask.sum().item()) == 0:
        raise ValueError("validation_batch contains no valid SDF validation observations")
    asset_features = torch.as_tensor(
        np.asarray(batch.characteristics, dtype=np.float32), dtype=torch.float32, device=device
    )
    context_features = None
    if batch.context_features is not None:
        context_features = torch.as_tensor(
            np.asarray(batch.context_features, dtype=np.float32), dtype=torch.float32, device=device
        )
    return returns, mask, n_obs_per_asset, asset_features, context_features


def _validation_metrics(
    *,
    stochastic_discount_factor_net: Any,
    moment_net: Any,
    val_tensors: tuple[Any, ...],
    phase: str,
    nn: Any,
    torch: Any,
) -> tuple[float, float]:
    """CPZ per-phase validation loss and SDF-portfolio Sharpe on the full val panel."""
    returns, mask, n_obs_per_asset, asset_features, context_features = val_tensors
    was_training = stochastic_discount_factor_net.training
    stochastic_discount_factor_net.eval()
    with torch.no_grad():
        weights, _ = stochastic_discount_factor_net(
            asset_features, context_features=context_features, mask=mask
        )
        if phase == "unconditional":
            loss, sdf_values = nn.unconditional_loss(weights, returns, mask, n_obs_per_asset)
        else:
            moment_net.eval()
            instruments, _ = moment_net(asset_features, context_features=context_features)
            loss, sdf_values = nn.conditional_loss(
                weights, instruments, returns, mask, n_obs_per_asset
            )
        sharpe = nn.compute_sharpe(sdf_values)
    if was_training:
        stochastic_discount_factor_net.train()
    return float(loss.item()), float(sharpe.item())


def _import_stochastic_discount_factor_nn() -> Any:
    from ml4t.models._internal import stochastic_discount_factor_nn

    return stochastic_discount_factor_nn
