"""Phase-aware stochastic discount factor model."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

import numpy as np

from ml4t.models._internal.latent_factor_utils import (
    resolve_checkpoint_epochs,
    select_checkpoint_epoch,
)
from ml4t.models._internal.torch_runtime import import_torch, resolve_device, seed_torch
from ml4t.models.configs import SDFConfig
from ml4t.models.sdf.base import BaseSDFModel
from ml4t.models.types import CrossSectionBatch, FitSummary, SDFState


class SDFModel(BaseSDFModel):
    """Stochastic discount factor model with weight-native structural outputs."""

    def __init__(self, config: SDFConfig) -> None:
        super().__init__(config)
        self._checkpoint_states: dict[int, dict[str, dict[str, Any]]] = {}
        self._asset_ids: tuple[str, ...] = ()
        self._n_characteristics: int | None = None
        self._n_context_features: int = 0
        self._history: tuple[dict[str, float | str], ...] = ()

    @property
    def available_checkpoints(self) -> tuple[int, ...]:
        return tuple(sorted(self._checkpoint_states))

    def fit(self, batch: CrossSectionBatch) -> FitSummary:
        if batch.returns is None:
            raise ValueError("SDF requires returns in the training batch")
        if self.config.output_mode != "weights":
            raise ValueError("SDFModel is weight-native; use a mapper for expected-return output")
        if self.config.expected_return_mapper != "linear":
            raise ValueError("expected_return_mapper must be 'linear'")

        torch = import_torch()
        nn = _import_sdf_nn()
        device = resolve_device(torch, self.config.device)

        returns_raw = torch.as_tensor(np.asarray(batch.returns, dtype=np.float32), device=device)
        mask = _resolve_mask(batch, torch, device)
        returns = torch.where(mask, returns_raw, torch.zeros_like(returns_raw))
        n_obs_per_asset = mask.float().sum(dim=0)
        if int(mask.sum().item()) == 0:
            raise ValueError("SDF received no valid training observations")

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

        sdf_net = nn.SDFNetwork(
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

        seed_torch(torch, self.config.seed, device)
        np.random.seed(self.config.seed)

        sdf_optimizer = torch.optim.Adam(
            sdf_net.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        moment_optimizer = torch.optim.Adam(
            moment_net.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        total_epochs = self.config.n_epochs_unc + self.config.n_epochs_cond
        checkpoint_epochs = tuple(
            resolve_checkpoint_epochs(
                total_epochs,
                checkpoint_interval=self.config.checkpoint_interval,
                checkpoint_epochs=list(self.config.checkpoint_epochs) or None,
            )
        )
        self._checkpoint_states = defaultdict(dict)
        history: list[dict[str, float | str]] = []
        sdf_epoch = 0

        for _ in range(self.config.n_epochs_unc):
            sdf_epoch += 1
            sdf_net.train()
            weights, _ = sdf_net(asset_features, context_features=context_features, mask=mask)
            loss, sdf_values = nn.unconditional_loss(weights, returns, mask, n_obs_per_asset)
            sdf_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            sdf_optimizer.step()

            sharpe = nn.compute_sharpe(sdf_values)
            history.append(
                {
                    "epoch": float(sdf_epoch),
                    "phase": "unconditional",
                    "train_loss": float(loss.item()),
                    "train_sharpe": float(sharpe.item()),
                }
            )
            self._maybe_store_checkpoint(
                epoch=sdf_epoch,
                checkpoint_epochs=checkpoint_epochs,
                sdf_net=sdf_net,
                moment_net=moment_net,
            )

        best_moment_state = deepcopy(moment_net.state_dict())
        best_moment_loss = float("-inf")
        for _ in range(self.config.n_epochs_moment):
            sdf_net.eval()
            moment_net.train()
            with torch.no_grad():
                weights, _ = sdf_net(asset_features, context_features=context_features, mask=mask)
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

        for _ in range(self.config.n_epochs_cond):
            sdf_epoch += 1
            sdf_net.train()
            moment_net.eval()
            weights, _ = sdf_net(asset_features, context_features=context_features, mask=mask)
            with torch.no_grad():
                instruments, _ = moment_net(asset_features, context_features=context_features)
            loss, sdf_values = nn.conditional_loss(
                weights, instruments, returns, mask, n_obs_per_asset
            )
            sdf_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            sdf_optimizer.step()

            sharpe = nn.compute_sharpe(sdf_values)
            history.append(
                {
                    "epoch": float(sdf_epoch),
                    "phase": "conditional",
                    "train_loss": float(loss.item()),
                    "train_sharpe": float(sharpe.item()),
                }
            )
            self._maybe_store_checkpoint(
                epoch=sdf_epoch,
                checkpoint_epochs=checkpoint_epochs,
                sdf_net=sdf_net,
                moment_net=moment_net,
            )

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
                "n_checkpoints": float(len(checkpoint_epochs)),
            },
            best_epoch=select_checkpoint_epoch(
                checkpoint=None,
                configured_default=self.config.default_checkpoint,
                available=checkpoint_epochs,
            ),
            history=self._history,
            notes=("Phase-aware SDF training with weight-native checkpoint extraction.",),
        )

    def extract(
        self,
        batch: CrossSectionBatch,
        *,
        checkpoint: int | None = None,
    ) -> SDFState:
        if not self.is_fitted or self._n_characteristics is None or not self._checkpoint_states:
            raise RuntimeError("SDF model must be fitted before extract()")
        if batch.characteristics.shape[2] != self._n_characteristics:
            raise ValueError(
                "characteristics feature dimension does not match fitted SDF model; "
                f"expected {self._n_characteristics}, got {batch.characteristics.shape[2]}"
            )
        n_context_features = (
            0 if batch.context_features is None else batch.context_features.shape[1]
        )
        if n_context_features != self._n_context_features:
            raise ValueError(
                "context feature dimension does not match fitted SDF model; "
                f"expected {self._n_context_features}, got {n_context_features}"
            )

        torch = import_torch()
        nn = _import_sdf_nn()
        device = resolve_device(torch, self.config.device)
        selected_checkpoint = select_checkpoint_epoch(
            checkpoint=checkpoint,
            configured_default=self.config.default_checkpoint,
            available=self.available_checkpoints,
        )
        checkpoint_state = self._checkpoint_states[selected_checkpoint]

        sdf_net = nn.SDFNetwork(
            n_asset_features=self._n_characteristics,
            n_context_features=self._n_context_features,
            state_dim=self.config.state_dim_sdf,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        ).to(device)
        sdf_net.load_state_dict(deepcopy(checkpoint_state["sdf"]))
        sdf_net.eval()

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
            weights_flat, _ = sdf_net(asset_features, context_features=context_features, mask=mask)
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
                sdf_series = nn.construct_sdf(returns, weights_flat, mask)
            sdf_values = sdf_series.detach().cpu().numpy().astype(np.float64)

        return SDFState(
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
        epoch: int,
        checkpoint_epochs: tuple[int, ...],
        sdf_net: Any,
        moment_net: Any,
    ) -> None:
        if epoch <= self.config.burn_in_epochs or epoch not in checkpoint_epochs:
            return
        self._checkpoint_states[epoch] = {
            "sdf": _cpu_state_dict(sdf_net),
            "moment": _cpu_state_dict(moment_net),
        }


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


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _import_sdf_nn() -> Any:
    from ml4t.models._internal import sdf_nn

    return sdf_nn
