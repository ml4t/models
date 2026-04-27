"""Supervised autoencoder structural extractor."""

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
from ml4t.models.api import PanelBatch
from ml4t.models.configs import SAEConfig
from ml4t.models.latent_factors.base import BaseLatentFactorModel
from ml4t.models.types import CrossSectionBatch, FitSummary, LatentFactorState


class SAEModel(BaseLatentFactorModel[SAEConfig]):
    """Supervised autoencoder with checkpoint-aware latent extraction."""

    def __init__(self, config: SAEConfig) -> None:
        super().__init__(config)
        self._checkpoint_states: dict[int, list[dict[str, Any]]] = {}
        self._asset_ids: tuple[str, ...] = ()
        self._n_characteristics: int | None = None
        self._history: tuple[dict[str, float | str], ...] = ()

    @property
    def available_checkpoints(self) -> tuple[int, ...]:
        return tuple(sorted(self._checkpoint_states))

    def fit(self, batch: PanelBatch) -> FitSummary:
        cross_section = _require_cross_section(batch)
        if cross_section.returns is None:
            raise ValueError("SAE requires returns in the training batch")
        if self.config.n_factors < 1:
            raise ValueError(f"n_factors must be positive; got {self.config.n_factors}")
        if self.config.n_ensemble < 1:
            raise ValueError(f"n_ensemble must be positive; got {self.config.n_ensemble}")

        torch = import_torch()
        nn = _import_sae_nn()
        device = resolve_device(torch, self.config.device)
        checkpoint_epochs = tuple(
            resolve_checkpoint_epochs(
                self.config.n_epochs,
                checkpoint_interval=self.config.checkpoint_interval,
                checkpoint_epochs=list(self.config.checkpoint_epochs) or None,
            )
        )
        hidden_units = _resolve_hidden_units(
            self.config.hidden_units, n_factors=self.config.n_factors
        )
        dropout_rates = _resolve_dropout_rates(self.config.dropout_rates)

        chars_train = torch.as_tensor(
            np.asarray(cross_section.characteristics, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        returns_train = torch.as_tensor(
            np.asarray(cross_section.returns, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        mask_train = _resolve_mask(cross_section)
        output_activation = "sigmoid" if self.config.task_type == "classification" else "linear"

        self._checkpoint_states = defaultdict(list)
        loss_sums = dict.fromkeys(checkpoint_epochs, 0.0)

        for ensemble_idx in range(self.config.n_ensemble):
            seed = self.config.seed + ensemble_idx
            np.random.seed(seed)
            seed_torch(torch, seed, device)

            model = nn.SupervisedAutoencoder(
                n_features=cross_section.characteristics.shape[2],
                n_labels=1,
                hidden_units=hidden_units,
                dropout_rates=dropout_rates,
                noise_std=self.config.noise_std,
                output_activation=output_activation,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)

            for epoch in range(1, self.config.n_epochs + 1):
                model.train()
                epoch_loss = 0.0
                n_batches = 0

                for date_idx in range(cross_section.n_periods):
                    valid = (
                        mask_train[date_idx] & torch.isfinite(returns_train[date_idx]).cpu().numpy()
                    )
                    if not valid.any():
                        continue

                    features_t = chars_train[date_idx, valid]
                    target_t = returns_train[date_idx, valid]
                    decoded, aux_pred, main_pred = model(features_t)

                    reconstruction_loss = torch.nn.functional.mse_loss(decoded, features_t)
                    if self.config.task_type == "classification":
                        target_t = target_t.clamp(0.0, 1.0)
                        main_loss = torch.nn.functional.binary_cross_entropy(
                            main_pred.squeeze(-1),
                            target_t,
                        )
                        aux_loss = torch.nn.functional.binary_cross_entropy(
                            aux_pred.squeeze(-1),
                            target_t,
                        )
                    else:
                        main_loss = torch.nn.functional.mse_loss(main_pred.squeeze(-1), target_t)
                        aux_loss = torch.nn.functional.mse_loss(aux_pred.squeeze(-1), target_t)

                    loss = (
                        reconstruction_loss
                        + self.config.alpha * main_loss
                        + self.config.aux_weight * aux_loss
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += float(loss.item())
                    n_batches += 1

                mean_loss = epoch_loss / max(n_batches, 1)
                if epoch in checkpoint_epochs:
                    self._checkpoint_states[epoch].append(_cpu_state_dict(model))
                    loss_sums[epoch] += mean_loss

        history: list[dict[str, float | str]] = []
        for epoch in checkpoint_epochs:
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": loss_sums[epoch] / self.config.n_ensemble,
                }
            )
        self._history = tuple(history)
        self._asset_ids = cross_section.asset_ids
        self._n_characteristics = cross_section.characteristics.shape[2]
        self._mark_fitted()

        return FitSummary(
            converged=True,
            train_metrics={
                "n_train_periods": float(cross_section.n_periods),
                "n_checkpoints": float(len(checkpoint_epochs)),
                "n_ensemble": float(self.config.n_ensemble),
            },
            best_epoch=_default_checkpoint(self.config.default_checkpoint, checkpoint_epochs),
            history=self._history,
            notes=("Supervised autoencoder checkpoints store bottleneck factor exposures.",),
        )

    def extract(
        self,
        batch: PanelBatch,
        *,
        checkpoint: int | None = None,
    ) -> LatentFactorState:
        cross_section = _require_cross_section(batch)
        if not self.is_fitted or self._n_characteristics is None or not self._checkpoint_states:
            raise RuntimeError("SAE model must be fitted before extract()")
        if cross_section.characteristics.shape[2] != self._n_characteristics:
            raise ValueError(
                "characteristics feature dimension does not match fitted SAE model; "
                f"expected {self._n_characteristics}, got {cross_section.characteristics.shape[2]}"
            )

        torch = import_torch()
        nn = _import_sae_nn()
        device = resolve_device(torch, self.config.device)
        selected_checkpoint = select_checkpoint_epoch(
            checkpoint=checkpoint,
            configured_default=self.config.default_checkpoint,
            available=self.available_checkpoints,
        )
        hidden_units = _resolve_hidden_units(
            self.config.hidden_units, n_factors=self.config.n_factors
        )
        dropout_rates = _resolve_dropout_rates(self.config.dropout_rates)
        output_activation = "sigmoid" if self.config.task_type == "classification" else "linear"

        mask = _resolve_mask(cross_section)
        factor_targets = None
        if cross_section.factor_returns is not None:
            factor_targets = np.asarray(cross_section.factor_returns, dtype=np.float64)
        elif cross_section.returns is not None:
            factor_targets = np.asarray(cross_section.returns, dtype=np.float64)

        ensemble_betas: list[np.ndarray] = []
        ensemble_factors: list[np.ndarray] = []
        for state_dict in self._checkpoint_states[selected_checkpoint]:
            model = nn.SupervisedAutoencoder(
                n_features=self._n_characteristics,
                n_labels=1,
                hidden_units=hidden_units,
                dropout_rates=dropout_rates,
                noise_std=self.config.noise_std,
                output_activation=output_activation,
            ).to(device)
            model.load_state_dict(deepcopy(state_dict))
            model.eval()
            betas_t, factors_t = _extract_sae_state(
                torch=torch,
                model=model,
                characteristics=cross_section.characteristics,
                returns=factor_targets,
                mask=mask,
                n_factors=self.config.n_factors,
                device=device,
            )
            ensemble_betas.append(betas_t)
            if factors_t is not None:
                ensemble_factors.append(factors_t)

        asset_betas = np.nanmean(np.stack(ensemble_betas, axis=0), axis=0)
        factor_returns = None
        if factor_targets is not None and ensemble_factors:
            factor_returns = np.nanmean(np.stack(ensemble_factors, axis=0), axis=0)

        return LatentFactorState(
            asset_betas=asset_betas,
            factor_returns=factor_returns,
            checkpoint_epoch=selected_checkpoint,
            timestamps=cross_section.timestamps,
            asset_ids=cross_section.asset_ids or self._asset_ids,
            metadata={
                "model_name": self.config.model_name,
                "persistent_entities": False,
                "task_type": self.config.task_type,
                "available_checkpoints": self.available_checkpoints,
            },
        )


def _require_cross_section(batch: PanelBatch) -> CrossSectionBatch:
    if not isinstance(batch, CrossSectionBatch):
        raise TypeError("SAE requires CrossSectionBatch input")
    return batch


def _resolve_hidden_units(
    hidden_units: tuple[int, ...] | None,
    *,
    n_factors: int,
) -> tuple[int, ...]:
    if hidden_units is None:
        return (n_factors, 96, 896, 448, 448, 256)
    resolved = list(hidden_units)
    if len(resolved) != 6:
        raise ValueError(f"hidden_units must have 6 entries; got {len(resolved)}")
    resolved[0] = n_factors
    return tuple(int(unit) for unit in resolved)


def _resolve_dropout_rates(dropout_rates: tuple[float, ...] | None) -> tuple[float, ...]:
    if dropout_rates is None:
        return (0.035, 0.038, 0.424, 0.104, 0.492, 0.320, 0.272, 0.438)
    if len(dropout_rates) != 8:
        raise ValueError(f"dropout_rates must have 8 entries; got {len(dropout_rates)}")
    return tuple(float(rate) for rate in dropout_rates)


def _resolve_mask(batch: CrossSectionBatch) -> np.ndarray:
    if batch.mask is None:
        return np.asarray(np.isfinite(batch.characteristics).all(axis=2), dtype=bool)
    mask = np.asarray(batch.mask, dtype=bool)
    return mask & np.isfinite(batch.characteristics).all(axis=2)


def _default_checkpoint(configured_default: int | None, available: tuple[int, ...]) -> int:
    return select_checkpoint_epoch(
        checkpoint=None,
        configured_default=configured_default,
        available=available,
    )


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _extract_sae_state(
    *,
    torch: Any,
    model: Any,
    characteristics: np.ndarray,
    returns: np.ndarray | None,
    mask: np.ndarray,
    n_factors: int,
    device: Any,
) -> tuple[np.ndarray, np.ndarray | None]:
    asset_betas = np.full(
        (characteristics.shape[0], characteristics.shape[1], n_factors),
        np.nan,
        dtype=np.float64,
    )
    factor_returns = None
    if returns is not None:
        factor_returns = np.full((characteristics.shape[0], n_factors), np.nan, dtype=np.float64)

    with torch.no_grad():
        for date_idx in range(characteristics.shape[0]):
            valid = mask[date_idx]
            if not valid.any():
                continue
            features_t = torch.as_tensor(
                characteristics[date_idx, valid],
                dtype=torch.float32,
                device=device,
            )
            betas_t = model.get_betas(features_t).detach().cpu().numpy().astype(np.float64)
            asset_betas[date_idx, valid] = betas_t

            if returns is not None:
                assert factor_returns is not None
                returns_t = returns[date_idx, valid]
                gram = betas_t.T @ betas_t + 1e-6 * np.eye(n_factors, dtype=np.float64)
                rhs = betas_t.T @ returns_t.astype(np.float64)
                factor_returns[date_idx] = _solve_linear_system(gram, rhs)

    return asset_betas, factor_returns


def _solve_linear_system(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def _import_sae_nn() -> Any:
    from ml4t.models._internal import sae_nn

    return sae_nn
