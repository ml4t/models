"""Conditional autoencoder structural extractor."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

import numpy as np

from ml4t.models._internal.latent_factor_utils import select_checkpoint_epoch
from ml4t.models._internal.torch_runtime import import_torch, resolve_device, seed_torch
from ml4t.models.api import PanelBatch
from ml4t.models.configs import CAEConfig
from ml4t.models.latent_factors.base import BaseLatentFactorModel
from ml4t.models.types import CrossSectionBatch, FitSummary, LatentFactorState


class CAEModel(BaseLatentFactorModel[CAEConfig]):
    """Conditional autoencoder with checkpoint-aware structural extraction."""

    def __init__(self, config: CAEConfig) -> None:
        super().__init__(config)
        self._checkpoint_states: dict[int, list[dict[str, Any]]] = {}
        self._asset_ids: tuple[str, ...] = ()
        self._n_characteristics: int | None = None
        self._n_instruments: int | None = None
        self._history: tuple[dict[str, float | str], ...] = ()

    @property
    def available_checkpoints(self) -> tuple[int, ...]:
        return tuple(sorted(self._checkpoint_states))

    def fit(self, batch: PanelBatch) -> FitSummary:
        cross_section = _require_cross_section(batch)
        if cross_section.returns is None:
            raise ValueError("CAE requires returns in the training batch")
        if self.config.n_factors < 1:
            raise ValueError(f"n_factors must be positive; got {self.config.n_factors}")
        if self.config.n_ensemble < 1:
            raise ValueError(f"n_ensemble must be positive; got {self.config.n_ensemble}")
        if self.config.task_type == "classification" and cross_section.factor_returns is None:
            raise ValueError(
                "Classification CAE requires factor_returns for managed-portfolio construction"
            )

        torch = import_torch()
        nn = _import_cae_nn()
        checkpoint_epochs = _resolve_training_checkpoints(self.config)
        device = resolve_device(torch, self.config.device)

        portfolio_returns = _portfolio_returns(cross_section)
        assert portfolio_returns is not None
        managed_portfolios = _compute_managed_portfolios(
            characteristics=cross_section.characteristics,
            returns=portfolio_returns,
        )

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
        portfolios_train = torch.as_tensor(
            managed_portfolios,
            dtype=torch.float32,
            device=device,
        )
        mask_train = _resolve_mask(cross_section)

        self._checkpoint_states = defaultdict(list)
        loss_sums = dict.fromkeys(checkpoint_epochs, 0.0)

        for ensemble_idx in range(self.config.n_ensemble):
            seed = self.config.seed + ensemble_idx
            np.random.seed(seed)
            seed_torch(torch, seed, device)

            model = nn.ConditionalAutoencoder(
                n_characteristics=cross_section.characteristics.shape[2],
                n_instruments=managed_portfolios.shape[2],
                n_factors=self.config.n_factors,
                hidden_units=self.config.hidden_units,
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
                    portfolios_t = portfolios_train[date_idx, valid]
                    scores_t = model(features_t, portfolios_t)

                    if self.config.task_type == "classification":
                        main_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            scores_t,
                            target_t.clamp(0.0, 1.0),
                        )
                    else:
                        main_loss = torch.nn.functional.mse_loss(scores_t, target_t)

                    loss = main_loss + nn.l1_regularization(model, self.config.lambda_l1)
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
        self._n_instruments = managed_portfolios.shape[2]
        self._mark_fitted()

        return FitSummary(
            converged=True,
            train_metrics={
                "n_train_periods": float(cross_section.n_periods),
                "n_checkpoints": float(len(checkpoint_epochs)),
                "n_ensemble": float(self.config.n_ensemble),
            },
            best_epoch=_default_checkpoint(self.config, checkpoint_epochs),
            history=self._history,
            notes=("Neural betas and linear factors stored at configurable checkpoints.",),
        )

    def extract(
        self,
        batch: PanelBatch,
        *,
        checkpoint: int | None = None,
    ) -> LatentFactorState:
        cross_section = _require_cross_section(batch)
        if (
            not self.is_fitted
            or self._n_characteristics is None
            or self._n_instruments is None
            or not self._checkpoint_states
        ):
            raise RuntimeError("CAE model must be fitted before extract()")
        if cross_section.characteristics.shape[2] != self._n_characteristics:
            raise ValueError(
                "characteristics feature dimension does not match fitted CAE model; "
                f"expected {self._n_characteristics}, got {cross_section.characteristics.shape[2]}"
            )

        torch = import_torch()
        nn = _import_cae_nn()
        selected_checkpoint = _select_checkpoint(
            checkpoint=checkpoint,
            configured_default=self.config.default_checkpoint,
            available=self.available_checkpoints,
        )
        device = resolve_device(torch, self.config.device)
        mask = _resolve_mask(cross_section)
        portfolio_returns = _portfolio_returns(cross_section)
        managed_portfolios = None
        if portfolio_returns is not None:
            managed_portfolios = _compute_managed_portfolios(
                characteristics=cross_section.characteristics,
                returns=portfolio_returns,
            )

        ensemble_betas: list[np.ndarray] = []
        ensemble_factors: list[np.ndarray] = []
        factor_returns_available = managed_portfolios is not None

        for state_dict in self._checkpoint_states[selected_checkpoint]:
            model = nn.ConditionalAutoencoder(
                n_characteristics=self._n_characteristics,
                n_instruments=self._n_instruments,
                n_factors=self.config.n_factors,
                hidden_units=self.config.hidden_units,
            ).to(device)
            model.load_state_dict(deepcopy(state_dict))
            model.eval()
            betas_t, factors_t = _extract_cae_state(
                torch=torch,
                model=model,
                characteristics=cross_section.characteristics,
                managed_portfolios=managed_portfolios,
                mask=mask,
                n_factors=self.config.n_factors,
                device=device,
            )
            ensemble_betas.append(betas_t)
            if factors_t is not None:
                ensemble_factors.append(factors_t)
            else:
                factor_returns_available = False

        asset_betas = np.nanmean(np.stack(ensemble_betas, axis=0), axis=0)
        factor_returns = None
        if factor_returns_available and ensemble_factors:
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
        raise TypeError("CAE requires CrossSectionBatch input")
    return batch


def _resolve_training_checkpoints(config: CAEConfig) -> tuple[int, ...]:
    from ml4t.models._internal.latent_factor_utils import resolve_checkpoint_epochs

    checkpoints = resolve_checkpoint_epochs(
        config.n_epochs,
        checkpoint_interval=config.checkpoint_interval,
        checkpoint_epochs=list(config.checkpoint_epochs) or None,
    )
    return tuple(checkpoints)


def _default_checkpoint(config: CAEConfig, available: tuple[int, ...]) -> int:
    if config.default_checkpoint is not None:
        if config.default_checkpoint not in available:
            raise ValueError(
                f"default_checkpoint={config.default_checkpoint} is not in {available}"
            )
        return config.default_checkpoint
    return available[-1]


def _select_checkpoint(
    *,
    checkpoint: int | None,
    configured_default: int | None,
    available: tuple[int, ...],
) -> int:
    return select_checkpoint_epoch(
        checkpoint=checkpoint,
        configured_default=configured_default,
        available=available,
    )


def _portfolio_returns(batch: CrossSectionBatch) -> np.ndarray | None:
    if batch.factor_returns is not None:
        return np.asarray(batch.factor_returns, dtype=np.float64)
    if batch.returns is None:
        return None
    return np.asarray(batch.returns, dtype=np.float64)


def _resolve_mask(batch: CrossSectionBatch) -> np.ndarray:
    if batch.mask is None:
        return np.asarray(np.isfinite(batch.characteristics).all(axis=2), dtype=bool)
    mask = np.asarray(batch.mask, dtype=bool)
    return mask & np.isfinite(batch.characteristics).all(axis=2)


def _compute_managed_portfolios(
    *,
    characteristics: np.ndarray,
    returns: np.ndarray,
) -> np.ndarray:
    from ml4t.models._internal.latent_factor_utils import compute_managed_portfolios

    return compute_managed_portfolios(
        np.asarray(characteristics, dtype=np.float64),
        np.asarray(returns, dtype=np.float64),
    )


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _extract_cae_state(
    *,
    torch: Any,
    model: Any,
    characteristics: np.ndarray,
    managed_portfolios: np.ndarray | None,
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
    if managed_portfolios is not None:
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

            if managed_portfolios is not None:
                assert factor_returns is not None
                portfolios_t = torch.as_tensor(
                    managed_portfolios[date_idx, valid][:1],
                    dtype=torch.float32,
                    device=device,
                )
                factor_t = model.get_factors(portfolios_t).squeeze(0)
                factor_returns[date_idx] = factor_t.detach().cpu().numpy().astype(np.float64)

    return asset_betas, factor_returns


def _import_cae_nn() -> Any:
    from ml4t.models._internal import cae_nn

    return cae_nn
