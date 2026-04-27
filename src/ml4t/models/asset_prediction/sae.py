"""Supervised autoencoder direct predictor."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from ml4t.models._internal.latent_factor_utils import (
    resolve_checkpoint_epochs,
    select_checkpoint_epoch,
)
from ml4t.models._internal.torch_runtime import import_torch, resolve_device, seed_torch
from ml4t.models.asset_prediction.base import BaseAssetPredictionModel
from ml4t.models.configs import SAEConfig
from ml4t.models.types import AssetSignalResult, CrossSectionBatch, FitSummary


class SAEModel(BaseAssetPredictionModel[SAEConfig]):
    """Checkpointed supervised autoencoder for direct asset prediction."""

    def __init__(self, config: SAEConfig) -> None:
        super().__init__(config)
        self._checkpoint_states: dict[int, dict[str, Any]] = {}
        self._n_features: int | None = None
        self._asset_ids: tuple[str, ...] = ()
        self._history: tuple[dict[str, float | str], ...] = ()

    @property
    def available_checkpoints(self) -> tuple[int, ...]:
        return tuple(sorted(self._checkpoint_states))

    def fit(
        self,
        batch: CrossSectionBatch,
        *,
        validation_batch: CrossSectionBatch | None = None,
    ) -> FitSummary:
        if batch.returns is None:
            raise ValueError("SAE training requires returns in the batch")

        torch = import_torch()
        nn = _import_sae_nn()
        device = resolve_device(torch, self.config.device)
        seed_torch(torch, self.config.seed, device)
        np.random.seed(self.config.seed)

        checkpoint_epochs = tuple(
            resolve_checkpoint_epochs(
                self.config.n_epochs,
                checkpoint_interval=self.config.checkpoint_interval,
                checkpoint_epochs=list(self.config.checkpoint_epochs) or None,
            )
        )
        model = nn.SupervisedAutoencoder(
            n_features=batch.characteristics.shape[2],
            n_labels=1,
            hidden_units=_resolve_hidden_units(self.config),
            dropout_rates=_resolve_dropout_rates(self.config.dropout_rates),
            noise_std=self.config.noise_std,
            output_activation="sigmoid" if self.config.task_type == "classification" else "linear",
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)

        train_features, train_targets = _flatten_supervision(batch)
        if train_features.shape[0] == 0:
            raise ValueError("SAE training received no valid observations")
        train_features_t = torch.as_tensor(train_features, dtype=torch.float32, device=device)
        train_targets_t = torch.as_tensor(train_targets, dtype=torch.float32, device=device)

        val_features_t = None
        val_targets_t = None
        if validation_batch is not None:
            if validation_batch.returns is None:
                raise ValueError("validation_batch must include returns")
            val_features, val_targets = _flatten_supervision(validation_batch)
            if val_features.shape[0] > 0:
                val_features_t = torch.as_tensor(val_features, dtype=torch.float32, device=device)
                val_targets_t = torch.as_tensor(val_targets, dtype=torch.float32, device=device)

        self._checkpoint_states = {}
        history: list[dict[str, float | str]] = []
        best_loss = float("inf")

        for epoch in range(1, self.config.n_epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for features_batch_t, targets_batch_t in _iterate_batches(
                train_features_t,
                train_targets_t,
                batch_size=self.config.batch_size,
                torch=torch,
                seed=self.config.seed + epoch,
            ):
                decoded, aux_pred, main_pred = model(features_batch_t)
                loss = _sae_loss(
                    torch=torch,
                    features=features_batch_t,
                    targets=targets_batch_t,
                    decoded=decoded,
                    aux_pred=aux_pred,
                    main_pred=main_pred,
                    task_type=self.config.task_type,
                    alpha=self.config.alpha,
                    aux_weight=self.config.aux_weight,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1

            train_loss = epoch_loss / max(n_batches, 1)
            val_loss = train_loss
            if val_features_t is not None and val_targets_t is not None:
                model.eval()
                with torch.no_grad():
                    decoded, aux_pred, main_pred = model(val_features_t)
                    val_loss = float(
                        _sae_loss(
                            torch=torch,
                            features=val_features_t,
                            targets=val_targets_t,
                            decoded=decoded,
                            aux_pred=aux_pred,
                            main_pred=main_pred,
                            task_type=self.config.task_type,
                            alpha=self.config.alpha,
                            aux_weight=self.config.aux_weight,
                        ).item()
                    )

            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )
            if val_loss <= best_loss:
                best_loss = val_loss
            if epoch in checkpoint_epochs:
                self._checkpoint_states[epoch] = _cpu_state_dict(model)

        checkpoint_val_losses = {
            epoch: float(history[epoch - 1]["val_loss"]) for epoch in checkpoint_epochs
        }
        configured_default = self.config.default_checkpoint
        best_checkpoint = min(checkpoint_val_losses, key=checkpoint_val_losses.__getitem__)
        self._history = tuple(history)
        self._n_features = batch.characteristics.shape[2]
        self._asset_ids = batch.asset_ids
        self._mark_fitted()

        return FitSummary(
            converged=True,
            train_metrics={
                "n_train_observations": float(train_features.shape[0]),
                "n_checkpoints": float(len(checkpoint_epochs)),
            },
            val_metrics={} if validation_batch is None else {"best_val_loss": best_loss},
            best_epoch=best_checkpoint
            if configured_default is None
            else select_checkpoint_epoch(
                checkpoint=None,
                configured_default=configured_default,
                available=checkpoint_epochs,
            ),
            history=self._history,
            notes=("Direct supervised autoencoder signal model.",),
        )

    def predict(
        self,
        batch: CrossSectionBatch,
        *,
        checkpoint: int | None = None,
    ) -> AssetSignalResult:
        if not self.is_fitted or self._n_features is None or not self._checkpoint_states:
            raise RuntimeError("SAE model must be fitted before predict()")
        if batch.characteristics.shape[2] != self._n_features:
            raise ValueError(
                "characteristics feature dimension does not match fitted SAE model; "
                f"expected {self._n_features}, got {batch.characteristics.shape[2]}"
            )

        torch = import_torch()
        nn = _import_sae_nn()
        device = resolve_device(torch, self.config.device)
        selected_checkpoint = select_checkpoint_epoch(
            checkpoint=checkpoint,
            configured_default=self.config.default_checkpoint,
            available=self.available_checkpoints,
        )
        model = nn.SupervisedAutoencoder(
            n_features=self._n_features,
            n_labels=1,
            hidden_units=_resolve_hidden_units(self.config),
            dropout_rates=_resolve_dropout_rates(self.config.dropout_rates),
            noise_std=self.config.noise_std,
            output_activation="sigmoid" if self.config.task_type == "classification" else "linear",
        ).to(device)
        model.load_state_dict(deepcopy(self._checkpoint_states[selected_checkpoint]))
        model.eval()

        mask = _resolve_mask(batch)
        predictions = np.full((batch.n_periods, batch.n_assets), np.nan, dtype=np.float64)
        with torch.no_grad():
            for date_idx in range(batch.n_periods):
                valid = mask[date_idx]
                if not valid.any():
                    continue
                features_t = torch.as_tensor(
                    np.asarray(batch.characteristics[date_idx, valid], dtype=np.float32),
                    dtype=torch.float32,
                    device=device,
                )
                predictions_t = model.predict(features_t).squeeze(-1).detach().cpu().numpy()
                predictions[date_idx, valid] = predictions_t.astype(np.float64)

        return AssetSignalResult(
            signal_values=predictions,
            timestamps=batch.timestamps,
            asset_ids=batch.asset_ids or self._asset_ids,
            metadata={
                "model_name": self.config.model_name,
                "task_type": self.config.task_type,
                "checkpoint_epoch": selected_checkpoint,
                "available_checkpoints": self.available_checkpoints,
            },
        )


def _flatten_supervision(batch: CrossSectionBatch) -> tuple[np.ndarray, np.ndarray]:
    if batch.returns is None:
        raise ValueError("batch must include returns")
    mask = _resolve_mask(batch)
    valid = mask & np.isfinite(batch.returns)
    if not valid.any():
        return (
            np.zeros((0, batch.characteristics.shape[2]), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    return (
        np.asarray(batch.characteristics[valid], dtype=np.float64),
        np.asarray(batch.returns[valid], dtype=np.float64),
    )


def _iterate_batches(
    features: Any,
    targets: Any,
    *,
    batch_size: int | None,
    torch: Any,
    seed: int,
):
    n_obs = int(features.shape[0])
    if batch_size is None or batch_size <= 0 or batch_size >= n_obs:
        yield features, targets
        return
    generator = torch.Generator(device=features.device)
    generator.manual_seed(seed)
    order = torch.randperm(n_obs, generator=generator, device=features.device)
    for start in range(0, n_obs, batch_size):
        idx = order[start : start + batch_size]
        yield features[idx], targets[idx]


def _sae_loss(
    *,
    torch: Any,
    features: Any,
    targets: Any,
    decoded: Any,
    aux_pred: Any,
    main_pred: Any,
    task_type: str,
    alpha: float,
    aux_weight: float,
) -> Any:
    reconstruction_loss = torch.nn.functional.mse_loss(decoded, features)
    targets = targets.squeeze(-1) if targets.ndim > 1 else targets
    if task_type == "classification":
        targets = targets.clamp(0.0, 1.0)
        main_loss = torch.nn.functional.binary_cross_entropy(main_pred.squeeze(-1), targets)
        aux_loss = torch.nn.functional.binary_cross_entropy(aux_pred.squeeze(-1), targets)
    else:
        main_loss = torch.nn.functional.mse_loss(main_pred.squeeze(-1), targets)
        aux_loss = torch.nn.functional.mse_loss(aux_pred.squeeze(-1), targets)
    return reconstruction_loss + alpha * main_loss + aux_weight * aux_loss


def _resolve_hidden_units(config: SAEConfig) -> tuple[int, ...]:
    return (
        int(config.bottleneck_dim),
        int(config.aux_hidden_dim),
        *(int(unit) for unit in config.main_hidden_units),
    )


def _resolve_dropout_rates(dropout_rates: tuple[float, ...] | None) -> tuple[float, ...]:
    if dropout_rates is None:
        return (0.035, 0.038, 0.424, 0.104, 0.492, 0.320, 0.272, 0.438)
    if len(dropout_rates) != 8:
        raise ValueError(f"dropout_rates must have 8 entries; got {len(dropout_rates)}")
    return tuple(float(rate) for rate in dropout_rates)


def _resolve_mask(batch: CrossSectionBatch) -> np.ndarray:
    if batch.mask is None:
        return np.asarray(np.isfinite(batch.characteristics).all(axis=2), dtype=bool)
    return np.asarray(batch.mask, dtype=bool) & np.isfinite(batch.characteristics).all(axis=2)


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _import_sae_nn() -> Any:
    from ml4t.models._internal import sae_nn

    return sae_nn
