"""Optional predictive heads and mappers for stochastic discount factor models."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from ml4t.models._internal.latent_factor_utils import (
    resolve_checkpoint_epochs,
    select_checkpoint_epoch,
)
from ml4t.models._internal.torch_runtime import import_torch, resolve_device, seed_torch
from ml4t.models.configs import StochasticDiscountFactorConfig
from ml4t.models.types import (
    AssetForecastResult,
    AssetSignalResult,
    CrossSectionBatch,
    FitSummary,
    StochasticDiscountFactorState,
)


class LinearStochasticDiscountFactorReturnMapper:
    """Map stochastic discount factor weights to expected returns via a fitted linear projection."""

    def __init__(self) -> None:
        self._intercept = 0.0
        self._slope = 0.0
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, state: StochasticDiscountFactorState, batch: CrossSectionBatch) -> FitSummary:
        if batch.returns is None:
            raise ValueError(
                "LinearStochasticDiscountFactorReturnMapper requires returns in the training batch"
            )
        valid = np.isfinite(state.asset_weights) & np.isfinite(batch.returns)
        if valid.sum() < 2:
            self._intercept = 0.0
            self._slope = 0.0
        else:
            design = np.column_stack(
                [
                    np.ones(int(valid.sum()), dtype=np.float64),
                    state.asset_weights[valid].astype(np.float64),
                ]
            )
            coeffs, *_ = np.linalg.lstsq(
                design, batch.returns[valid].astype(np.float64), rcond=None
            )
            self._intercept = float(coeffs[0])
            self._slope = float(coeffs[1])
        self._is_fitted = True
        return FitSummary(
            converged=True,
            train_metrics={
                "intercept": self._intercept,
                "slope": self._slope,
            },
            notes=(
                "Linear projection from stochastic discount factor weights to expected returns.",
            ),
        )

    def predict(self, state: StochasticDiscountFactorState) -> AssetForecastResult:
        if not self._is_fitted:
            raise RuntimeError(
                "LinearStochasticDiscountFactorReturnMapper must be fitted before predict()"
            )
        expected_returns = self._intercept + self._slope * state.asset_weights
        expected_returns = np.where(np.isfinite(state.asset_weights), expected_returns, np.nan)
        return AssetForecastResult(
            expected_returns=expected_returns,
            timestamps=state.timestamps,
            asset_ids=state.asset_ids,
            metadata={"mapper": "linear_stochastic_discount_factor_return"},
        )


class StochasticDiscountFactorBetaNetworkHead:
    """Paper-faithful beta-network predictive head for stochastic discount factor models."""

    def __init__(self, config: StochasticDiscountFactorConfig) -> None:
        self.config = config
        self._checkpoint_states: dict[int, dict[str, Any]] = {}
        self._n_asset_features: int | None = None
        self._n_context_features: int = 0
        self._asset_ids: tuple[str, ...] = ()
        self._f_hat_scale: float = 1.0
        self._history: tuple[dict[str, float | str], ...] = ()

    @property
    def available_checkpoints(self) -> tuple[int, ...]:
        return tuple(sorted(self._checkpoint_states))

    @property
    def is_fitted(self) -> bool:
        return bool(self._checkpoint_states)

    def fit(
        self,
        state: StochasticDiscountFactorState,
        batch: CrossSectionBatch,
        *,
        validation_state: StochasticDiscountFactorState | None = None,
        validation_batch: CrossSectionBatch | None = None,
    ) -> FitSummary:
        if batch.returns is None:
            raise ValueError("beta-network training requires returns in the batch")
        if state.sdf_values is None:
            raise ValueError("training state must include sdf_values")

        torch = import_torch()
        nn = _import_stochastic_discount_factor_nn()
        device = resolve_device(torch, self.config.device)
        seed_torch(torch, self.config.seed, device)
        np.random.seed(self.config.seed)

        train_payload = _beta_training_payload(state, batch, scale=None)
        if train_payload is None:
            raise ValueError("beta-network training received no valid observations")
        self._f_hat_scale = train_payload["scale"]
        val_payload = None
        if validation_state is not None and validation_batch is not None:
            val_payload = _beta_training_payload(
                validation_state,
                validation_batch,
                scale=self._f_hat_scale,
            )

        checkpoint_epochs = tuple(
            resolve_checkpoint_epochs(
                self.config.beta_n_epochs,
                checkpoint_interval=self.config.beta_checkpoint_interval,
                checkpoint_epochs=list(self.config.beta_checkpoint_epochs) or None,
            )
        )
        model = nn.BetaNetwork(
            n_asset_features=batch.characteristics.shape[2],
            n_context_features=0
            if batch.context_features is None
            else batch.context_features.shape[1],
            state_dim=self.config.beta_state_dim,
            hidden_dim=self.config.beta_hidden_dim,
            dropout=self.config.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.beta_lr)

        self._checkpoint_states = {}
        history: list[dict[str, float | str]] = []
        best_loss = float("inf")

        for epoch in range(1, self.config.beta_n_epochs + 1):
            model.train()
            train_loss = _beta_loss(torch, model, train_payload)
            optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            optimizer.step()

            val_loss_value = float(train_loss.item())
            if val_payload is not None:
                model.eval()
                with torch.no_grad():
                    val_loss_value = float(_beta_loss(torch, model, val_payload).item())

            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(train_loss.item()),
                    "val_loss": val_loss_value,
                }
            )
            if val_loss_value <= best_loss:
                best_loss = val_loss_value
            if epoch in checkpoint_epochs:
                self._checkpoint_states[epoch] = _cpu_state_dict(model)

        self._n_asset_features = batch.characteristics.shape[2]
        self._n_context_features = (
            0 if batch.context_features is None else batch.context_features.shape[1]
        )
        self._asset_ids = batch.asset_ids
        self._history = tuple(history)
        checkpoint_val_losses = {
            epoch: float(history[epoch - 1]["val_loss"]) for epoch in checkpoint_epochs
        }
        best_checkpoint = min(checkpoint_val_losses, key=checkpoint_val_losses.__getitem__)

        return FitSummary(
            converged=True,
            train_metrics={"n_checkpoints": float(len(checkpoint_epochs))},
            val_metrics={} if val_payload is None else {"best_val_loss": best_loss},
            best_epoch=best_checkpoint
            if self.config.beta_default_checkpoint is None
            else select_checkpoint_epoch(
                checkpoint=None,
                configured_default=self.config.beta_default_checkpoint,
                available=checkpoint_epochs,
            ),
            history=self._history,
            notes=("Beta-network signal head trained on returns times realized stochastic discount factor portfolio return.",),
        )

    def predict(
        self,
        batch: CrossSectionBatch,
        *,
        checkpoint: int | None = None,
    ) -> AssetSignalResult:
        if not self.is_fitted or self._n_asset_features is None:
            raise RuntimeError("beta-network head must be fitted before predict()")
        if batch.characteristics.shape[2] != self._n_asset_features:
            raise ValueError(
                "characteristics feature dimension does not match fitted beta-network head; "
                f"expected {self._n_asset_features}, got {batch.characteristics.shape[2]}"
            )
        n_context_features = (
            0 if batch.context_features is None else batch.context_features.shape[1]
        )
        if n_context_features != self._n_context_features:
            raise ValueError(
                "context feature dimension does not match fitted beta-network head; "
                f"expected {self._n_context_features}, got {n_context_features}"
            )

        torch = import_torch()
        nn = _import_stochastic_discount_factor_nn()
        device = resolve_device(torch, self.config.device)
        selected_checkpoint = select_checkpoint_epoch(
            checkpoint=checkpoint,
            configured_default=self.config.beta_default_checkpoint,
            available=self.available_checkpoints,
        )
        model = nn.BetaNetwork(
            n_asset_features=self._n_asset_features,
            n_context_features=self._n_context_features,
            state_dim=self.config.beta_state_dim,
            hidden_dim=self.config.beta_hidden_dim,
            dropout=self.config.dropout,
        ).to(device)
        model.load_state_dict(deepcopy(self._checkpoint_states[selected_checkpoint]))
        model.eval()

        mask = _resolve_mask(batch)
        payload = {
            "asset_features": torch.as_tensor(
                np.asarray(batch.characteristics, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            "context_features": None
            if batch.context_features is None
            else torch.as_tensor(
                np.asarray(batch.context_features, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            "mask": torch.as_tensor(mask, dtype=torch.bool, device=device),
        }
        with torch.no_grad():
            beta_flat, _ = model(
                payload["asset_features"],
                context_features=payload["context_features"],
                mask=payload["mask"],
            )

        signal_values = np.full((batch.n_periods, batch.n_assets), np.nan, dtype=np.float64)
        signal_values[mask] = beta_flat.detach().cpu().numpy().astype(np.float64)
        return AssetSignalResult(
            signal_values=signal_values,
            timestamps=batch.timestamps,
            asset_ids=batch.asset_ids or self._asset_ids,
            metadata={
                "model_name": self.config.model_name,
                "signal_type": "stochastic_discount_factor_beta",
                "checkpoint_epoch": selected_checkpoint,
                "available_checkpoints": self.available_checkpoints,
                "f_hat_scale": self._f_hat_scale,
            },
        )


def _beta_training_payload(
    state: StochasticDiscountFactorState,
    batch: CrossSectionBatch,
    *,
    scale: float | None,
) -> dict[str, Any] | None:
    if batch.returns is None or state.sdf_values is None:
        return None
    mask = _resolve_mask(batch) & np.isfinite(batch.returns) & np.isfinite(state.sdf_values)[:, None]
    if not mask.any():
        return None
    f_hat = 1.0 - np.asarray(state.sdf_values, dtype=np.float64)
    if scale is None:
        scale = float(np.std(f_hat))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
    y = np.asarray(batch.returns, dtype=np.float64) * (f_hat[:, None] / scale)
    y = np.where(mask, y, 0.0)

    torch = import_torch()
    device = resolve_device(torch, "cpu")
    del device
    return {
        "asset_features": np.asarray(batch.characteristics, dtype=np.float32),
        "context_features": None
        if batch.context_features is None
        else np.asarray(batch.context_features, dtype=np.float32),
        "mask": mask,
        "target": y.astype(np.float32),
        "scale": scale,
    }


def _beta_loss(torch: Any, model: Any, payload: dict[str, Any]) -> Any:
    asset_features = torch.as_tensor(payload["asset_features"], dtype=torch.float32, device=next(model.parameters()).device)
    context_features = None
    if payload["context_features"] is not None:
        context_features = torch.as_tensor(
            payload["context_features"],
            dtype=torch.float32,
            device=next(model.parameters()).device,
        )
    mask = torch.as_tensor(payload["mask"], dtype=torch.bool, device=next(model.parameters()).device)
    target = torch.as_tensor(payload["target"], dtype=torch.float32, device=next(model.parameters()).device)
    pred, _ = model(asset_features, context_features=context_features, mask=mask)
    target_flat = target[mask]
    return torch.mean((pred - target_flat) ** 2)


def _resolve_mask(batch: CrossSectionBatch) -> np.ndarray:
    mask = np.isfinite(batch.characteristics).all(axis=2)
    if batch.mask is None:
        return np.asarray(mask, dtype=bool)
    return np.asarray(batch.mask, dtype=bool) & mask


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _import_stochastic_discount_factor_nn() -> Any:
    from ml4t.models._internal import stochastic_discount_factor_nn

    return stochastic_discount_factor_nn
