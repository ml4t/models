"""Sequence-based portfolio baseline."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from ml4t.models._internal.latent_factor_utils import select_checkpoint_epoch
from ml4t.models._internal.torch_runtime import resolve_device, seed_torch
from ml4t.models.configs import LSTMPortfolioConfig
from ml4t.models.portfolio.base import BasePortfolioModel
from ml4t.models.portfolio.components import FiLM, StaticContextEncoder, VariableSelection
from ml4t.models.portfolio.runtime import (
    costs_tensor,
    fit_policy_network,
    group_ids_tensor,
    mask_tensor,
    validate_portfolio_identity,
    validate_portfolio_prediction_batch,
    validate_portfolio_training_batch,
)
from ml4t.models.types import FitSummary, PortfolioSequenceBatch, PortfolioWeightsResult


class LSTMPortfolioPolicy(nn.Module):
    """Context-aware LSTM portfolio policy."""

    def __init__(
        self,
        *,
        n_assets: int,
        n_features: int,
        n_groups: int | None,
        config: LSTMPortfolioConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.context_encoder = StaticContextEncoder(
            n_assets=n_assets,
            n_groups=n_groups,
            config=config,
        )
        context_dim = self.context_encoder.context_dim

        self.film = FiLM(context_dim=context_dim, n_features=n_features)
        self.variable_selection = VariableSelection(
            n_features=n_features,
            d_model=config.hidden_size,
            context_dim=context_dim,
            hidden_dim=config.vvsn_hidden_dim,
            dropout=config.dropout,
        )
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
        )
        self.h0_projection = nn.Linear(context_dim, config.n_layers * config.hidden_size)
        self.c0_projection = nn.Linear(context_dim, config.n_layers * config.hidden_size)
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.output_head = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        features: torch.Tensor,
        *,
        mask: torch.Tensor,
        asset_indices: torch.Tensor,
        group_ids: torch.Tensor | None,
        costs: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, n_periods, n_assets, _ = features.shape
        context = self.context_encoder(
            asset_indices=asset_indices,
            group_ids=group_ids,
            costs=costs,
        )
        modulated = self.film(features, context)
        hidden = self.variable_selection(modulated, context)
        hidden = hidden.permute(0, 2, 1, 3).reshape(
            batch_size * n_assets,
            n_periods,
            self.config.hidden_size,
        )

        asset_context = (
            context.unsqueeze(0)
            .expand(batch_size, n_assets, -1)
            .reshape(
                batch_size * n_assets,
                -1,
            )
        )
        h0 = (
            self.h0_projection(asset_context)
            .reshape(batch_size * n_assets, self.config.n_layers, self.config.hidden_size)
            .permute(1, 0, 2)
            .contiguous()
        )
        c0 = (
            self.c0_projection(asset_context)
            .reshape(batch_size * n_assets, self.config.n_layers, self.config.hidden_size)
            .permute(1, 0, 2)
            .contiguous()
        )
        hidden, _ = self.lstm(hidden, (h0, c0))
        hidden = self.output_norm(hidden)
        weights = torch.tanh(self.output_head(hidden).squeeze(-1))
        weights = weights.reshape(batch_size, n_assets, n_periods).permute(0, 2, 1).contiguous()
        return weights * mask


class LSTMPortfolioModel(BasePortfolioModel):
    """Sequence-based end-to-end portfolio baseline."""

    def __init__(self, config: LSTMPortfolioConfig) -> None:
        super().__init__(config)
        self.config: LSTMPortfolioConfig = config
        self._model: LSTMPortfolioPolicy | None = None
        self._asset_ids: tuple[str, ...] = ()
        self._n_assets: int | None = None
        self._n_features: int | None = None
        self._n_groups: int | None = None
        self._checkpoint_states: dict[int, dict[str, torch.Tensor]] = {}
        self._history: tuple[dict[str, float | str], ...] = ()

    @property
    def available_checkpoints(self) -> tuple[int, ...]:
        return tuple(sorted(self._checkpoint_states))

    def fit(
        self,
        batch: PortfolioSequenceBatch,
        *,
        validation_batch: PortfolioSequenceBatch | None = None,
    ) -> FitSummary:
        validate_portfolio_training_batch(batch, self.config)
        validation_batch = validation_batch or batch
        validate_portfolio_training_batch(validation_batch, self.config)
        if batch.n_assets != validation_batch.n_assets:
            raise ValueError("train and validation batches must share the asset dimension")
        if batch.features.shape[3] != validation_batch.features.shape[3]:
            raise ValueError("train and validation batches must share the feature dimension")
        if batch.asset_ids != validation_batch.asset_ids:
            raise ValueError("train and validation batches must share asset_ids")

        device = resolve_device(torch, self.config.device)
        seed_torch(torch, self.config.seed, device)
        np.random.seed(self.config.seed)

        model = LSTMPortfolioPolicy(
            n_assets=batch.n_assets,
            n_features=batch.features.shape[3],
            n_groups=self._resolve_n_groups(batch),
            config=self.config,
        ).to(device)
        artifacts = fit_policy_network(
            model,
            batch=batch,
            validation_batch=validation_batch,
            config=self.config,
            device=device,
        )

        self._model = model
        self._asset_ids = batch.asset_ids
        self._n_assets = batch.n_assets
        self._n_features = batch.features.shape[3]
        self._n_groups = self._resolve_n_groups(batch)
        self._checkpoint_states = artifacts.checkpoint_states
        self._history = artifacts.history
        self._mark_fitted()

        return FitSummary(
            converged=True,
            train_metrics={"n_train_windows": float(batch.batch_size)},
            val_metrics={"best_validation_sharpe": float(artifacts.best_validation_sharpe)},
            best_epoch=artifacts.best_step,
            history=self._history,
            notes=("End-to-end portfolio weights trained with a sequence-only baseline.",),
        )

    def predict(
        self,
        batch: PortfolioSequenceBatch,
        *,
        checkpoint: int | None = None,
    ) -> PortfolioWeightsResult:
        validate_portfolio_prediction_batch(batch, self.config)
        if (
            not self.is_fitted
            or self._model is None
            or self._n_assets is None
            or self._n_features is None
        ):
            raise RuntimeError("LSTMPortfolioModel must be fitted before predict()")
        if batch.n_assets != self._n_assets or batch.features.shape[3] != self._n_features:
            raise ValueError("prediction batch shape does not match the fitted model")
        validate_portfolio_identity(batch, self._asset_ids)

        device = resolve_device(torch, self.config.device)
        selected_checkpoint = select_checkpoint_epoch(
            checkpoint=checkpoint,
            configured_default=self.config.default_checkpoint,
            available=self.available_checkpoints,
        )
        model = LSTMPortfolioPolicy(
            n_assets=self._n_assets,
            n_features=self._n_features,
            n_groups=self._n_groups,
            config=self.config,
        ).to(device)
        model.load_state_dict(deepcopy(self._checkpoint_states[selected_checkpoint]))
        model.eval()

        asset_indices = torch.arange(batch.n_assets, dtype=torch.long, device=device)
        features = torch.as_tensor(batch.features, dtype=torch.float32, device=device)
        mask = mask_tensor(batch, device)
        group_ids = group_ids_tensor(batch, device)
        costs = costs_tensor(batch, device)

        with torch.no_grad():
            weights = model(
                features,
                mask=mask,
                asset_indices=asset_indices,
                group_ids=group_ids,
                costs=costs,
            )
        return PortfolioWeightsResult(
            weights=weights.detach().cpu().numpy().astype(np.float64),
            checkpoint_step=selected_checkpoint,
            timestamps=batch.timestamps,
            asset_ids=batch.asset_ids or self._asset_ids,
            metadata={"model_name": self.config.model_name},
        )

    def _resolve_n_groups(self, batch: PortfolioSequenceBatch) -> int | None:
        if not self.config.use_group_embedding or batch.group_ids is None:
            return None
        return int(np.max(np.asarray(batch.group_ids, dtype=np.int64))) + 1
