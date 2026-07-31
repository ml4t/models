"""DeePM-style end-to-end portfolio model."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn

from ml4t.models._internal.latent_factor_utils import select_checkpoint_epoch
from ml4t.models._internal.lifecycle import atomic_fit
from ml4t.models._internal.persistence import (
    load_artifact,
    load_config,
    pack_tensor_tree,
    save_artifact,
    unpack_tensor_tree,
)
from ml4t.models._internal.torch_runtime import resolve_device, seed_torch
from ml4t.models.configs import DeepPortfolioConfig
from ml4t.models.portfolio.base import BasePortfolioModel
from ml4t.models.portfolio.components import (
    CrossSectionalAttention,
    FiLM,
    MacroGraphAttention,
    StaticContextEncoder,
    TemporalAttentionBlock,
    VariableSelection,
)
from ml4t.models.portfolio.runtime import (
    adjacency_mask_tensor,
    costs_tensor,
    fit_policy_network,
    group_ids_tensor,
    mask_tensor,
    validate_portfolio_identity,
    validate_portfolio_prediction_batch,
    validate_portfolio_training_batch,
)
from ml4t.models.types import FitSummary, PortfolioSequenceBatch, PortfolioWeightsResult


class DeepPortfolioPolicy(nn.Module):
    """DeePM-style policy network producing bounded risk weights."""

    def __init__(
        self,
        *,
        n_assets: int,
        n_features: int,
        n_groups: int | None,
        adjacency_mask: torch.Tensor | None,
        config: DeepPortfolioConfig,
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
            d_model=config.d_model,
            context_dim=context_dim,
            hidden_dim=config.vvsn_hidden_dim,
            dropout=config.dropout,
        )
        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
        )
        self.h0_projection = nn.Linear(context_dim, config.lstm_layers * config.d_model)
        self.c0_projection = nn.Linear(context_dim, config.lstm_layers * config.d_model)

        self.temporal_blocks = nn.ModuleList(
            [
                TemporalAttentionBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    dropout=config.dropout,
                    adapter_mult=config.adapter_hidden_mult,
                )
                for _ in range(config.temporal_mha_layers)
            ]
        )
        self.cross_attention = CrossSectionalAttention(
            d_model=config.d_model,
            n_heads=config.cross_attention_heads,
            dropout=config.dropout,
            lag=config.cross_attention_lag,
        )
        self.graph_attention: MacroGraphAttention | None = None
        if adjacency_mask is not None:
            self.graph_attention = MacroGraphAttention(
                d_model=config.d_model,
                n_heads=config.macro_gnn_heads,
                dropout=config.dropout,
                adjacency_mask=adjacency_mask,
            )

        self.output_head = nn.Linear(config.d_model, 1)

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
            self.config.d_model,
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
            .reshape(batch_size * n_assets, self.config.lstm_layers, self.config.d_model)
            .permute(1, 0, 2)
            .contiguous()
        )
        c0 = (
            self.c0_projection(asset_context)
            .reshape(batch_size * n_assets, self.config.lstm_layers, self.config.d_model)
            .permute(1, 0, 2)
            .contiguous()
        )
        hidden, _ = self.lstm(hidden, (h0, c0))
        for block in self.temporal_blocks:
            hidden = block(hidden)

        hidden = hidden.reshape(batch_size, n_assets, n_periods, self.config.d_model)
        hidden = hidden.permute(0, 2, 1, 3).contiguous()
        hidden = self.cross_attention(hidden, mask)
        if self.graph_attention is not None:
            hidden = self.graph_attention(hidden, mask)
        weights = torch.tanh(self.output_head(hidden).squeeze(-1))
        return weights * mask


class DeepPortfolioModel(BasePortfolioModel):
    """End-to-end portfolio learner following the DeePM architecture."""

    def __init__(self, config: DeepPortfolioConfig) -> None:
        super().__init__(config)
        self.config: DeepPortfolioConfig = config
        self._asset_ids: tuple[str, ...] = ()
        self._n_assets: int | None = None
        self._n_features: int | None = None
        self._n_groups: int | None = None
        self._adjacency_mask: np.ndarray | None = None
        self._checkpoint_states: dict[int, dict[str, torch.Tensor]] = {}
        self._history: tuple[dict[str, float | str], ...] = ()

    @property
    def available_checkpoints(self) -> tuple[int, ...]:
        return tuple(sorted(self._checkpoint_states))

    @atomic_fit(
        "_asset_ids",
        "_n_assets",
        "_n_features",
        "_n_groups",
        "_adjacency_mask",
        "_checkpoint_states",
        "_history",
        "_is_fitted",
    )
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
        if not _same_optional_array(batch.adjacency_mask, validation_batch.adjacency_mask):
            raise ValueError("train and validation batches must share adjacency_mask")

        device = resolve_device(torch, self.config.device)
        seed_torch(torch, self.config.seed, device)

        model = DeepPortfolioPolicy(
            n_assets=batch.n_assets,
            n_features=batch.features.shape[3],
            n_groups=self._resolve_n_groups(batch),
            adjacency_mask=adjacency_mask_tensor(batch, device),
            config=self.config,
        ).to(device)
        artifacts = fit_policy_network(
            model,
            batch=batch,
            validation_batch=validation_batch,
            config=self.config,
            device=device,
        )

        self._asset_ids = batch.asset_ids
        self._n_assets = batch.n_assets
        self._n_features = batch.features.shape[3]
        self._n_groups = self._resolve_n_groups(batch)
        self._adjacency_mask = (
            None
            if batch.adjacency_mask is None
            else np.asarray(batch.adjacency_mask, dtype=bool).copy()
        )
        self._checkpoint_states = artifacts.checkpoint_states
        self._history = artifacts.history
        self._mark_fitted()

        return FitSummary(
            converged=True,
            train_metrics={"n_train_windows": float(batch.batch_size)},
            val_metrics={"best_validation_sharpe": float(artifacts.best_validation_sharpe)},
            best_epoch=artifacts.best_step,
            history=self._history,
            notes=("End-to-end portfolio weights trained with a robust Sharpe objective.",),
        )

    def predict(
        self,
        batch: PortfolioSequenceBatch,
        *,
        checkpoint: int | None = None,
    ) -> PortfolioWeightsResult:
        validate_portfolio_prediction_batch(batch, self.config)
        if not self.is_fitted or self._n_assets is None or self._n_features is None:
            raise RuntimeError("DeepPortfolioModel must be fitted before predict()")
        if batch.n_assets != self._n_assets or batch.features.shape[3] != self._n_features:
            raise ValueError("prediction batch shape does not match the fitted model")
        validate_portfolio_identity(batch, self._asset_ids)

        device = resolve_device(torch, self.config.device)
        selected_checkpoint = select_checkpoint_epoch(
            checkpoint=checkpoint,
            configured_default=self.config.default_checkpoint,
            available=self.available_checkpoints,
        )
        model = DeepPortfolioPolicy(
            n_assets=self._n_assets,
            n_features=self._n_features,
            n_groups=self._n_groups,
            adjacency_mask=self._prediction_adjacency_mask(batch, device),
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

    def _prediction_adjacency_mask(
        self,
        batch: PortfolioSequenceBatch,
        device: torch.device,
    ) -> torch.Tensor | None:
        if batch.adjacency_mask is not None and not _same_optional_array(
            batch.adjacency_mask,
            self._adjacency_mask,
        ):
            raise ValueError("prediction adjacency_mask must match the fitted adjacency_mask")
        if self._adjacency_mask is None:
            return None
        return torch.as_tensor(self._adjacency_mask, dtype=torch.bool, device=device)

    def save(self, path: str | Path) -> Path:
        if (
            not self.is_fitted
            or self._n_assets is None
            or self._n_features is None
            or not self._checkpoint_states
        ):
            raise RuntimeError("DeepPortfolioModel must be fitted before save()")
        checkpoint_tree, arrays = pack_tensor_tree(self._checkpoint_states)
        if self._adjacency_mask is not None:
            arrays["adjacency_mask"] = self._adjacency_mask
        return save_artifact(
            path,
            model_type="ml4t.models.DeepPortfolioModel",
            config=self.config,
            state={
                "asset_ids": self._asset_ids,
                "n_assets": self._n_assets,
                "n_features": self._n_features,
                "n_groups": self._n_groups,
                "history": self._history,
                "has_adjacency_mask": self._adjacency_mask is not None,
                "checkpoint_tree": checkpoint_tree,
            },
            arrays=arrays,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
    ) -> DeepPortfolioModel:
        artifact = load_artifact(path, expected_model_type="ml4t.models.DeepPortfolioModel")
        model = cls(load_config(artifact, DeepPortfolioConfig, device=device))
        checkpoint_tree = artifact.state.get("checkpoint_tree")
        if not isinstance(checkpoint_tree, dict):
            raise ValueError("artifact DeepPortfolio checkpoint tree is invalid")
        tensor_arrays = {
            name: array for name, array in artifact.arrays.items() if name.startswith("tensor_")
        }
        model._checkpoint_states = cast(
            dict[int, dict[str, torch.Tensor]],
            unpack_tensor_tree(checkpoint_tree, tensor_arrays, torch=torch),
        )
        has_adjacency_mask = artifact.state.get("has_adjacency_mask")
        if not isinstance(has_adjacency_mask, bool):
            raise ValueError("artifact DeepPortfolio adjacency state is invalid")
        if has_adjacency_mask:
            try:
                adjacency_mask = artifact.arrays["adjacency_mask"]
            except KeyError as exc:
                raise ValueError("artifact DeepPortfolio is missing adjacency_mask") from exc
            if adjacency_mask.ndim != 2:
                raise ValueError("artifact DeepPortfolio adjacency_mask must be 2D")
            model._adjacency_mask = np.asarray(adjacency_mask, dtype=bool).copy()
        elif "adjacency_mask" in artifact.arrays:
            raise ValueError("artifact DeepPortfolio has an unexpected adjacency_mask")
        expected_arrays = set(tensor_arrays)
        if has_adjacency_mask:
            expected_arrays.add("adjacency_mask")
        if set(artifact.arrays) != expected_arrays:
            raise ValueError("artifact DeepPortfolio contains unexpected arrays")
        model._asset_ids = tuple(artifact.state.get("asset_ids", ()))
        model._n_assets = int(artifact.state["n_assets"])
        model._n_features = int(artifact.state["n_features"])
        raw_n_groups = artifact.state.get("n_groups")
        model._n_groups = None if raw_n_groups is None else int(raw_n_groups)
        model._history = tuple(artifact.state.get("history", ()))
        if not model._checkpoint_states:
            raise ValueError("artifact DeepPortfolio has no checkpoints")
        if model._asset_ids and len(model._asset_ids) != model._n_assets:
            raise ValueError("artifact DeepPortfolio asset identity disagrees with dimensions")
        if model._adjacency_mask is not None and model._adjacency_mask.shape != (
            model._n_assets,
            model._n_assets,
        ):
            raise ValueError("artifact DeepPortfolio adjacency dimensions disagree")
        model._mark_fitted()
        return model


def _same_optional_array(left: object | None, right: object | None) -> bool:
    if left is None or right is None:
        return left is right
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))
