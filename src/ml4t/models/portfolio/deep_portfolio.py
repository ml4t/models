"""DeePM-style end-to-end portfolio model."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from ml4t.models._internal.latent_factor_utils import (
    resolve_checkpoint_epochs,
    select_checkpoint_epoch,
)
from ml4t.models._internal.torch_runtime import resolve_device, seed_torch
from ml4t.models.configs import DeepPortfolioConfig
from ml4t.models.portfolio.base import BasePortfolioModel
from ml4t.models.portfolio.losses import robust_sharpe_loss
from ml4t.models.types import FitSummary, PortfolioSequenceBatch, PortfolioWeightsResult


class StaticContextEncoder(nn.Module):
    """Encode per-asset static context."""

    def __init__(
        self,
        *,
        n_assets: int,
        n_groups: int | None,
        config: DeepPortfolioConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.asset_embedding = nn.Embedding(n_assets, config.asset_embedding_dim)

        self.group_embedding: nn.Embedding | None = None
        if config.use_group_embedding:
            if n_groups is None:
                raise ValueError("n_groups is required when use_group_embedding is True")
            self.group_embedding = nn.Embedding(n_groups, config.group_embedding_dim)

        self.include_cost = config.use_cost_in_context

    @property
    def context_dim(self) -> int:
        dim = self.config.asset_embedding_dim
        if self.group_embedding is not None:
            dim += self.config.group_embedding_dim
        if self.include_cost:
            dim += 1
        return dim

    def forward(
        self,
        *,
        asset_indices: torch.Tensor,
        group_ids: torch.Tensor | None,
        costs: torch.Tensor | None,
    ) -> torch.Tensor:
        parts = [self.asset_embedding(asset_indices)]
        if self.group_embedding is not None:
            if group_ids is None:
                raise ValueError("group_ids are required when group embeddings are enabled")
            parts.append(self.group_embedding(group_ids))
        if self.include_cost:
            if costs is None:
                raise ValueError("costs are required when use_cost_in_context is True")
            parts.append(costs)
        return torch.cat(parts, dim=-1)


class FiLM(nn.Module):
    """Feature-wise linear modulation."""

    def __init__(self, *, context_dim: int, n_features: int) -> None:
        super().__init__()
        self.projection = nn.Linear(context_dim, 2 * n_features)
        self.n_features = n_features

    def forward(self, features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.projection(context)
        gamma = gamma_beta[:, : self.n_features].unsqueeze(0).unsqueeze(0)
        beta = gamma_beta[:, self.n_features :].unsqueeze(0).unsqueeze(0)
        return features * (1.0 + gamma) + beta


class VariableSelection(nn.Module):
    """Vectorized variable selection network."""

    def __init__(
        self,
        *,
        n_features: int,
        d_model: int,
        context_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.feature_weight = nn.Parameter(torch.empty(n_features, d_model))
        self.feature_bias = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.xavier_uniform_(self.feature_weight)
        self.selector = nn.Sequential(
            nn.Linear(n_features + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_features),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_size, n_periods, n_assets, _ = features.shape
        context_expanded = context.unsqueeze(0).unsqueeze(0).expand(batch_size, n_periods, n_assets, -1)
        logits = self.selector(torch.cat([features, context_expanded], dim=-1))
        weights = torch.softmax(logits, dim=-1)
        latent = torch.einsum("btnf,fd->btnfd", features, self.feature_weight) + self.feature_bias
        selected = (weights.unsqueeze(-1) * latent).sum(dim=-2)
        return self.output_norm(selected)


class AdapterBlock(nn.Module):
    """Feed-forward adapter block with residual connection."""

    def __init__(self, *, d_model: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        d_ff = int(hidden_mult * d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.feed_forward(self.layer_norm(x)))


class TemporalAttentionBlock(nn.Module):
    """Causal temporal self-attention."""

    def __init__(self, *, d_model: int, n_heads: int, dropout: float, adapter_mult: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.adapter = AdapterBlock(d_model=d_model, hidden_mult=adapter_mult, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_periods = x.shape[1]
        causal_mask = torch.triu(
            torch.ones((n_periods, n_periods), device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attended, _ = self.attention(x, x, x, attn_mask=causal_mask)
        x = self.layer_norm(x + self.dropout(attended))
        return self.adapter(x)


class CrossSectionalAttention(nn.Module):
    """Cross-asset attention with a causal lag."""

    def __init__(self, *, d_model: int, n_heads: int, dropout: float, lag: int) -> None:
        super().__init__()
        self.lag = int(lag)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, n_periods, n_assets, d_model = x.shape
        if self.lag > 0:
            pad = torch.zeros((batch_size, self.lag, n_assets, d_model), device=x.device, dtype=x.dtype)
            kv = torch.cat([pad, x[:, : n_periods - self.lag]], dim=1)
            pad_mask = torch.zeros((batch_size, self.lag, n_assets), device=mask.device, dtype=mask.dtype)
            kv_mask = torch.cat([pad_mask, mask[:, : n_periods - self.lag]], dim=1)
        else:
            kv = x
            kv_mask = mask

        query = x.reshape(batch_size * n_periods, n_assets, d_model)
        key_value = kv.reshape(batch_size * n_periods, n_assets, d_model)
        key_padding_mask = kv_mask.reshape(batch_size * n_periods, n_assets) < 0.5
        all_masked = key_padding_mask.all(dim=-1, keepdim=True)
        if all_masked.any():
            key_padding_mask = key_padding_mask & ~all_masked

        attended, _ = self.attention(query, key_value, key_value, key_padding_mask=key_padding_mask)
        attended = attended.reshape(batch_size, n_periods, n_assets, d_model)
        return self.layer_norm(x + self.dropout(attended))


class MacroGraphAttention(nn.Module):
    """Adjacency-masked cross-sectional attention."""

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        dropout: float,
        adjacency_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("_adjacency_mask_buffer", adjacency_mask.to(dtype=torch.bool))
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, n_periods, n_assets, d_model = x.shape
        query = x.reshape(batch_size * n_periods, n_assets, d_model)
        key_padding_mask = mask.reshape(batch_size * n_periods, n_assets) < 0.5
        adjacency_mask = self.get_buffer("_adjacency_mask_buffer")
        combined = adjacency_mask.unsqueeze(0) | key_padding_mask.unsqueeze(1)
        all_blocked = combined.all(dim=-1)
        if all_blocked.any():
            key_padding_mask = key_padding_mask & ~all_blocked
        attended, _ = self.attention(
            query,
            query,
            query,
            attn_mask=adjacency_mask,
            key_padding_mask=key_padding_mask,
        )
        attended = attended.reshape(batch_size, n_periods, n_assets, d_model)
        return self.layer_norm(x + self.dropout(attended))


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

        hidden = hidden.permute(0, 2, 1, 3).reshape(batch_size * n_assets, n_periods, self.config.d_model)
        asset_context = context.unsqueeze(0).expand(batch_size, n_assets, -1).reshape(batch_size * n_assets, -1)
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
        self._model: DeepPortfolioPolicy | None = None
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
        _validate_portfolio_batch(batch)
        validation_batch = validation_batch or batch
        _validate_portfolio_batch(validation_batch)
        if batch.n_assets != validation_batch.n_assets:
            raise ValueError("train and validation batches must share the asset dimension")
        if batch.features.shape[3] != validation_batch.features.shape[3]:
            raise ValueError("train and validation batches must share the feature dimension")

        device = resolve_device(torch, self.config.device)
        seed_torch(torch, self.config.seed, device)
        np.random.seed(self.config.seed)

        adjacency_mask = _adjacency_mask_tensor(batch, device)
        group_ids_train = _group_ids_tensor(batch, device)
        costs_train = _costs_tensor(batch, device)
        group_ids_val = _group_ids_tensor(validation_batch, device)
        costs_val = _costs_tensor(validation_batch, device)

        model = DeepPortfolioPolicy(
            n_assets=batch.n_assets,
            n_features=batch.features.shape[3],
            n_groups=self._resolve_n_groups(batch),
            adjacency_mask=adjacency_mask,
            config=self.config,
        ).to(device)
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        train_loader = _build_loader(batch, self.config.batch_size, shuffle=True)
        checkpoint_steps = tuple(
            resolve_checkpoint_epochs(
                self.config.max_iters,
                checkpoint_interval=self.config.checkpoint_every,
                checkpoint_epochs=list(self.config.checkpoint_steps) or None,
            )
        )

        asset_indices = torch.arange(batch.n_assets, dtype=torch.long, device=device)
        self._checkpoint_states = {}
        history: list[dict[str, float | str]] = []
        best_step = checkpoint_steps[-1]
        best_val_sharpe = float("-inf")
        ema_value: float | None = None
        bad_count = 0

        train_iter = iter(train_loader)
        for step in range(1, self.config.max_iters + 1):
            model.train()
            try:
                features, forward_returns, vol_scale, mask = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                features, forward_returns, vol_scale, mask = next(train_iter)

            features = features.to(device=device, dtype=torch.float32)
            forward_returns = forward_returns.to(device=device, dtype=torch.float32)
            vol_scale = vol_scale.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            weights = model(
                features,
                mask=mask,
                asset_indices=asset_indices,
                group_ids=group_ids_train,
                costs=costs_train,
            )
            loss_output = robust_sharpe_loss(
                weights=weights,
                forward_returns=forward_returns,
                vol_scale=vol_scale,
                mask=mask,
                costs=costs_train,
                burn_in=self.config.burn_in,
                gamma_cost=self.config.gamma_cost,
                annualization_factor=self.config.annualization_factor,
                eps=self.config.sharpe_eps,
                tau=self.config.softmin_tau,
                lambda_soft=self.config.softmin_lambda,
            )
            if torch.isnan(loss_output.loss) or torch.isinf(loss_output.loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            loss_output.loss.backward()
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config.max_grad_norm)
            optimizer.step()

            if step % self.config.eval_every != 0 and step not in checkpoint_steps:
                continue

            val_sharpe = _evaluate_pooled_sharpe(
                model,
                validation_batch,
                group_ids=group_ids_val,
                costs=costs_val,
                config=self.config,
                device=device,
            )
            ema_value = val_sharpe if ema_value is None else (
                self.config.metric_ema_alpha * val_sharpe
                + (1.0 - self.config.metric_ema_alpha) * ema_value
            )
            if step >= self.config.early_stopping_burn_in_iters:
                if ema_value >= best_val_sharpe + self.config.metric_min_delta:
                    bad_count = 0
                else:
                    bad_count += 1

            history.append(
                {
                    "step": float(step),
                    "train_objective": float(loss_output.objective.item()),
                    "train_sharpe_pool": float(loss_output.sharpe_pool.item()),
                    "validation_sharpe_pool": float(val_sharpe),
                }
            )
            if step in checkpoint_steps:
                self._checkpoint_states[step] = _cpu_state_dict(model)
            if val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe
                best_step = step
            if (
                step >= self.config.early_stopping_burn_in_iters
                and bad_count >= self.config.early_stopping_patience
            ):
                break

        if best_step not in self._checkpoint_states:
            self._checkpoint_states[best_step] = _cpu_state_dict(model)

        self._model = model
        self._asset_ids = batch.asset_ids
        self._n_assets = batch.n_assets
        self._n_features = batch.features.shape[3]
        self._n_groups = self._resolve_n_groups(batch)
        self._history = tuple(history)
        self._mark_fitted()

        return FitSummary(
            converged=True,
            train_metrics={"n_train_windows": float(batch.batch_size)},
            val_metrics={"best_validation_sharpe": float(best_val_sharpe)},
            best_epoch=best_step,
            history=self._history,
            notes=("End-to-end portfolio weights trained with a robust Sharpe objective.",),
        )

    def predict(
        self,
        batch: PortfolioSequenceBatch,
        *,
        checkpoint: int | None = None,
    ) -> PortfolioWeightsResult:
        _validate_portfolio_batch(batch)
        if (
            not self.is_fitted
            or self._model is None
            or self._n_assets is None
            or self._n_features is None
        ):
            raise RuntimeError("DeepPortfolioModel must be fitted before predict()")
        if batch.n_assets != self._n_assets or batch.features.shape[3] != self._n_features:
            raise ValueError("prediction batch shape does not match the fitted model")

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
            adjacency_mask=_adjacency_mask_tensor(batch, device),
            config=self.config,
        ).to(device)
        model.load_state_dict(deepcopy(self._checkpoint_states[selected_checkpoint]))
        model.eval()

        asset_indices = torch.arange(batch.n_assets, dtype=torch.long, device=device)
        features = torch.as_tensor(batch.features, dtype=torch.float32, device=device)
        mask = _mask_tensor(batch, device)
        group_ids = _group_ids_tensor(batch, device)
        costs = _costs_tensor(batch, device)

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


def _validate_portfolio_batch(batch: PortfolioSequenceBatch) -> None:
    if batch.returns is None:
        raise ValueError("portfolio training requires forward returns in the batch")
    if batch.vol_scale is None:
        raise ValueError("portfolio training requires vol_scale in the batch")


def _build_loader(
    batch: PortfolioSequenceBatch,
    batch_size: int,
    *,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(batch.features, dtype=torch.float32),
        torch.as_tensor(batch.returns, dtype=torch.float32),
        torch.as_tensor(batch.vol_scale, dtype=torch.float32),
        torch.as_tensor(
            np.asarray(batch.mask, dtype=np.float32)
            if batch.mask is not None
            else np.ones(batch.features.shape[:3], dtype=np.float32),
            dtype=torch.float32,
        ),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def _evaluate_pooled_sharpe(
    model: DeepPortfolioPolicy,
    batch: PortfolioSequenceBatch,
    *,
    group_ids: torch.Tensor | None,
    costs: torch.Tensor | None,
    config: DeepPortfolioConfig,
    device: torch.device,
) -> float:
    model.eval()
    features = torch.as_tensor(batch.features, dtype=torch.float32, device=device)
    forward_returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=device)
    vol_scale = torch.as_tensor(batch.vol_scale, dtype=torch.float32, device=device)
    mask = _mask_tensor(batch, device)
    asset_indices = torch.arange(batch.n_assets, dtype=torch.long, device=device)

    weights = model(
        features,
        mask=mask,
        asset_indices=asset_indices,
        group_ids=group_ids,
        costs=costs,
    )
    loss_output = robust_sharpe_loss(
        weights=weights,
        forward_returns=forward_returns,
        vol_scale=vol_scale,
        mask=mask,
        costs=costs,
        burn_in=config.burn_in,
        gamma_cost=config.gamma_cost,
        annualization_factor=config.annualization_factor,
        eps=config.sharpe_eps,
        tau=config.softmin_tau,
        lambda_soft=config.softmin_lambda,
    )
    return float(loss_output.sharpe_pool.item())


def _mask_tensor(batch: PortfolioSequenceBatch, device: torch.device) -> torch.Tensor:
    mask = (
        np.asarray(batch.mask, dtype=np.float32)
        if batch.mask is not None
        else np.ones(batch.features.shape[:3], dtype=np.float32)
    )
    return torch.as_tensor(mask, dtype=torch.float32, device=device)


def _group_ids_tensor(batch: PortfolioSequenceBatch, device: torch.device) -> torch.Tensor | None:
    if batch.group_ids is None:
        return None
    return torch.as_tensor(np.asarray(batch.group_ids, dtype=np.int64), dtype=torch.long, device=device)


def _costs_tensor(batch: PortfolioSequenceBatch, device: torch.device) -> torch.Tensor | None:
    if batch.costs is None:
        return None
    costs = np.asarray(batch.costs, dtype=np.float32)
    if costs.ndim == 1:
        costs = costs[:, None]
    return torch.as_tensor(costs, dtype=torch.float32, device=device)


def _adjacency_mask_tensor(batch: PortfolioSequenceBatch, device: torch.device) -> torch.Tensor | None:
    if batch.adjacency_mask is None:
        return None
    return torch.as_tensor(np.asarray(batch.adjacency_mask, dtype=bool), dtype=torch.bool, device=device)


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
