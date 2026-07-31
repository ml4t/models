"""Reusable neural components for portfolio-learning models."""

from __future__ import annotations

import torch
from torch import nn

from ml4t.models.configs.portfolio import PortfolioConfig


class StaticContextEncoder(nn.Module):
    """Encode per-asset static context."""

    def __init__(
        self,
        *,
        n_assets: int,
        n_groups: int | None,
        config: PortfolioConfig,
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
        context_expanded = (
            context.unsqueeze(0)
            .unsqueeze(0)
            .expand(
                batch_size,
                n_periods,
                n_assets,
                -1,
            )
        )
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
            pad = torch.zeros(
                (batch_size, self.lag, n_assets, d_model),
                device=x.device,
                dtype=x.dtype,
            )
            kv = torch.cat([pad, x[:, : n_periods - self.lag]], dim=1)
            pad_mask = torch.zeros(
                (batch_size, self.lag, n_assets),
                device=mask.device,
                dtype=mask.dtype,
            )
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

        attended, _ = self.attention(
            query,
            key_value,
            key_value,
            key_padding_mask=key_padding_mask,
        )
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
