"""Shared runtime helpers for portfolio-learning models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from ml4t.models._internal.latent_factor_utils import (
    resolve_checkpoint_epochs,
)
from ml4t.models.configs import PortfolioConfig
from ml4t.models.portfolio.losses import robust_sharpe_loss
from ml4t.models.types import PortfolioSequenceBatch


@dataclass(frozen=True, slots=True)
class PortfolioTrainingArtifacts:
    checkpoint_states: dict[int, dict[str, torch.Tensor]]
    history: tuple[dict[str, float | str], ...]
    best_step: int
    best_validation_sharpe: float


def validate_portfolio_training_batch(
    batch: PortfolioSequenceBatch,
    config: PortfolioConfig | None = None,
) -> None:
    if batch.returns is None:
        raise ValueError("portfolio training requires forward returns in the batch")
    if config is not None:
        _validate_context(batch, config)


def validate_portfolio_prediction_batch(
    batch: PortfolioSequenceBatch,
    config: PortfolioConfig,
) -> None:
    _validate_context(batch, config)


def validate_portfolio_identity(
    batch: PortfolioSequenceBatch,
    fitted_asset_ids: tuple[str, ...],
) -> None:
    if batch.asset_ids != fitted_asset_ids:
        raise ValueError(
            "prediction asset_ids must exactly match fitted asset_ids; "
            f"expected {fitted_asset_ids}, got {batch.asset_ids}"
        )


def _validate_context(batch: PortfolioSequenceBatch, config: PortfolioConfig) -> None:
    if config.use_group_embedding and batch.group_ids is None:
        raise ValueError("group_ids are required when use_group_embedding is True")
    if config.use_cost_in_context and batch.costs is None:
        raise ValueError("costs are required when use_cost_in_context is True")


def fit_policy_network(
    policy: nn.Module,
    *,
    batch: PortfolioSequenceBatch,
    validation_batch: PortfolioSequenceBatch,
    config: PortfolioConfig,
    device: torch.device,
) -> PortfolioTrainingArtifacts:
    group_ids_train = group_ids_tensor(batch, device)
    costs_train = costs_tensor(batch, device)
    group_ids_val = group_ids_tensor(validation_batch, device)
    costs_val = costs_tensor(validation_batch, device)

    optimizer = AdamW(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint_steps = tuple(
        resolve_checkpoint_epochs(
            config.max_iters,
            checkpoint_interval=config.checkpoint_every,
            checkpoint_epochs=list(config.checkpoint_steps) or None,
        )
    )
    asset_indices = torch.arange(batch.n_assets, dtype=torch.long, device=device)

    checkpoint_states: dict[int, dict[str, torch.Tensor]] = {}
    history: list[dict[str, float | str]] = []
    best_step = checkpoint_steps[-1]
    best_val_sharpe = float("-inf")
    ema_value: float | None = None
    bad_count = 0
    best_state: dict[str, torch.Tensor] | None = None
    features = torch.as_tensor(batch.features, dtype=torch.float32, device=device)
    forward_returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=device)
    vol_scale = torch.as_tensor(_vol_scale(batch), dtype=torch.float32, device=device)
    mask = mask_tensor(batch, device)
    prev_weights = previous_weights_tensor(batch, device)

    for step in range(1, config.max_iters + 1):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        weights = _policy_weights(
            policy,
            features,
            mask=mask,
            asset_indices=asset_indices,
            group_ids=group_ids_train,
            costs=costs_train,
            chunk_size=config.batch_size,
        )
        loss_output = robust_sharpe_loss(
            weights=weights,
            forward_returns=forward_returns,
            vol_scale=vol_scale,
            mask=mask,
            costs=costs_train,
            burn_in=config.burn_in,
            gamma_cost=config.gamma_cost,
            annualization_factor=config.annualization_factor,
            eps=config.sharpe_eps,
            tau=config.softmin_tau,
            lambda_soft=config.softmin_lambda,
            prev_weights=prev_weights,
            turnover_penalty=config.turnover_penalty,
        )
        if not bool(torch.isfinite(loss_output.loss).item()):
            raise FloatingPointError(f"non-finite training loss at step {step}")
        loss_output.loss.backward()
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
            for parameter in policy.parameters()
        ):
            raise FloatingPointError(f"non-finite training gradient at step {step}")
        if config.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), max_norm=config.max_grad_norm
            )
            if not bool(torch.isfinite(grad_norm).item()):
                raise FloatingPointError(f"non-finite training gradient norm at step {step}")
        optimizer.step()
        if any(
            not bool(torch.isfinite(parameter).all().item()) for parameter in policy.parameters()
        ):
            raise FloatingPointError(f"non-finite model parameter at step {step}")

        if step % config.eval_every != 0 and step not in checkpoint_steps:
            continue

        val_sharpe = evaluate_pooled_sharpe(
            policy,
            validation_batch,
            group_ids=group_ids_val,
            costs=costs_val,
            config=config,
            device=device,
        )
        if not np.isfinite(val_sharpe):
            raise FloatingPointError(f"non-finite validation Sharpe at step {step}")
        ema_value = (
            val_sharpe
            if ema_value is None
            else (
                config.metric_ema_alpha * val_sharpe + (1.0 - config.metric_ema_alpha) * ema_value
            )
        )
        if step >= config.early_stopping_burn_in_iters:
            if ema_value >= best_val_sharpe + config.metric_min_delta:
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
            checkpoint_states[step] = cpu_state_dict(policy)
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_step = step
            best_state = cpu_state_dict(policy)
        if (
            step >= config.early_stopping_burn_in_iters
            and bad_count >= config.early_stopping_patience
        ):
            break

    if best_state is None:
        raise RuntimeError("portfolio training completed without an evaluated checkpoint")
    checkpoint_states[best_step] = best_state

    return PortfolioTrainingArtifacts(
        checkpoint_states=checkpoint_states,
        history=tuple(history),
        best_step=best_step,
        best_validation_sharpe=best_val_sharpe,
    )


@torch.no_grad()
def evaluate_pooled_sharpe(
    policy: nn.Module,
    batch: PortfolioSequenceBatch,
    *,
    group_ids: torch.Tensor | None,
    costs: torch.Tensor | None,
    config: PortfolioConfig,
    device: torch.device,
) -> float:
    policy.eval()
    features = torch.as_tensor(batch.features, dtype=torch.float32, device=device)
    forward_returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=device)
    vol_scale = torch.as_tensor(_vol_scale(batch), dtype=torch.float32, device=device)
    mask = mask_tensor(batch, device)
    asset_indices = torch.arange(batch.n_assets, dtype=torch.long, device=device)

    weights = policy(
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
        prev_weights=previous_weights_tensor(batch, device),
        turnover_penalty=config.turnover_penalty,
    )
    return float(loss_output.sharpe_pool.item())


def mask_tensor(batch: PortfolioSequenceBatch, device: torch.device) -> torch.Tensor:
    mask = (
        np.asarray(batch.mask, dtype=np.float32)
        if batch.mask is not None
        else np.ones(batch.features.shape[:3], dtype=np.float32)
    )
    return torch.as_tensor(mask, dtype=torch.float32, device=device)


def group_ids_tensor(batch: PortfolioSequenceBatch, device: torch.device) -> torch.Tensor | None:
    if batch.group_ids is None:
        return None
    return torch.as_tensor(
        np.asarray(batch.group_ids, dtype=np.int64), dtype=torch.long, device=device
    )


def costs_tensor(batch: PortfolioSequenceBatch, device: torch.device) -> torch.Tensor | None:
    if batch.costs is None:
        return None
    costs = np.asarray(batch.costs, dtype=np.float32)
    if costs.ndim == 1:
        costs = costs[:, None]
    return torch.as_tensor(costs, dtype=torch.float32, device=device)


def previous_weights_tensor(
    batch: PortfolioSequenceBatch, device: torch.device
) -> torch.Tensor | None:
    if batch.prev_weights is None:
        return None
    return torch.as_tensor(batch.prev_weights, dtype=torch.float32, device=device)


def _vol_scale(batch: PortfolioSequenceBatch) -> np.ndarray:
    if batch.vol_scale is None:
        return np.ones(batch.features.shape[:3], dtype=np.float64)
    return np.asarray(batch.vol_scale, dtype=np.float64)


def adjacency_mask_tensor(
    batch: PortfolioSequenceBatch,
    device: torch.device,
) -> torch.Tensor | None:
    if batch.adjacency_mask is None:
        return None
    return torch.as_tensor(
        np.asarray(batch.adjacency_mask, dtype=bool), dtype=torch.bool, device=device
    )


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _policy_weights(
    policy: nn.Module,
    features: torch.Tensor,
    *,
    mask: torch.Tensor,
    asset_indices: torch.Tensor,
    group_ids: torch.Tensor | None,
    costs: torch.Tensor | None,
    chunk_size: int,
) -> torch.Tensor:
    chunks = []
    for start in range(0, features.shape[0], chunk_size):
        end = start + chunk_size
        chunks.append(
            policy(
                features[start:end],
                mask=mask[start:end],
                asset_indices=asset_indices,
                group_ids=group_ids,
                costs=costs,
            )
        )
    return torch.cat(chunks, dim=0)
