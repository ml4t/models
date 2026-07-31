"""Pipeline-level config types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MapperConfig:
    """Config for asset-return or weight mappers."""

    model_name: str = "beta_lambda"
