"""Backtest-facing integration helpers for ML4T model outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

from ml4t.models.integration.data import resolve_dataset_schema
from ml4t.models.integration.surfaces import (
    SurfaceFrame,
    context_surface_from_weights,
    prediction_surface_from_asset_forecast,
    weight_surface_from_asset_weights,
    weight_surface_from_portfolio_weights,
)
from ml4t.models.types import AssetForecastResult, AssetWeightsResult, PortfolioWeightsResult


@dataclass(frozen=True, slots=True)
class BacktestDataFeedInputs:
    """Structured handoff payload for ``ml4t.backtest.DataFeed``."""

    feed_spec: dict[str, Any]
    prices_frame: Any | None = None
    prices_path: str | Path | None = None
    signals: SurfaceFrame | None = None
    context: SurfaceFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.prices_frame is None and self.prices_path is None:
            raise ValueError("Provide either prices_frame or prices_path")
        if self.prices_frame is not None and self.prices_path is not None:
            raise ValueError("Provide prices_frame or prices_path, not both")

    def to_datafeed_kwargs(self) -> dict[str, Any]:
        """Return kwargs compatible with ``ml4t.backtest.DataFeed``."""

        kwargs: dict[str, Any] = {"feed_spec": dict(self.feed_spec)}
        if self.prices_frame is not None:
            kwargs["prices_df"] = self.prices_frame
        if self.prices_path is not None:
            kwargs["prices_path"] = str(self.prices_path)
        if self.signals is not None:
            kwargs["signals_df"] = self.signals.to_polars()
        if self.context is not None:
            kwargs["context_df"] = self.context.to_polars()
        return kwargs


def resolve_feed_spec_mapping(
    frame: Any | None = None,
    *,
    schema: Any | None = None,
    timestamp_col: str | None = None,
    entity_col: str | None = None,
    price_col: str | None = None,
    open_col: str | None = None,
    high_col: str | None = None,
    low_col: str | None = None,
    close_col: str | None = None,
    volume_col: str | None = None,
    bid_col: str | None = None,
    ask_col: str | None = None,
    mid_col: str | None = None,
    bid_size_col: str | None = None,
    ask_size_col: str | None = None,
    calendar: str | None = None,
    timezone: str | None = None,
    data_frequency: Any | None = None,
    bar_type: str | None = None,
    timestamp_semantics: str | None = None,
    session_start_time: str | None = None,
) -> dict[str, Any]:
    """Resolve a ``FeedSpec``-compatible mapping from schema metadata and overrides."""

    spec_mapping = _coerce_feed_spec_mapping(schema)

    if frame is not None and _supports_schema_resolution(frame):
        resolved_schema = resolve_dataset_schema(
            frame,
            schema=schema,
            timestamp_col=timestamp_col,
            entity_col=entity_col,
        )
        spec_mapping["timestamp_col"] = resolved_schema.timestamp_col
        spec_mapping["entity_col"] = resolved_schema.entity_col
    else:
        if timestamp_col is not None:
            spec_mapping["timestamp_col"] = timestamp_col
        else:
            spec_mapping.setdefault("timestamp_col", spec_mapping.get("timestamp_col", "timestamp"))
        if entity_col is not None:
            spec_mapping["entity_col"] = entity_col
        else:
            spec_mapping.setdefault("entity_col", spec_mapping.get("entity_col", "asset"))

    overrides = {
        "price_col": price_col,
        "open_col": open_col,
        "high_col": high_col,
        "low_col": low_col,
        "close_col": close_col,
        "volume_col": volume_col,
        "bid_col": bid_col,
        "ask_col": ask_col,
        "mid_col": mid_col,
        "bid_size_col": bid_size_col,
        "ask_size_col": ask_size_col,
        "calendar": calendar,
        "timezone": timezone,
        "data_frequency": data_frequency,
        "bar_type": bar_type,
        "timestamp_semantics": timestamp_semantics,
        "session_start_time": session_start_time,
    }
    for key, value in overrides.items():
        if value is not None:
            spec_mapping[key] = value

    if "close_col" in spec_mapping and "price_col" not in spec_mapping:
        spec_mapping["price_col"] = spec_mapping["close_col"]
    spec_mapping.setdefault("price_col", "close")
    spec_mapping.setdefault("open_col", "open")
    spec_mapping.setdefault("high_col", "high")
    spec_mapping.setdefault("low_col", "low")
    spec_mapping.setdefault("close_col", spec_mapping["price_col"])
    spec_mapping.setdefault("volume_col", "volume")
    return spec_mapping


def backtest_datafeed_inputs(
    *,
    prices_frame: Any | None = None,
    prices_path: str | Path | None = None,
    signals: SurfaceFrame | None = None,
    context: SurfaceFrame | None = None,
    schema: Any | None = None,
    timestamp_col: str | None = None,
    entity_col: str | None = None,
    price_col: str | None = None,
    open_col: str | None = None,
    high_col: str | None = None,
    low_col: str | None = None,
    close_col: str | None = None,
    volume_col: str | None = None,
    bid_col: str | None = None,
    ask_col: str | None = None,
    mid_col: str | None = None,
    bid_size_col: str | None = None,
    ask_size_col: str | None = None,
    calendar: str | None = None,
    timezone: str | None = None,
    data_frequency: Any | None = None,
    bar_type: str | None = None,
    timestamp_semantics: str | None = None,
    session_start_time: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> BacktestDataFeedInputs:
    """Build a structured ``DataFeed`` handoff from model outputs and market-data metadata."""

    if prices_frame is None and prices_path is None:
        raise ValueError("Provide either prices_frame or prices_path")
    if prices_frame is not None and prices_path is not None:
        raise ValueError("Provide prices_frame or prices_path, not both")

    feed_spec = resolve_feed_spec_mapping(
        prices_frame,
        schema=schema,
        timestamp_col=timestamp_col,
        entity_col=entity_col,
        price_col=price_col,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        volume_col=volume_col,
        bid_col=bid_col,
        ask_col=ask_col,
        mid_col=mid_col,
        bid_size_col=bid_size_col,
        ask_size_col=ask_size_col,
        calendar=calendar,
        timezone=timezone,
        data_frequency=data_frequency,
        bar_type=bar_type,
        timestamp_semantics=timestamp_semantics,
        session_start_time=session_start_time,
    )
    combined_metadata = dict(metadata or {})
    if signals is not None:
        combined_metadata.setdefault("signal_surface_type", signals.metadata.get("surface_type"))
    return BacktestDataFeedInputs(
        feed_spec=feed_spec,
        prices_frame=prices_frame,
        prices_path=prices_path,
        signals=signals,
        context=context,
        metadata=combined_metadata,
    )


def backtest_inputs_from_asset_forecast(
    forecast: AssetForecastResult,
    *,
    prices_frame: Any | None = None,
    prices_path: str | Path | None = None,
    schema: Any | None = None,
    context: SurfaceFrame | None = None,
    timestamp_col: str | None = None,
    entity_col: str | None = None,
    price_col: str | None = None,
    open_col: str | None = None,
    high_col: str | None = None,
    low_col: str | None = None,
    close_col: str | None = None,
    volume_col: str | None = None,
    bid_col: str | None = None,
    ask_col: str | None = None,
    mid_col: str | None = None,
    bid_size_col: str | None = None,
    ask_size_col: str | None = None,
    calendar: str | None = None,
    timezone: str | None = None,
    data_frequency: Any | None = None,
    bar_type: str | None = None,
    timestamp_semantics: str | None = None,
    session_start_time: str | None = None,
    constants: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> BacktestDataFeedInputs:
    """Build ``DataFeed`` inputs directly from an asset-forecast result."""

    return backtest_datafeed_inputs(
        prices_frame=prices_frame,
        prices_path=prices_path,
        signals=prediction_surface_from_asset_forecast(forecast, constants=constants),
        context=context,
        schema=schema,
        timestamp_col=timestamp_col,
        entity_col=entity_col,
        price_col=price_col,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        volume_col=volume_col,
        bid_col=bid_col,
        ask_col=ask_col,
        mid_col=mid_col,
        bid_size_col=bid_size_col,
        ask_size_col=ask_size_col,
        calendar=calendar,
        timezone=timezone,
        data_frequency=data_frequency,
        bar_type=bar_type,
        timestamp_semantics=timestamp_semantics,
        session_start_time=session_start_time,
        metadata=metadata,
    )


def backtest_inputs_from_weights(
    weights: AssetWeightsResult | PortfolioWeightsResult,
    *,
    prices_frame: Any | None = None,
    prices_path: str | Path | None = None,
    schema: Any | None = None,
    as_context: bool = False,
    context_prefix: str = "w_",
    timestamp_col: str | None = None,
    entity_col: str | None = None,
    price_col: str | None = None,
    open_col: str | None = None,
    high_col: str | None = None,
    low_col: str | None = None,
    close_col: str | None = None,
    volume_col: str | None = None,
    bid_col: str | None = None,
    ask_col: str | None = None,
    mid_col: str | None = None,
    bid_size_col: str | None = None,
    ask_size_col: str | None = None,
    calendar: str | None = None,
    timezone: str | None = None,
    data_frequency: Any | None = None,
    bar_type: str | None = None,
    timestamp_semantics: str | None = None,
    session_start_time: str | None = None,
    constants: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> BacktestDataFeedInputs:
    """Build ``DataFeed`` inputs directly from target-weight outputs."""

    signals = None if as_context else _weight_surface(weights, constants=constants)
    context = context_surface_from_weights(weights, prefix=context_prefix, constants=constants)
    return backtest_datafeed_inputs(
        prices_frame=prices_frame,
        prices_path=prices_path,
        signals=signals,
        context=context if as_context else None,
        schema=schema,
        timestamp_col=timestamp_col,
        entity_col=entity_col,
        price_col=price_col,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        volume_col=volume_col,
        bid_col=bid_col,
        ask_col=ask_col,
        mid_col=mid_col,
        bid_size_col=bid_size_col,
        ask_size_col=ask_size_col,
        calendar=calendar,
        timezone=timezone,
        data_frequency=data_frequency,
        bar_type=bar_type,
        timestamp_semantics=timestamp_semantics,
        session_start_time=session_start_time,
        metadata=metadata,
    )


def _coerce_feed_spec_mapping(schema: Any | None) -> dict[str, Any]:
    feed_spec_cls = _load_feed_spec_class()
    if feed_spec_cls is not None:
        return {
            key: value
            for key, value in asdict(feed_spec_cls.from_any(schema)).items()
            if value is not None
        }
    return _fallback_feed_spec_mapping(schema)


def _load_feed_spec_class() -> type[Any] | None:
    try:
        module = import_module("ml4t.specs.market_data")
    except ImportError:
        return None
    return getattr(module, "FeedSpec", None)


def _supports_schema_resolution(frame: Any) -> bool:
    return hasattr(frame, "columns") or isinstance(frame, dict)


def _fallback_feed_spec_mapping(schema: Any | None) -> dict[str, Any]:
    if schema is None:
        return {}

    schema_section = _get_nested_field(schema, "schema")
    semantics_section = _get_nested_field(schema, "semantics")
    source = schema if schema_section is _MISSING and semantics_section is _MISSING else None

    data: dict[str, Any] = {}
    if source is not None:
        _update_from_source(
            data,
            source,
            fields=(
                "timestamp_col",
                "time_col",
                "datetime_col",
                "entity_col",
                "symbol_col",
                "ticker_col",
                "asset_col",
                "group_col",
                "price_col",
                "open_col",
                "high_col",
                "low_col",
                "close_col",
                "volume_col",
                "bid_col",
                "ask_col",
                "mid_col",
                "bid_size_col",
                "ask_size_col",
            ),
        )
    else:
        _update_from_source(
            data,
            schema_section,
            fields=(
                "timestamp_col",
                "time_col",
                "datetime_col",
                "entity_col",
                "symbol_col",
                "ticker_col",
                "asset_col",
                "group_col",
                "price_col",
                "open_col",
                "high_col",
                "low_col",
                "close_col",
                "volume_col",
                "bid_col",
                "ask_col",
                "mid_col",
                "bid_size_col",
                "ask_size_col",
            ),
        )
        _update_from_source(
            data,
            semantics_section,
            fields=(
                "calendar",
                "timezone",
                "data_frequency",
                "frequency",
                "bar_type",
                "timestamp_semantics",
                "session_start_time",
            ),
        )

    return _normalize_feed_spec_aliases(data)


_MISSING = object()


def _get_nested_field(source: Any, name: str) -> Any:
    if source is None:
        return _MISSING
    if isinstance(source, dict):
        return source.get(name, _MISSING)
    return getattr(source, name, _MISSING)


def _update_from_source(data: dict[str, Any], source: Any, *, fields: tuple[str, ...]) -> None:
    if source in (_MISSING, None):
        return
    for field_name in fields:
        value = _get_nested_field(source, field_name)
        if value is not _MISSING:
            data[field_name] = value


def _normalize_feed_spec_aliases(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)

    if "time_col" in normalized and "timestamp_col" not in normalized:
        normalized["timestamp_col"] = normalized.pop("time_col")
    if "datetime_col" in normalized and "timestamp_col" not in normalized:
        normalized["timestamp_col"] = normalized.pop("datetime_col")

    for alias in ("symbol_col", "ticker_col", "asset_col", "group_col"):
        if alias in normalized and "entity_col" not in normalized:
            normalized["entity_col"] = normalized[alias]
        normalized.pop(alias, None)

    if "frequency" in normalized and "data_frequency" not in normalized:
        normalized["data_frequency"] = normalized.pop("frequency")

    return normalized


def _weight_surface(
    weights: AssetWeightsResult | PortfolioWeightsResult,
    *,
    constants: dict[str, Any] | None,
) -> SurfaceFrame:
    if isinstance(weights, AssetWeightsResult):
        return weight_surface_from_asset_weights(weights, constants=constants)
    return weight_surface_from_portfolio_weights(weights, constants=constants)
