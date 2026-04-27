"""Schema-aware adapters for tabular market and panel data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ml4t.models.types import CrossSectionBatch, PersistentPanelBatch

_MISSING = object()
_TIMESTAMP_ALIASES = ("timestamp_col", "time_col", "datetime_col", "date_col")
_ENTITY_ALIASES = ("entity_col", "symbol_col", "ticker_col", "asset_col", "group_col")


@dataclass(frozen=True, slots=True)
class ResolvedDatasetSchema:
    """Resolved timestamp/entity column contract for a tabular dataset."""

    timestamp_col: str
    entity_col: str
    metadata: dict[str, Any] = field(default_factory=dict)


def resolve_dataset_schema(
    frame: Any,
    *,
    schema: Any | None = None,
    timestamp_col: str | None = None,
    entity_col: str | None = None,
    timestamp_candidates: Sequence[str] = ("timestamp", "datetime", "date", "time"),
    entity_candidates: Sequence[str] = ("asset", "symbol", "ticker", "instrument", "security"),
) -> ResolvedDatasetSchema:
    """Resolve timestamp and entity columns from explicit names or ML4T-style metadata."""

    columns = tuple(_frame_columns(frame))
    inferred_schema = _coerce_schema(schema)

    resolved_timestamp = (
        timestamp_col
        or inferred_schema.get("timestamp_col")
        or _first_present(columns, timestamp_candidates)
    )
    if resolved_timestamp is None:
        raise ValueError(
            f"Could not resolve a timestamp column from columns {list(columns)}. "
            f"Expected one of {tuple(timestamp_candidates)} or explicit schema metadata."
        )
    if resolved_timestamp not in columns:
        raise ValueError(
            f"Resolved timestamp column {resolved_timestamp!r} not found in columns {list(columns)}"
        )

    resolved_entity = (
        entity_col or inferred_schema.get("entity_col") or _first_present(columns, entity_candidates)
    )
    if resolved_entity is None:
        raise ValueError(
            f"Could not resolve an entity column from columns {list(columns)}. "
            f"Expected one of {tuple(entity_candidates)} or explicit schema metadata."
        )
    if resolved_entity not in columns:
        raise ValueError(
            f"Resolved entity column {resolved_entity!r} not found in columns {list(columns)}"
        )

    return ResolvedDatasetSchema(
        timestamp_col=resolved_timestamp,
        entity_col=resolved_entity,
        metadata=inferred_schema,
    )


def persistent_panel_batch_from_long_frame(
    frame: Any,
    *,
    schema: Any | None = None,
    return_col: str | None = None,
    feature_cols: Sequence[str] = (),
    timestamp_col: str | None = None,
    entity_col: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PersistentPanelBatch:
    """Build a persistent panel batch from a long-format frame."""

    resolved = resolve_dataset_schema(
        frame,
        schema=schema,
        timestamp_col=timestamp_col,
        entity_col=entity_col,
    )
    records = _sorted_records(frame, timestamp_col=resolved.timestamp_col, entity_col=resolved.entity_col)
    timestamps = tuple(_ordered_unique(record[resolved.timestamp_col] for record in records))
    asset_ids = tuple(str(asset) for asset in _ordered_unique(record[resolved.entity_col] for record in records))
    time_index = {timestamp: idx for idx, timestamp in enumerate(timestamps)}
    asset_index = {asset: idx for idx, asset in enumerate(asset_ids)}

    returns = None
    if return_col is not None:
        returns = np.full((len(timestamps), len(asset_ids)), np.nan, dtype=np.float64)

    characteristics = None
    if feature_cols:
        characteristics = np.full(
            (len(timestamps), len(asset_ids), len(feature_cols)),
            np.nan,
            dtype=np.float64,
        )

    seen: set[tuple[Any, str]] = set()
    for record in records:
        key = (record[resolved.timestamp_col], str(record[resolved.entity_col]))
        if key in seen:
            raise ValueError(
                "Duplicate (timestamp, entity) row encountered in long-format panel data: "
                f"{key}"
            )
        seen.add(key)
        t_idx = time_index[record[resolved.timestamp_col]]
        a_idx = asset_index[str(record[resolved.entity_col])]
        if returns is not None and return_col is not None:
            return_value = record[return_col]
            returns[t_idx, a_idx] = float(return_value) if _is_finite(return_value) else np.nan
        if characteristics is not None:
            for f_idx, feature_col in enumerate(feature_cols):
                value = record[feature_col]
                characteristics[t_idx, a_idx, f_idx] = float(value) if _is_finite(value) else np.nan

    combined_metadata = dict(metadata or {})
    combined_metadata.update(resolved.metadata)
    return PersistentPanelBatch(
        returns=returns,
        characteristics=characteristics,
        timestamps=timestamps,
        asset_ids=asset_ids,
        metadata=combined_metadata,
    )


def cross_section_batch_from_long_frame(
    frame: Any,
    *,
    schema: Any | None = None,
    feature_cols: Sequence[str],
    return_col: str | None = None,
    context_cols: Sequence[str] = (),
    timestamp_col: str | None = None,
    entity_col: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CrossSectionBatch:
    """Build a ragged cross-sectional batch from a long-format frame."""

    resolved = resolve_dataset_schema(
        frame,
        schema=schema,
        timestamp_col=timestamp_col,
        entity_col=entity_col,
    )
    records = _sorted_records(frame, timestamp_col=resolved.timestamp_col, entity_col=resolved.entity_col)
    timestamps = tuple(_ordered_unique(record[resolved.timestamp_col] for record in records))
    grouped_assets = {
        timestamp: [record for record in records if record[resolved.timestamp_col] == timestamp]
        for timestamp in timestamps
    }
    max_assets = max((len(group) for group in grouped_assets.values()), default=0)
    characteristics = np.full(
        (len(timestamps), max_assets, len(feature_cols)),
        np.nan,
        dtype=np.float64,
    )
    returns = (
        np.full((len(timestamps), max_assets), np.nan, dtype=np.float64)
        if return_col is not None
        else None
    )
    mask = np.zeros((len(timestamps), max_assets), dtype=bool)
    asset_ids = tuple(f"slot_{idx}" for idx in range(max_assets))

    context_features = None
    if context_cols:
        context_features = np.full((len(timestamps), len(context_cols)), np.nan, dtype=np.float64)

    for t_idx, timestamp in enumerate(timestamps):
        records_t = grouped_assets[timestamp]
        for slot_idx, record in enumerate(records_t):
            mask[t_idx, slot_idx] = True
            for f_idx, feature_col in enumerate(feature_cols):
                value = record[feature_col]
                characteristics[t_idx, slot_idx, f_idx] = float(value) if _is_finite(value) else np.nan
            if returns is not None and return_col is not None:
                value = record[return_col]
                returns[t_idx, slot_idx] = float(value) if _is_finite(value) else np.nan

        if context_features is not None and records_t:
            for c_idx, context_col in enumerate(context_cols):
                values = np.asarray([record[context_col] for record in records_t], dtype=object)
                finite_values = [value for value in values if _is_finite(value)]
                if not finite_values:
                    context_features[t_idx, c_idx] = np.nan
                    continue
                first_value = float(finite_values[0])
                if any(abs(float(value) - first_value) > 1e-12 for value in finite_values[1:]):
                    raise ValueError(
                        f"context column {context_col!r} must be constant within timestamp "
                        f"{timestamp!r}"
                    )
                context_features[t_idx, c_idx] = first_value

    combined_metadata = dict(metadata or {})
    combined_metadata.update(resolved.metadata)
    return CrossSectionBatch(
        characteristics=characteristics,
        returns=returns,
        context_features=context_features,
        timestamps=timestamps,
        asset_ids=asset_ids,
        mask=mask,
        metadata=combined_metadata,
    )


def _coerce_schema(schema: Any | None) -> dict[str, Any]:
    if schema is None:
        return {}
    if isinstance(schema, Mapping):
        schema_mapping = dict(schema)
        if "metadata" in schema_mapping and schema_mapping["metadata"] is not None:
            return _coerce_schema(schema_mapping["metadata"])
        if "schema" in schema_mapping and schema_mapping["schema"] is not None:
            nested = _coerce_schema(schema_mapping["schema"])
            schema_mapping = {**schema_mapping, **nested}
        return _extract_schema_fields(schema_mapping)

    metadata = getattr(schema, "metadata", None)
    if metadata is not None:
        return _coerce_schema(metadata)

    nested_schema = getattr(schema, "schema", None)
    if nested_schema is not None:
        nested = _coerce_schema(nested_schema)
        current = _extract_schema_fields(schema)
        return {**current, **nested}

    return _extract_schema_fields(schema)


def _extract_schema_fields(source: Mapping[str, Any] | Any) -> dict[str, Any]:
    data: dict[str, Any] = {}
    timestamp = _pick_field(source, *_TIMESTAMP_ALIASES, "timestamp")
    entity = _pick_field(source, *_ENTITY_ALIASES, "symbol", "ticker", "asset")
    if timestamp is not _MISSING:
        data["timestamp_col"] = str(timestamp)
    if entity is not _MISSING:
        data["entity_col"] = str(entity)
    return data


def _pick_field(source: Mapping[str, Any] | Any, *names: str) -> Any:
    if isinstance(source, Mapping):
        for name in names:
            if name in source and source[name] is not None:
                return source[name]
        return _MISSING
    for name in names:
        if hasattr(source, name):
            value = getattr(source, name)
            if value is not None:
                return value
    return _MISSING


def _frame_columns(frame: Any) -> list[str]:
    if isinstance(frame, Mapping):
        return [str(column) for column in frame]
    columns = getattr(frame, "columns", None)
    if columns is None:
        raise TypeError("frame must be a mapping or a tabular object with a columns attribute")
    return [str(column) for column in columns]


def _frame_column(frame: Any, name: str) -> np.ndarray:
    values = frame[name]
    if hasattr(values, "to_numpy"):
        return np.asarray(values.to_numpy(), dtype=object)
    return np.asarray(values, dtype=object)


def _sorted_records(frame: Any, *, timestamp_col: str, entity_col: str) -> list[dict[str, Any]]:
    columns = _frame_columns(frame)
    arrays = {column: _frame_column(frame, column) for column in columns}
    if not arrays:
        return []
    n_rows = len(next(iter(arrays.values())))
    records = [
        {column: arrays[column][row_idx] for column in columns}
        for row_idx in range(n_rows)
    ]
    return sorted(records, key=lambda row: (row[timestamp_col], str(row[entity_col])))


def _ordered_unique(values: Sequence[Any] | Any) -> list[Any]:
    seen: list[Any] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _first_present(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(value))
    except TypeError:
        return False
