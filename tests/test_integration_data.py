from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from ml4t.models import (
    cross_section_batch_from_long_frame,
    persistent_panel_batch_from_long_frame,
    resolve_dataset_schema,
)
from ml4t.models.integration import data as data_module


@dataclass
class FeedMetadataLike:
    timestamp_col: str = "date"
    entity_col: str = "ticker"


def test_resolve_dataset_schema_accepts_metadata_like_object() -> None:
    frame = {
        "date": np.array(["2024-01-01", "2024-01-01"], dtype=object),
        "ticker": np.array(["AAPL", "MSFT"], dtype=object),
        "close": np.array([100.0, 200.0], dtype=np.float64),
    }
    resolved = resolve_dataset_schema(frame, schema=FeedMetadataLike())

    assert resolved.timestamp_col == "date"
    assert resolved.entity_col == "ticker"


def test_resolve_dataset_schema_rejects_missing_or_unknown_columns() -> None:
    with pytest.raises(ValueError, match="Could not resolve a timestamp"):
        resolve_dataset_schema({"value": np.ones(2)})
    with pytest.raises(ValueError, match="Could not resolve an entity"):
        resolve_dataset_schema({"timestamp": np.arange(2)})
    with pytest.raises(ValueError, match="timestamp column 'missing'.*not found"):
        resolve_dataset_schema(
            {"timestamp": np.arange(2), "asset": np.arange(2)},
            timestamp_col="missing",
        )
    with pytest.raises(ValueError, match="entity column 'missing'.*not found"):
        resolve_dataset_schema(
            {"timestamp": np.arange(2), "asset": np.arange(2)},
            entity_col="missing",
        )


def test_persistent_panel_batch_from_long_frame_uses_nested_schema_mapping() -> None:
    frame = {
        "date": np.array(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"], dtype=object),
        "asset_ticker": np.array(["AAPL", "MSFT", "AAPL", "MSFT"], dtype=object),
        "ret_1d": np.array([0.01, 0.02, -0.01, 0.03], dtype=np.float64),
        "size": np.array([10.0, 20.0, 11.0, 19.0], dtype=np.float64),
    }
    batch = persistent_panel_batch_from_long_frame(
        frame,
        schema={"schema": {"timestamp_col": "date", "entity_col": "asset_ticker"}},
        return_col="ret_1d",
        feature_cols=("size",),
    )

    assert batch.timestamps == ("2024-01-01", "2024-01-02")
    assert batch.asset_ids == ("AAPL", "MSFT")
    assert batch.returns is not None
    assert batch.returns.shape == (2, 2)
    assert batch.characteristics is not None
    assert batch.characteristics.shape == (2, 2, 1)
    assert batch.returns[1, 0] == -0.01


def test_persistent_panel_supports_metadata_only_and_nonfinite_values() -> None:
    frame = {
        "timestamp": np.array(["t1", "t2"], dtype=object),
        "asset": np.array(["A", "A"], dtype=object),
        "return": np.array(["missing", 0.1], dtype=object),
        "feature": np.array([np.inf, 2.0], dtype=object),
    }

    batch = persistent_panel_batch_from_long_frame(
        frame,
        return_col="return",
        feature_cols=("feature",),
        metadata={"source": "test"},
    )
    labels_only = persistent_panel_batch_from_long_frame(frame)

    assert batch.metadata["source"] == "test"
    assert batch.returns is not None and np.isnan(batch.returns[0, 0])
    assert batch.characteristics is not None and np.isnan(batch.characteristics[0, 0, 0])
    assert labels_only.returns is None
    assert labels_only.characteristics is None


def test_persistent_panel_rejects_duplicate_rows() -> None:
    frame = {
        "timestamp": np.array(["t1", "t1"], dtype=object),
        "asset": np.array(["A", "A"], dtype=object),
    }

    with pytest.raises(ValueError, match=r"Duplicate \(timestamp, entity\)"):
        persistent_panel_batch_from_long_frame(frame)


def test_cross_section_batch_from_long_frame_builds_mask_and_context_features() -> None:
    frame = {
        "datetime": np.array(
            ["2024-01-01", "2024-01-01", "2024-01-02"],
            dtype=object,
        ),
        "symbol": np.array(["AAPL", "MSFT", "AAPL"], dtype=object),
        "feature_1": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "feature_2": np.array([0.5, 0.25, 0.75], dtype=np.float64),
        "return_1d": np.array([0.01, -0.02, 0.03], dtype=np.float64),
        "regime_value": np.array([1.0, 1.0, 2.0], dtype=np.float64),
    }
    batch = cross_section_batch_from_long_frame(
        frame,
        feature_cols=("feature_1", "feature_2"),
        return_col="return_1d",
        context_cols=("regime_value",),
        schema={"timestamp_col": "datetime", "entity_col": "symbol"},
    )

    assert batch.characteristics.shape == (2, 2, 2)
    assert batch.returns is not None
    assert batch.returns.shape == (2, 2)
    assert batch.context_features is not None
    assert batch.context_features.shape == (2, 1)
    assert batch.mask is not None
    assert batch.mask[1, 1] == np.False_
    assert batch.asset_ids == ("AAPL", "MSFT")
    assert batch.context_features[1, 0] == 2.0


def test_cross_section_long_frame_preserves_identity_across_entry_exit_and_reorder() -> None:
    frame = {
        "date": np.array(
            ["2024-01-02", "2024-01-01", "2024-01-03", "2024-01-01", "2024-01-02"],
            dtype=object,
        ),
        "asset": np.array(["B", "B", "C", "A", "A"], dtype=object),
        "feature": np.array([20.0, 2.0, 30.0, 1.0, 10.0], dtype=np.float64),
    }

    batch = cross_section_batch_from_long_frame(frame, feature_cols=("feature",))

    assert batch.timestamps == ("2024-01-01", "2024-01-02", "2024-01-03")
    assert batch.asset_ids == ("A", "B", "C")
    assert batch.mask is not None
    assert batch.mask.tolist() == [
        [True, True, False],
        [True, True, False],
        [False, False, True],
    ]
    assert batch.characteristics[0, 0, 0] == 1.0
    assert batch.characteristics[0, 1, 0] == 2.0
    assert batch.characteristics[2, 2, 0] == 30.0


def test_cross_section_supports_unlabeled_nonfinite_rows_and_empty_context() -> None:
    frame = {
        "timestamp": np.array(["t1", "t1", "t2"], dtype=object),
        "asset": np.array(["A", "B", "A"], dtype=object),
        "feature": np.array(["missing", 2.0, 3.0], dtype=object),
        "context": np.array([None, np.nan, 1.0], dtype=object),
    }

    batch = cross_section_batch_from_long_frame(
        frame,
        feature_cols=("feature",),
        context_cols=("context",),
        metadata={"source": "test"},
    )

    assert batch.returns is None
    assert np.isnan(batch.characteristics[0, 0, 0])
    assert batch.context_features is not None
    assert np.isnan(batch.context_features[0, 0])
    assert batch.context_features[1, 0] == 1.0
    assert batch.metadata["source"] == "test"


def test_cross_section_rejects_duplicate_rows_and_varying_context() -> None:
    duplicate = {
        "timestamp": np.array(["t1", "t1"], dtype=object),
        "asset": np.array(["A", "A"], dtype=object),
        "feature": np.ones(2),
    }
    with pytest.raises(ValueError, match=r"Duplicate \(timestamp, entity\)"):
        cross_section_batch_from_long_frame(duplicate, feature_cols=("feature",))

    varying_context = {
        "timestamp": np.array(["t1", "t1"], dtype=object),
        "asset": np.array(["A", "B"], dtype=object),
        "feature": np.ones(2),
        "context": np.array([1.0, 2.0]),
    }
    with pytest.raises(ValueError, match="must be constant within timestamp"):
        cross_section_batch_from_long_frame(
            varying_context,
            feature_cols=("feature",),
            context_cols=("context",),
        )


def test_schema_coercion_supports_metadata_and_nested_object_shapes() -> None:
    from_mapping_metadata = data_module._coerce_schema(
        {"metadata": {"date_col": "date", "ticker_col": "ticker"}}
    )
    from_object_metadata = data_module._coerce_schema(
        SimpleNamespace(metadata={"time_col": "time", "symbol_col": "symbol"})
    )
    from_nested_object = data_module._coerce_schema(
        SimpleNamespace(
            timestamp_col="outer_time",
            schema=SimpleNamespace(date_col="inner_time", asset_col="asset"),
        )
    )

    assert from_mapping_metadata == {"timestamp_col": "date", "entity_col": "ticker"}
    assert from_object_metadata == {"timestamp_col": "time", "entity_col": "symbol"}
    assert from_nested_object == {"timestamp_col": "inner_time", "entity_col": "asset"}
    assert data_module._coerce_schema({"unrelated": "value"}) == {}
    assert (
        data_module._pick_field(SimpleNamespace(value=None), "value", "missing")
        is data_module._MISSING
    )


class _ArrayColumn:
    def __init__(self, values: list[object]) -> None:
        self._values = values

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self._values, dtype=object)


class _TabularFrame:
    columns = ("timestamp", "asset", "value")

    def __init__(self) -> None:
        self._columns = {
            "timestamp": _ArrayColumn(["t2", "t1"]),
            "asset": _ArrayColumn(["B", "A"]),
            "value": _ArrayColumn([2.0, 1.0]),
        }

    def __getitem__(self, name: str) -> _ArrayColumn:
        return self._columns[name]


def test_tabular_frame_helpers_support_to_numpy_columns() -> None:
    frame = _TabularFrame()

    resolved = resolve_dataset_schema(frame)
    records = data_module._sorted_records(frame, timestamp_col="timestamp", entity_col="asset")

    assert resolved.timestamp_col == "timestamp"
    assert records[0]["value"] == 1.0
    assert data_module._sorted_records({}, timestamp_col="timestamp", entity_col="asset") == []


def test_frame_helpers_reject_unsupported_objects_and_nonfinite_text() -> None:
    with pytest.raises(TypeError, match="frame must be a mapping"):
        data_module._frame_columns(object())

    assert data_module._first_present(("a",), ("b", "c")) is None
    assert not data_module._is_finite("not-a-number")
