from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml4t.models import (
    cross_section_batch_from_long_frame,
    persistent_panel_batch_from_long_frame,
    resolve_dataset_schema,
)


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
    assert batch.context_features[1, 0] == 2.0
