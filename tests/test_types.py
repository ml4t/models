from __future__ import annotations

import numpy as np
import pytest

from ml4t.models import CrossSectionBatch, PersistentPanelBatch, PortfolioSequenceBatch


def test_persistent_panel_can_infer_shape_from_metadata_only() -> None:
    batch = PersistentPanelBatch(
        timestamps=("2024-01", "2024-02"),
        asset_ids=("A", "B", "C"),
    )
    assert batch.n_periods == 2
    assert batch.n_assets == 3


def test_cross_section_batch_validates_panel_alignment() -> None:
    with pytest.raises(ValueError):
        CrossSectionBatch(
            characteristics=np.zeros((2, 3, 4), dtype=np.float64),
            returns=np.zeros((2, 4), dtype=np.float64),
        )


def test_cross_section_batch_validates_factor_return_alignment() -> None:
    with pytest.raises(ValueError):
        CrossSectionBatch(
            characteristics=np.zeros((2, 3, 4), dtype=np.float64),
            factor_returns=np.zeros((3, 3), dtype=np.float64),
        )


def test_cross_section_batch_validates_context_alignment() -> None:
    with pytest.raises(ValueError):
        CrossSectionBatch(
            characteristics=np.zeros((2, 3, 4), dtype=np.float64),
            context_features=np.zeros((3, 2), dtype=np.float64),
        )


def test_portfolio_sequence_batch_normalizes_cost_shape() -> None:
    batch = PortfolioSequenceBatch(
        features=np.zeros((2, 4, 3, 5), dtype=np.float64),
        returns=np.zeros((2, 4, 3), dtype=np.float64),
        vol_scale=np.ones((2, 4, 3), dtype=np.float64),
        costs=np.array([0.001, 0.002, 0.003], dtype=np.float64),
    )
    assert batch.costs is not None
    assert batch.costs.shape == (3, 1)
