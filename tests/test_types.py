from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from ml4t.models import (
    AssetForecastResult,
    AssetSignalResult,
    AssetWeightsResult,
    CrossSectionBatch,
    FactorForecastResult,
    LatentFactorState,
    PersistentPanelBatch,
    PortfolioSequenceBatch,
    PortfolioWeightsResult,
    StochasticDiscountFactorState,
)


def test_persistent_panel_can_infer_shape_from_metadata_only() -> None:
    batch = PersistentPanelBatch(
        timestamps=("2024-01", "2024-02"),
        asset_ids=("A", "B", "C"),
    )
    assert batch.n_periods == 2
    assert batch.n_assets == 3


def test_persistent_panel_infers_shape_from_characteristics() -> None:
    batch = PersistentPanelBatch(characteristics=np.ones((2, 3, 4)))

    assert batch.n_periods == 2
    assert batch.n_assets == 3


def test_persistent_panel_requires_shape_information() -> None:
    with pytest.raises(ValueError, match="determine panel shape"):
        PersistentPanelBatch()
    with pytest.raises(ValueError, match="returns must be 2D; got shape"):
        PersistentPanelBatch(returns=np.ones(2))


def test_persistent_panel_rejects_characteristic_and_return_misalignment() -> None:
    with pytest.raises(ValueError, match="characteristics and returns disagree"):
        PersistentPanelBatch(
            returns=np.ones((2, 3)),
            characteristics=np.ones((2, 4, 1)),
        )


def test_persistent_panel_rejects_timestamp_misalignment() -> None:
    with pytest.raises(ValueError, match="timestamps length"):
        PersistentPanelBatch(returns=np.ones((2, 3)), timestamps=("t1",))


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


def test_cross_section_batch_validates_timestamp_and_mask_alignment() -> None:
    with pytest.raises(ValueError, match="timestamps length"):
        CrossSectionBatch(characteristics=np.ones((2, 3, 1)), timestamps=("t1",))
    with pytest.raises(ValueError, match="mask must match"):
        CrossSectionBatch(
            characteristics=np.ones((2, 3, 1)),
            mask=np.ones((3, 2), dtype=bool),
        )


def test_cross_section_batch_properties_report_panel_shape() -> None:
    batch = CrossSectionBatch(characteristics=np.ones((2, 3, 1)))

    assert batch.n_periods == 2
    assert batch.n_assets == 3


def test_portfolio_sequence_batch_normalizes_cost_shape() -> None:
    batch = PortfolioSequenceBatch(
        features=np.zeros((2, 4, 3, 5), dtype=np.float64),
        returns=np.zeros((2, 4, 3), dtype=np.float64),
        vol_scale=np.ones((2, 4, 3), dtype=np.float64),
        costs=np.array([0.001, 0.002, 0.003], dtype=np.float64),
    )
    assert batch.costs is not None
    assert batch.costs.shape == (3, 1)


def test_portfolio_sequence_batch_accepts_column_costs_and_reports_shape() -> None:
    batch = PortfolioSequenceBatch(
        features=np.zeros((2, 4, 3, 5)),
        costs=np.ones((3, 1)),
        timestamps=("t1", "t2", "t3", "t4"),
    )

    assert batch.costs is not None
    assert batch.costs.shape == (3, 1)
    assert batch.batch_size == 2
    assert batch.n_periods == 4
    assert batch.n_assets == 3


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("returns", np.ones((3, 4, 3)), r"returns and features disagree on \(B, T\)"),
        ("returns", np.ones((2, 4, 2)), "returns and features disagree on N"),
        ("vol_scale", np.ones((2, 3, 3)), "vol_scale and features disagree"),
        ("prev_weights", np.ones((3, 3)), "prev_weights must have shape"),
        ("mask", np.ones((2, 3, 3)), "mask must match"),
        ("group_ids", np.ones(2), "group_ids must have shape"),
        ("costs", np.ones((3, 2)), "costs must have shape"),
        ("adjacency_mask", np.ones((3, 2)), "adjacency_mask must have shape"),
    ],
)
def test_portfolio_sequence_batch_rejects_misaligned_optional_arrays(
    field: str, value: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PortfolioSequenceBatch(features=np.zeros((2, 4, 3, 5)), **{field: value})


@pytest.mark.parametrize("timestamps", [("t1", "t2"), ("t1", "t2", "t3", "t4", "t5")])
def test_portfolio_sequence_batch_rejects_misaligned_timestamps(
    timestamps: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match=r"timestamps.*expected 4.*got"):
        PortfolioSequenceBatch(
            features=np.zeros((2, 4, 3, 5), dtype=np.float64),
            timestamps=timestamps,
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LatentFactorState(asset_betas=np.ones((2, 3, 1)), timestamps=("t1",)),
        lambda: FactorForecastResult(factor_premia=np.ones((2, 1)), timestamps=("t1",)),
        lambda: AssetForecastResult(expected_returns=np.ones((2, 3)), timestamps=("t1",)),
        lambda: AssetSignalResult(signal_values=np.ones((2, 3)), timestamps=("t1",)),
        lambda: AssetWeightsResult(weights=np.ones((2, 3)), timestamps=("t1",)),
        lambda: StochasticDiscountFactorState(asset_weights=np.ones((2, 3)), timestamps=("t1",)),
        lambda: PortfolioWeightsResult(weights=np.ones((1, 2, 3)), timestamps=("t1",)),
    ],
)
def test_result_types_reject_timestamp_misalignment(factory: Callable[[], object]) -> None:
    with pytest.raises(ValueError, match="timestamps length"):
        factory()


def test_factor_forecast_result_accepts_unlabeled_periods() -> None:
    result = FactorForecastResult(factor_premia=np.ones((2, 1)))

    assert result.factor_premia.shape == (2, 1)


@pytest.mark.parametrize(
    "asset_ids",
    [
        ("A",),
        ("A", "A"),
        ("A", ""),
    ],
)
@pytest.mark.parametrize(
    "factory",
    [
        lambda ids: PersistentPanelBatch(returns=np.ones((2, 2)), asset_ids=ids),
        lambda ids: CrossSectionBatch(characteristics=np.ones((2, 2, 1)), asset_ids=ids),
        lambda ids: PortfolioSequenceBatch(features=np.ones((1, 2, 2, 1)), asset_ids=ids),
        lambda ids: LatentFactorState(asset_betas=np.ones((2, 2, 1)), asset_ids=ids),
        lambda ids: AssetForecastResult(expected_returns=np.ones((2, 2)), asset_ids=ids),
        lambda ids: AssetSignalResult(signal_values=np.ones((2, 2)), asset_ids=ids),
        lambda ids: AssetWeightsResult(weights=np.ones((2, 2)), asset_ids=ids),
        lambda ids: StochasticDiscountFactorState(asset_weights=np.ones((2, 2)), asset_ids=ids),
        lambda ids: PortfolioWeightsResult(weights=np.ones((1, 2, 2)), asset_ids=ids),
    ],
)
def test_panel_and_result_types_reject_invalid_asset_ids(
    factory: Callable[[tuple[str, ...]], object], asset_ids: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="asset_ids"):
        factory(asset_ids)


def test_latent_factor_state_validates_factor_shape_and_reports_dimensions() -> None:
    with pytest.raises(ValueError, match=r"factor_returns must have shape \(T, K\)"):
        LatentFactorState(
            asset_betas=np.ones((2, 3, 1)),
            factor_returns=np.ones((2, 2)),
        )

    state = LatentFactorState(
        asset_betas=np.ones((2, 3, 1)),
        factor_returns=np.ones((2, 1)),
    )
    assert state.n_periods == 2
    assert state.n_assets == 3
    assert state.n_factors == 1


def test_stochastic_discount_factor_state_validates_values_and_reports_dimensions() -> None:
    with pytest.raises(ValueError, match=r"sdf_values must have shape \(T,\)"):
        StochasticDiscountFactorState(
            asset_weights=np.ones((2, 3)),
            sdf_values=np.ones(3),
        )

    state = StochasticDiscountFactorState(
        asset_weights=np.ones((2, 3)),
        sdf_values=np.ones(2),
    )
    assert state.n_periods == 2
    assert state.n_assets == 3
