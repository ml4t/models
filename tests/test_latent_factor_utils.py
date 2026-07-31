from __future__ import annotations

import numpy as np
import pytest

from ml4t.models._internal.latent_factor_utils import (
    average_ranks,
    compute_managed_portfolios,
    mean_cross_sectional_spearman,
    resolve_checkpoint_epochs,
    select_checkpoint_epoch,
    summarize_predictions,
    validate_panel_shapes,
)


def test_validate_panel_shapes_accepts_aligned_training_and_validation() -> None:
    validate_panel_shapes(
        np.ones((2, 3, 4)),
        np.ones((2, 3)),
        np.ones((1, 3, 4)),
        np.ones((1, 3)),
    )
    validate_panel_shapes(np.ones((2, 3, 4)), np.ones((2, 3)))


@pytest.mark.parametrize(
    ("chars_train", "returns_train", "chars_val", "returns_val", "message"),
    [
        (np.ones((2, 3)), np.ones((2, 3)), None, None, "chars_train must be 3D"),
        (np.ones((2, 3, 1)), np.ones((2, 3, 1)), None, None, "returns_train must be 2D"),
        (
            np.ones((2, 3, 1)),
            np.ones((3, 2)),
            None,
            None,
            "chars_train and returns_train disagree",
        ),
        (
            np.ones((2, 3, 1)),
            np.ones((2, 3)),
            np.ones((2, 3)),
            None,
            "chars_val must be 3D",
        ),
        (
            np.ones((2, 3, 1)),
            np.ones((2, 3)),
            np.ones((2, 3, 2)),
            None,
            "must share the feature dimension",
        ),
        (
            np.ones((2, 3, 1)),
            np.ones((2, 3)),
            np.ones((2, 3, 1)),
            np.ones((2, 3, 1)),
            "returns_val must be 2D",
        ),
        (
            np.ones((2, 3, 1)),
            np.ones((2, 3)),
            None,
            np.ones((2, 3)),
            "chars_val is required",
        ),
        (
            np.ones((2, 3, 1)),
            np.ones((2, 3)),
            np.ones((2, 3, 1)),
            np.ones((3, 2)),
            "chars_val and returns_val disagree",
        ),
    ],
)
def test_validate_panel_shapes_rejects_each_misalignment(
    chars_train: np.ndarray,
    returns_train: np.ndarray,
    chars_val: np.ndarray | None,
    returns_val: np.ndarray | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_panel_shapes(chars_train, returns_train, chars_val, returns_val)


def test_managed_portfolios_leave_dates_without_observations_at_zero() -> None:
    result = compute_managed_portfolios(
        np.array([[[1.0], [2.0]], [[3.0], [4.0]]]),
        np.array([[np.nan, np.nan], [1.0, 2.0]]),
    )

    np.testing.assert_array_equal(result[0], 0.0)
    assert np.isfinite(result[1]).all()


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"checkpoint_interval": 2}, [2, 4, 5]),
        ({"checkpoint_interval": 10}, [5]),
        ({"checkpoint_interval": None}, [5]),
        ({"checkpoint_interval": 0}, [5]),
        ({"checkpoint_epochs": [5, 2, 2, 8, 0]}, [2, 5]),
        ({"checkpoint_epochs": [2], "include_final": False}, [2]),
    ],
)
def test_resolve_checkpoint_epochs_covers_supported_schedules(
    kwargs: dict[str, object], expected: list[int]
) -> None:
    assert resolve_checkpoint_epochs(5, **kwargs) == expected


def test_resolve_checkpoint_epochs_rejects_empty_schedules() -> None:
    with pytest.raises(ValueError, match="max_epoch must be positive"):
        resolve_checkpoint_epochs(0)
    with pytest.raises(ValueError, match="did not contain a valid epoch"):
        resolve_checkpoint_epochs(5, checkpoint_epochs=[0, 6])


def test_select_checkpoint_epoch_honors_explicit_default_and_latest() -> None:
    available = (2, 4, 5)

    assert select_checkpoint_epoch(checkpoint=2, configured_default=4, available=available) == 2
    assert select_checkpoint_epoch(checkpoint=None, configured_default=4, available=available) == 4
    assert (
        select_checkpoint_epoch(checkpoint=None, configured_default=None, available=available) == 5
    )


def test_select_checkpoint_epoch_rejects_unavailable_selections() -> None:
    with pytest.raises(ValueError, match="checkpoint=3"):
        select_checkpoint_epoch(checkpoint=3, configured_default=None, available=(2, 4))
    with pytest.raises(ValueError, match="default_checkpoint=3"):
        select_checkpoint_epoch(checkpoint=None, configured_default=3, available=(2, 4))


def test_summarize_predictions_reports_empty_task_specific_metrics() -> None:
    missing = np.array([[np.nan]])

    assert summarize_predictions(missing, missing, task_type="classification") == {
        "n_validation_obs": 0,
        "validation_auc": None,
        "validation_log_loss": None,
    }
    assert summarize_predictions(missing, missing, task_type="regression") == {
        "n_validation_obs": 0,
        "validation_mean_cs_ic": None,
    }


def test_summarize_classification_predictions_matches_hand_computed_metrics() -> None:
    summary = summarize_predictions(
        np.array([[0.0, 1.0, 0.0, 1.0]]),
        np.array([[0.1, 0.9, 0.4, 0.8]]),
        task_type="classification",
    )

    assert summary["n_validation_obs"] == 4
    assert summary["validation_auc"] == pytest.approx(1.0)
    assert summary["validation_log_loss"] == pytest.approx(-np.mean(np.log([0.9, 0.9, 0.6, 0.8])))


def test_summarize_classification_predictions_handles_one_class_and_clipping() -> None:
    summary = summarize_predictions(
        np.zeros((1, 3)),
        np.array([[-1.0, 0.0, 2.0]]),
        task_type="classification",
    )

    assert summary["validation_auc"] is None
    assert np.isfinite(summary["validation_log_loss"])


def test_summarize_regression_predictions_uses_mean_cross_sectional_rank_ic() -> None:
    actual = summarize_predictions(
        np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
        np.array([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]]),
        task_type="regression",
    )

    assert actual == {"n_validation_obs": 6, "validation_mean_cs_ic": 0.0}


def test_mean_cross_sectional_spearman_skips_small_and_constant_sections() -> None:
    assert (
        mean_cross_sectional_spearman(
            np.array(
                [
                    [1.0, 2.0, np.nan],
                    [1.0, 2.0, 3.0],
                    [4.0, 4.0, 4.0],
                ]
            ),
            np.array(
                [
                    [1.0, 2.0, np.nan],
                    [4.0, 4.0, 4.0],
                    [1.0, 2.0, 3.0],
                ]
            ),
        )
        is None
    )


def test_average_ranks_is_stable_for_ties_and_empty_input() -> None:
    np.testing.assert_array_equal(
        average_ranks(np.array([3.0, 1.0, 1.0, 2.0])),
        np.array([4.0, 1.5, 1.5, 3.0]),
    )
    np.testing.assert_array_equal(average_ranks(np.array([])), np.array([]))
