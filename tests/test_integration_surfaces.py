from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml4t.models import (
    AssetForecastResult,
    AssetSignalResult,
    AssetWeightsResult,
    PortfolioWeightsResult,
    ResultsFrame,
    context_frame_from_weights,
    predictions_frame_from_asset_forecast,
    predictions_frame_from_asset_signal,
    signals_frame_from_asset_weights,
    signals_frame_from_portfolio_weights,
    weights_frame_from_asset_weights,
    weights_frame_from_portfolio_weights,
    write_backtest_frames,
)
from ml4t.models.integration import surfaces as surfaces_module


def test_results_frame_columnar_polars_and_parquet_exports(tmp_path: Path) -> None:
    frame = ResultsFrame(columns=("a", "b"), rows=((1, "x"), (2, "y")))

    assert frame.to_columnar() == {"a": [1, 2], "b": ["x", "y"]}
    assert frame.to_polars().to_dicts() == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

    output = frame.write_parquet(tmp_path / "nested" / "frame.parquet", compression="snappy")

    assert output.is_file()


def test_predictions_frame_uses_diagnostic_column_names() -> None:
    forecast = AssetForecastResult(
        expected_returns=np.array([[0.1, np.nan], [0.2, 0.3]], dtype=np.float64),
        timestamps=("2024-01-01", "2024-01-02"),
        asset_ids=("AAPL", "MSFT"),
        metadata={"model_name": "ipca"},
    )

    frame = predictions_frame_from_asset_forecast(
        forecast,
        constants={"config_name": "baseline"},
    )

    assert frame.columns == ("timestamp", "asset", "prediction_value", "config_name")
    assert frame.metadata["frame_type"] == "prediction"
    assert frame.to_dicts()[0]["asset"] == "AAPL"
    assert frame.to_dicts()[0]["config_name"] == "baseline"
    assert len(frame.rows) == 3


def test_predictions_frame_supports_generic_asset_signals() -> None:
    signal = AssetSignalResult(
        signal_values=np.array([[0.4, np.nan], [-0.2, 0.1]], dtype=np.float64),
        timestamps=("2024-01-01", "2024-01-02"),
        asset_ids=("AAPL", "MSFT"),
        metadata={"model_name": "sae"},
    )

    frame = predictions_frame_from_asset_signal(signal)

    assert frame.columns == ("timestamp", "asset", "prediction_value")
    assert frame.metadata["model_name"] == "sae"
    assert len(frame.rows) == 3


def test_signals_frame_uses_signal_value_and_selected() -> None:
    weights = PortfolioWeightsResult(
        weights=np.array([[[0.1, 0.0], [-0.2, 0.3]]], dtype=np.float64),
        checkpoint_step=5,
        timestamps=("2024-01-01", "2024-01-02"),
        asset_ids=("AAPL", "MSFT"),
    )

    frame = signals_frame_from_portfolio_weights(weights)

    assert frame.columns == ("timestamp", "asset", "signal_value", "selected")
    rows = frame.to_dicts()
    assert rows[0]["signal_value"] == 0.1
    assert rows[1]["selected"] is False
    assert frame.metadata["checkpoint_step"] == 5


def test_weights_frame_adds_batch_id_for_multi_batch_outputs() -> None:
    weights = PortfolioWeightsResult(
        weights=np.array(
            [
                [[0.1, -0.1]],
                [[0.2, -0.2]],
            ],
            dtype=np.float64,
        ),
        timestamps=("2024-01-01",),
        asset_ids=("AAPL", "MSFT"),
    )

    frame = weights_frame_from_portfolio_weights(weights, constants={"run_id": "r1"})

    assert frame.columns == ("timestamp", "asset", "batch_id", "weight", "selected", "run_id")
    rows = frame.to_dicts()
    assert rows[0]["batch_id"] == 0
    assert rows[-1]["batch_id"] == 1
    assert rows[-1]["run_id"] == "r1"


def test_asset_weight_frames_support_sdf_style_outputs() -> None:
    weights = AssetWeightsResult(
        weights=np.array([[0.4, -0.1], [0.0, 0.2]], dtype=np.float64),
        timestamps=("2024-01-01", "2024-01-02"),
        asset_ids=("AAPL", "MSFT"),
    )

    signals_frame = signals_frame_from_asset_weights(weights)
    weights_frame = weights_frame_from_asset_weights(weights)

    assert signals_frame.columns == ("timestamp", "asset", "signal_value", "selected")
    assert weights_frame.columns == ("timestamp", "asset", "weight", "selected")
    assert signals_frame.to_dicts()[1]["selected"] is True
    assert weights_frame.to_dicts()[2]["selected"] is False


def test_context_frame_from_weights_builds_wide_context_frame() -> None:
    weights = AssetWeightsResult(
        weights=np.array([[0.4, -0.1], [0.0, 0.2]], dtype=np.float64),
        timestamps=("2024-01-01", "2024-01-02"),
        asset_ids=("AAPL", "MSFT"),
        metadata={"family": "sdf"},
    )

    frame = context_frame_from_weights(weights, prefix="tw_", constants={"run_id": "r1"})

    assert frame.columns == ("timestamp", "tw_AAPL", "tw_MSFT", "run_id")
    rows = frame.to_dicts()
    assert rows[0]["tw_AAPL"] == 0.4
    assert rows[1]["tw_AAPL"] == 0.0
    assert rows[1]["run_id"] == "r1"
    assert frame.metadata["frame_type"] == "context"


def test_context_frame_from_portfolio_weights_records_checkpoint_and_fills_nan() -> None:
    weights = PortfolioWeightsResult(
        weights=np.array([[[0.4, np.nan]]]),
        checkpoint_step=7,
    )

    frame = context_frame_from_weights(weights)

    assert frame.to_dicts() == [{"timestamp": 0, "w_asset_0": 0.4, "w_asset_1": 0.0}]
    assert frame.metadata["checkpoint_step"] == 7


def test_context_frame_rejects_multiple_portfolio_batches() -> None:
    with pytest.raises(ValueError, match="requires a single portfolio-weight batch"):
        context_frame_from_weights(PortfolioWeightsResult(weights=np.ones((2, 1, 1))))


def test_frame_to_polars_requires_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = weights_frame_from_portfolio_weights(
        PortfolioWeightsResult(weights=np.array([[[0.1]]], dtype=np.float64))
    )

    def _raise_import_error(name: str) -> None:
        raise ImportError(name)

    monkeypatch.setattr("ml4t.models.integration.surfaces.import_module", _raise_import_error)
    with pytest.raises(ImportError, match="ml4t-models\\[integration\\]"):
        frame.to_polars()


def test_write_backtest_frames_uses_standard_artifact_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    predictions_frame = predictions_frame_from_asset_forecast(
        AssetForecastResult(expected_returns=np.array([[0.1]], dtype=np.float64))
    )
    weights_frame = weights_frame_from_portfolio_weights(
        PortfolioWeightsResult(weights=np.array([[[0.2]]], dtype=np.float64))
    )
    written_paths: list[str] = []

    def _write_parquet(self, path, *, compression="zstd"):
        written_paths.append(f"{Path(path).name}:{compression}")
        output = Path(path)
        output.write_bytes(b"parquet")
        return output

    monkeypatch.setattr(
        "ml4t.models.integration.surfaces.ResultsFrame.write_parquet",
        _write_parquet,
    )
    written = write_backtest_frames(
        tmp_path,
        predictions=predictions_frame,
        weights=weights_frame,
    )

    assert set(written) == {"predictions", "weights"}
    assert "predictions.parquet:zstd" in written_paths
    assert "weights.parquet:zstd" in written_paths
    assert (tmp_path / "manifest.json").is_file()


def test_write_backtest_frames_removes_omitted_stale_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    predictions_frame = predictions_frame_from_asset_forecast(
        AssetForecastResult(expected_returns=np.array([[0.1]], dtype=np.float64))
    )
    weights_frame = weights_frame_from_portfolio_weights(
        PortfolioWeightsResult(weights=np.array([[[0.2]]], dtype=np.float64))
    )

    def _write_parquet(self, path, *, compression="zstd"):
        del self, compression
        output = Path(path)
        output.write_bytes(output.name.encode())
        return output

    monkeypatch.setattr(
        "ml4t.models.integration.surfaces.ResultsFrame.write_parquet",
        _write_parquet,
    )
    output_dir = tmp_path / "artifacts"
    write_backtest_frames(output_dir, predictions=predictions_frame, weights=weights_frame)

    write_backtest_frames(output_dir, predictions=predictions_frame)

    assert (output_dir / "predictions.parquet").is_file()
    assert not (output_dir / "weights.parquet").exists()


def test_write_backtest_frames_preserves_prior_generation_on_write_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    predictions_frame = predictions_frame_from_asset_forecast(
        AssetForecastResult(expected_returns=np.array([[0.1]], dtype=np.float64))
    )
    weights_frame = weights_frame_from_portfolio_weights(
        PortfolioWeightsResult(weights=np.array([[[0.2]]], dtype=np.float64))
    )
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    (output_dir / "predictions.parquet").write_bytes(b"old predictions")
    (output_dir / "weights.parquet").write_bytes(b"old weights")
    (output_dir / "manifest.json").write_text("old manifest", encoding="utf-8")

    def _fail_second_write(self, path, *, compression="zstd"):
        del self, compression
        output = Path(path)
        if output.name == "weights.parquet":
            raise OSError("injected write failure")
        output.write_bytes(b"new predictions")
        return output

    monkeypatch.setattr(
        "ml4t.models.integration.surfaces.ResultsFrame.write_parquet",
        _fail_second_write,
    )

    with pytest.raises(OSError, match="injected write failure"):
        write_backtest_frames(
            output_dir,
            predictions=predictions_frame,
            weights=weights_frame,
        )

    assert (output_dir / "predictions.parquet").read_bytes() == b"old predictions"
    assert (output_dir / "weights.parquet").read_bytes() == b"old weights"
    assert (output_dir / "manifest.json").read_text(encoding="utf-8") == "old manifest"


def test_write_backtest_frames_restores_prior_generation_on_publish_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import os

    predictions_frame = predictions_frame_from_asset_forecast(
        AssetForecastResult(expected_returns=np.array([[0.1]], dtype=np.float64))
    )
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    (output_dir / "predictions.parquet").write_bytes(b"old predictions")
    (output_dir / "manifest.json").write_text("old manifest", encoding="utf-8")

    def _write_parquet(self, path, *, compression="zstd"):
        del self, compression
        output = Path(path)
        output.write_bytes(b"new predictions")
        return output

    real_replace = os.replace
    failed = False

    def _fail_publish(source, destination):
        nonlocal failed
        source_path = Path(source)
        if not failed and source_path.name.startswith(".artifacts.staging-"):
            failed = True
            raise OSError("injected publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(
        "ml4t.models.integration.surfaces.ResultsFrame.write_parquet",
        _write_parquet,
    )
    monkeypatch.setattr("ml4t.models.integration.surfaces.os.replace", _fail_publish)

    with pytest.raises(OSError, match="injected publish failure"):
        write_backtest_frames(output_dir, predictions=predictions_frame)

    assert (output_dir / "predictions.parquet").read_bytes() == b"old predictions"
    assert (output_dir / "manifest.json").read_text(encoding="utf-8") == "old manifest"


def test_weight_frames_skip_nonfinite_values() -> None:
    asset_frame = weights_frame_from_asset_weights(
        AssetWeightsResult(weights=np.array([[0.1, np.nan]]))
    )
    portfolio_frame = weights_frame_from_portfolio_weights(
        PortfolioWeightsResult(weights=np.array([[[0.1, np.nan]]]))
    )

    assert len(asset_frame.rows) == 1
    assert len(portfolio_frame.rows) == 1


def test_write_backtest_frames_rejects_file_target(tmp_path: Path) -> None:
    target = tmp_path / "artifact"
    target.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact_dir must be a directory"):
        write_backtest_frames(target)


def test_write_backtest_frames_rejects_writer_without_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frame = predictions_frame_from_asset_forecast(
        AssetForecastResult(expected_returns=np.array([[0.1]]))
    )

    def do_not_write(self: object, path: str | Path, *, compression: str = "zstd") -> Path:
        del self, compression
        return Path(path)

    monkeypatch.setattr(
        "ml4t.models.integration.surfaces.ResultsFrame.write_parquet",
        do_not_write,
    )

    with pytest.raises(OSError, match="writer did not create"):
        write_backtest_frames(tmp_path / "artifacts", predictions=frame)


def test_publish_failure_without_prior_generation_cleans_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    output = tmp_path / "output"

    def fail_replace(source: object, destination: object) -> None:
        raise OSError(f"cannot replace {source} with {destination}")

    monkeypatch.setattr(surfaces_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="cannot replace"):
        surfaces_module._publish_generation(staging, output)
