from __future__ import annotations

import numpy as np

from ml4t.models import (
    AssetForecastResult,
    persistent_panel_batch_from_long_frame,
    predictions_frame_from_asset_forecast,
)

frame = {
    "timestamp": np.array(["2024-01", "2024-01", "2024-02", "2024-02"]),
    "asset": np.array(["A", "B", "A", "B"]),
    "return": np.array([0.01, 0.02, -0.01, 0.03]),
}
batch = persistent_panel_batch_from_long_frame(frame, return_col="return")
assert batch.returns is not None and batch.returns.shape == (2, 2)
assert batch.asset_ids == ("A", "B")

forecast = AssetForecastResult(
    expected_returns=np.array([[0.02, 0.01]]),
    timestamps=("2024-03",),
    asset_ids=batch.asset_ids,
)
predictions = predictions_frame_from_asset_forecast(forecast)
assert predictions.columns == ("timestamp", "asset", "prediction_value")
assert len(predictions.rows) == 2
print(f"panel shape: {batch.returns.shape}; prediction rows: {len(predictions.rows)}")
