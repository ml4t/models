"""Compare repeated Chapter 14 qualification records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

MAX_TIMING_RANGE = 2.0
CUDA_RTOL = 1e-5
CUDA_ATOL = 1e-5


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"performance record must be an object: {path}")
    return value


def check(records: list[Path], arrays: list[Path]) -> None:
    if len(records) != 3 or len(arrays) != 3:
        raise ValueError(
            f"performance qualification requires three records and arrays; "
            f"got {len(records)} and {len(arrays)}"
        )
    loaded = [_load(path) for path in records]
    identities = [
        (value.get("device"), value.get("environment"), value.get("profile")) for value in loaded
    ]
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("performance records do not share one environment identity")

    results = [value["results"] for value in loaded]
    names = set(results[0])
    if any(set(value) != names for value in results[1:]):
        raise ValueError("performance records do not share one model inventory")
    for name in sorted(names):
        fit_times = [float(value[name]["fit_seconds"]) for value in results]
        if min(fit_times) <= 0.0 or max(fit_times) / min(fit_times) > MAX_TIMING_RANGE:
            raise AssertionError(
                f"{name} fit timing range exceeded {MAX_TIMING_RANGE}x: {fit_times}"
            )

    loaded_arrays = [np.load(path, allow_pickle=False) for path in arrays]
    try:
        for name in sorted(names):
            reference = loaded_arrays[0][name]
            for repetition in loaded_arrays[1:]:
                if not np.allclose(
                    reference,
                    repetition[name],
                    rtol=CUDA_RTOL,
                    atol=CUDA_ATOL,
                    equal_nan=True,
                ):
                    difference = float(np.nanmax(np.abs(reference - repetition[name])))
                    raise AssertionError(
                        f"{name} replay exceeded CUDA tolerance: max_abs_difference={difference}"
                    )
    finally:
        for value in loaded_arrays:
            value.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--arrays", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    check(args.records, args.arrays)


if __name__ == "__main__":
    main()
