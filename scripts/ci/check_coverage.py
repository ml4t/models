"""Enforce separate line and branch coverage thresholds."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

OVERALL_THRESHOLDS = (90.0, 85.0)
CRITICAL_THRESHOLDS = (95.0, 90.0)
CRITICAL_MODULES = (
    "src/ml4t/models/_internal/latent_factor_utils.py",
    "src/ml4t/models/_internal/persistence.py",
    "src/ml4t/models/integration/backtest.py",
    "src/ml4t/models/integration/data.py",
    "src/ml4t/models/integration/surfaces.py",
    "src/ml4t/models/portfolio/runtime.py",
    "src/ml4t/models/stochastic_discount_factor/mapper.py",
    "src/ml4t/models/stochastic_discount_factor/model.py",
    "src/ml4t/models/types.py",
)


def _percent(covered: int, total: int) -> float:
    return 100.0 if total == 0 else 100.0 * covered / total


def _measure(summary: dict[str, Any]) -> tuple[float, float]:
    return (
        _percent(int(summary["covered_lines"]), int(summary["num_statements"])),
        _percent(int(summary["covered_branches"]), int(summary["num_branches"])),
    )


def _failure(
    name: str,
    actual: tuple[float, float],
    required: tuple[float, float],
) -> str | None:
    failed = []
    for metric, value, threshold in zip(("lines", "branches"), actual, required, strict=True):
        if value < threshold:
            failed.append(f"{metric} {value:.2f}% < {threshold:.2f}%")
    return f"{name}: {', '.join(failed)}" if failed else None


def evaluate(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    overall_failure = _failure("overall", _measure(report["totals"]), OVERALL_THRESHOLDS)
    if overall_failure is not None:
        failures.append(overall_failure)

    files = report["files"]
    for module in CRITICAL_MODULES:
        if module not in files:
            failures.append(f"{module}: missing from coverage report")
            continue
        module_failure = _failure(module, _measure(files[module]["summary"]), CRITICAL_THRESHOLDS)
        if module_failure is not None:
            failures.append(module_failure)
    return failures


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: check_coverage.py COVERAGE_JSON", file=sys.stderr)
        return 2

    report_path = Path(arguments[0])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures = evaluate(report)
    if failures:
        print("Coverage requirements failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    line_coverage, branch_coverage = _measure(report["totals"])
    print(f"Coverage passed: lines {line_coverage:.2f}%, branches {branch_coverage:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
