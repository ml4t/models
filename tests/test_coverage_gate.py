from __future__ import annotations

from typing import Any

from scripts.ci import check_coverage


def _summary(
    covered_lines: int = 95,
    num_statements: int = 100,
    covered_branches: int = 90,
    num_branches: int = 100,
) -> dict[str, int]:
    return {
        "covered_lines": covered_lines,
        "num_statements": num_statements,
        "covered_branches": covered_branches,
        "num_branches": num_branches,
    }


def _report() -> dict[str, Any]:
    return {
        "totals": _summary(),
        "files": {module: {"summary": _summary()} for module in check_coverage.CRITICAL_MODULES},
    }


def test_coverage_gate_accepts_all_thresholds() -> None:
    assert check_coverage.evaluate(_report()) == []


def test_coverage_gate_checks_overall_lines_and_branches_separately() -> None:
    report = _report()
    report["totals"] = _summary(covered_lines=89, covered_branches=84)

    assert check_coverage.evaluate(report) == [
        "overall: lines 89.00% < 90.00%, branches 84.00% < 85.00%"
    ]


def test_coverage_gate_checks_each_critical_module() -> None:
    report = _report()
    module = check_coverage.CRITICAL_MODULES[0]
    report["files"][module]["summary"] = _summary(covered_lines=94, covered_branches=89)

    assert check_coverage.evaluate(report) == [
        f"{module}: lines 94.00% < 95.00%, branches 89.00% < 90.00%"
    ]


def test_coverage_gate_rejects_missing_critical_module() -> None:
    report = _report()
    module = check_coverage.CRITICAL_MODULES[0]
    del report["files"][module]

    assert check_coverage.evaluate(report) == [f"{module}: missing from coverage report"]
