from __future__ import annotations

import re
import subprocess
from pathlib import Path

import ml4t.models as models


def test_public_repo_has_no_tracked_internal_or_generated_artifacts() -> None:
    assert models.__version__
    root = Path(__file__).parents[1]
    output = subprocess.check_output(
        ["git", "ls-files"],
        cwd=root,
        text=True,
    )
    tracked_files = output.splitlines()
    forbidden_fragments = (
        "/archive/",
        "/_archive/",
        ".agents/",
        ".workspace/",
        "__pycache__/",
        ".coverage",
        ".pytest_cache/",
        ".ruff_cache/",
        "dist/",
        "site/",
    )

    offenders = [
        path
        for path in tracked_files
        if any(fragment in f"/{path}" for fragment in forbidden_fragments)
    ]

    assert offenders == []


def test_readme_ecosystem_image_is_tracked() -> None:
    root = Path(__file__).parents[1]
    assert (root / "docs/images/ml4t_ecosystem_workflow_color.png").is_file()


def test_workflow_actions_are_pinned_to_commits() -> None:
    root = Path(__file__).parents[1]
    workflows = tuple((root / ".github/workflows").glob("*.yml"))
    action_references = [
        line.strip()
        for workflow in workflows
        for line in workflow.read_text(encoding="utf-8").splitlines()
        if "uses:" in line
    ]

    assert action_references
    assert all(re.search(r"@[0-9a-f]{40}(?:\s|$)", line) for line in action_references)


def test_ci_qualifies_one_candidate_across_required_platforms() -> None:
    root = Path(__file__).parents[1]
    workflow = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "os: [ubuntu-latest, macos-latest, windows-latest]" in workflow
    assert 'python-version: ["3.12", "3.13", "3.14"]' in workflow
    assert 'python-version: "3.15"' in workflow
    assert "allow-prereleases: true" in workflow
    assert "name: release-candidate" in workflow
    assert "scripts/ci/test_wheel.py candidate" in workflow
    assert "scripts/ci/check_coverage.py coverage.json" in workflow
    assert "needs: [lint, typecheck, coverage, docs]" in workflow


def test_release_promotes_qualified_candidate_without_rebuilding() -> None:
    root = Path(__file__).parents[1]
    workflow = (root / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "uv build" not in workflow
    assert "--workflow ci.yml" in workflow
    assert '--expected-tag "${{ github.ref_name }}"' in workflow
    assert "packages-dir: candidate/dist/" in workflow
