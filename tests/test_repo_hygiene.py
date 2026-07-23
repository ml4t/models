from __future__ import annotations

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
