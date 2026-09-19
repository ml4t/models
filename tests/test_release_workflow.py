"""Behavioral contracts for commit-bound release publication."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from ml4t.models import __version__
from scripts.ci import readme_smoke, release, verify_docs_deployment

ROOT = Path(__file__).parents[1]
COMMIT = "a" * 40


def _workflow(name: str) -> dict:
    value = yaml.load(
        (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert isinstance(value, dict)
    return value


def test_preflight_requires_an_unpublished_current_main_candidate() -> None:
    assert (
        release.preflight_failures(
            version="1.2.3",
            source_version="1.2.3",
            candidate_commit=COMMIT,
            workflow_commit=COMMIT,
            checkout_commit=COMMIT,
            main_commit=COMMIT,
            tag_exists=False,
            release_exists=False,
            pypi_exists=False,
        )
        == []
    )

    failures = release.preflight_failures(
        version="v1.2",
        source_version="1.2.2",
        candidate_commit="HEAD",
        workflow_commit="b" * 40,
        checkout_commit="c" * 40,
        main_commit="d" * 40,
        tag_exists=True,
        release_exists=True,
        pypi_exists=True,
    )

    assert len(failures) == 9
    assert "candidate_commit must be a full lowercase commit SHA" in failures
    assert "PyPI version already exists: v1.2" in failures


def test_pypi_publication_must_match_candidate_metadata_and_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata = {
        "author_email": "Stefan Jansen <stefan@applied-ai.com>",
        "classifiers": ["Programming Language :: Python :: 3"],
        "description": "Finance-specific models for asset pricing, prediction, and portfolio learning.",
        "keywords": ["finance"],
        "license": "MIT",
        "maintainer_email": "Stefan Jansen <pm@ml4trading.io>",
        "project_urls": {"Homepage": "https://www.ml4trading.io/"},
        "requires_python": ">=3.12",
    }
    manifest = {
        "name": "ml4t-models",
        "version": "1.2.3",
        "metadata": metadata,
        "artifacts": [
            {"filename": "package.whl", "sha256": "b" * 64, "size": 10},
            {"filename": "package.tar.gz", "sha256": "c" * 64, "size": 20},
        ],
    }
    (tmp_path / "candidate.json").write_text(json.dumps(manifest), encoding="utf-8")
    response = {
        "info": {
            "name": "ml4t-models",
            "version": "1.2.3",
            "author_email": metadata["author_email"],
            "classifiers": metadata["classifiers"],
            "summary": metadata["description"],
            "keywords": "finance",
            "license": "MIT",
            "license_expression": None,
            "maintainer_email": metadata["maintainer_email"],
            "project_urls": metadata["project_urls"],
            "requires_python": metadata["requires_python"],
        },
        "urls": [
            {
                "filename": artifact["filename"],
                "digests": {"sha256": artifact["sha256"]},
                "size": artifact["size"],
            }
            for artifact in manifest["artifacts"]
        ],
    }
    monkeypatch.setattr(release, "_package", lambda _name, _version: response)

    release.verify_publication(tmp_path)
    response["urls"][0]["digests"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="artifacts do not match"):
        release.verify_publication(tmp_path)


def test_published_install_verification_retries_index_propagation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = iter(
        (
            subprocess.CompletedProcess([], 1),
            subprocess.CompletedProcess([], 1),
            subprocess.CompletedProcess([], 0),
        )
    )
    commands: list[list[str]] = []
    sleeps: list[int] = []

    def run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[bytes]:
        assert check is False
        commands.append(command)
        return next(results)

    monkeypatch.setattr(release.subprocess, "run", run)
    monkeypatch.setattr(release.time, "sleep", sleeps.append)
    release.verify_install(
        "ml4t-models",
        "1.2.3",
        Path("scripts/ci/readme_smoke.py"),
        attempts=3,
        retry_seconds=7,
    )

    assert len(commands) == 3
    assert "ml4t-models==1.2.3" in commands[0]
    assert commands[0][-2:] == ["--expected-version", "1.2.3"]
    assert sleeps == [7, 7]


def test_readme_quick_start_is_an_executable_installed_package_contract() -> None:
    readme = ROOT / "README.md"
    source = readme_smoke.extract_quick_start(readme.read_text(encoding="utf-8"))
    assert "IPCAModel" in source
    readme_smoke.run(readme, __version__)


def test_deployed_docs_default_retry_window_is_four_minutes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []
    sleeps: list[float] = []

    def read_identity(_url: str, _commit: str, attempt: int) -> object:
        attempts.append(attempt)
        return {"commit": "stale"}

    monkeypatch.setattr(verify_docs_deployment, "_read_identity", read_identity)
    monkeypatch.setattr(verify_docs_deployment.time, "sleep", sleeps.append)

    with pytest.raises(RuntimeError, match="deployed documentation identity did not match"):
        verify_docs_deployment.verify(
            ("https://example.test/release.json",),
            {"commit": COMMIT},
        )

    assert attempts == list(range(24))
    assert sleeps == [10] * 23


def test_rendered_docs_expose_exact_release_identity(tmp_path: Path) -> None:
    expected = {"commit": COMMIT, "library": "models", "version": __version__}
    environment = {
        **os.environ,
        "ML4T_DOCS_COMMIT": expected["commit"],
        "ML4T_DOCS_VERSION": expected["version"],
    }
    subprocess.run(
        [
            "uv",
            "run",
            "--extra",
            "docs",
            "mkdocs",
            "build",
            "--strict",
            "--site-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )

    verify_docs_deployment.verify_site(tmp_path, expected)
    index = tmp_path / "index.html"
    html = index.read_text(encoding="utf-8").replace(
        f'<meta name="ml4t-version" content="{__version__}">',
        '<meta name="ml4t-version" content="wrong">',
    )
    index.write_text(html, encoding="utf-8")
    with pytest.raises(RuntimeError, match="ml4t-version is 'wrong'"):
        verify_docs_deployment.verify_site(tmp_path, expected)


def test_release_workflow_reuses_one_commit_bound_candidate() -> None:
    ci = _workflow("ci.yml")
    release_workflow = _workflow("release.yml")
    jobs = release_workflow["jobs"]

    assert "workflow_call" in ci["on"]
    assert ci["on"]["workflow_call"]["inputs"]["release_version"]["type"] == "string"
    assert set(release_workflow["on"]) == {"workflow_dispatch"}
    assert set(release_workflow["on"]["workflow_dispatch"]["inputs"]) == {
        "version",
        "candidate_commit",
    }
    assert jobs["qualification"]["uses"] == "./.github/workflows/ci.yml"
    assert jobs["qualification"]["with"]["release_version"] == (
        "${{ needs.validate.outputs.version }}"
    )
    assert jobs["select-candidate"]["needs"] == [
        "validate",
        "ecosystem-qualification",
        "qualification",
    ]
    assert jobs["docs"]["needs"] == ["validate", "select-candidate"]
    assert jobs["publish"]["needs"] == [
        "validate",
        "ecosystem-qualification",
        "select-candidate",
        "docs",
    ]
    assert jobs["github-release"]["needs"] == ["validate", "publish"]
    assert jobs["verify-release"]["needs"] == ["validate", "github-release"]

    workflow_text = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "uv build" not in workflow_text
    assert '--target "$CANDIDATE_COMMIT"' in workflow_text
    assert "release.py verify candidate" in workflow_text
    assert "release.py smoke-test" in workflow_text
    assert "candidate.py verify released" in workflow_text
    assert "verify_docs_deployment.py" in workflow_text
    assert "needs.publish.result != 'skipped'" in workflow_text
    assert "Rerun only failed jobs" in workflow_text
    assert "Do not rebuild or reuse this version" in workflow_text


def test_only_release_workflow_can_deploy_documentation() -> None:
    for name in ("ci.yml", "docs.yml", "ecosystem.yml"):
        assert "push-to-another-repository" not in (
            ROOT / ".github" / "workflows" / name
        ).read_text(encoding="utf-8")


def test_standalone_docs_workflow_is_read_only_and_verifies_strict_build() -> None:
    workflow = _workflow("docs.yml")
    build = workflow["jobs"]["build"]
    commands = "\n".join(step.get("run", "") for step in build["steps"])

    assert workflow["permissions"] == {"contents": "read"}
    assert "permissions" not in build
    assert "uv run mkdocs build --strict" in commands
    assert "ML4T_DOCS_SITE=site" in commands
    assert "scripts/ci/verify_docs_deployment.py" in commands


def test_ci_uses_locked_dependencies_and_requires_cuda_for_releases() -> None:
    workflows = [
        (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        for name in ("ci.yml", "release.yml")
    ]
    ci = workflows[0]
    assert 'version: "latest"' not in ci
    assert 'UV_NO_SOURCES: "1"' in ci
    for workflow in workflows:
        sync_commands = [line for line in workflow.splitlines() if "uv sync" in line]
        assert sync_commands
        assert all("--locked" in command for command in sync_commands)
        assert all("--frozen" not in command for command in sync_commands)
    assert "inputs.release_version != ''" in ci
    assert 'test "$CUDA_RESULT" = success' in ci


def test_uv_locked_sync_supports_the_no_sources_policy() -> None:
    environment = {**os.environ, "UV_NO_SOURCES": "1"}
    result = subprocess.run(
        ["uv", "sync", "--locked", "--dev", "--dry-run"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
