from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.ci import candidate


@pytest.fixture(scope="module")
def candidate_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("candidate")
    subprocess.run(["uv", "build", "--out-dir", str(directory / "dist")], check=True)
    candidate.create(directory, commit_sha="a" * 40, git_tree="b" * 40)
    return directory


def test_candidate_manifest_binds_artifacts_to_source_identity(candidate_dir: Path) -> None:
    candidate.verify(
        candidate_dir,
        expected_commit="a" * 40,
        expected_tree="b" * 40,
        expected_tag="v0.1.0b0",
    )

    manifest = json.loads((candidate_dir / "candidate.json").read_text(encoding="utf-8"))

    assert manifest["name"] == "ml4t-models"
    assert manifest["version"] == "0.1.0b0"
    assert {record["filename"] for record in manifest["artifacts"]} == {
        "ml4t_models-0.1.0b0-py3-none-any.whl",
        "ml4t_models-0.1.0b0.tar.gz",
    }


def test_candidate_verification_rejects_wrong_tag(candidate_dir: Path) -> None:
    with pytest.raises(ValueError, match="does not match candidate version"):
        candidate.verify(candidate_dir, expected_tag="v0.1.0")


def test_candidate_verification_rejects_modified_artifact(candidate_dir: Path) -> None:
    wheel = next((candidate_dir / "dist").glob("*.whl"))
    original = wheel.read_bytes()
    try:
        wheel.write_bytes(original + b"modified")
        with pytest.raises(ValueError, match="integrity check failed"):
            candidate.verify(candidate_dir)
    finally:
        wheel.write_bytes(original)


def test_rebuilt_wheel_matches_candidate(candidate_dir: Path, tmp_path: Path) -> None:
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(tmp_path)], check=True)

    candidate.compare_wheel(candidate_dir, tmp_path)
