from __future__ import annotations

import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from scripts.ci import candidate, check_performance


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


def test_candidate_contains_typing_and_documentation_contracts(candidate_dir: Path) -> None:
    wheel, sdist = candidate._distribution_files(candidate_dir)
    with zipfile.ZipFile(wheel) as archive:
        wheel_members = set(archive.namelist())
    assert "ml4t/models/_version.py" in wheel_members
    assert "ml4t/models/py.typed" in wheel_members

    with tarfile.open(sdist, "r:gz") as archive:
        sdist_members = {member.name.partition("/")[2] for member in archive.getmembers()}
    assert "SECURITY.md" in sdist_members
    assert "mkdocs.yml" in sdist_members
    assert "docs/getting-started/installation.md" in sdist_members


def _performance_inputs(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    records: list[Path] = []
    arrays: list[Path] = []
    for repetition in range(3):
        record = tmp_path / f"performance-{repetition}.json"
        record.write_text(
            json.dumps(
                {
                    "device": "cuda",
                    "environment": {"gpu": "test"},
                    "profile": "canonical",
                    "results": {"model": {"fit_seconds": 1.0}},
                }
            ),
            encoding="utf-8",
        )
        array = tmp_path / f"performance-{repetition}.npz"
        np.savez_compressed(array, model=np.array([1.0]))
        records.append(record)
        arrays.append(array)
    return records, arrays


def test_performance_checker_accepts_consistent_repetitions(tmp_path: Path) -> None:
    records, arrays = _performance_inputs(tmp_path)

    check_performance.check(records, arrays)


def test_performance_checker_rejects_timing_and_replay_drift(tmp_path: Path) -> None:
    records, arrays = _performance_inputs(tmp_path)
    value = json.loads(records[2].read_text(encoding="utf-8"))
    value["results"]["model"]["fit_seconds"] = 3.0
    records[2].write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(AssertionError, match="timing range"):
        check_performance.check(records, arrays)

    value["results"]["model"]["fit_seconds"] = 1.0
    records[2].write_text(json.dumps(value), encoding="utf-8")
    np.savez_compressed(arrays[2], model=np.array([2.0]))
    with pytest.raises(AssertionError, match="replay exceeded"):
        check_performance.check(records, arrays)
