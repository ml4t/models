from __future__ import annotations

import json
import subprocess
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request

import numpy as np
import pytest

from ml4t.models import __version__
from scripts.ci import (
    candidate,
    check_performance,
    hardware_qualification,
    performance_qualification,
    test_wheel,
    verify_docs_deployment,
)


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
        expected_tag=f"v{__version__}",
    )

    manifest = json.loads((candidate_dir / "candidate.json").read_text(encoding="utf-8"))

    assert manifest["name"] == "ml4t-models"
    assert manifest["version"] == __version__
    assert {record["filename"] for record in manifest["artifacts"]} == {
        f"ml4t_models-{__version__}-py3-none-any.whl",
        f"ml4t_models-{__version__}.tar.gz",
    }


def test_candidate_verification_rejects_wrong_tag(candidate_dir: Path) -> None:
    with pytest.raises(ValueError, match="does not match candidate version"):
        candidate.verify(candidate_dir, expected_tag="v0.1.0b0")


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

    candidate.compare_wheel(candidate_dir, tmp_path, require_byte_identical=True)


def test_rebuilt_wheel_allows_platform_zip_metadata_only(
    candidate_dir: Path,
    tmp_path: Path,
) -> None:
    wheel = next((candidate_dir / "dist").glob("*.whl"))
    rebuilt = tmp_path / wheel.name
    with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(rebuilt, "w") as destination:
        for name in source.namelist():
            destination.writestr(name, source.read(name))

    candidate.compare_wheel(candidate_dir, tmp_path)
    with pytest.raises(ValueError, match="not byte-identical"):
        candidate.compare_wheel(candidate_dir, tmp_path, require_byte_identical=True)


def test_candidate_contains_typing_and_documentation_contracts(candidate_dir: Path) -> None:
    wheel, sdist = candidate._distribution_files(candidate_dir)
    with zipfile.ZipFile(wheel) as archive:
        wheel_members = set(archive.namelist())
        metadata_name = next(name for name in wheel_members if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode()
    assert "ml4t/models/_version.py" in wheel_members
    assert "ml4t/models/py.typed" in wheel_members
    assert "Development Status :: 5 - Production/Stable" in metadata

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


def test_performance_checker_ignores_submeasurement_timing_noise(tmp_path: Path) -> None:
    records, arrays = _performance_inputs(tmp_path)
    for index, record in enumerate(records, start=1):
        value = json.loads(record.read_text(encoding="utf-8"))
        value["results"]["model"]["fit_seconds"] = index * 0.02
        record.write_text(json.dumps(value), encoding="utf-8")

    check_performance.check(records, arrays)


def test_performance_checker_ignores_subsecond_timing_range(tmp_path: Path) -> None:
    records, arrays = _performance_inputs(tmp_path)
    for fit_seconds, record in zip((0.120, 0.136, 0.526), records, strict=True):
        value = json.loads(record.read_text(encoding="utf-8"))
        value["results"]["model"]["fit_seconds"] = fit_seconds
        record.write_text(json.dumps(value), encoding="utf-8")

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


def test_performance_profiles_cover_representative_public_model_families() -> None:
    expected = {
        "pca",
        "rp_pca",
        "ipca",
        "cae",
        "sdf",
        "sae",
        "linear_portfolio",
        "lstm_portfolio",
        "deep_portfolio",
    }

    assert set(performance_qualification.CANONICAL) == expected
    assert set(performance_qualification.SMOKE) == expected
    assert {
        probe.rsplit("_", maxsplit=1)[-1]
        for probe in performance_qualification.SCALING
        if probe.startswith(("cae_", "sdf_", "sae_", "lstm_", "deep_"))
    } >= {"assets", "periods", "epochs", "ensemble", "sequence", "iterations"}


def test_performance_memory_measurement_is_importable_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(performance_qualification.sys, "platform", "win32")

    assert performance_qualification._peak_rss_kib() is None


def test_installed_typecheck_targets_candidate_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    python = tmp_path / "candidate-python"
    consumer = tmp_path / "typed_consumer.py"
    observed: list[str] = []

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        observed.extend(command)
        return subprocess.CompletedProcess(command, 1, stdout="invalid-assignment", stderr="")

    monkeypatch.setattr(test_wheel.subprocess, "run", run)

    test_wheel._check_consumer_types(python, consumer, tmp_path)

    assert observed == [
        str(python),
        "-m",
        "ty",
        "check",
        "--python",
        str(python),
        str(consumer),
    ]


def test_hardware_qualification_separates_replay_and_cpu_recovery_tolerances() -> None:
    expected = np.zeros(1)
    cross_backend = np.full(1, 2e-5)

    with pytest.raises(AssertionError, match="replay tolerance"):
        hardware_qualification._assert_close("model", expected, cross_backend, "cuda")

    hardware_qualification._assert_close(
        "model",
        expected,
        cross_backend,
        "cuda",
        cpu_recovery=True,
    )


def test_docs_verifier_identifies_client_and_varies_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"commit": "a" * 40, "version": "0.1.0"}
    responses = iter(({"commit": "old", "version": "0.1.0"}, expected))
    requests: list[Request] = []

    def open_url(request: Request, *, timeout: int) -> BytesIO:
        requests.append(request)
        assert timeout == 20
        return BytesIO(json.dumps(next(responses)).encode())

    monkeypatch.setattr(verify_docs_deployment, "urlopen", open_url)
    monkeypatch.setattr(verify_docs_deployment.time, "sleep", lambda _: None)

    verify_docs_deployment.verify(
        ("https://ml4trading.io/docs/models/release.json",),
        expected,
        attempts=2,
        retry_seconds=0,
    )

    assert [parse_qs(urlparse(request.full_url).query)["attempt"] for request in requests] == [
        ["0"],
        ["1"],
    ]
    assert all(
        request.get_header("User-agent") == verify_docs_deployment.USER_AGENT
        for request in requests
    )
