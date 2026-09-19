"""Validate and verify commit-bound ml4t-models releases."""

from __future__ import annotations

import argparse
import json
import re
import runpy
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

PYPI_URL = "https://pypi.org/pypi/{name}/{version}/json"
USER_AGENT = "ml4t-models-release-verifier/1.0"
VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:(?:a|b|rc)[0-9]+)?$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _package(name: str, version: str) -> dict[str, Any]:
    request = Request(
        PYPI_URL.format(name=name, version=version), headers={"User-Agent": USER_AGENT}
    )
    with urlopen(request, timeout=20) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("PyPI response must be a JSON object")
    return value


def version_exists(name: str, version: str) -> bool:
    try:
        _package(name, version)
    except HTTPError as error:
        if error.code == 404:
            return False
        raise
    return True


def preflight_failures(
    *,
    version: str,
    source_version: str,
    candidate_commit: str,
    workflow_commit: str,
    checkout_commit: str,
    main_commit: str,
    tag_exists: bool,
    release_exists: bool,
    pypi_exists: bool,
) -> list[str]:
    failures = []
    if VERSION_PATTERN.fullmatch(version) is None:
        failures.append(f"version must be a stable or prerelease PEP 440 version: {version!r}")
    if COMMIT_PATTERN.fullmatch(candidate_commit) is None:
        failures.append("candidate_commit must be a full lowercase commit SHA")
    for label, observed in (
        ("workflow", workflow_commit),
        ("checkout", checkout_commit),
        ("origin/main", main_commit),
    ):
        if candidate_commit != observed:
            failures.append(
                f"candidate commit differs from {label}: {candidate_commit} != {observed}"
            )
    if source_version != version:
        failures.append(
            f"source version differs from requested release: {source_version} != {version}"
        )
    if tag_exists:
        failures.append(f"tag already exists: v{version}")
    if release_exists:
        failures.append(f"GitHub release already exists: v{version}")
    if pypi_exists:
        failures.append(f"PyPI version already exists: {version}")
    return failures


def _public_metadata(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "author_email": info.get("author_email"),
        "classifiers": sorted(info.get("classifiers", [])),
        "description": info.get("summary"),
        "keywords": sorted(
            keyword.strip() for keyword in (info.get("keywords") or "").split(",") if keyword
        ),
        "license": info.get("license_expression") or info.get("license"),
        "maintainer_email": info.get("maintainer_email"),
        "project_urls": info.get("project_urls"),
        "requires_python": info.get("requires_python"),
    }


def verify_publication(candidate_dir: Path) -> None:
    manifest = json.loads((candidate_dir / "candidate.json").read_text(encoding="utf-8"))
    package = _package(manifest["name"], manifest["version"])
    info = package.get("info")
    if not isinstance(info, dict):
        raise ValueError("PyPI response has no project metadata")
    if (info.get("name"), info.get("version")) != (manifest["name"], manifest["version"]):
        raise ValueError("PyPI project identity does not match the candidate manifest")
    if _public_metadata(info) != manifest.get("metadata"):
        raise ValueError("PyPI public metadata does not match the candidate manifest")

    published = {
        item.get("filename"): (item.get("digests", {}).get("sha256"), item.get("size"))
        for item in package.get("urls", [])
        if isinstance(item, dict)
    }
    expected = {item["filename"]: (item["sha256"], item["size"]) for item in manifest["artifacts"]}
    if published != expected:
        raise ValueError("PyPI artifacts do not match the candidate manifest")


def verify_install(
    name: str,
    version: str,
    script: Path,
    *,
    attempts: int = 12,
    retry_seconds: int = 10,
) -> None:
    command = [
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--refresh-package",
        name,
        "--with",
        f"{name}=={version}",
        "python",
        "-I",
        str(script),
        "--expected-version",
        version,
    ]
    for attempt in range(attempts):
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            return
        if attempt + 1 < attempts:
            time.sleep(retry_seconds)
    raise RuntimeError(f"failed to install and exercise {name} {version} from PyPI")


def _run_preflight(args: argparse.Namespace) -> None:
    source_version = runpy.run_path("src/ml4t/models/_version.py")["__version__"]
    tag = f"v{args.version}"
    failures = preflight_failures(
        version=args.version,
        source_version=source_version,
        candidate_commit=args.candidate_commit,
        workflow_commit=args.workflow_commit,
        checkout_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        main_commit=subprocess.check_output(["git", "rev-parse", "origin/main"], text=True).strip(),
        tag_exists=(
            subprocess.run(
                ["git", "show-ref", "--verify", "--quiet", f"refs/tags/{tag}"], check=False
            ).returncode
            == 0
        ),
        release_exists=(
            subprocess.run(
                ["gh", "release", "view", tag, "--repo", args.repository],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        ),
        pypi_exists=version_exists("ml4t-models", args.version),
    )
    if failures:
        raise SystemExit("\n".join(failures))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--version", required=True)
    preflight.add_argument("--candidate-commit", required=True)
    preflight.add_argument("--workflow-commit", required=True)
    preflight.add_argument("--repository", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("candidate_dir", type=Path)
    smoke = subparsers.add_parser("smoke-test")
    smoke.add_argument("name")
    smoke.add_argument("version")
    smoke.add_argument("script", type=Path)
    present = subparsers.add_parser("publication-exists")
    present.add_argument("name")
    present.add_argument("version")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "preflight":
        _run_preflight(args)
    elif args.command == "verify":
        verify_publication(args.candidate_dir)
    elif args.command == "smoke-test":
        verify_install(args.name, args.version, args.script)
    elif not version_exists(args.name, args.version):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
