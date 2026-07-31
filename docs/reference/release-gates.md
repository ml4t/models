# Release Gates

Stable publication promotes the exact wheel and source distribution qualified by the successful
main-branch CI run. The release workflow does not rebuild either artifact.

## Local Checks

Run these checks from the repository root:

```bash
uv run ruff check src/ tests/ examples/ scripts/
uv run ruff format --check src/ tests/ examples/ scripts/
uv run ty check
uv run pytest tests/ -q --cov-report=json:coverage.json
uv run python scripts/ci/check_coverage.py coverage.json
uv run mkdocs build --strict
uv build
uv run twine check dist/*
actionlint
```

The local hardware smoke uses CPU while exercising the same fit, persistence, recovery, and
replay paths as the CUDA and MPS jobs:

```bash
uv run python scripts/ci/hardware_qualification.py --device cpu
```

## Blocking CI Checks

The stable candidate requires all of these results:

- Ruff, formatting, ty, documentation, and coverage thresholds
- installed-wheel tests on Linux, macOS, and Windows for Python 3.12, 3.13, and 3.14
- installed core-wheel tests on the current Python 3.15 prerelease for all three operating systems
- current and supported-minimum dependency vulnerability scans
- reproducible wheel reconstruction on every stable operating-system and Python cell
- RTX 3090 CUDA fit, replay, checkpoint, persistence, CPU recovery, and CUDA recovery checks
- three fresh-process Chapter 14 performance runs on the CUDA reference host

MPS runs the public neural workflow matrix on an Apple Silicon hosted runner. It remains a
non-blocking compatibility result.

## Candidate Identity

CI records the source commit, Git tree, package name, package version, artifact names, sizes, and
SHA-256 digests in `candidate.json`. Each downstream job verifies that manifest before testing the
candidate.

A release tag must point to the same commit and match the candidate version exactly. The release
workflow selects the successful main-branch CI run for that commit, deploys revision-identified
documentation, verifies the deployed markers, and then publishes the retained distributions.
