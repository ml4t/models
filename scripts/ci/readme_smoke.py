"""Execute the first README quick-start example against an installed distribution."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def extract_quick_start(markdown: str) -> str:
    _, marker, remainder = markdown.partition("## Quick Start")
    if not marker:
        raise ValueError("README has no Quick Start section")
    section = remainder.partition("\n## ")[0]
    blocks = re.findall(r"```python\r?\n(.*?)```", section, re.DOTALL)
    if not blocks:
        raise ValueError("README Quick Start has no Python example")
    return blocks[0]


def run(readme: Path, expected_version: str) -> None:
    import numpy as np

    import ml4t.models as models

    if models.__version__ != expected_version:
        raise RuntimeError(
            f"installed version {models.__version__!r} does not match {expected_version!r}"
        )
    np.random.seed(0)
    source = extract_quick_start(readme.read_text(encoding="utf-8"))
    exec(compile(source, "README.md", "exec"), {"__name__": "__main__"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--readme", type=Path, default=Path(__file__).parents[2] / "README.md")
    args = parser.parse_args()
    run(args.readme, args.expected_version)


if __name__ == "__main__":
    main()
