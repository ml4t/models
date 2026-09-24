from __future__ import annotations

import re
import runpy
from pathlib import Path

import pytest

EXAMPLES = (
    "latent_factor_pipeline.py",
    "latent_factor_variants.py",
    "stochastic_discount_factor.py",
    "direct_asset_prediction.py",
    "portfolio_learning.py",
    "portfolio_neural.py",
    "integration_handoff.py",
)


@pytest.mark.parametrize("example", EXAMPLES)
def test_examples_execute(example: str) -> None:
    runpy.run_path(str(Path(__file__).parents[1] / "examples" / example))


def test_documented_quickstart_executes(capsys: pytest.CaptureFixture[str]) -> None:
    root = Path(__file__).parents[1]
    content = (root / "docs/getting-started/quickstart.md").read_text(encoding="utf-8")
    source = re.search(r"```python\r?\n(.*?)```", content, re.DOTALL)
    assert source is not None
    exec(compile(source.group(1), "quickstart.md", "exec"), {"__name__": "__main__"})
    assert capsys.readouterr().out == "forecast shape: (2, 6)\n"
