from __future__ import annotations

import ast
import re
import runpy
from pathlib import Path

import pytest

from ml4t.models import StochasticDiscountFactorConfig

EXAMPLES = (
    "latent_factor_pipeline.py",
    "stochastic_discount_factor.py",
    "direct_asset_prediction.py",
    "portfolio_learning.py",
)

DOCUMENTS = (
    "README.md",
    "docs/getting-started/quickstart.md",
    "docs/user-guide/stochastic-discount-factor.md",
)


@pytest.mark.parametrize("example", EXAMPLES)
def test_examples_execute(example: str) -> None:
    runpy.run_path(str(Path(__file__).parents[1] / "examples" / example))


@pytest.mark.parametrize(
    "document",
    DOCUMENTS,
)
def test_documented_sdf_checkpoint_configs_are_valid(document: str) -> None:
    root = Path(__file__).parents[1]
    content = (root / document).read_text(encoding="utf-8")
    code_blocks = re.findall(r"```python\r?\n(.*?)```", content, re.DOTALL)
    checked = 0
    for code_block in code_blocks:
        tree = ast.parse(code_block)
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if (
                not isinstance(call.func, ast.Name)
                or call.func.id != "StochasticDiscountFactorConfig"
            ):
                continue
            keywords = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in call.keywords
                if keyword.arg
                in {"checkpoint_epochs", "default_checkpoint", "n_epochs_unc", "n_epochs_cond"}
            }
            if not keywords:
                continue
            StochasticDiscountFactorConfig(**keywords)
            checked += 1
    assert checked > 0
