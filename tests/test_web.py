"""Checks that the web app (web/public) matches the Python version.

The browser has its own copy of the editor (editor.js) and of the AI's
explanations (ai.js). These tests run them with Node.js and compare them with
the Python originals, so the two can't quietly drift apart. They're skipped
if Node.js isn't installed.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from kelvin.auto.ai import EXPLANATIONS
from kelvin.editor import SLIDERS, apply_edits
from kelvin.ui import tokens

WEB = Path(__file__).parent.parent / "web" / "public"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")


def run_js(code: str, data=None):
    """Run a snippet of JavaScript (an ES module) with `data` available as DATA,
    and return whatever it passes to output()."""
    script = f"const DATA = {json.dumps(data)};\nconst output = (x) => console.log(JSON.stringify(x));\n{code}"
    result = subprocess.run(
        ["node", "--input-type=module", "-"], input=script, capture_output=True, text=True, cwd=WEB, check=True
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("seed", range(4))
def test_editor_matches_python(seed):
    torch.manual_seed(seed)
    photo = torch.rand(3, 16, 16)
    params = {s.name: round(torch.empty(1).uniform_(-1, 1).item(), 2) for s in SLIDERS}
    js = run_js(
        'import { applyEdits } from "./editor.js";'
        "output(Array.from(applyEdits(Float32Array.from(DATA.pixels), DATA.params)));",
        {"pixels": photo.permute(1, 2, 0).flatten().tolist(), "params": params},  # (3, H, W) -> r,g,b,r,g,b...
    )
    python = apply_edits(photo, params).permute(1, 2, 0).flatten()
    assert torch.allclose(torch.tensor(js), python, atol=1e-5)


def test_sliders_match_python():
    js = run_js('import { SLIDERS } from "./editor.js"; output(SLIDERS);')
    assert js == [{"name": s.name, "label": s.label, "help": s.help} for s in SLIDERS]


def test_explanations_match_python():
    js = run_js('import { EXPLANATIONS } from "./ai.js"; output(EXPLANATIONS);')
    assert js == {name: list(lines) for name, lines in EXPLANATIONS.items()}


def test_colours_match_design_tokens():
    css = (WEB / "style.css").read_text(encoding="utf-8")
    for name, value in vars(tokens).items():
        if name.isupper() and isinstance(value, str) and value.startswith("#"):
            css_name = "--" + name.lower().replace("_", "-")
            assert re.search(rf"{css_name}:\s*{value}\s*;", css, re.IGNORECASE), f"{css_name} should be {value}"
