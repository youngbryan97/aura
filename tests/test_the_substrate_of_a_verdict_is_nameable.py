"""A boundary that is disclaimed has to be crossable.

Every receipt the influence trials write says the same thing: "a verdict
here is not a verdict at the 27B. The substrate is a 1.5B and the result is
a result at that substrate." That warning was correct and complete, and the
substrate was a module constant, so nothing could be pointed at the 27B to
get the verdict the warning kept deferring.

A hard-coded substrate that every artifact then apologises for is a
measurement nobody can extend. It is nameable now, and the receipt reports
the model that actually produced it rather than a constant.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_influence_trials_on_the_substrate.py"
HARNESS = ROOT / "tools" / "matched" / "run_matched_substrate.py"


def test_the_loader_takes_a_substrate():
    import sys

    sys.path.insert(0, str(ROOT))
    try:
        from tools.matched.run_matched_substrate import _load
    finally:
        sys.path.pop(0)

    signature = inspect.signature(_load)
    assert "substrate" in signature.parameters
    # Defaulted, so every existing caller keeps the model it had.
    assert signature.parameters["substrate"].default is None


def test_the_default_is_still_the_substrate_the_harness_was_built_on():
    import sys

    sys.path.insert(0, str(ROOT))
    try:
        from tools.matched.run_matched_substrate import SUBSTRATE
    finally:
        sys.path.pop(0)

    assert SUBSTRATE == "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def test_the_runner_exposes_it_on_the_command_line():
    source = RUNNER.read_text(encoding="utf-8")
    assert '"--substrate"' in source


def test_the_receipt_reports_the_model_that_produced_it():
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    # The receipt's substrate field must read the parameter, not the constant.
    assert '"substrate": substrate,' in source
    assert '"substrate": SUBSTRATE,' not in source
    del tree


def test_the_disclaimer_names_the_model_rather_than_one_model():
    """It said "not a verdict at the 27B" even when run ON the 27B."""

    source = RUNNER.read_text(encoding="utf-8")
    assert "a verdict at any substrate but {substrate}" in source


def test_a_folded_pack_can_be_the_substrate_too():
    """The 27B-class pack does not load through mlx_lm on its own."""

    source = HARNESS.read_text(encoding="utf-8")
    assert "is_prism_hadamard_pack" in source
    assert "load_prism_hadamard_pack" in source
