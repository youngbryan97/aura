"""Four readers asked the arbiter to rank, and every answer said "Selected".

Only one caller can start the work. The others — executive authority, the
binding engine — arbitrate for their own purposes, and each scoring pass logged
"InitiativeArbiter: Selected '...'". The record then showed a goal chosen 190
times and started 24, which reads as an initiative that keeps being dropped and
is really four readers asking the same question.

LIVE 2026-08-29, asked what she could work out about herself from what she can
measure: "my 'thinking slowly and well' state coexists with a goal marked 93%
confident, yet the step count is still zero." She was reading this.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_the_arbiter_says_it_ranked() -> None:
    source = Path("core/agency/initiative_arbiter.py").read_text(encoding="utf-8")
    assert "ranked '" in source
    assert "f\"Selected '" not in source, "a ranking is not a decision"


def test_the_caller_that_can_start_the_work_says_it_acted() -> None:
    source = Path("core/initiative_synthesis.py").read_text(encoding="utf-8")
    assert "Synth: acting on" in source


def test_it_only_says_so_when_the_will_approved() -> None:
    tree = ast.parse(Path("core/initiative_synthesis.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Name) and test.id == "approved"):
            continue
        if "Synth: acting on" in ast.unparse(node):
            return
    raise AssertionError("the acted-on line is not guarded by approval")


def test_the_ranking_line_still_names_the_field_it_ranked() -> None:
    """A count nobody can compare against is not a measurement."""

    source = Path("core/agency/initiative_arbiter.py").read_text(encoding="utf-8")
    assert "first of {len(scored)}" in source
