"""The interventional cut, on systems whose answer is known.

A system with no coupling between its domains has nothing a cut can remove, so
no cut is decided. A star, whose hub drives every domain, loses that drive at
every cut, so every cut is decided. Four of the 511 bipartitions and twelve
anchors keep this fast; tools/validate_interventional_cut.py runs all of them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from core.subject.nulls import architecture
from core.subject.state import DOMAINS
from core.subject.v25_runtime import bipartitions

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("validate_interventional_cut", ROOT / "tools/validate_interventional_cut.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_interventional_cut"] = module
    spec.loader.exec_module(module)
    return module


CUTS = list(bipartitions(DOMAINS))[:4]


def _sweep(name: str) -> dict:
    tool = _tool()
    system = architecture(name, seed=7)
    step, start = tool.toy_step(system)
    return tool.sweep(step, start, sum(system.widths.values()) + system.hub_width, anchors=12, lag=2, cuts=CUTS)


def test_a_system_with_no_coupling_has_no_cut_worth_deciding() -> None:
    assert _sweep("independent")["decided"] == 0


def test_a_star_loses_its_hub_at_every_cut() -> None:
    result = _sweep("star")
    assert result["decided"] == result["cuts"] == len(CUTS)
