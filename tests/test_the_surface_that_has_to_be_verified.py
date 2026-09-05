"""The complexity criticism that is worth acting on, made checkable.

Size is not it. A sufficiently rich mind probably requires enormous
heterogeneity, and a neuron is not fifteen thousand lines long because minds
are simple. What makes software complexity dangerous is verification surface:
with n components whose invariants hold locally, checking stays about O(n);
with arbitrary cross-links, O(n²) relationships become possible; with
higher-order interactions, coalitions of 2^n become relevant. That is why the
coalition-credit problem exists at all.

Biological complexity is distributed. The question this measures is whether
Aura's is — and the existing ratchets cannot answer it, because a file of
twenty-nine thousand lines that nothing imports and one that everything
routes through are the same number to a line count.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.lint_convergence_surface import BASELINE, HOW_MANY_KEPT, measure


def test_a_utility_is_not_a_convergence_surface():
    """Fan-in alone is a utility. Everything imports the logger."""

    now = measure()
    by_name = {one["module"]: one for one in now["worst"]}
    errors = by_name.get("core.runtime.errors")
    engine = by_name.get("core.brain.cognitive_engine")
    if errors is None or engine is None:
        pytest.skip("the tree no longer has both of these")
    assert errors["paths"] > engine["paths"], "the error sink should mediate more"
    assert engine["surface"] > errors["surface"], (
        "a module reached by fifty and reaching sixty is the dangerous shape; "
        "one reached by a thousand and reaching eleven is a utility"
    )


def test_both_sides_have_to_be_large():
    """The number is the harmonic mean, so neither side can carry it alone."""

    for one in measure()["worst"]:
        if not one["fan_in"] or not one["fan_out"]:
            assert one["surface"] == 0.0
        else:
            assert one["surface"] <= 2 * min(one["fan_in"], one["fan_out"])


def test_the_baseline_exists_and_holds():
    assert BASELINE.exists(), "the ratchet has nothing to hold"
    was = json.loads(BASELINE.read_text(encoding="utf-8"))
    now = measure()
    assert float(now["worst_total"]) <= float(was["worst_total"]), (
        "a module many things reach into that also reaches many things is "
        "where an invariant stops being local, and the total may not grow"
    )


def test_the_baseline_pins_the_worst_of_them():
    was = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert len(was["worst"]) == HOW_MANY_KEPT
    assert was["worst"][0]["surface"] >= was["worst"][-1]["surface"]


def test_it_reads_the_organism_and_not_the_harness():
    """Tests and tools are not the organism, and a gate that counted them
    would measure how much measuring there is."""

    modules = {one["module"] for one in measure()["worst"]}
    assert not any(one.startswith(("tests.", "tools.")) for one in modules)
