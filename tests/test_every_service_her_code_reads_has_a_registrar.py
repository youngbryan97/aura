"""A service her code reads by name has something that registers it.

A container read with no registrar returns None for the life of the process,
and the code behind it returns at its first line. That is how her artificial
life, her repair engine and her homeostatic drives went unrun from April to
November 2026 behind a boot audit that read 100%: every tick asked for them and
nothing had registered them (core/providers/self_maintenance_provider.py).

Two ratchets. The kernel's phases run on every tick, so each phase read with no
registrar is named here with the reason it is allowed. Everywhere else, the
names recorded on 5 November are the baseline, and the baseline only shrinks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "config" / "unregistered_service_reads_baseline.json"

#: Phase reads with no registrar, and why each is allowed to stay.
PHASE_EXCUSED: dict[str, str] = {
    "unity_state": "a fallback after state.cognition.unity_state, which the unity runtime writes into her turn",
    "unity_fragmentation_report": "a fallback; the unity runtime writes its report into her turn, not the container",
    "unity_repair_plan": "a fallback; the unity runtime writes its plan into response_modifiers",
    "coherence_report": "a fallback after the coherence report the same phase writes into response_modifiers",
    "user_model": "reply path: wiring it changes what reaches her words, which waits until the paper's runs are done",
    "user_model_engine": "the third fallback after bryan_model_engine and bryan_model, which the boot registers",
}


@pytest.fixture(scope="module")
def census():
    import importlib.util

    spec = importlib.util.spec_from_file_location("census_service_reads", ROOT / "tools" / "census_service_reads.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unregistered(found: dict) -> set[str]:
    return {name for name, value in found.items() if not value["registered_by"]}


def test_every_phase_read_has_a_registrar_or_a_reason(census):
    missing = _unregistered(census.census(phases_only=True))
    unexplained = sorted(missing - set(PHASE_EXCUSED))
    assert not unexplained, f"a kernel phase reads these and nothing registers them: {unexplained}"


def test_no_phase_excuse_outlives_its_reason(census):
    missing = _unregistered(census.census(phases_only=True))
    stale = sorted(set(PHASE_EXCUSED) - missing)
    assert not stale, f"these now have a registrar or no reader; remove their excuse: {stale}"


def test_no_new_read_without_a_registrar(census):
    recorded = set(json.loads(BASELINE.read_text())["names"])
    new = sorted(_unregistered(census.census()) - recorded)
    assert not new, f"read from the container with no registrar anywhere: {new}"


def test_the_baseline_only_shrinks(census):
    recorded = set(json.loads(BASELINE.read_text())["names"])
    fixed = sorted(recorded - _unregistered(census.census()))
    assert not fixed, f"these are registered or no longer read; remove them from {BASELINE.name}: {fixed}"


def test_the_census_sees_the_registrations_that_matter(census):
    """A census that missed these would report the organs as dead again."""
    found = census.census()
    for name in ("alife_dynamics", "autopoiesis", "homeostasis_engine", "dead_letter_queue", "interiority"):
        assert found.get(name, {}).get("registered_by"), name
