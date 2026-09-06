"""A caller can be wrong about a tool before it runs it, and after.

CrewAI requires structured schemas both ways and normalises them, so a
provider quirk cannot leak upward and a consumer knows the shape of a result
without running it. Aura declared the argument side — every one of its
eighty-two tools says what it takes — and left the result to whatever the
caller happened to get. Skills carry `output`, which is prose: "Current date
and time string" tells a reader what to expect and tells a caller nothing it
can check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def engine():
    from core.capability_engine import CapabilityEngine

    return CapabilityEngine()


def test_every_tool_says_what_it_takes(engine):
    """The half that was already true, held."""

    silent = sorted(
        one.name for one in engine.skills.values() if not one.declares_its_arguments
    )
    assert not silent, silent


def test_the_whole_contract_reads_from_one_place(engine):
    one = next(iter(engine.skills.values()))
    promised = one.what_it_promises()
    assert set(promised) >= {
        "name",
        "contract_version",
        "arguments",
        "declares_its_arguments",
        "result",
        "declares_its_result",
        "effect_scope",
        "authority_class",
    }


def test_a_skill_that_declares_a_result_schema_carries_it_through(engine):
    """Clock declares one, so the path from the class to the metadata is live."""

    clock = engine.skills.get("clock")
    assert clock is not None, sorted(engine.skills)
    assert clock.declares_its_result, "the declaration did not reach the metadata"
    schema = clock.result_schema_def
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"ok", "time", "readable", "summary"}


def test_an_undeclared_result_is_an_open_object_and_says_so(engine):
    quiet = [one for one in engine.skills.values() if not one.declares_its_result]
    assert quiet, "nothing is undeclared, so this test measures nothing"
    schema = quiet[0].result_schema_def
    assert schema == {"additionalProperties": True, "properties": {}, "type": "object"}


def test_the_count_of_undeclared_results_only_falls():
    """The ratchet. Declaring is work; what matters is that it goes one way."""

    baseline = json.loads(
        (ROOT / "config" / "tool_contract_baseline.json").read_text()
    )
    import subprocess
    import sys

    from tools.lint_tool_contracts import THE_COUNTS, measure

    now = measure()
    for key in THE_COUNTS:
        assert now[key] <= baseline[key], (
            f"{key} rose from {baseline[key]} to {now[key]}: a tool stopped "
            "saying something it used to say"
        )


def test_the_ratchet_would_notice_a_regression():
    """A gate that cannot fail is not a gate."""

    from tools.lint_tool_contracts import THE_COUNTS

    assert "without_a_result" in THE_COUNTS
    assert "tools" not in THE_COUNTS, (
        "adding a tool would count as a regression, which stops the work the "
        "ratchet exists to encourage"
    )
