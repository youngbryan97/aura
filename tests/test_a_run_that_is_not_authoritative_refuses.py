"""A run says no rather than reporting numbers about a different organism.

A live cognitive loop that cannot be advanced by a count cannot be inside a
paired measurement. A required phase that raised was not run in either arm. A
reader that failed for most of the run made every column declaring it a
default. Each leaves the report describing something other than the organism
that lives here, and each was recorded in the report and otherwise ignored.

Recording it is not enough. A reader who takes a scorecard at face value has no
reason to open the notes, and an unauthoritative run is exactly the one
somebody quotes.
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _runner():
    spec = importlib.util.spec_from_file_location(
        "subject_core_runner", REPO / "tools" / "run_subject_core.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clean_evidence() -> dict:
    return {
        "notes": {"layers": {"unsteppable": {}, "failures": {}}},
        "organism": {"down": {}},
        "recording": {"misses": {}},
    }


def _runtime(**failures) -> types.SimpleNamespace:
    return types.SimpleNamespace(failures=dict(failures), failure_notes={})


def test_a_clean_run_is_authoritative() -> None:
    assert _runner()._authority_blockers(_runtime(), _clean_evidence()) == []


def test_a_loop_that_cannot_be_counted_blocks_the_run() -> None:
    module = _runner()
    evidence = _clean_evidence()
    evidence["notes"]["layers"]["unsteppable"] = {"NeuralMesh": "no entry point"}
    blockers = module._authority_blockers(_runtime(), evidence)
    assert any("advanced by a count" in line for line in blockers)


def test_a_layer_that_did_not_come_up_blocks_the_run() -> None:
    module = _runner()
    evidence = _clean_evidence()
    evidence["organism"]["down"] = {"unified_field": "boom"}
    blockers = module._authority_blockers(_runtime(), evidence)
    assert any("did not come up" in line for line in blockers)


def test_a_required_phase_that_raised_on_most_turns_blocks_the_run() -> None:
    module = _runner()
    blockers = module._authority_blockers(
        _runtime(AffectUpdatePhase=200), _clean_evidence(), turns=480
    )
    assert any("required phase raised" in line for line in blockers)


def test_a_required_phase_that_raised_once_does_not_block_the_run() -> None:
    """A phase that raised once in four hundred and eighty turns lost one turn's
    worth of that domain. A phase that raised on a fifth of them was not
    running, and the share between them is the one the battery already uses to
    decide that a reader was absent rather than unlucky."""
    module = _runner()
    assert module._authority_blockers(
        _runtime(AffectUpdatePhase=1), _clean_evidence(), turns=480
    ) == []


def test_a_phase_that_needs_a_cortex_does_not_block_the_run() -> None:
    """The run installs a deterministic mind on purpose.

    A phase that is absent by design is a different fact from one that raised,
    and a gate that cannot tell them apart refuses every run there is.
    """
    module = _runner()
    assert module._authority_blockers(
        _runtime(ResponseGenerationPhase=480), _clean_evidence(), turns=480
    ) == []


def test_a_reader_that_failed_for_most_of_the_run_blocks_it() -> None:
    module = _runner()
    from core.subject.battery import MISSING_SHARE

    evidence = _clean_evidence()
    evidence["recording"]["misses"] = {
        "organ:substrate.get_substrate_affect": {"share": MISSING_SHARE + 0.01}
    }
    assert any(
        "reader failed" in line
        for line in module._authority_blockers(_runtime(), evidence)
    )
    evidence["recording"]["misses"]["organ:substrate.get_substrate_affect"]["share"] = 0.01
    assert module._authority_blockers(_runtime(), evidence) == []


def test_every_required_phase_is_one_the_domains_are_read_from() -> None:
    """The list is a precondition, so it is named in advance rather than
    trimmed to whatever happened to pass."""
    module = _runner()
    from core.runtime.pipeline_blueprint import kernel_phase_attribute_order

    del kernel_phase_attribute_order
    assert module.REQUIRED_PHASES
    assert "ResponseGenerationPhase" not in module.REQUIRED_PHASES
    assert "AffectUpdatePhase" in module.REQUIRED_PHASES


@pytest.mark.parametrize("flag", ["--allow-degraded"])
def test_the_flag_that_records_a_degraded_run_exists(flag: str) -> None:
    source = (REPO / "tools" / "run_subject_core.py").read_text()
    assert flag in source
    assert "authoritative" in source
