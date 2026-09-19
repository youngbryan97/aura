"""Every new detector has a reader.

This repository's most expensive recurring defect is the half-wired
channel: a writer with no reader, and a live surface reporting a
measurement that was structurally impossible. Two flagship numbers shipped
that way.

Each detector added alongside these tests writes somewhere. This file
asserts the other half — that the runtime integrity block actually READS
them — so a future refactor that drops the reader fails here rather than
in six months when someone believes a zero.
"""
from __future__ import annotations

import pytest

from core.runtime.health_contract import _runtime_integrity_block


@pytest.fixture(scope="module")
def block() -> dict:
    return _runtime_integrity_block()


def test_no_block_errored(block):
    """Each add-on is isolated, so a failure shows up as a *_error key."""
    errored = {
        k: v
        for k, v in block.items()
        if k.endswith("_error")
        and k.split("_error")[0] in {"grounding", "chronic_faults", "judgement"}
    }
    assert not errored, f"integrity add-ons failed: {errored}"


def test_work_ledger_is_read(block):
    """The fabrication audit's evidence source must be visible."""
    assert "grounding" in block
    assert "work_ledger" in block["grounding"]
    assert "turns_tracked" in block["grounding"]["work_ledger"]


def test_injection_canaries_are_read(block):
    canaries = block["grounding"]["injection_canaries"]
    for key in ("evaluated", "incidents", "incident_rate", "blind", "inconclusive"):
        assert key in canaries


def test_a_blind_canary_lane_is_surfaced(block):
    """A detector that stopped detecting is itself the finding."""
    assert "blind" in block["grounding"]["injection_canaries"]


def test_chronic_faults_are_read(block):
    chronic = block["chronic_faults"]
    assert "signatures_tracked" in chronic
    assert "chronic" in chronic
    # The floor is reported so nobody has to read the module to learn that
    # habituation never fully silences anything.
    assert chronic["residual_floor"] == pytest.approx(0.4)


def test_harmful_memories_are_read(block):
    """Registered and reporting, or plainly absent — the same rule its
    sibling below already states for the ambient governor.

    An unregistered ledger used to RAISE here, so a process that simply
    had none — every test process, a tool, a partial boot — put
    judgement_error on the integrity surface and took the whole judgement
    block with it, the governor's reading included, which was there and
    fine. Not measured is not the same as measured and bad.
    """

    retrieval = block["judgement"]["retrieval"]
    assert "registered" in retrieval
    if retrieval["registered"]:
        assert "harmful_memories" in retrieval
        assert "graded" in retrieval
    else:
        # Never zeros that would read as a ledger seeing nothing harmful.
        assert "harmful_memories" not in retrieval
        assert "graded" not in retrieval


def test_ambient_restraint_is_read(block):
    ambient = block["judgement"]["ambient"]
    # Either it is registered and reports, or it says plainly that it is
    # not — never zeros that look like a quiet, running governor.
    assert "registered" in ambient
    if ambient["registered"]:
        assert "restraint_rate" in ambient
        assert "calibration" in ambient


def test_an_unregistered_governor_says_so_rather_than_reporting_zeros():
    """Absence must not be presentable as a clean result."""
    from core.container import ServiceContainer

    had = ServiceContainer.has("ambient_governor")
    if had:
        pytest.skip("governor already registered in this process")
    block = _runtime_integrity_block()
    assert block["judgement"]["ambient"] == {"registered": False}
