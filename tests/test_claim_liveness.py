"""Evidence that expires: a MEASURED_LIVE claim decays when its channel goes quiet.

The registry could previously only record what an author asserted at
registration time. These tests pin the property that makes it self-
correcting: the claim reads its own telemetry, and an undeclared or silent
channel demotes it rather than leaving a measurement standing on nothing.
"""
from __future__ import annotations

import time

import pytest

from core.fsw.telemetry_dictionary import (
    ChannelType,
    channel,
    get_telemetry,
    reset_telemetry_for_test,
    write,
)
from core.organism.claim_liveness import channel_liveness, effective_evidence, liveness_report
from core.organism.model_validation import Claim, Evidence


@pytest.fixture(autouse=True)
def _clean_telemetry():
    reset_telemetry_for_test()
    yield
    reset_telemetry_for_test()


def _declare(name: str, *, stale_after_s: float = 120.0, identifier: int = 0x0FE0) -> None:
    channel(
        identifier=identifier,
        name=name,
        type=ChannelType.FLOAT,
        unit="count",
        description="test channel",
        owner="tests",
        stale_after_s=stale_after_s,
    )


def test_undeclared_channel_does_not_read_as_healthy():
    """The whole hazard: state() answers NOMINAL for a name it never heard."""
    assert get_telemetry().state("core.typo.never_declared").value == "nominal"
    live = channel_liveness("core.typo.never_declared")
    assert not live.declared
    assert not live.supports
    assert "not declared" in live.reason()


def test_fresh_channel_leaves_declared_evidence_standing():
    _declare("test.fresh")
    write("test.fresh", 1.0)
    resolved, note, liveness = effective_evidence(Evidence.MEASURED_LIVE, ["test.fresh"])
    assert resolved is Evidence.MEASURED_LIVE
    assert note == ""
    assert liveness[0].supports


def test_silent_channel_demotes_a_live_claim():
    _declare("test.silent", stale_after_s=0.0)
    write("test.silent", 1.0)
    time.sleep(0.01)
    resolved, note, _ = effective_evidence(Evidence.MEASURED_LIVE, ["test.silent"])
    assert resolved is Evidence.UNMEASURED
    assert "not arriving" in note


def test_never_written_channel_demotes_a_live_claim():
    """A declared channel with no reading demotes, whichever way it got there.

    The demotion is the property. The sentence explaining it is not one
    sentence: a publisher that ran and left the channel empty is a fault in
    the organ behind it, and no publisher at all means this process is simply
    not the one taking readings. This asserted the first wording while setting
    up the second state, so it broke when the two were told apart -- which was
    the improvement, not the regression.
    """
    _declare("test.never_written")
    resolved, note, _ = effective_evidence(Evidence.MEASURED_LIVE, ["test.never_written"])
    assert resolved is Evidence.UNMEASURED
    assert "test.never_written" in note, note


def test_the_two_ways_a_channel_can_be_empty_read_differently(monkeypatch):
    """No publisher here is not the same finding as a publisher that wrote nothing."""
    import core.organism.claim_liveness as liveness

    _declare("test.two_ways")

    monkeypatch.setattr(liveness, "_a_publisher_ran_for", lambda channel: False)
    nobody = channel_liveness("test.two_ways")
    assert nobody.sampled_here is False
    assert "no publisher running in this process" in nobody.reason()

    monkeypatch.setattr(liveness, "_a_publisher_ran_for", lambda channel: True)
    ran_and_wrote_nothing = channel_liveness("test.two_ways")
    assert ran_and_wrote_nothing.sampled_here is True
    assert "has never been written" in ran_and_wrote_nothing.reason()


def test_unbound_claim_is_unchanged():
    resolved, note, liveness = effective_evidence(Evidence.MEASURED_LIVE, [])
    assert resolved is Evidence.MEASURED_LIVE
    assert note == ""
    assert liveness == []


def test_synthetic_evidence_is_not_overwritten_by_a_silent_channel():
    """A weaker-but-true label must not be replaced by a different weak one."""
    _declare("test.quiet", stale_after_s=0.0)
    write("test.quiet", 1.0)
    time.sleep(0.01)
    resolved, note, _ = effective_evidence(Evidence.MEASURED_SYNTHETIC, ["test.quiet"])
    assert resolved is Evidence.MEASURED_SYNTHETIC
    assert note == ""


def test_claim_stops_being_citable_when_its_channel_dies():
    _declare("test.citable", stale_after_s=0.0)
    write("test.citable", 1.0)
    time.sleep(0.01)
    claim = Claim(
        statement="a thing is measured",
        test="test_thing",
        owner="tests",
        live_channels=("test.citable",),
    )
    assert not claim.is_evidence_for_the_system
    payload = claim.to_dict()
    assert payload["evidence"] == "measured_live"
    assert payload["effective_evidence"] == "unmeasured"
    assert payload["citable_as_evidence"] is False


def test_claim_stays_citable_while_its_channel_lives():
    _declare("test.alive")
    write("test.alive", 1.0)
    claim = Claim(
        statement="a thing is measured",
        test="test_thing",
        owner="tests",
        live_channels=("test.alive",),
    )
    assert claim.is_evidence_for_the_system
    assert claim.to_dict()["effective_evidence"] == "measured_live"


def test_liveness_report_names_the_decayed_claims():
    _declare("test.dead", stale_after_s=0.0)
    write("test.dead", 1.0)
    _declare("test.ok", identifier=0x0FE1)
    write("test.ok", 1.0)
    time.sleep(0.01)
    decayed = Claim(
        statement="decayed", test="t1", owner="tests", live_channels=("test.dead",)
    )
    healthy = Claim(
        statement="healthy", test="t2", owner="tests", live_channels=("test.ok",)
    )
    unbound = Claim(statement="unbound", test="t3", owner="tests")
    report = liveness_report([decayed, healthy, unbound])
    assert report["claims_bound_to_telemetry"] == 2
    assert report["decayed_count"] == 1
    assert report["decayed"][0]["statement"] == "decayed"


def test_declared_evidence_is_preserved_not_rewritten():
    """Demotion must not erase the fact that it was once measured live."""
    _declare("test.history", stale_after_s=0.0)
    write("test.history", 1.0)
    time.sleep(0.01)
    claim = Claim(
        statement="s", test="t", owner="tests", live_channels=("test.history",)
    )
    assert claim.evidence is Evidence.MEASURED_LIVE
    assert claim.to_dict()["evidence"] == "measured_live"
