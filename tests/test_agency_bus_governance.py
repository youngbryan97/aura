from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from core.agency.agency_bus import AgencyBus
from core.orchestrator.mixins.autonomy import AutonomyMixin


@dataclass
class _Decision:
    receipt_id: str = "receipt-ok"
    reason: str = "ok"
    approved: bool = True

    def is_approved(self) -> bool:
        return self.approved


class _Will:
    def __init__(self, *, approved: bool = True, valid_receipt: bool = True):
        self.approved = approved
        self.valid_receipt = valid_receipt
        self.decisions = []
        self.verifications = []

    def decide(self, **kwargs):
        self.decisions.append(kwargs)
        return _Decision(approved=self.approved)

    def verify_receipt(self, receipt_id: str) -> bool:
        self.verifications.append(receipt_id)
        return receipt_id == "receipt-ok" and self.valid_receipt


def test_agency_bus_auto_acquires_and_verifies_will_receipt(monkeypatch):
    will = _Will()
    monkeypatch.setattr("core.will.get_will", lambda: will)
    bus = AgencyBus()

    proposal = {"origin": "test", "text": "hello", "priority_class": "duty"}

    assert bus.submit(proposal) is True
    assert proposal["will_receipt"] == "receipt-ok"
    assert bus.stats["recent_audit"]
    assert len(will.decisions) == 1
    assert will.verifications == ["receipt-ok"]
    assert will.decisions[0]["source"] == "test"
    context = will.decisions[0]["context"]
    # The bus's own claims, exactly. Asserted as a subset because provenance
    # keys are added by whatever handles the decision downstream, and an
    # equality check turned every added field into a failure here. What must
    # not change is what the bus itself said.
    assert {
        "source": "test",
        "autonomous": True,
        "agency_bus": True,
        "priority_class": "duty",
    }.items() <= context.items()
    # Anything else has to be namespaced provenance rather than a redefinition
    # of one of those four.
    extra = set(context) - {"source", "autonomous", "agency_bus", "priority_class"}
    assert all(key.startswith("action_executor_") for key in extra), extra


def test_agency_bus_fails_closed_when_will_unavailable(monkeypatch):
    will_resolution_attempts = []

    def unavailable():
        will_resolution_attempts.append("attempted")
        raise RuntimeError("will offline")

    monkeypatch.setattr("core.will.get_will", unavailable)
    bus = AgencyBus()

    assert bus.submit({"origin": "test", "text": "hello", "priority_class": "duty"}) is False
    assert will_resolution_attempts == ["attempted"]
    assert bus.stats["recent_audit"] == []


def test_agency_bus_rejects_unattributed_autonomous_proposal(monkeypatch):
    monkeypatch.setattr(
        "core.will.get_will",
        lambda: (_ for _ in ()).throw(AssertionError("Will must not be called")),
    )
    bus = AgencyBus()

    assert bus.submit({"text": "hello", "priority_class": "duty"}) is False
    assert bus.stats["recent_audit"] == []


def test_agency_bus_rejects_invalid_receipt(monkeypatch):
    monkeypatch.setattr("core.governance.will.get_will", lambda: _Will(valid_receipt=False))
    bus = AgencyBus()

    assert (
        bus.submit(
            {
                "origin": "test",
                "text": "hello",
                "priority_class": "duty",
                "will_receipt": "receipt-ok",
            }
        )
        is False
    )
    assert bus.stats["recent_audit"] == []


def test_boredom_gate_reuses_the_receipt_it_already_acquired(monkeypatch):
    proposals = []
    scheduled = []
    decision = _Decision()

    monkeypatch.setattr(
        "core.will.get_will",
        lambda: SimpleNamespace(decide=lambda **_kwargs: decision),
    )
    monkeypatch.setattr(
        "core.agency.agency_bus.AgencyBus.get",
        lambda: SimpleNamespace(submit=lambda proposal: proposals.append(proposal) or True),
    )

    async def governed_impulse(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        "core.orchestrator.mixins.autonomy.run_governed_impulse",
        governed_impulse,
    )

    class Runtime(AutonomyMixin):
        personality_engine = None

        def _fire_and_forget(self, awaitable, *, name):
            scheduled.append(name)
            awaitable.close()

    Runtime()._trigger_boredom_impulse()

    assert len(proposals) == 1
    assert proposals[0]["will_receipt"] == "receipt-ok"
    assert proposals[0]["origin"] == "orchestrator_boredom"
    assert scheduled == ["autonomy.boredom_impulse"]
