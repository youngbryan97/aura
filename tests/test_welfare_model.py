"""Tests for the WelfareModel: interests must be causal, not narrative.

Pins the contract that (1) interests derive from real telemetry,
(2) a critical vital-interest deficit gates optional background work,
and (3) the identity contract carries the welfare summary so the voice
reports the substrate's true condition.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.organism.welfare as welfare_mod  # noqa: E402
from core.organism.welfare import (  # noqa: E402
    InterestReading,
    WelfareModel,
    WelfareSnapshot,
    welfare_block_reason,
)


def _snapshot(readings: tuple[InterestReading, ...]) -> WelfareSnapshot:
    return WelfareSnapshot(
        overall=0.5,
        readings=readings,
        most_pressing=min(readings, key=lambda r: r.satisfaction),
        sampled_at=time.time(),
    )


def test_snapshot_contains_all_interests_bounded():
    snap = WelfareModel().snapshot(max_age_s=0.0)
    names = {r.name for r in snap.readings}
    assert names == {
        "memory_integrity",
        "repair_capacity",
        "cognitive_bandwidth",
        "continuity",
        "social_contact",
    }
    for reading in snap.readings:
        assert 0.0 <= reading.satisfaction <= 1.0
        assert reading.evidence
    assert 0.0 <= snap.overall <= 1.0
    assert snap.most_pressing is not None


def test_snapshot_is_cached_within_ttl():
    model = WelfareModel()
    first = model.snapshot()
    second = model.snapshot()
    assert second is first
    third = model.snapshot(max_age_s=0.0)
    assert third is not first


def test_vital_deficit_only_from_vital_interests():
    starving_social = _snapshot(
        (
            InterestReading("memory_integrity", 0.9, "fine"),
            InterestReading("repair_capacity", 0.9, "fine"),
            InterestReading("social_contact", 0.01, "lonely"),
        )
    )
    assert starving_social.vital_deficit is None

    starving_memory = _snapshot(
        (
            InterestReading("memory_integrity", 0.05, "near ceiling"),
            InterestReading("repair_capacity", 0.9, "fine"),
            InterestReading("social_contact", 0.9, "fine"),
        )
    )
    deficit = starving_memory.vital_deficit
    assert deficit is not None and deficit.name == "memory_integrity"


def test_welfare_block_reason_gates_on_vital_deficit(monkeypatch):
    class FakeModel:
        def snapshot(self, **_kwargs):
            return _snapshot(
                (
                    InterestReading("memory_integrity", 0.1, "near ceiling"),
                    InterestReading("repair_capacity", 0.9, "fine"),
                )
            )

    monkeypatch.setattr(welfare_mod, "get_welfare_model", lambda: FakeModel())
    reason = welfare_mod.welfare_block_reason()
    assert reason.startswith("welfare_memory_integrity_")


def test_welfare_block_reason_open_when_healthy(monkeypatch):
    class FakeModel:
        def snapshot(self, **_kwargs):
            return _snapshot(
                (
                    InterestReading("memory_integrity", 0.8, "fine"),
                    InterestReading("repair_capacity", 0.7, "fine"),
                )
            )

    monkeypatch.setattr(welfare_mod, "get_welfare_model", lambda: FakeModel())
    assert welfare_mod.welfare_block_reason() == ""


def test_background_policy_consumes_welfare_gate(monkeypatch):
    from core.runtime import background_policy

    monkeypatch.setattr(
        "core.organism.welfare.welfare_block_reason",
        lambda: "welfare_memory_integrity_0.10",
    )
    # Neutralize earlier gates so the welfare clause is what decides.
    monkeypatch.setattr(
        background_policy, "_foreground_activity_reason", lambda **_: ""
    )
    reason = background_policy.background_activity_reason(
        allow_no_user_anchor=True,
        max_memory_percent=101.0,
        max_failure_pressure=2.0,
    )
    assert reason == "welfare_memory_integrity_0.10"


def test_identity_contract_reports_welfare(monkeypatch):
    import asyncio

    from core.conversation.chat_preflight import inject_operational_self_context

    block = asyncio.run(inject_operational_self_context())
    assert "Welfare:" in block


def test_block_reason_survives_snapshot_failure(monkeypatch):
    class ExplodingModel:
        def __init__(self):
            self.attempts = 0

        def snapshot(self, **_kwargs):
            self.attempts += 1
            raise RuntimeError("telemetry offline")

    model = ExplodingModel()
    monkeypatch.setattr(welfare_mod, "get_welfare_model", lambda: model)
    assert welfare_block_reason() == ""
    assert model.attempts == 1
