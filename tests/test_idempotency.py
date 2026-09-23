"""Idempotency: a receipt proves what happened, not that it happened once.

A retried request arrives fully authorized — the gate approves it again
because on its own terms it is a legitimate call. These tests pin the
three properties that stop that becoming a second send: a key replays
instead of re-running, concurrent duplicates collapse to one execution,
and a failure does not poison the key.
"""
from __future__ import annotations

import asyncio

import pytest

from core.runtime.idempotency import (
    IdempotencyLedger,
    requires_idempotency_key,
)


@pytest.fixture
def ledger():
    return IdempotencyLedger(ttl_seconds=60.0)


# ── Replay ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_second_call_with_the_same_key_does_not_run_again(ledger):
    runs = []

    async def send():
        runs.append(1)
        return {"sent": True, "id": len(runs)}

    first = await ledger.run_once("k1", send)
    second = await ledger.run_once("k1", send)

    assert len(runs) == 1, "the action executed twice"
    assert first.replayed is False
    assert second.replayed is True
    assert second.value == first.value


@pytest.mark.asyncio
async def test_a_different_key_runs_again(ledger):
    runs = []

    async def send():
        runs.append(1)
        return len(runs)

    await ledger.run_once("k1", send)
    await ledger.run_once("k2", send)
    assert len(runs) == 2


@pytest.mark.asyncio
async def test_no_key_never_dedupes(ledger):
    runs = []

    async def send():
        runs.append(1)
        return len(runs)

    await ledger.run_once("", send)
    await ledger.run_once("   ", send)
    assert len(runs) == 2, "an absent key must not collapse unrelated calls"


# ── Single flight ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_duplicates_collapse_to_one_execution(ledger):
    """The real shape of a retry: the client resends while the first call
    is still in flight. A plain "have I seen this key" check does nothing
    here, because neither has finished."""
    runs = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_send():
        runs.append(1)
        started.set()
        await release.wait()
        return "sent-once"

    first = asyncio.create_task(ledger.run_once("k1", slow_send))
    await started.wait()
    second = asyncio.create_task(ledger.run_once("k1", slow_send))
    await asyncio.sleep(0)
    release.set()

    a, b = await asyncio.gather(first, second)

    assert len(runs) == 1, "concurrent duplicates both executed"
    assert a.value == b.value == "sent-once"
    assert {a.replayed, b.replayed} == {False, True}


@pytest.mark.asyncio
async def test_a_waiter_giving_up_does_not_cancel_the_real_action(ledger):
    """A waiter's own client can hang up. That must not cancel the
    in-flight action everyone else is depending on."""
    completed = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_send():
        started.set()
        await release.wait()
        completed.append(1)
        return "done"

    first = asyncio.create_task(ledger.run_once("k1", slow_send))
    await started.wait()
    waiter = asyncio.create_task(ledger.run_once("k1", slow_send))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    release.set()
    outcome = await first
    assert outcome.value == "done"
    assert completed == [1]


# ── Failure ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_failure_does_not_poison_the_key(ledger):
    """Caching an exception would turn one transient error into a
    permanently dead key — the retry could never succeed."""
    attempts = []

    async def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("network blip")
        return "sent"

    with pytest.raises(RuntimeError):
        await ledger.run_once("k1", flaky)

    outcome = await ledger.run_once("k1", flaky)
    assert outcome.value == "sent"
    assert outcome.replayed is False
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_concurrent_waiters_see_the_failure_too(ledger):
    started = asyncio.Event()
    release = asyncio.Event()

    async def failing():
        started.set()
        await release.wait()
        raise RuntimeError("boom")

    first = asyncio.create_task(ledger.run_once("k1", failing))
    await started.wait()
    second = asyncio.create_task(ledger.run_once("k1", failing))
    await asyncio.sleep(0)
    release.set()

    results = await asyncio.gather(first, second, return_exceptions=True)
    assert all(isinstance(item, RuntimeError) for item in results)


# ── Bounds ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_expired_keys_stop_replaying():
    tiny = IdempotencyLedger(ttl_seconds=0.01)
    runs = []

    async def send():
        runs.append(1)
        return len(runs)

    await tiny.run_once("k1", send)
    await asyncio.sleep(0.05)
    await tiny.run_once("k1", send)
    assert len(runs) == 2


@pytest.mark.asyncio
async def test_ledger_is_bounded_against_a_caller_minting_fresh_keys():
    small = IdempotencyLedger(ttl_seconds=600.0, max_entries=8)

    async def send():
        return True

    for index in range(50):
        await small.run_once(f"key-{index}", send)

    assert len(small._entries) <= 8


# ── Where a key is demanded ──────────────────────────────────────────


@pytest.mark.parametrize(
    "effect_scope,source,expected",
    [
        # Changes something, arrived from somewhere that can resend it.
        ("state_mutation", "paired_device:phone", True),
        ("privileged_mutation", "remote", True),
        ("external_io", "channel:telegram", True),
        # Reads cost nothing to repeat.
        ("read_only", "paired_device:phone", False),
        ("status", "remote", False),
        # Internal callers are not retried across a network, and demanding
        # a key from them would be ceremony with no failure to prevent.
        ("state_mutation", "autonomous", False),
        ("privileged_mutation", "desktop_ui", False),
    ],
)
def test_key_is_required_only_where_duplicates_can_arrive(effect_scope, source, expected):
    assert requires_idempotency_key(effect_scope=effect_scope, source=source) is expected


# ── Wiring ───────────────────────────────────────────────────────────
#
# Everything above tests the ledger, which proves nothing about whether
# the execution waist uses it. These drive CapabilityEngine.execute.


def _probe_engine(skill_name: str):
    import logging
    from types import SimpleNamespace

    from core.capability_engine import CapabilityEngine, SkillMetadata

    logger = logging.getLogger("test.idempotency.probe")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False

    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.logger = logger
    engine.error_boundary = lambda fn: fn
    engine.skills = {
        skill_name: SkillMetadata(
            name=skill_name,
            description="idempotency probe",
            skill_class=lambda: object(),
            metabolic_cost=1,
        )
    }
    engine.instances = {}
    engine.sandbox = None
    engine.rosetta_stone = None
    engine.temporal = None
    engine.orchestrator = SimpleNamespace(mycelium=None)
    engine.skill_last_errors = {}
    engine._emit_skill_status = lambda *a, **k: None
    engine.max_retries = 1
    engine.retry_delay = 0.0
    engine.timeout = 1.0
    return engine


@pytest.mark.asyncio
async def test_execute_refuses_remote_mutation_without_a_key(monkeypatch):
    from core.capability_engine import CapabilityEngine
    from core.runtime import idempotency

    monkeypatch.setattr(
        idempotency, "requires_idempotency_key", lambda **_kw: True, raising=True
    )
    monkeypatch.setattr(
        "core.capability_engine.requires_idempotency_key",
        lambda **_kw: True,
    )

    engine = _probe_engine("send_message")
    result = await CapabilityEngine.execute(
        engine,
        "send_message",
        {"to": "someone"},
        context={"origin": "paired_device:phone"},
    )

    assert result["ok"] is False
    assert result["status"] == "idempotency_key_required"


@pytest.mark.asyncio
async def test_execute_replays_instead_of_running_twice():
    """Drives the real entry point twice with one key.

    The dedupe wraps the outermost guarded body, so this holds regardless
    of what the body decides — which is what makes it testable without
    standing up the twenty gates between here and a real skill. The second
    call must come back marked as a replay rather than re-entering.
    """
    from core.capability_engine import CapabilityEngine
    from core.runtime.idempotency import reset_idempotency_ledger_for_test

    reset_idempotency_ledger_for_test()
    try:
        engine = _probe_engine("send_message")
        context = {"origin": "desktop_ui", "idempotency_key": "abc-123"}

        first = await CapabilityEngine.execute(
            engine, "send_message", {"to": "someone"}, context=dict(context)
        )
        second = await CapabilityEngine.execute(
            engine, "send_message", {"to": "someone"}, context=dict(context)
        )

        assert first.get("idempotent_replay") is None, "first call was not an execution"
        assert second.get("idempotent_replay") is True, "second call re-entered the body"
        assert second.get("idempotency_key") == "abc-123"
        # Same outcome, not a fresh decision.
        assert {k: v for k, v in second.items() if not k.startswith("idempoten")} == first
    finally:
        reset_idempotency_ledger_for_test()


def test_execute_actually_wires_the_ledger_and_the_requirement():
    """Source-level proof that the waist uses both halves — the behaviour
    above is only meaningful if execute() is the thing doing it."""
    from pathlib import Path

    source = Path("core/capability_engine.py").read_text(encoding="utf-8")
    assert "from core.runtime.idempotency import" in source
    assert "requires_idempotency_key(" in source
    assert "await get_idempotency_ledger().run_once(" in source
    assert '"status": "idempotency_key_required"' in source
    # The replay is marked by the outcome that knows it was one, and execute
    # hands back what the outcome says.
    assert "outcome.value_saying_so(idempotency_key)" in source
    ledger = Path("core/runtime/idempotency.py").read_text(encoding="utf-8")
    assert '"idempotent_replay": True' in ledger
