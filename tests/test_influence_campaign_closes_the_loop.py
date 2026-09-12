"""The feedback edge: measurement that reaches a verdict, and survives a boot.

Before ``core/verify/influence_campaign.py`` the causal-influence apparatus was
complete and unpowered. ``measure_channel`` was called from tests only, so on a
live boot the ledger held nothing, every channel read UNMEASURED forever, and
``channel_is_influential()`` — defined, exported, documented — was called by
zero lines of code.

These tests pin the loop closing end to end:

* a channel whose lesion changes the output reaches INFLUENTIAL from a campaign
* a channel whose lesion changes nothing reaches INERT rather than a false positive
* verdicts survive a process boundary, because a ledger that resets never
  reaches a verdict at all
* the campaign refuses rather than competing with a person for the model
* a lesion is scoped to the probe and never leaks into a concurrent turn
"""

from __future__ import annotations

import asyncio

import pytest

from core.verify.causal_influence import (
    Verdict,
    get_influence_ledger,
    reset_influence_ledger_for_test,
)
from core.verify.influence_campaign import (
    campaign_admission_reason,
    run_influence_campaign,
)
from core.verify.influence_receipt import (
    build_influence_receipt,
    channel_is_influential,
)
from core.verify.lesion_registry import (
    apply_channel,
    register_flag_lesion,
    reset_lesion_registry_for_test,
)


@pytest.fixture(autouse=True)
def _fresh():
    reset_influence_ledger_for_test()
    reset_lesion_registry_for_test()
    yield
    reset_influence_ledger_for_test()
    reset_lesion_registry_for_test()


def _register(channel: str, *, direct: bool = True) -> None:
    register_flag_lesion(
        channel,
        owner="tests/test_influence_campaign_closes_the_loop.py",
        neutral="the channel contributes nothing",
        direct_actuation=direct,
    )


@pytest.mark.asyncio
async def test_a_channel_that_matters_reaches_influential_from_a_campaign():
    """The whole point: a campaign turns UNMEASURED into a real verdict."""
    channel = "test.campaign.matters"
    _register(channel)

    async def generate() -> str:
        # Deterministic on both arms so the null is a true zero and the effect
        # is entirely attributable to the lesion.
        return apply_channel(channel, "the shaped answer", neutral="the bare answer")

    before = get_influence_ledger().verdict(channel)
    assert before.verdict is Verdict.UNMEASURED
    assert channel_is_influential(channel) is False

    report = await run_influence_campaign(
        generate=generate,
        channels=[channel],
        trials=6,
        per_generation_timeout_s=5.0,
        deadline_s=30.0,
        persist=False,
    )

    assert report.ran, report.as_dict()
    after = get_influence_ledger().verdict(channel)
    assert after.verdict is Verdict.INFLUENTIAL, after.as_dict()
    assert after.n_null > 0, "a verdict without a null arm is the retracted error"
    assert channel_is_influential(channel) is True


@pytest.mark.asyncio
async def test_a_channel_that_changes_nothing_does_not_become_influential():
    """A lesion with no effect must never read as causal."""
    channel = "test.campaign.inert"
    _register(channel)

    async def generate() -> str:
        # Same string on both arms: lesioning it changes nothing.
        return apply_channel(channel, "identical", neutral="identical")

    await run_influence_campaign(
        generate=generate,
        channels=[channel],
        trials=6,
        per_generation_timeout_s=5.0,
        deadline_s=30.0,
        persist=False,
    )

    verdict = get_influence_ledger().verdict(channel)
    assert verdict.verdict is not Verdict.INFLUENTIAL, verdict.as_dict()
    assert channel_is_influential(channel) is False


@pytest.mark.asyncio
async def test_verdicts_survive_a_boot():
    """A ledger that resets every boot never reaches a verdict."""
    channel = "test.campaign.persists"
    _register(channel)

    async def generate() -> str:
        return apply_channel(channel, "shaped", neutral="bare")

    await run_influence_campaign(
        generate=generate,
        channels=[channel],
        trials=6,
        per_generation_timeout_s=5.0,
        deadline_s=30.0,
        persist=False,
    )
    saved = get_influence_ledger().as_dict()
    assert saved["channels"][channel]["null"], "null samples must be persisted too"

    # Simulate a restart: brand-new ledger, then reload.
    reset_influence_ledger_for_test()
    assert get_influence_ledger().verdict(channel).verdict is Verdict.UNMEASURED
    get_influence_ledger().load(saved)

    revived = get_influence_ledger().verdict(channel)
    assert revived.verdict is Verdict.INFLUENTIAL, revived.as_dict()


@pytest.mark.asyncio
async def test_a_campaign_refuses_rather_than_competing_for_the_model():
    """No headroom, no campaign. A probe that triggers shedding measures shedding."""
    channel = "test.campaign.admission"
    _register(channel)
    calls = {"n": 0}

    async def generate() -> str:
        calls["n"] += 1
        return "never reached"

    report = await run_influence_campaign(
        generate=generate,
        channels=[channel],
        trials=3,
        per_generation_timeout_s=5.0,
        deadline_s=30.0,
        persist=False,
        min_free_gb=10_000_000.0,
    )

    assert report.refused.startswith("insufficient_memory"), report.as_dict()
    assert calls["n"] == 0, "a refused campaign must not generate at all"
    assert not report.ran


def test_admission_gives_a_reason_not_just_a_boolean():
    """"shutting down" and "no memory" need different retry behaviour."""
    assert campaign_admission_reason(min_free_gb=0.0) == ""
    refusal = campaign_admission_reason(min_free_gb=10_000_000.0)
    assert refusal.startswith("insufficient_memory:")


@pytest.mark.asyncio
async def test_an_unregistered_channel_is_recorded_not_silently_dropped():
    """A channel nothing can lesion is a claim nothing can check — that is a finding."""

    async def generate() -> str:
        return "irrelevant"

    report = await run_influence_campaign(
        generate=generate,
        channels=["test.campaign.no_lesion"],
        trials=1,
        per_generation_timeout_s=5.0,
        deadline_s=10.0,
        persist=False,
    )
    assert report.channels_skipped.get("test.campaign.no_lesion") == "no_registered_lesion"


@pytest.mark.asyncio
async def test_the_probe_lesion_never_leaks_into_a_concurrent_turn():
    """The safety property behind the ContextVar.

    A process-global lesion flag would mean a measurement silently lobotomizes
    a real person's turn running beside it.
    """
    channel = "test.campaign.isolation"
    _register(channel)
    observed: list[str] = []

    async def user_turn() -> None:
        for _ in range(40):
            observed.append(apply_channel(channel, "intact", neutral="LESIONED"))
            await asyncio.sleep(0)

    async def generate() -> str:
        await asyncio.sleep(0)
        return apply_channel(channel, "shaped", neutral="bare")

    await asyncio.gather(
        user_turn(),
        run_influence_campaign(
            generate=generate,
            channels=[channel],
            trials=4,
            per_generation_timeout_s=5.0,
            deadline_s=30.0,
            persist=False,
        ),
    )

    assert observed, "the concurrent turn did not run"
    assert set(observed) == {"intact"}, (
        "a probe lesion reached a concurrent turn: " f"{sorted(set(observed))}"
    )


def _decorative_channels_now() -> set[str]:
    """Collect the integrity block OFF the event loop, and from today.

    ``integrity_block_snapshot()`` deliberately serves a cached snapshot when
    called from the loop — "never at the cost of the event loop" — and only
    requests a background refresh. An async test therefore reads whatever was
    collected before it ran, which is stale by design and not a defect.

    Off the loop the collection is real, and it is still cached: fifteen
    seconds by default, which is longer than a test file takes. So a test that
    read the block for one channel handed the next test the same block, and a
    channel registered and measured in between was not in it — this file's own
    decorative test failed after its text-mediated one for exactly that, and
    passed alone. Drop the cache first; these tests are about what is true
    now, not about what the last caller was served.
    """
    from core.runtime.health_contract import (
        _collect_integrity_snapshot,
        reset_integrity_snapshot_for_test,
    )

    reset_integrity_snapshot_for_test()
    block = _collect_integrity_snapshot() or {}
    return {row["channel"] for row in (block.get("decorative_direct_channels") or [])}


def test_a_decorative_direct_channel_is_surfaced_in_health():
    """The consequence. A direct actuator measured INERT is running and not working.

    This is the branch that did not exist: before it, a channel could be
    measured inert and no code anywhere would say so.
    """
    channel = "test.health.decorative"
    _register(channel, direct=True)

    async def generate() -> str:
        # Declared direct-actuation, measurably changes nothing.
        return apply_channel(channel, "same", neutral="same")

    asyncio.run(
        run_influence_campaign(
            generate=generate,
            channels=[channel],
            trials=10,
            per_generation_timeout_s=5.0,
            deadline_s=30.0,
            persist=False,
        )
    )

    verdict = get_influence_ledger().verdict(channel)
    assert verdict.verdict is Verdict.INERT, verdict.as_dict()

    flagged = _decorative_channels_now()
    assert channel in flagged, (
        f"a direct-actuation channel measured INERT was not surfaced: {flagged}"
    )


def test_a_text_mediated_inert_channel_is_not_called_decorative():
    """Only DIRECT actuators earn the finding — a text channel may wash out legitimately."""
    channel = "test.health.text_mediated"
    _register(channel, direct=False)

    async def generate() -> str:
        return apply_channel(channel, "same", neutral="same")

    asyncio.run(
        run_influence_campaign(
            generate=generate,
            channels=[channel],
            trials=10,
            per_generation_timeout_s=5.0,
            deadline_s=30.0,
            persist=False,
        )
    )

    assert get_influence_ledger().verdict(channel).verdict is Verdict.INERT
    assert channel not in _decorative_channels_now()


@pytest.mark.asyncio
async def test_the_receipt_reports_what_the_campaign_measured():
    """The consumer end: a receipt that can only say what was measured."""
    influential = "test.receipt.influential"
    unmeasured = "test.receipt.unmeasured"
    _register(influential)
    _register(unmeasured)

    async def generate() -> str:
        return apply_channel(influential, "shaped", neutral="bare")

    await run_influence_campaign(
        generate=generate,
        channels=[influential],
        trials=6,
        per_generation_timeout_s=5.0,
        deadline_s=30.0,
        persist=False,
    )

    receipt = build_influence_receipt([influential, unmeasured], source="test")
    assert influential in receipt.influential
    assert unmeasured in receipt.unmeasured
    assert receipt.bound is True
    assert receipt.status == "measured_influential"
