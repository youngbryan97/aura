"""The causal campaign ran hourly for three weeks and measured nothing.

It did not fail. It reported success. The live ledger held nine channels, each
with three treatment samples and three null samples, and every one of the
fifty-four numbers was exactly 0.0 — a perfect null and a perfect treatment
from a model that was never called.

The chain, from the live logs:

1. the probe asks the inference gate for a background generation;
2. the Brainstem returns no text, and background requests have their local
   fallback suppressed to protect foreground latency, so the gate refuses;
3. ``_refuse_generation`` returns ``None``, which the caller turned into
   ``""``;
4. ``text_divergence("", "")`` is 0.0, and so is the null pair;
5. the probe counted three completed trials and the campaign logged
   "1 channel(s) measured".

Nine generations took four milliseconds. Nothing anywhere said a word.

These tests hold each link. The arm that produced nothing is a failed arm; a
campaign that completed no trial did not run; and the refusal reaches the
record by name instead of as silence.
"""

from __future__ import annotations

import asyncio

import pytest

from core.verify.causal_influence import InfluenceLedger, text_divergence
from core.verify.influence_probe import arm_produced_nothing, measure_channel
from core.verify.lesion_registry import LesionHandle, get_lesion_registry


@pytest.fixture
def a_lesionable_channel():
    registry = get_lesion_registry()
    name = "test.refused_arm"
    registry.register(
        LesionHandle(
            channel=name,
            lesion=lambda: None,
            restore=lambda: None,
            owner="test_a_campaign_that_measured_nothing_for_three_weeks",
            neutral_description="nothing, deliberately",
            direct_actuation=True,
        ),
        replace=True,
    )
    try:
        yield name
    finally:
        registry.unregister(name)


def test_two_arms_that_said_nothing_are_not_zero_apart():
    """The metric cannot tell the difference, so the probe has to."""

    assert text_divergence("", "") == 0.0
    assert arm_produced_nothing("")
    assert arm_produced_nothing("   \n ")
    assert arm_produced_nothing(None)
    assert not arm_produced_nothing("a reply")


def test_an_empty_vector_arm_is_also_nothing():
    assert arm_produced_nothing([], "vector")
    assert not arm_produced_nothing([0.0, 0.0], "vector")


def test_a_refused_generation_does_not_become_a_measurement(a_lesionable_channel):
    """This is the exact shape of the live ledger: three trials, all zeros."""

    async def refused() -> str:
        return ""  # what the gate's None became at the call site

    book = InfluenceLedger()
    report = asyncio.run(
        measure_channel(
            a_lesionable_channel,
            generate=refused,
            trials=3,
            per_generation_timeout_s=5.0,
            deadline_s=30.0,
            ledger=book,
        )
    )

    assert report.trials_completed == 0
    assert report.generation_failures == 3
    assert report.verdict is not None
    assert report.verdict.n_treatment == 0
    assert report.verdict.n_null == 0
    assert "no output" in " ".join(report.notes)


def test_an_arm_that_raises_names_why_it_did_not_generate(a_lesionable_channel):
    async def refused() -> str:
        raise RuntimeError("deferred: background_local_fallback_suppressed")

    book = InfluenceLedger()
    report = asyncio.run(
        measure_channel(
            a_lesionable_channel,
            generate=refused,
            trials=2,
            per_generation_timeout_s=5.0,
            deadline_s=30.0,
            ledger=book,
        )
    )

    assert report.trials_completed == 0
    assert "background_local_fallback_suppressed" in " ".join(report.notes)


def test_a_real_pair_is_still_recorded(a_lesionable_channel):
    """The guard must not cost a channel that does generate."""

    outputs = iter(
        [
            "the intact answer runs long",
            "lesioned",
            "the intact answer runs long again",
        ]
    )

    async def speaking() -> str:
        return next(outputs)

    book = InfluenceLedger()
    report = asyncio.run(
        measure_channel(
            a_lesionable_channel,
            generate=speaking,
            trials=1,
            per_generation_timeout_s=5.0,
            deadline_s=30.0,
            ledger=book,
        )
    )

    assert report.trials_completed == 1
    assert report.verdict is not None
    assert report.verdict.n_treatment == 1
    assert report.verdict.n_null == 1


def test_a_campaign_whose_arms_all_refused_did_not_run(a_lesionable_channel, monkeypatch):
    """``ran`` was ``bool(trials)``, and a trial object exists either way.

    That is what let the conductor write "ran" into the record an hour at a
    time while the ledger filled with zeros.
    """

    from core.verify import influence_campaign

    monkeypatch.setattr(influence_campaign, "campaign_admission_reason", lambda **_: "")

    async def refused() -> str:
        raise RuntimeError("deferred: background_local_fallback_suppressed")

    report = asyncio.run(
        influence_campaign.run_influence_campaign(
            generate=refused,
            channels=[a_lesionable_channel],
            trials=2,
            per_generation_timeout_s=5.0,
            deadline_s=30.0,
            persist=False,
        )
    )

    assert report.trials, "the channel was attempted"
    assert report.trials_completed == 0
    assert report.ran is False
    assert "background_local_fallback_suppressed" in report.why_nothing_landed
    assert report.as_dict()["trials_completed"] == 0


def test_a_campaign_that_completed_a_trial_ran(a_lesionable_channel, monkeypatch):
    from core.verify import influence_campaign

    monkeypatch.setattr(influence_campaign, "campaign_admission_reason", lambda **_: "")

    outputs = iter(["intact one", "lesioned", "intact two"])

    async def speaking() -> str:
        return next(outputs)

    report = asyncio.run(
        influence_campaign.run_influence_campaign(
            generate=speaking,
            channels=[a_lesionable_channel],
            trials=1,
            per_generation_timeout_s=5.0,
            deadline_s=30.0,
            persist=False,
        )
    )

    assert report.ran is True
    assert report.trials_completed == 1
    assert report.why_nothing_landed == ""
