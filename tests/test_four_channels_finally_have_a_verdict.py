"""Four faculty channels now carry an experimentally measured verdict.

The count stood at zero for as long as the apparatus existed, and the reason
was never statistical. The hourly campaign asks the inference gate for a
*background* generation, and every ``apply_channel`` site it rotates through
sits behind a guard a background call does not pass: the circumplex behind
``not is_background and origin_is_user_facing`` in the gate, the live-mind and
sampling-bias sites inside the clean-user-surface contract in the engine. The
job could not have moved one of the nine channels it was measuring, however
well its generations had gone — and they did not go at all.

A direct call can move four of them, because the advisory passes, the
circumplex and the production sampling fold all run without a turn.
``tools/run_influence_trials_on_the_substrate.py`` runs those as paired
trials, and the verdicts in ``artifacts/influence/substrate_trials.json`` are
what forty trials each produced: all four INERT, with the whole confidence
interval under the noise floor of two intact runs.

INERT is a result. It says a temperature shift of −0.18 to +0.17 and a token
budget moved by up to 56 does not displace a 1.5B's answer further than
free-running sampling already does. What it is not is a verdict at the 27B,
and the count is held apart from the live one so that it cannot be read as
one.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.verify.what_has_a_measured_effect import (
    how_much_is_measured,
    the_baseline,
    the_declared_lesions,
    what_the_substrate_trials_found,
)

RECEIPT = Path(__file__).resolve().parents[1] / "artifacts" / "influence" / "substrate_trials.json"


def test_the_receipt_exists_and_names_its_substrate():
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["substrate"], "a verdict with no substrate is not a verdict"
    assert payload["trials_per_channel"] >= 40
    assert payload["what_this_is_not"], "the boundary has to be written down"


def test_every_reported_verdict_came_from_completed_trials():
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    for channel, finding in payload["findings"].items():
        verdict = finding["verdict"]
        if verdict["verdict"] in {"inert", "influential"}:
            assert finding["trials_completed"] >= 40, channel
            assert verdict["n_treatment"] >= 40, channel
            assert verdict["n_null"] >= 40, channel


def test_a_lesion_that_moved_nothing_could_not_have_been_measured():
    """Each measured channel must have changed the sampling it was read into.

    This is the check that would have caught the live campaign before it ran
    for three weeks: a channel whose intact and lesioned arms sample
    identically has no counterfactual, and a verdict from it is a verdict
    about nothing.
    """
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    for channel, finding in payload["findings"].items():
        intact = finding["intact_sampling"]
        lesioned = finding["lesioned_sampling"]
        moved = (
            abs(intact["temperature"] - lesioned["temperature"]) > 1e-6
            or intact["max_tokens"] != lesioned["max_tokens"]
        )
        assert moved, f"{channel}: the lesion changed nothing the sampler reads"


def test_four_channels_are_measured_at_the_substrate():
    found = what_the_substrate_trials_found()
    assert found["measured"] >= 4
    assert set(found["channels"]) >= {
        "affect.circumplex_sampling",
        "spiking.sampling_bias",
        "imagination.sampling_bias",
        "bicameral.sampling_bias",
    }


def test_the_substrate_count_is_not_folded_into_the_live_one():
    """A 1.5B must not discharge the 27B's obligation."""
    counts = how_much_is_measured()
    assert counts["measured"] == 0
    assert counts["measured_at_substrate"] >= 4
    assert counts["substrate"]
    assert counts["measured_at_substrate_not_live"]


def test_the_loop_registered_channels_are_declared():
    """Three sampling biases register in a ``for`` over a literal tuple.

    The argument at the call is a bare name, which the counter refuses, so
    they were registered in the source, registered at runtime and missing
    from the declared count.
    """
    declared = set(the_declared_lesions())
    assert {
        "SPIKING_SAMPLING_BIAS",
        "IMAGINATION_SAMPLING_BIAS",
        "BICAMERAL_SAMPLING_BIAS",
    } <= declared


def test_the_decorator_form_is_declared_too():
    assert "AFFECT_GENERATION_CONTROLS" in set(the_declared_lesions())


def test_the_baseline_never_walks_back_a_verdict():
    held = the_baseline()
    now = how_much_is_measured()
    assert now["declared_lesions"] >= held["declared_lesions"]
    assert now["measured"] >= held["measured"]
    assert now["measured_at_substrate"] >= held["measured_at_substrate"]


def test_the_gate_site_is_behind_a_foreground_guard():
    """The fact that explains three weeks of zeros, held by a test.

    If somebody removes that guard the circumplex becomes reachable from a
    background call and this fails, which is the right way round: the claim
    is about the source, so the source is what checks it.
    """
    from core.verify.which_lesions_a_direct_call_can_bite import where_each_channel_acts

    sites = {one.channel: one for one in where_each_channel_acts(str(RECEIPT.parents[2]))}
    circumplex = sites["affect.circumplex_sampling"]
    assert "core/brain/inference_gate.py" in circumplex.applied_in
    assert "core/brain/inference_gate.py" in circumplex.foreground_only_in
    assert not circumplex.reachable_by("a background call through the gate")


def test_a_background_call_through_the_gate_can_bite_nothing():
    """Which is why the hourly job must not spend generations on it."""
    from core.verify.which_lesions_a_direct_call_can_bite import (
        what_a_background_gate_call_can_bite,
    )

    assert what_a_background_gate_call_can_bite(str(RECEIPT.parents[2])) == ()


def test_the_campaign_job_refuses_a_channel_it_cannot_move():
    import asyncio

    from core.container import ServiceContainer
    from core.runtime.autonomy_conductor import AutonomyConductor
    from core.verify import influence_campaign
    from core.verify.lesion_registry import LesionHandle, get_lesion_registry
    from core.verify.why_the_campaign_did_not_run import (
        forget_everything,
        how_the_campaign_has_gone,
    )

    registry = get_lesion_registry()
    registry.register(
        LesionHandle(
            channel="test.unreachable_channel",
            lesion=lambda: None,
            restore=lambda: None,
            owner="test_four_channels_finally_have_a_verdict",
            neutral_description="nothing",
            direct_actuation=True,
        ),
        replace=True,
    )

    class _Gate:
        async def generate(self, *_args, **_kwargs):  # pragma: no cover - never called
            raise AssertionError("a channel this job cannot move must cost no generations")

    forget_everything()
    original = influence_campaign.campaign_admission_reason
    influence_campaign.campaign_admission_reason = lambda **_: ""
    ServiceContainer.register("inference_gate", _Gate())
    try:
        result = asyncio.run(AutonomyConductor()._job_influence_campaign())
    finally:
        influence_campaign.campaign_admission_reason = original
        registry.unregister("test.unreachable_channel")

    assert result["status"] == "unreachable"
    assert "test.unreachable_channel" in result["unreachable"]
    assert how_the_campaign_has_gone()["counts"].get("unreachable") == 1
