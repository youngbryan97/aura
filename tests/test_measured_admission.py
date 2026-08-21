"""Admit work by what the machine does, not by what a formula allows.

The incidents behind this: a caller with a ten-second timeout allocated
fifteen seconds of work; expected foreground contention escalated into a
failure lockdown; a prompt cache that was never built and then cleared
every turn made conversation cost grow quadratically while the policy saw
nothing wrong.
"""
from __future__ import annotations

import pytest

from core.brain.llm.measured_admission import (
    AdmissionOutcome,
    Confidence,
    TaskShape,
    ThroughputEstimator,
    admit,
    measured_slack_seconds,
    recommended_completion_tokens,
    recommended_foreground_deadline,
)


@pytest.fixture(autouse=True)
def _fresh_estimator(monkeypatch):
    """A shared estimator across tests would leak measurements between them."""
    import core.brain.llm.measured_admission as module

    monkeypatch.setattr(module, "_ESTIMATOR", ThroughputEstimator())


def _teach(seconds_per_token: float, *, samples: int = 30, **over) -> None:
    """Give the estimator a machine with a known decode rate."""
    from core.brain.llm.measured_admission import record_generation

    payload = dict(
        model="q32b",
        prompt_tokens=1200,
        generated_tokens=500,
        prefill_seconds=0.5,
        decode_seconds=500 * seconds_per_token,
        overhead_seconds=0.2,
    )
    payload.update(over)
    for _ in range(samples):
        record_generation(**payload)


# ------------------------------------------------------- knowing what it knows


def test_an_unmeasured_shape_says_so_instead_of_inventing_a_number():
    decision = admit(
        model="never-seen",
        prompt_tokens=1000,
        requested_decode_tokens=400,
        deadline_seconds=60.0,
    )
    assert decision.confidence is Confidence.NO_SAMPLES
    assert decision.samples == 0


def test_an_unmeasured_shape_is_admitted_cautiously_not_confidently():
    """The conservative estimate must actually bind, or caution is decorative."""
    generous = admit(
        model="never-seen", prompt_tokens=1000, requested_decode_tokens=400,
        deadline_seconds=600.0,
    )
    tight = admit(
        model="never-seen", prompt_tokens=1000, requested_decode_tokens=400,
        deadline_seconds=5.0,
    )
    assert generous.outcome is AdmissionOutcome.ADMIT
    assert tight.outcome is not AdmissionOutcome.ADMIT


def test_unmeasured_prior_distinguishes_resident_32b_from_small_model():
    small = admit(
        model="Qwen2.5-1.5B-Instruct-4bit",
        prompt_tokens=2048,
        requested_decode_tokens=512,
        deadline_seconds=60.0,
    )
    resident = admit(
        model="Qwen2.5-32B-Instruct-4bit",
        prompt_tokens=2048,
        requested_decode_tokens=512,
        deadline_seconds=60.0,
    )

    assert small.confidence is Confidence.NO_SAMPLES
    assert resident.confidence is Confidence.NO_SAMPLES
    assert resident.predicted_seconds > small.predicted_seconds * 5
    assert resident.outcome is not AdmissionOutcome.ADMIT


def test_foreground_deadline_covers_cold_resident_decode_before_measurement():
    deadline, confidence, samples = recommended_foreground_deadline(
        model="Qwen2.5-32B-Instruct-4bit",
        prompt_tokens=2048,
        decode_tokens=512,
        minimum_seconds=112.0,
        maximum_seconds=240.0,
    )

    assert 190.0 <= deadline <= 240.0
    assert confidence is Confidence.NO_SAMPLES
    assert samples == 0


def test_foreground_deadline_converges_to_measured_throughput():
    _teach(0.012, model="Qwen2.5-32B-Instruct-4bit", prompt_tokens=2048)

    deadline, confidence, samples = recommended_foreground_deadline(
        model="Qwen2.5-32B-Instruct-4bit",
        prompt_tokens=2048,
        decode_tokens=512,
        minimum_seconds=112.0,
        maximum_seconds=240.0,
    )

    assert deadline == 112.0
    assert confidence is Confidence.MEASURED
    assert samples == 30


def test_completion_length_uses_prior_until_observed() -> None:
    tokens, confidence, samples = recommended_completion_tokens(
        model="Qwen2.5-32B-Instruct-4bit",
        prompt_tokens=2048,
        maximum_tokens=1920,
        prior_tokens=1280,
    )

    assert tokens == 1280
    assert confidence is Confidence.NO_SAMPLES
    assert samples == 0


def test_completion_length_converges_to_observed_p90() -> None:
    from core.brain.llm.measured_admission import record_generation

    for generated in (*([420] * 28), 500, 540):
        record_generation(
            model="Qwen2.5-32B-Instruct-4bit",
            prompt_tokens=2048,
            generated_tokens=generated,
            prefill_seconds=0.5,
            decode_seconds=generated * 0.02,
        )

    tokens, confidence, samples = recommended_completion_tokens(
        model="Qwen2.5-32B-Instruct-4bit",
        prompt_tokens=2048,
        maximum_tokens=1920,
        prior_tokens=1280,
    )

    assert tokens == 420
    assert confidence is Confidence.MEASURED
    assert samples == 30


def test_it_gets_less_wrong_as_it_measures():
    before = admit(
        model="q32b", prompt_tokens=1200, requested_decode_tokens=500,
        deadline_seconds=30.0,
    )
    _teach(0.012)  # 500 tokens in ~6s
    after = admit(
        model="q32b", prompt_tokens=1200, requested_decode_tokens=500,
        deadline_seconds=30.0,
    )
    assert before.confidence is Confidence.NO_SAMPLES
    assert after.confidence is Confidence.MEASURED
    assert after.predicted_seconds < before.predicted_seconds
    assert after.outcome is AdmissionOutcome.ADMIT


def test_a_slower_machine_allocates_less():
    _teach(0.10)  # 500 tokens would take ~50s
    decision = admit(
        model="q32b", prompt_tokens=1200, requested_decode_tokens=500,
        deadline_seconds=30.0,
    )
    assert decision.outcome is AdmissionOutcome.DOWNGRADE
    assert decision.granted_decode_tokens < 500


def test_sparse_samples_widen_rather_than_pretend_precision():
    _teach(0.012, samples=2)
    decision = admit(
        model="q32b", prompt_tokens=1200, requested_decode_tokens=500,
        deadline_seconds=30.0,
    )
    assert decision.confidence is Confidence.SPARSE
    assert decision.samples == 2


# ------------------------------------------------------------ the real refusals


def test_a_ten_second_caller_is_never_given_fifteen_seconds_of_work():
    """The measured incident, as an assertion."""
    _teach(0.030)  # 500 tokens ≈ 15s
    decision = admit(
        model="q32b", prompt_tokens=1200, requested_decode_tokens=500,
        deadline_seconds=10.0,
    )
    assert decision.outcome is not AdmissionOutcome.ADMIT
    if decision.outcome is AdmissionOutcome.DOWNGRADE:
        per_token = decision.predicted_seconds / 500
        assert decision.granted_decode_tokens * per_token <= 10.0


def test_decode_is_never_queued_against_a_loading_model():
    """What turned one slow answer into a cluster of 503s."""
    _teach(0.001)
    decision = admit(
        model="q32b", prompt_tokens=100, requested_decode_tokens=64,
        deadline_seconds=600.0, model_loading=True,
    )
    assert decision.outcome is AdmissionOutcome.REFUSE
    assert decision.reason == "model_is_loading"


def test_a_request_that_cannot_sit_beside_the_resident_model_is_refused():
    _teach(0.001)
    decision = admit(
        model="q32b", prompt_tokens=100, requested_decode_tokens=64,
        deadline_seconds=600.0,
        free_memory_bytes=8 * 1024**3,
        model_footprint_bytes=20 * 1024**3,
    )
    assert decision.outcome is AdmissionOutcome.REFUSE
    assert "memory" in decision.reason


def test_a_caller_that_declares_no_deadline_is_refused_not_guessed_for():
    decision = admit(
        model="q32b", prompt_tokens=100, requested_decode_tokens=64,
        deadline_seconds=0.0,
    )
    assert decision.outcome is AdmissionOutcome.REFUSE
    assert decision.reason == "no_deadline_declared"


def test_queue_depth_counts_against_the_deadline():
    """Latency the caller experiences but this request does not spend."""
    _teach(0.012)
    alone = admit(
        model="q32b", prompt_tokens=1200, requested_decode_tokens=500,
        deadline_seconds=20.0, queue_depth=0,
    )
    behind = admit(
        model="q32b", prompt_tokens=1200, requested_decode_tokens=500,
        deadline_seconds=20.0, queue_depth=3,
    )
    assert alone.outcome is AdmissionOutcome.ADMIT
    assert behind.outcome is not AdmissionOutcome.ADMIT


def test_nothing_can_be_admitted_with_zero_tokens_requested():
    assert admit(
        model="q32b", prompt_tokens=100, requested_decode_tokens=0,
        deadline_seconds=60.0,
    ).outcome is AdmissionOutcome.REFUSE


# ------------------------------------------------------------------ background


def test_background_work_yields_to_the_foreground_lease():
    """It takes measured slack; it does not compete and get classified after."""
    _teach(0.001)
    decision = admit(
        model="q32b", prompt_tokens=100, requested_decode_tokens=64,
        deadline_seconds=600.0, foreground=False,
        foreground_lease_held_by_other=True,
    )
    assert decision.outcome is AdmissionOutcome.REFUSE
    assert "foreground_lease" in decision.reason


def test_background_work_proceeds_when_the_foreground_is_idle():
    """The control: background cognition must not be permanently starved."""
    _teach(0.001, foreground=False)
    decision = admit(
        model="q32b", prompt_tokens=1200, requested_decode_tokens=64,
        deadline_seconds=600.0, foreground=False,
        foreground_lease_held_by_other=False,
    )
    assert decision.outcome is AdmissionOutcome.ADMIT


def test_measured_slack_is_zero_while_the_foreground_holds_the_lease():
    assert measured_slack_seconds(
        deadline_seconds=120.0, foreground_lease_held_by_other=True, queue_depth=0
    ) == 0.0
    assert measured_slack_seconds(
        deadline_seconds=120.0, foreground_lease_held_by_other=False, queue_depth=0
    ) > 0.0


def test_slack_shrinks_as_the_queue_grows():
    idle = measured_slack_seconds(
        deadline_seconds=120.0, foreground_lease_held_by_other=False, queue_depth=0
    )
    busy = measured_slack_seconds(
        deadline_seconds=120.0, foreground_lease_held_by_other=False, queue_depth=5
    )
    assert busy < idle


# ------------------------------------------------------------------- shaping


def test_a_warm_cache_is_not_mixed_with_a_cold_one():
    """The difference is large enough that mixing makes both predictions wrong."""
    _teach(0.002, cache_warm=True)
    cold = admit(
        model="q32b", prompt_tokens=1200, requested_decode_tokens=500,
        deadline_seconds=30.0, cache_warm=False,
    )
    warm = admit(
        model="q32b", prompt_tokens=1200, requested_decode_tokens=500,
        deadline_seconds=30.0, cache_warm=True,
    )
    assert cold.confidence is Confidence.NO_SAMPLES
    assert warm.confidence is Confidence.MEASURED


def test_prompt_buckets_are_coarse_enough_to_accumulate_samples():
    """A key so specific that every request is unique is no estimator at all."""
    assert TaskShape.bucket_for(100) == TaskShape.bucket_for(500)
    assert TaskShape.bucket_for(100) != TaskShape.bucket_for(5000)


def test_degenerate_observations_do_not_poison_a_shape():
    from core.brain.llm.measured_admission import get_throughput_estimator, record_generation

    for _ in range(10):
        record_generation(
            model="q32b", prompt_tokens=1200, generated_tokens=0,
            prefill_seconds=0.5, decode_seconds=0.0,
        )
    shape = TaskShape("q32b", TaskShape.bucket_for(1200), False, True)
    assert get_throughput_estimator().confidence(shape)[0] is Confidence.NO_SAMPLES


def test_the_window_is_bounded():
    _teach(0.012, samples=500)
    from core.brain.llm.measured_admission import get_throughput_estimator

    shape = TaskShape("q32b", TaskShape.bucket_for(1200), False, True)
    assert get_throughput_estimator().confidence(shape)[1] <= 64


def test_the_decision_carries_the_reasoning_to_argue_with_it():
    _teach(0.012)
    payload = admit(
        model="q32b", prompt_tokens=1200, requested_decode_tokens=500,
        deadline_seconds=30.0,
    ).to_dict()
    for key in ("outcome", "confidence", "reason", "predicted_seconds", "samples"):
        assert key in payload


# ------------------------------------------------------------- the live seam
#
# An estimator nothing feeds is a formula with extra steps. This pins that
# the real generation completion path samples into it.


def test_the_mlx_client_samples_every_completed_generation():
    """AST, not grep: a comment mentioning it must not satisfy this."""
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "core" / "brain" / "llm" / "mlx_client.py"
    ).read_text("utf-8")
    tree = ast.parse(source)

    sampler = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_record_throughput_sample"
        ),
        None,
    )
    assert sampler is not None, "the client has no throughput sampler"
    called = {
        node.func.id
        for node in ast.walk(sampler)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "record_generation" in called, (
        "the sampler does not feed the admission estimator"
    )

    # And something must actually call the sampler.
    callers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_record_throughput_sample"
    ]
    assert callers, "nothing calls _record_throughput_sample; the estimator is fed nothing"


def test_a_bad_sample_can_never_fail_a_turn_that_worked():
    """Telemetry on the completion path must be total."""
    from core.brain.llm.measured_admission import record_generation

    for bad in (
        dict(prompt_tokens=-1, generated_tokens=10, prefill_seconds=1.0, decode_seconds=1.0),
        dict(prompt_tokens=10, generated_tokens=10, prefill_seconds=float("nan"), decode_seconds=1.0),
        dict(prompt_tokens=10, generated_tokens=10, prefill_seconds=1.0, decode_seconds=float("inf")),
    ):
        record_generation(model="q32b", **bad)  # must not raise
