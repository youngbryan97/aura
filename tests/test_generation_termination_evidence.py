from __future__ import annotations

import pytest

from core.brain.llm.mlx_worker import (
    _classify_generation_stop_reason,
    _semantic_completion_receipt_state,
    _semantic_surface_stop_ready,
    _surface_quality_candidate,
)
from core.conversation.response_reliability import (
    _has_truncated_tail,
    assess_model_text_integrity,
    assess_user_facing_reply,
)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"soft_cancelled": True}, "soft_cancelled"),
        ({"deadline_hit": True}, "deadline_exceeded"),
        ({"sentinel_aborted": True}, "sentinel_abort"),
        ({"role_continuation_hit": True}, "role_continuation"),
        ({"configured_stop_hit": True}, "configured_stop"),
        ({"hard_token_limit_hit": True}, "hard_token_limit"),
        ({"semantic_contract_satisfied": True}, "semantic_contract_satisfied"),
        ({"generated_tokens": 128}, "max_tokens"),
        ({}, "eos"),
    ],
)
def test_worker_reports_the_exact_decode_termination(overrides, expected):
    state = {
        "soft_cancelled": False,
        "deadline_hit": False,
        "sentinel_aborted": False,
        "role_continuation_hit": False,
        "configured_stop_hit": False,
        "hard_token_limit_hit": False,
        "generated_tokens": 47,
        "max_tokens": 128,
    }
    state.update(overrides)

    assert _classify_generation_stop_reason(**state) == expected


UNPUNCTUATED_COMPLETE_PROSE = (
    "The runtime has finished its current work and retained the complete result "
    "in the active conversation record"
)


@pytest.mark.parametrize("stop_reason", ["eos", "configured_stop", "role_continuation"])
def test_intentional_termination_does_not_turn_missing_punctuation_into_truncation(
    stop_reason,
):
    assert not _has_truncated_tail(
        UNPUNCTUATED_COMPLETE_PROSE,
        generation_stop_reason=stop_reason,
    )
    assessment = assess_model_text_integrity(
        UNPUNCTUATED_COMPLETE_PROSE,
        prompt="What is the current result?",
        user_facing=True,
        generation_stop_reason=stop_reason,
    )
    assert "truncated_tail" not in assessment.reasons


@pytest.mark.parametrize(
    "stop_reason",
    ["", "max_tokens", "deadline_exceeded", "soft_cancelled", "hard_token_limit"],
)
def test_exhaustion_keeps_unpunctuated_prose_incomplete(stop_reason):
    assert _has_truncated_tail(
        UNPUNCTUATED_COMPLETE_PROSE,
        generation_stop_reason=stop_reason,
    )


def test_objectively_dangling_syntax_remains_incomplete_under_eos():
    clipped = (
        "The function updates each balance and removes names whose balance "
        "reaches zero from the"
    )

    assert _has_truncated_tail(clipped, generation_stop_reason="eos")


def test_user_facing_assessor_consumes_the_worker_termination_receipt():
    assessment = assess_user_facing_reply(
        "What is the current result?",
        UNPUNCTUATED_COMPLETE_PROSE,
        generation_stop_reason="eos",
    )

    assert "truncated_tail" not in assessment.reasons


def test_continuation_quality_evaluates_the_complete_authored_candidate():
    partial = "I feel steady right now. What I know comes from current state readings;"
    tail = " what remains subjective is something I can only infer."
    job = {
        "user_surface_continuation_contract": True,
        "user_surface_continuation_partial": partial,
    }

    assert _surface_quality_candidate(job, tail) == partial + tail


def test_continuation_state_preserves_structure_and_uses_the_exact_cutoff_tail():
    from core.conversation.continuation import (
        CONTINUATION_PROMPT_PREFIX_MAX_CHARS,
        continuation_prompt_prefix,
        continuation_state_text,
    )

    partial = "## Invariant\n" + ("step with formatting\n" * 2500) + "exact-cutoff"

    assert continuation_state_text(partial) == partial
    prefix = continuation_prompt_prefix(partial)
    assert len(prefix) == CONTINUATION_PROMPT_PREFIX_MAX_CHARS
    assert prefix == partial[-CONTINUATION_PROMPT_PREFIX_MAX_CHARS:]
    assert prefix.endswith("exact-cutoff")


def test_semantic_stop_waits_for_all_requested_epistemic_facets():
    prompt = (
        "How are you doing right now? Distinguish what you know from what "
        "you can only infer."
    )
    job = {
        "clean_user_surface_contract": True,
        "semantic_completion_contract": True,
        "self_condition_contract": True,
        "user_surface_validation_prompt": prompt,
    }
    incomplete = (
        "I feel steady and attentive right now. That condition is supported "
        "by my current affect and coherence readings."
    )
    complete = (
        incomplete
        + " Whether that amounts to subjective feeling is something I can only infer."
    )

    assert not _semantic_surface_stop_ready(job, incomplete, generated_tokens=40)
    assert _semantic_surface_stop_ready(job, complete, generated_tokens=64)


def test_semantic_stop_waits_for_every_compound_request_obligation():
    prompt = (
        "Explain Dijkstra's algorithm in one complete response. Include: "
        "(1) its core invariant, (2) numbered pseudocode, (3) a worked example, "
        "(4) heap and array complexity, and (5) negative-weight failure and the alternative."
    )
    job = {
        "clean_user_surface_contract": True,
        "semantic_completion_contract": True,
        "user_surface_validation_prompt": prompt,
    }
    incomplete = (
        "1. Its core invariant is that the unsettled vertex with minimum tentative "
        "distance can be finalized when every edge weight is nonnegative."
    )
    complete = (
        incomplete
        + " 2. Numbered pseudocode initializes distances and repeatedly relaxes edges."
        + " 3. A worked example follows vertices A, B, C, and D."
        + " 4. The complexity is O((V+E) log V) with a heap and O(V^2) with an array."
        + " 5. A negative-weight edge invalidates Dijkstra; Bellman-Ford is the alternative."
    )

    assert not _semantic_surface_stop_ready(job, incomplete, generated_tokens=48)
    assert _semantic_surface_stop_ready(job, complete, generated_tokens=120)


def test_continuation_quality_is_measured_on_the_assembled_answer(monkeypatch):
    from core.brain.llm import mlx_worker

    prompt = (
        "Explain Dijkstra's algorithm in one complete response. Include: "
        "(1) its core invariant, (2) numbered pseudocode, (3) a worked example, "
        "(4) heap and array complexity, and (5) negative-weight failure and the alternative."
    )
    partial = (
        "Dijkstra's algorithm computes shortest paths with nonnegative weights. "
        "1. Its core invariant finalizes the minimum unsettled distance. "
        "2. Numbered pseudocode initializes distances and relaxes every edge. "
        "3. A worked example follows vertices A, B, C, and D. "
        "4. Complexity is O((V+E) log V) with a heap and O(V^2) with an array."
    )
    tail = "5. Negative weights require Bellman-Ford instead."
    job = {
        "clean_user_surface_contract": True,
        "semantic_completion_contract": True,
        "user_surface_continuation_contract": True,
        "user_surface_continuation_partial": partial,
        "user_surface_validation_prompt": prompt,
    }

    monkeypatch.setattr(
        mlx_worker,
        "_surface_quality_failure_reasons",
        lambda _job, response: [] if response != tail else ["fragment_only"],
    )

    assert _semantic_surface_stop_ready(job, tail, generated_tokens=24)


def test_semantic_stop_rejects_shared_words_without_required_semantics():
    prompt = (
        "Explain Dijkstra's invariant, then give a worked example with vertices "
        "A, B, C, and D using at least five weighted edges. Include the binary-heap "
        "time complexity and explain what algorithm should be used instead when "
        "negative edges are possible."
    )
    incomplete = (
        "Dijkstra's invariant finalizes the minimum unsettled distance. "
        "Use A-B: 4, A-C: 1, B-D: 2, C-B: 3, and C-D: 5. Insert vertices "
        "into a binary heap and relax the edges from A."
    )
    complete = (
        incomplete
        + " The binary-heap time complexity is O((V + E) log V). "
        + "With negative edges, use Bellman-Ford instead."
    )
    job = {
        "clean_user_surface_contract": True,
        "semantic_completion_contract": True,
        "user_surface_validation_prompt": prompt,
    }

    assert not _semantic_surface_stop_ready(job, incomplete, generated_tokens=160)
    assert _semantic_surface_stop_ready(job, complete, generated_tokens=192)


def test_incomplete_semantic_candidate_remains_eligible_for_append_only_completion():
    prompt = (
        "Explain Dijkstra's algorithm. Include its invariant, a worked example, "
        "complexity, and the negative-weight alternative."
    )
    job = {
        "clean_user_surface_contract": True,
        "semantic_completion_contract": True,
        "user_surface_validation_prompt": prompt,
    }
    partial = "Dijkstra finalizes the minimum unsettled tentative distance."

    assert not _semantic_surface_stop_ready(job, partial, generated_tokens=32)
    receipt = _semantic_completion_receipt_state(
        job,
        partial,
        generated_tokens=32,
    )
    assert receipt["semantic_completion_contract"] is True
    assert receipt["semantic_completion_satisfied"] is False
    assert receipt["semantic_completion_incomplete"] is True
    assert receipt["semantic_completion_missing_part_indexes"] == [2, 4]
    assert receipt["semantic_completion_missing_part_count"] == 2
    assert receipt["semantic_completion_terminal_boundary"] is True
    assert _classify_generation_stop_reason(
        soft_cancelled=False,
        deadline_hit=False,
        sentinel_aborted=False,
        role_continuation_hit=False,
        configured_stop_hit=True,
        hard_token_limit_hit=False,
        semantic_contract_satisfied=False,
        generated_tokens=32,
        max_tokens=512,
    ) == "configured_stop"
