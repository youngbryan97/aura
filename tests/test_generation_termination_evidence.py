from __future__ import annotations

import pytest

from core.brain.llm.mlx_worker import (
    _classify_generation_stop_reason,
    _continuation_resume_should_bind,
    _continuation_resume_unavailable_reason,
    _conversation_resume_boundary_complete,
    _job_requires_exact_continuation_cache,
    _semantic_completion_receipt_state,
    _semantic_surface_stop_ready,
    _semantic_terminal_grace_eligible,
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


def test_complete_deadline_answer_does_not_claim_resume_failure() -> None:
    assert (
        _continuation_resume_unavailable_reason(
            resume_required=False,
            cache_lru_available=False,
            cache_disabled=True,
            final_cache_available=False,
            sentinel_aborted=False,
            response_present=True,
        )
        == ""
    )


def test_cache_bypass_does_not_disable_exact_transaction_continuation() -> None:
    assert _job_requires_exact_continuation_cache(
        {
            "disable_prompt_cache": True,
            "semantic_completion_contract": True,
        }
    )
    assert _job_requires_exact_continuation_cache(
        {
            "disable_prompt_cache": True,
            "user_surface_continuation_resume_handle": "d" * 32,
        }
    )
    assert _job_requires_exact_continuation_cache(
        {
            "clean_user_surface_contract": True,
        }
    )
    assert _job_requires_exact_continuation_cache(
        {
            "user_surface_conversation_resume_handle": "e" * 32,
        }
    )


def test_unrelated_cache_bypass_does_not_allocate_continuation_state() -> None:
    assert not _job_requires_exact_continuation_cache(
        {
            "disable_prompt_cache": True,
            "health_probe": True,
        }
    )


def test_incomplete_deadline_answer_names_missing_resume_cache() -> None:
    assert (
        _continuation_resume_unavailable_reason(
            resume_required=True,
            cache_lru_available=False,
            cache_disabled=False,
            final_cache_available=False,
            sentinel_aborted=False,
            response_present=True,
        )
        == "cache_lru_unavailable"
    )


@pytest.mark.parametrize(
    ("stop_reason", "semantic_incomplete", "expected"),
    [
        ("semantic_contract_satisfied", False, True),
        ("semantic_contract_satisfied", True, True),
        ("deadline_exceeded", True, True),
        ("max_tokens", True, True),
        ("soft_cancelled", True, True),
        ("deadline_exceeded", False, False),
        ("eos", True, False),
        ("configured_stop", True, False),
        ("sentinel_abort", True, False),
    ],
)
def test_worker_retains_exact_state_at_only_resumable_boundaries(
    stop_reason: str,
    semantic_incomplete: bool,
    expected: bool,
) -> None:
    assert (
        _continuation_resume_should_bind(
            generation_stop_reason=stop_reason,
            semantic_completion_incomplete=semantic_incomplete,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("eos", True),
        ("configured_stop", False),
        ("role_continuation", False),
        ("semantic_contract_satisfied", False),
        ("deadline_exceeded", False),
        ("max_tokens", False),
        ("soft_cancelled", False),
    ],
)
def test_new_conversation_turn_requires_a_native_assistant_boundary(
    stop_reason: str,
    expected: bool,
) -> None:
    assert _conversation_resume_boundary_complete(stop_reason) is expected


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


def test_cognitive_engine_uses_the_full_shared_completion_assessment():
    from core.brain.cognitive_engine import _truncation_verdict

    clipped = (
        "One consequence I can distinguish from expectation is that the live "
        "runtime now reports a smaller resident-model footprint, while"
    )

    assert _truncation_verdict(clipped, generation_stop_reason="eos")


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


def test_a_quoted_word_does_not_end_the_surrounding_sentence() -> None:
    """A closing quote is syntax, not terminal punctuation."""
    from core.language.terminal_boundary import has_terminal_sentence_boundary

    prompt = (
        "What does Dijkstra's shortest-path algorithm guarantee, and why do "
        "negative edge weights break that guarantee?"
    )
    live_cutoff = (
        "Dijkstra finalizes the smallest tentative distance because nonnegative "
        "edges cannot later reduce it. Negative weights can reduce a settled "
        "distance, so the greedy choice is only safe when “closest”"
    )
    job = {
        "clean_user_surface_contract": True,
        "semantic_completion_contract": True,
        "user_surface_validation_prompt": prompt,
    }

    assert not has_terminal_sentence_boundary(live_cutoff)
    assert has_terminal_sentence_boundary('The invariant is called “safe.”')
    assert has_terminal_sentence_boundary(
        'The invariant holds (when weights are nonnegative.)'
    )
    assert not _semantic_surface_stop_ready(job, live_cutoff, generated_tokens=200)
    receipt = _semantic_completion_receipt_state(job, live_cutoff, generated_tokens=200)
    assert receipt["semantic_completion_satisfied"] is False
    assert receipt["semantic_completion_terminal_boundary"] is False
    assert _has_truncated_tail(
        live_cutoff,
        generation_stop_reason="semantic_contract_satisfied",
    )


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


def test_semantic_stop_accepts_numbered_sections_after_fenced_pseudocode():
    prompt = (
        "Explain Dijkstra's algorithm in one complete response. Include: "
        "(1) its core invariant, (2) numbered pseudocode, (3) a worked example "
        "with at least five weighted edges, (4) heap and array complexity, and "
        "(5) negative-weight failure and the alternative. Do not stop mid-sentence "
        "or omit a requested part."
    )
    answer = """Dijkstra computes shortest paths on nonnegative graphs.
## (1) Core invariant
The core invariant finalizes the minimum unsettled distance.
## (2) Numbered pseudocode
```text
1. Initialize distances.
2. Extract the minimum.
3. Relax outgoing edges.
4. Repeat.
```
## (3) Worked example
Use A-B: 4, A-C: 2, C-B: 1, B-D: 5, and C-D: 8. The final distance to D is 8.
## (4) Complexity
A binary heap takes O((V+E) log V), while an array takes O(V^2 + E).
## (5) Failure and alternative
Negative weights break the invariant; use Bellman-Ford instead.
"""
    job = {
        "clean_user_surface_contract": True,
        "semantic_completion_contract": True,
        "user_surface_validation_prompt": prompt,
    }

    assert _semantic_surface_stop_ready(job, answer, generated_tokens=160)


def test_deadline_terminal_grace_requires_all_semantics_except_final_boundary():
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
    complete_without_boundary = (
        "1. Its core invariant is that the unsettled vertex with minimum tentative "
        "distance can be finalized when every edge weight is nonnegative. "
        "2. Numbered pseudocode initializes distances and repeatedly relaxes edges. "
        "3. A worked example follows vertices A, B, C, and D. "
        "4. The complexity is O((V+E) log V) with a heap and O(V^2) with an array. "
        "5. A negative-weight edge invalidates Dijkstra; Bellman-Ford is the alternative"
    )
    still_missing_work = (
        "1. The invariant finalizes the minimum unsettled distance. "
        "2. Numbered pseudocode initializes distances"
    )

    assert _semantic_terminal_grace_eligible(
        job,
        complete_without_boundary,
        generated_tokens=120,
    )
    assert not _semantic_terminal_grace_eligible(
        job,
        still_missing_work,
        generated_tokens=48,
    )


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


def test_post_generation_completion_accepts_a_short_terminal_answer():
    job = {
        "clean_user_surface_contract": True,
        "semantic_completion_contract": True,
        "user_surface_validation_prompt": "Are you ready?",
    }

    receipt = _semantic_completion_receipt_state(
        job,
        "Yes.",
        generated_tokens=2,
    )

    assert receipt["semantic_completion_satisfied"] is True
    assert receipt["semantic_completion_incomplete"] is False


def test_natural_eos_accepts_a_complete_unpunctuated_entity_answer():
    job = {
        "clean_user_surface_contract": True,
        "semantic_completion_contract": True,
        "user_surface_validation_prompt": "Who wrote the novel Solaris?",
    }

    receipt = _semantic_completion_receipt_state(
        job,
        "Stanisław Lem",
        generated_tokens=5,
        generation_stop_reason="eos",
    )

    assert receipt["semantic_completion_satisfied"] is True
    assert receipt["semantic_completion_incomplete"] is False
    assert receipt["semantic_completion_terminal_boundary"] is False
    assert receipt["semantic_completion_eos_boundary"] is True


def test_natural_eos_does_not_accept_dangling_syntax():
    job = {
        "clean_user_surface_contract": True,
        "semantic_completion_contract": True,
        "user_surface_validation_prompt": "Name the latest release and cite its source.",
    }

    receipt = _semantic_completion_receipt_state(
        job,
        "The latest release is",
        generated_tokens=5,
        generation_stop_reason="eos",
    )

    assert receipt["semantic_completion_satisfied"] is False
    assert receipt["semantic_completion_incomplete"] is True
    assert receipt["semantic_completion_eos_boundary"] is False


def test_post_generation_completion_rejects_a_clipped_short_answer():
    job = {
        "clean_user_surface_contract": True,
        "semantic_completion_contract": True,
        "user_surface_validation_prompt": "Name the latest release and cite its source.",
    }

    receipt = _semantic_completion_receipt_state(
        job,
        "The latest release is",
        generated_tokens=5,
    )

    assert receipt["semantic_completion_satisfied"] is False
    assert receipt["semantic_completion_incomplete"] is True
    assert receipt["semantic_completion_terminal_boundary"] is False


def test_semantic_stop_waits_for_the_answers_own_declared_items() -> None:
    job = {
        "clean_user_surface_contract": True,
        "semantic_completion_contract": True,
        "user_surface_validation_prompt": "Do you know why that broke?",
    }
    partial = (
        "The break is likely one of two things:\n"
        "1. The automation timed out while waiting for verification."
    )
    complete = partial + "\n2. The application lookup never found an installed target."

    assert not _semantic_surface_stop_ready(job, partial, generated_tokens=88)
    assert _semantic_surface_stop_ready(job, complete, generated_tokens=112)
    assert _has_truncated_tail(partial)
    assert not _has_truncated_tail(complete)

    receipt = _semantic_completion_receipt_state(
        job,
        partial,
        generated_tokens=88,
    )
    assert receipt["semantic_completion_satisfied"] is False
    assert receipt["semantic_completion_incomplete"] is True
    assert receipt["semantic_completion_unfulfilled_discourse_count"] == 1
    assert receipt["semantic_completion_unfulfilled_discourse"] == [
        {
            "expected_count": 2,
            "observed_count": 1,
            "kind": "things",
            "declaration": "one of two things:",
        }
    ]


def test_semantic_stop_waits_for_both_sides_of_one_comparison() -> None:
    job = {
        "clean_user_surface_contract": True,
        "semantic_completion_contract": True,
        "user_surface_validation_prompt": (
            "What is the difference between a prism and a mirror?"
        ),
    }
    partial = (
        "A mirror is a flat coated surface that reflects light according to "
        "the law of reflection: angle in equals angle out."
    )
    complete = (
        partial
        + " A prism instead refracts light at its surfaces and can disperse it "
        "into component wavelengths."
    )

    assert not _semantic_surface_stop_ready(job, partial, generated_tokens=24)
    assert _semantic_surface_stop_ready(job, complete, generated_tokens=48)
