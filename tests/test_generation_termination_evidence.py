from __future__ import annotations

import math

import pytest

from core.brain.llm.mlx_worker import (
    _build_semantic_completion_eos_guard,
    _classify_generation_stop_reason,
    _semantic_completion_terminal_ids,
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


def test_semantic_eos_guard_blocks_termination_until_coverage_is_complete():
    mx = pytest.importorskip("mlx.core")

    class Tokenizer:
        eos_token_ids = {9}

        @staticmethod
        def decode(token_ids):
            return "".join(chr(int(token_id)) for token_id in token_ids)

    prompt = (
        "Explain Dijkstra's algorithm. Include: (1) its core invariant, "
        "(2) numbered pseudocode, (3) a worked example, (4) heap and array "
        "complexity, and (5) negative-weight failure and the alternative."
    )
    job = {
        "clean_user_surface_contract": True,
        "semantic_completion_contract": True,
        "user_surface_validation_prompt": prompt,
    }
    guard = _build_semantic_completion_eos_guard(
        Tokenizer(),
        job,
        prompt_token_count=2,
        tensor_ops=mx,
    )
    assert guard is not None

    incomplete_text = (
        "1. Its core invariant finalizes the minimum unsettled distance "
        "when edge weights are nonnegative."
    )
    incomplete = [1, 2, *map(ord, incomplete_text)]
    logits = mx.zeros((1, 128))
    blocked = guard(mx.array(incomplete), logits)
    assert math.isinf(float(blocked[0, 9]))
    assert float(blocked[0, 9]) < 0

    checkpoint_text = (
        incomplete_text
        + " The unsettled queue therefore preserves that invariant after every "
        + "successful edge relaxation, and finalized distances never decrease."
    )
    checkpoint = [1, 2, *map(ord, checkpoint_text)]
    checkpoint_logits = mx.zeros((1, 128))
    checkpoint_logits[0, 9] = 4.0
    checkpoint_blocked = guard(mx.array(checkpoint), checkpoint_logits)
    assert math.isinf(float(checkpoint_blocked[0, 9]))
    assert float(checkpoint_blocked[0, 9]) < 0

    complete_text = (
        incomplete_text
        + " 2. Numbered pseudocode initializes distances and repeatedly relaxes edges."
        + " 3. A worked example follows vertices A, B, C, and D."
        + " 4. The complexity is O((V+E) log V) with a heap and O(V^2) with an array."
        + " 5. A negative-weight edge invalidates Dijkstra; Bellman-Ford is the alternative."
    )
    complete = [1, 2, *map(ord, complete_text)]
    allowed = guard(mx.array(complete), logits)
    assert float(allowed[0, 9]) == 0.0


def test_semantic_completion_masks_chat_protocol_terminator_not_exposed_as_eos():
    mx = pytest.importorskip("mlx.core")

    class Tokenizer:
        eos_token_id = 9

        @staticmethod
        def convert_tokens_to_ids(token):
            return {"<|im_end|>": 17, "<|im_start|>": 18}.get(token, -1)

        @staticmethod
        def decode(token_ids):
            if token_ids == [17]:
                return "<|im_end|>"
            if token_ids == [18]:
                return "<|im_start|>"
            return "".join(chr(int(token_id)) for token_id in token_ids)

    tokenizer = Tokenizer()
    assert _semantic_completion_terminal_ids(tokenizer) == (9, 17, 18)

    prompt = (
        "Explain Dijkstra's algorithm in one complete response. Include: "
        "(1) its core invariant, (2) numbered pseudocode, (3) a worked example, "
        "(4) heap and array complexity, and (5) negative-weight failure and the alternative."
    )
    guard = _build_semantic_completion_eos_guard(
        tokenizer,
        {
            "clean_user_surface_contract": True,
            "semantic_completion_contract": True,
            "user_surface_validation_prompt": prompt,
        },
        prompt_token_count=2,
        tensor_ops=mx,
    )
    assert guard is not None
    partial = [
        1,
        2,
        *map(
            ord,
            (
                "1. Its core invariant finalizes the minimum unsettled distance "
                "when edge weights are nonnegative."
            ),
        ),
    ]
    logits = mx.zeros((1, 128))
    logits[0, 17] = 4.0
    blocked = guard(mx.array(partial), logits)
    assert math.isinf(float(blocked[0, 17]))
    assert float(blocked[0, 17]) < 0
    assert math.isinf(float(blocked[0, 18]))
    assert float(blocked[0, 18]) < 0
