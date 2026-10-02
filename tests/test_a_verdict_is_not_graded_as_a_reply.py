"""Her verdict on her own result is hers, and is not graded as somebody's reply.

LIVE 2026-10-02. Her cortex wrote a 3804-character verdict on her test result.
The draft check graded it against a question read out of its own prompt; the
prompt carried her scores, so the "question" looked like arithmetic, and the
verdict was thrown away as arithmetic_answer_missing. The call had said it was
internal, but that declaration was never copied to where the check reads it.
Then the gate fell back to the brainstem, capped at 384 tokens, and its verdict
went out as hers, cut off mid-sentence, though the call required her own lane.
"""
from __future__ import annotations

import inspect

import pytest

from core.brain import inference_gate
from core.brain.inference_gate_turn_setup import _SetsTheTurnUp

pytestmark = pytest.mark.unit


def test_the_internal_declaration_reaches_the_draft_check():
    morpho, _temperature = _SetsTheTurnUp._generate_with_metadata_sink_affective_circumplex_let(
        {"internal_inference": True}
    )
    assert morpho.get("internal_inference") is True
    check = inspect.getsource(inference_gate.InferenceGate._generate_with_client)
    assert 'kwargs.get("internal_inference"' in check


def test_a_call_only_she_can_answer_is_never_handed_to_a_lower_lane():
    body = inspect.getsource(inference_gate.InferenceGate._generate_with_metadata_sink)
    refusal = body.index('"lower_lane_fallback_refused"')
    fallback = body.index("is still recovering. Falling back to")
    condition = body.rindex("if (", 0, refusal)
    assert 'context.get("own_lane_required")' in body[condition:refusal]
    assert refusal < fallback
