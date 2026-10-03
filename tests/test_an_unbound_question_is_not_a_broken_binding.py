"""A turn with no bound question is graded against the question it carries.

LIVE 2026-10-02 20:28. Asked "Bam 83 point game or Kobe 81 point game?", her
cortex wrote a 1435-character answer in 115 seconds. The worker refused it::

    Rejected live user-surface draft reasons=surface_validation_prompt_binding_version
        validation_source=unknown validation_sha256=e3b0c44298fc validation_chars=0

The MLX client puts ``{}`` in the job when its caller bound nothing, and the
contract read that empty mapping as a binding with no version. Every draft on
that path failed, the 2-bit model answered "Kobe's 81." from an 85-second
prefill, and the person waited four minutes for it.
"""

from __future__ import annotations

from core.brain.llm.mlx_worker_surface_quality import _surface_quality_failure_reasons
from core.conversation.user_surface_contract import (
    make_user_surface_prompt_binding,
    resolve_user_surface_prompt,
)

QUESTION = "Bam 83 point game or Kobe 81 point game?"
HER_ANSWER = (
    "Kobe's 81, no contest on the feeling of it. Eighty-one points in a game "
    "where you're playing four-on-five for half the second quarter, and you "
    "just don't stop. But tell me about Bam's game if you've seen it."
)


def _job(binding: object) -> dict[str, object]:
    return {
        "clean_user_surface_contract": True,
        "user_surface_validation_prompt": QUESTION,
        "user_surface_prompt_binding": binding,
        "messages": [{"role": "user", "content": QUESTION}],
    }


def test_an_empty_binding_is_no_binding() -> None:
    resolution = resolve_user_surface_prompt(_job({}))
    assert not resolution.bound
    assert resolution.valid
    assert resolution.prompt == QUESTION


def test_her_answer_is_not_refused_for_a_binding_nobody_made() -> None:
    reasons = _surface_quality_failure_reasons(_job({}), HER_ANSWER)
    assert not any(reason.startswith("surface_validation_prompt") for reason in reasons), reasons


def test_a_binding_that_is_there_and_wrong_is_still_refused() -> None:
    forged = dict(make_user_surface_prompt_binding(QUESTION, source="chat"), version=0)
    assert _surface_quality_failure_reasons(_job(forged), HER_ANSWER) == [
        "surface_validation_prompt_binding_version"
    ]
    other = make_user_surface_prompt_binding("what is 7919 times 6421?", source="chat")
    assert _surface_quality_failure_reasons(_job(other), HER_ANSWER) == [
        "surface_validation_prompt_binding_value_mismatch"
    ]
