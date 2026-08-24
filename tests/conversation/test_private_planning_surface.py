"""Private composition work never becomes Aura's visible answer."""

from __future__ import annotations

from core.conversation.response_reliability import (
    assess_user_facing_reply,
    strip_private_planning_prefix,
)
from core.language.answer_surface import split_private_planning_prefix

PRIVATE_PLAN = (
    "We need answer user's request. Need explain Dijkstra in one complete "
    "response. We must include the invariant, pseudocode, a worked example, "
    "both complexities, and the negative-weight alternative."
)
PUBLIC_ANSWER = (
    "Dijkstra's algorithm settles the unvisited vertex with the smallest "
    "tentative distance; with non-negative edges, that distance is final."
)


def test_repeated_composition_plan_is_separated_from_exact_public_suffix() -> None:
    generated = f"{PRIVATE_PLAN}\n\n{PUBLIC_ANSWER}"

    split = split_private_planning_prefix(generated)

    assert split.separated
    assert split.planning_units >= 2
    assert split.public_answer == PUBLIC_ANSWER
    assert strip_private_planning_prefix(generated) == PUBLIC_ANSWER


def test_private_plan_is_an_internal_surface_leak_before_separation() -> None:
    generated = f"{PRIVATE_PLAN}\n\n{PUBLIC_ANSWER}"

    assessment = assess_user_facing_reply("Explain Dijkstra.", generated)

    assert "internal_task_prompt_leak" in assessment.reasons


def test_one_public_metalinguistic_statement_is_not_cut() -> None:
    public = (
        "We need to answer that question carefully because the available "
        "evidence conflicts. I would start by verifying the source receipt."
    )

    assert not split_private_planning_prefix(public).separated
    assert strip_private_planning_prefix(public) == public


def test_plan_without_an_authored_answer_is_not_misrepresented_as_an_answer() -> None:
    split = split_private_planning_prefix(PRIVATE_PLAN)

    assert split.planning_units >= 2
    assert not split.separated
    assert split.public_answer == ""
    assert strip_private_planning_prefix(PRIVATE_PLAN) == PRIVATE_PLAN


def test_code_and_math_answer_bytes_are_preserved() -> None:
    answer = "### Result\n\n```python\ndist[u] = 0\n```\n\nThe cost is $O(E \\log V)$."
    generated = f"{PRIVATE_PLAN}\n\n{answer}"

    assert strip_private_planning_prefix(generated) == answer
