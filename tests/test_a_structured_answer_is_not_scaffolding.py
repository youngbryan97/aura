"""Organising an answer must not destroy it.

LIVE, 2026-08-18. Asked to model out disk growth and show the numbers, she
answered: "I couldn't get to an answer I'd stand behind on that one, and I
won't send you a thinner one and pass it off as the real thing."

Nothing had failed to generate. The log shows the whole chain:

    [MLX] Worker rejected the visible draft for semantic quality
        (prompt_artifact); preserving the resident lane.
    Cortex exhausted its worker-owned semantic quality retries
    InferenceGate refused generation: kind=lanes_exhausted
        reason=worker_semantic_quality_retries_exhausted
    failure_class=user_cycle_no_response

The draft's offence was writing a structured answer. The artifact pattern
matched any line beginning `state:`, `mood:`, `goals:`, `history:`, `voice:`
or `recalled:` — which is how anyone lays out a model — so each retry produced
another well-organised draft and each was destroyed for the same reason. The
more carefully she structured the answer, the more certainly it was thrown
away, and the person got a canned apology.

One such heading is prose. A stack of them is the internal state block, which
is what the guard is actually for.
"""

from __future__ import annotations

import pytest

from core.phases.dialogue_policy import _contains_prompt_artifact

#: The shape of answer the request asked for.
STRUCTURED_ANSWER = """Here's the model.

Current state: 92% of 1.8 TB, so about 1.66 TB used.
History: over the last 6 hours it moved from 91% to 92%.
Growth: roughly 0.17 percentage points per hour.

At that rate you hit 100% in about 47 hours.
"""

#: The compact internal block the guard exists to catch.
SCAFFOLD_BLOCK = """obj: reach 4096
prev_obj: none
state: thinking
mood: steady
goals: win
narr: the story so far
"""


def test_the_answer_that_was_destroyed_now_survives():
    assert not _contains_prompt_artifact(STRUCTURED_ANSWER)


def test_the_internal_block_is_still_caught():
    assert _contains_prompt_artifact(SCAFFOLD_BLOCK)


@pytest.mark.parametrize(
    "line", ["narr: the story so far", "usr: bryan", "ctx: chat", "prev_obj: x", "phenom: warm"]
)
def test_a_key_that_is_never_english_condemns_on_its_own(line):
    """No reply writes "narr:". One is enough."""
    assert _contains_prompt_artifact(line), line


@pytest.mark.parametrize(
    "heading",
    [
        "History: it grew 1% today",
        "Goals: keep at least 10% free",
        "State: healthy, nothing pending",
        "Mood: steady",
        "Voice: warm and direct",
    ],
)
def test_one_ordinary_heading_is_prose(heading):
    assert not _contains_prompt_artifact(heading), heading


def test_two_headings_are_still_prose():
    """A short labelled answer is a normal thing to write."""
    assert not _contains_prompt_artifact("State: healthy\nMood: steady")


def test_three_stacked_state_keys_read_as_the_block():
    assert _contains_prompt_artifact("State: x\nMood: y\nGoals: z")


@pytest.mark.parametrize(
    "marker",
    ["[ACTIVE GROUNDING EVIDENCE]", "[FETCHED PAGE CONTENT]", "[INTERNAL MEMORY RECALL]"],
)
def test_explicit_scaffold_markers_are_untouched(marker):
    assert _contains_prompt_artifact(f"{marker} something")


def test_a_code_fence_may_contain_anything():
    """Fenced content is quoted material, not her own prose."""
    fenced = "Here is the format:\n\n```\nobj: reach 4096\nnarr: story\n```\n"
    assert not _contains_prompt_artifact(fenced)
