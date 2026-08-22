"""Being told not to do something is not being told to do it.

LIVE, 2026-08-22, typed into the window: "...can you look at it and tell me
what's actually wrong? don't just make the test pass." The turn routed to the
desktop actuation lane, where os_automation spent thirty-seven seconds failing
to compile AppleScript for a Python question and logged the same tool handoff
ninety times before timing out.

The sentence that sent it there was the one telling her not to touch the code.
The filesystem-observation exemption — which exists for exactly this request,
with a comment about the same failure three days earlier — refused to fire
because the message "asks for a change". It asks for the opposite.

Two vocabularies caused it. The mutation detector reads `make` as a change;
the list deciding whether a change was NEGATED did not contain make, fix,
change, edit, patch or modify. Two lists answering "is this an action" and
"was it negated" drift apart, and this is what that looks like.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.runtime.desktop_objective_intent import (
    _FILESYSTEM_MUTATION_RE,
    looks_like_desktop_objective,
    looks_like_filesystem_observation,
    normalize_memory_intent_text,
    strip_negated_action_spans,
)


def mutates_after_negation(text: str) -> bool:
    stripped = strip_negated_action_spans(normalize_memory_intent_text(text).lower()).lower()
    return bool(_FILESYSTEM_MUTATION_RE.search(stripped))


@pytest.mark.parametrize(
    "text",
    [
        "don't just make the test pass",
        "don't fix it, just tell me",
        "do not delete anything",
        "without changing the file, tell me what it does",
        "don't edit the config",
        "never modify the source",
    ],
)
def test_a_negated_change_is_not_a_change(text: str):
    assert not mutates_after_negation(text), text


@pytest.mark.parametrize(
    "text",
    [
        "read it and fix the bug",
        "fix the bug in the parser",
        "delete the stale rows",
        "edit the config and restart",
    ],
)
def test_a_real_change_is_still_a_change(text: str):
    assert mutates_after_negation(text), text


def test_the_negation_vocabulary_covers_the_mutation_vocabulary():
    """The drift itself, as a check: anything the detector calls a change has
    to be something the negation can negate."""
    from core.runtime.skill_task_bridge import _ACTION_VERBS

    mutation_verbs = {
        "write", "save", "create", "make", "append", "edit", "modify", "patch",
        "fix", "update", "delete", "remove", "rename", "move", "copy",
    }
    missing = sorted(mutation_verbs - set(_ACTION_VERBS))
    assert not missing, f"the negation cannot negate: {missing}"


def test_the_project_question_reaches_the_reading_lane(tmp_path: Path):
    project = tmp_path / "ledger"
    project.mkdir()
    asked = (
        f"there's a small python project at {project} — one of its tests fails and I "
        "can't see why. can you look at it and tell me what's actually wrong? "
        "don't just make the test pass."
    )
    assert looks_like_filesystem_observation(asked)
    assert not looks_like_desktop_objective(asked)


def test_a_real_desktop_request_is_still_one():
    for asked in ("open Safari and go to apple.com", "click the save button in Notes"):
        assert looks_like_desktop_objective(asked), asked
