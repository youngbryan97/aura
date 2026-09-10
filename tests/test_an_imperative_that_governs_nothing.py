""""Wait" at the head of a sentence is usually a change of mind, not an order.

`wait` is in the action verbs for good reasons — "wait for the build" is an
instruction — and the imperative pattern matched a bare verb with no
requirement that it govern anything. So one of the commonest ways an English
sentence changes its mind was read as a request for work.

LIVE 2026-09-07: "WAIT that's a perfect response, Aura. Those are opinions."
classified as TASK and would have been handed to the desktop task engine. A
test named for the continuity wrapper had been failing on this for some time;
the wrapper was never the cause, the bare sentence classifies the same way.

The distinction is syntactic and needs no list of which words double as
interjections: an imperative governs an object, a prepositional phrase or an
infinitive. A subject and a finite verb after it is a new clause, so the verb
heads nothing.
"""

from __future__ import annotations

import pytest

from core.conversation.request_mood import assess_request_mood
from core.runtime.turn_analysis import analyze_turn


@pytest.mark.parametrize(
    "message",
    [
        "WAIT that's a perfect response, Aura. Those are opinions.",
        "wait — I meant the other one",
        "wait, that is not what I asked",
        "look, I think you are wrong",
        "hold that thought, this is really interesting",
        "stop, you already did that",
    ],
)
def test_an_interjection_before_a_clause_asks_for_nothing(message: str) -> None:
    assert assess_request_mood(message).asks_for_action is False, message
    assert analyze_turn(message).intent_type == "CHAT", message


@pytest.mark.parametrize(
    "message",
    [
        "wait for the build to finish",
        "read the file /tmp/notes.txt",
        "open Safari",
        "check that the build passed",
        "put BUILD-42 on my clipboard",
        "keep watching the log for errors",
    ],
)
def test_an_imperative_that_governs_something_still_asks(message: str) -> None:
    assert analyze_turn(message).intent_type != "CHAT", message


@pytest.mark.parametrize(
    "message",
    [
        # Neither of these puts a finite verb straight after a subject, so the
        # rule does not see them at all. They read as CHAT for their own
        # reasons — both ask for an answer in this reply — and this pins that
        # the new rule is not what decided it.
        "show me what you found",
        "see if it works",
    ],
)
def test_the_rule_leaves_alone_what_it_does_not_match(message: str) -> None:
    from core.conversation.request_mood import _INTERJECTION_BEFORE_A_CLAUSE_RE

    assert _INTERJECTION_BEFORE_A_CLAUSE_RE.search(message) is None, message


def test_the_wrapper_was_never_the_cause() -> None:
    """The test that caught this blamed a continuity wrapper. It was the verb."""
    bare = "WAIT that's a perfect response, Aura. Those are opinions."
    wrapped = (
        "[Continuity context — earlier in this conversation]\n"
        "User asked: Tell me about yourself.\n"
        "You answered: I like mysteries.\n"
        "[End continuity context]\n\n" + bare
    )
    assert analyze_turn(bare).intent_type == analyze_turn(wrapped).intent_type == "CHAT"


@pytest.mark.parametrize(
    "message",
    [
        "Open Notes, click into a new note, type hello, then come back and report what happened.",
        "read the log, check for errors",
        "open Safari, go to the docs, find the install section",
    ],
)
def test_instructions_listed_with_commas_are_still_instructions(message: str) -> None:
    """English lists instructions with commas as readily as with "and".

    Splitting only on the conjunction left the whole turn as one clause, and a
    retrospective frame in its tail — "report what happened" — outranked the
    imperative at its head. The multi-step desktop instruction read as a
    question about the past.
    """
    assert assess_request_mood(message).asks_for_action is True, message


@pytest.mark.parametrize(
    "message",
    [
        "open the file, the one in /tmp",
        "send the email, please",
        "I like mysteries, well-crafted stories, and long walks",
    ],
)
def test_a_comma_before_something_that_is_not_a_verb_is_not_a_boundary(
    message: str,
) -> None:
    from core.conversation.request_mood import _split_independent_clauses

    assert len(_split_independent_clauses(message)) == 1, message


def test_a_fronted_time_stays_with_the_instruction_it_times() -> None:
    """A clause has a verb; a fragment before a comma that has none does not.

    Splitting on a comma before an imperative cut "Tomorrow, create a reminder
    to inspect the training receipt" in two, and the scheduled scope went with
    the half that had no verb.
    """
    verdict = assess_request_mood(
        "Tomorrow, create a reminder to inspect the training receipt."
    )
    assert verdict.asks_for_action is True
    assert verdict.temporal_scope == "scheduled"


@pytest.mark.parametrize(
    "message",
    [
        "Maybe you should create a concise report from those measurements.",
        "Perhaps you could write up the results.",
    ],
)
def test_an_artifact_someone_asks_for_is_not_words_in_this_reply(message: str) -> None:
    """An asking clause with no recognised external object falls to "words".

    That elimination is only as good as the object list, which held documents,
    notes, spreadsheets and presentations — and not the commonest word for the
    same kind of thing. "Maybe you should create a concise report from those
    measurements" was answered in the reply rather than planned.
    """
    assert analyze_turn(message).intent_type != "CHAT", message


def test_the_object_list_is_what_limits_this_and_not_the_phrasing() -> None:
    """Recorded, not fixed: "you might want to X" is not read as a request.

    The artifact nouns above close one hole. The indirect-request frames are a
    separate list with holes of their own, and widening a pattern is the move
    `docs/LEARNED_LANGUAGE_INTERPRETATION_TODO.md` says will be needed again.
    This pins the boundary so the gap is a known one rather than a surprise.
    """
    assert analyze_turn("you might want to draft a plan").intent_type == "CHAT"
    assert (
        analyze_turn("Perhaps you could draft a plan").intent_type != "CHAT"
    ), "the frames that ARE recognised must keep working"


@pytest.mark.parametrize(
    "message",
    [
        "what do you think about reports?",
        "tell me about the report you read",
        "summarize this in two sentences",
    ],
)
def test_talking_about_an_artifact_is_still_talking(message: str) -> None:
    assert analyze_turn(message).intent_type == "CHAT", message
