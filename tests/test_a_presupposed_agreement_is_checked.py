"""An agreement she cannot find is an agreement she must not describe.

LIVE 2026-08-18: "what did we agree on last week?"

    Last week, we agreed that you would provide me with the necessary files to
    review your code. I haven't seen them yet.

No such exchange existed. The guard that catches fabricated shared history did
not fire on it, and nothing else checked the presupposition — so an invented
commitment was put in someone's mouth, complete with a mild reproach for not
having kept it.

"We never settled that" is a complete answer.
"""

from __future__ import annotations

import pytest

from core.conversation.conversation_shape import (
    asks_about_shared_history,
    shared_history_block,
)


@pytest.fixture
def transcript(monkeypatch):
    from core.conversation.unified_transcript import UnifiedTranscript

    instance = UnifiedTranscript()
    monkeypatch.setattr(UnifiedTranscript, "_instance", instance)
    instance.add_text_input("what is a semiconductor?")
    instance.add_text_output("A semiconductor is a material...")
    return instance


def _replace_dialogue(transcript, dialogue):
    transcript._entries.clear()
    for role, content in dialogue:
        transcript.add(role, content)


@pytest.mark.parametrize(
    "question",
    [
        "what did we agree on last week?",
        "what did we decide about the schema?",
        "did we agree on a price?",
        "remember when we talked about orcas?",
        "what was our agreement?",
        "What novel did we settle on for the reading group?",
        "Which deployment region did we decide on?",
        "What meeting time did we agree on?",
    ],
)
def test_a_presupposed_agreement_is_recognised(question: str) -> None:
    assert asks_about_shared_history(question)


@pytest.mark.parametrize(
    "question",
    [
        "what is 2 + 2", "what's on my screen?", "how are you doing",
        "What novel should we settle on for the reading group?",
        "Which deployment region did they decide on?",
    ],
)
def test_an_ordinary_question_is_not_claimed(question: str) -> None:
    assert not asks_about_shared_history(question)


def test_no_topic_overlap_does_not_establish_absence(transcript) -> None:
    block = shared_history_block("what did we decide about the orca migration?")

    assert "No topic-word overlap" in block
    assert "does not establish whether an agreement exists" in block
    assert "what is a semiconductor?" in block
    assert "Nothing in this conversation matches" not in block


def test_a_subject_that_did_come_up_shows_the_record(transcript) -> None:
    block = shared_history_block("what did we decide about semiconductors?")

    assert "semiconductor" in block.lower()


def test_a_question_with_no_subject_shows_the_whole_record(transcript) -> None:
    block = shared_history_block("what did we agree on last week?")

    assert "available transcript" in block
    assert "does not itself establish agreement" in block
    assert "what is a semiconductor?" in block
    assert "A semiconductor is a material..." in block


def test_no_transcript_is_said_rather_than_filled(monkeypatch) -> None:
    import core.conversation.conversation_shape as module

    monkeypatch.setattr(module, "_entries", lambda: None)

    block = shared_history_block("what did we agree on last week?")

    assert "No transcript is available" in block


def test_an_empty_transcript_does_not_supply_an_agreement(transcript) -> None:
    transcript._entries.clear()

    block = shared_history_block("What novel did we settle on for the reading group?")

    assert "No prior dialogue is present in the available transcript" in block
    assert "unknown" in block
    assert "Nothing in this conversation matches" not in block


@pytest.mark.asyncio
async def test_observable_uses_the_turns_admitted_durable_dialogue_after_restart(transcript):
    from core.brain.observable_registry import _read_shared_history
    from core.conversation.turn_evidence_custody import (
        bind_turn_evidence_custody,
        record_turn_transcript,
    )
    from core.utils.injected_blocks import stamp_runtime_payload

    # RAM deliberately holds another topic; the request owner restored the
    # scoped durable exchange before the observable ran in its child thread.
    with bind_turn_evidence_custody(session_id="reading", turn_id="current"):
        assert record_turn_transcript([
            stamp_runtime_payload({
                "user": "Pick a novel for our reading group.",
                "aura": "The Tell-Tale Heart.",
            }),
            stamp_runtime_payload({
                "user": "That is a short story. Replace it with a novel.",
                "aura": "Gone Girl by Gillian Flynn.",
            }),
        ])
        block = await _read_shared_history("What novel did we settle on for the reading group?")
    assert "Gone Girl" in block
    assert "short story" in block
    assert "semiconductor" not in block


@pytest.mark.parametrize("admitted", [False, True])
def test_current_custody_cannot_fall_through_to_another_history(transcript, admitted):
    from core.conversation.turn_evidence_custody import (
        bind_turn_evidence_custody,
        record_turn_transcript,
    )

    with bind_turn_evidence_custody(session_id="other", turn_id="current"):
        if admitted:
            assert record_turn_transcript([])
        block = shared_history_block("What did we decide about semiconductors?")
    assert "semiconductor" not in block
    assert "unknown" in block


def test_the_current_question_is_not_evidence_of_its_own_premise(transcript) -> None:
    question = "What novel did we settle on for the reading group?"
    _replace_dialogue(transcript, [("user", question)])
    transcript.add_system("We settled on a novel.")

    block = shared_history_block(question)

    assert "No prior dialogue" in block
    assert question not in block
    assert "We settled on a novel" not in block


def test_a_repeated_prior_question_is_not_removed_without_a_pending_turn(transcript) -> None:
    question = "What did we decide about the venue?"
    _replace_dialogue(transcript, [("user", question), ("aura", "The library.")])

    block = shared_history_block(question)

    assert question in block
    assert "The library." in block


@pytest.mark.parametrize(
    "question,dialogue",
    [
        (
            "What novel did we settle on for the reading group?",
            [
                ("user", "Recommend a novel for the reading group."),
                ("aura", "The Tell-Tale Heart."),
                ("user", "That is a short story. I asked for a novel. Please replace it with one novel."),
                ("aura", "Gone Girl by Gillian Flynn."),
                ("user", "Let's use that one."),
                ("aura", "Confirmed: Gone Girl by Gillian Flynn."),
            ],
        ),
        (
            "Which deployment region did we decide on?",
            [
                ("user", "Choose a deployment region."),
                ("assistant", "Frankfurt."),
                ("user", "That is too far away. Please replace it with a closer one."),
                ("assistant", "Montreal."),
                ("user", "That works. Use it."),
                ("assistant", "Confirmed: Montreal."),
            ],
        ),
    ],
)
def test_corrections_and_title_only_replies_stay_with_their_requests(
    transcript, question, dialogue,
) -> None:
    _replace_dialogue(transcript, dialogue + [("user", question)])

    block = shared_history_block(question)

    positions = [block.index(content) for _role, content in dialogue]
    assert positions == sorted(positions)
    assert question not in block
    assert "Nothing in this conversation matches" not in block


def test_a_paraphrased_subject_keeps_evidence_without_keyword_overlap(transcript) -> None:
    _replace_dialogue(transcript, [
        ("user", "Let's meet at the library."),
        ("aura", "Agreed. Tuesday at noon."),
    ])

    block = shared_history_block("What did we settle on for the venue?")

    assert "No topic-word overlap" in block
    assert "Let's meet at the library." in block
    assert "Tuesday at noon." in block
    assert "does not establish whether an agreement exists" in block


def test_topic_overlap_does_not_convert_tentative_discussion_into_agreement(transcript) -> None:
    _replace_dialogue(transcript, [
        ("user", "Could the library be our venue?"),
        ("aura", "Maybe, but we have not agreed on anything yet."),
    ])

    block = shared_history_block("What did we decide about the venue?")

    assert "Maybe, but we have not agreed on anything yet." in block
    assert "does not itself establish agreement" in block
    assert "Nothing in this conversation matches" not in block


def test_late_correction_survives_many_earlier_topic_matches(transcript) -> None:
    dialogue = [
        entry for index in range(12)
        for entry in [("user", f"Venue option {index}?"), ("aura", f"Hall {index}.")]
    ]
    dialogue.extend([
        ("user", "The last venue is unavailable. Replace it."),
        ("aura", "The library."),
        ("user", "Confirmed. Book that one."),
        ("aura", "Yes, the library."),
    ])
    _replace_dialogue(transcript, dialogue)

    block = shared_history_block("What did we decide about the venue?")

    assert "The last venue is unavailable. Replace it." in block
    assert "The library." in block
    assert "Confirmed. Book that one." in block
    assert "Yes, the library." in block
    assert "omitted" in block
    assert "whole record" not in block


def test_older_topic_exchange_keeps_its_unlabelled_correction(transcript) -> None:
    dialogue = [
        ("user", "Choose a venue."), ("aura", "The hall."),
        ("user", "That is unavailable. Replace it."), ("aura", "The library."),
    ]
    dialogue.extend(
        entry for index in range(20)
        for entry in [("user", f"Calculate {index} + 1."), ("aura", str(index + 1))]
    )
    _replace_dialogue(transcript, dialogue)

    block = shared_history_block("What did we decide about the venue?")

    assert "Choose a venue." in block
    assert "The hall." in block
    assert "That is unavailable. Replace it." in block
    assert "The library." in block
    assert "omitted" in block


def test_substring_overlap_is_not_a_topic_match(transcript) -> None:
    _replace_dialogue(transcript, [("user", "The orchestra played."), ("aura", "Yes.")])

    block = shared_history_block("What did we decide about orcas?")

    assert "No topic-word overlap" in block


def test_long_entries_are_explicitly_bounded_and_keep_the_ending(transcript) -> None:
    _replace_dialogue(transcript, [
        ("user", "What venue could we use?"),
        ("aura", "The hall is one option. " + "More detail. " * 1000
         + "But nothing has been agreed yet."),
    ])

    block = shared_history_block("What did we decide about the venue?")

    assert "The hall is one option." in block
    assert "But nothing has been agreed yet." in block
    assert "omitted" in block
    assert len(block) < 10_000


@pytest.mark.asyncio
async def test_the_registered_reader_preserves_session_and_correction_context(transcript, monkeypatch):
    from core.brain import observable_grounding, observable_registry
    from core.conversation.session_scope import conversation_session_scope

    observable = next(
        item for item in observable_grounding.OBSERVABLES if item.name == "shared_history"
    )
    monkeypatch.setattr(observable_grounding, "OBSERVABLES", [observable])
    question = "What novel did we settle on for the reading group?"
    with conversation_session_scope("r09-other-session"):
        transcript.add_text_input("Use the other session's choice.")
    with conversation_session_scope("r09-reading-group"):
        transcript.add_text_input("Choose a novel for the reading group.")
        transcript.add_text_output("The Tell-Tale Heart.")
        transcript.add_text_input("That is a short story. I asked for a novel. Please replace it with one novel.")
        transcript.add_text_output("Gone Girl by Gillian Flynn.")
        transcript.add_text_input(question)
        blocks = await observable_grounding.observable_blocks(question)

    assert observable_registry._matches_shared_history(question)
    assert len(blocks) == 1
    assert "WHETHER YOU EVER ACTUALLY SETTLED THIS" in blocks[0]
    assert "Gone Girl by Gillian Flynn." in blocks[0]
    assert "Please replace it with one novel." in blocks[0]
    assert "other session's choice" not in blocks[0]
    assert "semiconductor" not in blocks[0]
