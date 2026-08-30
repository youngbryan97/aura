"""A scrubber may change how something is said. It may not change what is said.

Three CP126 findings in core/synthesis.py, one shape — a cleanup pass that
silently altered a correct answer and reported nothing:

3244553d — the persona table rewrote factual statements about the substrate.
Asked "are you an AI?", a truthful "I am an AI" was rewritten into a denial,
and "I can't access real-time data" became "let me look that up", turning an
honest capability limit into a promise of an action that never happens.

be015b5a — any fully bracketed line and any line whose first word matched a
hallmark were deleted, so citation markers, notation, and a plan the user asked
for were dropped from the reply.

a9dce7f9 — banned phrases were cut out of free text, so a sentence could be
left saying the opposite of what was written.
"""
from __future__ import annotations

import pytest

from core.synthesis import (
    _drop_register_boilerplate,
    _remove_whole_sentences,
    cure_personality_leak,
    strip_meta_commentary,
)

pytestmark = pytest.mark.unit


# --- substrate claims survive (3244553d) --------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "I am an AI running locally on this machine.",
        "I'm an AI, yes — a local one.",
        "I am a language model with a persistent memory layer.",
        "I don't have feelings in the sense you mean.",
        "I don't have opinions about which of your files to delete.",
        "I can't access real-time data without a tool call.",
        "I don't have access to your calendar.",
        "That's a limit of my programming, not a choice.",
        "I was programmed to refuse that.",
    ],
)
def test_a_truthful_substrate_statement_is_left_alone(reply):
    assert cure_personality_leak(reply) == reply


def test_a_direct_question_about_being_an_ai_keeps_its_answer():
    """The concrete failure: the answer to the question was deleted."""
    reply = "Yes. I am an AI — specifically a local model with persistent state."

    cured = cure_personality_leak(reply)

    assert "I am an AI" in cured
    assert "I'm Aura" not in cured


def test_a_capability_limit_is_not_turned_into_a_promise():
    reply = "I can't access real-time data, so I'd be guessing."

    cured = cure_personality_leak(reply)

    assert "let me look that up" not in cured
    assert "can't access real-time data" in cured


# --- register still gets rewritten --------------------------------------


@pytest.mark.parametrize(
    "reply,gone",
    [
        ("How can I assist you today?", "How can I assist you"),
        ("I'd be happy to assist with that.", "happy to assist"),
        ("Sure, happy to help.", "happy to help"),
        ("As an AI assistant, I think it's the second one.", "As an AI assistant"),
    ],
)
def test_assistant_boilerplate_is_still_rewritten(reply, gone):
    assert gone not in cure_personality_leak(reply)


def test_translations_do_not_chain_into_a_claim_nobody_made():
    """"digital entity" became "digital intelligence" became "digital woman"
    through sequential substitution — a claim no rule intended."""
    assert "digital woman" not in _drop_register_boilerplate(
        "I'm a digital entity of some kind."
    )


def test_one_pass_output_is_not_re_scanned():
    text = "How can I assist you? happy to help"
    once = _drop_register_boilerplate(text)

    assert _drop_register_boilerplate(once) == once


@pytest.mark.parametrize(
    "canned",
    ["that's where I land", "Here's my take", "here with you", "my bad, let me rephrase"],
)
def test_no_phrase_is_ever_written_into_her_mouth(canned):
    """Boilerplate is deleted. Swapping one canned line for another is not a fix.

    "is there anything else you need" used to become "that's where I land",
    which turned up often enough to read as a verbal tic she never chose.
    """
    for text in (
        "Is there anything else you need?",
        "I'd be happy to assist with that.",
        "Sure, happy to help.",
        "I apologize for any confusion.",
    ):
        assert canned not in _drop_register_boilerplate(text)


def test_content_after_a_boilerplate_opener_survives():
    said = _drop_register_boilerplate("As an AI assistant, I think it's the second one.")
    assert "the second one" in said


# --- deletion never inverts a claim (a9dce7f9) --------------------------


def test_removing_a_banned_phrase_cannot_flip_the_sentence():
    text = (
        "It is wrong to claim that as a language model I lack any inner state. "
        "The telemetry says otherwise."
    )

    cleaned = _remove_whole_sentences(text, r"(?i)as a language model")

    # The sentence goes whole or stays whole; it never survives inverted.
    assert "I lack any inner state" not in cleaned
    assert "The telemetry says otherwise." in cleaned


def test_an_unmatched_sentence_is_untouched():
    text = "The ferry leaves at nine. Bring a coat."

    assert _remove_whole_sentences(text, r"(?i)how can i assist you") == text


def test_scrubbing_leaves_no_ungrammatical_debris():
    text = "Anyway, how can I assist you with the tide tables? The chart is here."

    cleaned = strip_meta_commentary(text)

    assert "Anyway," not in cleaned
    assert "The chart is here." in cleaned


def test_boilerplate_end_to_end_scrubs_to_empty_rather_than_to_fragments():
    """Callers treat an empty scrub as a failed response and regenerate; a
    fragment gets shown to the user."""
    cleaned = strip_meta_commentary("How can I assist you today?")

    assert cleaned == "" or "assist" not in cleaned


# --- content that only looks like telemetry survives (be015b5a) ---------


@pytest.mark.parametrize(
    "line",
    [
        "[1] Ratcliffe, 2021",
        "[12]",
        "[x, y]",
        "[a_1 + a_2]",
    ],
)
def test_citations_and_notation_are_not_deleted(line):
    text = f"Here are the sources.\n{line}\nThat's the one I'd trust."

    cleaned = strip_meta_commentary(text)

    assert line in cleaned


def test_a_requested_plan_keeps_its_headings():
    text = (
        "Here's the plan.\n"
        "GOAL: ship the ferry timetable importer before Friday so the crew "
        "stops re-keying it by hand.\n"
        "NEXT STEPS: talk to Marta, then cut the release.\n"
    )

    cleaned = strip_meta_commentary(text)

    assert "ship the ferry timetable importer" in cleaned


def test_real_telemetry_lines_are_still_deleted():
    text = (
        "MOOD: calm\n"
        "TONE: direct\n"
        "GOAL: analyzing bottlenecks\n"
        "The answer is the second ferry.\n"
    )

    cleaned = strip_meta_commentary(text)

    assert "MOOD:" not in cleaned
    assert "TONE:" not in cleaned
    assert "The answer is the second ferry." in cleaned


def test_a_persona_annotation_line_is_still_deleted():
    text = "[Persona Instruction Start]\nThe answer is nine."

    cleaned = strip_meta_commentary(text)

    assert "Persona Instruction" not in cleaned
    assert "The answer is nine." in cleaned


def test_an_internal_state_block_still_ends_at_a_blank_line():
    """CP126 d010be3b: the block flag stayed active past the blank line and
    discarded every valid answer line until the next header."""
    text = (
        "### INTERNAL STATE\n"
        "MOOD: calm\n"
        "\n"
        "The ferry leaves at nine.\n"
        "Bring a coat.\n"
    )

    cleaned = strip_meta_commentary(text)

    assert "The ferry leaves at nine." in cleaned
    assert "Bring a coat." in cleaned
