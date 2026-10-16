"""A closing block after a list must not inherit list-item requirements."""

import pytest

from core.conversation.response_reliability import _has_truncated_tail, complete_truncated_tail

LIST = (
    "1. Persist the session before acknowledging delivery.\n"
    "2. Resume from the last acknowledged event.\n"
    "3. Commit exactly one final response.\n"
)


@pytest.mark.parametrize(
    "footer",
    [
        "R09-7F2C-COMPLETE",
        "build-2026-09-08-verification-complete",
        "All checks complete",
        "The session can now resume normally.",
    ],
)
@pytest.mark.parametrize("stop", ["", "eos", "length"])
def test_complete_footer_is_not_an_unfinished_list_item(footer, stop):
    assert not _has_truncated_tail(
        LIST + "\n" + footer, generation_stop_reason=stop
    )
    assert complete_truncated_tail(LIST + "\n" + footer) == LIST + "\n" + footer


@pytest.mark.parametrize(
    "footer",
    [
        "The next operation depends on",
        "The session is ready, but",
        "**Remaining work:**",
        'The final record says "delivery is complete',
    ],
)
def test_incomplete_footer_is_still_incomplete_at_eos(footer):
    assert _has_truncated_tail(LIST + "\n" + footer, generation_stop_reason="eos")


def test_list_does_not_hide_a_budget_clipped_concluding_paragraph():
    footer = (
        "The reconnect path must preserve the committed session state while\n"
        "the delivery worker reconciles the journal with the last acknowledged event"
    )
    assert _has_truncated_tail(LIST + "\n" + footer, generation_stop_reason="length")
    assert not _has_truncated_tail(LIST + "\n" + footer, generation_stop_reason="eos")


@pytest.mark.parametrize("tail", ["4.", "4. Resume", "4. The final response depends on"])
def test_unfinished_list_item_is_not_a_footer(tail):
    assert _has_truncated_tail(LIST + tail, generation_stop_reason="eos")


def test_unpunctuated_list_remains_complete():
    assert not _has_truncated_tail(
        "1. Persist session state\n2. Restore delivery position\n3. Commit final response",
        generation_stop_reason="eos",
    )


def test_a_list_of_single_words_ends_on_a_whole_item():
    """Live 2026-10-16: words sorted one to a line were refused as cut, and the right answer never reached her."""
    sorted_words = "Here they are:\n\n1. basil\n2. ember\n3. lantern\n4. orchard\n5. quarry\n6. velvet"
    assert not _has_truncated_tail(sorted_words, generation_stop_reason="eos")


@pytest.mark.parametrize(
    "cut",
    [
        "Here are the words sorted alphabetically from your list:\n\n1. basil",
        "1. Preheat the oven to 350 degrees.\n2. Mix the flour and the sugar.\n3. Bake",
        "1. Persist session state\n2. Restore delivery position\n3. Commit",
    ],
)
def test_a_last_item_shorter_than_its_list_makes_them_is_still_cut(cut):
    assert _has_truncated_tail(cut, generation_stop_reason="eos")
