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
