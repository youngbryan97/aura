"""A citation that names no source is worse than no citation.

LIVE, 2026-08-22, typed into the window and asked for links: the reply ended
"Source: [Live web search]". No URL, no title, nothing anyone could open — the
shape of a citation standing in for one, on a turn the log recorded as
source_present=False. A reader skims that as evidence.

The same turn opened "I checked live web evidence", which the runtime itself
writes when repairing provenance, and it wrote that with no URL in hand.
"""

from __future__ import annotations

import pytest

from interface.routes.chat import (
    _cites_nothing,
    _evidence_grounded_desktop_search_reply,
    _repair_required_search_reply_provenance,
    _strip_empty_citations,
)


@pytest.mark.parametrize(
    "reply",
    [
        "Hugging Face was founded in 2016. Source: [Live web search]",
        "They raised a lot. Sources: web search.",
        "Source: my own memory.",
        "Founded 2016. Source: unknown",
    ],
)
def test_a_citation_with_nothing_in_it_is_caught(reply: str):
    assert _cites_nothing(reply), reply
    assert "Source" not in _strip_empty_citations(reply)


@pytest.mark.parametrize(
    "reply",
    [
        "Founded 2016. Source: https://huggingface.co/about",
        "The source of the leak was a config file.",
        "I read the source code.",
    ],
)
def test_a_real_citation_and_ordinary_prose_survive(reply: str):
    assert not _cites_nothing(reply), reply
    assert _strip_empty_citations(reply) == reply


def test_the_grounded_rebuild_does_not_claim_a_check_it_cannot_show():
    without_url = {
        "ok": True,
        "result": {"ok": True, "summary": "Hugging Face builds open source ML tooling."},
    }
    said = _evidence_grounded_desktop_search_reply(without_url)
    assert "I checked live web evidence" not in said
    assert "open source ML tooling" in said

    with_url = {
        "ok": True,
        "result": {
            "ok": True,
            "results": [
                {
                    "title": "About",
                    "url": "https://huggingface.co/about",
                    "snippet": "Founded in 2016.",
                }
            ],
        },
    }
    grounded = _evidence_grounded_desktop_search_reply(with_url)
    assert "I checked live web evidence" in grounded
    assert "https://huggingface.co/about" in grounded


def test_an_empty_citation_is_removed_when_there_is_no_rebuild():
    """With no grounded rebuild available the original text was returned
    unchanged, so the fake citation was served."""
    evidence = {"ok": True, "result": {"ok": False}}
    repaired = _repair_required_search_reply_provenance(
        "Hugging Face was founded in 2016. Source: [Live web search]", evidence
    )
    assert "Live web search" not in repaired
    assert "founded in 2016" in repaired
