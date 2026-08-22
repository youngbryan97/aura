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


def test_an_offline_snapshot_says_it_is_offline():
    """LIVE, 2026-08-22: the search had degraded to the local corpus and the
    reply called it "live web evidence". The result said so itself —
    provenance local_corpus, offline_fallback true — and the reply overrode
    it."""
    from interface.routes.chat import (
        _claims_a_live_check,
        _evidence_came_from_the_network,
    )

    offline = {
        "ok": True,
        "result": {
            "ok": True,
            "offline_fallback": True,
            "provenance": "local_corpus",
            "results": [{"title": "HF", "snippet": "An ML company.", "source": "corpus"}],
        },
    }
    said = _evidence_grounded_desktop_search_reply(offline)
    assert "live" not in said.lower().split("rather than a live check")[0]
    assert "offline reference snapshot" in said
    assert not _evidence_came_from_the_network(offline["result"])

    live = {
        "ok": True,
        "result": {
            "ok": True,
            "results": [
                {"title": "About", "url": "https://huggingface.co/about", "snippet": "Founded 2016."}
            ],
        },
    }
    assert "I checked live web evidence." in _evidence_grounded_desktop_search_reply(live)
    assert _evidence_came_from_the_network(live["result"])


def test_claiming_a_live_check_over_a_snapshot_is_false_provenance():
    from interface.routes.chat import _repair_required_search_reply_provenance

    offline = {
        "ok": True,
        "result": {
            "ok": True,
            "offline_fallback": True,
            "provenance": "local_corpus",
            "results": [{"title": "HF", "snippet": "An ML company.", "source": "corpus"}],
        },
    }
    repaired = _repair_required_search_reply_provenance(
        "I checked live web evidence. Hugging Face is an ML company.", offline
    )
    assert "offline reference snapshot" in repaired


def test_ordinary_sentences_are_not_live_check_claims():
    from interface.routes.chat import _claims_a_live_check

    for text in ("I read the file.", "The web page was long.", "I checked the log."):
        assert not _claims_a_live_check(text), text
