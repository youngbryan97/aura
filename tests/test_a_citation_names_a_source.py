"""A citation that names no source is worse than no citation.

LIVE, 2026-08-22, typed into the window and asked for links: the reply ended
"Source: [Live web search]". No URL, no title, nothing anyone could open — the
shape of a citation standing in for one, on a turn the log recorded as
source_present=False. A reader skims that as evidence.

The same turn opened "I checked live web evidence", which the runtime itself
wrote when repairing provenance. Since 15 November 2026 the runtime writes no
reply of its own here: her reply is kept, what the record refutes is taken out
of it, and the source is added.
"""

from __future__ import annotations

import pytest

from interface.routes.chat import (
    _cites_nothing,
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


def test_her_reply_is_never_replaced_by_a_headline():
    """LIVE 15 November 2026: a remark about sleep was answered with a Reddit thread's title and blurb."""
    evidence = {
        "ok": True,
        "result": {"ok": True, "results": [{"title": "About", "url": "https://huggingface.co/about", "snippet": "Founded in 2016."}]},
    }
    for reply in ("I remember you saying that. Hugging Face started in 2016.", "From memory, it started in 2016."):
        repaired = _repair_required_search_reply_provenance(reply, evidence)
        assert "I checked live web evidence" not in repaired
        assert "About:" not in repaired and repaired.endswith("Source: https://huggingface.co/about")
    assert _repair_required_search_reply_provenance("I remember you saying that.", evidence).startswith("I remember you")
    assert _repair_required_search_reply_provenance("From memory, it started in 2016.", evidence).startswith("It started")


def test_an_empty_citation_is_removed_when_there_is_no_rebuild():
    """With no grounded rebuild available the original text was returned
    unchanged, so the fake citation was served."""
    evidence = {"ok": True, "result": {"ok": False}}
    repaired = _repair_required_search_reply_provenance(
        "Hugging Face was founded in 2016. Source: [Live web search]", evidence
    )
    assert "Live web search" not in repaired
    assert "founded in 2016" in repaired


def test_where_the_evidence_came_from_is_read_from_the_result():
    """LIVE, 2026-08-22: the search had degraded to the local corpus (provenance local_corpus, offline_fallback
    true) and the reply called it "live web evidence"."""
    from interface.routes.chat import _evidence_came_from_the_network

    offline = {"ok": True, "offline_fallback": True, "provenance": "local_corpus",
               "results": [{"title": "HF", "snippet": "An ML company.", "source": "corpus"}]}
    live = {"ok": True, "results": [{"title": "About", "url": "https://huggingface.co/about", "snippet": "Founded 2016."}]}
    assert not _evidence_came_from_the_network(offline)
    assert _evidence_came_from_the_network(live)


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
    assert repaired == "Hugging Face is an ML company."


def test_ordinary_sentences_are_not_live_check_claims():
    from interface.routes.chat import _claims_a_live_check

    for text in ("I read the file.", "The web page was long.", "I checked the log."):
        assert not _claims_a_live_check(text), text


def test_an_honest_reply_without_an_address_keeps_its_words_and_gains_the_source():
    """LIVE 2026-10-03: her reply about Bam's 83 was replaced by a pasted headline."""
    from interface.routes.chat import _repair_required_search_reply_provenance

    evidence = {
        "ok": True,
        "result": {
            "ok": True,
            "results": [{"title": "Bam! 83-point historic night", "url": "https://www.nba.com/news/bam-adebayo-83-point-game",
                         "snippet": "The veteran big man passes Kobe Bryant for the second-highest game."}],
        },
    }
    reply = "He really did it: 83 against the Wizards, second only to Wilt, and past Kobe's 81."
    repaired = _repair_required_search_reply_provenance(reply, evidence)
    assert repaired.startswith(reply)
    assert repaired.endswith("Source: https://www.nba.com/news/bam-adebayo-83-point-game")
    assert "I checked live web evidence" not in repaired
