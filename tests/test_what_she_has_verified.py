"""L10: what she checks online is kept, and held as verified only when independent, reputable sources agree; a
disagreement is held as disputed; only checkable factual claims are kept; the store answers with no network."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

WIKI = "https://en.wikipedia.org/wiki/Mount_Everest"
NASA = "https://earthobservatory.nasa.gov/images/everest"
NEWS = "https://www.reuters.com/world/everest-height"
BLOG = "https://someone.medium.com/everest"
FORUM = "https://www.reddit.com/r/geography/everest"


def test_sources_are_ranked_by_what_they_are_and_grouped_by_who_holds_them():
    from core.epistemics.what_she_has_verified import publisher, tier_of

    assert [tier_of(u) for u in (WIKI, NASA, NEWS, BLOG, FORUM)] == [
        "reference", "official", "news", "blog_forum_social", "blog_forum_social"]
    assert tier_of("https://www.ons.gov.uk/x") == "official" and tier_of("https://www.ox.ac.uk/x") == "primary"
    assert {tier_of(u) for u in ("https://www.culture.gouv.fr/x", "https://www.inegi.gob.mx/x", "https://www.stat.go.jp/x",
                                 "https://www.statcan.gc.ca/x", "https://www.bfs.admin.ch/x")} == {"official"}
    assert publisher("https://www.culture.gouv.fr/x") == "culture.gouv.fr"
    assert tier_of("https://doi.org/10.1/x") == "primary" and tier_of("https://example.com/x") == "unranked"
    assert publisher("https://en.wikipedia.org/x") == publisher("https://de.wikipedia.org/y") == "wikimedia"
    assert publisher("https://www.wikidata.org/wiki/Q243") == "wikimedia"  # Wikipedia cannot vouch for itself
    assert publisher("https://www.bbc.co.uk/news") == publisher("https://www.bbc.com/news") == "bbc"


def test_only_checkable_factual_claims_are_kept():
    from core.epistemics.what_she_has_verified import checkable

    assert checkable("Mount Everest is 8849 metres tall.")
    assert checkable("Paris is the capital of France.")
    for not_a_fact in ("I think Everest is the best mountain.", "Everest will be taller in a million years.",
                       "Is Everest 8849 metres tall?", "everything is fine", "You should climb Everest."):
        assert not checkable(not_a_fact), not_a_fact


def test_a_claim_is_verified_only_when_two_independent_sources_agree_and_one_is_reputable(tmp_path):
    from core.epistemics.what_she_has_verified import (
        CONTRADICTED, DISPUTED, SINGLE_SOURCE, SUPPORTED, VERIFIED, VerifiedFacts,
    )

    store = VerifiedFacts(tmp_path / "facts.json")
    claim = "Mount Everest is 8849 metres tall."
    assert store.attest(claim, WIKI, SUPPORTED, "Everest's height is 8,849 m").status == SINGLE_SOURCE
    assert store.attest(claim, "https://de.wikipedia.org/wiki/Everest", SUPPORTED).status == SINGLE_SOURCE  # one publisher
    assert store.attest(claim, "https://www.wikidata.org/wiki/Q513", SUPPORTED).status == SINGLE_SOURCE  # still Wikimedia
    assert store.attest(claim, BLOG, SUPPORTED).status == SINGLE_SOURCE  # a blog is kept and counts for nothing
    assert store.attest(claim, NASA, SUPPORTED).status == VERIFIED
    assert store.attest(claim, NEWS, CONTRADICTED).status == VERIFIED  # news does not outrank reference and official
    assert store.attest(claim, "https://www.usgs.gov/everest", CONTRADICTED).status == DISPUTED  # held, not resolved


def test_a_program_settles_what_it_can_compute(tmp_path):
    from core.epistemics.what_she_has_verified import REFUTED, SUPPORTED, VERIFIED, VerifiedFacts

    store = VerifiedFacts(tmp_path / "facts.json")
    assert store.attest("17 x 23 = 391", WIKI, SUPPORTED).status == VERIFIED
    assert store.attest("17 x 23 = 392", WIKI, SUPPORTED).status == REFUTED


def test_the_store_is_kept_and_answers_with_the_network_blocked(tmp_path, monkeypatch):
    import socket

    from core.epistemics.what_she_has_verified import SUPPORTED, VERIFIED, VerifiedFacts, as_evidence

    store = VerifiedFacts(tmp_path / "facts.json")
    claim = "Mount Everest is 8849 metres tall."
    store.attest(claim, WIKI, SUPPORTED)
    store.attest(claim, NASA, SUPPORTED)

    def no_network(*_args, **_kwargs):
        raise OSError("the network is blocked")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    again = VerifiedFacts(tmp_path / "facts.json")  # a new process, offline
    assert again.get("mount everest is 8849 metres tall").status == VERIFIED
    known = again.bearing_on("How tall is Mount Everest, in metres?")
    assert [fact.claim for fact in known] == [claim]
    block = as_evidence(known, today="2026-10-19")
    assert "verified" in block and "wikimedia (reference, supported" in block and "nasa.gov (official" in block


def test_a_dated_fact_is_read_again_on_a_later_day(tmp_path):
    from core.epistemics.what_she_has_verified import SUPPORTED, VerifiedFacts

    store = VerifiedFacts(tmp_path / "facts.json")
    current = store.attest("The current population of Iceland is 383726.", WIKI, SUPPORTED)
    lasting = store.attest("Iceland joined NATO in 1949.", WIKI, SUPPORTED)
    later = "9999-12-31"
    assert current.due_again(later) and not current.due_again(current.last_read)
    assert not lasting.due_again(later)
