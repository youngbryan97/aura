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
        CONTRADICTED,
        DISPUTED,
        SINGLE_SOURCE,
        SUPPORTED,
        VERIFIED,
        VerifiedFacts,
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

    from core.epistemics.what_she_has_verified import (
        SUPPORTED,
        VERIFIED,
        VerifiedFacts,
        as_evidence,
    )

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


@pytest.mark.parametrize(("url", "tier"), [
    ("https://www.britannica.com/science/boiling-point", "reference"),
    ("https://plato.stanford.edu/entries/truth/", "reference"),
    ("https://www.cdc.gov/flu/", "official"),
    ("https://www.ons.gov.uk/census", "official"),
    ("https://www.house.gov/representatives", "statement"),
    ("https://www.stanford.edu/research", "primary"),
    ("https://cs.stanford.edu/~someone/notes.html", "unranked"),
    ("https://www.nature.com/articles/x", "primary"),
    ("https://arxiv.org/abs/2410.00001", "preprint"),
    ("https://www.reuters.com/world/", "news"),
])
def test_a_source_is_what_the_operators_list_says_and_its_classes_are_checked(url, tier):
    from core.epistemics.what_she_has_verified import counts, tier_of

    assert tier_of(url) == tier
    assert counts(tier) is (tier in ("reference", "official", "primary", "news"))


def test_the_store_is_written_after_its_lock_is_let_go(tmp_path, monkeypatch):
    """LIVE 2026-11-10: 'LOCKDEP blocking_op_under_lock: fsync attempted while holding
    ['what_she_has_verified.facts']', and her runtime was tainted for it."""
    import json

    import core.runtime.atomic_writer as writer
    from core.epistemics.what_she_has_verified import VerifiedFacts
    from core.runtime import lockdep

    held: list[list[str]] = []
    original = writer.atomic_write_text_behind

    def probe(path, text, **kwargs):
        held.append(lockdep.get_validator().held_names())
        return original(path, text, **kwargs)

    monkeypatch.setattr(writer, "atomic_write_text_behind", probe)
    store = VerifiedFacts(tmp_path / "verified_facts.json")
    store.attest("The Eiffel Tower opened in 1889 in Paris.", "https://www.britannica.com/topic/Eiffel-Tower",
                 "SUPPORTED", "opened in 1889")
    assert held and all("what_she_has_verified.facts" not in names for names in held)
    assert json.loads((tmp_path / "verified_facts.json").read_text())[0]["claim"].startswith("The Eiffel Tower")


def test_one_page_read_twice_on_one_day_is_one_reading(tmp_path):
    """LIVE 2026-10-20: her check read Wikipedia's page twice for one claim and her store kept both, so her evidence
    named the same source twice. A later day's reading is kept beside it."""
    import json

    from core.epistemics.what_she_has_verified import (
        SUPPORTED,
        Attestation,
        VerifiedFacts,
        as_evidence,
    )

    path = tmp_path / "verified_facts.json"
    claim = "The speed of light in a vacuum is exactly 299,792,458 metres per second."
    page = "https://en.wikipedia.org/wiki/Speed_of_light"
    kept = {"url": page, "verdict": SUPPORTED, "quote": "exactly equ", "read_at": "2026-10-20"}
    path.write_text(json.dumps([{"claim": claim, "attestations": [kept, kept]}]))
    store = VerifiedFacts(path)
    assert len(store.get(claim).attestations) == 1  # the record she already holds reads as one reading
    store.attest(claim, page, SUPPORTED, "exactly equal to 299792458 m/s")
    store.attest(claim, page, SUPPORTED, "exactly equal to 299792458 m/s, the later quote")
    readings = store.get(claim).attestations
    assert [a.read_at for a in readings] == ["2026-10-20", Attestation(url=page, verdict=SUPPORTED).read_at]
    assert readings[-1].quote.endswith("the later quote")
    assert as_evidence([store.get(claim)]).count("2026-10-20") == 1


LIGHT = "the speed of light in a vacuum is exactly 299,792,458 metres per second"


def _verified_light(tmp_path):
    from core.epistemics.what_she_has_verified import SUPPORTED, VerifiedFacts

    store = VerifiedFacts(tmp_path / "facts.json")
    store.attest(LIGHT, WIKI, SUPPORTED)
    store.attest(LIGHT, NASA, SUPPORTED)
    return store


def test_a_question_gets_the_fact_it_asks_for_though_it_holds_few_of_its_words(tmp_path):
    """L10, 13 November 2026: a question never holds the answer's figure or units, so by words alone her verified
    speed of light never reached "How fast does light travel in a vacuum, exactly?". Judged by meaning, it does, and
    a question about another speed does not."""
    from core.language.learned_matcher import embed_sentences

    if not embed_sentences(["a sentence"]):
        pytest.skip("no embedder in this process; the words decide, as the next test holds")
    store = _verified_light(tmp_path)
    assert [fact.claim for fact in store.bearing_on("How fast does light travel in a vacuum, exactly?")] == [LIGHT]
    assert store.bearing_on("How fast can a cheetah run?") == []
    assert store.bearing_on("What did you have for lunch?") == []


def test_where_meaning_cannot_judge_the_shared_words_decide(tmp_path, monkeypatch):
    from core.epistemics import what_she_has_verified

    monkeypatch.setattr(what_she_has_verified, "_bearing_by_meaning", lambda text, claims, features=None: [])
    store = _verified_light(tmp_path)
    assert [f.claim for f in store.bearing_on("Is the speed of light in a vacuum exactly 299,792,458 metres per second?")] == [LIGHT]
    assert store.bearing_on("How fast does light travel in a vacuum, exactly?") == []


def test_the_boundary_is_measured_from_the_declared_pairs():
    from core.epistemics import what_she_has_verified as w

    about = {sentence for pair in w._BEARS_ON for sentence in pair}

    def features(sentences):
        # A declared "about" pair lies along the first direction and every other declared sentence along the second,
        # so about-pairs score 1 and the rest 0; two undeclared requests lean towards and away from the claims.
        lean = {"towards": [1.0, 1.0], "away": [-1.0, 1.0]}
        return [[1.0, 0.0] if s in about else lean.get(s.split()[0], [0.0, 1.0]) for s in sentences]

    boundary = w._bearing_boundary(features)
    assert boundary.separable and (boundary.decide_true_above, boundary.decide_false_below) == (0.0, 1.0)
    claim = [w._BEARS_ON[0][1]]
    assert w._bearing_by_meaning("towards the claim", claim, features)[0][0] is True  # above every request not about it
    assert w._bearing_by_meaning("away from it", claim, features)[0][0] is False
