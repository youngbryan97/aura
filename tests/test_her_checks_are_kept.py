"""L10 through her own check (core/epistemics/epistemic_reach.py): every permitted source is read and its verdict
kept; a claim two independent reputable sources support is verified; checked again, it is answered from what she
kept, with nothing fetched."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

CLAIM = "The Eiffel Tower opened in 1889 in Paris."


class _Source:
    def __init__(self, name: str, url: str, evidence: str) -> None:
        self.NAME, self.url, self.evidence, self.asked = name, url, evidence, 0

    def is_permitted(self, _policy) -> bool:
        return True

    async def lookup(self, _gateway, _terms):
        self.asked += 1
        return self.evidence, self.url


def test_two_independent_sources_verify_a_claim_and_it_is_not_fetched_again():
    from core.epistemics.epistemic_reach import VERDICT_SUPPORTED, EpistemicReachEngine
    from core.epistemics.what_she_has_verified import VERIFIED, her_verified_facts

    sources = [_Source("wikipedia", "https://en.wikipedia.org/wiki/Eiffel_Tower",
                       "The Eiffel Tower is a wrought-iron tower in Paris, opened in 1889."),
               _Source("culture ministry", "https://www.culture.gouv.fr/eiffel",
                       "In Paris the Eiffel Tower opened to the public in 1889.")]
    gateway = SimpleNamespace(policy=SimpleNamespace(read_hosts=frozenset(), mutate_hosts=frozenset()))
    engine = EpistemicReachEngine(gateway=gateway, sources=sources)
    felt = SimpleNamespace(fingerprint="f1", spikes=[])

    first = asyncio.run(engine._verify_async(CLAIM, felt))
    assert first.verdict == VERDICT_SUPPORTED and [s.asked for s in sources] == [1, 1]  # both read, not only the first
    kept = her_verified_facts().get(CLAIM)
    assert kept.status == VERIFIED and {a.publisher for a in kept.attestations} == {"wikimedia", "culture.gouv.fr"}

    again = asyncio.run(engine._verify_async(CLAIM, felt))
    assert again.verdict == VERDICT_SUPPORTED and "verified before" in again.reason
    assert [s.asked for s in sources] == [1, 1]  # answered from what she kept: nothing fetched


def test_one_source_alone_is_kept_and_not_verified():
    from core.epistemics.epistemic_reach import EpistemicReachEngine
    from core.epistemics.what_she_has_verified import SINGLE_SOURCE, her_verified_facts

    source = _Source("wikipedia", "https://en.wikipedia.org/wiki/Eiffel_Tower",
                     "The Eiffel Tower is a wrought-iron tower in Paris, opened in 1889.")
    gateway = SimpleNamespace(policy=SimpleNamespace(read_hosts=frozenset(), mutate_hosts=frozenset()))
    engine = EpistemicReachEngine(gateway=gateway, sources=[source])
    asyncio.run(engine._verify_async(CLAIM, SimpleNamespace(fingerprint="f1", spikes=[])))
    assert her_verified_facts().get(CLAIM).status == SINGLE_SOURCE
    asyncio.run(engine._verify_async(CLAIM, SimpleNamespace(fingerprint="f1", spikes=[])))
    assert source.asked == 2  # a single source settles nothing, so it is read again


def test_what_she_verified_reaches_her_next_turn_on_it():
    from core.brain.what_she_brings_to_a_turn import grounding_she_brings
    from core.epistemics.what_she_has_verified import SUPPORTED, her_verified_facts

    store = her_verified_facts()
    store.attest(CLAIM, "https://en.wikipedia.org/wiki/Eiffel_Tower", SUPPORTED)
    store.attest(CLAIM, "https://www.culture.gouv.fr/eiffel", SUPPORTED)
    blocks = grounding_she_brings({}, "When did the Eiffel Tower open in Paris?")
    assert len(blocks) == 1 and CLAIM in blocks[0] and "verified" in blocks[0]
    assert grounding_she_brings({}, "What is a good name for a cat?") == []


class _AllowlistGateway:
    """Stands in for the reach gateway: only allowlisted hosts answer, each with a canned page."""

    def __init__(self, read_hosts: set[str], pages: dict[str, str]) -> None:
        self.policy = SimpleNamespace(read_hosts=frozenset(read_hosts), mutate_hosts=frozenset())
        self.pages, self.asked = pages, []

    async def get(self, url: str, **_kw):
        from urllib.parse import urlparse

        self.asked.append(url)
        if urlparse(url).hostname not in self.policy.read_hosts:
            return SimpleNamespace(ok=False, body_preview="", status=0)
        body = next((page for fragment, page in self.pages.items() if fragment in url), None)
        return SimpleNamespace(ok=body is not None, body_preview=body or "", status=200 if body else 404)


def test_a_reputable_site_on_the_allowlist_is_a_second_publisher_and_a_claim_is_verified():
    import json

    from core.epistemics.epistemic_reach import VERDICT_SUPPORTED, EpistemicReachEngine
    from core.epistemics.searched_sources import searched_sources
    from core.epistemics.what_she_has_verified import VERIFIED, her_verified_facts

    britannica = "https://www.britannica.com/topic/Eiffel-Tower"
    pages = {
        "action=opensearch": json.dumps(["q", ["Eiffel Tower"], [], []]),
        "/page/summary/": json.dumps({"extract": "The Eiffel Tower is a wrought-iron tower in Paris, opened in 1889.",
                                      "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Eiffel_Tower"}}}),
        "html.duckduckgo.com": ('<a class="result__a" href="https://www.reddit.com/r/paris/eiffel">Eiffel?</a>'
                                f'<a class="result__a" href="{britannica}">Eiffel Tower | Britannica</a>'),
        "britannica.com/topic": "<html><body><p>The Eiffel Tower, in Paris, was opened in 1889 for the fair.</p></body></html>",
    }
    hosts = {"en.wikipedia.org", "html.duckduckgo.com", "www.britannica.com", "www.reddit.com"}
    gateway = _AllowlistGateway(hosts, pages)
    assert [s.NAME for s in searched_sources(gateway.policy)] == ["searched:open"]

    engine = EpistemicReachEngine(gateway=gateway)
    verdict = asyncio.run(engine._verify_async(CLAIM, SimpleNamespace(fingerprint="f1", spikes=[])))
    assert verdict.verdict == VERDICT_SUPPORTED
    kept = her_verified_facts().get(CLAIM)
    assert kept.status == VERIFIED and {a.publisher for a in kept.attestations} == {"wikimedia", "britannica.com"}
    from urllib.parse import urlparse

    assert {urlparse(u).hostname for u in gateway.asked} <= gateway.policy.read_hosts  # nothing off the allowlist asked
    assert not any("reddit" in u for u in gateway.asked), "a forum the engine found is not read, allowlisted or not"


def test_with_no_search_engine_allowlisted_a_site_is_not_searched():
    from core.epistemics.searched_sources import searched_sources

    policy = SimpleNamespace(read_hosts=frozenset({"www.britannica.com"}), mutate_hosts=frozenset())
    assert searched_sources(policy) == []


def test_the_operators_reputable_list_admits_its_hosts_and_what_is_under_them_for_reading_only(monkeypatch):
    from core.skills.reach_gateway import ReachPolicy

    monkeypatch.setenv("AURA_REACH_READ_HOSTS", "@reputable,example.org")
    policy = ReachPolicy()
    assert policy.reads("www.cdc.gov") and policy.reads("www.britannica.com") and policy.reads("plato.stanford.edu")
    assert policy.reads("html.duckduckgo.com") and policy.reads("example.org")
    assert not policy.reads("www.reddit.com") and not policy.reads("britannica.com.evil.net")
    assert policy.decide("POST", "www.cdc.gov")[0] is False
    monkeypatch.setenv("AURA_REACH_READ_HOSTS", "en.wikipedia.org")
    assert not ReachPolicy().reads("www.cdc.gov"), "without @reputable the list is not read"


def test_a_search_is_read_until_what_was_read_settles_the_claim():
    from core.epistemics.epistemic_reach import VERDICT_SUPPORTED, EpistemicReachEngine
    from core.epistemics.what_she_has_verified import VERIFIED, her_verified_facts

    found = ["https://www.britannica.com/topic/Eiffel-Tower", "https://www.culture.gouv.fr/eiffel",
             "https://www.nationalgeographic.com/eiffel", "https://www.reuters.com/eiffel"]
    page = "<html><body><p>The Eiffel Tower, in Paris, was opened in 1889 for the fair.</p></body></html>"
    pages = {"html.duckduckgo.com": "".join(f'<a class="result__a" href="{u}">Eiffel</a>' for u in found),
             "britannica.com/": page, "culture.gouv.fr/": page, "nationalgeographic.com/": page, "reuters.com/": page}
    gateway = _AllowlistGateway({"html.duckduckgo.com", "www.britannica.com", "www.culture.gouv.fr",
                                 "www.nationalgeographic.com", "www.reuters.com"}, pages)
    engine = EpistemicReachEngine(gateway=gateway)
    verdict = asyncio.run(engine._verify_async(CLAIM, SimpleNamespace(fingerprint="f1", spikes=[])))
    assert verdict.verdict == VERDICT_SUPPORTED and her_verified_facts().get(CLAIM).status == VERIFIED
    assert not any("nationalgeographic" in u or "reuters" in u for u in gateway.asked), "settled after two"


def test_a_passage_with_every_word_of_a_claim_that_does_not_state_it_does_not_support_it():
    """LIVE 19 October 2026: "Sydney is the capital of Australia" was held verified by two passages holding every
    word of it. Read by the entailment reader, the passage says otherwise."""
    from core.epistemics.epistemic_reach import (
        VERDICT_CONTRADICTED,
        VERDICT_SUPPORTED,
        EpistemicReachEngine,
    )
    from core.epistemics.how_a_source_reads import how_it_reads

    passage = "Sydney is the capital of the state of New South Wales and the largest city in Australia."
    if how_it_reads(passage, "Sydney is the capital of Australia.") is None:
        pytest.skip("no entailment reader on this machine")
    judged = EpistemicReachEngine.judge("Sydney is the capital of Australia.", passage, "https://en.wikipedia.org/wiki/Sydney")
    assert judged.verdict == VERDICT_CONTRADICTED
    stated = EpistemicReachEngine.judge("Mount Everest is 8,849 metres tall.",
                                        "Mount Everest's elevation of 8,849 m (29,032 ft) was most recently established in 2020.",
                                        "https://en.wikipedia.org/wiki/Mount_Everest")
    assert stated.verdict == VERDICT_SUPPORTED, "stated in other words: elevation for tall, m for metres"


def test_a_figure_rounded_as_the_claim_rounds_it_is_the_claims_figure():
    from core.epistemics.epistemic_reach import _figures_borne_out

    assert _figures_borne_out("Everest is 8,849 metres tall.", "Its elevation is 8,848.86 m.")
    assert not _figures_borne_out("Everest is 8,849 metres tall.", "Its elevation is 8,611 m.")
    assert _figures_borne_out("Pi is about 3.14.", "Pi is 3.14159.") and not _figures_borne_out("Pi is 3.15.", "Pi is 3.14159.")
