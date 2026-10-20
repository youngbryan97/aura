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
