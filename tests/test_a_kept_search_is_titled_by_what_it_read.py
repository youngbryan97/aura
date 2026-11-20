"""A kept web search is titled by the pages it read, never by the words it was asked (U07, attribution).

LIVE 19 November 2026, typed into her window after a restart: asked "What is the locker code for the Brimvaltor
depot?", she answered 6182, which she had been told before the restart, and gave a game guide on polygon.com as its
source. Earlier that evening a search had run on "Please remember this for later: the locker code for the Brimvaltor
depot is 6182", and her research pipeline kept it in her corpus titled by that query, beside the pages it read. Read
back, the person's own words stood as the title of a page. Nine kept searches were titled that way, all from requests to
remember. A kept search is now read back titled by the pages it cites, without its query, and kept that way from now on.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from core.conversation.what_the_world_says import _from_corpus, what_a_kept_search_read
from core.knowledge import local_corpus
from core.knowledge.local_corpus import RETAINED_SOURCE, LocalCorpusStore

pytestmark = pytest.mark.unit

ASKED = "Please remember this for later: the locker code for the Brimvaltor depot is 6182"
PAGE = {"title": "Modern Warfare 4 DMZ keys and locations", "url": "https://www.polygon.com/mw4-dmz-keys/"}
NOTE = (f"[WebLearning 2026-11-19 17:12:21] Query: {ASKED}\n"
        "Answer: DMZ keys open locked rooms across the map.\n"
        "Facts:\n- The Brimvaltor area has no keyed lockers in the game.\n"
        f"Sources:\n- {PAGE['title']}: {PAGE['url']}")


@pytest.fixture
def corpus(tmp_path):
    store = LocalCorpusStore(tmp_path / "corpus.db")
    store.add_documents([("Depot", "A depot is a place where goods are stored.", "wikipedia")])
    return store


def _kept_source(store):
    found = _from_corpus(["Brimvaltor depot locker code"], ["brimvaltor", "locker"], store=store)
    kept = [source for source in found if source.origin.startswith("kept from a web search")]
    assert len(kept) == 1, found
    return kept[0]


def test_a_note_kept_under_its_query_is_read_back_titled_by_its_pages(corpus):
    corpus.add_retained_document(ASKED.lower(), NOTE, artifact_id="before-the-fix")
    kept = _kept_source(corpus)
    assert kept.title == PAGE["title"] and kept.location == PAGE["url"]
    assert "6182" not in kept.title + kept.text + kept.full_text, "the person's words are not the page's"


def test_a_note_with_no_page_cited_is_not_titled_by_what_it_asked():
    title, read = what_a_kept_search_read(f"[WebLearning 2026-11-19 17:12:21] Query: {ASKED}\nAnswer: none\nSources:\n"
                                          "- No citations recorded")
    assert "6182" not in title + read


def test_a_search_kept_from_now_on_is_titled_by_what_it_read(corpus, monkeypatch):
    from core.search.research_pipeline import ResearchSearchPipeline, SearchArtifact

    monkeypatch.setattr(local_corpus, "get_local_corpus_store", lambda *_args, **_kwargs: corpus)
    pipeline = ResearchSearchPipeline.__new__(ResearchSearchPipeline)
    pipeline.artifact_store = []
    now = time.time()
    artifact = SearchArtifact(
        artifact_id="kept-after", query=ASKED, normalized_query=ASKED.lower(),
        answer="DMZ keys open locked rooms across the map.", summary="", facts=[], citations=[PAGE], evidence=[],
        created_at=now, updated_at=now, freshness_seconds=3600, confidence=0.6, current=False, source="web_search")
    asyncio.run(pipeline._retain_artifact(artifact, {"memory_facade": None, "semantic_memory": object()}))
    titles = [title for (title,) in corpus._connect_ro().execute(
        "SELECT title FROM docs WHERE source = ?", (RETAINED_SOURCE,))]
    assert titles == [PAGE["title"]]
