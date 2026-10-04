import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.autonomy.research_cycle import ResearchCycle
from core.search.research_pipeline import (
    ResearchSearchPipeline,
    SearchArtifact,
    SearchArtifactStore,
    SearchHit,
    SearchPage,
)


@pytest.mark.asyncio
async def test_search_pipeline_reuses_fresh_retained_artifact(tmp_path: Path, monkeypatch):
    store = SearchArtifactStore(tmp_path / "web_artifacts.jsonl")
    pipeline = ResearchSearchPipeline(store)
    now = time.time()
    artifact = SearchArtifact(
        artifact_id="artifact123",
        query="rayleigh scattering",
        normalized_query="rayleigh scattering",
        answer="Rayleigh scattering makes the sky appear blue.",
        summary="Rayleigh scattering makes blue wavelengths scatter more strongly.",
        facts=["Shorter wavelengths scatter more strongly in the atmosphere."],
        citations=[{"title": "Example", "url": "https://example.com/rayleigh"}],
        evidence=[
            {
                "title": "Example",
                "url": "https://example.com/rayleigh",
                "text": "Rayleigh scattering explains why the sky looks blue during the day.",
                "score": 0.91,
            }
        ],
        created_at=now,
        updated_at=now,
        freshness_seconds=24 * 60 * 60,
        confidence=0.82,
        current=False,
        source="https://example.com/rayleigh",
    )
    store.append(artifact)

    search_calls = []

    async def _unexpected_search(*args, **kwargs):
        search_calls.append((args, kwargs))
        raise AssertionError("live search should not run when a fresh retained artifact exists")

    monkeypatch.setattr(pipeline, "_search_candidates", _unexpected_search)

    result = await pipeline.search("rayleigh scattering", context={})

    assert result["ok"] is True
    assert result["cached"] is True
    assert "sky appear blue" in result["answer"]


@pytest.mark.asyncio
async def test_search_pipeline_cached_artifact_reports_retained_when_requested(tmp_path: Path):
    store = SearchArtifactStore(tmp_path / "web_artifacts.jsonl")
    pipeline = ResearchSearchPipeline(store)
    now = time.time()
    artifact = SearchArtifact(
        artifact_id="artifact-retained",
        query="python 3.12 release notes",
        normalized_query="python 3.12 release notes",
        answer="Python 3.12 adds type parameter syntax and performance improvements.",
        summary="Python 3.12 adds type parameter syntax and performance improvements.",
        facts=["Python 3.12 introduces PEP 695 style type parameter syntax."],
        citations=[{"title": "Release Notes", "url": "https://docs.python.org/3/whatsnew/3.12.html"}],
        evidence=[
            {
                "title": "Release Notes",
                "url": "https://docs.python.org/3/whatsnew/3.12.html",
                "text": "Python 3.12 introduces new syntax for type parameters and other improvements.",
                "score": 0.95,
            }
        ],
        created_at=now,
        updated_at=now,
        freshness_seconds=24 * 60 * 60,
        confidence=0.91,
        current=True,
        source="https://docs.python.org/3/whatsnew/3.12.html",
    )
    store.append(artifact)

    result = await pipeline.search("python 3.12 release notes", retain=True, context={})

    assert result["ok"] is True
    assert result["cached"] is True
    assert result["retained"] is True
    assert result["artifact_id"] == "artifact-retained"


class _FakeSemanticMemory:
    def __init__(self):
        self.entries = []

    async def remember(self, content, metadata=None):
        self.entries.append((content, metadata or {}))


@pytest.mark.asyncio
async def test_search_pipeline_retains_successful_search(tmp_path: Path):
    store = SearchArtifactStore(tmp_path / "web_artifacts.jsonl")
    pipeline = ResearchSearchPipeline(store)
    semantic_memory = _FakeSemanticMemory()

    text = (
        "Rayleigh scattering causes shorter wavelengths of visible light to scatter more strongly "
        "than longer wavelengths in the atmosphere. This is why the daytime sky often appears blue. "
        "At sunrise and sunset the light travels through more atmosphere, so red and orange wavelengths "
        "become more prominent."
    )

    async def _expand_queries(query, context):
        return [query]

    async def _search_candidates(queries, *, num_results):
        return [
            SearchHit(
                title="Rayleigh scattering",
                url="https://example.com/rayleigh",
                snippet="Why the sky appears blue.",
                source_engine="test",
                position=1,
            )
        ]

    async def _fetch_pages(hits, *, deep):
        return [
            SearchPage(
                url="https://example.com/rayleigh",
                title="Rayleigh scattering",
                text=text,
                snippet="Why the sky appears blue.",
                source_engine="test",
                position=1,
            )
        ]

    pipeline._expand_queries = _expand_queries  # type: ignore[method-assign]
    pipeline._search_candidates = _search_candidates  # type: ignore[method-assign]
    pipeline._fetch_pages = _fetch_pages  # type: ignore[method-assign]

    result = await pipeline.search(
        "rayleigh scattering",
        deep=True,
        retain=True,
        context={"semantic_memory": semantic_memory, "origin": "research_cycle"},
    )

    retained = store.find_best("rayleigh scattering", freshness_seconds=24 * 60 * 60)

    assert result["ok"] is True
    assert result["retained"] is True
    assert result["chunks"][0]["evidence_kind"] == "article_body"
    assert result["chunks"][0]["fetched"] is True
    assert len(result["chunks"][0]["document_sha256"]) == 64
    assert retained is not None
    assert semantic_memory.entries
    note, metadata = semantic_memory.entries[0]
    assert "Rayleigh scattering" in note
    assert metadata["source"] == "web_search"


@pytest.mark.asyncio
async def test_search_pipeline_skips_ddgs_when_runtime_disables_it(tmp_path: Path, monkeypatch):
    store = SearchArtifactStore(tmp_path / "web_artifacts.jsonl")
    pipeline = ResearchSearchPipeline(store)
    calls = {"ddgs": 0, "legacy": 0}

    def _unexpected_ddgs(*args, **kwargs):
        calls["ddgs"] += 1
        return []

    def _fake_legacy(query, num_results):
        calls["legacy"] += 1
        return [
            SearchHit(
                title="Example",
                url="https://example.com/result",
                snippet="Fallback result",
                source_engine="test",
                position=1,
            )
        ]

    monkeypatch.setattr("core.search.research_pipeline._ddgs_enabled", lambda: False)
    monkeypatch.setattr(pipeline, "_ddgs_search", _unexpected_ddgs)
    monkeypatch.setattr(pipeline, "_legacy_html_search", _fake_legacy)

    hits = await pipeline._search_candidates(["fallback query"], num_results=1)

    assert calls["ddgs"] == 0
    assert calls["legacy"] == 1
    assert hits[0].url == "https://example.com/result"


@pytest.mark.asyncio
async def test_expanded_candidate_searches_run_concurrently(tmp_path: Path, monkeypatch):
    pipeline = ResearchSearchPipeline(SearchArtifactStore(tmp_path / "web_artifacts.jsonl"))
    barrier = threading.Barrier(2, timeout=2.0)

    def _fake_legacy(query, _num_results):
        barrier.wait()
        return [
            SearchHit(
                title=query,
                url=f"https://example.com/{query}",
                snippet="Independent result",
                source_engine="test",
                position=1,
            )
        ]

    monkeypatch.setattr("core.search.research_pipeline._ddgs_enabled", lambda: False)
    monkeypatch.setattr(pipeline, "_legacy_html_search", _fake_legacy)

    hits = await pipeline._search_candidates(["one", "two"], num_results=2)

    assert [hit.title for hit in hits] == ["one", "two"]


@pytest.mark.asyncio
async def test_deep_fetch_recovers_only_missing_pages_in_parallel(tmp_path: Path, monkeypatch):
    pipeline = ResearchSearchPipeline(SearchArtifactStore(tmp_path / "web_artifacts.jsonl"))
    hits = [
        SearchHit(
            title=f"Article {index}",
            url=f"https://example.com/{index}",
            snippet="Article snippet",
            source_engine="test",
            position=index,
        )
        for index in range(1, 4)
    ]
    active = 0
    max_active = 0
    browser_urls = []

    async def _fetch_page(_client, hit, *, timeout_val):
        del timeout_val
        if hit is hits[0]:
            return SearchPage(
                url=hit.url,
                title=hit.title,
                text="complete article body " * 80,
                snippet=hit.snippet,
                source_engine="test",
                position=hit.position,
            )
        return None

    async def _fetch_browser(hit):
        nonlocal active, max_active
        browser_urls.append(hit.url)
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        active -= 1
        return SearchPage(
            url=hit.url,
            title=hit.title,
            text="browser recovered article body " * 50,
            snippet=hit.snippet,
            source_engine="browser:test",
            position=hit.position,
        )

    monkeypatch.setattr(pipeline, "_fetch_page", _fetch_page)
    monkeypatch.setattr(pipeline, "_fetch_page_with_browser", _fetch_browser)

    pages = await pipeline._fetch_pages(hits, deep=True)

    assert {page.url for page in pages} == {hit.url for hit in hits}
    assert hits[0].url not in browser_urls
    assert set(browser_urls) == {hits[1].url, hits[2].url}
    assert max_active == 2


_WIKIPEDIA_LIKE_PAGE = (
    "<html><head><title>Kaseya Center - Wikipedia</title></head><body>"
    + "".join(
        f"<li>{label}</li>"
        for label in (
            "Jump to content", "Main menu", "Navigation", "Main page", "Contents",
            "Current events", "Random article", "About Wikipedia", "Contact us",
            "Search", "Donate", "Create account", "Log in", "Personal tools",
        )
    )
    + "<p>Kaseya Center is a multi-purpose arena on Biscayne Bay in Miami, Florida. "
    "It opened on December 31, 1999, and is the home of the Miami Heat.</p>"
    "<p>The arena was designed by Arquitectonica and 360 Architecture. "
    "Construction cost about $213 million, paid in part by the county.</p>"
    + "".join(f"<li>{label}</li>" for label in ("Privacy policy", "Disclaimers", "Mobile view"))
    + "</body></html>"
)


def _gateway_returning(body: str, content_type: str):
    class _Gateway:
        def request(self, *_args, **_kwargs):
            return {
                "ok": True,
                "status_code": 200,
                "headers": {"content-type": content_type},
                "content": body.encode("utf-8"),
            }

    return _Gateway()


@pytest.mark.asyncio
async def test_a_fetched_page_keeps_its_prose_not_its_menus(tmp_path: Path, monkeypatch):
    # LIVE 2026-10-03: research she kept from a search began "Kaseya Center -
    # Wikipedia Jump to content Main menu ..." because the page text was the
    # whole converted HTML, navigation first.
    import core.search.research_pipeline as research_pipeline

    monkeypatch.setattr(
        research_pipeline,
        "get_network_gateway",
        lambda: _gateway_returning(_WIKIPEDIA_LIKE_PAGE, "text/html; charset=utf-8"),
    )
    pipeline = ResearchSearchPipeline(SearchArtifactStore(tmp_path / "web_artifacts.jsonl"))
    hit = SearchHit(title="Kaseya Center", url="https://en.wikipedia.org/wiki/Kaseya_Center")

    page = await pipeline._fetch_page(None, hit, timeout_val=1.0)

    assert page is not None
    assert page.text.startswith("Kaseya Center is a multi-purpose arena")
    assert "Main menu" not in page.text
    assert "Privacy policy" not in page.text
    assert page.read_chars > len(page.text)


@pytest.mark.asyncio
async def test_a_json_body_is_kept_whole(tmp_path: Path, monkeypatch):
    import core.search.research_pipeline as research_pipeline

    body = '{"name": "Kaseya Center", "opened": "1999-12-31"}\n' + '{"note": "x"}\n' * 40
    monkeypatch.setattr(
        research_pipeline,
        "get_network_gateway",
        lambda: _gateway_returning(body, "application/json"),
    )
    pipeline = ResearchSearchPipeline(SearchArtifactStore(tmp_path / "web_artifacts.jsonl"))
    hit = SearchHit(title="Kaseya Center", url="https://example.com/arena.json")

    page = await pipeline._fetch_page(None, hit, timeout_val=1.0)

    assert page is not None
    assert page.text.startswith('{"name": "Kaseya Center"')
    assert page.read_chars == len(page.text)


@pytest.mark.asyncio
async def test_deep_fetch_judges_completeness_by_what_was_read_not_the_prose(
    tmp_path: Path, monkeypatch
):
    pipeline = ResearchSearchPipeline(SearchArtifactStore(tmp_path / "web_artifacts.jsonl"))
    hit = SearchHit(title="Short news", url="https://example.com/news", position=1)
    browser_urls = []

    async def _fetch_page(_client, hit, *, timeout_val):
        del timeout_val
        return SearchPage(url=hit.url, title=hit.title, text="A short story. " * 20, read_chars=4000)

    async def _fetch_browser(hit):
        browser_urls.append(hit.url)
        return None

    monkeypatch.setattr(pipeline, "_fetch_page", _fetch_page)
    monkeypatch.setattr(pipeline, "_fetch_page_with_browser", _fetch_browser)

    pages = await pipeline._fetch_pages([hit], deep=True)

    assert [page.url for page in pages] == [hit.url]
    assert browser_urls == []


@pytest.mark.asyncio
async def test_evidence_only_search_skips_model_work_and_preserves_source_diversity(
    tmp_path: Path,
    monkeypatch,
):
    pipeline = ResearchSearchPipeline(SearchArtifactStore(tmp_path / "web_artifacts.jsonl"))
    observed = {}

    async def _no_reason(*_args, **_kwargs):
        raise AssertionError("evidence-only retrieval must not allocate a synthesis model")

    async def _search_candidates(queries, *, num_results):
        observed["queries"] = list(queries)
        observed["num_results"] = num_results
        return [
            SearchHit(
                title=f"Recent article {index}",
                url=f"https://example.com/article-{index}",
                snippet="Recent independent reporting",
                source_engine="test",
                position=index,
            )
            for index in range(1, 4)
        ]

    async def _fetch_pages(hits, *, deep):
        observed["fetched"] = len(hits)
        observed["deep"] = deep
        return [
            SearchPage(
                url=hit.url,
                title=hit.title,
                text=(
                    f"Article {hit.position} reports a distinct current finding about "
                    "orca cognition, social learning, and cooperative behavior. " * 12
                ),
                snippet=hit.snippet,
                source_engine="test",
                position=hit.position,
            )
            for hit in hits
        ]

    monkeypatch.setattr(pipeline, "_reason", _no_reason)
    monkeypatch.setattr(pipeline, "_search_candidates", _search_candidates)
    monkeypatch.setattr(pipeline, "_fetch_pages", _fetch_pages)

    result = await pipeline.search(
        "find 3 recent articles about orca cognition",
        num_results=3,
        deep=True,
        retain=False,
        force_refresh=True,
        context={"evidence_only": True},
    )

    assert result["ok"] is True
    assert result["synthesis_mode"] == "deterministic_evidence"
    assert set(result["timing_ms"]) == {
        "query_expansion",
        "candidate_search",
        "source_fetch",
        "synthesis",
        "total",
    }
    assert observed == {
        "queries": ["find 3 recent articles about orca cognition"],
        "num_results": 3,
        "fetched": 3,
        "deep": True,
    }
    assert len({chunk["url"] for chunk in result["chunks"][:3]}) == 3


@pytest.mark.asyncio
async def test_research_cycle_integrates_findings_into_semantic_memory(monkeypatch):
    kg_entries = []
    semantic_entries = []
    state = SimpleNamespace(cognition=SimpleNamespace(long_term_memory=[]))

    class _FakeKG:
        def add_knowledge(self, *, content, type, source, confidence):
            kg_entries.append((content, type, source, confidence))

    class _FakeSemantic:
        async def remember(self, content, metadata=None):
            semantic_entries.append((content, metadata or {}))

    services = {
        "knowledge_graph": _FakeKG(),
        "memory_facade": None,
        "semantic_memory": _FakeSemantic(),
    }

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: services.get(name, default),
    )

    cycle = ResearchCycle(SimpleNamespace())
    cycle._get_state = lambda: state  # type: ignore[method-assign]

    await cycle._integrate_knowledge(
        ["Rayleigh scattering makes short wavelengths scatter more strongly."],
        "Research and learn something new about atmospheric optics",
        "curiosity",
    )

    assert kg_entries
    assert semantic_entries
    assert state.cognition.long_term_memory
