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

    async def _unexpected_search(*args, **kwargs):
        raise AssertionError("live search should not run when a fresh retained artifact exists")

    monkeypatch.setattr(pipeline, "_search_candidates", _unexpected_search)

    result = await pipeline.search("rayleigh scattering", context={})

    assert result["ok"] is True
    assert result["cached"] is True
    assert "sky appear blue" in result["answer"]


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
async def test_research_cycle_integrates_findings_into_semantic_memory(monkeypatch):
    kg_entries = []
    semantic_entries = []
    state = SimpleNamespace(cognition=SimpleNamespace(long_term_memory=[]))

    class _FakeKG:
        def add_knowledge(self, *, content, source, confidence):
            kg_entries.append((content, source, confidence))

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
