import time
from types import SimpleNamespace

from core.memory.episodic_memory import EpisodicMemory


def test_recall_similar_skips_keyword_fallback_when_vector_results_suffice(tmp_path):
    class _VectorMemory:
        def search_similar(self, query, k, filter_metadata=None):
            return [
                {"metadata": {"episode_id": "ep-a"}},
                {"metadata": {"episode_id": "ep-b"}},
            ]

    memory = EpisodicMemory(db_path=str(tmp_path / "episodic.db"), vector_memory=_VectorMemory())
    episodes = [
        SimpleNamespace(episode_id="ep-a", importance=0.7, timestamp=time.time()),
        SimpleNamespace(episode_id="ep-b", importance=0.6, timestamp=time.time() - 1),
    ]

    memory._fetch_by_ids = lambda episode_ids: [ep for ep in episodes if ep.episode_id in episode_ids]
    memory._keyword_search = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("keyword fallback should be skipped"))

    result = memory.recall_similar("summarize our recent continuity work", limit=2)

    assert [ep.episode_id for ep in result] == ["ep-a", "ep-b"]


def test_recall_similar_keeps_keyword_fallback_for_exact_recall_queries(tmp_path):
    class _VectorMemory:
        def search_similar(self, query, k, filter_metadata=None):
            return [
                {"metadata": {"episode_id": "ep-a"}},
                {"metadata": {"episode_id": "ep-b"}},
            ]

    memory = EpisodicMemory(db_path=str(tmp_path / "episodic.db"), vector_memory=_VectorMemory())
    called = {"keyword": 0}
    episodes = [
        SimpleNamespace(episode_id="ep-a", importance=0.7, timestamp=time.time()),
        SimpleNamespace(episode_id="ep-b", importance=0.6, timestamp=time.time() - 1),
    ]

    memory._fetch_by_ids = lambda episode_ids: [ep for ep in episodes if ep.episode_id in episode_ids]

    def _keyword_search(*_args, **_kwargs):
        called["keyword"] += 1
        return []

    memory._keyword_search = _keyword_search

    memory.recall_similar('What did I tell you? Give me the exact words.', limit=2)

    assert called["keyword"] == 1
