import copy

import pytest

from core.memory.temporal_rag import TimeWeightedRetriever


@pytest.mark.asyncio
async def test_facade_score_and_provenance_drive_recall(monkeypatch):
    now = 2_000_000_000.0
    monkeypatch.setattr("core.memory.temporal_rag.time.time", lambda: now)
    records = [
        {"text": "weak", "score": 0.1, "metadata": {"timestamp": now}},
        {"text": "episode", "score": 0.9, "metadata": {"timestamp": now - 86400}},
    ]
    before = copy.deepcopy(records)
    result = await TimeWeightedRetriever().rerank_and_format(records, limit=1)
    assert "[1 days ago] episode" in result
    assert "weak" not in result
    assert records == before


@pytest.mark.asyncio
@pytest.mark.parametrize("timestamp", [None, 0, float("nan"), "unknown"])
async def test_unknown_time_is_not_today(timestamp):
    result = await TimeWeightedRetriever().rerank_and_format([
        {"text": "remembered", "metadata": {"timestamp": timestamp}},
    ])
    assert "[Time unknown] remembered" in result
    assert "[Today]" not in result


def test_within_a_day_a_memory_says_how_many_minutes_or_hours_ago():
    """LIVE 8 November 2026: "[Today]" put a sitting three hours gone beside the one going on."""
    import time

    from core.memory.temporal_rag import TimeWeightedRetriever

    retriever = TimeWeightedRetriever()
    now = time.time()
    assert retriever._apply_seasonal_context("x", now - 30) == "[1 minute ago] x"
    assert retriever._apply_seasonal_context("x", now - 12 * 60) == "[12 minutes ago] x"
    assert retriever._apply_seasonal_context("x", now - 3 * 3600 - 60) == "[3 hours ago] x"
    assert retriever._apply_seasonal_context("x", now - 3600 - 60) == "[1 hour ago] x"
