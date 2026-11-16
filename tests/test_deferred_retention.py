"""A deferral must not destroy completed work — and must not become a bypass.

Live evidence (2026-07-25 idle window): a web search ran for 10.8s, produced
facts and citations, and the whole artifact was discarded because the Will
deferred the memory write on welfare grounds. The log called it
``all memory backends rejected the artifact``, which was not true — nobody
rejected it, the runtime just said "later" and nothing was holding it.

Both halves are pinned here. The queue must hold "later" writes until they
land, and it must NOT retry a decided refusal — persistence past a veto is a
bypass, and that is the failure this file exists to prevent.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from core.memory.deferred_retention import (
    MAX_ATTEMPTS,
    MAX_ENTRIES,
    DeferredRetentionQueue,
    is_deferral,
)

pytestmark = pytest.mark.unit


class FakeFacade:
    """A memory facade that reports its verdicts the way the real one does."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self._last_add_memory_status = {"ok": False, "reason": "pending"}
        self.accepted: list[str] = []

    def add_memory(self, text, metadata=None):
        verdict = self._verdicts.pop(0) if self._verdicts else True
        if verdict is True:
            self._last_add_memory_status = {"ok": True, "reason": ""}
            self.accepted.append(text)
            return True
        self._last_add_memory_status = {"ok": False, "reason": verdict}
        return False


@pytest.fixture()
def queue(tmp_path):
    return DeferredRetentionQueue(queue_path=tmp_path / "deferred.jsonl")


class TestDeferralClassification:
    @pytest.mark.parametrize(
        "reason",
        [
            "aura_now_defer: present-state policy requires stabilization first",
            "welfare_recovery_required_before_action",
            "constitutional_gate_unavailable",
            "resource_busy",
            "unity_memory_defer",
        ],
    )
    def test_not_now_is_a_deferral(self, reason):
        assert is_deferral(reason)

    @pytest.mark.parametrize(
        "reason",
        [
            "constitutional_violation: identity rewrite",
            "content_rejected",
            "provenance_missing",
            "write_rejected",
            "",
        ],
    )
    def test_a_decided_no_is_not_a_deferral(self, reason):
        assert not is_deferral(reason)

    def test_an_override_beats_a_deferral_word(self):
        """Fail-closed: a refusal that happens to mention deferral is a refusal."""
        assert not is_deferral("constitutional_violation: defer to owner")


class TestHoldingDeferredWork:
    def test_a_deferred_write_is_held(self, queue):
        assert queue.enqueue("web learning", {"q": "x"}, reason="aura_now_defer")
        assert len(queue.pending()) == 1

    def test_a_refused_write_is_never_held(self, queue):
        assert not queue.enqueue("bad", reason="constitutional_violation")
        assert queue.pending() == []

    def test_holding_is_idempotent_by_content(self, queue):
        for _ in range(5):
            queue.enqueue("same artifact", reason="aura_now_defer")
        assert len(queue.pending()) == 1

    def test_the_queue_is_bounded(self, queue):
        for i in range(MAX_ENTRIES + 40):
            queue.enqueue(f"artifact {i}", reason="aura_now_defer")
        assert len(queue.pending()) <= MAX_ENTRIES

    def test_empty_text_is_not_held(self, queue):
        assert not queue.enqueue("   ", reason="aura_now_defer")

    def test_a_torn_line_loses_one_write_not_the_queue(self, queue):
        queue.enqueue("good one", reason="aura_now_defer")
        with queue.path.open("a", encoding="utf-8") as handle:
            handle.write("{not json\n")
        assert len(queue.pending()) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_write", [False, True])
    async def test_directory_setup_can_outlive_its_grant(self, queue, monkeypatch, async_write):
        from core.governance_context import get_active_governance
        from core.runtime.file_write_gateway import get_file_write_gateway

        monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")
        gateway = get_file_write_gateway()
        expired = []

        def expire_directory_grant():
            token = get_active_governance()
            assert token is not None and token.domain == "state_mutation"
            token.mono_timestamp -= token.ttl + 1
            assert token.expired
            expired.append(token.lease_id)

        if async_write:
            ensure = gateway.ensure_directory_async

            async def ensure_async_then_expire(*args, **kwargs):
                result = await ensure(*args, **kwargs)
                expire_directory_grant()
                return result

            monkeypatch.setattr(gateway, "ensure_directory_async", ensure_async_then_expire)
            held = await queue.enqueue_async("held across a slow directory setup", reason="aura_now_defer")
        else:
            ensure = gateway.ensure_directory

            def ensure_then_expire(*args, **kwargs):
                result = ensure(*args, **kwargs)
                expire_directory_grant()
                return result

            monkeypatch.setattr(gateway, "ensure_directory", ensure_then_expire)
            held = await asyncio.to_thread(queue.enqueue, "held across a slow directory setup", reason="aura_now_defer")

        assert held and len(expired) == 1
        pending = await asyncio.to_thread(queue.pending)
        assert [entry["text"] for entry in pending] == ["held across a slow directory setup"]


class TestReplay:
    @pytest.mark.asyncio
    async def test_work_deferred_once_lands_later(self, queue):
        queue.enqueue("hard-won research", {"source": "web_search"}, reason="aura_now_defer")
        facade = FakeFacade([True])

        report = await queue.replay(facade)

        assert report.committed == 1
        assert facade.accepted == ["hard-won research"]
        assert queue.pending() == [], "a landed write must not be replayed forever"

    @pytest.mark.asyncio
    async def test_still_deferred_work_keeps_waiting(self, queue):
        queue.enqueue("research", reason="aura_now_defer")

        report = await queue.replay(FakeFacade(["welfare_recovery_required_before_action"]))

        assert report.still_deferred == 1
        assert len(queue.pending()) == 1

    @pytest.mark.asyncio
    async def test_a_refusal_on_retry_drops_the_entry(self, queue):
        """The gate can change its mind toward NO; persistence must not fight it."""
        queue.enqueue("research", reason="aura_now_defer")

        report = await queue.replay(FakeFacade(["constitutional_violation"]))

        assert report.refused == 1
        assert queue.pending() == []

    @pytest.mark.asyncio
    async def test_an_exception_is_transient_not_a_verdict(self, queue):
        class Exploding:
            _last_add_memory_status: dict = {}

            def add_memory(self, text, metadata=None):
                raise RuntimeError("backend down")

        queue.enqueue("research", reason="aura_now_defer")

        report = await queue.replay(Exploding())

        assert report.still_deferred == 1
        assert len(queue.pending()) == 1

    @pytest.mark.asyncio
    async def test_forever_deferred_work_ages_out(self, queue):
        queue.enqueue("research", reason="aura_now_defer")
        entries = queue.pending()
        entries[0]["attempts"] = MAX_ATTEMPTS
        queue._store(entries)

        report = await queue.replay(FakeFacade(["aura_now_defer"]))

        assert report.expired == 1
        assert queue.pending() == []

    @pytest.mark.asyncio
    async def test_stale_work_ages_out_by_wall_clock(self, queue):
        queue.enqueue("research", reason="aura_now_defer")
        entries = queue.pending()
        entries[0]["queued_at"] = time.time() - (8 * 24 * 3600)
        queue._store(entries)

        report = await queue.replay(FakeFacade([True]))

        assert report.expired == 1

    @pytest.mark.asyncio
    async def test_no_facade_means_keep_holding(self, queue):
        queue.enqueue("research", reason="aura_now_defer")

        report = await queue.replay(object())

        assert report.still_deferred == 1
        assert len(queue.pending()) == 1

    @pytest.mark.asyncio
    async def test_empty_queue_replay_is_a_no_op(self, queue):
        report = await queue.replay(FakeFacade([]))
        assert report.as_dict() == {
            "committed": 0, "still_deferred": 0, "refused": 0,
            "expired": 0, "remaining": 0,
        }
        assert "no deferred memory writes" in report.narrative()

    @pytest.mark.asyncio
    async def test_a_mixed_batch_is_reported_honestly(self, queue):
        for i, _ in enumerate(range(3)):
            queue.enqueue(f"artifact {i}", reason="aura_now_defer")

        report = await queue.replay(
            FakeFacade([True, "aura_now_defer", "content_rejected"])
        )

        assert (report.committed, report.still_deferred, report.refused) == (1, 1, 1)
        assert report.remaining == 1
        assert "1 write(s) finally landed" in report.narrative()


class TestResearchPipelineIntegration:
    @pytest.mark.asyncio
    async def test_deferred_research_is_held_rather_than_discarded(
        self, tmp_path, monkeypatch
    ):
        """The exact live shape: every backend declines, artifact must survive."""
        from core.memory import deferred_retention
        from core.search.research_pipeline import ResearchSearchPipeline, SearchArtifact

        monkeypatch.setattr(
            deferred_retention, "_QUEUE",
            DeferredRetentionQueue(queue_path=tmp_path / "held.jsonl"),
        )
        held = deferred_retention.get_deferred_retention_queue()

        pipeline = ResearchSearchPipeline.__new__(ResearchSearchPipeline)
        pipeline.artifact_store = []
        now = time.time()
        artifact = SearchArtifact(
            artifact_id="a1",
            query="what changed in the runtime",
            normalized_query="what changed in the runtime",
            answer="a real answer that cost 10 seconds of network work",
            summary="summary",
            facts=["fact one", "fact two"],
            citations=[{"title": "src", "url": "https://example.invalid"}],
            evidence=[],
            created_at=now,
            updated_at=now,
            freshness_seconds=3600,
            confidence=0.7,
            current=True,
            source="web_search",
        )

        facade = FakeFacade(["aura_now_defer: requires stabilization first"])
        await pipeline._retain_artifact(
            artifact, {"memory_facade": facade, "semantic_memory": object()}
        )

        pending = held.pending()
        assert len(pending) == 1, "completed research must not be discarded"
        assert "a real answer" in pending[0]["text"]
        assert pending[0]["origin"] == "research_pipeline"
