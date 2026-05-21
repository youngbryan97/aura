from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.concept_linker import ConceptLinker


@pytest.mark.asyncio
async def test_batch_linking_prefers_contradiction_over_resonance():
    class Epistemic:
        def __init__(self):
            self.signaled = []

        def get_profile(self):
            return SimpleNamespace(
                strong_nodes=[
                    SimpleNamespace(concept="memory writes are durable"),
                    SimpleNamespace(concept="memory writes are not durable"),
                ],
                weak_nodes=[],
            )

        def signal_contradiction(self, a, b):
            self.signaled.append((a, b))

    class Challenger:
        def __init__(self):
            self.challenged = []

        async def challenge_pair(self, a, b):
            self.challenged.append((a, b))

    linker = ConceptLinker()
    linker._epistemic = Epistemic()
    linker._challenger = Challenger()

    await linker.run_batch_linking()

    assert len(linker._links) == 1
    assert linker._links[0].link_type == "contradiction"
    assert linker._epistemic.signaled
    assert linker._challenger.challenged


@pytest.mark.asyncio
async def test_downstream_failures_do_not_drop_link():
    class Epistemic:
        def signal_contradiction(self, a, b):
            if a and b:
                raise RuntimeError("epistemic signal offline")

    class Challenger:
        async def challenge_pair(self, a, b):
            if a and b:
                raise RuntimeError("challenger offline")

    class Journal:
        async def record_insight(self, **kwargs):
            if kwargs:
                raise RuntimeError("journal offline")

    linker = ConceptLinker()
    linker._epistemic = Epistemic()
    linker._challenger = Challenger()
    linker._journal = Journal()

    await linker._establish_link(
        "agency is coherent", "agency is not coherent", 0.95, "contradiction"
    )

    assert len(linker._links) == 1
    assert linker._links[0].strength == 0.95


@pytest.mark.asyncio
async def test_link_loop_survives_batch_failure():
    linker = ConceptLinker(scan_interval_seconds=0.01, sleep_slice_seconds=0.01)
    calls = 0

    async def fail_then_stop():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("batch failure")
        linker.running = False

    linker.run_batch_linking = fail_then_stop
    linker.running = True

    await asyncio.wait_for(linker._link_loop(), timeout=1.0)

    assert calls == 2


@pytest.mark.asyncio
async def test_batch_scan_respects_pair_budget():
    class Epistemic:
        def get_profile(self):
            return SimpleNamespace(
                strong_nodes=[
                    SimpleNamespace(concept="alpha shared concept"),
                    SimpleNamespace(concept="beta shared concept"),
                    SimpleNamespace(concept="gamma shared concept"),
                ],
                weak_nodes=[],
            )

    linker = ConceptLinker(max_batch_pairs=1)
    linker._epistemic = Epistemic()

    await linker.run_batch_linking()

    assert len(linker._links) <= 1
