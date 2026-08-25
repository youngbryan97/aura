"""tests/test_cognitive_adaptations.py -- Automated tests for learnable assertiveness and dream fragments.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import pytest

from core.consciousness.dreaming import DreamingProcess
from core.governance.will import WillState, get_will


@dataclass
class WelfareTransactionRecordFixture:
    outcome: str = "success"
    welfare_delta: dict[str, float] = field(default_factory=dict)
    body_delta: dict[str, float] = field(default_factory=dict)
    integrity_preserved: bool = True
    truth_preserved: bool = True


@pytest.mark.asyncio
async def test_learnable_assertiveness():
    will = get_will()
    original_state = will._state
    will._state = WillState(assertiveness=0.5)
    try:
        # 1. Negative outcome: distress spike and integrity compromise
        bad_record = WelfareTransactionRecordFixture(
            outcome="failure",
            welfare_delta={"distress": 0.15},
            body_delta={"fatigue": 0.1},
            integrity_preserved=False,
            truth_preserved=True,
        )
        will.record_outcome("receipt_bad", bad_record)

        # Assertiveness should have decreased below 0.5
        assert will._state.assertiveness < 0.5

        # Save assertiveness state
        lowered_assertiveness = will._state.assertiveness

        # 2. Positive outcome: clean success with relief
        good_record = WelfareTransactionRecordFixture(
            outcome="success",
            welfare_delta={"relief": 0.1},
            body_delta={},
            integrity_preserved=True,
            truth_preserved=True,
        )
        will.record_outcome("receipt_good", good_record)

        # Assertiveness should have increased from the lowered value
        assert will._state.assertiveness > lowered_assertiveness
    finally:
        will._state = original_state


@pytest.mark.asyncio
async def test_dream_fragments(tmp_path):
    from core.config import Paths
    orig_cache = Paths._runtime_home_cache
    Paths._runtime_home_cache = tmp_path
    try:
        from core.container import ServiceContainer
        # Resolve dependencies through deterministic in-memory fixtures.
        class TestOrchestrator:
            def __init__(self):
                self._last_user_interaction_time = 0
                self.state_repo = None
                self.messages = []

            def enqueue_message(self, msg):
                self.messages.append(msg)

        class TestPhase:
            pass

        test_orchestrator = TestOrchestrator()

        # Test fragment writing
        from core.kernel.aura_kernel import AuraKernel
        # Since kernel initialization requires database, let's create a minimal test setup
        # and exercise the _record_dream_fragment function on a narrow test subclass.
        @dataclass
        class TestStatus:
            cycle_count: int = 1

        class TestKernel(AuraKernel):
            def __init__(self):
                self._phases = [TestPhase()]
                self.state = None
                self.status = TestStatus(cycle_count=12)

        kernel = TestKernel()

        fragment_file = tmp_path / "data" / "dream_fragments.jsonl"
        assert not fragment_file.exists()

        # Awaited. The recorder moved onto the async lane — every caller is a
        # preemption path with a user already waiting, so a blocking append
        # plus fsync was happening at the worst possible moment — and calling
        # it without awaiting produced a coroutine nobody ran and a file
        # nobody wrote.
        await kernel._record_dream_fragment(
            "Optimize swarms", kernel._phases[0], "TestPhase"
        )

        assert fragment_file.exists()
        fragment_text = await asyncio.to_thread(fragment_file.read_text, encoding="utf-8")
        lines = fragment_text.splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["objective"] == "Optimize swarms"
        assert data["preempted_at_phase"] == "TestPhase"
        assert "TestPhase" in data["completed_phases"]

        # Register deterministic identity/narrator services for DreamingProcess.
        class TestIdentity:
            def __init__(self):
                self.evolutions = []

            def record_evolution(self, source, reflection):
                self.evolutions.append((source, reflection))

        ServiceContainer.register_instance("identity_service", TestIdentity(), required=False)
        ServiceContainer.register_instance("narrator", object(), required=False)

        # Test dream process ingestion
        dp = DreamingProcess(test_orchestrator, interval=300.0)
        summary = await dp._get_recent_summary()

        assert "[Dream Fragment]" in summary
        assert "Optimize swarms" in summary
        assert "TestPhase" in summary

        # After get_recent_summary runs, it should clear/empty the fragment file
        remaining = (
            (await asyncio.to_thread(fragment_file.read_text, encoding="utf-8")).strip()
            if fragment_file.exists()
            else ""
        )
        assert remaining == ""

    finally:
        Paths._runtime_home_cache = orig_cache
