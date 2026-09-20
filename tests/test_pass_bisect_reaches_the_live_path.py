"""AURA_PASS_BISECT_LIMIT has to work on the pipeline that serves traffic.

Two defects, both of which made the documented debugging entry point do
nothing where it was needed:

1. ``should_run()`` was consulted by ``core/pipeline/pass_manager.py`` and by
   the kernel tick. Chat drives the legacy phase loop in
   ``core/brain/cognitive_engine.py`` — by roughly 479 turns to 3 — and that
   loop never asked. The good instrumentation was on the pipeline that
   mostly is not running.

2. Pass numbering was monotonic for the process, so ``BISECT_LIMIT=5`` meant
   "the first five passes since boot". Correct on the first turn, and total
   silence on every turn after it. Worse than not having the knob, because
   the second turn's empty output looks like a finding.
"""
from __future__ import annotations

import asyncio

import pytest

from core.pipeline.pass_manager import get_instrumentation
from tests.source_contract import family_text


@pytest.fixture
def instrumentation():
    inst = get_instrumentation()
    original_limit = inst.bisect_limit()
    original_trace = inst._trace
    yield inst
    inst.set_bisect_limit(original_limit)
    inst.set_trace(original_trace)


def _run_a_pipeline(inst, phases: list[str]) -> list[str]:
    """Drive `phases` through the seam exactly as a phase loop does."""
    inst.begin_run("test_pipeline")
    ran = []
    for name in phases:
        should, _ordinal, _reason = inst.should_run(name)
        if should:
            ran.append(name)
    return ran


class TestNumberingIsPerTurn:
    def test_the_limit_means_the_same_thing_on_every_turn(self, instrumentation):
        phases = [f"phase_{i}" for i in range(8)]
        instrumentation.set_bisect_limit(3)

        first = _run_a_pipeline(instrumentation, phases)
        second = _run_a_pipeline(instrumentation, phases)
        third = _run_a_pipeline(instrumentation, phases)

        assert first == ["phase_0", "phase_1", "phase_2"]
        assert second == first, "turn 2 numbered from where turn 1 stopped"
        assert third == first

    def test_without_begin_run_numbering_would_have_run_away(self, instrumentation):
        """The old behaviour, shown directly: no reset, so turn 2 is silent."""
        phases = [f"phase_{i}" for i in range(4)]
        instrumentation.set_bisect_limit(3)
        instrumentation.begin_run("once")

        first = [n for n in phases if instrumentation.should_run(n)[0]]
        # Deliberately NOT calling begin_run again.
        second = [n for n in phases if instrumentation.should_run(n)[0]]

        assert first == ["phase_0", "phase_1", "phase_2"]
        assert second == [], "this is what every turn after the first used to do"

    def test_no_limit_runs_everything(self, instrumentation):
        instrumentation.set_bisect_limit(None)
        phases = [f"phase_{i}" for i in range(6)]
        assert _run_a_pipeline(instrumentation, phases) == phases

    def test_a_limit_of_zero_runs_nothing(self, instrumentation):
        instrumentation.set_bisect_limit(0)
        assert _run_a_pipeline(instrumentation, ["a", "b"]) == []

    def test_two_turns_in_flight_number_their_own_passes(self, instrumentation):
        """Concurrency: the counter is per-task, not shared."""
        instrumentation.set_bisect_limit(2)
        phases = [f"phase_{i}" for i in range(5)]

        async def one_turn():
            instrumentation.begin_run("concurrent")
            ran = []
            for name in phases:
                should, _o, _r = instrumentation.should_run(name)
                if should:
                    ran.append(name)
                await asyncio.sleep(0)  # interleave with the other turn
            return ran

        async def both():
            return await asyncio.gather(one_turn(), one_turn())

        left, right = asyncio.run(both())
        assert left == ["phase_0", "phase_1"]
        assert right == ["phase_0", "phase_1"]


class TestTheLivePathConsultsTheSeam:
    """Structural, because standing up the whole legacy engine in a unit test
    would prove less than it costs. What must be true is that the loop asks."""

    def test_the_legacy_phase_loop_calls_should_run(self):
        from core.brain import cognitive_engine

        source = family_text(cognitive_engine)
        assert "_pass_instrumentation().should_run(" in source, (
            "the loop that serves chat does not consult the pass seam; "
            "AURA_PASS_BISECT_LIMIT does nothing on the live path"
        )

    def test_the_legacy_phase_loop_restarts_numbering_each_turn(self):
        from core.brain import cognitive_engine

        source = family_text(cognitive_engine)
        assert '_begin_pass_run("legacy_pipeline")' in source

    def test_the_kernel_tick_restarts_numbering_each_tick(self):
        from core.kernel import aura_kernel

        source = family_text(aura_kernel)
        assert "_begin_pass_run(" in source

    def test_both_pipelines_record_into_the_same_ledger(self, instrumentation):
        """One report, or the trace tells you about the wrong pipeline."""
        from core.brain.cognitive_engine import _record_legacy_pass

        before = len(instrumentation.records())
        _record_legacy_pass("ProbePhase", 1, 0.01, skipped=False)
        after = instrumentation.records()

        assert len(after) == before + 1
        assert after[-1].name == "legacy_pipeline/ProbePhase"

    def test_a_skipped_phase_is_recorded_with_its_reason(self, instrumentation):
        from core.brain.cognitive_engine import _record_legacy_pass

        _record_legacy_pass(
            "SkippedPhase", 9, 0.0, skipped=True, reason="opt-bisect: ordinal 9 > limit 3"
        )
        last = instrumentation.records()[-1]
        assert last.skipped is True
        assert "opt-bisect" in last.reason

    def test_recording_never_breaks_a_turn(self, monkeypatch):
        """A debug aid may not be able to stop Aura answering."""
        from core.brain import cognitive_engine

        def _explode():
            raise RuntimeError("instrumentation is down")

        monkeypatch.setattr(cognitive_engine, "_pass_instrumentation", _explode)
        # Neither of these may raise.
        cognitive_engine._begin_pass_run("legacy_pipeline")

    def test_a_missing_instrumentation_module_still_runs_every_phase(
        self, monkeypatch
    ):
        from core.brain import cognitive_engine

        null = cognitive_engine._NullPassInstrumentation()
        assert null.should_run("anything") == (True, 0, "")
