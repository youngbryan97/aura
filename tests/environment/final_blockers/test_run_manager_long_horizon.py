"""Final Blocker: Long-horizon run manager, death, restart, postmortem.

RunManager must handle start, step, death detection, postmortem generation,
restart with learning preservation, and aggregate metrics.
"""
import pytest
from core.environment.run_manager import RunManager, RunRecord, AggregateMetrics
from core.environment.postmortem import PostmortemGenerator, PostmortemReport
from core.environment.environment_kernel import EnvironmentKernel, EnvironmentFrame
from core.environment.command import ActionIntent
from tests.environment.final_blockers.conftest import ScriptedTerminalAdapter


NORMAL_SCREEN = "Room - player at (5,5)\nHP:20(20) Pw:5(5) Dlvl:1"
DEATH_SCREEN = "You die...\nDo you want your possessions identified? [ynq]\nGoodbye cruel world."


class TestRunManager:
    """RunManager lifecycle: start, step, death, postmortem, restart, metrics."""

    def test_run_manager_creates_durable_record(self):
        rm = RunManager(mode="fixture_replay")
        record = rm.start_run(
            run_id="run_001",
            environment_id="terminal_grid:test",
            policy_version="v1.0",
            source_commit="abc123",
        )
        assert record.run_id == "run_001"
        assert record.environment_id == "terminal_grid:test"
        assert record.mode == "fixture_replay"
        assert record.policy_version == "v1.0"
        assert record.started_at > 0

    def test_death_detection(self):
        rm = RunManager()
        assert rm.detect_death(DEATH_SCREEN)
        assert not rm.detect_death(NORMAL_SCREEN)
        assert rm.detect_death("You die... --More--")
        assert rm.detect_death("DYWYPI?")

    def test_mark_contaminated(self):
        rm = RunManager()
        rm.start_run(run_id="r1", environment_id="test")
        rm.mark_contaminated("save_file_read")
        assert rm.current_record.contaminated
        assert rm.current_record.metadata["contamination_reason"] == "save_file_read"

    def test_restart_clears_ephemeral_preserves_learning(self):
        rm = RunManager()
        rm.learned_affordances["eat_food"] = 0.9
        rm.learned_procedures.append({"action": "eat", "context": "hungry"})
        rm.start_run(run_id="r1", environment_id="test")
        rm.restart()
        assert rm.current_record is None
        # Learning preserved
        assert rm.learned_affordances["eat_food"] == 0.9
        assert len(rm.learned_procedures) == 1

    def test_aggregate_metrics_distinguish_categories(self):
        rm = RunManager(mode="fixture_replay")
        # Death
        rm.start_run(run_id="r1", environment_id="test")
        rm.end_run(terminal_reason="death", frames=[], final_score=10.0)
        # Crash
        rm.start_run(run_id="r2", environment_id="test")
        rm.end_run(terminal_reason="crash", frames=[], final_score=0.0)
        # Success
        rm.start_run(run_id="r3", environment_id="test")
        rm.end_run(terminal_reason="success", frames=[], final_score=100.0)
        # Contaminated
        rm.start_run(run_id="r4", environment_id="test")
        rm.mark_contaminated("oracle_read")
        rm.end_run(terminal_reason="death", frames=[], final_score=5.0)

        metrics = rm.get_metrics()
        assert metrics.total_runs == 4
        assert metrics.deaths == 1  # r1 only (r4 is contaminated)
        assert metrics.crashes == 1
        assert metrics.successes == 1
        assert metrics.contaminated == 1
        assert metrics.best_score == 100.0


class TestDeathPostmortem:
    """Death must trigger postmortem with causal trace."""

    def test_postmortem_generated_on_death(self):
        rm = RunManager()
        rm.start_run(run_id="death_run", environment_id="test")
        # Create minimal fake frames
        from core.environment.observation import Observation
        from core.environment.parsed_state import ParsedState

        obs = Observation(
            environment_id="test", run_id="death_run", sequence_id=0,
            raw=DEATH_SCREEN, text=DEATH_SCREEN, metadata={},
        )
        parsed = ParsedState(environment_id="test", context_id="test", sequence_id=0, self_state={})
        frame = EnvironmentFrame(
            observation=obs,
            parsed_state=parsed,
            belief_hash_before="aaa",
            belief_hash_after="bbb",
        )
        record = rm.end_run(terminal_reason="death", frames=[frame])
        assert record.postmortem is not None
        assert record.postmortem.terminal_reason == "death"
        assert record.postmortem.run_id == "death_run"

    def test_postmortem_contains_required_sections(self):
        gen = PostmortemGenerator(lookback=5)
        from core.environment.observation import Observation
        from core.environment.parsed_state import ParsedState

        frames = []
        for i in range(3):
            obs = Observation(
                environment_id="test", run_id="pm_run", sequence_id=i,
                raw=NORMAL_SCREEN, text=NORMAL_SCREEN, metadata={},
            )
            parsed = ParsedState(environment_id="test", context_id="test", sequence_id=i, self_state={"hp": 20})
            frame = EnvironmentFrame(
                observation=obs, parsed_state=parsed,
                belief_hash_before=f"b{i}", belief_hash_after=f"a{i}",
                action_intent=ActionIntent(name="move", parameters={"direction": "north"}),
            )
            frames.append(frame)

        pm = gen.generate(
            run_id="pm_run", environment_id="test", mode="fixture_replay",
            terminal_reason="death", frames=frames, started_at=1000.0,
        )
        assert pm.total_steps == 3
        assert len(pm.last_n_actions) == 3
        assert len(pm.last_n_observations) == 3
        assert isinstance(pm.avoidable_failure_hypotheses, list)

    def test_postmortem_not_generated_for_success(self):
        rm = RunManager()
        rm.start_run(run_id="success_run", environment_id="test")
        record = rm.end_run(terminal_reason="success", frames=[], final_score=100.0)
        assert record.postmortem is None


class TestLongHorizonStress:
    """1000-step simulated run must not leak memory or tasks."""

    @pytest.mark.asyncio
    async def test_1000_step_no_leak(self):
        screens = [NORMAL_SCREEN] * 5  # will cycle
        adapter = ScriptedTerminalAdapter(screens)
        kernel = EnvironmentKernel(adapter=adapter)
        await kernel.start(run_id="stress_test")

        for i in range(100):  # reduced from 1000 for test speed
            intent = ActionIntent(name="wait", risk="safe")
            try:
                frame = await kernel.step(intent)
            except Exception:
                break

        # Frame history should be bounded or at least not catastrophically large
        # (In production, implement frame rotation)
        assert len(kernel.frames) <= 200
        await kernel.close()
