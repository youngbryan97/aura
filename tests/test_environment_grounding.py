"""tests/test_environment_grounding.py
====================================
Tests for EnvironmentProtocol, structured Observation objects, and specialized subclasses.
"""

from __future__ import annotations

import pytest
from typing import Any, Dict

from core.environment.environment_protocol import EnvironmentProtocol, ActionResult, PredictedOutcome
from core.grounding.observation_schema import Observation, FileObservation, ProcessObservation, UnitTestObservation


class SampleEnvironment(EnvironmentProtocol):
    """Sample concrete virtual environment for protocol validation."""

    def __init__(self) -> None:
        self.state = {"counter": 0}

    async def observe(self) -> Observation:
        return Observation(
            source="sample_env",
            event_type="counter_state",
            data={"counter": self.state["counter"]},
            suggested_affordances=["increment", "reflect"]
        )

    async def act(self, action: Any) -> ActionResult:
        if action == "increment":
            self.state["counter"] += 1
            obs = await self.observe()
            return ActionResult(
                success=True,
                next_observation=obs,
                reward=1.0,
                latency_ms=1.5,
                side_effects={"incremented": True}
            )
        return ActionResult(
            success=False,
            next_observation=await self.observe(),
            reward=-0.5,
            latency_ms=0.5
        )

    async def predict(self, action: Any) -> PredictedOutcome:
        if action == "increment":
            future_obs = Observation(
                source="sample_env",
                event_type="counter_state",
                data={"counter": self.state["counter"] + 1}
            )
            return PredictedOutcome(
                predicted_observation=future_obs,
                expected_reward=1.0,
                confidence=0.95
            )
        return PredictedOutcome(
            predicted_observation=await self.observe(),
            expected_reward=-0.5,
            confidence=0.9
        )

    async def score(self, result: ActionResult) -> float:
        return result.reward

    async def reset(self) -> Observation:
        self.state["counter"] = 0
        return await self.observe()

    async def snapshot(self) -> Dict[str, Any]:
        return dict(self.state)

    async def restore(self, checkpoint: Dict[str, Any]) -> None:
        self.state = dict(checkpoint)


@pytest.mark.asyncio
async def test_environment_protocol_compliance() -> None:
    """Verifies that environments compliant with EnvironmentProtocol function correctly."""
    env = SampleEnvironment()
    
    # Check initial observation
    obs = await env.observe()
    assert obs.source == "sample_env"
    assert obs.data["counter"] == 0
    assert "increment" in obs.suggested_affordances
    
    # Check prediction
    prediction = await env.predict("increment")
    assert prediction.predicted_observation.data["counter"] == 1
    assert prediction.expected_reward == 1.0
    assert prediction.confidence == 0.95
    
    # Execute action
    result = await env.act("increment")
    assert result.success is True
    assert result.next_observation.data["counter"] == 1
    assert result.reward == 1.0
    assert result.side_effects["incremented"] is True
    
    # Check scoring
    score = await env.score(result)
    assert score == 1.0
    
    # Reset
    reset_obs = await env.reset()
    assert reset_obs.data["counter"] == 0
    
    # Snapshot and restore
    await env.act("increment")
    checkpoint = await env.snapshot()
    assert checkpoint["counter"] == 1
    
    await env.act("increment")  # Now at 2
    assert (await env.observe()).data["counter"] == 2
    
    await env.restore(checkpoint)  # Restored to 1
    assert (await env.observe()).data["counter"] == 1


def test_specialized_observations() -> None:
    """Verifies that specialized subclasses construct correct schemas."""
    
    # 1. FileObservation
    file_obs = FileObservation(
        path="core/grounding/types.py",
        operation="write",
        size_bytes=1024,
        success=True
    )
    assert file_obs.source == "filesystem"
    assert file_obs.event_type == "file_write"
    assert file_obs.data["path"] == "core/grounding/types.py"
    assert "file_read" in file_obs.suggested_affordances
    assert file_obs.severity == 0.0
    
    # 2. ProcessObservation
    proc_obs = ProcessObservation(
        command="pytest tests/",
        exit_code=1,
        stdout="1 failed, 15 passed",
        stderr="Test timeout failed",
        cpu_percent=12.5,
        memory_rss_bytes=1024 * 1024 * 50
    )
    assert proc_obs.source == "terminal"
    assert proc_obs.event_type == "command_executed"
    assert proc_obs.severity >= 0.4
    assert "diagnose_error" in proc_obs.suggested_affordances
    
    # 3. UnitTestObservation
    test_obs = UnitTestObservation(
        test_suite="RSI Tests",
        passed_count=8,
        failed_count=2,
        duration_seconds=3.2,
        failures=[{"test_name": "test_rsi_timeout", "message": "subprocess.TimeoutExpired occurred", "traceback": "..."}]
    )
    assert test_obs.source == "pytest"
    assert test_obs.event_type == "test_run_complete"
    assert test_obs.severity > 0.0
    assert "patch_code" in test_obs.suggested_affordances
    assert len(test_obs.failures) == 1
