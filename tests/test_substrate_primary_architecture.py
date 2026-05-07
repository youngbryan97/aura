from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def test_observation_vector_is_grounded_and_state_dependent():
    from core.brain.llm.sensorimotor_grounding import observation_to_vector

    visual = observation_to_vector(
        {"source": "camera", "summary": "bright window", "confidence": 0.9, "energy": 0.4},
        dim=64,
    )
    audio = observation_to_vector(
        {"source": "microphone", "transcript": "Bryan said run the tests", "confidence": 0.8, "rms": 0.2},
        dim=64,
    )

    assert visual.shape == (64,)
    assert audio.shape == (64,)
    assert float(np.linalg.norm(visual)) > 0.0
    assert float(np.linalg.norm(audio)) > 0.0
    assert not np.allclose(visual, audio)


def test_continuous_substrate_accepts_sensor_observations():
    from core.brain.llm.continuous_substrate import ContinuousSubstrate

    substrate = ContinuousSubstrate()
    substrate.inject_observation(
        {"source": "screen", "summary": "terminal test output changed", "confidence": 0.7, "energy": 0.3}
    )
    for _ in range(8):
        substrate._step_once()

    summary = substrate.get_state_summary()
    assert summary["grounded_observation"] is True
    assert summary["last_observation_source"] == "screen"
    assert float(np.linalg.norm(substrate.get_state_vector())) > 0.0


def test_substrate_token_generator_changes_output_under_lesion():
    from core.brain.llm.continuous_substrate import ContinuousSubstrate
    from core.brain.llm.substrate_token_generator import SubstrateTokenGenerator

    substrate = ContinuousSubstrate()
    generator = SubstrateTokenGenerator(substrate, threshold=1.0)

    substrate.inject_input(np.ones(64, dtype=np.float32) * 0.55)
    for _ in range(12):
        substrate._step_once()
    intact = generator.generate("continue the repair loop", force=True)

    substrate._state = np.zeros(64, dtype=np.float32)
    lesioned = generator.generate("continue the repair loop", force=True)

    assert intact.used_substrate is True
    assert lesioned.used_substrate is True
    assert intact.logits_checksum != lesioned.logits_checksum
    assert intact.token_ids != lesioned.token_ids


def test_substrate_token_generator_falls_back_on_high_prediction_error():
    from core.brain.llm.continuous_substrate import ContinuousSubstrate
    from core.brain.llm.substrate_token_generator import SubstrateTokenGenerator

    substrate = ContinuousSubstrate()
    generator = SubstrateTokenGenerator(substrate, threshold=0.05)

    result = generator.generate(
        "Explain a multi-file architecture migration with hidden external baselines and receipts?",
        max_tokens=16,
    )

    assert result.used_substrate is False
    assert result.fallback_reason == "prediction_error_exceeded"
    assert result.prediction_error > result.threshold


@pytest.mark.asyncio
async def test_llm_router_uses_substrate_before_transformer(monkeypatch):
    from core.brain.llm.continuous_substrate import ContinuousSubstrate
    from core.brain.llm.llm_router import IntelligentLLMRouter
    from core.container import ServiceContainer

    ServiceContainer.clear()
    substrate = ContinuousSubstrate()
    substrate.inject_input(np.ones(64, dtype=np.float32) * 0.6)
    for _ in range(12):
        substrate._step_once()
    ServiceContainer.register_instance("continuous_substrate", substrate, required=False)

    monkeypatch.setenv("AURA_SUBSTRATE_PRIMARY", "1")
    router = IntelligentLLMRouter()
    text = await router.think("quiet status", force_substrate=True, max_tokens=8, origin="user")

    assert text.startswith("Substrate path:")
    assert router.last_user_tier == "substrate"


@pytest.mark.asyncio
async def test_online_lora_governor_blocks_when_training_is_running(tmp_path):
    from core.adaptation.online_lora_governor import OnlineLoRAGovernor

    fake_proc = SimpleNamespace(
        info={
            "pid": 123,
            "cmdline": ["python", "-m", "mlx_lm", "lora", "--train"],
            "name": "python",
        }
    )
    governor = OnlineLoRAGovernor(receipt_path=tmp_path / "receipts.jsonl", process_iter=lambda attrs: [fake_proc])

    receipt = await governor.maybe_update_from_reflection("I noticed a repair pattern.")

    assert receipt.status == "blocked_existing_training"
    assert "pid=123" in receipt.reason


@pytest.mark.asyncio
async def test_default_goal_seeder_creates_tool_attached_goals(tmp_path):
    from core.goals.default_goals import DEFAULT_AUTONOMY_GOALS, seed_default_autonomy_goals
    from core.goals.goal_engine import GoalEngine

    engine = GoalEngine(db_path=str(tmp_path / "goals.sqlite3"))
    seeded = await seed_default_autonomy_goals(engine)

    assert len(seeded) == len(DEFAULT_AUTONOMY_GOALS)
    assert all(goal["status"] == "in_progress" for goal in seeded)
    assert all(goal["required_tools"] for goal in seeded)


@pytest.mark.asyncio
async def test_overt_action_loop_executes_verifies_and_receipts(tmp_path):
    from core.runtime.overt_action_loop import OvertActionLoop
    from core.runtime.receipts import ReceiptStore

    class FakeSynth:
        async def start(self):
            return None

        async def synthesize(self, state):
            return SimpleNamespace(
                winner={
                    "goal": "Run a light environment self-audit",
                    "source": "test_goal",
                    "urgency": 0.6,
                    "metadata": {"required_skills": ["environment_info"]},
                },
                will_receipt_id="will-test-1",
            )

    class FakeEngine:
        async def execute(self, skill_name, params, context=None):
            return {
                "ok": True,
                "summary": "Environment audit completed.",
                "result": {"hostname": "test-host"},
            }

    loop = OvertActionLoop(
        capability_engine=FakeEngine(),
        synthesizer=FakeSynth(),
        receipt_store=ReceiptStore(tmp_path / "receipts"),
        state_provider=lambda: SimpleNamespace(cognition=SimpleNamespace(pending_initiatives=[])),
    )
    loop._record_life_trace = lambda result, raw: setattr(result, "life_trace_id", "life-test-1")

    result = await loop.run_once(force=True)

    assert result["status"] == "verified"
    assert result["skill"] == "environment_info"
    assert result["verified"] is True
    assert result["tool_receipt_id"].startswith("tool_execution-")
    assert result["autonomy_receipt_id"].startswith("autonomy-")
    assert loop.status()["actions_verified"] == 1
