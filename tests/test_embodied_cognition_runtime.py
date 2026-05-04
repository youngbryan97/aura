from __future__ import annotations

from dataclasses import dataclass

import pytest

from core.perception.cognitive_runtime import EmbodiedCognitionRuntime
from core.perception.environment_parser import EnvironmentParser, EnvironmentState
from core.perception.nethack_parser import NetHackParser
from core.phases.action_grounding import ground_response
from core.world_state import WorldState


class ToyParser(EnvironmentParser):
    def __init__(self, state: EnvironmentState):
        self.state = state

    def parse(self, raw_input):
        self.state.refresh_observation_id()
        return self.state


def test_runtime_uses_generic_prompt_skill_for_modal_state(tmp_path):
    state = EnvironmentState(
        domain="toy",
        context_id="room_1",
        self_state={"hp": 2, "max_hp": 10},
        messages=["Is this ok? [ynq]"],
        active_prompts=["Is this ok? [ynq]"],
    )
    runtime = EmbodiedCognitionRuntime(
        domain="toy",
        parser=ToyParser(state),
        legal_actions=["confirm", "cancel", "move"],
        prompt_actions={"is this ok": "confirm", "cancel": "cancel"},
        storage_root=tmp_path,
    )

    frame = runtime.observe("raw")

    assert frame.risk.level == "critical"
    assert frame.goal.name == "STABILIZE_CRITICAL_RISK"
    assert frame.skill.name == "resolve_active_prompt"
    assert "ENVIRONMENT STATE" in frame.to_prompt()


def test_action_gateway_replaces_normal_action_during_prompt(tmp_path):
    state = EnvironmentState(
        domain="toy",
        context_id="room_1",
        active_prompts=["Is this ok? [ynq]"],
    )
    runtime = EmbodiedCognitionRuntime(
        domain="toy",
        parser=ToyParser(state),
        legal_actions=["confirm", "cancel", "move"],
        prompt_actions={"is this ok": "confirm", "cancel": "cancel"},
        storage_root=tmp_path,
    )
    runtime.observe("raw")

    decision = runtime.approve_action("move", tags=["movement"])

    assert decision.approved is True
    assert decision.action == "confirm"
    assert decision.replaced is True


def test_action_gateway_blocks_irreversible_action_under_high_uncertainty(tmp_path):
    state = EnvironmentState(domain="toy", context_id="room_1")
    runtime = EmbodiedCognitionRuntime(
        domain="toy",
        parser=ToyParser(state),
        legal_actions=["use", "inspect"],
        storage_root=tmp_path,
    )
    runtime.belief.ensure_hypotheses("mystery object", ["safe", "harmful"])
    runtime.observe("raw")

    decision = runtime.approve_action("use", tags=["irreversible"])

    assert decision.approved is False
    assert "high_uncertainty_blocks_irreversible_action" in decision.vetoes


def test_runtime_bridges_observations_into_existing_world_state(service_container, tmp_path):
    world_state = WorldState()
    service_container.register_instance("world_state", world_state)
    state = EnvironmentState(domain="toy", context_id="room_1", messages=["new signal"])
    runtime = EmbodiedCognitionRuntime(
        domain="toy",
        parser=ToyParser(state),
        storage_root=tmp_path,
    )

    frame = runtime.observe("raw")

    assert world_state.get_belief("environment.toy.risk_level") == frame.risk.level
    events = world_state.get_salient_events()
    assert any("toy observation" in event["description"] for event in events)


def test_runtime_loads_existing_macro_skill_library(service_container, tmp_path):
    @dataclass
    class MacroSkill:
        description: str = "Existing macro skill"
        reliability: float = 0.9

    class MacroLibrary:
        skills = {"safe_probe": MacroSkill()}

    service_container.register_instance("skill_library", MacroLibrary())
    state = EnvironmentState(domain="toy", context_id="room_1")

    runtime = EmbodiedCognitionRuntime(
        domain="toy",
        parser=ToyParser(state),
        storage_root=tmp_path,
    )

    assert "macro:safe_probe" in runtime.skill_graph.options
    assert runtime.skill_graph.options["macro:safe_probe"].metadata["source"] == "core.agency.skill_library"


@pytest.mark.asyncio
async def test_action_grounding_preserves_marker_args_for_embodied_actions():
    class Engine:
        def __init__(self):
            self.calls = []

        async def execute(self, name, params, context):
            self.calls.append((name, params))
            return {"ok": True, "summary": "sent"}

    engine = Engine()
    result = await ground_response(
        "[ACTION:execute_nethack_action(key='y')]",
        capability_engine=engine,
    )

    assert result.dispatched_ok == 1
    assert engine.calls[0][1]["action"] == "y"


def test_nethack_parser_is_an_adapter_to_general_environment_state():
    lines = ["You see here an orange potion."]
    lines.extend([""] * 21)
    map_line = list(" " * 80)
    map_line[10] = "@"
    map_line[11] = "!"
    lines[1] = "".join(map_line)
    lines.append("[Aura the Test] St:10 Dx:10 Co:10 In:10 Wi:10 Ch:10 Neutral")
    lines.append("Dlvl:1 $:0 HP:10(20) Pw:5(5) AC:9 Xp:1/0 T:12 Hungry")
    screen = "\n".join(lines)

    state = NetHackParser().parse(screen)

    assert state.domain == "nethack"
    assert state.context_id == "dlvl_1"
    assert state.self_state["hp"] == 10
    assert state.self_state["hunger"] == "Hungry"
    assert any(entity.get("glyph") == "!" for entity in state.entities)
    assert state.uncertainty["visible_unknown_items"] >= 0.6
