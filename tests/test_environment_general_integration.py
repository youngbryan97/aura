from __future__ import annotations

from pathlib import Path

import pytest

from core.environment import ActionIntent, CommandCompiler, EnvironmentCapabilityMatrix, EnvironmentKernel, ResourceState
from core.environment.adapter import ExecutionResult
from core.environment.belief_graph import EnvironmentBeliefGraph
from core.environment.command import CommandSpec
from core.environment.generic_command_handlers import register_generic_handlers
from core.environment.observation import Observation
from core.environment.outcome.semantic_diff import SemanticDiffLearner
from core.environment.parsed_state import ParsedState
from core.environment.policy.candidate_generator import CandidateGenerator
from core.environments.terminal_grid import NetHackCommandCompiler, NetHackStateCompiler


FIXTURE_DIR = Path(__file__).parent / "environments" / "terminal_grid" / "fixtures"


class ScriptedNetHackAdapter:
    environment_id = "terminal_grid:nethack"

    def __init__(self, screens: list[str]) -> None:
        self.screens = screens
        self.index = 0
        self.run_id = "unstarted"
        self.alive = False
        self.executed: list[CommandSpec] = []

    async def start(self, *, run_id: str, seed: int | None = None) -> None:
        self.run_id = run_id
        self.alive = True

    async def observe(self) -> Observation:
        text = self.screens[min(self.index, len(self.screens) - 1)]
        return Observation(
            environment_id=self.environment_id,
            run_id=self.run_id,
            sequence_id=self.index + 1,
            text=text,
            raw=text,
        )

    async def execute(self, command: CommandSpec) -> ExecutionResult:
        self.executed.append(command)
        self.index += 1
        return ExecutionResult(True, command.command_id, await self.observe())

    async def close(self) -> None:
        self.alive = False

    def is_alive(self) -> bool:
        return self.alive


def _death_screen() -> str:
    lines = ["You die... Do you want your possessions identified? [ynq]"]
    lines.extend([""] * 21)
    lines.append("[Aura the Agent] St:10 Dx:10 Co:10 In:10 Wi:10 Ch:10 Neutral")
    lines.append("Dlvl:1 $:0 HP:0(12) Pw:7(7) AC:10 Xp:1/0 T:2")
    return "\n".join(line[:80].ljust(80) for line in lines)


def test_generic_command_handlers_bind_to_concrete_environment_id():
    compiler = CommandCompiler("browser:task")
    register_generic_handlers(compiler)

    command = compiler.compile(ActionIntent(name="observe"))

    assert command.environment_id == "browser:task"
    assert command.command_id.startswith("cmd_browser_task_")


def test_policy_reads_inventory_items_and_emits_generic_stair_intent():
    inventory_text = (FIXTURE_DIR / "nethack_inventory.txt").read_text(encoding="utf-8")
    parsed = NetHackStateCompiler().parse_text(inventory_text)

    candidates = CandidateGenerator().generate(parsed, belief=EnvironmentBeliefGraph(), recent_frames=[])
    names = {candidate.name for candidate in candidates}

    assert "eat" in names
    assert "wield" in names

    start = NetHackStateCompiler().parse_text((FIXTURE_DIR / "nethack_start.txt").read_text(encoding="utf-8"))
    transition = next(obj for obj in start.objects if obj.kind == "transition")
    start.self_state["local_coordinates"] = transition.position
    graph = EnvironmentBeliefGraph()
    graph.update_from_parsed_state(start)
    candidates = CandidateGenerator().generate(start, belief=graph, recent_frames=[])
    names = {candidate.name for candidate in candidates}
    assert "use_stairs" in names
    NetHackCommandCompiler().compile(next(candidate for candidate in candidates if candidate.name == "use_stairs"))


def test_belief_spatial_memory_keeps_metadata_and_legacy_kind_lookup():
    graph = EnvironmentBeliefGraph()
    graph.upsert_spatial("ctx", 3, 4, kind="trap", confidence=0.9)
    graph.upsert_spatial("ctx", 3, 4, kind="player", confidence=0.2)

    cell = graph.spatial[("ctx", 3, 4)]

    assert cell == "trap"
    assert cell["kind"] == "trap"
    assert cell["confidence"] >= 0.8


def test_semantic_diff_reports_resources_modal_and_new_entities():
    before = ParsedState(environment_id="env", context_id="ctx", self_state={"local_coordinates": (1, 1)})
    after = ParsedState(environment_id="env", context_id="ctx", self_state={"local_coordinates": (2, 1)})
    before.resources["health"] = ResourceState("health", 10, max_value=10)
    after.resources["health"] = ResourceState("health", 6, max_value=10)
    after.observed_ids.add("env:object:new")

    names = {event.name for event in SemanticDiffLearner().compute_diff(before, after)}

    assert "position_changed" in names
    assert "resource_health_decreased" in names
    assert "new_object_or_entity_observed" in names


@pytest.mark.asyncio
async def test_kernel_lifecycle_records_terminal_death_and_postmortem():
    start = (FIXTURE_DIR / "nethack_start.txt").read_text(encoding="utf-8")
    adapter = ScriptedNetHackAdapter([start, _death_screen()])
    kernel = EnvironmentKernel(
        adapter=adapter,
        state_compiler=NetHackStateCompiler(),
        command_compiler=NetHackCommandCompiler(),
    )
    await kernel.start(run_id="death-regression", seed=1)

    frame = await kernel.step(ActionIntent(name="wait", expected_effect="turn_passed"))

    assert frame.outcome_assessment is not None
    assert frame.outcome_assessment.is_death
    assert kernel.run_manager.records[-1].terminal_reason == "death"
    assert kernel.run_manager.records[-1].postmortem is not None
    assert kernel.run_manager.current_record is None
    await kernel.close()


@pytest.mark.asyncio
async def test_environment_capability_matrix_is_executable_and_clean():
    start = (FIXTURE_DIR / "nethack_start.txt").read_text(encoding="utf-8")
    kernel = EnvironmentKernel(
        adapter=ScriptedNetHackAdapter([start]),
        state_compiler=NetHackStateCompiler(),
        command_compiler=NetHackCommandCompiler(),
    )
    await kernel.start(run_id="capability-matrix", seed=1)

    report = EnvironmentCapabilityMatrix().audit(kernel)

    assert report.score == 1.0
    report.require_clean()
    await kernel.close()
