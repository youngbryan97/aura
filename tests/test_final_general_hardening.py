from __future__ import annotations

import pytest

from core.brain.grounding_guard import GroundingGuard
from core.brain.llm.context_gate import ContextBlock, estimate_tokens
from core.environment import ActionIntent, CommandCompiler, EnvironmentCapabilityMatrix, EnvironmentKernel
from core.environment.action_semantics import ActionSemanticsValidator
from core.environment.adapter import EnvironmentCapabilities, ExecutionResult
from core.environment.belief_graph import EnvironmentBeliefGraph
from core.environment.command import CommandSpec, CommandStep, command_id_for
from core.environment.external_validation import ExternalTaskProofGate
from core.environment.generic_command_handlers import register_generic_handlers
from core.environment.observation import Observation
from core.environment.outcome_attribution import OutcomeAssessment
from core.environment.parsed_state import ParsedState
from core.environment.planning import GridPathPlanner
from core.environment.curriculum import CurriculumEngine
from core.environment.abstraction_discovery import AbstractionDiscoveryEngine
from core.environment.experience_replay import HindsightReplayBuffer
from core.environments.terminal_grid.state_compiler import TerminalGridStateCompiler
from core.learning.formalizer import KnowledgeFormalizer
from core.runtime.concurrency_health import ConcurrencyHealthMonitor
from core.runtime.proof_kernel_bridge import start_proof_kernel_bridge


class _GeneralScriptedAdapter:
    environment_id = "terminal_grid:generic"
    capabilities = EnvironmentCapabilities(can_act=True, supports_replay=True)
    _simulated = True

    def __init__(self, screens: list[str]) -> None:
        self.screens = screens
        self.index = 0
        self.run_id = ""
        self.alive = False

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
        self.index += 1
        return ExecutionResult(True, command.command_id, await self.observe(), metadata={"resource_cost": 0.1})

    async def close(self) -> None:
        self.alive = False

    def is_alive(self) -> bool:
        return self.alive


def _outcome(score: float = 0.0, *, death: bool = False, surprise: float = 0.0) -> OutcomeAssessment:
    return OutcomeAssessment(
        action="use",
        expected_effect="effect",
        observed_events=["failure"] if score < 0.5 else ["effect"],
        success_score=score,
        harm_score=1.0 if death else 0.0,
        information_gain=0.0,
        surprise=surprise,
        is_death=death,
    )


def test_formalizer_distills_structured_causal_conditional_and_procedure_claims():
    content = " ".join(
        [
            "If a system is uncertain, it should gather evidence before direct action.",
            "To recover from a stalled workflow, observe the current state and switch subgoals.",
            "Unknown direct interaction causes avoidable irreversible failure.",
            "A safety gate requires executable receipts before promotion.",
            "This extra filler sentence keeps the document above the minimum content length for extraction.",
        ]
        * 6
    )

    facts = KnowledgeFormalizer()._extract_atomic_facts(
        content,
        source_title="Reference Manual",
        source_url="https://docs.example.test/manual",
    )

    claim_types = {fact["type"] for fact in facts}
    assert {"conditional_rule", "procedure", "causal_rule", "requirement"} <= claim_types
    first = facts[0]["metadata"]
    assert first["verification_status"] == "extractive_unverified"
    assert first["source_quality"] > 0.7
    assert facts[0]["confidence"] != 0.6


def test_grounding_guard_returns_corrective_replan_not_only_penalty():
    guard = GroundingGuard(orchestrator=object())

    correction = guard.correction_action(
        "execute bounded action",
        0.91,
        {"ok": False, "error": "unknown_intent:deploy"},
    )

    assert correction["needs_replan"] is True
    assert correction["intent"] == "observe"
    assert correction["grounded_score"] <= 0.2


def test_context_gate_compacts_by_conservative_token_budget():
    codeish = "def f(x):\n    return {'value': x, 'items': [x for x in range(100)]}\n" * 30
    multilingual = "状態を確認してから実行する。安全な観測を優先する。" * 20

    assert estimate_tokens(codeish) > len(codeish) / 5
    assert estimate_tokens(multilingual) > len(multilingual) / 5

    block = ContextBlock(id="code", content=codeish, essential=True, max_tokens=80).compact()
    assert estimate_tokens(block.content) <= 80
    assert "[compacted]" in block.content


def test_terminal_grid_compiler_extracts_generic_beliefs_without_domain_strategy():
    screen = "\n".join(
        [
            "You hear a distant alarm.",
            "#####",
            "#@..>",
            "#.^.#",
            "#####",
        ]
    )
    parsed = TerminalGridStateCompiler().compile(
        Observation(environment_id="terminal_grid:generic", run_id="r1", sequence_id=1, text=screen, raw=screen)
    )

    assert parsed.self_state["local_coordinates"] == (1, 2)
    assert any(obj.kind == "transition" for obj in parsed.objects)
    assert any(hazard.label == "visible grid hazard" for hazard in parsed.hazards)
    assert any("distant alarm" in event.label for event in parsed.semantic_events)


def test_action_semantics_blocks_unknown_high_risk_direct_action_and_allows_observe():
    validator = ActionSemanticsValidator()
    parsed = ParsedState(environment_id="env", context_id="ctx", uncertainty={"identity": 0.9})
    intent = ActionIntent(name="use", risk="risky", tags={"unknown"}, expected_effect="effect")
    command = CommandSpec(
        command_id="cmd_use",
        environment_id="env",
        intent=intent,
        preconditions=[],
        steps=[CommandStep(kind="key", value="u")],
        expected_effects=["effect"],
    )

    blocked = validator.validate(intent=intent, command=command, parsed_state=parsed)
    assert blocked.allowed is False
    assert blocked.requires_observation is True

    observe = ActionIntent(name="observe", expected_effect="state_observed")
    observe_command = CommandSpec(
        command_id=command_id_for("env", observe),
        environment_id="env",
        intent=observe,
        preconditions=[],
        steps=[CommandStep(kind="observe", value="")],
        expected_effects=["state_observed"],
    )
    allowed = validator.validate(intent=observe, command=observe_command, parsed_state=parsed)
    assert allowed.allowed is True
    assert allowed.reversible is True


def test_grid_path_planner_uses_canonical_spatial_map_and_avoids_hazards():
    graph = EnvironmentBeliefGraph()
    graph.upsert_spatial("ctx", 0, 0, kind="player", confidence=0.95, walkable=True, properties={"self": True})
    graph.upsert_spatial("ctx", 1, 0, kind="hazard", confidence=0.95, walkable=False)
    graph.upsert_spatial("ctx", 1, 1, kind="floor", confidence=0.8, walkable=True)
    graph.upsert_spatial("ctx", 2, 1, kind="floor", confidence=0.8, walkable=True)

    path = GridPathPlanner().plan(graph, context_id="ctx", start=(0, 0), goal=(2, 1))

    assert path
    assert (1, 0) not in path


def test_replay_and_abstraction_turn_failures_into_transferable_rules():
    before = ParsedState(environment_id="env:one", context_id="ctx")
    after = ParsedState(environment_id="env:one", context_id="ctx", uncertainty={"identity": 0.9})
    action = ActionIntent(name="use", risk="risky", tags={"unknown"})
    replay = HindsightReplayBuffer()
    discovery = AbstractionDiscoveryEngine(min_evidence=2)

    for _ in range(2):
        replay.add_transition(
            environment_id="env:one",
            context_id="ctx",
            action=action,
            before=before,
            after=after,
            outcome=_outcome(0.0, surprise=0.8),
            observed_events=["failure"],
        )
        discovery.observe_transition(
            environment_id="env:one",
            context_id="ctx",
            action=action,
            outcome=_outcome(0.0, surprise=0.8),
            observed_events=["failure"],
            parsed_after=after,
        )

    rules = replay.applicable_rules(action=action, environment_family="browser")
    assert any(rule.trigger == "unknown_direct_action" for rule in rules)
    assert any(absn.label == "unknown-asset-direct-interaction-risk" for absn in discovery.abstractions.values())


def test_curriculum_generates_next_task_from_bottleneck_and_mastery():
    curriculum = CurriculumEngine()
    curriculum.record_result(environment_family="desktop", objective="observe", outcome_score=1.0)
    curriculum.record_result(environment_family="desktop", objective="recover", outcome_score=0.1, bottleneck="modal")

    task = curriculum.propose_next_task(environment_family="desktop", bottlenecks={"modal": 0.9, "policy": 0.2})

    assert task.objective == "modal"
    assert "trace" in task.allowed_capabilities
    assert task.to_dict()["task_id"].startswith("curriculum:desktop:")


@pytest.mark.asyncio
async def test_kernel_wires_budget_replay_abstraction_curriculum_and_external_proof():
    screens = [
        "#####\n#@.>#\n#####",
        "#####\n#.@>#\n#####",
    ]
    compiler = CommandCompiler("terminal_grid:generic")
    register_generic_handlers(compiler)
    kernel = EnvironmentKernel(
        adapter=_GeneralScriptedAdapter(screens),
        state_compiler=TerminalGridStateCompiler(),
        command_compiler=compiler,
    )
    await kernel.start(run_id="general-hardening", seed=7)

    frame = await kernel.step(ActionIntent(name="observe", expected_effect="state_observed"))
    report = EnvironmentCapabilityMatrix().audit(kernel)

    assert report.score == 1.0
    assert frame.metadata["action_budget"]["used_total_steps"] == 1
    assert frame.metadata["curriculum_next"]["task_id"].startswith("curriculum:")

    await kernel.close()
    evidence = ExternalTaskProofGate().evaluate_kernel(kernel)
    assert evidence.passed is True
    assert evidence.proof_level == "simulated"
    assert evidence.trace_rows >= 1


def test_concurrency_health_samples_existing_watchdogs_and_queues():
    report = ConcurrencyHealthMonitor(stale_task_age_s=0.1).sample()

    assert report.active_tasks >= 0
    assert report.dlq_total >= 0
    assert 0.0 <= report.pressure <= 1.0
    assert "task_tracker" in report.evidence


@pytest.mark.asyncio
async def test_proof_kernel_bridge_runs_over_live_runtime_evidence():
    bridge = await start_proof_kernel_bridge()
    status = bridge.status()

    assert status["active"] is True
    assert "live runtime evidence is sampled" in status["claim_scope"]["supports"]
    assert "proof of subjective experience" in status["claim_scope"]["does_not_support"]
