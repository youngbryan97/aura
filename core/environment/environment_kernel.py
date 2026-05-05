"""Universal typed observe -> model -> gate -> act -> learn kernel."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from .action_gateway import EnvironmentActionGateway, GatewayDecision
from .adapter import EnvironmentAdapter, ExecutionResult
from .belief_graph import EnvironmentBeliefGraph
from .blackbox import BlackBoxRecorder, BlackBoxRow
from .command import ActionIntent, CommandCompiler, CommandSpec
from .crisis import CrisisManager
from .episode_manager import EpisodeManager
from .homeostasis import Homeostasis
from .modal import ModalManager
from .observation import Observation
from .options import OptionCompiler, OptionLibrary
from .outcome_attribution import OutcomeAssessment, OutcomeAttributor
from .parsed_state import ParsedState
from .prediction_error import PredictionErrorComputer
from .receipt_chain import EnvironmentActionReceipt
from .simulation import TacticalSimulator
from .state_compiler import StateCompiler


@dataclass
class EnvironmentFrame:
    observation: Observation
    parsed_state: ParsedState
    belief_hash_before: str
    belief_hash_after: str
    selected_option: str = ""
    action_intent: ActionIntent | None = None
    gateway_decision: GatewayDecision | None = None
    receipt: EnvironmentActionReceipt | None = None
    execution_result: ExecutionResult | None = None
    post_observation: Observation | None = None
    post_parsed_state: ParsedState | None = None
    outcome_assessment: OutcomeAssessment | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EnvironmentKernel:
    """Small environment OS kernel.

    It does not replace Aura's older embodied runtime; it supplies the typed
    contract and evidence trail that adapters and existing systems can share.
    """

    def __init__(
        self,
        *,
        adapter: EnvironmentAdapter,
        state_compiler: StateCompiler | None = None,
        command_compiler: CommandCompiler | None = None,
        trace_path: str | Path | None = None,
    ) -> None:
        self.adapter = adapter
        self.environment_id = adapter.environment_id
        if state_compiler is None:
            if "nethack" in adapter.environment_id:
                from core.embodiment.games.nethack.state_compiler import NetHackStateCompiler
                self.state_compiler = NetHackStateCompiler()
            else:
                self.state_compiler = StateCompiler()
        else:
            self.state_compiler = state_compiler
        self.command_compiler = command_compiler or CommandCompiler(self.environment_id)
        from .generic_command_handlers import register_generic_handlers
        register_generic_handlers(self.command_compiler)
        self.belief = EnvironmentBeliefGraph()
        from .policy.policy_orchestrator import PolicyOrchestrator
        self.policy = PolicyOrchestrator()
        from .governance_bridge import EnvironmentGovernanceBridge
        self.governance_bridge = EnvironmentGovernanceBridge()
        self.modal_manager = ModalManager()
        self.gateway = EnvironmentActionGateway(modal_manager=self.modal_manager)
        self.homeostasis = Homeostasis()
        self.options = OptionLibrary()
        self.option_compiler = OptionCompiler()
        self.simulator = TacticalSimulator()
        self.prediction_error = PredictionErrorComputer()
        self.outcomes = OutcomeAttributor()
        self.crisis = CrisisManager()
        from .outcome.semantic_diff import SemanticDiffLearner
        self.semantic_diff = SemanticDiffLearner()
        from .boundary_guard import BoundaryGuard
        self.boundary_guard = BoundaryGuard()
        from .run_manager import RunManager
        self.run_manager = RunManager()
        self.htn_planner = self.policy.strategic_policy.planner
        self.blackbox = BlackBoxRecorder(trace_path)
        self.episode: EpisodeManager | None = None
        self.run_id = ""
        self.frames: list[EnvironmentFrame] = []
        # Cross-episode learning stores
        from core.memory.procedural.store import ProceduralMemoryStore
        from core.environment.outcome.ledger import OutcomeLedger
        
        aura_data_dir = Path.home() / ".aura" / "data" / self.environment_id.replace(":", "_")
        self.causal_model = None       # set externally to CausalWorldModel if available
        self.outcome_ledger = OutcomeLedger(aura_data_dir / "outcome_ledger.json")
        self.procedural_store = ProceduralMemoryStore(aura_data_dir / "procedural_store.json")
        self.episodic_memory = None    # set externally to EpisodicMemory if available
        self.macro_inducer = None      # set externally to MacroInducer if available
        self.competence_tracker: dict[str, dict] = {}  # action_name -> {attempts, successes, last_score}
        self.world_state = None
        self._bind_aura_services()

    async def start(self, *, run_id: str, seed: int | None = None) -> None:
        self.run_id = run_id
        self.episode = EpisodeManager(run_id, self.environment_id)
        
        # Seed Aura's goals
        from core.environment.strategy.goal_seeder import seed_aura_goals
        seed_aura_goals(self.htn_planner, self.environment_id)
        await self.adapter.start(run_id=run_id, seed=seed)
        self.run_manager.mode = self._infer_run_mode()
        capabilities = getattr(self.adapter, "capabilities", None)
        capabilities_payload = asdict(capabilities) if is_dataclass(capabilities) else {}
        self.run_manager.start_run(
            run_id=run_id,
            environment_id=self.environment_id,
            adapter_config={
                "adapter": self.adapter.__class__.__name__,
                "capabilities": capabilities_payload,
            },
        )

    async def observe(self) -> EnvironmentFrame:
        observation = await self.adapter.observe()
        belief_before = self.belief.stable_hash()
        parsed = self.state_compiler.compile(observation)
        self.belief.update_from_parsed_state(parsed)
        belief_after = self.belief.stable_hash()
        resources = self.homeostasis.extract(parsed)
        homeostasis = self.homeostasis.assess(resources)
        if self.episode:
            self.episode.transition(
                valid_state=True,
                stable=not homeostasis.critical_resources and parsed.modal_state is None,
                critical_risk=bool(homeostasis.critical_resources),
            )
        frame = EnvironmentFrame(
            observation=observation,
            parsed_state=parsed,
            belief_hash_before=belief_before,
            belief_hash_after=belief_after,
            metadata={"homeostasis": asdict(homeostasis)},
        )
        self.frames.append(frame)
        self._publish_observation(frame)
        self._trace(frame)
        return frame

    async def step(self, intent: ActionIntent | None = None) -> EnvironmentFrame:
        started = time.time()
        frame = await self.observe()
        parsed = frame.parsed_state
        if parsed.modal_state and parsed.modal_state.requires_resolution:
            safe_response = self.modal_manager.resolve(
                parsed.modal_state, 
                intent_name=intent.name if intent else None, 
                intent_parameters=intent.parameters if intent else None
            )
            intent = ActionIntent(
                name="resolve_modal",
                parameters={"response": safe_response},
                expected_effect="modal_cleared",
                risk="safe",
                tags={"modal"},
            )
            frame.selected_option = "RESOLVE_MODAL"
        elif intent is None:
            intent = self.policy.select_action(
                parsed_state=parsed,
                belief=self.belief,
                homeostasis=self.homeostasis,
                episode=self.episode,
                recent_frames=self.frames,
            )
            frame.selected_option = intent.name

        sim = self.simulator.simulate(self.belief, intent)
        
        gov_decision = await self.governance_bridge.decide_action(intent)
        if not gov_decision.approved:
            decision = GatewayDecision(
                approved=False,
                action_intent=None,
                reason=gov_decision.reason or "governance_veto",
                vetoes=[gov_decision.reason or "governance_veto"],
                decision_id="gov_veto",
            )
            decision_id = decision.decision_id
            will_id = None
            auth_id = None
        else:
            decision = self.gateway.approve(
                intent,
                modal_state=parsed.modal_state,
                simulation=sim,
                uncertainty=max(parsed.uncertainty.values()) if parsed.uncertainty else 0.0,
                context_id=parsed.context_id or "default",
                authority_receipt_id=gov_decision.authority_receipt_id,
            )
            decision_id = decision.decision_id
            will_id = gov_decision.will_receipt_id
            auth_id = gov_decision.authority_receipt_id
            
        frame.action_intent = intent
        frame.gateway_decision = decision
        receipt = EnvironmentActionReceipt(
            receipt_id=f"envrcpt_{self.run_id}_{parsed.sequence_id}_{len(self.frames)}",
            run_id=self.run_id,
            sequence_id=parsed.sequence_id,
            environment_id=self.environment_id,
            observation_id=frame.observation.stable_hash(),
            belief_hash_before=frame.belief_hash_before,
            action_intent_id=intent.intent_id(),
            gateway_decision_id=decision_id,
            will_receipt_id=will_id,
            authority_receipt_id=auth_id,
        )
        frame.receipt = receipt
        if not decision.approved:
            receipt.finalize(status="blocked", belief_hash_after=frame.belief_hash_after)
            self.gateway.record_failure(intent.name, parsed.context_id or "default")
            self._trace(frame, latency_ms=(time.time() - started) * 1000)
            return frame
        command = self.command_compiler.compile(intent, trace_id=receipt.receipt_id, receipt_id=receipt.receipt_id)
        receipt.command_id = command.command_id
        result = await self.adapter.execute(command)
        frame.execution_result = result
        
        # Observe and parse after execution
        obs_after = result.observation_after or await self.adapter.observe()
        parsed_after = self.state_compiler.compile(obs_after)
        self.belief.update_from_parsed_state(parsed_after)
        frame.post_observation = obs_after
        frame.post_parsed_state = parsed_after
        
        # Semantic diff
        semantic_events = self.semantic_diff.compute_diff(parsed, parsed_after)
        observed_events = [e.name for e in semantic_events]
        self.semantic_diff.learn_from_transition(parsed, intent, parsed_after)
        
        expected = sim.hypotheses[0].predicted_events if sim.hypotheses else []
        pe = self.prediction_error.compute(action_id=intent.intent_id(), expected_events=expected, observed_events=observed_events)
        before_resources = {name: float(res.value) for name, res in parsed.resources.items()}
        after_resources = {name: float(res.value) for name, res in parsed_after.resources.items()}
        resource_delta = self.outcomes.compute_resource_delta(before_resources, after_resources)
        messages = [event.label for event in parsed_after.semantic_events]
        raw_after = obs_after.text or (obs_after.raw if isinstance(obs_after.raw, str) else "")
        if raw_after:
            first_line = raw_after.splitlines()[0] if raw_after.splitlines() else raw_after
            if first_line:
                messages.append(first_line)
        outcome = self.outcomes.assess(
            action=intent.name,
            expected_effect=intent.expected_effect,
            observed_events=observed_events,
            prediction_error=pe,
            resource_delta=resource_delta,
            information_gain=sim.hypotheses[0].information_gain if sim.hypotheses else 0.0,
            messages=messages,
        )
        frame.outcome_assessment = outcome
        receipt.execution_result_id = result.command_id
        receipt.finalize(status="executed" if result.ok else "failed", belief_hash_after=self.belief.stable_hash())
        frame.belief_hash_after = self.belief.stable_hash()

        # --- Cross-system wiring: outcome -> learning stores ---
        # 1. Causal model: ground with empirical observations
        if self.causal_model and observed_events:
            for event_name in observed_events:
                self.causal_model.add_observation(intent.name, event_name, 1.0 if outcome.success_score > 0.5 else -0.5)

        # 2. Outcome ledger: record structured outcome for cross-episode transfer
        if self.outcome_ledger:
            try:
                self.outcome_ledger.record_outcome(
                    action=intent.name,
                    env_id=self.environment_id,
                    ctx_id=parsed.context_id or "default",
                    success=outcome.success_score > 0.5,
                    score=outcome.success_score,
                    consequences=observed_events,
                )
            except Exception:
                pass  # non-critical

        # 3. Procedural memory: record successful procedures
        if self.procedural_store and outcome.success_score > 0.7:
            try:
                self.procedural_store.record(
                    environment_family=self.environment_id.split(":")[0],
                    context_signature=parsed.context_id or "default",
                    procedure={"action": intent.name, "parameters": intent.parameters, "effect": observed_events},
                )
            except Exception:
                pass

        # 4. Episodic memory: record notable events (deaths, surprises, milestones)
        if self.episodic_memory and (outcome.is_death or outcome.surprise > 0.5):
            try:
                self.episodic_memory.store(
                    content=f"Action {intent.name} resulted in {', '.join(observed_events)}",
                    importance=1.0 if outcome.is_death else outcome.surprise,
                    tags=[self.environment_id, intent.name] + observed_events[:3],
                )
            except Exception:
                pass

        # 5. HTN planner: feed semantic events as completion signals
        self.htn_planner.update(parsed_after, self.belief)

        # 6. Competence tracker: structured capability measurement
        ct = self.competence_tracker.setdefault(intent.name, {"attempts": 0, "successes": 0, "last_score": 0.0})
        ct["attempts"] += 1
        if outcome.success_score > 0.5:
            ct["successes"] += 1
        ct["last_score"] = outcome.success_score

        # 7. Macro inducer: collect action traces for N-gram mining
        if self.macro_inducer:
            try:
                self.macro_inducer.record_step(intent.name)
            except Exception:
                pass

        # 8. RunManager: record step
        self.run_manager.record_step(frame)
        self._publish_outcome(frame, semantic_events, resource_delta)

        terminal_reason = self._terminal_reason(outcome, obs_after, parsed_after)
        if terminal_reason:
            if self.episode:
                self.episode.transition(terminal=True)
            if self.run_manager.current_record is not None:
                self.run_manager.end_run(
                    terminal_reason=terminal_reason,
                    frames=self.frames,
                    final_score=outcome.success_score,
                )

        self._trace(frame, latency_ms=(time.time() - started) * 1000)
        return frame

    async def close(self) -> None:
        if self.run_manager.current_record is not None:
            self.run_manager.end_run(
                terminal_reason="closed",
                frames=self.frames,
                final_score=self.frames[-1].outcome_assessment.success_score
                if self.frames and self.frames[-1].outcome_assessment
                else 0.0,
            )
        if self.episode:
            self.episode.transition(terminal=True)
            self.episode.transition()
        self.save_learning_stores()
        await self.adapter.close()

    def save_learning_stores(self) -> None:
        if self.outcome_ledger:
            try:
                self.outcome_ledger.save()
            except Exception:
                pass
        if self.procedural_store:
            try:
                self.procedural_store.save()
            except Exception:
                pass

    def _trace(self, frame: EnvironmentFrame, *, latency_ms: float = 0.0) -> None:
        parsed = frame.post_parsed_state or frame.parsed_state
        self.blackbox.record(
            BlackBoxRow(
                run_id=self.run_id or frame.observation.run_id,
                sequence_id=frame.observation.sequence_id,
                environment_id=self.environment_id,
                context_id=parsed.context_id,
                raw_observation_hash=frame.observation.stable_hash(),
                parsed_state_ref=parsed.stable_hash(),
                belief_hash_before=frame.belief_hash_before,
                selected_option=frame.selected_option,
                action_intent=asdict(frame.action_intent) if frame.action_intent else {},
                gateway_decision=asdict(frame.gateway_decision) if frame.gateway_decision else {},
                will_receipt_id=frame.receipt.will_receipt_id if frame.receipt else None,
                command_spec={"command_id": frame.receipt.command_id} if frame.receipt and frame.receipt.command_id else {},
                execution_result=asdict(frame.execution_result) if frame.execution_result else {},
                semantic_events=[event.to_dict() for event in parsed.semantic_events],
                outcome_assessment=asdict(frame.outcome_assessment) if frame.outcome_assessment else {},
                belief_hash_after=frame.belief_hash_after,
                latency_ms=latency_ms,
            )
        )

    def _bind_aura_services(self) -> None:
        """Connect kernel learning to Aura's existing organs when present."""
        try:
            from core.container import ServiceContainer
        except Exception:
            return
        self.world_state = ServiceContainer.get("world_state", default=None)
        self.causal_model = ServiceContainer.get("causal_world_model", default=self.causal_model)
        self.episodic_memory = ServiceContainer.get("episodic_memory", default=self.episodic_memory)
        self.macro_inducer = ServiceContainer.get("macro_inducer", default=self.macro_inducer)
        try:
            ServiceContainer.register_instance(f"environment_kernel:{self.environment_id}", self, required=False)
        except Exception:
            pass

    def _publish_observation(self, frame: EnvironmentFrame) -> None:
        if self.world_state is None:
            return
        try:
            parsed = frame.parsed_state
            self.world_state.record_event(
                f"{self.environment_id} observation context={parsed.context_id}",
                source=f"environment:{self.environment_id}",
                salience=0.5 + min(0.4, max(parsed.uncertainty.values()) if parsed.uncertainty else 0.0),
                ttl=900.0,
                observation_id=frame.observation.stable_hash(),
                parsed_state=parsed.stable_hash(),
                context_id=parsed.context_id,
            )
            self.world_state.set_belief(
                f"environment.{self.environment_id}.context",
                parsed.context_id,
                confidence=0.85,
                source="environment_kernel",
                ttl=900.0,
            )
        except Exception:
            pass

    def _publish_outcome(self, frame: EnvironmentFrame, semantic_events: list, resource_delta: dict[str, float]) -> None:
        if self.world_state is None or frame.action_intent is None or frame.outcome_assessment is None:
            return
        try:
            self.world_state.record_event(
                f"{self.environment_id} action {frame.action_intent.name} score={frame.outcome_assessment.success_score:.2f}",
                source=f"environment:{self.environment_id}",
                salience=0.65 if frame.outcome_assessment.success_score >= 0.5 else 0.85,
                ttl=1800.0,
                action=frame.action_intent.name,
                observed_events=[event.name for event in semantic_events],
                resource_delta=resource_delta,
                receipt_id=frame.receipt.receipt_id if frame.receipt else None,
            )
        except Exception:
            pass

    def _terminal_reason(
        self,
        outcome: OutcomeAssessment,
        observation: Observation,
        parsed: ParsedState,
    ) -> str:
        text = observation.text or (observation.raw if isinstance(observation.raw, str) else "")
        if outcome.is_death or self.run_manager.detect_death(text):
            return "death"
        labels = " ".join(event.label.lower() for event in parsed.semantic_events)
        if any(token in labels for token in ("victory", "task_completed", "ascended", "success")):
            return "success"
        return ""

    def _infer_run_mode(self):
        mode = getattr(self.adapter, "mode", None)
        value = getattr(mode, "value", None)
        if value == "simulated" or getattr(self.adapter, "_simulated", False):
            return "simulated_canary"
        if getattr(self.adapter, "mode", "") == "fixture_replay":
            return "fixture_replay"
        return "strict_real"


__all__ = ["EnvironmentFrame", "EnvironmentKernel"]
