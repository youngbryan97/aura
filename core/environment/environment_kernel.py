"""Universal typed observe -> model -> gate -> act -> learn kernel."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
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
        self.state_compiler = state_compiler or StateCompiler()
        self.command_compiler = command_compiler or CommandCompiler(self.environment_id)
        self.belief = EnvironmentBeliefGraph()
        self.modal_manager = ModalManager()
        self.gateway = EnvironmentActionGateway(modal_manager=self.modal_manager)
        self.homeostasis = Homeostasis()
        self.options = OptionLibrary()
        self.option_compiler = OptionCompiler()
        self.simulator = TacticalSimulator()
        self.prediction_error = PredictionErrorComputer()
        self.outcomes = OutcomeAttributor()
        self.crisis = CrisisManager()
        self.blackbox = BlackBoxRecorder(trace_path)
        self.episode: EpisodeManager | None = None
        self.run_id = ""
        self.frames: list[EnvironmentFrame] = []

    async def start(self, *, run_id: str, seed: int | None = None) -> None:
        self.run_id = run_id
        self.episode = EpisodeManager(run_id, self.environment_id)
        await self.adapter.start(run_id=run_id, seed=seed)

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
            resources = self.homeostasis.assess(self.homeostasis.extract(parsed))
            if resources.critical_resources:
                option = self.options.get("STABILIZE_RESOURCE")
            else:
                option = self.options.get("OBSERVE_MORE")
            frame.selected_option = option.name
            intent = self.option_compiler.compile(option)

        sim = self.simulator.simulate(self.belief, intent)
        decision = self.gateway.approve(
            intent,
            modal_state=parsed.modal_state,
            simulation=sim,
            uncertainty=max(parsed.uncertainty.values()) if parsed.uncertainty else 0.0,
            context_id=parsed.context_id or "default",
            authority_receipt_id=None if intent.requires_authority else "not_required",
        )
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
            gateway_decision_id=decision.decision_id,
            will_receipt_id="not_required" if not intent.requires_authority else None,
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
        observed_events = ["execution_ok" if result.ok else "execution_error"]
        expected = sim.hypotheses[0].predicted_events if sim.hypotheses else []
        pe = self.prediction_error.compute(action_id=intent.intent_id(), expected_events=expected, observed_events=observed_events)
        outcome = self.outcomes.assess(
            action=intent.name,
            expected_effect=intent.expected_effect,
            observed_events=observed_events,
            prediction_error=pe,
            information_gain=sim.hypotheses[0].information_gain if sim.hypotheses else 0.0,
        )
        frame.outcome_assessment = outcome
        receipt.execution_result_id = result.command_id
        receipt.finalize(status="executed" if result.ok else "failed", belief_hash_after=self.belief.stable_hash())
        self._trace(frame, latency_ms=(time.time() - started) * 1000)
        return frame

    async def close(self) -> None:
        if self.episode:
            self.episode.transition(terminal=True)
            self.episode.transition()
        await self.adapter.close()

    def _trace(self, frame: EnvironmentFrame, *, latency_ms: float = 0.0) -> None:
        parsed = frame.parsed_state
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


__all__ = ["EnvironmentFrame", "EnvironmentKernel"]
