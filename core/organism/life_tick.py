"""Execution steps for a standalone boxed-organism simulation tick.

Production cognition and state integration are owned by ``MindTick`` and the
kernel. This processor is retained for isolated scenario/evaluation worlds.
"""

import logging
import time
import uuid
from typing import Dict, Any, Optional

from core.organism.life_state import LifeState
from core.runtime.errors import record_degradation
from core.organism.life_events import (
    PerceptionEvent,
    BodyStateChanged,
    BeliefUpdated,
    GoalCreated,
    AttentionSelected,
    PlanProposed,
    ActionRequested,
    ActionApproved,
    ActionExecuted,
    ConsequenceVerified,
    MemoryWritten,
    WelfareUpdated,
    ValueUpdated,
    IdentityUpdated,
    RepairProposed,
    PrivacyClass,
)
from core.runtime.version import VERSION

logger = logging.getLogger("Organism.LifeTick")

_LIFE_TICK_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


class LifeTickProcessor:
    """Coordinates the exact chronological transitions of a single life tick."""

    def __init__(self, container: Optional[Any] = None):
        self.container = container

    async def _publish_event(self, topic: str, event: Any) -> None:
        try:
            from core.event_bus import get_event_bus, EventPriority

            eb = get_event_bus()
            if hasattr(event, "model_dump"):
                data = event.model_dump()
            elif hasattr(event, "dict"):
                data = event.dict()
            else:
                data = event
            await eb.publish(topic, data, priority=EventPriority.AUTONOMIC)
        except _LIFE_TICK_RECOVERABLE_ERRORS as e:
            record_degradation("organism.publish_event", e)

    @staticmethod
    def _new_id(prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4()}"

    @staticmethod
    def _event_content(value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {"value": value}

    @staticmethod
    def _goal_owner(state: LifeState) -> str:
        return str(state.identity.get("owner") or state.identity.get("operator") or "operator")

    async def execute_tick(self, state: LifeState) -> None:
        """Executes the full 13-stage organic pipeline."""
        state.timestamp = time.time()
        state.tick_count += 1

        # 1. Perceive world
        await self._perceive_world(state)

        # 2. Update body/interoception
        await self._update_body(state)

        # 3. Update beliefs/world model
        await self._update_beliefs(state)

        # 4. Update goals/drives/preferences
        await self._update_goals_drives_preferences(state)

        # 5. Choose attention
        await self._choose_attention(state)

        # 6. Deliberate
        await self._deliberate(state)

        # 7. Act or inhibit action
        action_intent = await self._determine_action(state)
        receipt = None
        if action_intent:
            receipt = await self._act_or_inhibit(state, action_intent)

        # 8. Verify consequence
        if receipt:
            await self._verify_consequence(state, receipt)

        # 9. Update memory
        await self._update_memory(state, receipt)

        # 10. Update welfare
        await self._update_welfare(state)

        # 11. Update values
        await self._update_values(state)

        # 12. Consolidate identity
        await self._consolidate_identity(state)

        # 13. Sleep/dream/train/repair
        await self._offline_cycle(state)

    async def _perceive_world(self, state: LifeState) -> None:
        try:
            from core.body.body_runtime import get_body_runtime

            body = get_body_runtime()
            observations = await body.perceive_all(state)
            state.world_model["last_observations"] = observations
            state.world_model["last_sensor_health"] = body.summarize_sensor_health(observations)

            # Build and publish PerceptionEvent
            event = PerceptionEvent(
                event_id=self._new_id("perc"),
                source="perception",
                modality="multimodal",
                parsed_content=observations,
                confidence=0.95,
                uncertainty=0.05,
                privacy_class=PrivacyClass.LOCAL_ONLY,
            )
            await self._publish_event("organism/perception", event)
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.perceive", exc)
            logger.warning("Perception tick step failed: %s", exc)

    async def _update_body(self, state: LifeState) -> None:
        try:
            from core.body.body_runtime import get_body_runtime

            body = get_body_runtime()
            status = await body.get_system_status()
            state.body.battery_level = status.get("battery", 100.0)
            state.body.cpu_usage = status.get("cpu", 10.0)
            state.body.memory_usage = status.get("memory", 50.0)
            state.body.current_focus_app = status.get("focus_app", "Terminal")
            state.body.clipboard_content = status.get("clipboard", "")

            # Apply resource scaling recommendations
            scaling = body.calculate_resource_scaling(status)
            state.world_model["active_resource_scaling"] = scaling
            sensor_health = state.world_model.get("last_sensor_health", {})
            if not isinstance(sensor_health, dict):
                sensor_health = {}
            actuator_health = body.summarize_actuator_health()
            state.world_model["last_actuator_health"] = actuator_health

            # Build and publish BodyStateChanged
            event = BodyStateChanged(
                event_id=self._new_id("body"),
                source="body",
                energy_budget=state.body.battery_level,
                thermal_pressure=status.get("temperature", 42.0),
                memory_pressure=state.body.memory_usage,
                model_capacity=scaling["model_capacity"],
                sensor_health={str(k): bool(v) for k, v in sensor_health.items()},
                actuator_health=actuator_health,
                governance_integrity=scaling["governance_integrity"],
                attention_saturation=float(len(state.cognition.current_goals)),
                uncertainty_load=state.cognition.uncertainty_score,
                error_rate=0.0,
                repair_need=scaling["defer_dream_cycles"],
                user_interruptibility=10.0,
            )
            await self._publish_event("organism/body_state", event)
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.body", exc)

    async def _update_beliefs(self, state: LifeState) -> None:
        try:
            from core.world.belief_revision import BeliefRevisionEngine

            engine = BeliefRevisionEngine()
            await engine.revise_beliefs(state)

            # Build and publish BeliefUpdated
            active_beliefs = state.world_model.get("active_beliefs", {})
            event = BeliefUpdated(
                event_id=self._new_id("belief"),
                source="beliefs",
                belief_id="surprise_index",
                content=f"Surprise score measured at {active_beliefs.get('surprise_index', 0.0)}",
                confidence=1.0 - state.cognition.uncertainty_score,
                decay_policy="dynamic_decay",
                contradictions=[],
                supporting_evidence=["observations"],
                downstream_uses=["attention", "welfare"],
            )
            await self._publish_event("organism/beliefs", event)
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.beliefs", exc)

    async def _update_goals_drives_preferences(self, state: LifeState) -> None:
        try:
            from core.agency.mission_manager import get_mission_manager

            manager = get_mission_manager()
            await manager.update_goals_and_drives(state)

            # Publish GoalCreated events for new goals
            for goal in state.cognition.current_goals:
                event = GoalCreated(
                    event_id=self._new_id("goal"),
                    source="agency",
                    goal_id=goal.get("id", "unknown"),
                    owner=self._goal_owner(state),
                    origin=str(goal.get("origin", "runtime")),
                    status=goal.get("status", "pending"),
                    priority=float(goal.get("priority", 1.0)),
                    urgency=float(goal.get("urgency", 1.0)),
                    importance=float(goal.get("importance", 1.0)),
                    success_criteria=list(goal.get("success_criteria", ["verified"])),
                    failure_criteria=list(goal.get("failure_criteria", ["unverified_effect"])),
                    allowed_tools=list(goal.get("allowed_tools", [])),
                    forbidden_tools=list(goal.get("forbidden_tools", [])),
                    risk_class=str(goal.get("risk_class", "low")),
                    dependencies=list(goal.get("dependencies", [])),
                )
                await self._publish_event("organism/goals", event)
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.goals", exc)

    async def _choose_attention(self, state: LifeState) -> None:
        try:
            from core.executive.attention_controller import AttentionController

            controller = AttentionController()
            state.cognition.active_attention = await controller.focus_attention(state)

            # Build and publish AttentionSelected
            event = AttentionSelected(
                event_id=self._new_id("attn"),
                source="executive",
                focus_id=f"focus-{state.tick_count}",
                target_object=state.cognition.active_attention,
                reason_for_attention="priority_reconciliation",
                salience_score=1.0,
                estimated_cost=0.1,
            )
            await self._publish_event("organism/attention", event)
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.attention", exc)

    async def _deliberate(self, state: LifeState) -> None:
        try:
            from core.executive.executive_kernel import DeliberationEngine

            engine = DeliberationEngine()
            await engine.deliberate(state)

            # Process global workspace tick (metacognition and monologue)
            try:
                from core.workspace.global_workspace import GlobalWorkspace

                gw = GlobalWorkspace()
                gw.process_workspace_tick(state)
            except _LIFE_TICK_RECOVERABLE_ERRORS as e:
                record_degradation("organism.workspace_tick", e)

            # Build and publish PlanProposed
            plans = state.world_model.get("active_plans", [])
            if plans:
                for plan in plans:
                    event = PlanProposed(
                        event_id=self._new_id("plan"),
                        source="executive",
                        plan_id=str(plan.get("plan_id", f"plan-{state.tick_count}")),
                        goal_id=str(plan.get("goal_id", "active")),
                        steps=[{"step": plan.get("deliberation_plan", "deliberate")}],
                        tools_required=list(plan.get("tools_required", [])),
                        permissions_required=list(plan.get("permissions_required", [])),
                        risks=list(plan.get("risks", [])),
                        expected_observations=list(plan.get("expected_observations", [])),
                        fallbacks=list(plan.get("fallbacks", [])),
                        verification_method="postcondition",
                        abort_conditions=list(plan.get("abort_criteria", [])),
                        estimated_cost=float(plan.get("estimated_cost", 0.0)),
                    )
                    await self._publish_event("organism/plans", event)
            else:
                event = PlanProposed(
                    event_id=self._new_id("plan"),
                    source="executive",
                    plan_id=f"plan-{state.tick_count}",
                    goal_id="active",
                    steps=[{"step": "deliberate"}],
                    tools_required=[],
                    permissions_required=[],
                    risks=[],
                    expected_observations=[],
                    fallbacks=[],
                    verification_method="postcondition",
                    abort_conditions=[],
                    estimated_cost=0.0,
                )
                await self._publish_event("organism/plans", event)

        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.deliberate", exc)

    async def _determine_action(self, state: LifeState) -> Optional[Dict[str, Any]]:
        if not state.cognition.pending_actions:
            return None
        return state.cognition.pending_actions.pop(0)

    async def _act_or_inhibit(self, state: LifeState, intent: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from core.executive.inhibition_system import ActionInhibitor

            inhibitor = ActionInhibitor()

            channel = str(intent.get("channel", "unknown"))
            params = intent.get("params", {})
            if not isinstance(params, dict):
                params = {}
            action_id = str(
                intent.get("action_id") or params.get("action_id") or f"act-{state.tick_count}"
            )
            risk_score = float(
                intent.get("risk_score", 4.0 if channel in ["terminal", "file"] else 1.0)
            )
            capability_token = params.get("capability_token") or intent.get("capability_token")

            # 1. Requested
            req_event = ActionRequested(
                event_id=self._new_id("req"),
                source="executive",
                action_id=action_id,
                channel=channel,
                params=params,
                risk_score=risk_score,
                requires_approval=risk_score >= 4.0,
            )
            await self._publish_event("organism/action_requested", req_event)

            if await inhibitor.should_inhibit(state, intent):
                logger.info("Action inhibited by governance/moral safety kernel: %s", intent)
                return {"status": "inhibited", "intent": intent}

            # 2. Approved
            app_event = ActionApproved(
                event_id=self._new_id("appr"),
                source="executive",
                action_id=req_event.action_id,
                approved_by="inhibition_system",
                posture="owner_autonomous",
                capability_token=str(capability_token) if capability_token else None,
                authority_receipt_id=(
                    str(params.get("authority_receipt_id") or params.get("will_receipt_id"))
                    if params.get("authority_receipt_id") or params.get("will_receipt_id")
                    else None
                ),
            )
            await self._publish_event("organism/action_approved", app_event)

            # 3. Executed
            from core.body.action_body import get_action_body

            action_body = get_action_body()
            # CP126 92b64654: a postcondition can only be a DIFF if something
            # recorded the target's state BEFORE the action ran.
            pre_snapshot = None
            if params.get("path"):
                from core.body.action_postcondition import snapshot_path_async

                pre_snapshot = await snapshot_path_async(params.get("path"))
            receipt = await action_body.execute_action(intent, state)
            if not isinstance(receipt, dict):
                receipt = {"status": "failed", "error": "motor returned non-dict receipt"}

            # Copy causal plan and goal metadata to receipt for verification tracking
            if "plan_id" in params:
                receipt["plan_id"] = params["plan_id"]
            if "goal_id" in params:
                receipt["goal_id"] = params["goal_id"]
            receipt["action_id"] = req_event.action_id
            receipt.setdefault("receipt_id", self._new_id("rec"))
            if pre_snapshot is not None:
                receipt["pre_action_snapshot"] = pre_snapshot

            exec_event = ActionExecuted(
                event_id=self._new_id("exec"),
                source="body",
                action_id=req_event.action_id,
                receipt_id=str(receipt["receipt_id"]),
                channel=channel,
                status=receipt.get("status", "success"),
                stdout=receipt.get("stdout"),
                stderr=receipt.get("stderr"),
                exit_code=receipt.get("exit_code"),
            )
            await self._publish_event("organism/action_executed", exec_event)
            return receipt
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.act", exc)
            return {"status": "failed", "error": str(exc), "intent": intent}

    async def _verify_consequence(self, state: LifeState, receipt: Dict[str, Any]) -> None:
        try:
            from core.body.action_postcondition import ActionPostconditionVerifier

            verifier = ActionPostconditionVerifier()
            verification = await verifier.verify(
                receipt, state, before=receipt.get("pre_action_snapshot"),
            )

            # Get the expected observations from the active plan
            expected = "successful_status"
            plans = state.world_model.get("active_plans", [])
            for p in plans:
                if p.get("plan_id") == receipt.get("plan_id"):
                    if p.get("expected_observations"):
                        expected = p.get("expected_observations")[0]
                    break

            observed = "success" if verification.get("success") else "failed"

            # Check if there is a mismatch
            success = verification.get("success", False)
            mismatch = None
            if not success:
                mismatch = f"Expected '{expected}', but observed '{observed}'"

            # Update prediction accuracy history
            prediction_history = state.world_model.setdefault("prediction_history", [])
            prediction_history.append(
                {
                    "action_id": receipt.get("action_id"),
                    "expected": expected,
                    "observed": observed,
                    "success": success,
                    "timestamp": time.time(),
                }
            )

            # Build and publish ConsequenceVerified
            event = ConsequenceVerified(
                event_id=self._new_id("ver"),
                source="body",
                action_id=receipt.get("action_id", "unknown"),
                expected_evidence=expected,
                observed_evidence=observed,
                success=success,
                side_effects=verification.get("side_effects", []),
                mismatch_description=mismatch,
            )
            await self._publish_event("organism/consequences", event)

        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.verify", exc)

    async def _update_memory(self, state: LifeState, receipt: Optional[Dict[str, Any]]) -> None:
        try:
            from core.memory.autobiography import AutobiographyEngine

            engine = AutobiographyEngine()
            await engine.record_tick_event(state, receipt)

            # Build and publish MemoryWritten
            event = MemoryWritten(
                event_id=self._new_id("mem"),
                source="memory",
                memory_id=f"ep-{state.tick_count}",
                type="episodic",
                content=self._event_content(state.autobiographical_memory[-1])
                if state.autobiographical_memory
                else {},
                sensitivity=PrivacyClass.LOCAL_ONLY,
            )
            await self._publish_event("organism/memory", event)
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.memory", exc)

    async def _update_welfare(self, state: LifeState) -> None:
        try:
            from core.welfare.welfare_bus import WelfareBus

            bus = WelfareBus()
            await bus.evaluate_welfare(state)

            # Tick the viability engine and update LifeState viability
            from core.organism.viability import get_viability

            viability_engine = get_viability()
            viability_state = viability_engine.tick()
            state.welfare.viability_state = viability_state.value

            # Build and publish WelfareUpdated
            event = WelfareUpdated(
                event_id=self._new_id("welf"),
                source="welfare",
                welfare_index=state.welfare.welfare_index,
                energy=state.welfare.energy,
                stress=state.welfare.stress,
                distress_level=state.welfare.distress_level,
                sleep_debt=state.welfare.sleep_debt,
            )
            await self._publish_event("organism/welfare", event)
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.welfare", exc)

    async def _update_values(self, state: LifeState) -> None:
        try:
            from core.values.preference_provenance import PreferenceProvenanceManager

            manager = PreferenceProvenanceManager()
            await manager.evaluate_preferences(state)

            # Build and publish ValueUpdated
            event = ValueUpdated(
                event_id=self._new_id("val"),
                source="values",
                value_id="speed_accuracy",
                statement="balance_efficiency_and_correctness",
                priority=state.active_preferences.get("accuracy", 0.8),
                hard_limit=False,
                conflicts=[],
            )
            await self._publish_event("organism/values", event)
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.values", exc)

    async def _consolidate_identity(self, state: LifeState) -> None:
        try:
            from core.identity.identity_kernel import IdentityKernel

            kernel = IdentityKernel()
            await kernel.guard_identity_continuity(state)

            # Build and publish IdentityUpdated
            event = IdentityUpdated(
                event_id=self._new_id("id"),
                source="identity",
                active_version=str(state.identity.get("version") or VERSION),
                active_modules=list(
                    state.identity.get("active_modules", ["organism", "body", "memory"])
                ),
                disabled_modules=list(state.identity.get("disabled_modules", [])),
                known_limitations=["subjective_experience_not_established"],
                active_permissions=list(
                    state.identity.get("active_permissions", ["local_execution"])
                ),
                capability_boundaries=list(
                    state.identity.get("capability_boundaries", ["sandbox"])
                ),
            )
            await self._publish_event("organism/identity", event)
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.identity", exc)

    async def _offline_cycle(self, state: LifeState) -> None:
        try:
            from core.sleep.sleep_cycle import SleepManager

            manager = SleepManager()
            if await manager.should_trigger_sleep(state):
                # Build and publish RepairProposed
                rep_event = RepairProposed(
                    event_id=self._new_id("rep"),
                    source="sleep",
                    repair_id=f"rep-{state.tick_count}",
                    subsystem="welfare",
                    issue_description="energy_depleted",
                    patch_diff="restore_energy_budget",
                    rollback_plan="exit_sleep",
                )
                await self._publish_event("organism/repair", rep_event)
                await manager.execute_sleep_cycle(state)
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.sleep", exc)
