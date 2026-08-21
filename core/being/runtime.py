from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Any

from core.governance.recovery_authority import (
    is_internal_recovery_context,
    is_restorative_consolidation,
)
from core.governance_context import (
    get_active_governance,
    normalize_governance_domain,
)
from core.runtime.errors import record_degradation
from core.runtime.numeric_safety import validated_unit

from .affective_valence import AffectiveValenceEngine
from .aura_now import (
    AuraNow,
    BodyState,
    MemoryContext,
    ReportBoundary,
    SelfState,
    WillStateSnapshot,
    WorldState,
)
from .blind_introspection import BlindIntrospector
from .body_state_service import BodyStateService
from .continuous_substrate import ContinuousSelfField
from .functional_soul import FunctionalSoul
from .higher_order_monitor import HigherOrderMonitor
from .interoceptive_model import InteroceptiveModel
from .introspection_renderer import IntrospectionRenderer
from .policy_coupler import ClosedLoopPolicyCoupler
from .self_model_attractor import FunctionalIAttractor
from .self_ownership import OwnershipTracker
from .self_report_calibrator import SelfReportCalibrator
from .semantic_stream import SemanticStream
from .welfare_state import WelfareState
from .workspace_ignition import WorkspaceIgnition

logger = logging.getLogger("Aura.BeingRuntime")


#: Single source of truth for which action domains are consequential.
#: CP126 c62962b4: this was a duplicated inline allowlist, so a misspelled or
#: newly added domain skipped body accounting and every consequential defer.
CONSEQUENTIAL_ACTION_DOMAINS = frozenset({
    "tool_execution",
    "memory_write",
    "state_mutation",
    "initiative",
    "exploration",
    "semantic_weight_update",
    "belief_update",
    "environment_action",
    "external_action",
    "file_write",
    "network_call",
    "cloud_call",
    "ci_cd",
    "self_modification",
    "cloud_fallback",
})

#: Domains that are known and NOT consequential. Anything outside the union of
#: these two sets is unknown, and unknown fails closed to consequential.
NON_CONSEQUENTIAL_ACTION_DOMAINS = frozenset({
    "response",
    "reflection",
    "stabilization",
    "observation",
    "introspection",
    "",
})

KNOWN_ACTION_DOMAINS = CONSEQUENTIAL_ACTION_DOMAINS | NON_CONSEQUENTIAL_ACTION_DOMAINS

_RUNTIME_BOUND_PASSIVE_OBSERVATIONS = frozenset(
    {
        (
            "environment_action",
            "browser_controller",
            "browser_controller.get_open_tabs",
        ),
    }
)


def is_consequential_domain(domain_name: str) -> bool:
    """Whether this domain must pay body cost and honour consequential defers.

    An unrecognized domain is treated as consequential: the failure mode of
    guessing "harmless" is that a new sink bypasses the whole policy.
    """
    name = str(domain_name or "").strip().lower()
    if name in CONSEQUENTIAL_ACTION_DOMAINS:
        return True
    return name not in NON_CONSEQUENTIAL_ACTION_DOMAINS


def is_runtime_bound_passive_observation(
    domain_name: str,
    context: dict[str, Any] | None,
) -> bool:
    """Recognize an exact read contract stamped by ActionExecutor.

    This does not grant authority. It only prevents a passive observation,
    already admitted by Will and privacy policy, from inheriting action-only
    welfare/body defers. The action name and source are stamped after caller
    context is merged, so payload booleans alone cannot enter this lane.
    """

    ctx = dict(context or {})
    contract = (
        str(domain_name or "").strip().lower(),
        str(ctx.get("action_executor_source") or "").strip(),
        str(ctx.get("action_executor_action_name") or "").strip(),
    )
    return bool(
        contract in _RUNTIME_BOUND_PASSIVE_OBSERVATIONS
        and ctx.get("passive_observation") is True
        and ctx.get("read_only") is True
        and str(ctx.get("effect_scope") or "").strip().lower() == "read_only"
        and ctx.get("no_external_effects") is True
        and ctx.get("user_visible_desktop_effect") is False
    )


def attested_context_flag(
    context: dict,
    flag: str,
    *,
    domain: str,
    action: str,
    consume: bool = False,
    child_receipt: str = "",
) -> bool:
    """A context flag that is only true when a capability token backs it.

    CP126 3b1a9177 / 310a67ee: the foreground-desktop exception and the
    continuity defers were cleared by plain booleans in a caller-supplied
    dictionary — no signed principal, capability lease, gesture nonce, target,
    scope or expiry. A caller could therefore hand itself the authority the
    policy was meant to withhold.

    The flag must now be accompanied by a capability token that validates for
    THIS domain and action. Without the token the flag is ignored, and the
    refusal is recorded so an operator can see the attempt.
    """
    if not context.get(flag):
        return False
    token_str = str(context.get("capability_token") or "").strip()
    if not token_str:
        logger.warning(
            "Context flag %r for %s/%s carried no capability token; ignoring it.",
            flag, domain, action,
        )
        return False
    try:
        from core.agency.capability_token import get_token_store

        store = get_token_store()
        if consume:
            store.validate_and_consume(
                token_str,
                domain=domain,
                action=action,
                child_receipt=child_receipt or f"being_runtime:{flag}",
                side_effects=[f"context_flag:{flag}"],
            )
        else:
            store.validate(token_str, domain=domain, action=action)
    except (PermissionError, ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "being_runtime",
            exc,
            severity="warning",
            action=f"ignored unattested context flag {flag!r} for {domain}/{action}",
        )
        return False
    return True


def active_governance_attests_private_maintenance(
    context: dict[str, Any],
    *,
    domain: str,
) -> bool:
    """Verify one private maintenance flag against the live local lease.

    Internal capture housekeeping already executes inside a short-lived local
    governance scope. Reusing that exact lease is stronger than minting a
    second bearer token: the attestation cannot survive the scope, move to a
    different task, or silently change operation.
    """

    if not context.get("internal_runtime_maintenance"):
        return False
    active = get_active_governance()
    if active is None or not active.authorizes:
        return False
    constraints = dict(active.constraints or ())
    return bool(
        normalize_governance_domain(active.domain)
        == normalize_governance_domain(domain)
        and active.receipt_id
        == str(context.get("capability_token_id") or "").strip()
        and active.source
        == str(context.get("authority_origin") or "").strip()
        and constraints.get("governance_origin") == "local_internal"
        and constraints.get("runtime_generated") is True
        and str(constraints.get("op") or "").strip().lower()
        == str(context.get("maintenance_operation") or "").strip().lower()
    )


def _is_explicit_owner_request(context: Any) -> bool:
    """Did the person ask for this, in the foreground, just now?

    Deliberately requires an EXPLICIT signal rather than merely "not
    autonomous". A background loop that forgets to set a flag must not inherit
    the owner's standing; the flags below are set by the routes that actually
    carry a live request.
    """

    ctx = dict(context or {}) if isinstance(context, Mapping) else {}
    return any(
        bool(ctx.get(flag))
        for flag in (
            "foreground_request",
            "user_explicitly_authorized",
            "user_explicit_action_request",
            "desktop_execution_contract",
            "user_visible_desktop_action",
        )
    )


class BeingRuntime:
    """Canonical LAMP/AuraNow runtime surface.

    Welfare, body, semantic stream, blind introspection, and self-report
    calibration are wired into every sample() call. The BeingRuntime
    action-constraint path consumes these signals before consequential
    behavior.
    """

    def __init__(self, *, field_dim: int = 32) -> None:
        # CP126 c28972f3 / 5d2fe427: sample() mutated interoception, body,
        # welfare, the semantic stream, the introspector, the field, ownership
        # and every _last_* attribute with no lock, so action_policy could
        # combine `now` from one request with welfare or felt state from
        # another — and any mid-sequence exception left a partially advanced
        # self-state. One reentrant lock serializes the whole sample.
        self._sample_lock = threading.RLock()
        self.field = ContinuousSelfField(dim=field_dim)
        self.interoception = InteroceptiveModel()
        self.affect = AffectiveValenceEngine()
        self.workspace = WorkspaceIgnition()
        self.ownership = OwnershipTracker()
        self.monitor = HigherOrderMonitor()
        self.soul = FunctionalSoul()
        self.renderer = IntrospectionRenderer()
        self._last_sample_monotonic = time.monotonic()
        self._last_now: AuraNow | None = None
        self._running = False

        # Welfare/body/introspection subsystems.
        self.body_service = BodyStateService.get()
        self.welfare = WelfareState.get()
        self.blind_introspector = BlindIntrospector()
        self.self_report_calibrator = SelfReportCalibrator()
        self.semantic_stream = SemanticStream.get()
        self._last_welfare: Any | None = None
        self._last_blind_report: Any | None = None
        self._last_body_snapshot: Any | None = None
        self._last_causal_self_vector: Any | None = None
        self._last_causal_valenced_workspace: Any | None = None
        self._last_unified_felt: Any | None = None
        self._lesion_controller_registered = False

        # The functional "I" and the policy it couples into.
        #
        # Both classes were substantial, correct and unreachable: nothing in
        # any production path constructed either one, so a controller that
        # computes continuity, coherence, integrity, identity tension, agency
        # readiness and first-person confidence — and maps them onto
        # temperature, verification threshold, retrieval depth and tool risk —
        # was doing none of that to anything. The attractor's own docstring
        # names the condition it failed: "real only when ... policy changes
        # downstream". A beautiful unused controller is not part of Aura's
        # operational cognition, and describing it as though it were was the
        # imprecision worth correcting.
        #
        # They live here because this is where the evidence they need already
        # is: AuraNow, welfare, the causal self vector and the action policy
        # all pass through one lock in _sample_locked.
        self.self_attractor = FunctionalIAttractor()
        self.policy_coupler = ClosedLoopPolicyCoupler(production_mode=True)
        self._last_self_attractor_state: Any | None = None
        self._last_closed_loop_policy: Any | None = None

    def start(self, *, hz: float = 20.0) -> None:
        if self._running:
            return
        self.field.start(hz=hz)
        self._register_lesion_targets()
        self._running = True

    def stop(self) -> None:
        self.field.stop()
        self._running = False

    def _register_lesion_targets(self) -> None:
        """Register all subsystems with the canonical LesionController."""
        if self._lesion_controller_registered:
            return
        try:
            from core.runtime.lesion_controller import LesionController
        except (ImportError, RuntimeError) as exc:
            record_degradation(
                "being_runtime",
                exc,
                severity="warning",
                action="deferred lesion-controller registration during BeingRuntime start",
            )
            logger.warning("LesionController registration deferred: %s", exc)
            return

        ctrl = LesionController.get()
        failed_targets: list[str] = []
        for name, subsystem in (
            ("welfare", self.welfare),
            ("body", self.body_service),
            ("introspection", self.blind_introspector),
            ("self_report", self.self_report_calibrator),
            ("semantic_stream", self.semantic_stream),
            ("affect", self.affect),
            ("workspace", self.workspace),
        ):
            try:
                ctrl.register(name, subsystem)
            except (AttributeError, TypeError, ValueError) as exc:
                failed_targets.append(name)
                record_degradation(
                    "being_runtime",
                    exc,
                    severity="warning",
                    action=f"left lesion target {name} unregistered until interface is fixed",
                    extra={"lesion_target": name},
                )

        if not failed_targets:
            self._lesion_controller_registered = True
        else:
            logger.warning("Lesion target registration incomplete: %s", failed_targets)

    @property
    def last_now(self) -> AuraNow | None:
        return self._last_now

    def sample(
        self,
        state: Any | None = None,
        *,
        objective: str = "",
        candidate_action: str = "",
        predicted_outcome: str = "",
        actual_outcome: str = "",
        tool_failed: bool = False,
        external_override: bool = False,
        lesions: set[str] | None = None,
    ) -> AuraNow:
        # One coherent, serialized sample (CP126 c28972f3 / 5d2fe427).
        with self._sample_lock:
            return self._sample_locked(
                state,
                objective=objective,
                candidate_action=candidate_action,
                predicted_outcome=predicted_outcome,
                actual_outcome=actual_outcome,
                tool_failed=tool_failed,
                external_override=external_override,
                lesions=lesions,
            )

    def _sample_locked(
        self,
        state: Any | None = None,
        *,
        objective: str = "",
        candidate_action: str = "",
        predicted_outcome: str = "",
        actual_outcome: str = "",
        tool_failed: bool = False,
        external_override: bool = False,
        lesions: set[str] | None = None,
    ) -> AuraNow:
        lesions = set(lesions or set())
        monotonic_now = time.monotonic()
        idle_elapsed = max(0.0, monotonic_now - self._last_sample_monotonic)
        self._last_sample_monotonic = monotonic_now

        body = BodyState.from_aura_state(state, idle_elapsed_s=idle_elapsed)
        world = WorldState.from_aura_state(state, objective=objective)
        prediction = self.interoception.compare(body, candidate_action=candidate_action or objective)

        # Feed raw body state into the body service before welfare is computed.
        self.body_service.update_body(body)
        body_snapshot = self.body_service.snapshot()

        aura_affect = getattr(state, "affect", None)
        base_valence = float(getattr(aura_affect, "valence", 0.0) or 0.0)
        base_arousal = float(getattr(aura_affect, "arousal", 0.5) or 0.5)
        affect = self.affect.compute(
            body=body,
            prediction=prediction,
            world=world,
            base_valence=base_valence,
            base_arousal=base_arousal,
            lesion="affect" in lesions,
        )

        self_state = self._build_self_state(state, lesions)
        memory_context = self._build_memory_context(state, lesions)

        # Compute welfare from body, affect, prediction, memory, and continuity.
        welfare_inputs = self.welfare.gather_inputs(
            body=body_snapshot,
            affect_distress=affect.distress,
            affect_valence=affect.valence,
            prediction_error=prediction.free_energy,
            memory_coherence=1.0 - memory_context.memory_conflict,
            tool_reliability=max(0.0, 1.0 - body.tool_failure_pressure),
            model_stability=self._measure_model_stability(),
            continuity_risk=self_state.continuity_risk,
        )
        welfare_outputs = self.welfare.compute(welfare_inputs)

        # Update the semantic stream with current welfare/body state.
        self.semantic_stream.update_welfare(
            welfare_score=welfare_outputs.welfare_score,
            distress=welfare_outputs.distress,
            fatigue=body_snapshot.fatigue,
            recovery_drive=welfare_outputs.recovery_drive,
            body_health=body_snapshot.operational_health,
        )
        if objective:
            self.semantic_stream.update_situation("active_task")
        self.semantic_stream.evolve()

        # Classify internal state from control variables, without label prompting.
        blind_trace = self.blind_introspector.build_trace(
            distress=welfare_outputs.distress,
            body_pressure=body_snapshot.total_pressure,
            prediction_error=prediction.free_energy,
            memory_coherence=welfare_inputs.memory_coherence,
            tool_reliability=welfare_inputs.tool_reliability,
            goal_frustration=welfare_inputs.goal_frustration,
            social_trust=welfare_inputs.social_trust,
            continuity_risk=welfare_inputs.continuity_risk,
            fatigue=body_snapshot.fatigue,
            recovery_debt=body_snapshot.recovery_debt,
            curiosity=affect.curiosity,
            confidence=welfare_outputs.confidence,
        )
        blind_report = self.blind_introspector.introspect(blind_trace)

        projection = (
            affect.valence,
            affect.arousal,
            affect.distress,
            affect.curiosity,
            prediction.free_energy,
            prediction.controllability,
        )
        field_packet = self.field.step(
            {
                "body_pressure": body.total_pressure,
                "prediction_error": prediction.free_energy,
                "attention_salience": 1.0 if world.focal_object else 0.0,
            },
            projection,
        )

        coalitions = self.workspace.build_coalitions(body=body, affect=affect, world=world)
        workspace_state, attention = self.workspace.ignite(
            coalitions,
            lesion="workspace_ignition" in lesions,
        )
        if world.focal_object and workspace_state.winner == "user_request":
            attention = attention.__class__(
                focal_object=world.focal_object,
                why_selected=attention.why_selected,
                stability=attention.stability,
                competing_objects=attention.competing_objects,
                control=attention.control,
            )

        ownership = self.ownership.assess(
            intended_action=candidate_action or objective,
            predicted_outcome=predicted_outcome,
            actual_outcome=actual_outcome,
            tool_failed=tool_failed,
            external_override=external_override,
            memory_influence=bool(getattr(getattr(state, "cognition", None), "long_term_memory", None)),
        )

        will = self._build_will_snapshot()
        provisional = AuraNow(
            tick=field_packet.tick,
            timestamp=field_packet.timestamp,
            monotonic_time=field_packet.monotonic_time,
            continuous_field=field_packet.state,
            body=body,
            world=world,
            attention=attention,
            affect=affect,
            self_model=self_state,
            memory_context=memory_context,
            workspace=workspace_state,
            will=will,
            prediction=prediction,
            ownership=ownership,
            report_boundary=ReportBoundary(),
            higher_order=(),
            private_residue_hash=field_packet.private_residue_hash,
        )
        higher_order = tuple(obs.to_dict() for obs in self.monitor.observe(provisional))
        now = AuraNow(
            **{
                **asdict(provisional),
                "body": body,
                "world": world,
                "attention": attention,
                "affect": affect,
                "self_model": self_state,
                "memory_context": memory_context,
                "workspace": workspace_state,
                "will": will,
                "prediction": prediction,
                "ownership": ownership,
                "report_boundary": provisional.report_boundary,
                "higher_order": higher_order,
            }
        )
        self._last_now = now
        self._last_welfare = welfare_outputs
        self._last_blind_report = blind_report
        self._last_body_snapshot = body_snapshot
        self._refresh_causal_self_vector(now)
        self._refresh_unified_felt_state(state, now, welfare_outputs)
        self._publish(now)
        return now

    def _refresh_unified_felt_state(self, state: Any, now: AuraNow, welfare_outputs: Any) -> None:
        """Reconcile the kernel affect/phi track with this being track into one
        authoritative felt-state, and measure their coherence. Best-effort: a failure
        here must never break a sample (the felt-state is additive to existing gates)."""
        try:
            from core.being.unified_felt_state import get_unified_felt_state

            self._last_unified_felt = get_unified_felt_state().reconcile(
                kernel_state=state, aura_now=now, welfare=welfare_outputs
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "being_runtime",
                exc,
                severity="debug",
                action="continued without unified felt-state reconciliation this sample",
            )
            self._last_unified_felt = None

    def _measure_model_stability(self) -> float:
        """Live LLM availability/reliability, not a hardcoded perfect score.

        CP126 7a841f23: this was pinned at 1.0, so welfare's integrity term
        always read the model as perfectly stable — the substrate could be
        failing every call and welfare would never notice.
        """
        try:
            from core.runtime.service_registry import get_runtime_service

            router = get_runtime_service("llm_router", default=None)
            if router is None:
                # Unknown is not perfect. Mid-scale, and say so.
                return 0.5
            for attribute in ("stability_score", "health_score", "reliability"):
                value = getattr(router, attribute, None)
                if callable(value):
                    value = value()
                if isinstance(value, (int, float)):
                    return float(validated_unit(value, name="model_stability"))
            health = getattr(router, "get_health", None)
            if callable(health):
                report = health() or {}
                if isinstance(report, dict):
                    for key in ("stability", "score", "availability"):
                        if isinstance(report.get(key), (int, float)):
                            return float(
                                validated_unit(report[key], name="model_stability")
                            )
                    if report.get("healthy") is not None:
                        return 1.0 if report.get("healthy") else 0.25
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "being_runtime",
                exc,
                severity="debug",
                action="used a neutral model-stability reading after the probe failed",
            )
        return 0.5

    def action_policy(
        self,
        now: AuraNow,
        *,
        domain: str = "",
        priority: float = 0.5,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Derive action constraints from the live AuraNow state + welfare.

        This is the operational bridge between the "inner life" substrate and
        behavior. Welfare, body, affect, active-inference prediction,
        workspace ignition, and ownership model MUST change consequential
        decisions — not merely decorate prompts.

        MANDATORY: Body cost is paid for every consequential action.
        """
        domain_name = str(domain or "").strip().lower()
        context = dict(context or {})
        # CP126 310a67ee: these converted qualifying writes from defer to
        # constrain on the strength of caller-supplied booleans alone. The
        # flags now require a capability token bound to this domain+action.
        memory_binding = str(context.get("memory_write_binding") or "").strip()
        continuity_action = (
            f"continuity_memory_write:{memory_binding}" if memory_binding else ""
        )
        continuity_memory_write = bool(
            domain_name == "memory_write"
            and continuity_action
            and (
                attested_context_flag(
                    context, "conversation_continuity",
                    domain=domain_name,
                    action=continuity_action,
                    consume=True,
                    child_receipt=f"being_runtime:{memory_binding}",
                )
                or attested_context_flag(
                    context, "explicit_observational_memory_write",
                    domain=domain_name,
                    action=continuity_action,
                    consume=True,
                    child_receipt=f"being_runtime:{memory_binding}",
                )
            )
            and not context.get("high_risk_memory_write")
        )
        evidence_binding = str(context.get("memory_write_binding") or "").strip()
        evidence_action = (
            f"internal_evidence_memory_write:{evidence_binding}"
            if evidence_binding
            else ""
        )
        internal_evidence_memory_write = bool(
            domain_name == "memory_write"
            and evidence_action
            and attested_context_flag(
                context,
                "internal_evidence_memory_write",
                domain=domain_name,
                action=evidence_action,
                consume=True,
                child_receipt=f"being_runtime:evidence:{evidence_binding}",
            )
            and not context.get("high_risk_memory_write")
        )
        foreground_continuity_state = bool(
            domain_name == "state_mutation"
            and attested_context_flag(
                context, "foreground_continuity_state",
                domain=domain_name, action="foreground_continuity_state",
            )
        )
        internal_runtime_maintenance = bool(
            domain_name == "file_write"
            and context.get("effect_scope") == "private_runtime_maintenance"
            and context.get("no_external_effects") is True
            and active_governance_attests_private_maintenance(
                context,
                domain=domain_name,
            )
        )
        explicit_foreground_desktop_tool = bool(
            domain_name in {"tool_execution", "environment_action", "external_action", "state_mutation"}
            # CP126 3b1a9177: the foreground-desktop exception trusted six
            # context booleans and cleared multiple defers. The two that
            # actually assert USER AUTHORITY must now be attested.
            and context.get("desktop_execution_contract")
            and context.get("foreground_request")
            and attested_context_flag(
                context, "user_explicitly_authorized",
                domain=domain_name, action="foreground_desktop_action",
            )
            and context.get("user_visible_desktop_action")
            and (
                domain_name == "tool_execution"
                or context.get("local_desktop_action")
                or context.get("desktop_task_owned_by")
            )
            and context.get("verification_required")
        )
        # CP126 c62962b4: consequential status was a duplicated inline string
        # allowlist, so a misspelled or newly added domain silently skipped
        # body accounting and every consequential defer. It is now one
        # module-level set, and an UNKNOWN domain is treated as consequential
        # — the fail-closed direction.
        constraints: list[str] = []
        blocks: list[str] = []
        defers: list[str] = []

        nominally_consequential = is_consequential_domain(domain_name)
        passive_observation = is_runtime_bound_passive_observation(
            domain_name,
            context,
        )
        consequential = nominally_consequential and not passive_observation
        # A source-bound append of already-observed runtime evidence is still a
        # consequential write, so body and integrity accounting remain active.
        # It does not require deliberative workspace, prediction, or agency to
        # reconstruct an answer: those gates can only discard the observation.
        deliberative_consequential = consequential and not internal_evidence_memory_write
        unknown_domain = bool(domain_name) and domain_name not in KNOWN_ACTION_DOMAINS
        if unknown_domain:
            constraints.append(
                f"unknown_action_domain_treated_as_consequential: {domain_name!r}"
            )
        if passive_observation:
            constraints.append("runtime_bound_passive_observation:read_only")
        repair_lane = (
            domain_name in {"stabilization", "reflection"}
            or is_internal_recovery_context(domain_name, context)
            or internal_runtime_maintenance
        )

        body_pressure = float(now.body.total_pressure)
        distress = float(now.affect.distress)
        controllability = float(now.prediction.controllability)
        free_energy = float(now.prediction.free_energy)
        ignition = float(now.workspace.ignition_strength)
        agency = float(now.ownership.agency_confidence)

        # Welfare-driven constraints.
        welfare = getattr(self, "_last_welfare", None)
        if welfare is not None:
            if welfare.action_inhibition > 0.5 and consequential and not repair_lane:
                defers.append(f"welfare_action_inhibition={welfare.action_inhibition:.3f}")
                constraints.append(f"welfare_inhibition_high: {welfare.action_inhibition:.3f}")

            if welfare.should_protect_integrity() and domain_name in {
                "memory_write", "self_modification", "belief_update",
            }:
                constraints.append(f"welfare_integrity_guard={welfare.integrity_guard:.3f}")
                if welfare.integrity_guard > 0.8 and not repair_lane:
                    defers.append("welfare_integrity_protection_active")

            if welfare.recovery_drive > 0.6 and consequential and not repair_lane:
                # Sleep-class restoration is exempt from THIS rule only:
                # deferring consolidation because recovery is needed is the
                # deadlock that froze the interior life live (11,738 defers,
                # 2,428 blocked consolidations, 7/17-7/21) — recovery drive
                # can only fall if restoration is allowed to run. Integrity
                # guards and action inhibition above still apply unchanged.
                if is_restorative_consolidation(context.get("source"), context) and domain_name in {
                    "state_mutation",
                    "memory_write",
                }:
                    constraints.append(
                        f"restorative_consolidation_lane: recovery_drive={welfare.recovery_drive:.3f}"
                    )
                elif internal_evidence_memory_write:
                    constraints.append(
                        "internal_evidence_commit_lane:"
                        f"recovery_drive={welfare.recovery_drive:.3f}"
                    )
                elif _is_explicit_owner_request(context):
                    # Rest instead of doing MORE OF YOUR OWN WORK is sound.
                    # Rest instead of doing the thing the person just asked for
                    # is not, and it is silent: they get "the desktop task lane
                    # did not complete", never "I am depleted".
                    #
                    # MEASURED live 2026-08-18. recovery_drive sat at 0.60-0.64
                    # for a whole session — just over the line — so every
                    # consequential action was deferred, 10,330 times in one
                    # log. At the moment an owner request was refused the
                    # context read foreground_request=True,
                    # user_explicitly_authorized=True,
                    # desktop_execution_contract=True,
                    # user_visible_desktop_action=True. Everything needed to
                    # know whose request it was, present and unused.
                    #
                    # The state is still recorded, so depletion stays visible
                    # and she can say so; what it no longer does is decide, on
                    # her behalf, that the person can wait.
                    constraints.append(
                        f"welfare_recovery_drive={welfare.recovery_drive:.3f}"
                        "; owner_request_proceeds_while_depleted"
                    )
                else:
                    constraints.append(f"welfare_recovery_drive={welfare.recovery_drive:.3f}")
                    defers.append("welfare_recovery_required_before_action")

            if welfare.should_verify_before_claiming() and domain_name == "response":
                constraints.append(f"welfare_verify_before_claim: self_report_conf={welfare.self_report_confidence:.3f}")

        # Policy evaluation only quotes cost. Charging before the outcome was
        # known made every defer increase fatigue, which raised recovery drive
        # and generated a self-reinforcing defer storm. UnifiedWill commits the
        # quote exactly once after approval using its signed receipt.
        body_cost_estimate: dict[str, float] = {}
        if consequential:
            try:
                body_cost_estimate = self.body_service.estimate_cost(
                    domain_name,
                    cost_multiplier=priority,
                )
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation(
                    "being_runtime",
                    exc,
                    severity="degraded",
                    action="constrained consequential action after body-cost accounting failure",
                    extra={"domain": domain_name, "priority": priority},
                )
                constraints.append("body_cost_accounting_failed")
                if not repair_lane:
                    defers.append("body_cost_accounting_required_before_action")

        if ignition < 0.12:
            constraints.append(f"aura_now_workspace_low: ignition={ignition:.3f}")
            if deliberative_consequential and not repair_lane:
                defers.append("workspace_not_ignited")
        elif ignition < 0.35:
            constraints.append(f"aura_now_workspace_strained: ignition={ignition:.3f}")

        if agency < 0.28:
            constraints.append(f"aura_now_ownership_low: agency={agency:.3f}")
            if deliberative_consequential and not repair_lane:
                blocks.append("ownership_too_low_for_consequential_action")
        elif agency < 0.50:
            constraints.append(f"aura_now_ownership_mixed: agency={agency:.3f}")

        if controllability < 0.18:
            constraints.append(f"aura_now_controllability_low: controllability={controllability:.3f}")
            if deliberative_consequential and not repair_lane:
                defers.append("action_controllability_too_low")
        elif controllability < 0.35:
            constraints.append(f"aura_now_controllability_strained: controllability={controllability:.3f}")

        if distress > 0.86:
            constraints.append(f"aura_now_distress_high: distress={distress:.3f}")
            if consequential and not repair_lane:
                defers.append("distress_requires_stabilization_first")
        elif distress > 0.62:
            constraints.append(f"aura_now_distress_strained: distress={distress:.3f}")

        if body_pressure > 0.92:
            constraints.append(f"aura_now_body_pressure_high: pressure={body_pressure:.3f}")
            if consequential and not repair_lane:
                defers.append("body_pressure_requires_cooling")
        elif body_pressure > 0.75:
            constraints.append(f"aura_now_body_pressure_strained: pressure={body_pressure:.3f}")

        if free_energy > 0.88 and controllability < 0.42:
            constraints.append(
                f"aura_now_prediction_error_high: free_energy={free_energy:.3f} controllability={controllability:.3f}"
            )
            if deliberative_consequential and not repair_lane:
                defers.append("prediction_error_requires_observation_or_plan")

        if (
            not now.workspace.broadcast_targets
            and deliberative_consequential
            and not repair_lane
        ):
            constraints.append("aura_now_no_workspace_broadcast")
            defers.append("no_workspace_broadcast_for_consequential_action")

        # Felt-state coherence gate: if the kernel and being felt tracks have diverged
        # (Aura's action-regulating state and her reasoning/speech state disagree), that
        # is an internal model mismatch. Acting consequentially on an incoherent self is
        # exactly when to be conservative — defer until the felt-state is reconciled.
        unified = getattr(self, "_last_unified_felt", None)
        if unified is not None and not unified.coherent:
            constraints.append(f"aura_now_felt_incoherent: coherence={unified.coherence:.3f}")
            if deliberative_consequential and not repair_lane:
                defers.append("felt_state_incoherent_resolve_before_action")

        if explicit_foreground_desktop_tool and defers and not blocks:
            desktop_soft_defers = {
                "action_controllability_too_low",
                "workspace_not_ignited",
                "no_workspace_broadcast_for_consequential_action",
                "prediction_error_requires_observation_or_plan",
                # Welfare's graded brakes yield to an EXPLICIT owner action
                # carrying the full desktop execution contract: the owner is
                # the authority, the strain still shapes budgets through the
                # allocation economy, and the note below receipts it. Live
                # evidence (Jul 2026): recovery_drive 0.84 vetoed the owner's
                # RENDER THIS click indefinitely — a broken brake, not
                # protection. Integrity protection and body-cost accounting
                # failures remain hard.
                "welfare_recovery_required_before_action",
            }
            desktop_soft_defer_prefixes = ("welfare_action_inhibition=",)
            hard_defers = [
                item
                for item in defers
                if item not in desktop_soft_defers
                and not item.startswith(desktop_soft_defer_prefixes)
            ]
            if not hard_defers:
                constraints.append("foreground_desktop_action_constrained:not_deferred")
                constraints.extend(f"foreground_desktop_note:{item}" for item in defers[:4])
                defers = []

        # Autonomy latitude (operator-authorized "less fettered" envelope): for a
        # REVERSIBLE, low-blast, non-external, non-self-modifying action there is no reason
        # she must be at a cognitive peak to act — she can simply undo it. Relax only the
        # over-cautious "not in peak state" SOFT defers; every brake that protects HER
        # (distress, body pressure, recovery, felt incoherence, ownership block, body cost)
        # and the strict gate on irreversible/external/self-mod actions stay fully intact.
        if consequential and defers and not blocks:
            try:
                from core.agency.autonomy_latitude import SOFT_DEFERS, get_autonomy_latitude

                latitude = get_autonomy_latitude().classify(domain_name, context=context)
                if latitude.relax_soft_defers:
                    relaxed = [d for d in defers if d in SOFT_DEFERS]
                    hard = [d for d in defers if d not in SOFT_DEFERS]
                    if relaxed:
                        constraints.append(f"autonomy_latitude_widened:{latitude.blast_radius}")
                        constraints.extend(f"latitude_note:{d}" for d in relaxed[:4])
                        defers = hard  # keep the protective defers; drop only the soft ones
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    "being_runtime", exc, severity="debug",
                    action="continued without autonomy-latitude widening this action",
                )

        if blocks:
            outcome = "refuse"
        elif defers:
            if continuity_memory_write or foreground_continuity_state:
                constraints.append(
                    "continuity_memory_write_constrained:not_deferred"
                    if continuity_memory_write
                    else "foreground_state_commit_constrained:not_deferred"
                )
                note_prefix = "continuity_memory_note" if continuity_memory_write else "foreground_state_note"
                constraints.extend(f"{note_prefix}:{item}" for item in defers[:4])
                outcome = "constrain"
            else:
                outcome = "defer"
        elif constraints:
            outcome = "constrain"
        else:
            outcome = "proceed"

        last_body_snapshot = getattr(self, "_last_body_snapshot", None)
        body_fatigue = float(getattr(last_body_snapshot, "fatigue", 0.0) or 0.0)

        policy = {
            "outcome": outcome,
            "constraints": constraints,
            "blocks": blocks,
            "defers": defers,
            "evidence": {
                "state_hash": now.state_hash,
                "tick": now.tick,
                "dominant_drive": now.affect.dominant_drive,
                "workspace_winner": now.workspace.winner,
                "workspace_ignition": round(ignition, 4),
                "agency_confidence": round(agency, 4),
                "controllability": round(controllability, 4),
                "distress": round(distress, 4),
                "body_pressure": round(body_pressure, 4),
                "body_cost_applied": {},
                "body_cost_estimate": {
                    key: round(float(value), 4)
                    for key, value in body_cost_estimate.items()
                },
                "welfare_score": round(welfare.welfare_score, 4) if welfare else 0.5,
                "welfare_distress": round(welfare.distress, 4) if welfare else 0.0,
                "welfare_integrity_guard": round(welfare.integrity_guard, 4) if welfare else 0.5,
                "welfare_truth_protection": round(welfare.truth_protection, 4) if welfare else 0.5,
                "welfare_action_inhibition": round(welfare.action_inhibition, 4) if welfare else 0.0,
                "welfare_recovery_drive": round(welfare.recovery_drive, 4) if welfare else 0.0,
                "welfare_self_report_confidence": round(welfare.self_report_confidence, 4) if welfare else 0.5,
                "body_fatigue": round(body_fatigue, 4),
                "felt_coherence": round(unified.coherence, 4) if unified is not None else 1.0,
            },
        }
        causal_vector = self._refresh_causal_self_vector(now, action_policy=policy)
        if causal_vector is not None:
            workspace = getattr(causal_vector, "causal_valenced_workspace", None)
            policy["evidence"]["causal_vector"] = {
                "organismal_coherence": round(causal_vector.value("organismal_coherence"), 4),
                "experience_candidate_strength": round(
                    causal_vector.value("experience_candidate_strength"),
                    6,
                ),
                "sentience_candidate_strength": round(
                    causal_vector.value("sentience_candidate_strength"),
                    6,
                ),
                "verification_need": round(causal_vector.value("verification_need"), 4),
                "governance_pressure": round(causal_vector.value("governance_pressure"), 4),
            }
            if workspace is not None:
                policy["evidence"]["causal_valenced_workspace"] = workspace.to_dict()
        return policy

    def _refresh_causal_self_vector(
        self,
        now: AuraNow,
        *,
        action_policy: dict[str, Any] | None = None,
    ) -> Any | None:
        """Publish the latest causal self vector from live runtime evidence."""

        try:
            from core.being.causal_self_state import vector_from_aura_now

            causal_vector = vector_from_aura_now(
                now,
                welfare_outputs=getattr(self, "_last_welfare", None),
                blind_report=getattr(self, "_last_blind_report", None),
                action_policy=action_policy,
            )
            self._last_causal_self_vector = causal_vector
            self._last_causal_valenced_workspace = causal_vector.causal_valenced_workspace
            self._refresh_functional_i(now, causal_vector, action_policy=action_policy)
            return causal_vector
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "being_runtime",
                exc,
                severity="warning",
                action="continued without causal valenced workspace vector for this sample",
            )
            return None

    def _refresh_functional_i(
        self,
        now: AuraNow,
        causal_vector: Any,
        *,
        action_policy: dict[str, Any] | None = None,
    ) -> None:
        """Update the functional "I" and the policy it constrains.

        The attractor sets its own bar: it is real only when derived from live
        evidence, injected back into the causal vector, and changing policy
        downstream. Two of those three now happen here — the derivation and
        the feedback — and the third happens where generation controls are
        assembled, in core/brain/cognitive_engine.py, which reads
        :meth:`closed_loop_policy`.

        Runs inside ``_sample_locked``'s reentrant lock, with the same vector
        and action policy the rest of the sample used, so the "I" cannot be
        computed from one request's evidence and applied to another's.
        """

        try:
            self_state = self.self_attractor.update(
                now=now,
                vector=causal_vector,
                action_policy=action_policy,
            )
            self._last_self_attractor_state = self_state

            # The feedback leg. Without it the attractor reads the vector and
            # never touches it, which is a monitor rather than an attractor —
            # continuity pressure, self integrity and trust debt are exactly
            # the dimensions identity tension should move.
            contributions = self.self_attractor.vector_contributions(self_state)
            signals = getattr(causal_vector, "signals", None)
            if isinstance(signals, dict):
                for name, value in contributions.items():
                    signal = signals.get(name)
                    if signal is None:
                        continue
                    # ``replace`` rather than a mutation, and the source is
                    # rewritten with it: a signal whose value came from the
                    # attractor and whose provenance still named its original
                    # sensor would be a false attribution in the one structure
                    # that exists to carry attribution.
                    # cautious_high: these three are continuity pressure, self
                    # integrity and trust debt. An unusable value must not read
                    # as "nothing is wrong" on any of them.
                    adjusted = validated_unit(
                        value, name=f"functional_i.{name}", cautious_high=True
                    )
                    signals[name] = replace(
                        signal,
                        value=float(adjusted),
                        source=f"{signal.source}+functional_i_attractor",
                        note=(
                            f"{signal.note}; adjusted by identity tension "
                            f"{self_state.identity_tension:.3f}"
                        ).strip("; "),
                    )

            self._last_closed_loop_policy = self.policy_coupler.modulate(
                vector=causal_vector,
                self_state=self_state,
                action_policy=action_policy,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            record_degradation(
                "being_runtime",
                exc,
                severity="warning",
                action="continued without a functional-I policy for this sample",
            )

    def self_attractor_state(self) -> Any | None:
        """The latest functional-"I" state, or None before the first sample."""
        return self._last_self_attractor_state

    def closed_loop_policy(self) -> Any | None:
        """The generation and action constraints the current "I" implies.

        Read by the generation-control assembly. Returns None before the first
        sample, and callers must treat that as "no constraint" rather than as
        a permissive one — an absent self-model is not evidence of a calm one.
        """
        return self._last_closed_loop_policy

    def _build_self_state(self, state: Any | None, lesions: set[str]) -> SelfState:
        identity = getattr(state, "identity", None)
        cognition = getattr(state, "cognition", None)
        commitments = []
        for goal in list(getattr(cognition, "active_goals", []) or [])[:4]:
            commitments.append(goal.get("goal") or goal.get("description") if isinstance(goal, dict) else str(goal))
        continuity_hash = ""
        try:
            if state is not None and hasattr(state, "get_continuity_hash"):
                continuity_hash = state.get_continuity_hash()
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("being_runtime", exc)
        if not continuity_hash:
            continuity_hash = self.soul.continuity_hash
        continuity_risk = 0.8 if "functional_soul" in lesions else 0.0
        return SelfState(
            identity_name=str(getattr(identity, "name", "Aura Luna") or "Aura Luna"),
            continuity_hash=continuity_hash,
            identity_stability=float(getattr(identity, "stability", 1.0) or 1.0),
            commitments=tuple(item for item in commitments if item),
            continuity_risk=continuity_risk,
        )

    def _build_memory_context(self, state: Any | None, lesions: set[str]) -> MemoryContext:
        cognition = getattr(state, "cognition", None)
        active = len(getattr(cognition, "long_term_memory", []) or [])
        working = len(getattr(cognition, "working_memory", []) or [])
        soul_policy = self.soul.influence_policy(lesioned="functional_soul" in lesions)
        centrality = 0.0 if "functional_soul" in lesions else soul_policy["memory_centrality_bonus"]
        return MemoryContext(
            active_items=active,
            autobiographical_pressure=max(0.0, min(1.0, active / 8.0)),
            semantic_centrality=round(centrality, 4),
            memory_conflict=max(0.0, min(1.0, working / 128.0)),
        )

    def _build_will_snapshot(self) -> WillStateSnapshot:
        try:
            from core.will import get_will

            will = get_will()
            status = will.get_status() if hasattr(will, "get_status") else {}
            recent = will.get_recent_decisions(1) if hasattr(will, "get_recent_decisions") else []
            return WillStateSnapshot(
                confidence=float(status.get("confidence", 0.7) or 0.7),
                assertiveness=float(status.get("assertiveness", 0.5) or 0.5),
                refusal_pressure=float(status.get("refuse_rate", 0.0) or 0.0),
                last_receipt_id=str(recent[-1].get("receipt_id", "") if recent else ""),
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("being_runtime", exc)
            return WillStateSnapshot()

    def _publish(self, now: AuraNow) -> None:
        try:
            from core.runtime.service_registry import register_runtime_service

            register_runtime_service("aura_now", now, required=False)
            register_runtime_service("being_runtime", self, required=False)
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("being_runtime", exc)
            logger.debug("AuraNow publish skipped: %s", exc)

    def prompt_block(self, state: Any | None = None, *, objective: str = "") -> str:
        """Full prompt block: AuraNow + renderer + semantic stream + blind introspection."""
        now = self.sample(state, objective=objective)
        parts = [now.compact_prompt_block()]
        organismal_block = self.organismal_workspace_prompt_block()
        if organismal_block:
            parts.append(organismal_block)

        # Semantic stream context.
        parts.append(self.semantic_stream.state.to_prompt_block())

        # Blind introspection (structured, non-persona).
        blind = getattr(self, "_last_blind_report", None)
        if blind and blind.confidence > 0:
            parts.append(
                f"## BLIND INTROSPECTION (non-persona)\n"
                f"- State: {blind.predicted_state_class} (confidence={blind.confidence:.2f})\n"
                f"- Behavior: {', '.join(blind.expected_behavior_shifts)}\n"
                f"- Welfare estimate: {blind.welfare_estimate:.2f}\n"
                f"- Urgency: {blind.urgency:.2f}\n\n"
            )

        # Welfare summary.
        welfare = getattr(self, "_last_welfare", None)
        if welfare:
            parts.append(
                f"## WELFARE STATE\n"
                f"- Score: {welfare.welfare_score:.2f}, Distress: {welfare.distress:.2f}\n"
                f"- Integrity guard: {welfare.integrity_guard:.2f}, "
                f"Truth protection: {welfare.truth_protection:.2f}\n"
                f"- Recovery drive: {welfare.recovery_drive:.2f}, "
                f"Action inhibition: {welfare.action_inhibition:.2f}\n"
                f"- Self-report confidence: {welfare.self_report_confidence:.2f}\n\n"
            )

        # State-grounded introspection.
        parts.append(self.renderer.render_prompt_block(now))

        return "".join(parts)

    def organismal_workspace_prompt_block(self, *, compact: bool = False) -> str:
        """Return the latest Causal Valenced Workspace block for prompts."""
        cvw = getattr(self, "_last_causal_valenced_workspace", None)
        if cvw is None or not hasattr(cvw, "prompt_block"):
            return ""
        return cvw.prompt_block(compact=compact)


_RUNTIME: BeingRuntime | None = None


def get_being_runtime() -> BeingRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = BeingRuntime()
    return _RUNTIME


def reset_being_runtime_for_test() -> None:
    global _RUNTIME
    if _RUNTIME is not None:
        _RUNTIME.stop()
    _RUNTIME = None
