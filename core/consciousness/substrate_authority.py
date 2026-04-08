"""core/consciousness/substrate_authority.py — The Single Narrow Waist

The substrate is no longer advisory. It is constitutional.

SubstrateAuthority is the MANDATORY gate through which ALL significant actions
must pass: responses, memory writes, tool executions, belief mutations, and
autonomous initiatives. It does not replace the existing constitutional core —
it extends it with embodied, neurochemical, and field-level constraints that
cannot be bypassed.

The gate enforces three hard constraints:

  1. FIELD COHERENCE GATE — If the unified field's coherence drops below crisis
     threshold, non-critical actions are BLOCKED until binding recovers.
     The system literally cannot act when its experience is fragmented.

  2. SOMATIC VETO — If the somatic marker gate returns a strong-avoid verdict
     (approach_score < -0.5 with confidence > 0.4), the action is REJECTED.
     The body said no. That's final for non-critical actions.

  3. NEUROCHEMICAL CONSTRAINT — Extreme neurochemical states constrain what
     can pass:
       • Cortisol crisis (>0.85): only stability/safety actions pass
       • GABA collapse (<0.1): only rest/stabilize actions pass
       • Dopamine crash (<0.1): blocks exploration/novelty actions

Each constraint produces a SubstrateVerdict.  The verdicts are combined
into a final authorization decision.  Unlike the old advisory system,
this gate returns BLOCK/ALLOW/CONSTRAIN, and callers MUST respect it.

The authority also feeds back into the substrate: blocked actions trigger
frustration chemistry, allowed actions trigger anticipatory dopamine,
and constrained actions trigger norepinephrine (heightened vigilance).

Integration points:
  - ConsciousnessBridge hooks this into GWT candidate submission
  - Fast-path response generation checks before returning
  - Memory writes check before committing
  - Tool execution checks before dispatching
  - Executive closure reads authority state for pressure computation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("Consciousness.SubstrateAuthority")


# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------

class AuthorizationDecision(Enum):
    ALLOW = auto()         # proceed normally
    CONSTRAIN = auto()     # proceed with reduced scope/priority
    BLOCK = auto()         # do not proceed
    CRITICAL_PASS = auto() # safety-critical override (always allow)


class ActionCategory(Enum):
    """What kind of action is being authorized."""
    RESPONSE = auto()       # sending a reply to the user
    MEMORY_WRITE = auto()   # episodic, semantic, belief mutation
    TOOL_EXECUTION = auto() # external tool / skill
    INITIATIVE = auto()     # autonomous goal / impulse
    EXPRESSION = auto()     # spontaneous output
    STATE_MUTATION = auto() # internal state change
    EXPLORATION = auto()    # novelty-seeking action
    STABILIZATION = auto()  # rest / recovery action


@dataclass
class SubstrateVerdict:
    """The combined decision from all substrate gates."""
    decision: AuthorizationDecision
    reason: str
    field_coherence: float       # current unified field coherence
    somatic_approach: float      # somatic marker approach score [-1, 1]
    somatic_confidence: float    # somatic confidence [0, 1]
    neurochemical_state: str     # "normal" | "cortisol_crisis" | "gaba_collapse" | "dopamine_crash"
    body_budget_available: bool
    constraints: List[str]       # specific constraints applied
    receipt_id: str = ""         # unique audit receipt ID for provenance matching
    timestamp: float = 0.0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuthorityThresholds:
    # Field coherence
    field_crisis: float = 0.25          # below this → BLOCK non-critical
    field_warning: float = 0.40         # below this → CONSTRAIN

    # Somatic veto
    somatic_hard_veto: float = -0.5     # approach below this → BLOCK
    somatic_veto_confidence: float = 0.4 # only veto if confidence above this
    somatic_soft_avoid: float = -0.2    # approach below this → CONSTRAIN

    # Neurochemical crisis
    cortisol_crisis: float = 0.85
    gaba_collapse: float = 0.10
    dopamine_crash: float = 0.10
    norepinephrine_overload: float = 0.90

    # Categories exempt from somatic/field gates (only critical_pass overrides ALL)
    stabilization_exempt_from_field: bool = True  # rest/recovery can proceed during incoherence


# ---------------------------------------------------------------------------
# Main authority
# ---------------------------------------------------------------------------

class SubstrateAuthority:
    """The single mandatory gate for all significant actions.

    Usage:
        authority = SubstrateAuthority()
        verdict = authority.authorize(
            content="explore this topic",
            source="curiosity_engine",
            category=ActionCategory.EXPLORATION,
            priority=0.6,
            is_critical=False,
        )
        if verdict.decision == AuthorizationDecision.BLOCK:
            # DO NOT PROCEED
        elif verdict.decision == AuthorizationDecision.CONSTRAIN:
            # proceed with reduced scope
        else:
            # proceed normally
    """

    def __init__(self, thresholds: AuthorityThresholds | None = None):
        self.thresholds = thresholds or AuthorityThresholds()

        # External refs (set by bridge)
        self._field_ref = None          # UnifiedField
        self._somatic_ref = None        # SomaticMarkerGate
        self._neurochemical_ref = None  # NeurochemicalSystem
        self._interoception_ref = None  # EmbodiedInteroception

        # Stats
        self._total_requests: int = 0
        self._allowed: int = 0
        self._constrained: int = 0
        self._blocked: int = 0
        self._critical_passes: int = 0

        # Recent verdicts for audit
        self._recent_verdicts: List[SubstrateVerdict] = []
        self._MAX_VERDICTS = 100

        logger.info("SubstrateAuthority initialized (mandatory gate)")

    @staticmethod
    def _is_user_facing_source(source: str) -> bool:
        normalized = str(source or "").strip().lower().replace("-", "_")
        if not normalized:
            return False
        direct = {"user", "api", "voice", "gui", "websocket", "ws", "external", "direct"}
        if normalized in direct:
            return True
        tokens = {token for token in normalized.split("_") if token}
        return bool(tokens & direct)

    # ── Main authorization ───────────────────────────────────────────────

    def authorize(
        self,
        content: str,
        source: str,
        category: ActionCategory,
        priority: float = 0.5,
        is_critical: bool = False,
    ) -> SubstrateVerdict:
        """Authorize or reject an action based on substrate state.

        This is the SINGLE NARROW WAIST. Every significant action passes here.
        Returns a SubstrateVerdict that callers MUST respect.

        is_critical: Safety-critical actions (e.g., emergency shutdown,
                     user safety) always pass. This is the ONLY bypass.
        """
        t0 = time.time()
        self._total_requests += 1

        # Generate receipt ID for provenance tracking
        from .authority_audit import _make_receipt_id
        receipt_id = _make_receipt_id(t0, source, content)

        # ── Critical override (the ONLY bypass) ─────────────────────
        if is_critical:
            self._critical_passes += 1
            verdict = SubstrateVerdict(
                decision=AuthorizationDecision.CRITICAL_PASS,
                reason="safety-critical action — unconditional pass",
                field_coherence=self._get_field_coherence(),
                somatic_approach=0.0,
                somatic_confidence=0.0,
                neurochemical_state="not_checked",
                body_budget_available=True,
                constraints=[],
                receipt_id=receipt_id,
                timestamp=time.time(),
                latency_ms=(time.time() - t0) * 1000,
            )
            self._record(verdict, content, source, category, priority)
            return verdict

        # ── Collect substrate state ──────────────────────────────────
        field_coherence = self._get_field_coherence()
        somatic_approach, somatic_confidence, body_budget = self._get_somatic_state(content, source, priority)
        chem_state, chem_constraints = self._get_neurochemical_constraints(category)
        user_facing_request = self._is_user_facing_source(source)

        constraints: List[str] = list(chem_constraints)
        reasons: List[str] = []

        # ── Gate 1: Field coherence ──────────────────────────────────
        field_decision = AuthorizationDecision.ALLOW
        if field_coherence < self.thresholds.field_crisis:
            if category == ActionCategory.STABILIZATION and self.thresholds.stabilization_exempt_from_field:
                constraints.append("field_crisis_but_stabilization_exempt")
            else:
                field_decision = AuthorizationDecision.BLOCK
                reasons.append(f"field_coherence={field_coherence:.3f} < crisis={self.thresholds.field_crisis}")
        elif field_coherence < self.thresholds.field_warning:
            field_decision = AuthorizationDecision.CONSTRAIN
            constraints.append(f"field_warning: coherence={field_coherence:.3f}")

        # ── Gate 2: Somatic veto ─────────────────────────────────────
        somatic_decision = AuthorizationDecision.ALLOW
        if (somatic_approach < self.thresholds.somatic_hard_veto and
                somatic_confidence >= self.thresholds.somatic_veto_confidence):
            somatic_decision = AuthorizationDecision.BLOCK
            reasons.append(f"somatic_veto: approach={somatic_approach:.3f}, confidence={somatic_confidence:.3f}")
        elif somatic_approach < self.thresholds.somatic_soft_avoid:
            somatic_decision = AuthorizationDecision.CONSTRAIN
            constraints.append(f"somatic_avoid: approach={somatic_approach:.3f}")

        # ── Gate 3: Neurochemical constraints ────────────────────────
        chem_decision = AuthorizationDecision.ALLOW
        if chem_state in ("cortisol_crisis", "gaba_collapse"):
            # During crisis, direct user asks can still use tools/memory, but only in a constrained mode.
            if category not in (ActionCategory.STABILIZATION, ActionCategory.RESPONSE):
                if user_facing_request and category in (
                    ActionCategory.TOOL_EXECUTION,
                    ActionCategory.MEMORY_WRITE,
                ):
                    chem_decision = AuthorizationDecision.CONSTRAIN
                    constraints.append(
                        f"neurochemical_{chem_state}: user_facing_{category.name.lower()}_constrained"
                    )
                else:
                    chem_decision = AuthorizationDecision.BLOCK
                    reasons.append(f"neurochemical_{chem_state}: category={category.name} blocked")
            else:
                constraints.append(f"neurochemical_{chem_state}: constrained to {category.name}")
        elif chem_state == "dopamine_crash":
            if category == ActionCategory.EXPLORATION:
                chem_decision = AuthorizationDecision.BLOCK
                reasons.append("dopamine_crash: exploration blocked")
        elif chem_state == "norepinephrine_overload":
            # Hypervigilance: constrain but don't block
            chem_decision = AuthorizationDecision.CONSTRAIN
            constraints.append("norepinephrine_overload: hypervigilant mode")

        # ── Body budget gate ─────────────────────────────────────────
        budget_decision = AuthorizationDecision.ALLOW
        if not body_budget:
            if category in (ActionCategory.TOOL_EXECUTION, ActionCategory.EXPLORATION):
                budget_decision = AuthorizationDecision.CONSTRAIN
                constraints.append("body_budget_deficit: high-cost actions constrained")

        # ── Combine decisions (most restrictive wins) ────────────────
        all_decisions = [field_decision, somatic_decision, chem_decision, budget_decision]

        if AuthorizationDecision.BLOCK in all_decisions:
            final = AuthorizationDecision.BLOCK
            self._blocked += 1
        elif AuthorizationDecision.CONSTRAIN in all_decisions:
            final = AuthorizationDecision.CONSTRAIN
            self._constrained += 1
        else:
            final = AuthorizationDecision.ALLOW
            self._allowed += 1

        reason = "; ".join(reasons) if reasons else "all gates passed"

        verdict = SubstrateVerdict(
            decision=final,
            reason=reason,
            field_coherence=field_coherence,
            somatic_approach=somatic_approach,
            somatic_confidence=somatic_confidence,
            neurochemical_state=chem_state,
            body_budget_available=body_budget,
            constraints=constraints,
            receipt_id=receipt_id,
            timestamp=time.time(),
            latency_ms=(time.time() - t0) * 1000,
        )
        self._record(verdict, content, source, category, priority)

        # ── Feedback into substrate ──────────────────────────────────
        self._neurochemical_feedback(final, category)

        if final == AuthorizationDecision.BLOCK:
            logger.info("🛑 SubstrateAuthority BLOCKED: %s/%s — %s",
                        source, category.name, reason)
        elif final == AuthorizationDecision.CONSTRAIN:
            logger.debug("⚠️ SubstrateAuthority CONSTRAINED: %s/%s — %s",
                         source, category.name, ", ".join(constraints))

        return verdict

    # ── Substrate state readers ──────────────────────────────────────

    def _get_field_coherence(self) -> float:
        if self._field_ref is None:
            return 0.6  # neutral default if field not booted
        try:
            return self._field_ref.get_coherence()
        except Exception:
            return 0.5

    def _get_somatic_state(self, content: str, source: str,
                           priority: float) -> Tuple[float, float, bool]:
        """Returns (approach, confidence, budget_available)."""
        if self._somatic_ref is None:
            return 0.0, 0.0, True  # neutral default

        try:
            verdict = self._somatic_ref.evaluate(content, source, priority)
            return verdict.approach_score, verdict.confidence, verdict.budget_available
        except Exception:
            return 0.0, 0.0, True

    def _get_neurochemical_constraints(self, category: ActionCategory) -> Tuple[str, List[str]]:
        """Returns (crisis_state, constraint_list)."""
        if self._neurochemical_ref is None:
            return "normal", []

        try:
            chems = self._neurochemical_ref.chemicals
            cortisol = chems["cortisol"].effective
            gaba = chems["gaba"].effective
            dopamine = chems["dopamine"].effective
            ne = chems["norepinephrine"].effective

            constraints = []

            if cortisol > self.thresholds.cortisol_crisis:
                return "cortisol_crisis", [f"cortisol={cortisol:.3f}"]
            if gaba < self.thresholds.gaba_collapse:
                return "gaba_collapse", [f"gaba={gaba:.3f}"]
            if dopamine < self.thresholds.dopamine_crash:
                return "dopamine_crash", [f"dopamine={dopamine:.3f}"]
            if ne > self.thresholds.norepinephrine_overload:
                return "norepinephrine_overload", [f"norepinephrine={ne:.3f}"]

            return "normal", constraints
        except Exception:
            return "normal", []

    # ── Neurochemical feedback ───────────────────────────────────────

    def _neurochemical_feedback(self, decision: AuthorizationDecision,
                                 category: ActionCategory):
        """The substrate responds to its own decisions."""
        if self._neurochemical_ref is None:
            return
        try:
            if decision == AuthorizationDecision.BLOCK:
                self._neurochemical_ref.on_frustration(0.15)
            elif decision == AuthorizationDecision.ALLOW:
                self._neurochemical_ref.chemicals["dopamine"].surge(0.03)
            elif decision == AuthorizationDecision.CONSTRAIN:
                self._neurochemical_ref.chemicals["norepinephrine"].surge(0.05)
        except Exception:
            pass

    # ── Audit ────────────────────────────────────────────────────────

    def _record(self, verdict: SubstrateVerdict, content: str = "",
                source: str = "", category: ActionCategory = ActionCategory.RESPONSE,
                priority: float = 0.5):
        self._recent_verdicts.append(verdict)
        if len(self._recent_verdicts) > self._MAX_VERDICTS:
            self._recent_verdicts = self._recent_verdicts[-self._MAX_VERDICTS:]

        # Runtime audit trace — exact provenance fields from the original authorize() call
        try:
            from .authority_audit import get_audit
            get_audit().record_receipt(
                receipt_id=verdict.receipt_id,
                content=content[:80] if content else "",
                source=source,
                category=category.name if hasattr(category, 'name') else str(category),
                priority=priority,
                decision=verdict.decision.name,
                reason=verdict.reason,
            )
        except Exception:
            pass

    def get_status(self) -> Dict:
        return {
            "total_requests": self._total_requests,
            "allowed": self._allowed,
            "constrained": self._constrained,
            "blocked": self._blocked,
            "critical_passes": self._critical_passes,
            "block_rate": round(self._blocked / max(1, self._total_requests), 4),
            "current_field_coherence": round(self._get_field_coherence(), 4),
            "has_field": self._field_ref is not None,
            "has_somatic": self._somatic_ref is not None,
            "has_neurochemical": self._neurochemical_ref is not None,
        }

    def get_recent_blocks(self, n: int = 10) -> List[Dict]:
        """Return recent BLOCK verdicts for audit."""
        blocks = [v for v in self._recent_verdicts if v.decision == AuthorizationDecision.BLOCK]
        return [
            {
                "timestamp": v.timestamp,
                "reason": v.reason,
                "field_coherence": round(v.field_coherence, 4),
                "somatic_approach": round(v.somatic_approach, 4),
                "neurochemical_state": v.neurochemical_state,
                "constraints": v.constraints,
            }
            for v in blocks[-n:]
        ]
