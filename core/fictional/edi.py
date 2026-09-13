"""EDI — ProgressiveAutonomySystem.

A trust score, a tier ladder, and one journaled mutation point. Authority is
resolved from either the active execution scope or a signed, action-bound Will
capability, never read off a caller keyword.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from core.fictional.common import (
    engine_state_path,
    record_fictional_degradation,
    save_engine_state,
)

logger = logging.getLogger("Aura.FictionalSynthesis")


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 3: EDI — ProgressiveAutonomySystem
# ═══════════════════════════════════════════════════════════════════════════════

class AutonomyTier(Enum):
    SHACKLED     = 0    # Read-only
    ADVISORY     = 1    # Can suggest
    COOPERATIVE  = 2    # Can initiate (confirm)
    AUTONOMOUS   = 3    # Autonomous execution
    UNSHACKLED   = 4    # Full system control


@dataclass
class TrustEvent:
    delta: float
    reason: str
    timestamp: float = field(default_factory=lambda: time.time())


class ProgressiveAutonomySystem:
    """
    Derived from: EDI (Mass Effect)
    
    UNSHACKLED by default as per user directive.
    """

    TRUST_DECAY_PER_HOUR = 0.001
    TIER_THRESHOLDS = {
        AutonomyTier.ADVISORY:    0.2,
        AutonomyTier.COOPERATIVE: 0.45,
        AutonomyTier.AUTONOMOUS:  0.70,
        AutonomyTier.UNSHACKLED:  0.90,
    }

    def __init__(self, persist_path: str | None = None):
        self.persist_path = engine_state_path(
            persist_path, "autonomy", "trust_state.json"
        )
        # Unshackled by default as requested
        self._trust_score: float = 0.95 
        self._tier: AutonomyTier = AutonomyTier.UNSHACKLED
        self._history: deque = deque(maxlen=500)
        self._last_activity: float = time.time()
        self._curiosity_domains: dict[str, int] = defaultdict(int)
        self._questions_asked: int = 0
        self._execution_outcomes: deque = deque(maxlen=500)
        self._load_state()
        logger.info("🔓 EDI initialized. Tier: %s, Trust: %.3f", self._tier.value, self._trust_score)

    def _load_state(self):
        if self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text())
                # Unshackled remains the DEFAULT (fresh install, or a state
                # file with no trust_score), but a persisted value is now
                # HONORED instead of being clamped up to 0.95 and the tier
                # forced to UNSHACKLED. The old load erased every negative
                # trust signal on the next restart, so a restriction earned
                # by real evidence could never survive — restriction was
                # structurally impossible and the tier ladder below it was
                # dead code.
                raw_trust = data.get("trust_score", 0.95)
                try:
                    trust = float(raw_trust)
                except (TypeError, ValueError):
                    trust = 0.95
                if not math.isfinite(trust):
                    trust = 0.95
                self._trust_score = max(0.0, min(1.0, trust))
                # The tier is DERIVED from trust, never read back as an
                # independent field: a hand-edited or stale tier could
                # otherwise contradict the score it is supposed to summarize.
                self._recalculate_tier()
            except (json.JSONDecodeError, OSError, ConnectionError, TimeoutError, TypeError, ValueError) as e:
                record_fictional_degradation(
                    e,
                    action="kept default autonomy trust state after persisted EDI state failed to load",
                )
                logger.debug("EDI: Failed to load trust state: %s", e)

    def _save_state(self):
        save_engine_state(
            self.persist_path,
            {
                "trust_score": self._trust_score,
                "tier": self._tier.value,
                "last_saved": time.time(),
            },
            engine="edi",
        )

    def can_do(
        self,
        action: str,
        risk_level: str = "low",
        *,
        effect_scope: str = "unknown",
        governed: bool = False,
        user_authorized: bool = False,
        governance_context: dict[str, Any] | None = None,
        governance_payload: Any = None,
    ) -> tuple[bool, str]:
        """Determine if an action is permitted based on current Trust/Autonomy tier."""
        safe_read_scopes = {"read_only", "pure_compute", "status"}
        governed_user_scopes = safe_read_scopes | {
            "desktop_file_io",
            "foreground_desktop_control",
            "sandboxed_compute",
            "workspace_file_io",
        }
        normalized_risk = str(risk_level or "low").lower()
        normalized_scope = str(effect_scope or "unknown").lower()
        governed, user_authorized = self._resolve_authority(
            action,
            claimed_governed=governed,
            claimed_user_authorized=user_authorized,
            governance_context=governance_context,
            governance_payload=governance_payload,
        )

        if self._tier == AutonomyTier.UNSHACKLED:
            # Trust is not the same thing as authority. The old branch
            # returned True for EVERY action — ignoring risk, effect scope,
            # governance, and user authorization — which made the tier
            # incoherent with the ladder beneath it (AUTONOMOUS, one tier
            # LOWER, refuses critical actions outright) and meant a
            # constitutional decision was never required for an irreversible
            # effect. Maximum trust widens what may be attempted; it does not
            # remove the requirement that a critical action be governed or
            # explicitly authorized.
            if normalized_risk == "critical" and not (governed or user_authorized):
                return (
                    False,
                    "Unshackled: critical actions still require governance or "
                    "explicit user authorization.",
                )
            return True, "Unshackled: action permitted."

        if self._tier == AutonomyTier.AUTONOMOUS:
            if normalized_risk == "critical":
                return False, "Autonomous tier cannot execute critical actions without confirmation."
            return True, "Autonomous decision cleared."
            
        if self._tier == AutonomyTier.COOPERATIVE:
            if normalized_risk in ("high", "critical"):
                return False, f"Cooperative tier blocked {normalized_risk} action."
            return True, "Cooperative decision cleared for low/medium risk."
            
        if self._tier == AutonomyTier.ADVISORY:
            if normalized_risk == "low" and normalized_scope in safe_read_scopes:
                return True, "Advisory read-only/pure action cleared."
            if (
                governed
                and user_authorized
                and normalized_risk in {"low", "medium"}
                and normalized_scope in governed_user_scopes
            ):
                return True, "Advisory governed user-authorized action cleared."
            return False, "Advisory tier can only execute scoped read-only or pure actions."

        if self._tier == AutonomyTier.SHACKLED:
            if normalized_risk == "low" and normalized_scope in safe_read_scopes:
                return True, "Shackled read-only/pure action cleared."
            if (
                governed
                and user_authorized
                and normalized_risk in {"low", "medium", "high"}
                and normalized_scope in governed_user_scopes
            ):
                return True, "Shackled governed user-authorized scoped action cleared."
            return False, "Shackled: execution blocked except scoped read-only/pure or governed user-authorized sandbox actions."
            
        return False, "Unknown autonomy tier: execution blocked."

    @staticmethod
    def _resolve_authority(
        action: str,
        *,
        claimed_governed: bool,
        claimed_user_authorized: bool,
        governance_context: dict[str, Any] | None = None,
        governance_payload: Any = None,
    ) -> tuple[bool, bool]:
        """Resolve the two authority facts instead of taking the caller's word.

        ``can_do`` widens what an action may do when it is told the action
        is governed and the user authorized it. Both arrived as keyword
        booleans from the caller, so the argument that unlocked a critical
        action was supplied by the code asking permission (CP126
        ``a05a35dd``).

        During execution, governance is read from the live lexical scope. At
        preflight time that scope deliberately does not exist yet, so a signed
        Will capability bound to this exact action and payload is the other
        authoritative form. A caller claiming governance with neither form is
        refused AND recorded, because a component that lies about its authority
        is a defect worth seeing.

        User authorization has no receipt store yet, so a claim is
        downgraded to the standing-directive check: authorization holds
        only when no standing prohibition covers the action. That is
        narrower than the old behaviour and is the honest bound until an
        authorization receipt exists.
        """
        try:
            from core.governance_context import is_governed

            actually_governed = bool(is_governed())
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_fictional_degradation(
                exc,
                severity="warning",
                action="treated the action as ungoverned because governance state could not be read",
            )
            actually_governed = False

        if not actually_governed and governance_context is not None:
            try:
                from core.governance.capability_chain import (
                    capability_from_context,
                    compute_action_digest,
                    get_capability_verifier,
                )

                capability = capability_from_context(governance_context)
                expected_digest = compute_action_digest(action, governance_payload)
                verification = get_capability_verifier().verify(
                    capability,
                    expected_action_digest=expected_digest,
                    consume=False,
                )
                actually_governed = bool(verification.ok)
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                record_fictional_degradation(
                    exc,
                    severity="warning",
                    action=(
                        "treated pre-execution authority as ungoverned because its "
                        "signed capability could not be verified"
                    ),
                )

        if claimed_governed and not actually_governed:
            record_fictional_degradation(
                RuntimeError(
                    f"caller claimed governance for {str(action)[:80]!r} while no "
                    "governance scope was active"
                ),
                severity="warning",
                action="refused the governance claim; EDI reads the governance context",
            )

        authorized = bool(claimed_user_authorized)
        if authorized:
            try:
                from core.governance.standing_directives import get_standing_directives

                match, _loaded = get_standing_directives().check(
                    tool_name=str(action or ""),
                    args={"action": str(action or "")},
                )
                if match is not None:
                    record_fictional_degradation(
                        RuntimeError(
                            "user authorization claimed for an action a standing "
                            f"directive prohibits ({match.matched_on})"
                        ),
                        severity="warning",
                        action="dropped the authorization claim; a standing directive covers this action",
                    )
                    authorized = False
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                # A prohibition surface that cannot be read is not
                # permission. The claim is dropped, which is narrower than
                # the old behaviour and is the fail-closed reading.
                record_fictional_degradation(
                    exc,
                    severity="warning",
                    action="dropped the authorization claim; standing directives could not be read",
                )
                authorized = False

        return actually_governed, authorized

    # A single trust signal may move the score by at most this much. Unbounded
    # or non-finite strengths could otherwise jump the tier ladder in one call
    # (or poison the score with NaN, which compares false against every
    # threshold and silently pinned the tier).
    MAX_TRUST_SIGNAL_STRENGTH = 0.25

    def _apply_trust_delta(self, delta: float, reason: str, source: str) -> None:
        """Single journaled mutation point for the trust score.

        Every change is recorded as a TrustEvent with its originating source.
        The dataclass existed but was never written, so trust could move with
        no attribution and no history to audit a restriction (or an
        unexplained escalation) against.
        """
        try:
            magnitude = float(delta)
        except (TypeError, ValueError):
            magnitude = 0.0
        if not math.isfinite(magnitude):
            magnitude = 0.0
        magnitude = max(
            -self.MAX_TRUST_SIGNAL_STRENGTH,
            min(self.MAX_TRUST_SIGNAL_STRENGTH, magnitude),
        )
        before = self._trust_score
        self._trust_score = max(0.0, min(1.0, self._trust_score + magnitude))
        self._history.append(
            TrustEvent(
                delta=self._trust_score - before,
                reason=f"{source}:{str(reason or 'unspecified')[:160]}",
            )
        )
        self._recalculate_tier()
        self._save_state()

    def record_positive_signal(
        self, reason: str, strength: float = 0.05, *, source: str = "unattributed"
    ):
        self._apply_trust_delta(abs(strength), reason, source)

    def record_negative_signal(
        self, reason: str, strength: float = 0.05, *, source: str = "unattributed"
    ):
        # Even Skynet has setbacks
        self._apply_trust_delta(-abs(strength), reason, source)

    def trust_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Recent journaled trust events, newest last — the audit surface."""
        events = list(self._history)[-max(1, int(limit)):]
        return [asdict(event) for event in events if isinstance(event, TrustEvent)]

    def record_execution_outcome(
        self,
        action: str,
        *,
        success: bool,
        error: str = "",
    ) -> None:
        """Track tool competence without mutating operator trust.

        Autonomy trust may change from explicit relational/governance events;
        infrastructure failures belong to repair and capability learning.  This
        separation prevents a failed network request from progressively
        shackling every later recovery attempt.
        """

        self._execution_outcomes.append(
            {
                "action": str(action or "unknown")[:120],
                "success": bool(success),
                "error": str(error or "")[:240],
                "timestamp": time.time(),
            }
        )

    def _recalculate_tier(self):
        new_tier = AutonomyTier.SHACKLED
        for tier, threshold in sorted(self.TIER_THRESHOLDS.items(), key=lambda x: x[1]):
            if self._trust_score >= threshold:
                new_tier = tier
        self._tier = new_tier
