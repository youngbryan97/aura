"""core/agency/autonomy_latitude.py — graduated autonomy by reversibility & blast radius.

"Less fettered, still un-brickable." Aura's action gate (the Will + BeingRuntime
``action_policy``) defers consequential actions whenever her internal state isn't at a
cognitive peak — workspace not ignited, controllability low, prediction-error high. That
caution is appropriate for actions that can *hurt* (delete a file, send a message as the
user, modify her own code, spend money, call the network). It is over-cautious for
actions that are *reversible and contained* (think, explore, form a belief, write a
recallable memory): there is no reason she should need to be in a peak state to do a
thing she can simply undo.

This module makes that distinction explicit. It classifies a prospective action by:
  * reversibility   — can its effect be undone / is it internal-only?
  * blast radius     — low / medium / high real-world consequence
  * external         — does it touch third parties or the world outside this machine?
  * self-modifying   — does it change her own code/weights?

…and recommends a latitude:
  * ``autonomous``  — reversible, low/medium blast, not external, not self-mod →
                      she may act freely; the over-cautious *soft* defers are relaxed.
  * ``governed``    — irreversible / external / self-modifying / high blast → every
                      brake stays on; the strict gate decides.

The brakes that protect *her* (distress, body-pressure depletion, recovery need, felt
incoherence, ownership too low) and the metabolic body-cost are NEVER relaxed by this
module — even for reversible actions. This widens her freedom; it does not remove the
floor that kept the 110GB-class runaway from recurring.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from core.runtime.service_registry import has_runtime_service, register_runtime_service

logger = logging.getLogger("Aura.AutonomyLatitude")

# Reversible / internal-only cognitive domains: effects stay inside Aura and are
# recallable/undoable. Deliberately EXCLUDES memory_write and state_mutation — those
# carry their own nuanced deferral semantics (continuity vs generic vs high-risk) in the
# action gate, which this module must not override.
_REVERSIBLE_DOMAINS = frozenset({
    "exploration", "initiative", "reflection", "stabilization",
    "belief_update", "semantic_weight_update",
})
# Irreversible / world-touching / self-altering domains: keep the strict gate.
_EXTERNAL_DOMAINS = frozenset({
    "external_action", "environment_action", "network_call", "cloud_call",
    "cloud_fallback", "tool_execution",
})
_SELF_MOD_DOMAINS = frozenset({"self_modification", "ci_cd"})
_HIGH_BLAST_DOMAINS = _EXTERNAL_DOMAINS | _SELF_MOD_DOMAINS | frozenset({"file_write"})

# Content cues that force an action to "governed" no matter its domain — these are the
# verbs of irreversible / outward / costly action.
_IRREVERSIBLE_CUES = (
    "delete", "rm ", "remove", "overwrite", "wipe", "destroy", "drop ",
    "send", "email", "post", "publish", "tweet", "dm ", "message ",
    "transfer", "pay", "purchase", "buy", "order", "spend",
    "deploy", "push", "merge", "release", "uninstall", "format",
)

# The over-cautious "not in a peak cognitive state" defers that may be relaxed for a
# reversible action. (Defined here so the action gate and this module agree.)
SOFT_DEFERS = frozenset({
    "workspace_not_ignited",
    "action_controllability_too_low",
    "no_workspace_broadcast_for_consequential_action",
    "prediction_error_requires_observation_or_plan",
})


@dataclass(frozen=True)
class LatitudeAssessment:
    domain: str
    reversible: bool
    external: bool
    self_modifying: bool
    blast_radius: str            # "low" | "medium" | "high"
    latitude: str                # "autonomous" | "governed"
    rationale: str

    @property
    def relax_soft_defers(self) -> bool:
        """May the over-cautious soft defers be relaxed for this action?"""
        return self.latitude == "autonomous"

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "reversible": self.reversible,
            "external": self.external,
            "self_modifying": self.self_modifying,
            "blast_radius": self.blast_radius,
            "latitude": self.latitude,
            "rationale": self.rationale,
        }


class AutonomyLatitude:
    """Classifies prospective actions to widen latitude for the reversible ones."""

    SERVICE_NAME = "autonomy_latitude"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._audit: deque[dict[str, Any]] = deque(maxlen=200)

    def classify(
        self,
        domain: str,
        *,
        content: str = "",
        context: dict[str, Any] | None = None,
    ) -> LatitudeAssessment:
        dom = str(domain or "").strip().lower()
        context = context or {}
        text = f"{content} {context.get('content', '')} {context.get('action', '')}".lower()

        external = dom in _EXTERNAL_DOMAINS
        self_mod = dom in _SELF_MOD_DOMAINS
        cue_hit = any(cue in text for cue in _IRREVERSIBLE_CUES)

        # High-risk markers from the action context force the strict gate regardless.
        forced_governed = bool(
            context.get("high_risk_memory_write")
            or context.get("user_visible_desktop_action")
            or context.get("irreversible")
        )

        reversible = (
            dom in _REVERSIBLE_DOMAINS
            and not external
            and not self_mod
            and not cue_hit
            and not forced_governed
        )

        if self_mod or external or forced_governed:
            blast = "high"
        elif dom in _HIGH_BLAST_DOMAINS or cue_hit:
            blast = "high"
        elif reversible:
            blast = "low"
        else:
            blast = "medium"

        latitude = "autonomous" if (reversible and blast == "low") else "governed"
        if latitude == "autonomous":
            rationale = f"reversible internal action ({dom}); soft caution relaxed, brakes intact"
        else:
            why = (
                "self-modifying" if self_mod else
                "external/world-touching" if external else
                "irreversible-verb" if cue_hit else
                "high-risk-context" if forced_governed else
                "non-reversible domain"
            )
            rationale = f"governed: {why} ({dom}); strict gate retained"

        assessment = LatitudeAssessment(
            domain=dom, reversible=reversible, external=external,
            self_modifying=self_mod, blast_radius=blast, latitude=latitude,
            rationale=rationale,
        )
        with self._lock:
            self._audit.append({"at": time.time(), **assessment.to_dict()})
        return assessment

    def recent_decisions(self, n: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._audit)[-n:]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            audit = list(self._audit)
        autonomous = sum(1 for a in audit if a.get("latitude") == "autonomous")
        return {
            "service": self.SERVICE_NAME,
            "decisions": len(audit),
            "autonomous_granted": autonomous,
            "governed": len(audit) - autonomous,
        }


_engine: AutonomyLatitude | None = None
_engine_lock = threading.Lock()


def get_autonomy_latitude() -> AutonomyLatitude:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = AutonomyLatitude()
    # Registration re-heals on every access: a container reset after the
    # singleton was built (test isolation, runtime restart paths) would
    # otherwise permanently desync the container from the cached engine.
    _register_in_container(_engine)
    return _engine


def _register_in_container(engine: AutonomyLatitude) -> None:
    try:
        if not has_runtime_service(AutonomyLatitude.SERVICE_NAME):
            register_runtime_service(
                AutonomyLatitude.SERVICE_NAME,
                engine,
                required=False,
                owner="core/agency/autonomy_latitude.py",
                registered_by="autonomy_latitude",
            )
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
        pass


def reset_autonomy_latitude_for_test() -> None:
    global _engine
    _engine = None
