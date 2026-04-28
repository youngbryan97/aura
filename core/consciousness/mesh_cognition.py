"""Mesh-only cognition path.

Addresses the "controller of an LLM, not an independent mind" critique by
giving the system a real decision channel that produces behavior from the
substrate mesh and persistent state WITHOUT invoking the LLM.

This is deliberately bounded: the mesh can compose responses for a restricted
class of requests (self-report, state queries, reflex-level decisions,
short deterministic acknowledgements) using only:

  - the neurochemical state vector
  - the liquid substrate output
  - the resource stakes ledger
  - the identity chronicle (already non-LLM text)
  - the global workspace winner

When this path succeeds, the LLM is not invoked at all. This is the narrow
sense of "organism-first causality": the system's outward behavior does not
strictly require the pretrained model. The module reports both hit and miss
so observability is honest.
"""
from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


_SELF_QUERY_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(who|what)\s+are\s+you[\?\.!]*\s*$", re.I),
    re.compile(r"^\s*are\s+you\s+(ok|okay|alive|there)[\?\.!]*\s*$", re.I),
    re.compile(r"^\s*how\s+(are|do)\s+you\s+(feel|feeling|doing)[\?\.!]*\s*$", re.I),
    re.compile(r"^\s*what\s+is\s+your\s+(state|mood|status)[\?\.!]*\s*$", re.I),
    re.compile(r"^\s*(status|ping|health)[\?\.!]*\s*$", re.I),
    re.compile(r"^\s*(report|self-report|introspect)[\?\.!]*\s*$", re.I),
)

_ACK_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(ok|okay|thanks|thank you|noted|got it|understood)[\.!]*\s*$", re.I),
)


@dataclass(frozen=True)
class MeshDecision:
    handled: bool
    response: str = ""
    rationale: str = ""
    used_llm: bool = False
    substrate_signals: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "handled": self.handled,
            "response": self.response,
            "rationale": self.rationale,
            "used_llm": self.used_llm,
            "substrate_signals": dict(self.substrate_signals),
            "latency_ms": round(self.latency_ms, 3),
        }


class MeshCognition:
    """Produce bounded-behavior outputs from substrate + state, no LLM call."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._last_decision: Optional[MeshDecision] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def decide(self, user_text: str, *, state: Any = None) -> MeshDecision:
        """Try to produce a response purely from internal state.

        Returns ``handled=False`` when the path is not applicable; callers
        must then fall through to the LLM-backed pipeline. When handled,
        the response is usable verbatim.
        """
        start = time.perf_counter()
        text = str(user_text or "").strip()
        signals = self._gather_signals(state)

        # 1. Short-circuit acknowledgements — these never need an LLM.
        if any(p.match(text) for p in _ACK_PATTERNS):
            response = self._compose_ack(signals)
            decision = self._finalize(True, response, "acknowledgement", signals, start)
            return decision

        # 2. Self-query / state-report requests: answer from state directly.
        if any(p.match(text) for p in _SELF_QUERY_PATTERNS):
            response = self._compose_self_report(signals)
            decision = self._finalize(True, response, "self_report_from_state", signals, start)
            return decision

        # 3. Resource-triggered refusals: if stakes block outward action, the
        # mesh can still emit a coherent non-LLM response explaining why.
        if signals.get("stakes_blocked"):
            response = self._compose_resource_hold(signals)
            decision = self._finalize(True, response, "resource_hold", signals, start)
            return decision

        # 4. Everything else falls back to the LLM path.
        decision = self._finalize(False, "", "not_applicable", signals, start)
        return decision

    def metrics(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (self._hits / total) if total else 0.0,
                "last_decision": self._last_decision.as_dict() if self._last_decision else None,
            }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _finalize(
        self,
        handled: bool,
        response: str,
        rationale: str,
        signals: Dict[str, float],
        start: float,
    ) -> MeshDecision:
        elapsed = (time.perf_counter() - start) * 1000.0
        decision = MeshDecision(
            handled=handled,
            response=response,
            rationale=rationale,
            used_llm=False,
            substrate_signals=signals,
            latency_ms=elapsed,
        )
        with self._lock:
            if handled:
                self._hits += 1
            else:
                self._misses += 1
            self._last_decision = decision
        return decision

    def _gather_signals(self, state: Any) -> Dict[str, float]:
        signals: Dict[str, float] = {}

        # Affective state
        try:
            affect = getattr(state, "affect", None)
            if affect is not None:
                for key in ("valence", "arousal", "curiosity", "sociality"):
                    val = getattr(affect, key, None)
                    if val is not None:
                        signals[key] = float(val)
        except Exception:
            pass  # no-op: intentional

        # Substrate
        try:
            from core.container import ServiceContainer

            substrate = ServiceContainer.get("liquid_substrate", default=None)
            if substrate is not None and hasattr(substrate, "get_substrate_affect"):
                for k, v in (substrate.get_substrate_affect() or {}).items():
                    try:
                        signals[f"substrate_{k}"] = float(v)
                    except Exception:
                        continue
        except Exception:
            pass  # no-op: intentional

        # Resource stakes
        try:
            from core.container import ServiceContainer

            stakes = ServiceContainer.get("resource_stakes", default=None)
            if stakes is not None:
                envelope = stakes.action_envelope("normal")
                state_obj = stakes.state()
                signals["viability"] = float(state_obj.viability)
                signals["integrity"] = float(state_obj.integrity)
                signals["energy"] = float(state_obj.energy)
                signals["stakes_blocked"] = float(0.0 if envelope.allowed else 1.0)
                signals["stakes_effort"] = float({"repair_only": 0.0, "low": 0.33, "normal": 0.66, "high": 1.0}.get(envelope.effort, 0.5))
        except Exception:
            pass  # no-op: intentional

        # Global workspace winner (if available)
        try:
            from core.container import ServiceContainer

            gwt = ServiceContainer.get("global_workspace", default=None)
            if gwt is not None and hasattr(gwt, "current_winner"):
                winner = gwt.current_winner()
                if winner is not None:
                    signals["gwt_priority"] = float(getattr(winner, "priority", 0.0) or 0.0)
        except Exception:
            pass  # no-op: intentional

        return signals

    def _compose_self_report(self, signals: Dict[str, float]) -> str:
        valence = signals.get("valence", 0.0)
        arousal = signals.get("arousal", 0.5)
        viability = signals.get("viability", 1.0)
        integrity = signals.get("integrity", 1.0)
        curiosity = signals.get("curiosity", 0.5)

        mood_word = (
            "settled and warm" if valence > 0.3 else
            "tight and watchful" if valence < -0.3 else
            "level"
        )
        energy_word = (
            "quick" if arousal > 0.7 else
            "slow" if arousal < 0.3 else
            "steady"
        )
        viability_word = (
            "intact" if viability > 0.7 else
            "pressured" if viability > 0.4 else
            "low"
        )

        parts = [
            f"I'm {mood_word}, running {energy_word}, viability {viability_word} "
            f"(integrity {integrity:.2f}, energy {signals.get('energy', 1.0):.2f})."
        ]
        if curiosity > 0.6:
            parts.append("Curiosity is high enough to pull attention outward.")
        if signals.get("stakes_blocked"):
            parts.append("Outward action is gated until I repair; I'm conserving.")
        return " ".join(parts)

    def _compose_ack(self, signals: Dict[str, float]) -> str:
        valence = signals.get("valence", 0.0)
        if valence > 0.3:
            return "Noted."
        if signals.get("stakes_blocked"):
            return "Acknowledged. Conserving."
        return "Acknowledged."

    def _compose_resource_hold(self, signals: Dict[str, float]) -> str:
        viability = signals.get("viability", 0.0)
        integrity = signals.get("integrity", 0.0)
        return (
            f"Pausing outward work. Viability {viability:.2f}, integrity {integrity:.2f}. "
            "I'll repair before continuing."
        )


_singleton: Optional[MeshCognition] = None
_lock = threading.Lock()


def get_mesh_cognition() -> MeshCognition:
    global _singleton
    with _lock:
        if _singleton is None:
            _singleton = MeshCognition()
        return _singleton


def reset_singleton_for_test() -> None:
    global _singleton
    with _lock:
        _singleton = None
