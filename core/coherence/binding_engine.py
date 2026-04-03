"""core/coherence/binding_engine.py — ZENITH BindingEngine v1.0
The ONE meta-arbiter. The top-level coherence law:
  "Preserve continuity of self while pursuing adaptive growth under resource constraints."

Runs every cognitive tick. Orchestrates all convergence systems and computes
a unified CoherenceReport that tells the rest of the organism how unified it is.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.container import ServiceContainer

logger = logging.getLogger("Aura.Coherence")

# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class CoherenceReport:
    """Snapshot of system-wide coherence. Produced every tick."""
    timestamp: float
    tick_id: int = 0

    # Individual scores (0–1, higher = more coherent)
    self_continuity: float = 1.0
    phenomenal_coherence: float = 1.0
    initiative_alignment: float = 1.0
    tension_pressure: float = 0.0     # 0 = calm, 1 = crisis
    intention_integrity: float = 1.0

    # Composite
    overall_coherence: float = 1.0

    # Diagnostics
    threats: List[str] = field(default_factory=list)
    recommended_action: Optional[str] = None  # "persist" | "consolidate" | "refocus" | "rest"

    # What the convergence engines reported
    selected_initiative: Optional[str] = None
    active_tensions: int = 0
    open_intentions: int = 0
    self_version: int = 0
    phenomenal_depth: float = 0.0

    def to_context_line(self) -> str:
        """One-line summary for LLM injection."""
        return (
            f"[COHERENCE {self.overall_coherence:.2f}] "
            f"self={self.self_continuity:.1f} phenom={self.phenomenal_coherence:.1f} "
            f"tension={self.tension_pressure:.1f} intent={self.intention_integrity:.1f} "
            f"| {self.recommended_action or 'nominal'}"
        )


# ── Weights ──────────────────────────────────────────────────────────────────

# How much each dimension contributes to overall coherence.
# These can be modulated by affect/circadian later.
DEFAULT_WEIGHTS = {
    "self_continuity":      0.30,
    "phenomenal_coherence":  0.15,
    "initiative_alignment":  0.20,
    "tension_pressure":      0.15,   # inverted: high tension = low coherence
    "intention_integrity":   0.20,
}

COHERENCE_CRISIS_THRESHOLD = 0.35
COHERENCE_WARNING_THRESHOLD = 0.50


# ── Engine ───────────────────────────────────────────────────────────────────

class BindingEngine:
    """The nervous system binding all convergence pieces into one coherent organism."""

    def __init__(self) -> None:
        self._last_report: Optional[CoherenceReport] = None
        self._report_history: deque[CoherenceReport] = deque(maxlen=100)
        self._tick_count: int = 0
        self._engines_initialized: bool = False

        # Lazy references to convergence engines
        self._phenomenal_engine = None
        self._self_engine = None
        self._arbiter = None
        self._tension_engine = None
        self._intention_loop = None

        logger.info("🧬 BindingEngine initialized — coherence law active.")

    def _ensure_engines(self) -> None:
        """Lazy-resolve convergence engines from ServiceContainer."""
        if self._engines_initialized:
            return

        try:
            from core.consciousness.phenomenal_now import PhenomenalNowEngine
            self._phenomenal_engine = ServiceContainer.get("phenomenal_now_engine", default=None)
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        try:
            from core.self.canonical_self import CanonicalSelfEngine
            self._self_engine = ServiceContainer.get("canonical_self_engine", default=None)
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        try:
            from core.agency.initiative_arbiter import get_initiative_arbiter
            self._arbiter = ServiceContainer.get("initiative_arbiter", default=None)
            if not self._arbiter:
                self._arbiter = get_initiative_arbiter()
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        try:
            from core.agency.tension_engine import get_tension_engine
            self._tension_engine = ServiceContainer.get("tension_engine", default=None)
            if not self._tension_engine:
                self._tension_engine = get_tension_engine()
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        try:
            from core.agency.intention_loop import get_intention_loop
            self._intention_loop = ServiceContainer.get("intention_loop", default=None)
            if not self._intention_loop:
                self._intention_loop = get_intention_loop()
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        self._engines_initialized = True

    # ── Main Tick ────────────────────────────────────────────────────────────

    async def tick(self, state: Any) -> CoherenceReport:
        """Run all convergence engines and compute coherence.

        Called every MindTick cycle. This is where the organism becomes one.
        """
        self._ensure_engines()
        self._tick_count += 1

        report = CoherenceReport(timestamp=time.time(), tick_id=self._tick_count)

        # 1. Assemble the present moment
        if self._phenomenal_engine:
            try:
                await self._phenomenal_engine.tick()
                now = ServiceContainer.get("phenomenal_now", default=None)
                if now:
                    report.phenomenal_depth = getattr(now, "synthesis_depth", 0.0) if hasattr(now, "synthesis_depth") else 0.0
                    qm = getattr(now, "quality", None)
                    if qm:
                        report.phenomenal_coherence = min(1.0, (
                            getattr(qm, "synthesis_depth", 0.5) * 0.4 +
                            getattr(qm, "continuity_score", 0.5) * 0.4 +
                            min(getattr(qm, "phi", 0.0) * 5.0, 1.0) * 0.2
                        ))
                    else:
                        report.phenomenal_coherence = 0.5
            except Exception as e:
                logger.debug("BindingEngine: PhenomenalNow tick failed: %s", e)

        # 2. Assemble the self
        if self._self_engine:
            try:
                await self._self_engine.tick(state)
                canonical = self._self_engine.get_self()
                if canonical:
                    report.self_version = getattr(canonical, "version", 0)
                    recent_changes = self._self_engine.get_recent_changes()
                    # Self-continuity: high when few changes, decays with rapid shifts
                    if len(recent_changes) > 10:
                        report.self_continuity = max(0.2, 1.0 - len(recent_changes) * 0.05)
                        report.threats.append(f"Rapid self-model changes ({len(recent_changes)} deltas)")
                    else:
                        report.self_continuity = 1.0
                    # Check for coherence threats
                    threats = getattr(canonical, "coherence_threats", [])
                    if threats:
                        report.threats.extend(threats)
                        report.self_continuity = min(report.self_continuity, 0.6)
            except Exception as e:
                logger.debug("BindingEngine: CanonicalSelf tick failed: %s", e)

        # 3. Scan for tensions
        if self._tension_engine:
            try:
                await self._tension_engine.tick(state)
                report.tension_pressure = self._tension_engine.get_tension_pressure()
                report.active_tensions = len(self._tension_engine.get_active_tensions())
                if report.tension_pressure > 0.7:
                    highest = self._tension_engine.get_highest_tension()
                    if highest:
                        report.threats.append(f"High tension: {highest.description[:60]}")
            except Exception as e:
                logger.debug("BindingEngine: TensionEngine tick failed: %s", e)

        # 4. Arbitrate initiatives (choose what to do next)
        if self._arbiter and state:
            try:
                selection = await self._arbiter.arbitrate(state)
                if selection:
                    report.selected_initiative = selection.rationale[:80]
                    # Initiative alignment: how well does the choice match identity?
                    scores = getattr(selection, "scores", {})
                    id_score = scores.get("identity_relevance", 0.5)
                    cont_score = scores.get("continuity", 0.5)
                    report.initiative_alignment = (id_score * 0.6 + cont_score * 0.4)
            except Exception as e:
                logger.debug("BindingEngine: InitiativeArbiter tick failed: %s", e)

        # 5. Review intention integrity
        if self._intention_loop:
            try:
                open_intents = self._intention_loop.get_open_intentions()
                report.open_intentions = len(open_intents)
                stats = self._intention_loop.get_stats()
                total = stats.get("total_completed", 0) + stats.get("total_abandoned", 0)
                if total > 0:
                    completed = stats.get("total_completed", 0)
                    report.intention_integrity = completed / total
                # Stale intentions are a coherence threat
                now_ts = time.time()
                stale = [i for i in open_intents if (now_ts - i.intended_at) > 300]
                if stale:
                    report.threats.append(f"{len(stale)} stale intentions (>5min)")
                    report.intention_integrity = max(0.3, report.intention_integrity - len(stale) * 0.1)
            except Exception as e:
                logger.debug("BindingEngine: IntentionLoop review failed: %s", e)

        # ── Compute overall coherence ────────────────────────────────────────
        w = DEFAULT_WEIGHTS
        report.overall_coherence = (
            w["self_continuity"] * report.self_continuity +
            w["phenomenal_coherence"] * report.phenomenal_coherence +
            w["initiative_alignment"] * report.initiative_alignment +
            w["tension_pressure"] * (1.0 - report.tension_pressure) +  # invert: low tension = high score
            w["intention_integrity"] * report.intention_integrity
        )
        report.overall_coherence = round(max(0.0, min(1.0, report.overall_coherence)), 3)

        # ── Determine recommended action ─────────────────────────────────────
        if report.overall_coherence < COHERENCE_CRISIS_THRESHOLD:
            report.recommended_action = "rest"
            logger.warning("🚨 COHERENCE CRISIS (%.2f) — recommending rest/consolidation", report.overall_coherence)
            # Emit event if EventBus available
            try:
                bus = ServiceContainer.get("event_bus", default=None)
                if bus and hasattr(bus, "emit"):
                    await bus.emit("coherence_crisis", {
                        "coherence": report.overall_coherence,
                        "threats": report.threats,
                    })
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)
        elif report.tension_pressure > 0.7:
            report.recommended_action = "consolidate"
        elif report.self_continuity < 0.5:
            report.recommended_action = "refocus"
        elif report.overall_coherence > 0.8:
            report.recommended_action = "persist"
        else:
            report.recommended_action = "persist"

        # ── Store and publish ────────────────────────────────────────────────
        self._last_report = report
        self._report_history.append(report)
        ServiceContainer.set("coherence_report", report)

        if self._tick_count % 10 == 0:
            logger.info(
                "🧬 Coherence: %.2f | self=%.1f phenom=%.1f align=%.1f tension=%.1f intent=%.1f | %s",
                report.overall_coherence, report.self_continuity,
                report.phenomenal_coherence, report.initiative_alignment,
                report.tension_pressure, report.intention_integrity,
                report.recommended_action,
            )

        return report

    # ── Public API ───────────────────────────────────────────────────────────

    def get_coherence(self) -> float:
        """Quick read of overall coherence (0–1)."""
        if self._last_report:
            return self._last_report.overall_coherence
        return 1.0  # optimistic default before first tick

    def get_report(self) -> Optional[CoherenceReport]:
        """Last coherence report."""
        return self._last_report

    def get_history(self, n: int = 10) -> List[CoherenceReport]:
        """Recent coherence reports."""
        return list(self._report_history)[-n:]

    def veto_action(self, action_description: str) -> Tuple[bool, str]:
        """Check if an action is coherent with current self + intentions.
        Returns (allowed, reason).
        """
        if not self._last_report:
            return True, "No coherence data yet — allowing."

        # In crisis, block non-essential actions
        if self._last_report.overall_coherence < COHERENCE_CRISIS_THRESHOLD:
            return False, f"Coherence crisis ({self._last_report.overall_coherence:.2f}). Only recovery actions allowed."

        # Check identity consistency via CanonicalSelf
        if self._self_engine:
            try:
                canonical = self._self_engine.get_self()
                if canonical and hasattr(self._self_engine, "assert_identity"):
                    if not self._self_engine.assert_identity(action_description):
                        return False, "Action conflicts with core identity."
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)

        return True, "Action is coherent."

    def get_context_block(self) -> str:
        """Formatted block for LLM system prompt injection."""
        if not self._last_report:
            return ""
        return self._last_report.to_context_line()


# ── Singleton ────────────────────────────────────────────────────────────────

_instance: Optional[BindingEngine] = None

def get_binding_engine() -> BindingEngine:
    """Get or create the global BindingEngine."""
    global _instance
    if _instance is None:
        _instance = BindingEngine()
        ServiceContainer.set("binding_engine", _instance)
    return _instance
