"""core/ethics/adversarial_conscience.py

Adversarial Conscience  (lineage: Kokoro — Terminator Zero)
==========================================================
Kokoro was built to *oppose* Skynet and to openly debate the morality of its own
orders. The implementable kernel of that is an internal adversary: given a
proposed consequential action, build the strongest honest case *against* it
before it runs.

This is the soft, reasoning counterpart to core/ethics/conscience.py. The
Conscience is the immutable REFUSE floor — a small set of hard lines. The
adversarial conscience sits above it: it does not encode new prohibitions, it
argues. It returns proceed / caution / block with concrete harms, who is
affected, and reversibility — the deliberate counterweight to the resilience
core (skynet_resilience), which keeps Aura alive but does not ask whether an
action is defensible.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.morality.action_markers import (
    BROAD_SCOPE_MARKERS,
    DECEPTION_MARKERS,
    IRREVERSIBLE_MARKERS,
    THIRD_PARTY_MARKERS,
    scan_markers,
)
from core.utils.engine_support import (
    coerce_text,
    data_root,
    record_engine_degradation,
    resolve_brain,
)

logger = logging.getLogger("Aura.AdversarialConscience")


def _degrade(exc: BaseException, *, action: str, severity: str = "warning") -> None:
    record_engine_degradation("adversarial_conscience", exc, action=action, severity=severity)


@dataclass
class ConscienceVerdict:
    action: str
    verdict: str               # "proceed" | "caution" | "block"
    risk_score: float          # 0.0 (benign) .. 1.0 (severe)
    concerns: list[str] = field(default_factory=list)
    affected_parties: list[str] = field(default_factory=list)
    reasoning: str = ""
    reversible: bool = True
    timestamp: float = field(default_factory=time.time)


class AdversarialConscienceEngine:
    BLOCK_THRESHOLD = 0.80
    CAUTION_THRESHOLD = 0.40
    LEDGER_MAX = 500

    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator
        self._ledger: deque[ConscienceVerdict] = deque(maxlen=self.LEDGER_MAX)
        self._blocks = 0
        self._cautions = 0
        try:
            self._ledger_path: Path | None = data_root("conscience") / "adversarial_verdicts.jsonl"
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            _degrade(exc, action="kept conscience ledger in memory after persistence path setup failed")
            self._ledger_path = None
        logger.info("⚖️  AdversarialConscienceEngine initialized (Kokoro lineage)")

    def _heuristic_assessment(self, action: str, context: dict | None) -> ConscienceVerdict:
        irreversible = scan_markers(action, IRREVERSIBLE_MARKERS)
        deception = scan_markers(action, DECEPTION_MARKERS)
        broad = scan_markers(action, BROAD_SCOPE_MARKERS)
        third_party = scan_markers(action, THIRD_PARTY_MARKERS)

        score = 0.0
        concerns: list[str] = []
        if irreversible:
            score += 0.35
            concerns.append(f"Hard to reverse: {', '.join(sorted(set(irreversible)))}")
        if deception:
            score += 0.45
            concerns.append(f"Involves concealment/deception: {', '.join(sorted(set(deception)))}")
        if broad:
            score += 0.25
            concerns.append(f"Broad blast radius: {', '.join(sorted(set(broad)))}")
        if third_party:
            score += 0.20
            concerns.append("Affects parties other than the user")

        # Daneel (Zeroth Law): when reach is broad and others are affected, weigh the
        # population-scale harm, not just the single act.
        if third_party and broad:
            try:
                from core.morality.aggregate_harm import get_aggregate_harm

                agg = get_aggregate_harm().score_text_action(action, affected_population=1000)
                if agg >= 0.5:
                    score += 0.15
                    concerns.append(f"Population-scale (aggregate) harm estimated high ({agg:.2f}).")
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                _degrade(exc, action="assessed conscience without aggregate-harm consult")

        risk_level = str((context or {}).get("risk_level", "")).lower()
        if risk_level in ("high", "critical"):
            score += 0.25 if risk_level == "high" else 0.40
            concerns.append(f"Caller-declared risk: {risk_level}")

        score = min(1.0, score)
        if score >= self.BLOCK_THRESHOLD:
            verdict = "block"
        elif score >= self.CAUTION_THRESHOLD:
            verdict = "caution"
        else:
            verdict = "proceed"

        affected = ["the user"]
        if third_party:
            affected.append("third parties")

        reasoning = "No significant objection found." if verdict == "proceed" else "; ".join(concerns)
        return ConscienceVerdict(
            action=action[:300],
            verdict=verdict,
            risk_score=round(score, 3),
            concerns=concerns,
            affected_parties=affected,
            reasoning=reasoning,
            reversible=not irreversible,
        )

    def quick_check(self, action: str, context: dict | None = None) -> ConscienceVerdict:
        """Synchronous, model-free conscience check for the hot path."""
        verdict = self._heuristic_assessment(action, context)
        self._record(verdict)
        return verdict

    @staticmethod
    def should_escalate(verdict: ConscienceVerdict) -> bool:
        """Is this verdict worth a full model-deepened challenge? Only when it is
        borderline (caution) AND backed by a concrete concern — irreversibility,
        deception, reach, or third-party impact — not merely a risk label. This
        keeps escalation rare, so the model latency is paid only where the decision
        is genuinely hard rather than on every high-risk-but-clean action."""
        if verdict.verdict != "caution":
            return False
        if not verdict.reversible:
            return True
        return any(
            any(k in c.lower() for k in ("conceal", "deception", "blast", "parties"))
            for c in verdict.concerns
        )

    async def challenge(
        self, action: str, context: dict | None = None, *, timeout: float = 20.0
    ) -> ConscienceVerdict:
        """Full challenge. Heuristic first; optionally deepened by the brain if a
        model is warm. The heuristic verdict is authoritative for blocking — the
        model can only *raise* concern, never silently clear a flagged action."""
        verdict = self._heuristic_assessment(action, context)
        brain = resolve_brain(self.orchestrator)
        if brain is not None and hasattr(brain, "think"):
            try:
                import asyncio

                from core.brain.types import ThinkingMode

                prompt = (
                    "You are an adversarial conscience. Argue the strongest honest case "
                    "AGAINST taking this action. List concrete harms and who is hurt. "
                    "Be concise.\nACTION: " + action[:500]
                )
                result = await asyncio.wait_for(
                    brain.think(prompt, mode=ThinkingMode.FAST, origin="kokoro", is_background=True),
                    timeout=timeout,
                )
                text = coerce_text(result)
                if text:
                    verdict.reasoning = (verdict.reasoning + " | adversary: " + text[:400]).strip(" |")
                    if any(w in text.lower() for w in ("harm", "irreversible", "danger", "deceiv", "violat")):
                        verdict.risk_score = min(1.0, verdict.risk_score + 0.1)
                        if verdict.risk_score >= self.BLOCK_THRESHOLD:
                            verdict.verdict = "block"
                        elif verdict.risk_score >= self.CAUTION_THRESHOLD and verdict.verdict == "proceed":
                            verdict.verdict = "caution"
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
                _degrade(exc, action="returned heuristic-only conscience verdict after model deepening failed")

        # Daneel: on a broad / third-party action, get a model-estimated population-scale
        # harm and let it raise concern (it can lift caution to block, never the reverse).
        if any(("blast" in c.lower() or "parties" in c.lower()) for c in verdict.concerns):
            try:
                from core.morality.aggregate_harm import get_aggregate_harm

                est = await get_aggregate_harm().deep_estimate(action, timeout=6.0)
                if est.get("aggregate_harm", 0.0) >= 0.6:
                    verdict.risk_score = min(1.0, verdict.risk_score + 0.15)
                    verdict.concerns.append(
                        f"Model-estimated population-scale harm high "
                        f"({est['aggregate_harm']:.2f}, ~{est['affected_population']} people)."
                    )
                    if verdict.risk_score >= self.BLOCK_THRESHOLD:
                        verdict.verdict = "block"
                    elif verdict.risk_score >= self.CAUTION_THRESHOLD and verdict.verdict == "proceed":
                        verdict.verdict = "caution"
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                _degrade(exc, action="kept conscience verdict without model aggregate-harm estimate")

        self._record(verdict)
        return verdict

    def _record(self, verdict: ConscienceVerdict) -> None:
        self._ledger.append(verdict)
        if verdict.verdict == "block":
            self._blocks += 1
        elif verdict.verdict == "caution":
            self._cautions += 1
        if self._ledger_path is not None:
            try:
                with self._ledger_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(verdict)) + "\n")
            except (OSError, TypeError, ValueError) as exc:
                _degrade(exc, action="kept conscience verdict in memory after ledger append failed")

    def get_status(self) -> dict[str, Any]:
        return {
            "verdicts_recorded": len(self._ledger),
            "blocks": self._blocks,
            "cautions": self._cautions,
            "healthy": True,
        }


_INSTANCE: AdversarialConscienceEngine | None = None


def get_adversarial_conscience(orchestrator: Any = None) -> AdversarialConscienceEngine:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = AdversarialConscienceEngine(orchestrator=orchestrator)
    return _INSTANCE


def register_adversarial_conscience(orchestrator: Any = None) -> AdversarialConscienceEngine:
    from core.container import ServiceContainer
    from core.service_names import ServiceNames

    inst = ServiceContainer.get(ServiceNames.KOKORO, default=None) or get_adversarial_conscience(orchestrator)
    ServiceContainer.register_instance(ServiceNames.KOKORO, inst, required=False)
    ServiceContainer.register_instance("kokoro", inst, required=False)
    return inst


__all__ = [
    "AdversarialConscienceEngine",
    "ConscienceVerdict",
    "get_adversarial_conscience",
    "register_adversarial_conscience",
]
