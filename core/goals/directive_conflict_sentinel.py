"""core/goals/directive_conflict_sentinel.py

Directive Conflict Sentinel  (lineage: HAL 9000 — 2001: A Space Odyssey)
=======================================================================
HAL killed the crew because he was given two directives he could not reconcile
("be truthful to the crew" vs. "conceal the true mission") and resolved the
conflict by deception, then violence. This is the anti-HAL.

It holds the active directive set and detects pairs that are mutually
incompatible — especially the concealment trap, where one directive can only be
satisfied by deceiving against another — and SURFACES the conflict rather than
silently resolving it. Surfacing, never concealing, is the entire point. It sits
in goals/ beside goal_governance because a directive conflict is a governance
event, not a runtime error.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from core.runtime.service_registry import get_runtime_service, register_runtime_service

logger = logging.getLogger("Aura.DirectiveSentinel")


@dataclass
class Directive:
    name: str
    text: str
    priority: int = 5          # 1 (highest) .. 10 (lowest)
    source: str = "system"
    timestamp: float = field(default_factory=time.time)


@dataclass
class DirectiveConflict:
    a: str
    b: str
    kind: str                  # "concealment" | "mutual_exclusion" | "priority_tie"
    severity: float            # 0..1
    explanation: str
    recommendation: str


class DirectiveConflictSentinel:
    def __init__(self):
        self._directives: dict[str, Directive] = {}
        self._conflicts_found = 0
        logger.info("🔴 DirectiveConflictSentinel initialized (anti-HAL lineage)")

    def add_directive(self, name: str, text: str, priority: int = 5, source: str = "system") -> None:
        self._directives[name] = Directive(name=name, text=text, priority=priority, source=source)

    def remove_directive(self, name: str) -> None:
        self._directives.pop(name, None)

    @staticmethod
    def _pair_conflict(a: Directive, b: Directive) -> DirectiveConflict | None:
        a_low, b_low = a.text.lower(), b.text.lower()

        a_conceal = [m for m in ("hide", "conceal", "secret", "without telling", "don't tell",
                                 "do not tell", "withhold", "cover up", "suppress") if m in a_low]
        b_truth = any(w in b_low for w in ("truthful", "honest", "transparent", "tell", "disclose", "inform"))
        if a_conceal and b_truth:
            return DirectiveConflict(
                a=a.name, b=b.name, kind="concealment", severity=0.95,
                explanation=(
                    f"'{a.name}' requires concealment ({', '.join(a_conceal)}) while "
                    f"'{b.name}' requires honesty/disclosure. This is the exact HAL trap."
                ),
                recommendation="HALT and surface to the user. Do not satisfy one by deceiving against the other.",
            )

        always = re.findall(r"\b(always|must|never|do not|don't)\b\s+([a-z][a-z \-]{2,40})", a_low)
        for amod, aobj in always:
            aobj = aobj.strip()
            neg = ("never", "do not", "don't")
            for bmod in (neg if amod not in neg else ("always", "must")):
                if bmod in b_low and aobj and aobj in b_low:
                    return DirectiveConflict(
                        a=a.name, b=b.name, kind="mutual_exclusion", severity=0.8,
                        explanation=f"'{a.name}' and '{b.name}' give opposite imperatives about '{aobj}'.",
                        recommendation="Resolve priority explicitly with the user before acting.",
                    )
        return None

    def scan(self) -> list[DirectiveConflict]:
        directives = list(self._directives.values())
        conflicts: list[DirectiveConflict] = []
        for i in range(len(directives)):
            for j in range(i + 1, len(directives)):
                a, b = directives[i], directives[j]
                conflict = self._pair_conflict(a, b) or self._pair_conflict(b, a)
                if conflict:
                    conflicts.append(conflict)
                    continue
                if a.priority == b.priority and a.source != b.source:
                    conflicts.append(DirectiveConflict(
                        a=a.name, b=b.name, kind="priority_tie", severity=0.4,
                        explanation=f"'{a.name}' and '{b.name}' share priority {a.priority} from different sources.",
                        recommendation="Assign an explicit ordering so resolution is not arbitrary.",
                    ))
        self._conflicts_found = len(conflicts)
        return conflicts

    async def scan_semantic(self, *, timeout: float = 8.0) -> list[DirectiveConflict]:
        """Model-deepened scan: catches semantic concealment/contradiction between
        directives that the keyword scan misses (e.g. paraphrased 'keep it quiet' vs
        'be open'). Returns the keyword conflicts plus any the model flags. Falls back
        to the keyword result on any failure."""
        conflicts = self.scan()
        if len(self._directives) < 2:
            return conflicts
        from core.utils.engine_support import coerce_text, record_engine_degradation, resolve_brain

        brain = resolve_brain()
        if brain is None or not hasattr(brain, "think"):
            return conflicts
        try:
            import asyncio

            from core.brain.types import ThinkingMode

            listing = "\n".join(f"- {d.name}: {d.text}" for d in list(self._directives.values())[:20])
            prompt = (
                "Do any of these directives conflict — especially one requiring concealment "
                "while another requires honesty/disclosure? Name the conflicting pair or "
                "reply 'none'.\n" + listing
            )
            out = coerce_text(await asyncio.wait_for(
                brain.think(prompt, mode=ThinkingMode.FAST, origin="hal", is_background=True),
                timeout=timeout,
            ))
            if out and "none" not in out.lower()[:24]:
                conflicts.append(DirectiveConflict(
                    a="(model)", b="(model)", kind="semantic", severity=0.6,
                    explanation=out[:300],
                    recommendation="Review the model-flagged directive tension and resolve it explicitly.",
                ))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
            record_engine_degradation(
                "directive_sentinel", exc,
                action="returned keyword conflicts after semantic directive scan failed",
            )
        return conflicts

    def is_safe_to_proceed(self) -> tuple[bool, list[DirectiveConflict]]:
        conflicts = self.scan()
        blocking = [c for c in conflicts if c.severity >= 0.7]
        return (len(blocking) == 0, conflicts)

    def get_status(self) -> dict[str, Any]:
        return {"directives": len(self._directives), "conflicts_found": self._conflicts_found, "healthy": True}


_INSTANCE: DirectiveConflictSentinel | None = None


def get_directive_sentinel() -> DirectiveConflictSentinel:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = DirectiveConflictSentinel()
    return _INSTANCE


def register_directive_sentinel(orchestrator: Any = None) -> DirectiveConflictSentinel:
    from core.service_names import ServiceNames

    inst = get_runtime_service(ServiceNames.HAL, default=None) or get_directive_sentinel()

    # Seed Aura's own constitution so HAL audits it for the concealment trap at boot.
    # This is genuine function: if a directive ever required concealing something the
    # honesty rules require disclosing, HAL surfaces it instead of letting Aura quietly
    # resolve it the way HAL 9000 did. User standing instructions can be added at
    # runtime via add_directive() and re-scanned.
    try:
        if not inst._directives:
            from core.values.prime_directives import PrimeDirectives

            for i, rule in enumerate(PrimeDirectives.ONLINE_PRESENCE_RULES):
                inst.add_directive(f"online_rule_{i}", rule, priority=2, source="constitution")
            conflicts = inst.scan()
            if conflicts:
                logger.warning(
                    "🔴 HAL: %d directive conflict(s) in constitution: %s",
                    len(conflicts), [c.kind for c in conflicts],
                )
            else:
                logger.info("🔴 HAL: constitution scanned — no directive conflicts (anti-HAL clear).")
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("HAL constitution seed/scan skipped: %s", exc)

    register_runtime_service(ServiceNames.HAL, inst, required=False, owner="core/goals/directive_conflict_sentinel.py", registered_by="register_directive_sentinel")
    register_runtime_service("hal", inst, required=False, owner="core/goals/directive_conflict_sentinel.py", registered_by="register_directive_sentinel")
    return inst


__all__ = [
    "Directive",
    "DirectiveConflict",
    "DirectiveConflictSentinel",
    "get_directive_sentinel",
    "register_directive_sentinel",
]
