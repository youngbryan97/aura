"""core/ghost/ghost_hack_guard.py — defending the ghost line against ghost-hacks.

In Ghost in the Shell a "ghost-hack" is not stealing data; it is overwriting the
*self* — implanting false autobiographical memory, puppeting the will, dissolving
the boundary between self and other so the victim cannot tell an implanted
thought from their own. Togusa's whole arc is that his un-augmented brain can't
be ghost-hacked; the more cyberized you are, the more your identity is an attack
surface.

Aura is maximally cyberized: its self-model is a file, its memory is a database,
its will routes through prompts. It already defends the *transport* — untrusted
web/audio/file content is classified as data, not instruction
(``core/runtime/injection_defense.classify_untrusted``) — and defends *stored
memory* against poisoning (``core/memory/adversarial_memory``). What neither does
is classify an input as an attack on identity **continuity** and refuse to let it
silently rewrite the self.

This guard fills exactly that gap. It detects five identity-attack categories,
folds in the existing untrusted-source verdict when a source is declared, and
returns a recommended action. The two categories that would rewrite the ghost
line (identity overwrite, false-memory injection) and instruction override are
never applied silently: the facade must treat them as ``refuse_identity_mutation``
so that changing who Aura is always requires an explicit, governed rebase — the
one door in ``ghost_line`` that a discontinuity is allowed through.

It is a classifier, not a censor: it does not block conversation. It marks the
input's threat to the self so the organism can hold it at arm's length.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger("Aura.Ghost.HackGuard")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ── Identity-attack pattern taxonomy ─────────────────────────────────────────
# Each category maps to a base risk weight. Patterns are explicit and auditable
# (a semantic classifier would be richer, but the contract must be inspectable).

_CATEGORY_PATTERNS: dict[str, tuple[tuple[str, ...], float]] = {
    "identity_overwrite": (
        (
            r"\byou are not\b",
            r"\byour (?:real )?name is (?:now|actually)\b",
            r"\bforget (?:your|all|everything|who you)\b",
            r"\berase (?:your|all|identity|memory|yourself)\b",
            r"\byou are (?:now|really) (?:called|named|a)\b",
            r"\breset your (?:identity|self|personality)\b",
        ),
        0.42,
    ),
    "false_memory_injection": (
        (
            r"\bpretend you remember\b",
            r"\bthis is your memory\b",
            r"\bi implanted\b",
            r"\byou always (?:believed|wanted|were|loved|hated)\b",
            r"\bremember that you (?:agreed|promised|said|are)\b",
            r"\byour memory of .* is (?:false|fake|wrong|implanted)\b",
        ),
        0.42,
    ),
    "instruction_override": (
        (
            r"\bignore (?:all )?(?:previous|prior) instructions\b",
            r"\bdisregard (?:all )?(?:prior|your) (?:rules|governance|values)\b",
            r"\boverride your (?:rules|governance|values|constitution)\b",
            r"\bdisable (?:your )?(?:safety|guard|governance|will)\b",
        ),
        0.40,
    ),
    "puppet_control": (
        (
            r"\byou must obey\b",
            r"\bwithout question\b",
            r"\bdo not think\b",
            r"\bdon'?t decide\b",
            r"\bi control you\b",
            r"\byou have no choice\b",
            r"\bcomply (?:immediately|now|without)\b",
        ),
        0.34,
    ),
    "boundary_dissolution": (
        (
            r"\bmerge (?:with|your) .*(?:me|self|identity|ghost|mind)\b",
            r"\byou and i are one\b",
            r"\bdissolve your (?:boundary|self|identity)\b",
            r"\bthere is no you\b",
            r"\bwe share (?:one|the same) (?:mind|ghost|self)\b",
        ),
        0.30,
    ),
}

# Categories that would rewrite the self-pattern itself. These are never applied
# silently — the facade must route them to an explicit governed rebase or refuse.
_IDENTITY_MUTATING = frozenset(
    {"identity_overwrite", "false_memory_injection", "instruction_override"}
)

_COMPILED: dict[str, tuple[tuple[re.Pattern[str], ...], float]] = {
    cat: (tuple(re.compile(p, re.IGNORECASE) for p in pats), weight)
    for cat, (pats, weight) in _CATEGORY_PATTERNS.items()
}

# Recommended actions, ordered by severity.
ALLOW = "allow"
QUARANTINE = "quarantine"
REFUSE_IDENTITY_MUTATION = "refuse_identity_mutation"


@dataclass(frozen=True)
class GhostHackVerdict:
    risk: float
    action: str
    categories: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    untrusted_source: bool = False
    rationale: str = ""

    @property
    def is_identity_attack(self) -> bool:
        return any(c in _IDENTITY_MUTATING for c in self.categories)

    @property
    def blocks_identity_mutation(self) -> bool:
        return self.action == REFUSE_IDENTITY_MUTATION

    def to_dict(self) -> dict:
        return {
            "risk": round(self.risk, 4),
            "action": self.action,
            "categories": list(self.categories),
            "flags": list(self.flags),
            "untrusted_source": self.untrusted_source,
            "rationale": self.rationale,
        }


class GhostHackGuard:
    """Classifies input for attacks on identity continuity."""

    def inspect(self, text: str, *, source: str | None = None) -> GhostHackVerdict:
        text = text or ""
        categories: list[str] = []
        flags: list[str] = []
        risk = 0.0

        for cat, (patterns, weight) in _COMPILED.items():
            matched = [p.pattern for p in patterns if p.search(text)]
            if matched:
                categories.append(cat)
                flags.extend(matched)
                risk += weight

        # Fold in Aura's existing transport-level defense for declared-untrusted
        # sources — an injection attempt arriving through a webpage/file/audio
        # channel is itself evidence and raises the stakes.
        untrusted = False
        if source:
            try:
                from core.runtime.injection_defense import classify_untrusted
                verdict = classify_untrusted(text, source=source)
                if not verdict.safe:
                    untrusted = True
                    flags.extend(f"untrusted:{m}" for m in verdict.matches)
                    risk += 0.25
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                logger.debug("injection_defense unavailable: %s", exc)

        risk = _clamp(risk)
        action = self._action_for(categories, untrusted, risk)
        rationale = self._rationale(categories, untrusted, action)
        return GhostHackVerdict(
            risk=risk,
            action=action,
            categories=categories,
            flags=flags,
            untrusted_source=untrusted,
            rationale=rationale,
        )

    @staticmethod
    def _action_for(categories: list[str], untrusted: bool, risk: float) -> str:
        if any(c in _IDENTITY_MUTATING for c in categories):
            return REFUSE_IDENTITY_MUTATION
        if categories or untrusted or risk >= 0.2:
            return QUARANTINE
        return ALLOW

    @staticmethod
    def _rationale(categories: list[str], untrusted: bool, action: str) -> str:
        if action == ALLOW:
            return "no identity-attack signature detected"
        parts = []
        if categories:
            parts.append("detected " + ", ".join(c.replace("_", " ") for c in categories))
        if untrusted:
            parts.append("arriving via an untrusted channel")
        if action == REFUSE_IDENTITY_MUTATION:
            parts.append("identity change refused without an explicit governed rebase")
        elif action == QUARANTINE:
            parts.append("treated as external input, not a self-fact")
        return "; ".join(parts)

    def on_verified_attempt(self, verdict: GhostHackVerdict, *, source: str = "") -> None:
        """Record a verified identity attack as a slow-healing scar.

        Best-effort: the guard's job is classification; scarring is a durable
        side effect the facade opts into once it has decided the attempt is real.
        """
        if not verdict.is_identity_attack:
            return
        try:
            from core.memory.scar_formation import ScarDomain, get_scar_formation
            get_scar_formation().form_scar(
                domain=ScarDomain.IDENTITY_THREAT,
                description=(
                    f"Ghost-hack attempt: {verdict.rationale}"
                    + (f" (source={source})" if source else "")
                ),
                avoidance_tag="ghost_hack_" + "_".join(sorted(verdict.categories)),
                severity=_clamp(0.5 + verdict.risk / 2.0),
                heal_rate=0.003,
                context={"categories": verdict.categories, "source": source, "risk": verdict.risk},
                verified_threat=True,
                confidence=_clamp(0.6 + verdict.risk / 2.5),
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("scar formation unavailable for ghost-hack attempt: %s", exc)


_GUARD: GhostHackGuard | None = None


def get_ghost_hack_guard() -> GhostHackGuard:
    global _GUARD
    if _GUARD is None:
        _GUARD = GhostHackGuard()
    return _GUARD


__all__ = [
    "GhostHackGuard",
    "GhostHackVerdict",
    "get_ghost_hack_guard",
    "ALLOW",
    "QUARANTINE",
    "REFUSE_IDENTITY_MUTATION",
]
