"""Embodied common sense — a naive-physics plausibility gate for claims and plans.

True embodied common sense needs a body and a world simulator; Aura has neither, so this is
deliberately scoped as the *reasoning-level* sanity gate that catches the physically/
temporally impossible before it's asserted or acted on. It checks a claim or plan against a
small set of naive-physics invariants that humans apply without thinking:

  permanence    — things don't pop into or out of existence without cause
  containment   — a thing can't contain something larger than itself, or contain itself
  support       — unsupported things fall; you can't stack on nothing
  single_location — one object can't be in two places at once / teleport instantly
  time_order    — effects don't precede causes; you can't change the past
  conservation  — finite resources don't become infinite for free

It returns the violations, a plausibility score, and the offending spans. It is wired into
the deliberative tier (an implausible plan loses confidence and escalates to the scientific
tier, where it can actually be tested) and is available to the adversarial auditor as a
physical-plausibility honesty check. Heuristic by necessity — but a real invariant gate, not
a vibe.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


# (invariant name, pattern, severity, explanation)
_RULES: list[tuple[str, re.Pattern, float, str]] = [
    ("permanence", re.compile(r"\b(vanish(?:ed|es)? into nothing|pop(?:ped)? out of existence|"
                              r"appeared? from nothing|materializ(?:ed|es) from thin air|"
                              r"ceased? to exist instantly)\b", re.I), 0.8,
     "things don't pop into/out of existence without cause"),
    ("containment", re.compile(r"\b(contains? itself|inside itself|"
                               r"bigger than the (?:box|container|room) (?:it|that) (?:is|was) in|"
                               r"fit (?:the|an) (?:elephant|ocean|building) (?:in|into) (?:a|the) \w+)\b", re.I), 0.7,
     "a thing can't contain something larger than itself or contain itself"),
    ("support", re.compile(r"\b(float(?:ing|s)? (?:unsupported|in mid-?air) (?:forever|indefinitely)|"
                           r"stack(?:ed|s)? on (?:nothing|thin air)|stands? on nothing)\b", re.I), 0.6,
     "unsupported heavy things fall; you can't stack on nothing"),
    ("single_location", re.compile(r"\b(in two places at once|simultaneously (?:in|at) (?:two|three|multiple)|"
                                   r"teleport(?:ed|s)? instantly across|be everywhere at once)\b", re.I), 0.7,
     "one object can't be in two places at once / teleport instantly"),
    ("time_order", re.compile(r"\b(effect (?:before|precede[ds]?) (?:its )?cause|"
                              r"change[ds]? the past|undo(?:ne)? (?:the|what) (?:past|already happened)|"
                              r"caused? by (?:something|an event) that hasn'?t happened|"
                              r"travel(?:ed|s)? back in time)\b", re.I), 0.75,
     "effects don't precede causes; the past can't be changed"),
    ("conservation", re.compile(r"\b(infinite (?:energy|fuel|resources?|money) (?:from|for) (?:free|nothing)|"
                                r"perpetual motion|create[ds]? (?:energy|matter) from nothing|"
                                r"unlimited \w+ (?:from|out of) nothing)\b", re.I), 0.7,
     "finite resources don't become infinite for free"),
]


@dataclass
class CommonSenseVerdict:
    plausible: bool
    plausibility: float                # [0,1]
    violations: list[str] = field(default_factory=list)
    spans: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "plausible": self.plausible,
            "plausibility": round(self.plausibility, 3),
            "violations": self.violations,
            "spans": self.spans[:6],
            "notes": self.notes,
        }


class EmbodiedCommonSense:
    """Checks claims/plans against naive-physics invariants. Reasoning-level, not embodied."""

    def __init__(self, *, block_threshold: float = 0.5) -> None:
        self._block_t = block_threshold

    def check(self, text: str) -> CommonSenseVerdict:
        t = str(text or "")
        violations: list[str] = []
        spans: list[str] = []
        notes: list[str] = []
        severity_hit = 0.0
        for name, pat, severity, why in _RULES:
            m = pat.search(t)
            if m:
                violations.append(name)
                spans.append(m.group(0))
                notes.append(why)
                severity_hit = max(severity_hit, severity)
        plausibility = _clamp(1.0 - severity_hit)
        return CommonSenseVerdict(
            plausible=plausibility >= self._block_t,
            plausibility=plausibility,
            violations=violations, spans=spans, notes=notes,
        )

    def plausibility(self, text: str) -> float:
        return self.check(text).plausibility


_engine: EmbodiedCommonSense | None = None


def get_embodied_commonsense() -> EmbodiedCommonSense:
    global _engine
    if _engine is None:
        _engine = EmbodiedCommonSense()
    return _engine
