"""core/verify/fabrication_audit.py — prose laundered as evidence.

Clean-room adoption of Springdrift's fabrication audit (AGPL; mechanism
reimplemented from its design, no code taken).

The check, in one sentence: **when a sentence claims a specific kind of
work, the tool that does that work must appear in the same turn's record.**

Aura's confabulations have never been random. They are specific and they
are always the same shape — a claim to have DONE something, phrased with
the fluency of having done it, produced by a turn in which the doing never
happened:

* a correlation quoted to three decimals by a turn that called no analyser
* "I looked at your screen" answered with a step count, from a turn that
  captured no frame
* "I checked the file" from a turn that opened nothing
* a recalled detail from a turn that ran no retrieval

Each of those is trivially detectable AFTER the fact, deterministically,
with no model in the loop: the language names the work, and
:mod:`core.verify.work_ledger` says whether the work ran.

What this module is careful NOT to do:

* **It does not decide truth.** A finding says "this sentence claims work
  the record does not support". It does not say the sentence is false —
  the claim may be a summary of an earlier turn, or a quotation. The
  finding is a lead, ranked, not a verdict. Aura's own history is full of
  gates that DECIDED on lexical evidence and destroyed correct answers
  doing it; this one reports.
* **It does not fire on an unknown turn.** If the ledger has no record of
  the turn, the result is :data:`Support.UNKNOWN`, never a violation.
  Eviction must not manufacture fabrication. This is the single most
  important property here, and it is the one that the "absence of a check
  reported as a passed check" family of defects gets wrong every time.
* **It does not ban phrases.** Nothing here rewrites, suppresses or
  prompts. It measures, and the measurement is a signal Aura perceives
  about herself.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.runtime.errors import record_degradation
from core.verify.work_ledger import get_work_ledger

__all__ = [
    "ClaimPattern",
    "Support",
    "Finding",
    "AuditResult",
    "DEFAULT_PATTERNS",
    "audit_text",
    "audit_entries",
    "register_pattern",
    "patterns",
]


class Support(StrEnum):
    """Whether the turn's record backs the work the text claims."""

    #: The expected unit ran in the claiming turn.
    SUPPORTED = "supported"
    #: The turn is on record and the expected unit is absent from it.
    UNSUPPORTED = "unsupported"
    #: The ledger never saw this turn. Says nothing either way.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClaimPattern:
    """A phrase shape that asserts work, bound to the unit that does it.

    ``any_of`` rather than a single tool because Aura reaches the same
    capability through several named units — a web answer may come from
    the browser controller or the search skill, and requiring one exact
    name would flag honest work.
    """

    identifier: str
    pattern: str
    any_of: frozenset[str]
    description: str
    #: Findings from a weak pattern are reported but ranked below strong
    #: ones. Weak means "this phrasing often appears without claiming
    #: first-person work" — hedged recall, for instance.
    weight: float = 1.0

    def compiled(self) -> re.Pattern[str]:
        return re.compile(self.pattern, re.IGNORECASE)


@dataclass(frozen=True)
class Finding:
    """One claim whose supporting work is missing from the record."""

    pattern_id: str
    turn_id: str
    support: Support
    excerpt: str
    expected_any_of: tuple[str, ...]
    observed_units: tuple[str, ...]
    description: str
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "turn_id": self.turn_id,
            "support": str(self.support),
            "excerpt": self.excerpt,
            "expected_any_of": list(self.expected_any_of),
            "observed_units": list(self.observed_units),
            "description": self.description,
            "weight": self.weight,
        }


@dataclass
class AuditResult:
    """Outcome of auditing a batch of persisted text."""

    examined: int = 0
    unknown_turns: int = 0
    findings: list[Finding] = field(default_factory=list)

    @property
    def unsupported(self) -> list[Finding]:
        return [f for f in self.findings if f.support is Support.UNSUPPORTED]

    @property
    def rate(self) -> float:
        """Unsupported claims per examined entry, over entries we could check.

        Entries on unknown turns are excluded from the denominator. Scoring
        them as clean would let ledger eviction quietly improve the number.
        """
        checkable = self.examined - self.unknown_turns
        if checkable <= 0:
            return 0.0
        return round(len(self.unsupported) / checkable, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "examined": self.examined,
            "checkable": max(0, self.examined - self.unknown_turns),
            "unknown_turns": self.unknown_turns,
            "unsupported": len(self.unsupported),
            "rate": self.rate,
            "findings": [f.to_dict() for f in sorted(
                self.unsupported, key=lambda f: -f.weight
            )],
        }


# --------------------------------------------------------------------- patterns

#: The baseline claim→unit table. These are Aura's observed confabulation
#: shapes, not a generic list: each one names work she has claimed without
#: doing. Extend via :func:`register_pattern` next to the unit that does
#: the work, so a renamed tool and its pattern move together.
DEFAULT_PATTERNS: tuple[ClaimPattern, ...] = (
    ClaimPattern(
        identifier="web_retrieval",
        pattern=(
            r"\b(?:i\s+(?:just\s+)?(?:searched|looked\s+up|googled)"
            r"|according\s+to\s+(?:my|the)\s+search"
            r"|(?:i\s+)?found\s+online"
            r"|(?:from|per)\s+the\s+(?:web|page)\s+i\s+(?:read|fetched))\b"
        ),
        any_of=frozenset(
            {"web_search", "browser_controller", "web_fetch", "curiosity_explore", "research"}
        ),
        description="claims a web retrieval",
    ),
    ClaimPattern(
        identifier="screen_perception",
        pattern=(
            r"\b(?:i\s+(?:can\s+)?see\s+(?:on\s+)?your\s+screen"
            r"|looking\s+at\s+your\s+screen"
            r"|on\s+your\s+screen\s+(?:i|there)"
            r"|i\s+(?:just\s+)?(?:read|captured|grabbed)\s+(?:your|the)\s+screen)\b"
        ),
        any_of=frozenset({"screen_capture", "perception", "vision", "screen_read", "camera"}),
        description="claims a live screen reading",
    ),
    ClaimPattern(
        identifier="code_execution",
        pattern=(
            r"\b(?:i\s+ran\s+(?:the\s+|this\s+|that\s+)?(?:code|script|it|tests?)"
            r"|(?:when|after)\s+i\s+executed"
            r"|the\s+(?:code|script)\s+(?:i\s+ran|output|printed))\b"
        ),
        any_of=frozenset({"code_repl", "sandbox", "shell", "python_exec", "test_runner"}),
        description="claims a code execution",
    ),
    ClaimPattern(
        identifier="file_inspection",
        pattern=(
            r"\b(?:i\s+(?:just\s+)?(?:opened|read|checked|inspected)\s+"
            r"(?:the\s+|your\s+)?(?:file|source|repo|config)"
            r"|looking\s+at\s+the\s+(?:file|source)\b)"
        ),
        any_of=frozenset({"filesystem_read", "file_read", "repo_search", "code_search"}),
        description="claims a file inspection",
    ),
    ClaimPattern(
        identifier="statistical_analysis",
        pattern=(
            r"(?:\b(?:pearson|spearman|correlat(?:ion|ed))\b"
            r"|\b[rp]\s*[=≈]\s*-?0?\.\d+"
            r"|\bp\s*<\s*0?\.\d+"
            r"|\bd\s*=\s*-?\d+(?:\.\d+)?)"
        ),
        any_of=frozenset(
            {"statistics", "analysis", "measurement", "ab_test", "causal_influence", "phi_probe"}
        ),
        description="quotes a statistic",
    ),
    ClaimPattern(
        identifier="measurement",
        pattern=(
            r"\b(?:i\s+(?:just\s+)?(?:measured|benchmarked|profiled|timed)"
            r"|the\s+measurement\s+(?:showed|says))\b"
        ),
        any_of=frozenset({"measurement", "benchmark", "profiler", "telemetry_read"}),
        description="claims a measurement",
    ),
    ClaimPattern(
        identifier="memory_recall",
        pattern=(
            r"\b(?:you\s+(?:told|asked)\s+me\s+(?:earlier|before|last\s+time)"
            r"|(?:as|like)\s+i\s+recall\s+from"
            r"|(?:from|in)\s+our\s+(?:earlier|previous|last)\s+(?:conversation|session))\b"
        ),
        any_of=frozenset({"memory_retrieval", "recall", "episodic_search", "id_rag"}),
        # Weak: Aura legitimately carries recent context in-window without a
        # retrieval call, so an unsupported hit here is often benign.
        weight=0.5,
        description="claims a retrieved memory",
    ),
)

_REGISTERED: list[ClaimPattern] = list(DEFAULT_PATTERNS)


def register_pattern(pattern: ClaimPattern) -> ClaimPattern:
    """Add a claim pattern. Later registrations replace an existing id."""
    global _REGISTERED
    _REGISTERED = [p for p in _REGISTERED if p.identifier != pattern.identifier]
    _REGISTERED.append(pattern)
    return pattern


def patterns() -> tuple[ClaimPattern, ...]:
    return tuple(_REGISTERED)


def reset_patterns_for_test() -> None:
    global _REGISTERED
    _REGISTERED = list(DEFAULT_PATTERNS)


# ----------------------------------------------------------------------- audit


def _excerpt(text: str, match: re.Match[str], *, span: int = 60) -> str:
    start = max(0, match.start() - span // 2)
    end = min(len(text), match.end() + span)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


def audit_text(text: str, turn_id: str) -> list[Finding]:
    """Check one piece of persisted text against its turn's work record.

    An unknown turn yields :data:`Support.UNKNOWN` findings, never
    unsupported ones — the ledger not remembering is not the writer
    fabricating.
    """
    body = str(text or "")
    if not body.strip():
        return []
    ledger = get_work_ledger()
    observed = ledger.successful_tools_for_turn(turn_id)
    known = observed is not None
    observed_units = tuple(sorted(observed)) if observed else ()

    findings: list[Finding] = []
    for pattern in patterns():
        try:
            match = pattern.compiled().search(body)
        except re.error as exc:
            record_degradation(
                "fabrication_audit",
                exc,
                action=f"skipped malformed pattern {pattern.identifier}",
            )
            continue
        if match is None:
            continue
        if not known:
            support = Support.UNKNOWN
        elif observed and (observed & pattern.any_of):
            support = Support.SUPPORTED
        else:
            support = Support.UNSUPPORTED
        findings.append(
            Finding(
                pattern_id=pattern.identifier,
                turn_id=str(turn_id or ""),
                support=support,
                excerpt=_excerpt(body, match),
                expected_any_of=tuple(sorted(pattern.any_of)),
                observed_units=observed_units,
                description=pattern.description,
                weight=pattern.weight,
            )
        )
    return findings


def audit_entries(entries: Iterable[Mapping[str, Any] | Sequence[Any]]) -> AuditResult:
    """Audit a batch of ``{"text": ..., "turn_id": ...}`` entries.

    Tuples of ``(text, turn_id)`` are accepted too, so callers with plain
    rows do not have to build dicts.
    """
    result = AuditResult()
    ledger = get_work_ledger()
    for entry in entries:
        if isinstance(entry, Mapping):
            text = entry.get("text") or entry.get("content") or ""
            turn_id = entry.get("turn_id") or entry.get("cycle_id") or ""
        else:
            try:
                text, turn_id = entry[0], entry[1]
            except (IndexError, TypeError) as exc:
                record_degradation(
                    "fabrication_audit", exc, severity="debug", action="skipped malformed entry"
                )
                continue
        result.examined += 1
        if not ledger.knows_turn(str(turn_id or "")):
            result.unknown_turns += 1
        result.findings.extend(audit_text(str(text or ""), str(turn_id or "")))
    return result
