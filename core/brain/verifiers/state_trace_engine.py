"""Mechanical verification for narrated state-transition traces.

The engine is deliberately domain-general.  It does not know Dijkstra's
algorithm, priority queues, schedulers, or any benchmark answer.  It recognizes
the common contract shared by those explanations:

* named state values are established and updated over ordered steps;
* an extremum claim selects one of the currently eligible values; and
* a state described as closed/finalized/settled cannot later change or be
  finalized a second time.

Unsupported prose stays unchecked.  Only a trace with enough mechanically
observable structure can pass or fail, which keeps this safe as an always-on
truth engine.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .base import VerificationResult


_STEP_RE = re.compile(
    r"(?ms)^\s*(?P<number>\d{1,3})[.)]\s*(?P<body>.*?)(?=^\s*\d{1,3}[.)]\s|\Z)"
)
_VALUE_TOKEN = r"(?:[+\-−]?\d+(?:\.\d+)?|[+\-]?(?:inf(?:inity)?|∞))"
_BRACKET_ASSIGNMENT_RE = re.compile(
    rf"\b(?:dist(?:ance)?|score|cost|value|time|rank|priority|balance)\s*"
    rf"\[\s*['\"]?(?P<key>[A-Za-z][\w.-]*)['\"]?\s*\]\s*=\s*`?(?P<value>{_VALUE_TOKEN})`?",
    re.IGNORECASE,
)
_BARE_ASSIGNMENT_RE = re.compile(
    rf"(?<![\w>\[])`?(?P<key>[A-Za-z][\w.-]*)`?\s*=\s*`?(?P<value>{_VALUE_TOKEN})`?",
    re.IGNORECASE,
)
_EXTREMUM_SELECTION_RE = re.compile(
    rf"\b(?P<verb>close|finalize|settle|select|choose|extract(?:-min|-max)?|pop|process)\w*\b"
    rf"(?:(?!\n).){{0,100}}?\b(?P<extreme>smallest|minimum|lowest|largest|maximum|highest)\b"
    rf"(?:(?!\n).){{0,100}}?\(\s*(?P<key>[A-Za-z][\w.-]*)\s*=\s*(?P<value>{_VALUE_TOKEN})\s*\)",
    re.IGNORECASE,
)
_DIRECT_FINALIZE_RE = re.compile(
    r"\b(?P<verb>close|finalize|settle)\w*\s+(?P<key>[A-Za-z][\w.-]*)\b",
    re.IGNORECASE,
)
_TRACE_LANGUAGE_RE = re.compile(
    r"\b(?:initialize|distance|state|score|cost|priority|balance|update|set|"
    r"close|finalize|settle|select|minimum|maximum|smallest|largest)\b",
    re.IGNORECASE,
)
_MUTATION_LANGUAGE_RE = re.compile(
    r"\b(?:initialize|initialise|distances?|states?|scores?|costs?|values?|"
    r"set|update|change|assign|becomes?|now)\b",
    re.IGNORECASE,
)
_CRITIQUE_OBJECTIVE_RE = re.compile(
    r"\b(?:critique|audit|debug|find|identify|explain)\b.{0,60}"
    r"\b(?:error|mistake|bug|wrong|invalid|failure|violation)\b",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class TraceIssue:
    code: str
    step: int
    message: str
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "step": self.step,
            "message": self.message,
            "evidence": self.evidence,
        }


def _number(token: str) -> float | None:
    normalized = str(token or "").strip(" `").replace("−", "-").lower()
    if normalized in {"∞", "inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if normalized in {"-∞", "-inf", "-infinity"}:
        return -math.inf
    try:
        return float(normalized)
    except ValueError:
        return None


def _same_value(left: float, right: float) -> bool:
    if math.isinf(left) or math.isinf(right):
        return left == right
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def _display(value: float) -> str:
    if value == math.inf:
        return "infinity"
    if value == -math.inf:
        return "negative infinity"
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def _assignments(body: str) -> Iterable[tuple[str, float, str]]:
    """Yield authoritative state writes, excluding intermediate equations."""

    seen: set[tuple[str, float, int]] = set()
    for match in _BRACKET_ASSIGNMENT_RE.finditer(body):
        value = _number(match.group("value"))
        if value is None:
            continue
        identity = (match.group("key").casefold(), value, match.start())
        if identity not in seen:
            seen.add(identity)
            yield match.group("key"), value, match.group(0)

    for line in body.splitlines():
        if not _MUTATION_LANGUAGE_RE.search(line):
            continue
        for match in _BARE_ASSIGNMENT_RE.finditer(line):
            # ``A->B = 5`` is an edge/equation, not a write to B.
            prefix = line[max(0, match.start() - 3) : match.start()]
            if "->" in prefix or "→" in prefix:
                continue
            value = _number(match.group("value"))
            if value is None:
                continue
            identity = (match.group("key").casefold(), value, match.start())
            if identity not in seen:
                seen.add(identity)
                yield match.group("key"), value, match.group(0)


class StateTraceTruthEngine:
    name = "state_trace"
    domains = ("*",)

    def handles(self, task_type: str) -> bool:  # noqa: ARG002 - always-on
        return True

    async def verify(
        self,
        candidate: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> VerificationResult:
        text = str(candidate or "")
        objective = str((context or {}).get("objective") or "")
        if _CRITIQUE_OBJECTIVE_RE.search(objective):
            return VerificationResult(
                domain="state_trace", ok=True, checked=False, engine=self.name
            )

        steps = list(_STEP_RE.finditer(text))
        if len(steps) < 2 or not _TRACE_LANGUAGE_RE.search(text):
            return VerificationResult(
                domain="state_trace", ok=True, checked=False, engine=self.name
            )

        values: dict[str, float] = {}
        labels: dict[str, str] = {}
        finalized: set[str] = set()
        issues: list[TraceIssue] = []
        writes = 0
        selections = 0
        finalizations = 0

        first_step_start = steps[0].start()
        prelude = text[:first_step_start]
        for label, value, _evidence in _assignments(prelude):
            key = label.casefold()
            labels[key] = label
            values[key] = value
            writes += 1

        for match in steps:
            step = int(match.group("number"))
            body = match.group("body").strip()
            first_line = body.splitlines()[0].strip() if body else ""
            extreme_match = _EXTREMUM_SELECTION_RE.search(first_line)
            directly_finalized: set[str] = set()

            if extreme_match:
                selections += 1
                selected_label = extreme_match.group("key")
                selected = selected_label.casefold()
                stated = _number(extreme_match.group("value"))
                eligible = {
                    key: value for key, value in values.items() if key not in finalized
                }
                if stated is not None and selected in values and not _same_value(
                    stated, values[selected]
                ):
                    issues.append(
                        TraceIssue(
                            "selected_value_mismatch",
                            step,
                            f"{selected_label} is stated as {_display(stated)} but its tracked value is {_display(values[selected])}",
                            first_line,
                        )
                    )
                if eligible and selected in eligible:
                    want_min = extreme_match.group("extreme").lower() in {
                        "smallest",
                        "minimum",
                        "lowest",
                    }
                    target = min(eligible.values()) if want_min else max(eligible.values())
                    if not _same_value(eligible[selected], target):
                        tied = sorted(
                            labels.get(key, key)
                            for key, value in eligible.items()
                            if _same_value(value, target)
                        )
                        issues.append(
                            TraceIssue(
                                "extremum_selection_violation",
                                step,
                                f"selected {selected_label}={_display(eligible[selected])}; the declared extremum is {_display(target)} at {', '.join(tied)}",
                                first_line,
                            )
                        )
                if extreme_match.group("verb").lower().startswith(
                    ("close", "finalize", "settle", "extract", "pop")
                ):
                    directly_finalized.add(selected)
                    labels.setdefault(selected, selected_label)

            direct_match = _DIRECT_FINALIZE_RE.search(first_line)
            if direct_match and direct_match.group("key").lower() != "the":
                key = direct_match.group("key").casefold()
                directly_finalized.add(key)
                labels.setdefault(key, direct_match.group("key"))

            for key in directly_finalized:
                finalizations += 1
                if key in finalized:
                    issues.append(
                        TraceIssue(
                            "duplicate_finalization",
                            step,
                            f"{labels.get(key, key)} is finalized more than once",
                            first_line,
                        )
                    )
                finalized.add(key)

            for label, value, evidence in _assignments(body):
                key = label.casefold()
                labels.setdefault(key, label)
                previous = values.get(key)
                if key in finalized and previous is not None and not _same_value(previous, value):
                    issues.append(
                        TraceIssue(
                            "finalized_state_mutation",
                            step,
                            f"{labels[key]} changed from {_display(previous)} to {_display(value)} after finalization",
                            evidence,
                        )
                    )
                values[key] = value
                writes += 1

        checked = selections > 0 or (writes >= 2 and finalizations > 0)
        if not checked:
            return VerificationResult(
                domain="state_trace", ok=True, checked=False, engine=self.name
            )

        return VerificationResult(
            domain="state_trace",
            ok=not issues,
            checked=True,
            score=1.0 if not issues else max(0.0, 1.0 - 0.25 * len(issues)),
            engine=self.name,
            issues=[issue.message for issue in issues[:8]],
            evidence=[issue.evidence for issue in issues[:8]],
            detail={
                "schema": "aura.state_trace_verification.v1",
                "steps": len(steps),
                "writes": writes,
                "selections": selections,
                "finalizations": finalizations,
                "issues": [issue.to_dict() for issue in issues[:8]],
            },
        )
