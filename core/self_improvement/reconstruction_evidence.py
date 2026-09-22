"""Take a piece, and let it constrain the whole.

Reconstructing from published rules is the easy case. The real one is being
handed fragments — a binary, a screenshot, a packet capture, a log, a leaked
snippet, one file of a format — and having to infer the rest. That is what
makes this a reverse-engineering engine rather than a from-description code
generator, and it is what was missing: the plan came from her knowledge and
some research notes, so evidence in hand did nothing to the outcome.

Three ideas do the work here.

**Evidence becomes constraints, not prose.** A string table in a binary is a
vocabulary the reconstruction must contain. A screenshot's labels are a UI
surface it must offer. An observed request/response pair is behaviour it must
reproduce. Held as constraints, they can be *checked* against the artifact —
which is the difference between evidence informing a guess and evidence
grading it.

**Coverage is measured, so gaps are named.** Every constraint is either
addressed by the plan or it is not, and the ones that are not are exactly the
next questions to research. A reconstruction converges when its uncovered set
shrinks; without this it just stops when someone loses patience.

**Confidence is carried and never laundered.** A symbol read out of a binary is
near-certain. A behaviour inferred from a screenshot is a guess. Both are
useful, and merging them into one undifferentiated "evidence" is how a
reconstruction ends up asserting invented detail with the same voice as
observed fact.

Nothing here decompiles, unpacks, or circumvents anything: it reads what is
already readable in material the user supplies, which is the same authorization
boundary the rest of the Program DNA lane works under.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


_RECOVERABLE = (RuntimeError, AttributeError, TypeError, ValueError, OSError, KeyError)


class EvidenceKind(StrEnum):
    """Where a piece of evidence came from, which sets how much it is worth."""

    SOURCE_FRAGMENT = "source_fragment"        # actual code: near-certain
    OBSERVED_IO = "observed_io"                # ran it and watched: near-certain
    BINARY_STRINGS = "binary_strings"          # readable strings: strong
    FILE_FORMAT_SAMPLE = "file_format_sample"  # a real artifact it produced
    NETWORK_TRACE = "network_trace"            # endpoints and payload shapes
    LOG_OUTPUT = "log_output"                  # its own messages
    UI_CAPTURE = "ui_capture"                  # labels and layout: suggestive
    DOCUMENTATION = "documentation"            # what it claims about itself
    PRIOR_ART = "prior_art"                    # similar programs she has built


# How far a kind of evidence can be trusted on its own. Deliberately not 1.0
# anywhere: a string in a binary is real, but what it *means* is inference.
_KIND_CONFIDENCE: dict[EvidenceKind, float] = {
    EvidenceKind.SOURCE_FRAGMENT: 0.95,
    EvidenceKind.OBSERVED_IO: 0.9,
    EvidenceKind.FILE_FORMAT_SAMPLE: 0.85,
    EvidenceKind.BINARY_STRINGS: 0.7,
    EvidenceKind.NETWORK_TRACE: 0.7,
    EvidenceKind.LOG_OUTPUT: 0.6,
    EvidenceKind.UI_CAPTURE: 0.5,
    EvidenceKind.DOCUMENTATION: 0.5,
    EvidenceKind.PRIOR_ART: 0.35,
}


@dataclass(frozen=True)
class Evidence:
    """One piece the user supplied, with where it came from."""

    kind: EvidenceKind
    content: str
    source: str = ""

    @property
    def confidence(self) -> float:
        return _KIND_CONFIDENCE.get(self.kind, 0.4)


@dataclass(frozen=True)
class Constraint:
    """Something the reconstruction must account for, and why."""

    category: str          # vocabulary | behaviour | interface | format | structure
    statement: str
    confidence: float
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "statement": self.statement,
            "confidence": round(self.confidence, 3),
            "provenance": self.provenance,
        }


@dataclass
class GapReport:
    """What the evidence demands that the plan does not yet answer."""

    covered: list[Constraint] = field(default_factory=list)
    uncovered: list[Constraint] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        total = len(self.covered) + len(self.uncovered)
        return (len(self.covered) / total) if total else 1.0

    def next_questions(self, limit: int = 6) -> list[str]:
        """What to research next: the strongest evidence still unaccounted for.

        Ordered by confidence, because an unaddressed near-certainty is a hole
        in the reconstruction, while an unaddressed guess is merely a lead.
        """
        ranked = sorted(self.uncovered, key=lambda item: -item.confidence)
        return [
            f"{item.category}: {item.statement} (from {item.provenance})"
            for item in ranked[:limit]
        ]

    def summary(self) -> str:
        total = len(self.covered) + len(self.uncovered)
        if not total:
            return "no evidence supplied, so nothing constrains this reconstruction"
        return (
            f"{len(self.covered)}/{total} evidence constraints addressed "
            f"({self.coverage:.0%} coverage)"
        )


# ── Extractors: a piece becomes constraints ────────────────────────────────

_PRINTABLE_RUN = re.compile(rb"[ -~]{5,}")
_IDENTIFIER = re.compile(r"\b[a-z_][a-z0-9_]{2,40}\b")
_FUNCTION_DEF = re.compile(r"(?:def|function|fn|func)\s+([A-Za-z_][A-Za-z0-9_]*)")
_CLASS_DEF = re.compile(r"(?:class|struct|interface)\s+([A-Za-z_][A-Za-z0-9_]*)")
_HTTP_LINE = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE)\s+(/[^\s?]*)")
_UI_LABEL = re.compile(r"^[A-Z][A-Za-z0-9 '&/-]{1,28}$")
# Words too generic to constrain anything — they appear in every program.
_NOISE = frozenset(
    {
        "error", "warning", "true", "false", "null", "none", "string", "value",
        "object", "return", "self", "this", "data", "text", "name", "type",
        "list", "index", "print", "format", "input", "output", "file", "line",
    }
)


def _binary_strings(blob: bytes | str, limit: int = 60) -> list[str]:
    raw = blob.encode("utf-8", "ignore") if isinstance(blob, str) else blob
    found = [match.group(0).decode("ascii", "ignore") for match in _PRINTABLE_RUN.finditer(raw)]
    ranked = Counter(found)
    return [text for text, _ in ranked.most_common(limit)]


def extract_constraints(evidence: Evidence) -> list[Constraint]:
    """Turn one supplied piece into things the reconstruction must satisfy."""
    kind = evidence.kind
    content = str(evidence.content or "")
    provenance = evidence.source or kind.value
    confidence = evidence.confidence
    constraints: list[Constraint] = []

    def add(category: str, statement: str, scale: float = 1.0) -> None:
        statement = " ".join(statement.split())[:220]
        if statement:
            constraints.append(
                Constraint(category, statement, min(0.99, confidence * scale), provenance)
            )

    if kind is EvidenceKind.SOURCE_FRAGMENT:
        for name in dict.fromkeys(_FUNCTION_DEF.findall(content)):
            add("interface", f"defines a function named {name}")
        for name in dict.fromkeys(_CLASS_DEF.findall(content)):
            add("structure", f"has a type named {name}")

    elif kind is EvidenceKind.BINARY_STRINGS:
        for text in _binary_strings(content):
            lowered = text.strip().lower()
            if lowered in _NOISE or len(lowered) < 5:
                continue
            add("vocabulary", f"contains the string {text.strip()!r}", scale=0.9)

    elif kind is EvidenceKind.OBSERVED_IO:
        for line in content.splitlines():
            if "->" in line:
                given, _, produced = line.partition("->")
                add("behaviour", f"given {given.strip()} it produces {produced.strip()}")

    elif kind is EvidenceKind.NETWORK_TRACE:
        for verb, path in dict.fromkeys(_HTTP_LINE.findall(content)):
            add("interface", f"serves or calls {verb} {path}")

    elif kind is EvidenceKind.UI_CAPTURE:
        for line in content.splitlines():
            label = line.strip()
            if _UI_LABEL.match(label) and label.lower() not in _NOISE:
                add("interface", f"presents a control or label {label!r}", scale=0.9)

    elif kind is EvidenceKind.FILE_FORMAT_SAMPLE:
        head = content[:16].strip()
        if head:
            add("format", f"reads or writes files beginning {head!r}")
        for token in dict.fromkeys(_IDENTIFIER.findall(content))[:20]:
            if token not in _NOISE:
                add("format", f"its file format uses the key {token!r}", scale=0.8)

    elif kind is EvidenceKind.LOG_OUTPUT:
        for line in content.splitlines()[:40]:
            stripped = line.strip()
            if len(stripped) > 12:
                add("behaviour", f"emits the message {stripped!r}", scale=0.85)

    elif kind in {EvidenceKind.DOCUMENTATION, EvidenceKind.PRIOR_ART}:
        for sentence in re.split(r"(?<=[.!?])\s+", content):
            stripped = sentence.strip()
            if len(stripped) > 20:
                add("behaviour", stripped, scale=0.9)

    return constraints


def fuse_evidence(pieces: list[Evidence]) -> list[Constraint]:
    """All the pieces, as one constraint set, strongest evidence winning.

    The same fact from two independent kinds is worth more than from either
    alone, so a repeated statement keeps the higher confidence and records both
    sources rather than appearing twice.
    """
    merged: dict[tuple[str, str], Constraint] = {}
    for piece in pieces:
        try:
            extracted = extract_constraints(piece)
        except _RECOVERABLE:
            continue
        for constraint in extracted:
            key = (constraint.category, constraint.statement.lower())
            existing = merged.get(key)
            if existing is None:
                merged[key] = constraint
                continue
            # Independent corroboration: keep the stronger, name both sources,
            # and lift confidence without ever reaching certainty.
            if constraint.provenance not in existing.provenance:
                merged[key] = Constraint(
                    category=existing.category,
                    statement=existing.statement,
                    confidence=min(0.99, max(existing.confidence, constraint.confidence) + 0.08),
                    provenance=f"{existing.provenance}, {constraint.provenance}",
                )
            elif constraint.confidence > existing.confidence:
                merged[key] = constraint
    return sorted(merged.values(), key=lambda item: (-item.confidence, item.category))


def assess_coverage(plan: Any, constraints: list[Constraint]) -> GapReport:
    """Which constraints the plan actually accounts for, and which it ignores.

    Coverage is textual on purpose: a constraint is addressed when the plan
    mentions it — in an entry point, a component, a worked example, an invariant
    or a research note. That is a low bar, and it is meant to be. The failure
    this catches is not subtle disagreement; it is evidence being collected and
    then never looked at again.
    """
    haystack_parts: list[str] = []
    try:
        haystack_parts.append(str(getattr(plan, "target", "")))
        haystack_parts.append(str(getattr(plan, "summary", "")))
        haystack_parts.extend(str(name) for name in (getattr(plan, "entry_points", ()) or ()))
        for component in getattr(plan, "components", ()) or ():
            haystack_parts.append(str(getattr(component, "name", "")))
            haystack_parts.append(str(getattr(component, "responsibility", "")))
        for example in getattr(plan, "worked_examples", ()) or ():
            haystack_parts.append(str(getattr(example, "entry_point", "")))
            haystack_parts.append(str(getattr(example, "argument", "")))
            haystack_parts.append(str(getattr(example, "expected", "")))
        for invariant in getattr(plan, "invariants", ()) or ():
            haystack_parts.append(str(getattr(invariant, "description", "")))
            haystack_parts.append(str(getattr(invariant, "expression", "")))
        haystack_parts.extend(str(note) for note in (getattr(plan, "research_notes", ()) or ()))
    except _RECOVERABLE as exc:
        logger.debug(
            "part of the plan would not render, so the search text is incomplete (%s: %s)",
            type(exc).__name__,
            exc,
        )
    haystack = " ".join(haystack_parts).lower()

    report = GapReport()
    for constraint in constraints:
        tokens = [
            token
            for token in re.findall(r"[a-z0-9_]{3,}", constraint.statement.lower())
            if token not in _NOISE
        ]
        # A constraint is addressed when its distinctive words appear. Requiring
        # every token would make any rewording a gap; requiring one would make
        # everything covered.
        if not tokens:
            report.covered.append(constraint)
            continue
        hits = sum(1 for token in tokens if token in haystack)
        if hits >= max(1, len(tokens) // 3):
            report.covered.append(constraint)
        else:
            report.uncovered.append(constraint)
    return report


__all__ = [
    "Constraint",
    "Evidence",
    "EvidenceKind",
    "GapReport",
    "assess_coverage",
    "extract_constraints",
    "fuse_evidence",
]
