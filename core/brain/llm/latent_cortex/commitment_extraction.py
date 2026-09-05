"""Where a commitment comes from — evidence, never assertion.

The ratchet is only as good as what it commits. A constraint invented by the
model about its own answer is the model marking its own homework, and that is
the failure this codebase has already paid for twice: candidate-local scores
becoming Wilson trials, and "the faculty reports it ran" being read as "the
faculty mattered".

So every constraint here has to come from one of three places, in order of
strength:

  ELIMINATION   a candidate answer was REFUTED by a deterministic verifier,
                so the answer is not that. This is the strongest kind: the
                evidence is a failed check, not an opinion, and the resulting
                EXCLUDES constraint is exactly what a chain-of-thought token
                does when it rules a branch out;

  AGREEMENT     independently sampled candidates agree on a structural
                property (they are all numbers; they all mention X; they all
                lie in a range). Agreement across independent samples is
                evidence about the problem, not about any one sample. The
                bar is unanimity, because a majority-vote constraint is a
                consensus mechanism and consensus collapses — which this
                codebase measured as "collapse is cheapest";

  STATED        the PROMPT says so. "Answer in one word", "give three
                examples", "in kilometres". These are free, exact, and
                routinely ignored by a model eight passes deep; committing
                them makes them structural rather than hopeful.

Nothing here proposes a constraint from a latent vector. A vector cannot be
checked, and an unchecked commitment is a sentence in a prompt.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from core.brain.llm.latent_cortex.commitment_ratchet import (
    Constraint,
    ConstraintKind,
    _list_items,
    _normalize,
    _numbers_in,
)

#: Unanimity, not majority. A property two of three candidates share is a
#: property of those two candidates.
AGREEMENT_MIN_CANDIDATES = 3


# ───────────────────────────────────────────────────────── from the prompt


_ONE_WORD_RE = re.compile(
    r"\b(?:answer|reply|respond)\s+(?:with\s+)?(?:in\s+)?(?:exactly\s+)?one\s+word\b",
    re.IGNORECASE,
)
_ONLY_NUMBER_RE = re.compile(
    r"\b(?:answer|reply|respond|give)\s+(?:with\s+)?(?:only\s+)?(?:a\s+)?number\b",
    re.IGNORECASE,
)
_YES_NO_RE = re.compile(r"\banswer\s+(?:with\s+)?(?:yes\s+or\s+no|true\s+or\s+false)\b", re.IGNORECASE)
_COUNT_RE = re.compile(
    r"\b(?:give|list|name|provide)\s+(?:me\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.IGNORECASE,
)
_UNIT_RE = re.compile(
    r"\bin\s+(kilometres|kilometers|km|miles|metres|meters|m|celsius|fahrenheit|"
    r"kelvin|seconds|minutes|hours|days|dollars|usd|euros|percent|%)\b",
    re.IGNORECASE,
)
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def constraints_from_prompt(objective: str, *, step: int = 0) -> list[Constraint]:
    """Format and unit requirements the person actually stated.

    Free, exact, and the thing a model eight passes deep quietly drops. A
    stated requirement is not a guess about the answer; it is a fact about
    the request, which is why it can be committed without a candidate pool.
    """
    text = str(objective or "")
    if not text.strip():
        return []
    found: list[Constraint] = []
    source = "prompt"

    if _ONE_WORD_RE.search(text):
        found.append(
            Constraint(kind=ConstraintKind.CARDINALITY, subject="words",
                       args=(1.0,), source=source, step=step)
        )
    if _YES_NO_RE.search(text):
        found.append(
            Constraint(kind=ConstraintKind.ANSWER_TYPE, subject="boolean",
                       source=source, step=step)
        )
    elif _ONLY_NUMBER_RE.search(text):
        found.append(
            Constraint(kind=ConstraintKind.ANSWER_TYPE, subject="number",
                       source=source, step=step)
        )
    match = _COUNT_RE.search(text)
    if match:
        raw = match.group(1).lower()
        count = _WORD_NUMBERS.get(raw)
        if count is None:
            try:
                count = int(raw)
            except ValueError:
                count = None
        if count and 1 <= count <= 50:
            found.append(
                Constraint(kind=ConstraintKind.CARDINALITY, subject="items",
                           args=(float(count),), source=source, step=step)
            )
    unit = _UNIT_RE.search(text)
    if unit:
        found.append(
            Constraint(kind=ConstraintKind.UNIT, subject=unit.group(1).lower(),
                       source=source, step=step)
        )
    return found


# ──────────────────────────────────────────────────── from refuted answers


def constraints_from_refutations(
    refuted: Sequence[str], *, source: str = "verifier", step: int = 0
) -> list[Constraint]:
    """Every REFUTED candidate becomes "the answer is not that".

    This is the strongest kind and the closest analogue of what a
    chain-of-thought token does. The evidence is a deterministic verifier
    saying no — not a score, not a preference. A refuted branch that is
    merely dropped can be re-derived by the next pass, which is one concrete
    way eight passes end up doing one pass's work eight times. Committed, it
    cannot come back.
    """
    out: list[Constraint] = []
    seen: set[str] = set()
    for candidate in refuted:
        normalised = _normalize(candidate)
        # A whole refuted essay is not a usable exclusion; the short,
        # answer-shaped ones are. A long EXCLUDES would almost never match
        # and would narrow nothing while costing context.
        if not normalised or len(normalised) > 120 or normalised in seen:
            continue
        seen.add(normalised)
        out.append(
            Constraint(kind=ConstraintKind.EXCLUDES, subject=candidate.strip()[:120],
                       source=source, step=step)
        )
    return out


# ──────────────────────────────────────────── from independent agreement


def constraints_from_agreement(
    candidates: Sequence[str], *, source: str = "agreement", step: int = 0
) -> list[Constraint]:
    """Structural properties EVERY independent candidate already has.

    Unanimity is the bar. A property shared by a majority is a property of
    the majority, and promoting it is a consensus mechanism — the thing this
    codebase measured collapsing into a local basin. Unanimity across
    independently sampled candidates is instead evidence about the problem:
    if every sample produced a number, the answer is a number, and saying so
    out loud stops pass six from producing a paragraph.
    """
    usable = [str(item).strip() for item in candidates if str(item).strip()]
    if len(usable) < AGREEMENT_MIN_CANDIDATES:
        return []

    out: list[Constraint] = []

    for answer_type in ("number", "boolean", "date"):
        probe = Constraint(kind=ConstraintKind.ANSWER_TYPE, subject=answer_type)
        if all(probe.check(candidate) is True for candidate in usable):
            out.append(
                Constraint(kind=ConstraintKind.ANSWER_TYPE, subject=answer_type,
                           source=source, step=step)
            )
            break

    # A numeric range every sample already lands in. Widened to the observed
    # hull, not the mean: this rules out the far field, it does not pick a
    # winner, and a constraint that picks the winner is just voting.
    numeric = [_numbers_in(candidate) for candidate in usable]
    if all(values for values in numeric):
        lows = [min(values) for values in numeric]
        highs = [max(values) for values in numeric]
        low, high = min(lows), max(highs)
        if high > low:
            out.append(
                Constraint(kind=ConstraintKind.NUMERIC_RANGE, subject="value",
                           args=(low, high), source=source, step=step)
            )

    counts = {len(_list_items(candidate)) for candidate in usable}
    if len(counts) == 1:
        count = counts.pop()
        if count >= 2:
            out.append(
                Constraint(kind=ConstraintKind.CARDINALITY, subject="items",
                           args=(float(count),), source=source, step=step)
            )
    return out


# ─────────────────────────────────────────────────────────── the gatherer


def propose_constraints(
    *,
    objective: str = "",
    candidates: Sequence[str] = (),
    refuted: Sequence[str] = (),
    step: int = 0,
) -> list[Constraint]:
    """All admissible constraints for this step, strongest evidence first.

    Ordering matters: eliminations are committed before agreements, so a
    range derived from candidates that include a refuted one cannot lock in
    around it.
    """
    proposals: list[Constraint] = []
    proposals.extend(constraints_from_refutations(refuted, step=step))
    proposals.extend(constraints_from_prompt(objective, step=step))
    proposals.extend(constraints_from_agreement(candidates, step=step))
    return proposals


def evidence_summary(proposals: Sequence[Constraint]) -> dict[str, Any]:
    """What kinds of evidence this step actually had.

    A step whose only proposals are STATED ones learned nothing from the
    episode; it re-read the prompt. Worth knowing, and worth not confusing
    with a step that eliminated three candidates.
    """
    by_source: dict[str, int] = {}
    for constraint in proposals:
        by_source[constraint.source] = by_source.get(constraint.source, 0) + 1
    return {
        "proposed": len(proposals),
        "by_source": by_source,
        "has_elimination_evidence": by_source.get("verifier", 0) > 0,
        "has_agreement_evidence": by_source.get("agreement", 0) > 0,
        "prompt_only": set(by_source) <= {"prompt"} and bool(by_source),
    }


__all__ = [
    "AGREEMENT_MIN_CANDIDATES",
    "constraints_from_agreement",
    "constraints_from_prompt",
    "constraints_from_refutations",
    "evidence_summary",
    "propose_constraints",
]
