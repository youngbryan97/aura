"""The commitment ratchet: token-free irreversible narrowing.

WHY THIS EXISTS
───────────────
Latent recurrence in this codebase has failed, repeatedly and honestly:
``cos(pass1, pass2) = 0.9994``, the frozen loop refuted, vanilla beating
every latent arm, and the eventual verdict that "recurrence itself is the
damage". Those are not tuning failures. They are the predicted behaviour of
iterating a fixed operator on its own output: that is a contraction, it
reaches its fixed point in one step, and depth 8 therefore computes what
depth 1 computed. More depth on a contraction is a longer identity.

The question that follows is what chain-of-thought is actually buying, since
it demonstrably buys something. The usual answer — "more computation" — is
wrong, because the latent loop is also more computation and it buys nothing.

The answer this module is built on:

    A chain of thought is a sequence of IRREVERSIBLE COMMITMENTS.

Emitting a token collapses a distribution. Every later pass conditions on a
decision that can no longer be unmade, and the hypothesis space monotonically
shrinks. That is an information-theoretic act, not an arithmetic one. Latent
recurrence bought computation without commitment: it carries a superposition
forward and smooths it, and averaging a superposition is exactly the
contraction we measured.

So the missing organ is a token-free commitment device. The token-free
analogue of writing a token is not a vector — a vector can be blended, and
anything that can be blended can be un-decided. It is a CONSTRAINT: a
discrete proposition the final answer must satisfy.

A constraint has the four properties a latent state lacks:

  discrete       committing is a decision, not a blend;
  irreversible   within an episode it cannot be retracted, so later passes
                 cannot drift back through it;
  narrowing      it strictly removes admissible answers, and the amount is
                 MEASURED against a real candidate pool, never asserted;
  checkable      a deterministic function decides whether a text satisfies
                 it, so the claim "this helped" is falsifiable.

THE STRUCTURAL GUARANTEE
────────────────────────
Because pass N+1 is conditioned on a constraint set that pass N did not
have, it is solving a strictly different problem. ``cos(pass_n, pass_n+1) ≈ 1``
cannot be the normal case any more — and where a step commits nothing, the
ratchet says so and the step is CANCELLED rather than spent. That is the
direct structural answer to the 0.9994 result: an identity step is not
performed, and a performed step is not an identity.

WHAT THIS MODULE DOES NOT CLAIM
───────────────────────────────
It does not claim a reasoning gain. It makes one measurable. Narrowing is
only reported when it has been measured against a candidate pool; a
constraint committed without a pool is marked ``unmeasured`` and cannot count
toward any authority decision. The falsification harness that can refute the
whole idea lives in ``commitment_ratchet_ablations.py``, and its shuffle arm
is designed to kill this module if the constraints are not doing the work.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

RATCHET_SCHEMA = "aura.rlc.commitment_ratchet.v2"
COMMIT_SCHEMA = "aura.rlc.commitment_ratchet.commit.v1"

#: A ratchet with unbounded teeth is a prompt-stuffing device. Past this the
#: conditioning block costs more context than the narrowing is worth.
MAX_TEETH = 24

#: A constraint that removes nothing from the live pool has not narrowed
#: anything, whatever it asserts about the world. Committing it would grow
#: the conditioning block for free — the exact failure mode of "add more
#: instructions and hope".
MIN_MEASURED_NARROWING = 1e-9


class RatchetRefusal(RuntimeError):  # noqa: N818 - public compatibility name
    """A commit was refused. The ratchet only turns one way."""


class ConstraintKind(StrEnum):
    """The checkable vocabulary.

    Deliberately small. Every kind here has a deterministic checker below,
    because a constraint nothing can check is a sentence, and a sentence in
    the prompt is what we already had.
    """

    #: The answer's surface form: number / boolean / list / name / date.
    ANSWER_TYPE = "answer_type"
    #: A term that must appear in the answer.
    MUST_MENTION = "must_mention"
    #: A value the answer must not be or contain — a killed candidate.
    EXCLUDES = "excludes"
    #: The answer's numeric value lies within [low, high].
    NUMERIC_RANGE = "numeric_range"
    #: The answer carries a unit.
    UNIT = "unit"
    #: The answer enumerates exactly N items.
    CARDINALITY = "cardinality"
    #: One term precedes another in the answer.
    ORDERING = "ordering"
    #: The answer, normalised, equals this value.
    MUST_EQUAL = "must_equal"


_ANSWER_TYPES = frozenset({"number", "boolean", "list", "name", "date"})

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
_BOOLEAN_RE = re.compile(r"\b(yes|no|true|false)\b", re.IGNORECASE)
_DATE_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2})\b",
    re.IGNORECASE,
)
_LIST_SPLIT_RE = re.compile(r"(?:\n\s*[-*•]\s*|\n\s*\d+[.)]\s*|,\s+|;\s+)")


def _normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _numbers_in(text: str) -> list[float]:
    values: list[float] = []
    for token in _NUMBER_RE.findall(text):
        try:
            values.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return values


def _list_items(text: str) -> list[str]:
    parts = [part.strip(" \t-*•") for part in _LIST_SPLIT_RE.split(text)]
    return [part for part in parts if part]


@dataclass(frozen=True)
class Constraint:
    """One committed proposition about the answer.

    ``subject`` is what the constraint is about (a term, a unit, the literal
    string "value"); ``args`` carries the numbers a kind needs. Both are part
    of the identity, so two constraints that say different things about the
    same subject are detectably different rather than silently merged.
    """

    kind: ConstraintKind
    subject: str = ""
    args: tuple[float, ...] = ()
    #: Why this was committed — a latent step index, a verifier name, a
    #: prompt span. Provenance is not decoration: the ablation harness
    #: permutes it, and a gain that survives permutation was never causal.
    source: str = ""
    step: int = 0

    @property
    def constraint_id(self) -> str:
        payload = json.dumps(
            {
                "kind": self.kind.value,
                "subject": _normalize(self.subject),
                "args": [round(float(value), 9) for value in self.args],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    # ── the checker ──────────────────────────────────────────────────────

    def check(self, text: str) -> bool | None:
        """Does this text satisfy the constraint?

        Returns None when the constraint is not decidable on this text —
        which is different from False, and must stay different. Treating
        "I could not tell" as "it failed" is how a filter becomes a
        shredder.
        """
        body = str(text or "").strip()
        if not body:
            return None
        try:
            return self._check(body)
        except (ValueError, TypeError, IndexError):
            return None

    def _check(self, body: str) -> bool | None:
        lowered = _normalize(body)
        if self.kind is ConstraintKind.ANSWER_TYPE:
            return _check_answer_type(self.subject, body, lowered)
        if self.kind is ConstraintKind.MUST_MENTION:
            term = _normalize(self.subject)
            return bool(term) and term in lowered
        if self.kind is ConstraintKind.EXCLUDES:
            term = _normalize(self.subject)
            return bool(term) and term not in lowered
        if self.kind is ConstraintKind.NUMERIC_RANGE:
            if len(self.args) != 2:
                return None
            low, high = float(self.args[0]), float(self.args[1])
            values = _numbers_in(body)
            if not values:
                return None
            return any(low <= value <= high for value in values)
        if self.kind is ConstraintKind.UNIT:
            unit = _normalize(self.subject)
            return bool(unit) and unit in lowered
        if self.kind is ConstraintKind.CARDINALITY:
            if not self.args:
                return None
            items = _list_items(body)
            if not items:
                return None
            return len(items) == int(self.args[0])
        if self.kind is ConstraintKind.ORDERING:
            first, _, second = self.subject.partition("<")
            first, second = _normalize(first), _normalize(second)
            if not first or not second:
                return None
            if first not in lowered or second not in lowered:
                return None
            return lowered.index(first) < lowered.index(second)
        if self.kind is ConstraintKind.MUST_EQUAL:
            return _normalize(self.subject) == lowered
        return None

    def render(self) -> str:
        """The constraint as one line a model can act on."""
        if self.kind is ConstraintKind.ANSWER_TYPE:
            return f"The answer is a {self.subject}."
        if self.kind is ConstraintKind.MUST_MENTION:
            return f"The answer must refer to {self.subject!r}."
        if self.kind is ConstraintKind.EXCLUDES:
            return f"The answer is NOT {self.subject!r} — that was ruled out."
        if self.kind is ConstraintKind.NUMERIC_RANGE:
            low, high = self.args[0], self.args[1]
            return f"The value lies between {low:g} and {high:g}."
        if self.kind is ConstraintKind.UNIT:
            return f"The answer is expressed in {self.subject}."
        if self.kind is ConstraintKind.CARDINALITY:
            return f"The answer names exactly {int(self.args[0])} item(s)."
        if self.kind is ConstraintKind.ORDERING:
            first, _, second = self.subject.partition("<")
            return f"{first.strip()} comes before {second.strip()}."
        if self.kind is ConstraintKind.MUST_EQUAL:
            return f"The answer is exactly {self.subject!r}."
        return f"{self.kind.value}: {self.subject}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.constraint_id,
            "kind": self.kind.value,
            "subject": self.subject[:200],
            "args": [float(value) for value in self.args],
            "source": self.source[:120],
            "step": int(self.step),
            "text": self.render(),
        }


def _check_answer_type(subject: str, body: str, lowered: str) -> bool | None:
    wanted = _normalize(subject)
    if wanted not in _ANSWER_TYPES:
        return None
    if wanted == "number":
        return bool(_numbers_in(body))
    if wanted == "boolean":
        return bool(_BOOLEAN_RE.search(body))
    if wanted == "date":
        return bool(_DATE_RE.search(body))
    if wanted == "list":
        return len(_list_items(body)) >= 2
    if wanted == "name":
        # A name is a short noun phrase with no sentence punctuation and at
        # least one capitalised token in the original casing.
        if len(lowered.split()) > 8 or "." in body.strip()[:-1]:
            return False
        return any(token[:1].isupper() for token in body.split())
    return None


# ────────────────────────────────────── contradiction between constraints


def contradiction_between(first: Constraint, second: Constraint) -> str:
    """Why these two cannot both hold, or "" when they can.

    This is the ratchet's integrity property and the reason it is a ratchet
    rather than a pile. A device that can commit ¬P after committing P has
    not narrowed anything — it has accumulated noise, and the "monotone
    shrinking" story is false. Every commit is checked against every tooth
    already cut.
    """
    if first.kind is not second.kind:
        return _cross_kind_contradiction(first, second)

    subject_a, subject_b = _normalize(first.subject), _normalize(second.subject)
    if first.kind is ConstraintKind.ANSWER_TYPE and subject_a != subject_b:
        return f"answer cannot be both a {subject_a} and a {subject_b}"
    if first.kind is ConstraintKind.MUST_EQUAL and subject_a != subject_b:
        return f"answer cannot equal both {subject_a!r} and {subject_b!r}"
    if first.kind is ConstraintKind.CARDINALITY and first.args != second.args:
        return "answer cannot have two different item counts"
    if first.kind is ConstraintKind.UNIT and subject_a != subject_b:
        return f"answer cannot be in both {subject_a} and {subject_b}"
    if first.kind is ConstraintKind.NUMERIC_RANGE:
        if len(first.args) == 2 and len(second.args) == 2:
            low = max(first.args[0], second.args[0])
            high = min(first.args[1], second.args[1])
            if low > high:
                return "numeric ranges are disjoint"
    if first.kind is ConstraintKind.ORDERING:
        a1, _, a2 = subject_a.partition("<")
        b1, _, b2 = subject_b.partition("<")
        if a1.strip() == b2.strip() and a2.strip() == b1.strip():
            return "orderings are mutually exclusive"
    return ""


def _cross_kind_contradiction(first: Constraint, second: Constraint) -> str:
    pair = {first.kind: first, second.kind: second}
    if ConstraintKind.MUST_MENTION in pair and ConstraintKind.EXCLUDES in pair:
        mention = _normalize(pair[ConstraintKind.MUST_MENTION].subject)
        excluded = _normalize(pair[ConstraintKind.EXCLUDES].subject)
        if mention and mention == excluded:
            return f"{mention!r} is both required and excluded"
    if ConstraintKind.MUST_EQUAL in pair and ConstraintKind.EXCLUDES in pair:
        equals = _normalize(pair[ConstraintKind.MUST_EQUAL].subject)
        excluded = _normalize(pair[ConstraintKind.EXCLUDES].subject)
        if equals and equals == excluded:
            return f"answer cannot equal an excluded value {equals!r}"
    if ConstraintKind.MUST_EQUAL in pair and ConstraintKind.NUMERIC_RANGE in pair:
        equals = pair[ConstraintKind.MUST_EQUAL].subject
        bounds = pair[ConstraintKind.NUMERIC_RANGE].args
        values = _numbers_in(equals)
        if values and len(bounds) == 2:
            if not any(bounds[0] <= value <= bounds[1] for value in values):
                return "the required value lies outside the required range"
    return ""


# ───────────────────────────────────────────────────── the ratchet itself


@dataclass(frozen=True)
class CommitReceipt:
    """What one turn of the ratchet actually did."""

    committed: bool
    constraint: Constraint | None
    reason: str
    #: Fraction of the live candidate pool this constraint eliminated.
    #: ``None`` means NOT MEASURED — there was no pool — and must never be
    #: read as zero or as "fine".
    narrowing: float | None
    pool_before: int
    pool_after: int
    at: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": COMMIT_SCHEMA,
            "committed": self.committed,
            "reason": self.reason,
            "constraint": self.constraint.to_dict() if self.constraint else None,
            "narrowing": self.narrowing,
            "narrowing_measured": self.narrowing is not None,
            "pool_before": self.pool_before,
            "pool_after": self.pool_after,
            "at": round(self.at, 3),
        }


#: Sources whose constraints narrow the FUTURE rather than the present pool.
#:
#: A requirement the person stated — "answer in one word", "in kilometres" —
#: may eliminate none of the candidates already drawn, because those were
#: drawn while the requirement was still in the prompt. Its value is that it
#: keeps being true on pass six, when a model several passes deep has
#: quietly stopped honouring it. Refusing it for narrowing nothing would
#: discard the cheapest and most reliable constraint there is.
#:
#: These commit with narrowing recorded but not enforced, so they never
#: contribute to a measured-narrowing claim.
_FUTURE_NARROWING_SOURCES = frozenset({"prompt"})


def _is_stated_requirement(constraint: Constraint) -> bool:
    return constraint.source in _FUTURE_NARROWING_SOURCES


class CommitmentRatchet:
    """Accumulates irreversible, consistent, measured narrowings.

    One instance per episode. It holds the committed constraint set and, when
    the caller supplies one, the live candidate pool those constraints are
    measured against.
    """

    def __init__(
        self,
        candidates: Iterable[str] | None = None,
        *,
        max_teeth: int = MAX_TEETH,
    ) -> None:
        self._teeth: list[Constraint] = []
        self._receipts: list[CommitReceipt] = []
        self._pool: list[str] = [str(item) for item in (candidates or []) if str(item).strip()]
        self._initial_pool = len(self._pool)
        self._max_teeth = max(1, int(max_teeth))
        self._sealed = False

    # ── state ────────────────────────────────────────────────────────────

    @property
    def teeth(self) -> tuple[Constraint, ...]:
        return tuple(self._teeth)

    @property
    def turns(self) -> int:
        return len(self._teeth)

    @property
    def pool(self) -> tuple[str, ...]:
        return tuple(self._pool)

    @property
    def sealed(self) -> bool:
        return self._sealed

    def seal(self) -> None:
        """No further turns. The episode's commitments are final."""
        self._sealed = True

    @property
    def measured_narrowing(self) -> float:
        """Total measured fraction of the initial pool eliminated.

        Reported only over pool-measured commits. A ratchet that never had a
        pool reports 0.0 here AND ``measured_commits == 0``; the two must be
        read together, which is why both are on the receipt.
        """
        if not self._initial_pool:
            return 0.0
        return 1.0 - (len(self._pool) / float(self._initial_pool))

    @property
    def measured_commits(self) -> int:
        """Commits whose narrowing was MEASURED against the live pool.

        A stated requirement is excluded even when a pool was present: it
        was committed for its effect on later draws, and counting it here
        would let restating the prompt inflate a measured-narrowing claim.
        """
        return sum(
            1
            for receipt in self._receipts
            if receipt.committed
            and receipt.narrowing is not None
            and receipt.narrowing > MIN_MEASURED_NARROWING
        )

    # ── the turn ─────────────────────────────────────────────────────────

    def consistency_of(self, constraint: Constraint) -> str:
        """Why this constraint cannot join the set, or "" when it can."""
        for tooth in self._teeth:
            reason = contradiction_between(tooth, constraint)
            if reason:
                return reason
        return ""

    def would_narrow(self, constraint: Constraint) -> tuple[float | None, int, int]:
        """(narrowing, pool_before, pool_after) without committing anything."""
        before = len(self._pool)
        if not self._pool:
            return None, 0, 0
        survivors = [
            candidate
            for candidate in self._pool
            # None (undecidable) SURVIVES. A constraint that cannot be
            # evaluated against a candidate has not ruled it out, and
            # eliminating what we could not check is how a ratchet becomes
            # a random filter.
            if constraint.check(candidate) is not False
        ]
        after = len(survivors)
        return (before - after) / float(before), before, after

    def commit(self, constraint: Constraint) -> CommitReceipt:
        """Cut one tooth, or refuse and say why.

        Refusals are receipts too. A step whose commit was refused did not
        narrow anything, and the caller is expected to CANCEL that latent
        step rather than spend a forward pass on an unchanged problem —
        which is precisely the identity step that produced cos ≈ 0.9994.
        """
        if self._sealed:
            return self._record(False, None, "ratchet_sealed", None, 0, 0)
        if not isinstance(constraint, Constraint):
            return self._record(False, None, "not_a_constraint", None, 0, 0)
        if len(self._teeth) >= self._max_teeth:
            return self._record(
                False, constraint, "max_teeth_reached", None, len(self._pool), len(self._pool)
            )
        if any(tooth.constraint_id == constraint.constraint_id for tooth in self._teeth):
            return self._record(
                False, constraint, "already_committed", None, len(self._pool), len(self._pool)
            )
        conflict = self.consistency_of(constraint)
        if conflict:
            # THE integrity property. A device that admits ¬P after P is not
            # a ratchet; it is a pile, and a pile does not narrow.
            return self._record(
                False, constraint, f"contradicts_committed_set: {conflict}",
                None, len(self._pool), len(self._pool),
            )

        narrowing, before, after = self.would_narrow(constraint)
        if (
            narrowing is not None
            and narrowing <= MIN_MEASURED_NARROWING
            and not _is_stated_requirement(constraint)
        ):
            return self._record(
                False, constraint, "no_measured_narrowing", narrowing, before, after
            )

        if narrowing is not None:
            self._pool = [
                candidate
                for candidate in self._pool
                if constraint.check(candidate) is not False
            ]
        self._teeth.append(constraint)
        return self._record(True, constraint, "committed", narrowing, before, after)

    def _record(
        self,
        committed: bool,
        constraint: Constraint | None,
        reason: str,
        narrowing: float | None,
        before: int,
        after: int,
    ) -> CommitReceipt:
        receipt = CommitReceipt(
            committed=committed,
            constraint=constraint,
            reason=reason,
            narrowing=narrowing,
            pool_before=before,
            pool_after=after,
        )
        self._receipts.append(receipt)
        return receipt

    # ── conditioning the next pass ───────────────────────────────────────

    def conditioning_block(self, *, include_exclusions: bool = False) -> str:
        """What pass N+1 sees that pass N did not.

        This is the whole mechanism. The next pass is not re-running the
        same problem with a smoothed state; it is running a strictly more
        constrained problem. If this block is empty, nothing was committed
        and the next pass would be an identity — the caller should not
        spend it.
        """
        if not self._teeth:
            return ""

        # Exclusions and requirements need OPPOSITE instructions, and the
        # first version of this block gave them the same one.
        #
        # Measured 2026-08-09 on Qwen2.5-1.5B, 24 hard multiplications: the
        # exclusion arm scored WORSE than i.i.d. (-8%), compliance was 0.43,
        # and distinct coverage FELL from 5.46 to 2.58. The block ended with
        # "Work within these ... reopening them repeats work already done" —
        # a convergence instruction. Told to settle, the model settled, and
        # collapsed onto the answers it had been told were wrong.
        #
        # A requirement says "stay inside this". An exclusion says "go
        # somewhere else". Rendering them under one heading with one closing
        # instruction inverted the mechanism.
        # Exclusions are OFF by default because listing them in a prompt was
        # measured and lost: 46.9% vs 48.1% for plain i.i.d. on 160 paired
        # tasks. Naming a wrong answer does not remove it from the
        # distribution — it adds tokens that anchor on it. The correct
        # implementation of "draw from p restricted to A \\ R" is REJECTION:
        # draw unconditioned, discard draws landing in R, redraw. Callers do
        # that with `teeth`; the ablation harness asks for them explicitly
        # because measuring the losing arm is its job.
        exclusions = (
            [tooth for tooth in self._teeth if tooth.kind is ConstraintKind.EXCLUDES]
            if include_exclusions
            else []
        )
        requirements = [
            tooth for tooth in self._teeth if tooth.kind is not ConstraintKind.EXCLUDES
        ]

        lines: list[str] = []
        if exclusions:
            lines.append(
                "[RULED OUT — each of these was CHECKED and found wrong. "
                "Do not produce any of them again; work out a DIFFERENT "
                "answer.]"
            )
            lines.extend(f"- not {tooth.subject}" for tooth in exclusions)
        if requirements:
            if lines:
                lines.append("")
            lines.append(
                "[REQUIRED — established earlier in this episode, not up for "
                "revision]"
            )
            lines.extend(f"- {tooth.render()}" for tooth in requirements)
        if exclusions:
            # Last line, because it is the one nearest the generation point.
            lines.append(
                f"Your answer must differ from all {len(exclusions)} ruled-out "
                "answer(s) above."
            )
        return "\n".join(lines)

    def satisfies(self, text: str) -> dict[str, Any]:
        """Evaluate a candidate answer against every committed constraint."""
        results = []
        violated: list[str] = []
        undecidable = 0
        for tooth in self._teeth:
            verdict = tooth.check(text)
            results.append({**tooth.to_dict(), "satisfied": verdict})
            if verdict is False:
                violated.append(tooth.constraint_id)
            elif verdict is None:
                undecidable += 1
        return {
            "schema": RATCHET_SCHEMA,
            "constraints": len(self._teeth),
            "violated": violated,
            "undecidable": undecidable,
            "satisfied": not violated,
            # Satisfying zero constraints is not satisfying constraints.
            # A caller reading `satisfied` alone on an empty ratchet would
            # be reading "nothing was checked" as "everything passed".
            "checked_any": bool(self._teeth) and undecidable < len(self._teeth),
            "items": results,
        }

    def penalty(self, text: str) -> float:
        """Fraction of decidable constraints this text violates, in [0, 1].

        The differentiable-ish handle for test-time adaptation: an episode's
        fast weights can be pushed to satisfy the episode's own commitments.
        Undecidable constraints are excluded from the denominator rather than
        counted as passes.
        """
        decidable = [
            verdict
            for verdict in (tooth.check(text) for tooth in self._teeth)
            if verdict is not None
        ]
        if not decidable:
            return 0.0
        return sum(1 for verdict in decidable if verdict is False) / len(decidable)

    # ── receipt ──────────────────────────────────────────────────────────

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": RATCHET_SCHEMA,
            "turns": self.turns,
            "sealed": self._sealed,
            "constraints": [tooth.to_dict() for tooth in self._teeth],
            "commits": [receipt.to_dict() for receipt in self._receipts],
            "refusals": [
                receipt.to_dict()
                for receipt in self._receipts
                if not receipt.committed
            ],
            "pool_initial": self._initial_pool,
            "pool_remaining": len(self._pool),
            "measured_narrowing": round(self.measured_narrowing, 6),
            "measured_commits": self.measured_commits,
            # Read these two together or not at all: a large narrowing over
            # zero measured commits is not a large narrowing, it is an
            # unmeasured one.
            "narrowing_is_measured": self.measured_commits > 0,
            "conditioning_chars": len(self.conditioning_block()),
        }
        encoded = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return {
            **body,
            "receipt_sha256": hashlib.sha256(encoded).hexdigest(),
        }


def ratchet_from_receipt(payload: Mapping[str, Any]) -> CommitmentRatchet:
    """Rebuild a ratchet's committed set from its receipt.

    Used by the ablation harness and by any consumer that has to re-derive
    what an episode committed without re-running it.
    """
    ratchet = CommitmentRatchet()
    for row in payload.get("constraints") or ():
        if not isinstance(row, Mapping):
            continue
        try:
            kind = ConstraintKind(str(row.get("kind")))
        except ValueError:
            continue
        ratchet._teeth.append(
            Constraint(
                kind=kind,
                subject=str(row.get("subject") or ""),
                args=tuple(float(value) for value in (row.get("args") or ())),
                source=str(row.get("source") or ""),
                step=int(row.get("step") or 0),
            )
        )
    return ratchet


def constraints_from_texts(
    texts: Sequence[str], *, source: str = "", step: int = 0
) -> list[Constraint]:
    """Parse rendered constraint lines back into Constraints.

    Exists so the ablation harness can round-trip a constraint set through
    text without a second, divergent parser.
    """
    out: list[Constraint] = []
    for line in texts:
        parsed = _parse_rendered(str(line).strip().lstrip("- ").strip())
        if parsed is not None:
            out.append(
                Constraint(
                    kind=parsed[0], subject=parsed[1], args=parsed[2],
                    source=source, step=step,
                )
            )
    return out


_RENDER_PATTERNS: tuple[tuple[re.Pattern[str], ConstraintKind], ...] = (
    (re.compile(r"^The answer is a (\w+)\.$"), ConstraintKind.ANSWER_TYPE),
    (re.compile(r"^The answer must refer to '(.+)'\.$"), ConstraintKind.MUST_MENTION),
    (re.compile(r"^The answer is NOT '(.+)' —"), ConstraintKind.EXCLUDES),
    (re.compile(r"^The answer is expressed in (.+)\.$"), ConstraintKind.UNIT),
    (re.compile(r"^The answer is exactly '(.+)'\.$"), ConstraintKind.MUST_EQUAL),
)


def _parse_rendered(line: str) -> tuple[ConstraintKind, str, tuple[float, ...]] | None:
    for pattern, kind in _RENDER_PATTERNS:
        match = pattern.match(line)
        if match:
            return kind, match.group(1), ()
    match = re.match(r"^The value lies between (-?[\d.]+) and (-?[\d.]+)\.$", line)
    if match:
        return ConstraintKind.NUMERIC_RANGE, "value", (
            float(match.group(1)),
            float(match.group(2)),
        )
    match = re.match(r"^The answer names exactly (\d+) item\(s\)\.$", line)
    if match:
        return ConstraintKind.CARDINALITY, "items", (float(match.group(1)),)
    match = re.match(r"^(.+) comes before (.+)\.$", line)
    if match:
        return ConstraintKind.ORDERING, f"{match.group(1)}<{match.group(2)}", ()
    return None


__all__ = [
    "COMMIT_SCHEMA",
    "MAX_TEETH",
    "MIN_MEASURED_NARROWING",
    "RATCHET_SCHEMA",
    "CommitReceipt",
    "CommitmentRatchet",
    "Constraint",
    "ConstraintKind",
    "RatchetRefusal",
    "constraints_from_texts",
    "contradiction_between",
    "ratchet_from_receipt",
]
