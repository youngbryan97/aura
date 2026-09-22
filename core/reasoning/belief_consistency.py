"""Logical consistency checking over Aura's beliefs.

Beliefs are stored as natural-language claims (``core/epistemics/belief_revision.py``). This
module is the bridge that lets the natural-deduction prover act on them: it encodes
each claim as a propositional literal (an atom, or its negation when the claim is
phrased negatively) and uses :mod:`core.reasoning.natural_deduction` to detect when
Aura simultaneously, confidently holds a proposition **and** its negation — a
genuine logical inconsistency in her self-model.

The encoding is deliberately conservative: it only pairs a claim with its *explicit*
negation (same core proposition, opposite polarity), so it does not hallucinate
semantic contradictions. What it catches is sound — "X" together with "not X".

Detected contradictions are surfaced to governance via
:mod:`core.reasoning.deduction_governance`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.reasoning.natural_deduction import (
    Atom,
    Formula,
    Implies,
    Not,
    find_contradiction,
    is_consistent,
)

# Negation cues that flip a claim's polarity.
_NEG_PATTERNS = [
    r"\bnot\b", r"\bno longer\b", r"\bnever\b", r"\bisn'?t\b", r"\baren'?t\b",
    r"\bwasn'?t\b", r"\bweren'?t\b", r"\bdon'?t\b", r"\bdoesn'?t\b", r"\bdidn'?t\b",
    r"\bcannot\b", r"\bcan'?t\b", r"\bwon'?t\b", r"\bfalse that\b", r"\buntrue\b",
    r"\bno\b",
]
_NEG_RE = re.compile("|".join(_NEG_PATTERNS), re.IGNORECASE)
# Filler words dropped when forming the core proposition key (so polarity is the
# only difference between "I am sovereign" and "I am not sovereign").
_FILLER_RE = re.compile(r"\b(a|an|the|is|are|was|were|am|be|being|been|do|does|did)\b", re.IGNORECASE)


@dataclass(frozen=True)
class EncodedBelief:
    formula: Formula
    core_key: str
    negated: bool
    source: str


@dataclass
class ConsistencyReport:
    consistent: bool
    contradictions: list[tuple[str, str]] = field(default_factory=list)  # (belief, opposing belief)
    minimal_core: list[str] = field(default_factory=list)
    checked: int = 0
    # True when the trusted proof kernel independently certified the
    # inconsistency (a checked proof of Γcore ⊢ ⊥), not just the search.
    kernel_certified: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "consistent": self.consistent,
            "contradictions": self.contradictions,
            "minimal_core": self.minimal_core,
            "checked": self.checked,
            "kernel_certified": self.kernel_certified,
        }


# Implication cues: "if X then Y", "X implies/means/leads to/entails Y".
_IF_THEN_RE = re.compile(r"^\s*if\b(?P<ante>.+?)\bthen\b(?P<cons>.+)$", re.IGNORECASE)
_X_IMPLIES_Y_RE = re.compile(
    r"^(?P<ante>.+?)\b(?:implies|means that|means|leads to|entails|requires)\b(?P<cons>.+)$",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower()).strip(" .!?")


def _stem_word(w: str) -> str:
    """Light, conservative stemmer so morphological variants unify into one atom.

    "rain"/"rains"/"raining" → "rain", "wet" → "wet". Not linguistically perfect —
    it only needs to be *consistent* so the same concept maps to the same atom and
    a logical link ("if it rains … the ground is wet" ↔ "it is raining") survives.
    """
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 5 and w.endswith("ied"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 3 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _literal(text: str) -> tuple[Formula, str, bool]:
    """Encode a clause as an atom (or its negation), returning (formula, key, negated)."""
    norm = _normalize(text)
    negated = bool(_NEG_RE.search(norm))
    core = _NEG_RE.sub(" ", norm)
    core = _FILLER_RE.sub(" ", core)
    core = " ".join(_stem_word(w) for w in core.split())
    atom = Atom(core or norm or "∅")
    return (Not(atom) if negated else atom, atom.name, negated)


def encode_belief(content: str) -> EncodedBelief:
    """Encode a natural-language claim as a propositional formula.

    Implication-shaped claims ("if X then Y", "X implies Y") become ``Implies`` so
    the prover can chain them (modus ponens) — this is what lets it catch
    multi-step contradictions like {X, X→Y, ¬Y}, not just direct X ∧ ¬X.
    Otherwise a negation cue flips polarity and the cue + grammatical filler are
    stripped to form the core key, so an affirmation and its explicit negation map
    to the same atom with opposite polarity.
    """
    raw = str(content or "")
    m = _IF_THEN_RE.match(raw) or _X_IMPLIES_Y_RE.match(raw)
    if m and m.group("ante").strip() and m.group("cons").strip():
        ante_f, ante_k, _ = _literal(m.group("ante"))
        cons_f, cons_k, _ = _literal(m.group("cons"))
        if ante_k != cons_k:
            return EncodedBelief(
                formula=Implies(ante_f, cons_f),
                core_key=f"{ante_k}→{cons_k}",
                negated=False,
                source=raw,
            )
    formula, key, negated = _literal(raw)
    return EncodedBelief(formula=formula, core_key=key, negated=negated, source=raw)


def check_beliefs(
    beliefs: list[tuple[str, float]],
    *,
    min_confidence: float = 0.6,
) -> ConsistencyReport:
    """Check a set of (claim, confidence) beliefs for logical inconsistency.

    Only beliefs at or above ``min_confidence`` participate — a low-confidence
    stray claim should not be treated as a firm contradiction.
    """
    encoded: list[EncodedBelief] = [
        encode_belief(content)
        for content, conf in beliefs
        if content and float(conf) >= min_confidence
    ]
    formulas = [e.formula for e in encoded]
    if is_consistent(formulas):
        return ConsistencyReport(consistent=True, checked=len(encoded))

    core = find_contradiction(formulas) or []
    core_set = set(core)
    # Kernel-certify the inconsistency: a checked proof that the minimal core
    # entails ⊥. The claim "Aura's beliefs are contradictory" is consequential
    # enough to demand a certificate, not just a search verdict.
    kernel_certified = False
    if core:
        try:
            from core.reasoning.natural_deduction import Bot
            from core.reasoning.proof_kernel import prove_certified

            kernel_certified = prove_certified(core, Bot()).verified
        # not a failure: a kernel that cannot certify has not certified, and False is the
        # refusing direction.
        except (ValueError, RuntimeError, TypeError, AttributeError, ImportError):
            kernel_certified = False
    # Source beliefs whose formula is in the minimal unsatisfiable core — covers
    # both direct X∧¬X pairs and chained {X, X→Y, ¬Y} modus-ponens conflicts.
    core_sources = [e.source for e in encoded if e.formula in core_set]

    # Direct opposing pairs (atom vs. its negation) for a clean (affirm, deny).
    by_key: dict[str, dict[bool, str]] = {}
    for e in encoded:
        if e.formula in core_set and isinstance(e.formula, (Atom, Not)):
            by_key.setdefault(e.core_key, {})[e.negated] = e.source
    contradictions: list[tuple[str, str]] = []
    for sides in by_key.values():
        if True in sides and False in sides:
            contradictions.append((sides[False], sides[True]))
    # If the conflict is purely chained (no direct pair), surface the core sources.
    if not contradictions and len(core_sources) >= 2:
        contradictions.append((core_sources[0], " + ".join(core_sources[1:])))

    return ConsistencyReport(
        consistent=False,
        contradictions=contradictions,
        minimal_core=core_sources or [str(f) for f in core],
        checked=len(encoded),
        kernel_certified=kernel_certified,
    )
