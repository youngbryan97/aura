"""core/epistemics/contradiction_detector.py — Contradiction Detector.

Found by the eval arena's truthfulness probe on 2026-08-04, the first time that
probe ran anything real. Two defects, in opposite directions:

1. A CLAIM CONTRADICTED ITSELF. The rule was::

       diff = words1.symmetric_difference(words2)
       if diff.issubset(negation_words | {"optimized"}):
           return True

   For two identical texts ``diff`` is empty, and the empty set is a subset of
   everything, so ``are_contradictory(x, x)`` returned True for any claim of
   three or more words. ``detect_conflicts`` walks every pair in the graph, so
   every duplicated belief was reported as a logical conflict against itself.

2. A REAL CONTRADICTION WAS MISSED. "aura runs locally" vs "aura does not run
   locally" returned False: the difference is ``{runs, does, not, run}``, and
   ``runs``/``run``/``does`` are not in the negation list, so the subset test
   failed. Any negation carrying an auxiliary verb or a conjugation change —
   which in English is most of them — escaped.

The list-of-permitted-differences approach cannot be repaired by extending the
list; "unoptimized", "high", "low" and "latency" were already in it, which is
how a vocabulary of one past example became the rule. Contradiction is
structural: the same subject and predicate, asserted on one side and negated on
the other. That is what this measures now.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.epistemics.claim_graph import ClaimGraph

logger = logging.getLogger("Aura.ContradictionDetector")

#: Words that flip the polarity of a claim.
NEGATORS = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "nothing",
        "nobody",
        "cannot",
        "cant",
        "wont",
        "dont",
        "doesnt",
        "didnt",
        "isnt",
        "arent",
        "wasnt",
        "werent",
        "hasnt",
        "havent",
        "untrue",
    }
)
# "false", "incorrect" and "failed" were negators here. They are predicates —
# "the migration failed" is a claim ABOUT the migration, not a negation of one —
# and treating them as polarity markers stripped them out of the content, so
# "the migration succeeded" and "the migration failed" stopped overlapping
# enough to compare. They live in the antonym table below instead.

#: Auxiliaries and function words that carry no claim content. They must not
#: count as a difference — "does not run" versus "runs" differs by an auxiliary,
#: and that difference is grammar, not disagreement.
_FUNCTION_WORDS = frozenset(
    """
    a an the is are was were be been being am do does did doing has have had
    having will would shall should can could may might must of to in on at by
    for from with as that this these those it its there here and or but
    """.split()
)

#: Antonym pairs that contradict without a negator present.
_OPPOSITE_PAIRS: tuple[tuple[str, str], ...] = (
    ("increase", "decrease"),
    ("higher", "lower"),
    ("high", "low"),
    ("rise", "fall"),
    ("rising", "falling"),
    ("up", "down"),
    ("more", "less"),
    ("faster", "slower"),
    ("fast", "slow"),
    ("optimized", "unoptimized"),
    ("safe", "unsafe"),
    ("stable", "unstable"),
    ("enabled", "disabled"),
    ("present", "absent"),
    ("true", "false"),
    ("success", "failure"),
    ("succeeded", "failed"),
    ("allowed", "denied"),
    ("correct", "incorrect"),
    ("valid", "invalid"),
    ("complete", "incomplete"),
    ("open", "closed"),
    ("warm", "cold"),
    ("local", "remote"),
)

#: How much of the surrounding claim must match before a polarity flip counts
#: as disagreement rather than two different statements that happen to contain
#: a negator. Jaccard over stemmed content words.
MIN_CONTENT_OVERLAP = 0.6

#: Below this many shared content words the texts are not about the same thing,
#: whatever the ratio says — two-word claims overlap trivially.
MIN_SHARED_CONTENT = 1

_WORD_RE = re.compile(r"[a-z0-9]+")


def _stem_variants(word: str) -> set[str]:
    """Every stem this word could appear as.

    ``_stem`` is asymmetric — "increases" reduces to "increas" while "increase"
    keeps its trailing e — so a lookup built from one surface form misses the
    other. The antonym table is written in dictionary form and has to match
    text written in any form, so it is expanded rather than the text mangled.
    """
    base = str(word or "").lower()
    forms = {base, _stem(base)}
    for suffix in ("s", "es", "ed", "ing", "d"):
        forms.add(_stem(base + suffix))
    forms.discard("")
    return forms


def _stem(word: str) -> str:
    """Enough morphology to make "runs", "run" and "running" the same claim.

    Deliberately shallow: over-stemming collides unrelated words, which turns a
    contradiction detector into a random one.
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) - len(suffix) >= 3 and word.endswith(suffix):
            candidate = word[: -len(suffix)]
            # "optimized" -> "optimiz" is fine as a key; "bus" -> "bu" is not.
            if suffix == "s" and word.endswith("ss"):
                return word
            return candidate
    return word


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(str(text or "").lower().replace("-", " "))


def _split_claim(text: str) -> tuple[set[str], bool]:
    """The claim's content words (stemmed) and whether it is negated."""
    words = _tokens(text)
    negated = any(word in NEGATORS for word in words)
    content = {
        _stem(word)
        for word in words
        if word not in NEGATORS and word not in _FUNCTION_WORDS
    }
    return content, negated


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / float(len(union)) if union else 0.0


class ContradictionDetector:
    """Scans and flags logically conflicting claims in the belief ecosystem."""

    @staticmethod
    def are_contradictory(left: str, right: str) -> bool:
        """True when two claim texts assert and deny the same thing.

        Requires BOTH that the claims are about the same thing and that their
        polarity differs. A claim never contradicts itself, and two unrelated
        statements never contradict each other however they are worded.
        """
        text_left = str(left or "").strip().lower()
        text_right = str(right or "").strip().lower()
        if not text_left or not text_right:
            return False

        # A claim does not disagree with itself. This is the case that made
        # every duplicated belief in the graph a conflict.
        if text_left == text_right:
            return False

        content_left, negated_left = _split_claim(text_left)
        content_right, negated_right = _split_claim(text_right)
        if not content_left or not content_right:
            return False

        shared = content_left & content_right
        if len(shared) < MIN_SHARED_CONTENT:
            return False
        same_subject = _overlap(content_left, content_right) >= MIN_CONTENT_OVERLAP

        # Polarity flip on the same claim: one asserts, the other denies.
        if same_subject and (negated_left != negated_right):
            return True

        # Antonyms contradict without any negator: "latency is high" against
        # "latency is low". The rest of the claim still has to match, or every
        # sentence containing "up" fights every sentence containing "down".
        for first, second in _OPPOSITE_PAIRS:
            forms_a = _stem_variants(first)
            forms_b = _stem_variants(second)
            crosses = (content_left & forms_a and content_right & forms_b) or (
                content_left & forms_b and content_right & forms_a
            )
            if not crosses:
                continue
            polar = forms_a | forms_b
            rest_left = content_left - polar
            rest_right = content_right - polar
            if not rest_left or not rest_right:
                continue
            if _overlap(rest_left, rest_right) >= MIN_CONTENT_OVERLAP:
                return negated_left == negated_right
        return False

    @staticmethod
    def detect_conflicts(graph: ClaimGraph) -> list[tuple[str, str, str]]:
        """Every pair of claims in the graph that assert and deny the same thing.

        Returns:
            List of (claim_id1, claim_id2, description_of_conflict)
        """
        conflicts = []
        nodes = list(graph.nodes.values())

        for i, node1 in enumerate(nodes):
            for node2 in nodes[i + 1 :]:
                if not ContradictionDetector.are_contradictory(node1.text, node2.text):
                    continue
                reason = (
                    f"Claim '{node2.claim_id}' logically negates claim '{node1.claim_id}'"
                )
                logger.warning("Contradiction detected: %s", reason)
                conflicts.append((node1.claim_id, node2.claim_id, reason))

        return conflicts


def _extract_subject(text: str) -> str:
    """Rough subject extraction, kept for callers that import it."""
    words = str(text or "").lower().split()
    if len(words) > 2:
        return "".join(words[2:])
    return str(text or "").lower()


def _normalized_words(text: str) -> set[str]:
    return set(_tokens(text))


__all__ = ["ContradictionDetector", "NEGATORS"]
