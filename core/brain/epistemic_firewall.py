"""Epistemic firewall: admission control for evidence entering deep recurrence.

Deep latent reasoning amplifies whatever it is seeded with. Internally
consistent recurrence over bad evidence produces confident, well-structured
wrongness — the "hyper-hallucination" failure. Aura already tracks provenance
and verifies outputs; this module makes admission INTO the Recursive Latent
Cortex contingent on epistemic quality rather than mere retrieval relevance.

Before retrieved content may seed a cognitive slot, the firewall:

1. clusters near-duplicate reports so corroboration is counted by
   INDEPENDENT sources, never by repetition;
2. tracks freshness and provenance kind (observed fact vs claim vs
   inference) — observed facts outrank claims, claims outrank inferences;
3. builds a conflict graph between cluster representatives (numeric
   disagreement and polarity disagreement, detection method receipted);
4. resolves conflicts only by defensible rules (provenance rank, then
   decisive freshness) and otherwise refuses BOTH sides;
5. estimates whether retrieval coverage of the objective is sufficient;
6. forces explicit abstention — a caution the episode can feel — when
   contradictions remain unresolved, rather than letting one side in.

Everything here is deterministic, bounded, and receipted. The detectors are
honest about being lexical: the receipt names the method that fired
("numeric_disagreement", "polarity_disagreement"), and nothing claims
semantic understanding the code does not have.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.EpistemicFirewall")

EPISTEMIC_FIREWALL_SCHEMA = "aura.epistemic_firewall.v1"

_WORD_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_NEGATION_MARKERS = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "cannot",
        "can't",
        "won't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "doesn't",
        "don't",
        "didn't",
        "without",
        "false",
    }
)

_VALID_KINDS = ("observed_fact", "claim", "inference")
_KIND_RANK = {kind: rank for rank, kind in enumerate(_VALID_KINDS)}

MAX_ITEMS = 16
MAX_TEXT_CHARS = 800
MAX_ORIGIN_CHARS = 120


@dataclass
class EvidenceItem:
    """One retrieved report with the provenance the firewall reasons over."""

    text: str
    origin: str  # producing source, e.g. "memory_facade.search#2" or a URL
    channel: str = ""  # transport family, e.g. "episodic_memory", "web"
    observed_at: float | None = None  # unix time the content was recorded
    kind: str = "claim"  # observed_fact | claim | inference
    trust: float = 0.5  # prior source trust inside [0, 1]

    def validated(self) -> "EvidenceItem":
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("evidence item requires non-empty text")
        if not isinstance(self.origin, str) or not self.origin.strip():
            raise ValueError("evidence item requires a non-empty origin")
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"evidence kind must be one of {_VALID_KINDS}")
        if not isinstance(self.channel, str):
            raise ValueError("evidence channel must be a string")
        if self.observed_at is not None and (
            isinstance(self.observed_at, bool)
            or not isinstance(self.observed_at, (int, float))
            or not math.isfinite(float(self.observed_at))
            or float(self.observed_at) <= 0.0
        ):
            raise ValueError("evidence observed_at must be a positive unix time or None")
        if (
            isinstance(self.trust, bool)
            or not isinstance(self.trust, (int, float))
            or not math.isfinite(float(self.trust))
            or not 0.0 <= float(self.trust) <= 1.0
        ):
            raise ValueError("evidence trust must be inside [0, 1]")
        return EvidenceItem(
            text=self.text.strip()[:MAX_TEXT_CHARS],
            origin=self.origin.strip()[:MAX_ORIGIN_CHARS],
            channel=self.channel.strip()[:MAX_ORIGIN_CHARS],
            observed_at=None if self.observed_at is None else float(self.observed_at),
            kind=self.kind,
            trust=float(self.trust),
        )


@dataclass
class FirewallVerdict:
    """What may seed thought slots, what may not, and why."""

    admitted: list[dict[str, Any]] = field(default_factory=list)
    refused: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    clusters: list[list[int]] = field(default_factory=list)
    coverage: float = 0.0
    uncovered_terms: list[str] = field(default_factory=list)
    abstain: bool = False
    needs_more_retrieval: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema": EPISTEMIC_FIREWALL_SCHEMA,
            "admitted": [dict(row) for row in self.admitted],
            "refused": [dict(row) for row in self.refused],
            "conflicts": [dict(row) for row in self.conflicts],
            "clusters": [list(cluster) for cluster in self.clusters],
            "coverage": round(self.coverage, 4),
            "uncovered_terms": list(self.uncovered_terms[:8]),
            "abstain": self.abstain,
            "needs_more_retrieval": self.needs_more_retrieval,
            "reasons": sorted(set(self.reasons)),
        }

    def admitted_texts(self) -> list[str]:
        return [str(row["text"]) for row in self.admitted]

    def caution_text(self) -> str:
        """A bounded caution line an episode can be seeded with on abstention."""
        if not self.abstain and not self.needs_more_retrieval:
            return ""
        parts: list[str] = []
        if self.conflicts:
            unresolved = [c for c in self.conflicts if not c.get("resolved_by")]
            if unresolved:
                methods = sorted({str(c["method"]) for c in unresolved})
                parts.append(
                    f"{len(unresolved)} retrieved reports conflict "
                    f"({', '.join(methods)}); treat conclusions as provisional"
                )
        if self.needs_more_retrieval and self.uncovered_terms:
            parts.append(
                "retrieval coverage is thin on: "
                + ", ".join(self.uncovered_terms[:4])
            )
        if not parts:
            parts.append("retrieved evidence failed epistemic admission")
        return ("Evidence check: " + "; ".join(parts))[:400]


def _terms(text: str) -> set[str]:
    return {word.lower() for word in _WORD_RE.findall(text or "")}


# Function words carry no topic. They are excluded from the RELEVANCE test
# only — clustering and the polarity check keep the full term set, because
# negation words are exactly what the polarity detector reads.
_STOPWORDS = frozenset(
    {
        "the", "and", "but", "for", "not", "you", "your", "yours", "our", "ours",
        "his", "her", "hers", "its", "their", "theirs", "this", "that", "these",
        "those", "with", "without", "from", "into", "onto", "over", "under",
        "about", "after", "before", "between", "during", "than", "then", "them",
        "they", "was", "were", "are", "been", "being", "have", "has", "had",
        "does", "did", "doing", "done", "will", "would", "could", "should",
        "shall", "may", "might", "must", "can", "cannot", "one", "two", "any",
        "all", "some", "each", "both", "few", "more", "most", "other", "such",
        "only", "own", "same", "too", "very", "just", "here", "there", "when",
        "where", "which", "who", "whom", "what", "why", "how", "also", "because",
        "while", "since", "until", "upon", "out", "off", "again", "further",
        "once", "now", "get", "got", "make", "made", "take", "taken", "give",
        "given", "come", "came", "went", "say", "said", "see", "seen", "know",
        "known", "want", "like", "need", "use", "used", "show", "tell", "let",
        "put", "way", "thing", "things", "something", "anything", "everything",
        "nothing", "me", "my", "mine", "him", "she", "hers", "it", "we", "us",
    }
)


def _distinctive(text: str) -> set[str]:
    """Topic-bearing terms only."""
    return _terms(text) - _STOPWORDS


def _relevance(objective_terms: set[str], item_terms: set[str]) -> float:
    """How much this item and this objective are actually about each other.

    Two readings, and an item passes on the better of them: the share of the
    objective's distinctive terms it touches (a long, broad source), and the
    share of its OWN distinctive terms that the objective asked for (a short,
    precise fact). Either shape is real evidence; neither alone catches both.
    """
    if not objective_terms or not item_terms:
        return 0.0
    shared = objective_terms & item_terms
    if not shared:
        return 0.0
    share = len(shared) / len(objective_terms)
    containment = len(shared) / len(item_terms)
    if len(shared) < 2 and containment < 0.5:
        # One word in common is a coincidence, not a topic. The passage about
        # Andy Warhol "turning blue" shared exactly one term with a question
        # about blue marbles. The objective-share reading above still lets a
        # short, on-point fact in on a single term when the objective is
        # itself short — that is the case this must not break.
        containment = 0.0
    return max(share, containment)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _numbers(text: str) -> tuple[str, ...]:
    return tuple(sorted(_NUMBER_RE.findall(text or "")))


class EpistemicFirewall:
    """Deterministic evidence-admission gate for latent-cortex ingress."""

    def __init__(
        self,
        *,
        duplicate_jaccard: float = 0.6,
        conflict_jaccard: float = 0.45,
        stale_after_s: float = 90 * 24 * 3600.0,
        decisive_freshness_s: float = 24 * 3600.0,
        max_admitted: int = 4,
        min_coverage: float = 0.2,
        min_item_relevance: float = 0.12,
    ) -> None:
        if not 0.0 < duplicate_jaccard <= 1.0:
            raise ValueError("duplicate_jaccard must be inside (0, 1]")
        if not 0.0 < conflict_jaccard <= 1.0:
            raise ValueError("conflict_jaccard must be inside (0, 1]")
        if conflict_jaccard > duplicate_jaccard:
            raise ValueError("conflict_jaccard cannot exceed duplicate_jaccard")
        if stale_after_s <= 0 or decisive_freshness_s <= 0:
            raise ValueError("freshness horizons must be positive")
        if not 1 <= int(max_admitted) <= 8:
            raise ValueError("max_admitted must be inside [1, 8]")
        if not 0.0 <= min_coverage <= 1.0:
            raise ValueError("min_coverage must be inside [0, 1]")
        if not 0.0 <= min_item_relevance <= 1.0:
            raise ValueError("min_item_relevance must be inside [0, 1]")
        self.duplicate_jaccard = float(duplicate_jaccard)
        self.conflict_jaccard = float(conflict_jaccard)
        self.stale_after_s = float(stale_after_s)
        self.decisive_freshness_s = float(decisive_freshness_s)
        self.max_admitted = int(max_admitted)
        self.min_coverage = float(min_coverage)
        self.min_item_relevance = float(min_item_relevance)

    # ── Clustering: independence, not repetition ────────────────────────
    def _cluster(
        self,
        items: list[EvidenceItem],
        eligible: set[int] | None = None,
    ) -> list[list[int]]:
        parent = list(range(len(items)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        indices = [
            index
            for index in range(len(items))
            if eligible is None or index in eligible
        ]
        term_sets = [_terms(item.text) for item in items]
        for position, i in enumerate(indices):
            for j in indices[position + 1 :]:
                same_origin = items[i].origin == items[j].origin
                near_duplicate = (
                    _jaccard(term_sets[i], term_sets[j]) >= self.duplicate_jaccard
                )
                # Reports that DISAGREE are never duplicates, however similar
                # their wording — "12 replicas" vs "3 replicas" is a conflict
                # for the graph below, not repetition to collapse.
                if (same_origin or near_duplicate) and self._disagreement(
                    items[i], items[j]
                ) is None:
                    union(i, j)
        clusters: dict[int, list[int]] = {}
        for index in indices:
            clusters.setdefault(find(index), []).append(index)
        return sorted(clusters.values(), key=lambda members: members[0])

    @staticmethod
    def _representative(items: list[EvidenceItem], members: list[int]) -> int:
        def sort_key(index: int):
            item = items[index]
            freshness = item.observed_at if item.observed_at is not None else 0.0
            return (_KIND_RANK[item.kind], -freshness, -item.trust, index)

        return min(members, key=sort_key)

    # ── Conflict detection between representatives ──────────────────────
    @staticmethod
    def _disagreement(left: EvidenceItem, right: EvidenceItem) -> str | None:
        """The disagreement test alone, with no similarity floor."""
        left_numbers, right_numbers = _numbers(left.text), _numbers(right.text)
        if left_numbers and right_numbers and left_numbers != right_numbers:
            return "numeric_disagreement"
        left_terms, right_terms = _terms(left.text), _terms(right.text)
        left_negated = bool(left_terms & _NEGATION_MARKERS) or any(
            marker in left.text.lower() for marker in ("isn't", "can't", "won't", "didn't")
        )
        right_negated = bool(right_terms & _NEGATION_MARKERS) or any(
            marker in right.text.lower() for marker in ("isn't", "can't", "won't", "didn't")
        )
        if left_negated != right_negated:
            return "polarity_disagreement"
        return None

    def _detect_conflict(
        self, left: EvidenceItem, right: EvidenceItem
    ) -> str | None:
        overlap = _jaccard(_terms(left.text), _terms(right.text))
        if overlap < self.conflict_jaccard:
            return None
        return self._disagreement(left, right)

    def _resolve_conflict(
        self, left: EvidenceItem, right: EvidenceItem
    ) -> tuple[str, str] | None:
        """(winner_side, rule) when a conflict is defensibly resolvable."""
        left_rank, right_rank = _KIND_RANK[left.kind], _KIND_RANK[right.kind]
        if left_rank != right_rank:
            return ("left" if left_rank < right_rank else "right", "provenance_rank")
        if left.observed_at is not None and right.observed_at is not None:
            gap = float(left.observed_at) - float(right.observed_at)
            if abs(gap) >= self.decisive_freshness_s:
                return ("left" if gap > 0 else "right", "decisive_freshness")
        return None

    # ── The admission decision ──────────────────────────────────────────
    def review(
        self,
        objective: str,
        raw_items: list[EvidenceItem],
        *,
        now: float | None = None,
        also_relevant_to: str | None = None,
    ) -> FirewallVerdict:
        """Admit evidence for ``objective``.

        ``also_relevant_to`` is the refined query that actually retrieved these
        items, when it differs from the objective. Evidence is judged relevant
        if it speaks to EITHER: "compare scheduler lock strategies" retrieved
        via "find evidence about lease expiry after owner death" must not have
        the lease record refused for failing to mention schedulers. Coverage is
        still measured against the objective, which is what was asked.
        """
        verdict = FirewallVerdict()
        current_time = float(now) if now is not None else time.time()
        items: list[EvidenceItem] = []
        for raw in list(raw_items)[:MAX_ITEMS]:
            try:
                items.append(raw.validated())
            except (AttributeError, TypeError, ValueError) as exc:
                verdict.reasons.append(f"invalid_item:{exc}")
        if len(raw_items) > MAX_ITEMS:
            verdict.reasons.append("item_overflow_clipped")
        if not items:
            verdict.needs_more_retrieval = True
            verdict.reasons.append("no_valid_evidence")
            verdict.uncovered_terms = sorted(_terms(objective))[:8]
            return verdict

        # ── Relevance: is this item about the objective at all? ──────────
        # LIVE DEFECT, 2026-07-26. Asked "a bag has 3 red, 4 blue and 5 green
        # marbles... what's the probability both are the same colour", the
        # desktop turn admitted two local-corpus passages — one about Wilmette,
        # Illinois, one about Andy Warhol's death — as evidence, and served
        # "Do product of multiple exponent term simplify reflexion".
        #
        # Nothing here was broken in isolation: the pair were not duplicates,
        # did not conflict, and fit the slot budget. The gap is that admission
        # never asked the first question. Coverage WAS measured, reported
        # "insufficient_coverage", and then admitted the items anyway — a
        # measurement standing in for a gate. Retrieval relevance is not this
        # module's job, but "shares no topic with the objective" is not
        # relevance ranking, it is the floor below which nothing is evidence.
        objective_distinctive = _distinctive(objective)
        query_distinctive = _distinctive(also_relevant_to or "")
        eligible: set[int] = set(range(len(items)))
        if objective_distinctive and self.min_item_relevance > 0.0:
            for index, item in enumerate(items):
                item_distinctive = _distinctive(item.text)
                score = max(
                    _relevance(objective_distinctive, item_distinctive),
                    _relevance(query_distinctive, item_distinctive),
                )
                if score < self.min_item_relevance:
                    eligible.discard(index)
                    row = self._row(
                        items[index],
                        index,
                        current_time,
                        reason="irrelevant_to_objective",
                    )
                    row["relevance"] = round(score, 4)
                    verdict.refused.append(row)
            if not eligible:
                verdict.needs_more_retrieval = True
                verdict.reasons.append("no_relevant_evidence")
                verdict.uncovered_terms = sorted(objective_distinctive)[:8]
                return verdict

        clusters = self._cluster(items, eligible)
        verdict.clusters = clusters
        representatives: list[int] = []
        for members in clusters:
            representative = self._representative(items, members)
            representatives.append(representative)
            for member in members:
                if member != representative:
                    verdict.refused.append(
                        self._row(
                            items[member],
                            member,
                            current_time,
                            reason=f"duplicate_of:{representative}",
                        )
                    )

        # Conflict graph over independent representatives.
        losers: set[int] = set()
        unresolved: set[int] = set()
        for position, left_index in enumerate(representatives):
            for right_index in representatives[position + 1 :]:
                method = self._detect_conflict(items[left_index], items[right_index])
                if method is None:
                    continue
                resolution = self._resolve_conflict(
                    items[left_index], items[right_index]
                )
                edge: dict[str, Any] = {
                    "left": left_index,
                    "right": right_index,
                    "method": method,
                    "resolved_by": resolution[1] if resolution else "",
                }
                verdict.conflicts.append(edge)
                if resolution is None:
                    unresolved.update((left_index, right_index))
                else:
                    loser = right_index if resolution[0] == "left" else left_index
                    losers.add(loser)

        for index in sorted(unresolved):
            verdict.refused.append(
                self._row(items[index], index, current_time, reason="unresolved_conflict")
            )
        for index in sorted(losers - unresolved):
            verdict.refused.append(
                self._row(items[index], index, current_time, reason="conflict_loser")
            )
        if unresolved:
            verdict.abstain = True
            verdict.reasons.append("unresolved_conflict")

        admissible = [
            index
            for index in representatives
            if index not in unresolved and index not in losers
        ]
        cluster_by_representative = {
            self._representative(items, members): members for members in clusters
        }

        def admission_key(index: int):
            item = items[index]
            independent = len(
                {items[m].origin for m in cluster_by_representative[index]}
            )
            freshness = item.observed_at if item.observed_at is not None else 0.0
            return (_KIND_RANK[item.kind], -independent, -freshness, -item.trust, index)

        admissible.sort(key=admission_key)
        for index in admissible[: self.max_admitted]:
            row = self._row(items[index], index, current_time)
            members = cluster_by_representative[index]
            row["independent_sources"] = len({items[m].origin for m in members})
            row["cluster_size"] = len(members)
            verdict.admitted.append(row)
        for index in admissible[self.max_admitted :]:
            verdict.refused.append(
                self._row(items[index], index, current_time, reason="slot_budget")
            )

        # Coverage: which distinctive objective terms the admitted evidence
        # actually touches. A bounded heuristic, receipted as such.
        # Distinctive terms only: an admitted passage that shares "the" and
        # "was" with the question has covered nothing, and a coverage number
        # inflated by function words cannot tell anyone that.
        objective_terms = objective_distinctive
        if objective_terms:
            covered: set[str] = set()
            for row in verdict.admitted:
                covered |= _terms(str(row["text"])) & objective_terms
            verdict.coverage = len(covered) / len(objective_terms)
            verdict.uncovered_terms = sorted(objective_terms - covered)[:8]
        else:
            verdict.coverage = 1.0 if verdict.admitted else 0.0
        if verdict.coverage < self.min_coverage:
            verdict.needs_more_retrieval = True
            verdict.reasons.append("insufficient_coverage")
        return verdict

    def _row(
        self,
        item: EvidenceItem,
        index: int,
        current_time: float,
        *,
        reason: str = "",
    ) -> dict[str, Any]:
        age_s = (
            max(0.0, current_time - float(item.observed_at))
            if item.observed_at is not None
            else None
        )
        row: dict[str, Any] = {
            "index": index,
            "text": item.text,
            "origin": item.origin,
            "channel": item.channel,
            "kind": item.kind,
            "trust": round(item.trust, 4),
            "age_s": None if age_s is None else round(age_s, 1),
            "stale": bool(age_s is not None and age_s > self.stale_after_s),
        }
        if reason:
            row["reason"] = reason
        return row


__all__ = [
    "EPISTEMIC_FIREWALL_SCHEMA",
    "EvidenceItem",
    "EpistemicFirewall",
    "FirewallVerdict",
]
