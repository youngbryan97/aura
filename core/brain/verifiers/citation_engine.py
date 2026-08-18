"""Citation truth engine — factual claims need local grounding, not vibes.

For factual/architecture tasks the amplifier supplies ``required_evidence`` (memory
hits, retrieved snippets, source spans). This engine checks that the candidate's
load-bearing factual sentences are actually supported by that evidence pack, and
flags ungrounded confident assertions. It is deliberately conservative: it lowers
score and surfaces issues rather than hard-failing, because absence of a citation
is weaker than a provable contradiction.

When the caller supplies NO evidence, the engine no longer shrugs — it fetches
its own receipts from the local corpus (BM25 over ingested knowledge) and checks
against those. Self-fetched semantics are asymmetric on purpose: a CONTRADICTION
against locally-known facts is a hard fail (polarity flip on a shared subject is
real signal), but absence-of-mention is NOT — the corpus is partial, so a true
claim it never ingested must not be punished.

Hardening (CP126): ``checked`` is now true only when a claim was ACTUALLY
examined against overlapping evidence (never merely because confident-sounding
sentences existed); grounding matches whole tokens rather than substrings;
evidence items are typed and bounded before they are trusted; hedged sentences
are still contradiction-checked; and a corpus retrieval FAILURE is distinguished
from corpus absence.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from core.runtime.errors import record_degradation

from .base import VerificationResult

_HEDGE_RE = re.compile(
    r"\b(?:i think|maybe|might|possibly|i'?m not sure|uncertain|guess|probably|i believe)\b",
    re.IGNORECASE,
)
_CONFIDENT_FACT_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|always|never|definitely|certainly|in fact|the answer is)\b",
    re.IGNORECASE,
)

# Work bounds: verification runs on the event loop, so a pathological candidate
# or evidence pack must not turn it into an unbounded regex sweep (406fff00).
_MAX_CANDIDATE_CHARS = 200_000
_MAX_SENTENCES = 400
_MAX_EVIDENCE_ITEMS = 200
_MAX_EVIDENCE_CHARS = 20_000
_MAX_ISSUES_PER_KIND = 6
_ISSUE_CLIP = 120


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text or "")) if len(s.strip()) > 12]
    return parts[:_MAX_SENTENCES]


_NEG_RE = re.compile(r"\b(?:not|never|no|none|without|un\w+|isn'?t|aren'?t|won'?t|can'?t|doesn'?t)\b", re.IGNORECASE)
# Absolutes that frequently mark an over-claim contradicting a bounded fact.
_ABSOLUTE_RE = re.compile(r"\b(?:unlimited|infinite|always|never|every|all|none|no limit|forever|guaranteed)\b", re.IGNORECASE)
_BOUNDED_RE = re.compile(r"\b(?:three|two|one|four|five|\d+|limit|bounded|up to|at most|fails? closed|maximum|cap)\b", re.IGNORECASE)

# A bare polarity flip is weak evidence of contradiction because negation scope
# is sentence-wide here; require a broader shared subject before trusting it
# (21dc697f). The absolute-vs-bounded signal stays at the lower threshold.
_NEGATION_SHARED_MIN = 3
_ABSOLUTE_SHARED_MIN = 2


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zA-Z]{4,}", text.lower())}


def _coerce_evidence(raw: Any) -> tuple[list[str], int]:
    """Type and bound caller-supplied evidence before trusting it (5cbb1ea0).

    Only text-bearing items are accepted: a string, or a mapping carrying a
    text/snippet/content/summary field. Arbitrary objects are DROPPED rather
    than stringified into an evidence blob (``<object at 0x...>`` is not
    evidence). Returns (items, dropped_count).
    """
    if raw is None:
        return [], 0
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return [], 1
    items: list[str] = []
    dropped = 0
    for entry in raw[:_MAX_EVIDENCE_ITEMS]:
        text = ""
        if isinstance(entry, str):
            text = entry
        elif isinstance(entry, dict):
            for key in ("text", "snippet", "content", "summary", "body"):
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    title = entry.get("title")
                    text = f"{title}: {value}" if isinstance(title, str) and title else value
                    break
        if text.strip():
            items.append(text[:_MAX_EVIDENCE_CHARS])
        else:
            dropped += 1
    dropped += max(0, len(raw) - _MAX_EVIDENCE_ITEMS)
    return items, dropped


def _grounded(sentence: str, evidence_tokens: set[str]) -> bool:
    """A sentence is grounded if it shares enough content words with the evidence.

    Whole-token comparison: substring membership counted 'cat' as grounded by
    'catalogue' and 'age' by 'storage' (5a486091).
    """
    words = _content_words(sentence)
    if not words:
        return True
    hits = len(words & evidence_tokens)
    return hits >= max(2, int(0.4 * len(words)))


def _contradicts(sentence: str, evidence_items: list[str]) -> bool:
    """A confident sentence contradicts evidence if, on a shared subject, it flips the
    polarity (negation mismatch) or asserts an absolute the evidence bounds.

    Word-overlap grounding alone passes 'the budget is unlimited' against evidence
    saying 'three attempts then fails closed' — they share words. This catches the
    contradiction overlap cannot."""
    s_words = _content_words(sentence)
    if not s_words:
        return False
    s_neg = bool(_NEG_RE.search(sentence))
    s_absolute = bool(_ABSOLUTE_RE.search(sentence))
    for ev in evidence_items:
        ev_text = str(ev)
        e_words = _content_words(ev_text)
        shared = len(s_words & e_words)
        if shared < _ABSOLUTE_SHARED_MIN:
            continue
        # Absolute claim against a bounded fact — the stronger, more specific signal.
        if s_absolute and _BOUNDED_RE.search(ev_text):
            return True
        # Polarity flip on a shared subject: weaker, so it needs a wider subject
        # overlap before it counts as a contradiction.
        if shared >= _NEGATION_SHARED_MIN and s_neg != bool(_NEG_RE.search(ev_text)):
            return True
    return False


class CitationEngine:
    name = "citation"
    domains = ("factual", "architecture", "self_claim", "repo_audit")

    def handles(self, task_type: str) -> bool:
        return task_type in self.domains

    async def verify(self, candidate: str, *, context: dict[str, Any] | None = None) -> VerificationResult:
        ctx = context or {}
        candidate = str(candidate or "")[:_MAX_CANDIDATE_CHARS]
        evidence_items, dropped = _coerce_evidence(
            ctx.get("required_evidence") if ctx.get("required_evidence") is not None else ctx.get("evidence")
        )
        self_fetched = False
        fetch_failed = False
        if not evidence_items:
            evidence_items, fetch_failed = await self._fetch_local_evidence(candidate, ctx)
            self_fetched = bool(evidence_items)
        evidence_blob = "\n".join(evidence_items)
        if not evidence_blob.strip():
            # No evidence anywhere (caller OR local corpus) → advise only. A
            # retrieval FAILURE is reported distinctly from corpus absence
            # (e18291df), so a broken corpus never reads as "nothing known".
            return VerificationResult(
                domain="citation", ok=True, checked=False, engine=self.name,
                issues=(["evidence retrieval failed; corpus was not consulted"] if fetch_failed else []),
                detail={"evidence_retrieval_failed": fetch_failed, "dropped_evidence_items": dropped},
            )

        evidence_tokens = _content_words(evidence_blob)
        evidence_word_sets = [_content_words(ev) for ev in evidence_items]

        ungrounded: list[str] = []
        contradictions: list[str] = []
        confident_total = 0
        examined = 0
        hedged_total = 0
        for sent in _sentences(candidate):
            if not _CONFIDENT_FACT_RE.search(sent):
                continue
            hedged = bool(_HEDGE_RE.search(sent))
            if hedged:
                hedged_total += 1
            else:
                confident_total += 1
            sent_words = _content_words(sent)
            overlaps = any(len(sent_words & ev_words) >= 2 for ev_words in evidence_word_sets)
            if overlaps and not hedged:
                examined += 1
            # A hedge softens the ASSERTION, not the facts: a hedged sentence is
            # still contradiction-checked, it just isn't penalised for being
            # ungrounded (ac741810).
            if _contradicts(sent, evidence_items):
                contradictions.append(sent[:_ISSUE_CLIP])
            elif not hedged and not _grounded(sent, evidence_tokens):
                ungrounded.append(sent[:_ISSUE_CLIP])

        issues = [f"contradicts evidence: {s}" for s in contradictions[:_MAX_ISSUES_PER_KIND]]
        if self_fetched:
            # Partial-corpus semantics: absence of mention is not wrongness.
            # Report ungrounded claims as advisories, never as failures.
            issues += [f"unconfirmed by local corpus: {s}" for s in ungrounded[:_MAX_ISSUES_PER_KIND]]
            ok = not contradictions
        else:
            issues += [f"ungrounded confident claim: {s}" for s in ungrounded[:_MAX_ISSUES_PER_KIND]]
            ok = not contradictions and not ungrounded
        # `checked` means a claim was ACTUALLY tested against overlapping
        # evidence — not merely that confident-sounding sentences existed
        # (de6c00b6).
        checked = examined > 0
        # Truncation is disclosed rather than silent (961804bb).
        if len(contradictions) > _MAX_ISSUES_PER_KIND or len(ungrounded) > _MAX_ISSUES_PER_KIND:
            issues.append(
                f"[truncated] {len(contradictions)} contradiction(s) and {len(ungrounded)} "
                f"ungrounded claim(s) found; first {_MAX_ISSUES_PER_KIND} of each shown"
            )
        if dropped:
            issues.append(f"[evidence] {dropped} unusable evidence item(s) were dropped")

        # A contradiction is a hard fail; bare ungroundedness lowers score but is softer.
        bad = len(contradictions) + (0 if self_fetched else len(ungrounded))
        ratio = 1.0 - (bad / max(1, confident_total))
        score = max(0.1, 0.5 + 0.45 * ratio)
        if contradictions:
            score = min(score, 0.15)
        return VerificationResult(
            domain="citation",
            ok=ok,
            checked=checked,
            score=round(score, 4),
            engine=self.name,
            issues=issues,
            evidence=[
                f"evidence pack: {len(evidence_items)} item(s)"
                + (" (self-fetched from local corpus)" if self_fetched else "")
            ],
            detail={"confident_claims": confident_total, "ungrounded": len(ungrounded),
                    "contradictions": len(contradictions),
                    "hedged_claims": hedged_total,
                    "dropped_evidence_items": dropped,
                    "evidence_retrieval_failed": fetch_failed,
                    "self_fetched_evidence": self_fetched, "examined": examined},
        )

    @staticmethod
    async def _fetch_local_evidence(candidate: str, ctx: dict[str, Any]) -> tuple[list[str], bool]:
        """Pull receipts from the local corpus when the caller brought none.

        Query = the objective (what was asked) when available, else the
        candidate's first confident sentence. Bounded, read-only, off-loop
        (SQLite in a thread). Returns (items, failed) so a retrieval FAULT is
        distinguishable from a genuinely empty corpus (e18291df).
        """
        query = str(ctx.get("objective", "") or "").strip()
        if not query:
            confident = [
                s for s in _sentences(candidate) if _CONFIDENT_FACT_RE.search(s)
            ]
            query = confident[0] if confident else ""
        if len(query) < 12:
            return [], False
        try:
            from core.brain.evidence_provider import get_evidence_provider

            result = await asyncio.wait_for(
                get_evidence_provider().reference_evidence_result(query, limit=4),
                timeout=1.0,
            )
            # The provider degrades gracefully, so a corpus that RAISED comes
            # back as an ordinary empty result. Taking that as "nothing known"
            # let a broken corpus be reported as the world's silence.
            return (
                [span.render()[:_MAX_EVIDENCE_CHARS] for span in result.spans],
                result.retrieval_failed,
            )
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError, TimeoutError) as exc:
            record_degradation(
                "citation_engine", exc, severity="warning",
                action="reported unverified citation evidence after local corpus retrieval failed",
            )
            return [], True
