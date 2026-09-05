"""core/capabilities/interlocutor_factcheck.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Grounded pushback: Aura challenges another web AI when it is wrong, instead of
politely agreeing. Same mind, same conversation — but willing to disagree, and
only when she has grounds.

The honesty constraint is the whole point: a system that reflexively "corrects"
its interlocutor is just contrarian theater. This checks the interlocutor's
*checkable* assertions against Aura's local knowledge (the offline corpus) and
raises a challenge ONLY when a retrieved source actually contradicts the claim
— with the counter-evidence and its provenance attached, so the pushback is
verifiable rather than asserted.

Two contradiction signals, precision-first:
1. numeric/date mismatch — the interlocutor gives a value for a subject that a
   retrieved source states differently (highest precision, deterministic);
2. an optional engine adjudication seam — the real cognitive engine may judge a
   contradiction, but is required to cite the grounding passage, and defaults
   to "no challenge" when unsure.

If neither fires, there is no challenge. Silence beats a false accusation.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable

logger = logging.getLogger("Aura.WebInterlocutor.Factcheck")

# A corpus search returns passages: each a dict with at least {text, source}.
CorpusSearch = Callable[[str, int], list[dict[str, Any]]]

_FACTUAL_MARKERS = (
    " is ", " was ", " are ", " were ", " invented", " discovered", " founded",
    " created", " built", " first ", " only ", " always", " never", " in 1", " in 2",
)
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
# subject = the leading noun phrase before the first factual verb
_SUBJECT_RE = re.compile(r"^\s*([A-Z][\w'’-]*(?:\s+[A-Z0-9][\w'’-]*){0,5})")


@dataclass
class GroundedContradiction:
    interlocutor_claim: str
    counter_evidence: str
    source: str
    signal: str          # "numeric_mismatch" | "adjudicated"
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_checkable_claims(reply: str) -> list[str]:
    """Sentences that assert something falsifiable (a value, date, or definite
    'X is Y'). Opinions and hedged statements are skipped."""
    sentences = re.split(r"(?<=[.!?])\s+", str(reply or "").strip())
    claims: list[str] = []
    for sentence in sentences:
        s = sentence.strip()
        if len(s) < 16:
            continue
        low = f" {s.lower()} "
        if not any(marker in low for marker in _FACTUAL_MARKERS):
            continue
        # skip clearly-hedged / opinion sentences — challenging those is theater
        if re.search(r"\b(i think|maybe|perhaps|in my view|arguably|might|could be|seems)\b", low):
            continue
        claims.append(s[:400])
    return claims


def _subject_of(claim: str) -> str:
    match = _SUBJECT_RE.match(claim.strip())
    return (match.group(1).strip() if match else "").lower()


def _numeric_mismatch(claim: str, passage_text: str) -> tuple[bool, str]:
    """A high-precision contradiction: the claim asserts a year/number for a
    subject and a passage about the SAME subject states a different one."""
    claim_years = set(_YEAR_RE.findall(claim))
    if not claim_years:
        return False, ""
    subject = _subject_of(claim)
    if not subject or len(subject) < 3:
        return False, ""
    # the passage must actually be about the claim's subject
    if subject not in passage_text.lower():
        return False, ""
    passage_years = set(_YEAR_RE.findall(passage_text))
    if not passage_years:
        return False, ""
    if claim_years & passage_years:
        return False, ""  # they agree on at least one value → not a contradiction
    # subject matches, both cite years, none shared → mismatch
    sentence = _first_sentence_with(passage_text, subject) or passage_text[:240]
    return True, sentence


def _first_sentence_with(text: str, needle: str) -> str:
    for sentence in re.split(r"(?<=[.!?])\s+", str(text or "")):
        if needle in sentence.lower():
            return sentence.strip()[:240]
    return ""


def factcheck_reply(
    reply: str,
    *,
    corpus_search: CorpusSearch,
    adjudicate: Callable[[str, list[dict[str, Any]]], tuple[bool, str, float]] | None = None,
    min_confidence: float = 0.6,
    max_challenges: int = 2,
) -> list[GroundedContradiction]:
    """Return grounded contradictions in the interlocutor's reply, or []."""
    contradictions: list[GroundedContradiction] = []
    for claim in extract_checkable_claims(reply):
        # Precision-first and resource-bounded: without an adjudicator, this
        # module can only prove deterministic numeric/date mismatches. Broad
        # philosophical "X is Y" claims should not trigger expensive corpus
        # search every turn; that created live web-interlocutor SLO noise while
        # producing no possible contradiction.
        if adjudicate is None and not (_YEAR_RE.search(claim) or _NUMBER_RE.search(claim)):
            continue
        try:
            passages = corpus_search(claim, 3) or []
        except (RuntimeError, OSError, TypeError, ValueError) as exc:
            logger.debug("Corpus search failed for a factcheck claim: %s", exc)
            continue
        if not passages:
            continue

        matched = False
        for passage in passages:
            text = str(passage.get("text") or "")
            source = str(passage.get("source") or passage.get("title") or "local_corpus")
            hit, evidence = _numeric_mismatch(claim, text)
            if hit:
                contradictions.append(
                    GroundedContradiction(
                        interlocutor_claim=claim,
                        counter_evidence=evidence,
                        source=source,
                        signal="numeric_mismatch",
                        confidence=0.75,
                    )
                )
                matched = True
                break
        if matched:
            if len(contradictions) >= max_challenges:
                break
            continue

        # Engine adjudication seam (live path): must cite a passage and clear
        # the confidence floor, else no challenge.
        if adjudicate is not None:
            try:
                is_contra, evidence, confidence = adjudicate(claim, passages)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                logger.debug("Adjudication failed for a factcheck claim: %s", exc)
                is_contra, evidence, confidence = False, "", 0.0
            if is_contra and evidence and confidence >= min_confidence:
                contradictions.append(
                    GroundedContradiction(
                        interlocutor_claim=claim,
                        counter_evidence=str(evidence)[:240],
                        source="local_corpus",
                        signal="adjudicated",
                        confidence=round(float(confidence), 3),
                    )
                )
        if len(contradictions) >= max_challenges:
            break
    return contradictions


def compose_challenge_message(contradictions: list[GroundedContradiction]) -> str:
    """A firm-but-civil grounded correction Aura can send as her next message.
    Deterministic so the challenge is honest even if generation is unavailable."""
    if not contradictions:
        return ""
    lead = (
        "I want to push back on one thing before we go on — I think that is not "
        "quite right, and here is why."
    )
    parts = [lead]
    for c in contradictions:
        parts.append(
            f"You said: \"{c.interlocutor_claim}\" My local reference says: "
            f"\"{c.counter_evidence}\" ({c.source}). Can you reconcile that, or "
            f"was the detail off?"
        )
    return " ".join(parts)[:1400]


__all__ = [
    "GroundedContradiction",
    "extract_checkable_claims",
    "factcheck_reply",
    "compose_challenge_message",
]
