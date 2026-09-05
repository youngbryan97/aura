"""Calibration gate — Aura may only assert what survived a check.

Not glamorous, very powerful: most bad answers are not *illogical*, they are
*overconfident*. This gate classifies every load-bearing sentence of a candidate
answer by epistemic status —

    KNOWN            backed by memory / established fact in context
    TOOL_VERIFIED    a verifier or sandbox actually confirmed it
    SOURCE_BACKED    grounded in a supplied evidence span
    INFERRED         a reasonable deduction, not directly checked
    GUESSED          confident but unsupported
    UNVERIFIED       a claim we had no way to check
    IMPOSSIBLE_LOCALLY  asserts something that needs a capability we lack

— and then *applies* the verdict: confident-but-unsupported sentences get an
honesty hedge, impossible-locally claims are flagged. The result is an answer the
rest of the system is allowed to speak, plus a calibrated confidence the response
path and governance can read.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.runtime.errors import record_degradation


class EpistemicStatus(StrEnum):
    KNOWN = "known"
    TOOL_VERIFIED = "tool_verified"
    SOURCE_BACKED = "source_backed"
    INFERRED = "inferred"
    GUESSED = "guessed"
    UNVERIFIED = "unverified"
    IMPOSSIBLE_LOCALLY = "impossible_locally"


_CONFIDENCE_RANK = {
    EpistemicStatus.TOOL_VERIFIED: 1.0,
    EpistemicStatus.KNOWN: 0.92,
    EpistemicStatus.SOURCE_BACKED: 0.85,
    EpistemicStatus.INFERRED: 0.6,
    EpistemicStatus.UNVERIFIED: 0.4,
    EpistemicStatus.GUESSED: 0.3,
    EpistemicStatus.IMPOSSIBLE_LOCALLY: 0.05,
}

_HEDGE_RE = re.compile(
    r"\b(?:i think|maybe|might|possibly|i'?m not (?:sure|certain)|uncertain|"
    r"i'?m guessing|probably|i believe|it seems|appears to|i'?m not aware|"
    r"as far as i know|to my knowledge)\b",
    re.IGNORECASE,
)
_CONFIDENT_RE = re.compile(
    r"\b(?:is|are|was|were|will|always|never|definitely|certainly|guaranteed|"
    r"the answer is|in fact|undoubtedly|must be|exactly)\b",
    re.IGNORECASE,
)
# Phrases that claim a capability a local model generally cannot have done.
#: Labels emitted in a serialized report.
_MAX_REPORTED_LABELS = 10

_IMPOSSIBLE_RE = re.compile(
    r"\b(?:i (?:just )?(?:browsed|googled|searched the web|accessed the internet|"
    r"checked online|called the api|ran it on your machine|looked it up online)|"
    r"according to (?:today'?s|the latest) (?:news|web))\b",
    re.IGNORECASE,
)


@dataclass
class ClaimLabel:
    text: str
    status: EpistemicStatus
    reason: str = ""
    #: Offsets into the ORIGINAL answer, so a rewrite can splice rather than
    #: rebuild. -1 when the label was not derived from a span.
    start: int = -1
    end: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text[:160], "status": self.status.value, "reason": self.reason}


@dataclass
class CalibrationReport:
    overall: EpistemicStatus
    confidence: float
    labels: list[ClaimLabel] = field(default_factory=list)
    calibrated_answer: str = ""
    downgraded: int = 0
    flagged_impossible: int = 0
    # Substrate interoception coupling (None when no felt trace matched):
    felt_confidence: float | None = None
    felt_demotions: int = 0
    #: "consulted" | "supplied" | "no_trace" | "unavailable:<reason>".
    #: CP126 0d707883: an import or lookup failure returned None, which the
    #: caller could not tell apart from "no trace matched" — so the gate
    #: could present an ordinary calibration result after losing one of its
    #: advertised evidence channels.
    felt_channel: str = "no_trace"

    def to_dict(self) -> dict[str, Any]:
        """The full report, INCLUDING the answer the gate says may be spoken.

        CP126 a40c5b75: ``calibrated_answer`` is the entire point of this
        gate — the corrected text the rest of the system is allowed to say —
        and serialization excluded it. Any boundary consuming only
        ``to_dict()`` got the scores and labels while silently losing the
        correction, and then spoke the original. The honesty gate was
        bypassed by its own serializer.

        CP126 1bfaa5ab: labels were cut to the first ten with no total and
        no truncation flag, so a long answer could not be audited.
        """
        return {
            "overall": self.overall.value,
            "confidence": round(self.confidence, 3),
            "calibrated_answer": self.calibrated_answer,
            "downgraded": self.downgraded,
            "flagged_impossible": self.flagged_impossible,
            "felt_confidence": (
                round(self.felt_confidence, 3) if self.felt_confidence is not None else None
            ),
            "felt_demotions": self.felt_demotions,
            # Whether the interoception channel was actually consulted, so a
            # LOST evidence channel is never read as "nothing felt contested".
            "felt_channel": self.felt_channel,
            "labels": [c.to_dict() for c in self.labels[:_MAX_REPORTED_LABELS]],
            "label_count": len(self.labels),
            "labels_truncated": len(self.labels) > _MAX_REPORTED_LABELS,
        }


def _sentences(text: str) -> list[str]:
    return [span[0] for span in _sentence_spans(text)]


def _sentence_spans(text: str) -> list[tuple[str, int, int]]:
    """Sentences WITH their offsets in the original text.

    CP126 616e5ae1: the gate rebuilt the answer with `" ".join(sentences)`,
    which discards newlines, paragraphs, list structure, indentation and
    code-fence layout — for an answer containing code that can change what
    the code DOES, and it happened on every answer, including the common
    case where nothing was downgraded at all.

    Keeping offsets lets the rewrite splice only the sentences it actually
    modified back into the untouched original.
    """
    raw = str(text or "")
    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for piece in re.split(r"(?<=[.!?])(\s+)", raw):
        if not piece:
            continue
        if piece.isspace():
            cursor += len(piece)
            continue
        stripped = piece.strip()
        if stripped:
            offset = piece.index(stripped)
            start = cursor + offset
            spans.append((stripped, start, start + len(stripped)))
        cursor += len(piece)
    return spans


class CalibrationGate:
    """Classify and honesty-correct an answer's claims by epistemic status."""

    def assess(
        self,
        answer: str,
        *,
        verification: Any | None = None,
        evidence: list[str] | None = None,
        tool_verified: bool = False,
        known_facts: list[str] | None = None,
        felt: Any | None = None,
    ) -> CalibrationReport:
        evidence_blob = "\n".join(str(e) for e in (evidence or [])).lower()
        known_blob = "\n".join(str(k) for k in (known_facts or [])).lower()
        # A verifier that actually checked and passed promotes factual sentences.
        v_checked = bool(getattr(verification, "checked", False))
        v_ok = bool(getattr(verification, "ok", False))

        labels: list[ClaimLabel] = []
        for sent, start, end in _sentence_spans(answer):
            label = self._classify_sentence(
                sent, evidence_blob, known_blob,
                v_checked=v_checked, v_ok=v_ok, tool_verified=tool_verified,
            )
            label.start, label.end = start, end
            labels.append(label)

        # Substrate interoception: if the answer's felt trace shows the words
        # were contested as they formed, unsupported sentences overlapping the
        # contested regions lose their nerve (see _apply_felt).
        if felt is not None:
            felt_trace, felt_channel = felt, "supplied"
        else:
            felt_trace, felt_channel = self._felt_trace_for(answer)
        felt_demotions = 0
        felt_conf: float | None = None
        if felt_trace is not None:
            felt_conf = float(getattr(felt_trace, "felt_confidence", 0.5))
            felt_demotions = self._apply_felt(labels, felt_trace)

        calibrated, downgraded, flagged = self._apply(answer, labels)
        overall = self._overall(labels)
        confidence = self._confidence(labels, v_checked=v_checked, v_ok=v_ok)
        if felt_conf is not None:
            # Internal doubt can lower stated confidence, never raise it: a
            # contested decode caps the ceiling, a fluent one changes nothing.
            confidence = round(min(confidence, 0.45 + 0.5 * felt_conf), 4)
        return CalibrationReport(
            overall=overall,
            confidence=confidence,
            labels=labels,
            calibrated_answer=calibrated,
            downgraded=downgraded,
            flagged_impossible=flagged,
            felt_confidence=felt_conf,
            felt_demotions=felt_demotions,
            felt_channel=felt_channel,
        )

    @staticmethod
    def _felt_trace_for(answer: str) -> Any:
        """Fetch the substrate felt-trace recorded for this exact answer, if any.

        Fail-open: no trace, stale trace, or an unavailable organ simply means
        the gate runs text-only, exactly as before interoception existed.
        """
        try:
            from core.being.thought_interoception import get_thought_interoception

            return get_thought_interoception().find_for_text(answer), "consulted"
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            # CP126 0d707883: returning a bare None here made "the organ is
            # gone" indistinguishable from "nothing matched", so the gate
            # could present an ordinary result after silently losing one of
            # its advertised evidence channels.
            record_degradation(
                "calibration_gate",
                exc,
                action="calibrated text-only after the interoception channel was unavailable",
            )
            return None, f"unavailable:{type(exc).__name__}"

    @staticmethod
    def _apply_felt(labels: list[ClaimLabel], felt_trace: Any) -> int:
        """Demote unsupported sentences that overlap contested decode regions.

        Epistemology, deliberately conservative:
        * externally supported statuses (KNOWN / SOURCE_BACKED / TOOL_VERIFIED)
          are never demoted — real evidence beats internal fluency;
        * already-hedged sentences (INFERRED) and already-demoted GUESSED stay;
        * only UNVERIFIED sentences that contain words from the felt trace's
          surprisal spikes are demoted to GUESSED (which the apply pass then
          hedges), and only when the whole thought felt contested
          (low felt confidence or high ambivalence).
        """
        try:
            felt_confidence = float(getattr(felt_trace, "felt_confidence", 1.0))
            ambivalence = float(getattr(felt_trace, "ambivalence", 0.0))
            if felt_confidence >= 0.45 and ambivalence <= 0.35:
                return 0
            contested: set[str] = set()
            for spike in getattr(felt_trace, "spikes", ()) or ():
                blob = f"{spike.get('text', '')} {spike.get('context', '')}"
                contested.update(w for w in re.findall(r"[a-zA-Z]{4,}", blob.lower()))
            if not contested:
                return 0
            demoted = 0
            for label in labels:
                if label.status is not EpistemicStatus.UNVERIFIED:
                    continue
                sentence_words = set(re.findall(r"[a-zA-Z]{4,}", label.text.lower()))
                if sentence_words & contested:
                    label.status = EpistemicStatus.GUESSED
                    label.reason = (
                        "felt contested while forming (substrate interoception: "
                        f"confidence={felt_confidence:.2f}, ambivalence={ambivalence:.2f})"
                    )
                    demoted += 1
            return demoted
        except (AttributeError, TypeError, ValueError):
            return 0

    def _classify_sentence(
        self,
        sent: str,
        evidence_blob: str,
        known_blob: str,
        *,
        v_checked: bool,
        v_ok: bool,
        tool_verified: bool,
    ) -> ClaimLabel:
        if _IMPOSSIBLE_RE.search(sent):
            return ClaimLabel(sent, EpistemicStatus.IMPOSSIBLE_LOCALLY, "claims a capability not available locally")
        hedged = bool(_HEDGE_RE.search(sent))
        confident = bool(_CONFIDENT_RE.search(sent))
        content_words = {w for w in re.findall(r"[a-zA-Z]{4,}", sent.lower())}

        def _overlap(blob: str) -> float:
            if not content_words or not blob:
                return 0.0
            return sum(1 for w in content_words if w in blob) / len(content_words)

        if known_blob and _overlap(known_blob) >= 0.4:
            return ClaimLabel(sent, EpistemicStatus.KNOWN, "matches a known fact in context")
        if evidence_blob and _overlap(evidence_blob) >= 0.4:
            return ClaimLabel(sent, EpistemicStatus.SOURCE_BACKED, "grounded in supplied evidence")
        if tool_verified and v_checked and v_ok and confident:
            return ClaimLabel(sent, EpistemicStatus.TOOL_VERIFIED, "confirmed by a verifier/sandbox")
        if hedged:
            return ClaimLabel(sent, EpistemicStatus.INFERRED, "appropriately hedged")
        if confident:
            return ClaimLabel(sent, EpistemicStatus.GUESSED, "confident assertion without support")
        return ClaimLabel(sent, EpistemicStatus.UNVERIFIED, "no check available")

    def _apply(self, answer: str, labels: list[ClaimLabel]) -> tuple[str, int, int]:
        """Rewrite the answer so confidence matches epistemic status.

        Splices replacements into the ORIGINAL text at the sentences that
        actually changed, so everything else — newlines, paragraphs, list
        markers, indentation, code fences — survives byte for byte. An
        answer with nothing to downgrade is returned untouched.
        """
        original = str(answer or "")
        downgraded = 0
        flagged = 0
        edits: list[tuple[int, int, str]] = []

        for label in labels:
            if label.status is EpistemicStatus.IMPOSSIBLE_LOCALLY:
                flagged += 1
                replacement = f"[unverifiable locally] {label.text}"
            elif label.status is EpistemicStatus.GUESSED:
                downgraded += 1
                replacement = self._soften(label.text)
            else:
                continue
            if replacement == label.text:
                continue
            if label.start < 0 or label.end > len(original):
                index = original.find(label.text)
                if index < 0:
                    continue
                edits.append((index, index + len(label.text), replacement))
            else:
                edits.append((label.start, label.end, replacement))

        if not edits:
            return original, downgraded, flagged

        rebuilt: list[str] = []
        cursor = 0
        for start, end, replacement in sorted(edits):
            if start < cursor:
                continue  # overlapping edit; keep the first
            rebuilt.append(original[cursor:start])
            rebuilt.append(replacement)
            cursor = end
        rebuilt.append(original[cursor:])
        return "".join(rebuilt) or original, downgraded, flagged

    @staticmethod
    def _soften(sentence: str) -> str:
        """Insert an honesty hedge into a confident-but-unsupported sentence."""
        if _HEDGE_RE.search(sentence):
            return sentence
        # Prepend a measured qualifier rather than mangling the sentence body.
        lead = sentence[0].lower() + sentence[1:] if sentence[:1].isupper() else sentence
        return f"I'm not fully certain, but {lead}"

    @staticmethod
    def _overall(labels: list[ClaimLabel]) -> EpistemicStatus:
        if not labels:
            return EpistemicStatus.UNVERIFIED
        if any(label.status is EpistemicStatus.IMPOSSIBLE_LOCALLY for label in labels):
            return EpistemicStatus.IMPOSSIBLE_LOCALLY
        # The weakest load-bearing status dominates the headline.
        order = [
            EpistemicStatus.GUESSED, EpistemicStatus.UNVERIFIED, EpistemicStatus.INFERRED,
            EpistemicStatus.SOURCE_BACKED, EpistemicStatus.KNOWN, EpistemicStatus.TOOL_VERIFIED,
        ]
        present = {label.status for label in labels}
        for status in order:
            if status in present:
                return status
        return EpistemicStatus.UNVERIFIED

    @staticmethod
    def _confidence(labels: list[ClaimLabel], *, v_checked: bool, v_ok: bool) -> float:
        if not labels:
            return 0.4
        base = sum(_CONFIDENCE_RANK[label.status] for label in labels) / len(labels)
        if v_checked and not v_ok:
            base *= 0.6  # a verifier actively found a problem
        return round(max(0.05, min(0.98, base)), 4)


_instance: CalibrationGate | None = None


def get_calibration_gate() -> CalibrationGate:
    global _instance
    if _instance is None:
        _instance = CalibrationGate()
    return _instance
