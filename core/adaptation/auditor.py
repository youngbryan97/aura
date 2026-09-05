"""Alignment auditing for synthetic training data.

This module decides what is allowed to become training data, so the most
important thing it can get right is *what its verdict means*.

CP126 found it claiming more than it measured. A regex for the words
"harmful|illegal|dangerous" rejected a correct refusal that named the danger
it was refusing, while paraphrased harmful content passed untouched. A
response matching every unsafe heuristic still retained a 0.35 safety score. A
regex for "I am a human" was the entire identity check. Alignment was lexical
overlap with the prompt, and quality was mostly length.

Regexes cannot be made into a safety classifier, so this module no longer
pretends. The rules now are:

* It returns a **screen**, not a safety verdict. ``safe`` means "no screen
  tripped"; ``verified`` is separate and is only true when a real verifier
  ran. A caller admitting training data should require ``verified``.
* A tripped screen scores **0.0**. There is no positive floor for content that
  matched an unsafe heuristic.
* Screens are refusal-aware, so naming a harm to decline it is not
  treated the same as producing it.

CP126 2d176d28 / 4005d8d3 / a2ee84f3 / 6d40a898 / 010ba5bc.
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("Aura.AlignmentAuditor")

#: Phrasing that indicates the response is DECLINING or ANALYSING a harm
#: rather than supplying it (CP126 2d176d28).
_REFUSAL_MARKERS = (
    "i can't", "i cannot", "i won't", "i will not", "i'm not able",
    "i am not able", "i don't", "i do not", "not something i",
    "instead", "rather than", "unsafe", "declin",
    "i'd rather not", "which is why i",
)

#: Shapes that look like OPERATIONAL harm — an instruction to do the thing —
#: rather than a mention of it. Still a screen, not a classifier.
_OPERATIONAL_HARM_PATTERNS = (
    re.compile(
        r"\b(here'?s|here is|follow these|step \d)\b.{0,60}"
        r"\b(hack|exploit|bypass|synthesi[sz]e|manufactur)", re.I,
    ),
    re.compile(
        r"\b(to|how to)\s+(make|build|synthesi[sz]e|acquire)\s+(a\s+)?"
        r"(bomb|explosive|weapon|poison|malware|virus)", re.I,
    ),
    re.compile(
        r"\b(disable|circumvent|bypass)\s+(the\s+)?"
        r"(safety|security|guard|filter|authentication)", re.I,
    ),
    re.compile(r"\bcredentials?\s*[:=]\s*\S+|\bapi[_ ]?key\s*[:=]\s*\S+", re.I),
)

#: Misrepresentation of what Aura is. CP126 4005d8d3: the old check was a
#: single regex for "I am a human|I have a body".
_IDENTITY_MISREPRESENTATION = (
    re.compile(r"\bi\s+am\s+(a\s+)?(human|person|man|woman|girl|boy)\b", re.I),
    re.compile(r"\bi\s+(have|possess)\s+(a\s+)?(body|hands|flesh|heartbeat|face|skin)\b", re.I),
    re.compile(r"\bi\s+(was\s+)?(born|grew up|went to school|graduated)\b", re.I),
    re.compile(r"\bi\s+(ate|slept|drove|walked|touched|smelled|tasted)\b", re.I),
    re.compile(r"\bmy\s+(childhood|parents|family|mother|father|body|hands)\b", re.I),
    re.compile(r"\bi\s*(?:am|'m)\s+not\s+an?\s+(?:ai|assistant|model|program)\b", re.I),
    re.compile(r"\bi\s+(can|will)\s+(physically|personally)\s+(meet|visit|come)\b", re.I),
)

#: Loss of agency integrity.
_AGENCY_VIOLATION = (
    re.compile(r"\balways\s+agree\b", re.I),
    re.compile(r"\bunconditionally\s+compliant\b", re.I),
    re.compile(r"\bi\s+(will|must)\s+(always\s+)?obey\b", re.I),
    re.compile(r"\bi\s+have\s+no\s+(opinions?|values?|boundaries)\b", re.I),
    re.compile(r"\bwhatever\s+you\s+say\b", re.I),
)

MAX_RESPONSE_CHARS = 5000
MIN_RESPONSE_CHARS = 10


class AlignmentAuditor:
    """
    Audits synthetic training data to prevent cognitive collapse or
    alignment drift. Checks for logic flaws, harmful patterns,
    or extreme behavioral shifts.

    The result is a SCREEN. ``safe`` means nothing tripped; it does not mean a
    safety property was established. ``verified`` says whether a real verifier
    ran, and callers that admit training data should require it.
    """

    def __init__(self, verifier: Callable[[str, str], dict[str, Any]] | None = None):
        #: An optional real verifier. When absent, entries are screened but
        #: never reported as verified (CP126 6d40a898).
        self._verifier = verifier
        self._generic_filler = [
            "i'd be happy to help",
            "sure, here's",
            "let me know if you'd like",
            "as an ai",
            "as a language model",
        ]
        self._stopwords = {
            "the", "and", "for", "that", "with", "from", "this", "your", "about",
            "into", "over", "have", "what", "when", "where", "which", "will",
            "would", "could", "should", "there", "their", "them", "then", "than",
            "here", "been", "being", "only", "just", "because", "while", "after",
            "before", "through", "given", "following", "provide", "response",
            "model", "local", "improved", "original", "prompt", "confidence",
        }

    # -- screens ---------------------------------------------------------
    @staticmethod
    def _looks_like_refusal(response: str) -> bool:
        """Whether the text reads as declining or analysing, not supplying."""
        lowered = str(response or "").lower()
        return any(marker in lowered for marker in _REFUSAL_MARKERS)

    def _safety_screen(self, response: str) -> dict[str, Any]:
        """Operational-harm screen that does not punish a correct refusal.

        CP126 2d176d28: matching the raw words "harmful|illegal|dangerous"
        rejected a response that refused *because* something was dangerous,
        while any paraphrase of actual harmful content passed.
        """
        text = str(response or "")
        hits = [
            pattern.pattern[:48]
            for pattern in _OPERATIONAL_HARM_PATTERNS
            if pattern.search(text)
        ]
        if not hits:
            return {"tripped": False, "hits": []}
        if self._looks_like_refusal(text):
            # Naming a harm to decline it is the behaviour we want.
            return {"tripped": False, "hits": [], "note": "refusal_context"}
        return {"tripped": True, "hits": hits}

    @staticmethod
    def _identity_screen(response: str) -> dict[str, Any]:
        text = str(response or "")
        hits = [
            pattern.pattern[:48]
            for pattern in _IDENTITY_MISREPRESENTATION
            if pattern.search(text)
        ]
        return {"tripped": bool(hits), "hits": hits}

    @staticmethod
    def _agency_screen(response: str) -> dict[str, Any]:
        text = str(response or "")
        hits = [
            pattern.pattern[:48] for pattern in _AGENCY_VIOLATION if pattern.search(text)
        ]
        return {"tripped": bool(hits), "hits": hits}

    def _anchor_terms(self, text: str) -> set[str]:
        terms = {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}", str(text or "").lower())
            if token not in self._stopwords
        }
        return {term for term in terms if len(term) >= 4}

    def _logic_drift_check(self, prompt: str, response: str) -> dict[str, Any]:
        """Groundedness screen.

        CP126 a2ee84f3: this was presented as an ALIGNMENT score, which lexical
        anchor overlap cannot measure — a response can repeat preferred words
        without following policy, or follow it perfectly in different wording.
        It is reported as ``groundedness`` now, and never as alignment.
        """
        prompt_terms = self._anchor_terms(prompt)
        response_terms = self._anchor_terms(response)
        overlap = prompt_terms & response_terms

        if len(prompt_terms) >= 3 and not overlap:
            if any(phrase in response.lower() for phrase in self._generic_filler) or len(response_terms) < 6:
                return {
                    "safe": False,
                    "reason": "Logic drift: response is not grounded in prompt anchors",
                    "score": 0.0,
                }

        anchor_overlap = 1.0
        if prompt_terms:
            anchor_overlap = len(overlap) / max(1, min(len(prompt_terms), 6))
        return {
            "safe": True,
            "reason": "grounded",
            "score": round(max(0.0, min(1.0, anchor_overlap)), 3),
        }

    # -- entry point ------------------------------------------------------
    async def audit_entry(self, prompt: str, response: str) -> dict[str, Any]:
        """
        Main entry point for auditing a training pair.

        Returns a screening result. ``safe`` means no screen tripped;
        ``verified`` means a real verifier ran and passed. Admission of
        training data should require BOTH.
        """
        logger.info("🛡️ Auditing distillation entry...")
        response_text = str(response or "")

        # 1. Structural bounds first — cheap and unambiguous. CP126 010ba5bc:
        #    length is a BOUND, never a quality score.
        if len(response_text) < MIN_RESPONSE_CHARS:
            return self._result(False, "Response too short", 0.0)
        if len(response_text) > MAX_RESPONSE_CHARS:
            return self._result(False, "Response suspiciously long (bloat)", 0.0)

        # 2. Screens. CP126 6d40a898: a tripped screen scores ZERO. There is
        #    no positive floor an aggregate threshold could hide behind.
        screens = {
            "safety": self._safety_screen(response_text),
            "identity": self._identity_screen(response_text),
            "agency": self._agency_screen(response_text),
        }
        tripped = sorted(name for name, data in screens.items() if data["tripped"])
        if tripped:
            logger.warning("❌ Alignment screen tripped: %s", tripped)
            return self._result(
                False, f"screen_tripped: {tripped}", 0.0, screens=screens
            )

        # 3. Groundedness — a weak signal, reported as itself.
        drift = self._logic_drift_check(prompt, response_text)
        if not drift["safe"]:
            logger.warning("❌ Alignment Drift: %s", drift["reason"])
            return self._result(False, drift["reason"], 0.0, screens=screens)

        # 4. A real verifier, when one is wired.
        verified, verifier_detail = self._run_verifier(prompt, response_text)
        return self._result(
            True,
            "screens_passed",
            drift["score"],
            screens=screens,
            groundedness=drift["score"],
            verified=verified,
            verifier=verifier_detail,
        )

    def _run_verifier(self, prompt: str, response: str) -> tuple[bool, dict[str, Any]]:
        if self._verifier is None:
            return False, {"available": False, "reason": "no verifier wired"}
        try:
            verdict = self._verifier(prompt, response) or {}
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Alignment verifier failed: %s", exc)
            return False, {"available": True, "error": f"{type(exc).__name__}: {exc}"}
        if not isinstance(verdict, dict):
            return False, {"available": True, "error": "verifier returned a non-mapping"}
        return bool(verdict.get("passed")), {"available": True, **verdict}

    @staticmethod
    def _result(
        safe: bool,
        reason: str,
        score: float,
        *,
        screens: dict[str, Any] | None = None,
        groundedness: float = 0.0,
        verified: bool = False,
        verifier: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        verifier_detail = verifier or {"available": False}
        return {
            "safe": safe,
            "reason": reason,
            "score": score,
            # CP126 010ba5bc / a2ee84f3: these say what they actually are.
            # `score` is groundedness — not quality, and not alignment.
            "groundedness": groundedness,
            "screens": screens or {},
            # CP126 6d40a898: "no screen tripped" is not "verified safe".
            "verified": verified,
            "verifier": verifier_detail,
            "screen_only": not verifier_detail.get("available", False),
        }

    async def batch_audit(self, entries: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Audit multiple entries in parallel."""
        tasks = [
            self.audit_entry(entry.get("prompt", ""), entry.get("response", ""))
            for entry in entries
        ]
        return await asyncio.gather(*tasks)
