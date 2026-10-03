"""core/learning/learning_selfreport.py — Aura answers 'what have you learned?'

The learning stack writes receipts everywhere — the lineage ledger, the
flywheel's practice stats, the preference store, the scheduler's state — but
until now none of it reached the conversation lane: ask Aura what she has
been learning and the model would improvise. This mirrors the incident
narrator's pattern: a bounded, receipt-backed context block that fires ONLY
on learning-shaped questions, so ordinary turns pay nothing and learning
claims in chat are grounded in the same numbers /api/system/learning serves.

Honesty is structural here: the block states what the ledger verdict allows
and instructs the draft to claim nothing beyond it. A refused cycle or an
empty practice log is reported with the same directness as a promotion.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from core.runtime.errors import FallbackClassification, record_degradation

logger = logging.getLogger("Aura.LearningSelfReport")

_RECOVERABLE = (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError, KeyError)

_LEARNING_QUESTION_RE = re.compile(
    r"(?:what (?:have|did|do) you (?:learn|practi[cs]e)"
    r"|(?:have|are) you (?:been )?(?:learning|practicing|training|improving)"
    r"|learn(?:ed|t)? (?:anything|something|lately|today|recently)"
    r"|your (?:learning|training|practice|weights?|self.?improvement)"
    r"|(?:gotten|getting|become|becoming) (?:better|smarter|more capable)"
    r"|weight.?(?:update|training|compounding)"
    r"|improve[d]? yourself"
    r"|teach(?:ing)? yourself)",
    re.IGNORECASE,
)


def _asks_about_learning(objective: str) -> bool:
    return bool(objective) and bool(_LEARNING_QUESTION_RE.search(objective))


def _asks_about_a_lookup(objective: str) -> bool:
    """Whether the question is about what this turn's sources say, not her training.

    LIVE 2026-10-03: "what did you learn", asked right after she searched,
    got this block and nothing about the search. When the language substrate
    reads the question as asking what was found and the turn carries sources
    (core/conversation/what_the_world_says.py), those sources answer it.
    """
    try:
        from core.conversation.turn_evidence_custody import turn_world_evidence
        from core.language.search_request import asks_what_was_found

        return bool(turn_world_evidence()) and asks_what_was_found(objective) is True
    except _RECOVERABLE as exc:
        record_degradation(
            "learning_selfreport", exc, action="read the question as about her learning",
            classification=FallbackClassification.SAFE_FALLBACK, severity="debug",
        )
        return False


class LearningSelfReport:
    """Receipt-backed learning status for the conversation lane."""

    def __init__(self) -> None:
        self._cache: str | None = None
        self._cache_at = 0.0
        self._cache_ttl_s = 60.0

    def get_context_injection(self, objective: str = "") -> str:
        """A bounded learning-status block; empty unless the question is
        learning-shaped. Numbers come from the same receipts the API serves."""
        if not _asks_about_learning(objective) or _asks_about_a_lookup(objective):
            return ""
        now = time.monotonic()
        if self._cache is not None and now - self._cache_at < self._cache_ttl_s:
            return self._cache
        try:
            block = self._build_block()
        except _RECOVERABLE as exc:
            record_degradation(
                "learning_selfreport",
                exc,
                action="answered learning question without grounded status block",
                classification=FallbackClassification.SAFE_FALLBACK,
                severity="debug",
            )
            return ""
        self._cache = block
        self._cache_at = now
        return block

    # ── collection (mirrors /api/system/learning, tolerant per-section) ──────

    def _build_block(self) -> str:
        lines = [
            "## LEARNING SELF-KNOWLEDGE (real receipts — cite honestly, do not invent)"
        ]
        lines.extend(self._weight_cycle_lines())
        lines.extend(self._practice_lines())
        lines.append(
            "Claim exactly what these receipts support and no more. Promotions mean "
            "'a gated update passed its sealed capability check'; refusals and empty "
            "practice logs are stated plainly. Never imply capability growth unless "
            "the ledger verdict above says the curve increased."
        )
        return "\n".join(lines)

    def _weight_cycle_lines(self) -> list[str]:
        try:
            from core.runtime.service_access import resolve_weight_compounding

            scheduler = resolve_weight_compounding(default=None)
            status = scheduler.get_status() if scheduler is not None else {}
        except _RECOVERABLE:
            status = {}
        lineage = status.get("lineage") or {}
        generations = int(lineage.get("generations", 0) or 0)
        if generations <= 0:
            last = str(status.get("last_status", "never_attempted") or "never_attempted")
            return [
                "- Weight-level learning: the compounding loop is armed but no training "
                f"generation has completed yet (last scheduler status: {last}). Idle "
                "windows with enough verified data trigger it; nothing to claim yet."
            ]
        promoted = int(lineage.get("promoted", 0) or 0)
        refused = int(lineage.get("refused", 0) or 0)
        verdict = str(lineage.get("verdict", "") or "UNKNOWN")
        line = (
            f"- Weight-level learning: {generations} recorded generation(s) — "
            f"{promoted} promoted, {refused} refused; ledger verdict: {verdict}. "
            f"Last cycle: {status.get('last_status', 'unknown')}"
        )
        trigger = status.get("last_generation_id")
        if trigger:
            line += f" (generation {trigger})"
        return [line + "."]

    def _practice_lines(self) -> list[str]:
        state: dict[str, Any] = {}
        try:
            from core.config import get_config

            path = Path(get_config().paths.data_dir) / "learning" / "selfplay_flywheel.json"
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    state = raw
        except _RECOVERABLE:
            state = {}
        bursts = int(state.get("bursts", 0) or 0)
        attempts = int(state.get("total_attempts", 0) or 0)
        if bursts <= 0:
            return [
                "- Idle practice (self-play): no practice bursts recorded yet — "
                "practice runs only in genuinely idle windows."
            ]
        correct = int(state.get("total_correct", 0) or 0)
        pairs = int(state.get("total_pairs", 0) or 0)
        if attempts <= 0:
            return [
                f"- Idle practice (self-play): {bursts} burst(s) fired but every one "
                "yielded to foreground work before answering — no attempts banked yet."
            ]
        rate = correct / attempts
        ema = state.get("correct_rate_ema")
        line = (
            f"- Idle practice (self-play): {bursts} burst(s), {attempts} verified "
            f"attempt(s), {correct} exactly correct (~{rate:.0%}); {pairs} win/loss "
            "preference pair(s) banked for the next training window"
        )
        if isinstance(ema, (int, float)):
            line += f"; recent correct-rate trend ~{float(ema):.0%}"
        lines = [line + "."]
        lines.extend(self._direction_lines())
        return lines

    def _direction_lines(self) -> list[str]:
        """What the Practice Director is aiming practice at, receipts cited —
        the honest answer to 'why are you practicing that?'. Resolved from
        the service spine only (never self-created — a hermetic test must
        not read the real machine's practice ledger)."""
        try:
            from core.runtime.service_access import resolve_practice_director

            director = resolve_practice_director(default=None)
            if director is None or not hasattr(director, "why"):
                return []
            direction = director.why()
        except _RECOVERABLE:
            return []
        if not direction:
            return []
        return [f"- {direction}"]


_selfreport: LearningSelfReport | None = None


def get_learning_selfreport() -> LearningSelfReport:
    global _selfreport
    if _selfreport is None:
        _selfreport = LearningSelfReport()
    return _selfreport


def reset_learning_selfreport_for_test() -> None:
    global _selfreport
    _selfreport = None


__all__ = [
    "LearningSelfReport",
    "get_learning_selfreport",
    "reset_learning_selfreport_for_test",
]
