"""core/brain/llm/context_gate.py — Attentional Context Gate

Selects *what* enters the LLM prompt.  This is the prompt-level equivalent
of a thalamic filter: identity and user task always survive; background
telemetry only enters when it is salient, changed, or critical.

The problem this solves:
  Without gating, the system prompt concatenates identity + RAG + affect +
  continuity + goals + temporal finitude + meta-qualia + personhood modules
  + social memory + shared ground + discourse + capabilities + task status
  + few-shot examples + structural constraints … then trims *late*.  Even
  under the character cap, the prompt can become noisy.  The model does not
  overflow, but it can still get overloaded by irrelevant self-telemetry.

The right model:
  Do not ask "how do I trim this giant prompt?"
  Ask "which 5-8 facts are actually allowed into awareness this turn?"
"""
from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

logger = logging.getLogger("Brain.ContextGate")


# ── Utilities ─────────────────────────────────────────────────────────────

def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate.

    Keep this conservative.  For prompt budgeting, undercounting is worse
    than overcounting.
    """
    text = str(text or "")
    return max(1, math.ceil(len(text) / 3.5))


def _fingerprint(text: str) -> str:
    return hashlib.blake2b(str(text or "").encode("utf-8"), digest_size=8).hexdigest()


# ── Context Block ─────────────────────────────────────────────────────────

@dataclass
class ContextBlock:
    """A candidate block of text that might enter the system prompt."""
    id: str
    content: str
    priority: float = 0.5              # 0..1  (how important is this category?)
    salience: float = 0.5              # 0..1  (how relevant is it right now?)
    source: str = "unknown"
    essential: bool = False            # if True, always included
    max_tokens: int = 256
    timestamp: float = field(default_factory=time.time)
    include_if: Callable[[], bool] | None = None

    def compact(self) -> "ContextBlock":
        """Return a copy with content trimmed to max_tokens."""
        text = str(self.content or "").strip()
        if not text:
            return self

        max_chars = int(self.max_tokens * 3.5)
        if len(text) > max_chars:
            text = text[: max(0, max_chars - 20)].rstrip() + "\n...[compacted]"
        return ContextBlock(
            id=self.id,
            content=text,
            priority=self.priority,
            salience=self.salience,
            source=self.source,
            essential=self.essential,
            max_tokens=self.max_tokens,
            timestamp=self.timestamp,
            include_if=self.include_if,
        )


# ── Delta Tracker ─────────────────────────────────────────────────────────

class ContextDeltaTracker:
    """Tracks which internal-state values changed enough to deserve prompt space."""

    DEFAULT_THRESHOLDS = {
        "valence": 0.20,
        "arousal": 0.20,
        "curiosity": 0.25,
        "free_energy": 0.15,
        "phi": 0.10,
        "vitality": 0.10,
        "cpu_usage": 25.0,
        "vram_usage": 20.0,
        "last_thought_ms": 1500.0,
        "failure_pressure": 0.15,
    }

    def __init__(self):
        self._last: dict[str, float] = {}

    def changed(
        self,
        key: str,
        value: Any,
        *,
        threshold: float | None = None,
        critical: float | None = None,
    ) -> bool:
        """Return True if *value* moved enough from the last-seen value for *key*.

        Parameters
        ----------
        key : str
            Name of the metric.
        value : Any
            Current numeric value.
        threshold : float, optional
            Minimum absolute change to count as "moved".  Defaults to the
            per-key value in ``DEFAULT_THRESHOLDS``.
        critical : float, optional
            If the absolute value exceeds *critical*, always report changed
            (regardless of delta).  Useful for abnormal ranges.
        """
        try:
            v = float(value)
        except Exception:
            return False

        if critical is not None:
            # Always surface critical abnormal values.
            if key in {"cpu_usage", "vram_usage", "last_thought_ms"}:
                if v >= critical:
                    self._last[key] = v
                    return True
            else:
                if abs(v) >= critical:
                    self._last[key] = v
                    return True

        if key not in self._last:
            self._last[key] = v
            return False

        t = float(threshold if threshold is not None else self.DEFAULT_THRESHOLDS.get(key, 0.20))
        old = self._last[key]
        if abs(v - old) >= t:
            self._last[key] = v
            return True

        return False


# ── Attentional Gate ──────────────────────────────────────────────────────

class AttentionalContextGate:
    """Selects what enters the LLM prompt.

    Identity and user task always survive; background telemetry only enters
    when it is salient, changed, or critical.
    """

    def __init__(self):
        self.delta = ContextDeltaTracker()
        self._last_seen_blocks: dict[str, str] = {}

    def should_include_block(
        self,
        block: ContextBlock,
        *,
        focus_sources: set[str] | None = None,
    ) -> bool:
        if not block.content.strip():
            return False
        if block.essential:
            return True
        if block.include_if is not None:
            try:
                if not bool(block.include_if()):
                    return False
            except Exception:
                return False

        focus_sources = focus_sources or set()
        if block.source in focus_sources:
            return True

        # Salience gate: low-priority, low-salience blocks do not enter.
        score = (block.priority * 0.65) + (block.salience * 0.35)
        if score < 0.48:
            return False

        # Duplicate gate: exact same block → skip unless high priority.
        fp = _fingerprint(block.content)
        previous = self._last_seen_blocks.get(block.id)
        self._last_seen_blocks[block.id] = fp
        if previous == fp and not block.essential and score < 0.72:
            return False

        return True

    def select(
        self,
        blocks: Iterable[ContextBlock],
        *,
        token_budget: int,
        focus_sources: set[str] | None = None,
    ) -> list[ContextBlock]:
        """Select and budget context blocks under the given token limit."""
        candidates = []
        essentials = []

        for raw in blocks:
            block = raw.compact()
            if not self.should_include_block(block, focus_sources=focus_sources):
                continue
            if block.essential:
                essentials.append(block)
            else:
                candidates.append(block)

        # Essential blocks first, then by priority/salience/recency.
        candidates.sort(
            key=lambda b: (b.priority, b.salience, b.timestamp),
            reverse=True,
        )

        selected: list[ContextBlock] = []
        used = 0

        for block in essentials + candidates:
            cost = estimate_tokens(block.content)
            if block.essential or used + cost <= token_budget:
                selected.append(block)
                used += cost

        logger.debug(
            "ContextGate selected %d blocks / approx %d tokens budget=%d",
            len(selected),
            used,
            token_budget,
        )
        return selected


# ── Module-level singleton ────────────────────────────────────────────────

_gate: AttentionalContextGate | None = None


def get_context_gate() -> AttentionalContextGate:
    global _gate
    if _gate is None:
        _gate = AttentionalContextGate()
    return _gate
