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
import importlib.util
import logging
import math
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Iterable

logger = logging.getLogger("Brain.ContextGate")


# ── Utilities ─────────────────────────────────────────────────────────────

def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


@lru_cache(maxsize=1)
def _optional_tiktoken_encoding() -> Any:
    """Return a tokenizer when one is locally installed.

    Context gating must not depend on a cloud SDK or network download. If
    tiktoken is unavailable, the deterministic fallback below deliberately
    overestimates mixed code, punctuation-heavy, and non-English text.
    """
    if importlib.util.find_spec("tiktoken") is None:
        return None
    try:
        import tiktoken  # type: ignore

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def estimate_tokens(text: str) -> int:
    """Conservative prompt-token estimate.

    The old len/3.5 heuristic under-budgeted code, JSON, CJK, emoji, and
    punctuation-heavy terminal output. This function uses a real tokenizer
    when present and otherwise takes the maximum of several deterministic
    heuristics so budget errors fail toward compaction instead of overflow.
    """
    text = str(text or "")
    if not text:
        return 0
    encoding = _optional_tiktoken_encoding()
    if encoding is not None:
        try:
            return max(1, len(encoding.encode(text)))
        except Exception:
            pass

    words = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text)
    cjk = sum(1 for ch in text if "\u3400" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff" or "\uac00" <= ch <= "\ud7af")
    non_ascii = sum(1 for ch in text if ord(ch) > 127) - cjk
    punctuation = sum(1 for ch in text if ch in "{}[]()<>.,;:/\\|`~!@#$%^&*-+=?\"'")
    lines = text.count("\n") + 1
    codeish = sum(1 for ch in text if ch in "{}[]=;") + len(re.findall(r"\b(def|class|async|await|return|import|from|const|let|function)\b", text))

    char_model = math.ceil(len(text) / 3.2)
    lexical_model = math.ceil(len(words) * 1.18 + punctuation * 0.18 + cjk * 0.9 + non_ascii * 0.45)
    structure_model = math.ceil(lines * 1.5 + codeish * 0.6)
    return max(1, char_model, lexical_model, structure_model)


def _trim_to_token_budget(text: str, max_tokens: int) -> str:
    text = str(text or "")
    if estimate_tokens(text) <= max_tokens:
        return text
    marker = "\n...[compacted]"
    budget = max(1, int(max_tokens))
    low, high = 0, len(text)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid].rstrip() + marker
        if estimate_tokens(candidate) <= budget:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best or marker.strip()


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

        if estimate_tokens(text) > self.max_tokens:
            text = _trim_to_token_budget(text, self.max_tokens)
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
        dropped: list[ContextBlock] = []
        used = 0

        for block in essentials + candidates:
            cost = estimate_tokens(block.content)
            if block.essential or used + cost <= token_budget:
                selected.append(block)
                used += cost
            else:
                dropped.append(block)

        if dropped:
            dropped_summary = ", ".join(
                f"{b.id}(p={b.priority:.2f},s={b.salience:.2f},t={estimate_tokens(b.content)})"
                for b in dropped[:5]
            )
            logger.info(
                "ContextGate dropped %d blocks (budget=%d, used=%d): %s%s",
                len(dropped), token_budget, used, dropped_summary,
                f" +{len(dropped)-5} more" if len(dropped) > 5 else "",
            )

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
