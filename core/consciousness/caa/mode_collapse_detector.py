from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List


class CollapseSeverity(str, Enum):
    NONE = "none"
    WATCH = "watch"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class CollapseSignal:
    severity: CollapseSeverity
    reasons: List[str]
    token_count: int
    unique_token_ratio: float
    repeated_line_count: int
    repeated_ngram: str = ""
    repeated_ngram_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["severity"] = self.severity.value
        return payload


class ModeCollapseDetector:
    """Detects repetition and lexical collapse in model generations."""

    def __init__(self) -> None:
        self._history: List[Dict[str, Any]] = []
        self._last_signal = CollapseSignal(
            severity=CollapseSeverity.NONE,
            reasons=[],
            token_count=0,
            unique_token_ratio=1.0,
            repeated_line_count=0,
        )

    @staticmethod
    def _max_repeated_ngram(tokens: List[str], n: int = 3) -> tuple[str, int]:
        counts: Dict[str, int] = {}
        best = ("", 0)
        if len(tokens) < n:
            return best
        for idx in range(len(tokens) - n + 1):
            key = " ".join(tokens[idx : idx + n])
            counts[key] = counts.get(key, 0) + 1
            if counts[key] > best[1]:
                best = (key, counts[key])
        return best

    def observe(self, text: str) -> CollapseSignal:
        stripped = str(text or "").strip()
        tokens = [token for token in stripped.lower().split() if token]
        lines = [line.strip().lower() for line in stripped.splitlines() if line.strip()]
        unique_ratio = float(len(set(tokens)) / len(tokens)) if tokens else 1.0
        repeated_lines = 0
        if lines:
            line_counts: Dict[str, int] = {}
            for line in lines:
                line_counts[line] = line_counts.get(line, 0) + 1
            repeated_lines = max(line_counts.values())
        repeated_ngram, repeated_ngram_count = self._max_repeated_ngram(tokens, n=3)
        reasons: List[str] = []
        severity = CollapseSeverity.NONE
        if repeated_ngram_count >= 4:
            reasons.append(f"repeated_trigram:{repeated_ngram_count}")
            severity = CollapseSeverity.CRITICAL
        elif repeated_ngram_count == 3:
            reasons.append("repeated_trigram:3")
            severity = CollapseSeverity.WARNING
        if repeated_lines >= 3:
            reasons.append(f"repeated_lines:{repeated_lines}")
            severity = CollapseSeverity.CRITICAL
        elif repeated_lines == 2:
            reasons.append("repeated_lines:2")
            severity = max(severity, CollapseSeverity.WATCH, key=lambda item: list(CollapseSeverity).index(item))
        if len(tokens) >= 80 and unique_ratio < 0.22:
            reasons.append(f"unique_ratio:{unique_ratio:.3f}")
            severity = CollapseSeverity.CRITICAL
        elif len(tokens) >= 48 and unique_ratio < 0.32:
            reasons.append(f"unique_ratio:{unique_ratio:.3f}")
            severity = CollapseSeverity.WARNING
        elif len(tokens) >= 24 and unique_ratio < 0.45 and severity == CollapseSeverity.NONE:
            reasons.append(f"unique_ratio:{unique_ratio:.3f}")
            severity = CollapseSeverity.WATCH
        signal = CollapseSignal(
            severity=severity,
            reasons=reasons,
            token_count=len(tokens),
            unique_token_ratio=unique_ratio,
            repeated_line_count=repeated_lines,
            repeated_ngram=repeated_ngram,
            repeated_ngram_count=repeated_ngram_count,
        )
        self._last_signal = signal
        self._history.append(signal.to_dict())
        if len(self._history) > 64:
            self._history = self._history[-64:]
        return signal

    def status(self) -> Dict[str, Any]:
        return {
            "last_signal": self._last_signal.to_dict(),
            "history_size": len(self._history),
            "recent": self._history[-5:],
        }
