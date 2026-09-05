"""Reasoning memory — turn failures into future verifiers (Reflexion).

A storage bin remembers facts. A reasoning substrate remembers *how reasoning went
wrong* and refuses to repeat it. After every amplified episode this records a short
reflection keyed by a task signature: what the task was, whether it passed, which
verifier caught what, and a one-line lesson. Before the next episode of a similar
shape, :meth:`recall` pulls the relevant *failure modes* — not just similar facts —
so the amplifier can pre-arm the right guard:

    "Last time a repo-audit answer trusted a filename that did not exist →
     require file-span evidence this time."

Persisted as JSONL in the Aura data dir (durable atomic append), bounded, and
retrieved by content + task-type overlap. Pure local; no model required to read it.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_append_text
from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.ReasoningMemory")

_MAX_RECORDS = 2000
_STOPWORDS = frozenset(
    "the a an and or but if then so to of in on for with as is are was were be been "
    "this that these those it its i you we they he she him her them what which who how "
    "why when where can could would should will do does did not no yes".split()
)


def _signature_tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zA-Z]{3,}", str(text or "").lower()) if w not in _STOPWORDS}


@dataclass
class ReasoningReflection:
    task_type: str
    signature: str
    passed: bool
    lesson: str
    failure_mode: str = ""
    verifier_issues: list[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "signature": self.signature[:200],
            "passed": self.passed,
            "lesson": self.lesson[:300],
            "failure_mode": self.failure_mode[:200],
            "verifier_issues": self.verifier_issues[:6],
            "ts": round(self.ts, 2),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReasoningReflection:
        return cls(
            task_type=str(d.get("task_type", "generic")),
            signature=str(d.get("signature", "")),
            passed=bool(d.get("passed", False)),
            lesson=str(d.get("lesson", "")),
            failure_mode=str(d.get("failure_mode", "")),
            verifier_issues=list(d.get("verifier_issues", []) or []),
            ts=float(d.get("ts", 0.0) or 0.0),
        )


class ReasoningMemory:
    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else self._default_path()
        self._cache: list[ReasoningReflection] | None = None

    @staticmethod
    def _default_path() -> Path:
        try:
            from core.utils.paths import aura_data_dir

            base = aura_data_dir()
        except (ImportError, RuntimeError, OSError):
            base = state_root() / "data"
        base.mkdir(parents=True, exist_ok=True)
        return base / "reasoning_reflections.jsonl"

    def _load(self) -> list[ReasoningReflection]:
        if self._cache is not None:
            return self._cache
        records: list[ReasoningReflection] = []
        try:
            if self._path.exists():
                for line in self._path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(ReasoningReflection.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, ValueError, TypeError):
                        continue
        except (OSError, RuntimeError) as exc:
            record_degradation("reasoning_memory_load", exc)
        self._cache = records[-_MAX_RECORDS:]
        return self._cache

    def record(
        self,
        *,
        task_type: str,
        objective: str,
        passed: bool,
        lesson: str = "",
        verifier_issues: list[str] | None = None,
    ) -> ReasoningReflection:
        """Store one reflection. Failures derive a reusable failure-mode lesson."""
        verifier_issues = verifier_issues or []
        failure_mode = "" if passed else self._derive_failure_mode(verifier_issues, objective)
        if not lesson:
            lesson = (
                "verified clean" if passed
                else (failure_mode or "answer failed verification; re-check before asserting")
            )
        reflection = ReasoningReflection(
            task_type=str(task_type or "generic"),
            signature=str(objective or "")[:200],
            passed=bool(passed),
            lesson=lesson,
            failure_mode=failure_mode,
            verifier_issues=list(verifier_issues)[:6],
        )
        try:
            atomic_append_text(self._path, json.dumps(reflection.to_dict(), ensure_ascii=False) + "\n")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("reasoning_memory_record", exc)
        if self._cache is not None:
            self._cache.append(reflection)
            self._cache = self._cache[-_MAX_RECORDS:]
        return reflection

    def recall(
        self,
        objective: str,
        *,
        task_type: str | None = None,
        limit: int = 3,
        failures_only: bool = True,
    ) -> list[ReasoningReflection]:
        """Pull the most relevant prior reflections — failure modes first.

        Relevance = signature token overlap, boosted for same task type and for
        failures (which carry the reusable guard). The point of recall is to be
        reminded of how this *kind* of task previously went wrong.
        """
        records = self._load()
        if not records:
            return []
        want = _signature_tokens(objective)
        norm_type = str(task_type or "").strip().lower()
        scored: list[tuple[float, ReasoningReflection]] = []
        for rec in records:
            if failures_only and rec.passed and not rec.failure_mode:
                continue
            overlap = len(want & _signature_tokens(rec.signature))
            if overlap == 0 and rec.task_type != norm_type:
                continue
            score = float(overlap)
            if norm_type and rec.task_type == norm_type:
                score += 1.5
            if not rec.passed:
                score += 1.0  # failures are the load-bearing memories
            scored.append((score, rec))
        scored.sort(key=lambda kv: (kv[0], kv[1].ts), reverse=True)
        return [r for _s, r in scored[:limit]]

    def as_guard_text(self, objective: str, *, task_type: str | None = None, limit: int = 3) -> str:
        """Render recalled failure modes as a short guard block for a prompt."""
        hits = self.recall(objective, task_type=task_type, limit=limit)
        if not hits:
            return ""
        lines = [f"- {h.lesson}" for h in hits if h.lesson]
        if not lines:
            return ""
        return "Lessons from similar past reasoning (avoid repeating these):\n" + "\n".join(lines)

    @staticmethod
    def _derive_failure_mode(verifier_issues: list[str], objective: str) -> str:
        joined = " ".join(verifier_issues).lower()
        if "not found" in joined or "path" in joined:
            return "trusted a file/path reference that did not exist → require file-span evidence"
        if "arithmetic" in joined:
            return "made a calculation error → run the numbers in the symbolic sandbox"
        if "non-sequitur" in joined:
            return "drew a conclusion the premises did not support → check each inference step"
        if "ungrounded" in joined:
            return "asserted a fact without grounding → cite supplied evidence or hedge"
        if "syntax" in joined or "compile" in joined:
            return "produced code that does not compile → verify with py_compile before answering"
        if "not actionable" in joined or "verification step" in joined:
            return "wrote a vague plan → make steps actionable and end with a verification step"
        if verifier_issues:
            return f"verifier flagged: {verifier_issues[0][:120]}"
        return ""


_instance: ReasoningMemory | None = None


def get_reasoning_memory() -> ReasoningMemory:
    global _instance
    if _instance is None:
        _instance = ReasoningMemory()
    return _instance
