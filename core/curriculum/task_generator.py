"""Generate targeted learning tasks for a detected gap.

For unit tests we only need a deterministic, gap-driven task; a real
implementation would have the LLM craft challenge prompts seeded from
recent failures.  The structure here keeps the surface stable so the
real generator drops in later.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.curriculum.gap_detector import GapReport


@dataclass
class LearningTask:
    task_id: str
    belief: str
    modality: str
    prompt: str
    expected: Any
    strategy: str = "default"
    iteration: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskGenerator:
    """Synthesises learning tasks from a ``GapReport`` plus a strategy."""

    def __init__(self) -> None:
        self._counter = 0

    def _new_id(self) -> str:
        return f"task-{uuid.uuid4().hex[:12]}"

    def generate(
        self,
        *,
        gap: GapReport,
        strategy: str,
        seed_prompt: Optional[str] = None,
        seed_expected: Optional[Any] = None,
        iteration: int = 0,
    ) -> LearningTask:
        if not gap.has_gap and seed_prompt is None:
            raise ValueError("generate() requires either a gap or a seed prompt")
        belief = gap.belief or (seed_expected.get("belief", "unknown") if isinstance(seed_expected, dict) else "unknown")
        modality = gap.modality or "text"
        prompt = seed_prompt or f"[{strategy}] practice belief={belief} modality={modality}"
        expected = seed_expected if seed_expected is not None else {"belief": belief}
        return LearningTask(
            task_id=self._new_id(),
            belief=belief,
            modality=modality,
            prompt=prompt,
            expected=expected,
            strategy=strategy,
            iteration=iteration,
            metadata={"gap_n_resolved": gap.n_resolved, "gap_mean_brier": gap.mean_brier},
        )
