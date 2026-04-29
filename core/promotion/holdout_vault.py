"""HoldoutVault + LeakageDetector — anti-contamination layer.

The vault stores answer-bearing tasks on disk under a hash-keyed
JSON.  ``public_manifest()`` returns prompts + metadata so an
evaluator can solve them without seeing the answer; ``get_answer``
is the only path to the ground truth and is privileged.

``LeakageDetector`` checks similarity of held-out prompts against a
training-text corpus using character n-gram Jaccard similarity.  A
hit above the threshold flags the task as contaminated and the gate
should refuse to use it.
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

from core.promotion.dynamic_benchmark import Task


def _atomic_write_text(path: Union[str, Path], text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class VaultMissError(KeyError):
    pass


class HoldoutVault:
    """Append-only JSON store of (task_hash -> task_dict)."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.records: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            try:
                self.records = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(self.records, dict):
                    self.records = {}
            except json.JSONDecodeError:
                self.records = {}

    def add(self, tasks: Sequence[Task]) -> List[str]:
        ids: List[str] = []
        for task in tasks:
            tid = task.hash_public()
            self.records[tid] = dataclasses.asdict(task)
            ids.append(tid)
        self._flush()
        return ids

    def public_manifest(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for tid, payload in self.records.items():
            out.append(
                {
                    "id": tid,
                    "kind": payload.get("kind"),
                    "prompt": payload.get("prompt"),
                    "metadata": payload.get("metadata", {}),
                }
            )
        return out

    def get_answer(self, task_id: str) -> Any:
        if task_id not in self.records:
            raise VaultMissError(task_id)
        return self.records[task_id]["answer"]

    def size(self) -> int:
        return len(self.records)

    def _flush(self) -> None:
        _atomic_write_text(
            self.path, json.dumps(self.records, indent=2, sort_keys=True)
        )


class LeakageDetector:
    """Character n-gram Jaccard similarity between prompts + training texts."""

    def __init__(self, ngram: int = 5, threshold: float = 0.65):
        if ngram < 1:
            raise ValueError("ngram must be >= 1")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.ngram = ngram
        self.threshold = threshold

    def _ngrams(self, text: str) -> set:
        text = " ".join(text.lower().split())
        if not text:
            return set()
        if len(text) <= self.ngram:
            return {text}
        return {text[i : i + self.ngram] for i in range(len(text) - self.ngram + 1)}

    def similarity(self, a: str, b: str) -> float:
        A = self._ngrams(a)
        B = self._ngrams(b)
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)

    def contaminated(
        self, task: Task, training_texts: Sequence[str]
    ) -> Tuple[bool, float]:
        best = 0.0
        for text in training_texts:
            score = self.similarity(task.prompt, text)
            if score > best:
                best = score
                if best >= 1.0:
                    break
        return best >= self.threshold, best

    def filter_clean(
        self, tasks: Sequence[Task], training_texts: Sequence[str]
    ) -> List[Task]:
        return [t for t in tasks if not self.contaminated(t, training_texts)[0]]
