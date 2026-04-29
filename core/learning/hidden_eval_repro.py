"""Reproducible hidden evaluation packs.

The public manifest exposes task prompts and answer hashes, not answers. A
third party with the private seed can regenerate the same pack and verify that
reported scores were produced against the sealed answers.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from core.promotion.dynamic_benchmark import Task
from core.runtime.atomic_writer import atomic_write_text


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HiddenEvalManifest:
    pack_id: str
    seed_hash: str
    answer_salt_hash: str
    task_count: int
    answer_hashes: Dict[str, str]
    public_tasks: List[Dict[str, Any]]
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HiddenEvalResult:
    pack_id: str
    score: float
    passed: int
    total: int
    answer_hash_ok: bool
    manifest_hash: str
    runtime_s: float
    failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HiddenEvalPack:
    """Sealed hidden eval set with reproducible public manifest."""

    def __init__(self, *, seed: int, answer_salt: str, task_count: int = 50):
        if task_count <= 0:
            raise ValueError("task_count must be positive")
        self.seed = int(seed)
        self.answer_salt = str(answer_salt)
        self.task_count = int(task_count)
        self.tasks = self._generate_tasks()
        self.pack_id = _sha({
            "seed": self.seed,
            "salt": self.answer_salt,
            "task_count": self.task_count,
            "tasks": [task.public() for task in self.tasks],
        })[:24]

    def manifest(self) -> HiddenEvalManifest:
        answer_hashes = {
            task.hash_public(): _sha({"salt": self.answer_salt, "answer": task.answer})
            for task in self.tasks
        }
        return HiddenEvalManifest(
            pack_id=self.pack_id,
            seed_hash=_sha(self.seed),
            answer_salt_hash=_sha(self.answer_salt),
            task_count=len(self.tasks),
            answer_hashes=answer_hashes,
            public_tasks=[
                {
                    "id": task.hash_public(),
                    "kind": task.kind,
                    "prompt": task.prompt,
                    "metadata": task.metadata,
                }
                for task in self.tasks
            ],
        )

    def manifest_hash(self) -> str:
        payload = self.manifest().to_dict()
        payload.pop("created_at", None)
        return _sha(payload)

    def evaluate(self, solver: Callable[[Task], Any]) -> HiddenEvalResult:
        start = time.time()
        failures: List[str] = []
        passed = 0
        manifest = self.manifest()
        for task in self.tasks:
            task_id = task.hash_public()
            expected_hash = manifest.answer_hashes[task_id]
            if expected_hash != _sha({"salt": self.answer_salt, "answer": task.answer}):
                failures.append(f"answer_hash_mismatch:{task_id}")
                continue
            try:
                predicted = solver(task)
            except Exception as exc:
                failures.append(f"{task_id}:solver_error:{type(exc).__name__}")
                continue
            if predicted == task.answer:
                passed += 1
            else:
                failures.append(f"{task_id}:wrong")
        total = len(self.tasks)
        return HiddenEvalResult(
            pack_id=self.pack_id,
            score=passed / max(1, total),
            passed=passed,
            total=total,
            answer_hash_ok=not any("answer_hash_mismatch" in item for item in failures),
            manifest_hash=self.manifest_hash(),
            runtime_s=round(time.time() - start, 6),
            failures=failures[:100],
        )

    def write_reproduction_bundle(self, output_dir: Path | str) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest_path = out / f"{self.pack_id}_manifest.json"
        atomic_write_text(
            manifest_path,
            json.dumps(self.manifest().to_dict(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return manifest_path

    def _generate_tasks(self) -> List[Task]:
        rng = random.Random(self.seed)
        tasks: List[Task] = []
        for idx in range(self.task_count):
            kind = rng.choice(["gcd", "mod", "sort", "palindrome", "compose"])
            if kind == "gcd":
                a = rng.randint(10_000, 900_000)
                b = rng.randint(10_000, 900_000)
                tasks.append(Task(kind, f"Return gcd({a}, {b}) as an integer.", math.gcd(a, b), {"a": a, "b": b, "idx": idx}))
            elif kind == "mod":
                a = rng.randint(2, 10_000)
                b = rng.randint(2, 500)
                m = rng.randint(3, 50_000)
                tasks.append(Task(kind, f"Return ({a} ** {b}) mod {m}.", pow(a, b, m), {"a": a, "b": b, "m": m, "idx": idx}))
            elif kind == "sort":
                arr = [rng.randint(-100, 100) for _ in range(rng.randint(4, 12))]
                tasks.append(Task(kind, f"Sort this list ascending: {arr}", sorted(arr), {"arr": arr, "idx": idx}))
            elif kind == "palindrome":
                s = "".join(rng.choice("abcxyz") for _ in range(rng.randint(3, 12)))
                tasks.append(Task(kind, f"Is this string a palindrome? Answer true or false: {s}", s == s[::-1], {"s": s, "idx": idx}))
            else:
                x = rng.randint(-30, 30)
                a = rng.randint(-5, 5) or 1
                b = rng.randint(-10, 10)
                c = rng.randint(-5, 5) or 1
                d = rng.randint(-10, 10)
                answer = c * (a * x + b) + d
                tasks.append(
                    Task(
                        "compose",
                        f"Let f(x)={a}x+{b}, g(x)={c}x+{d}. Return g(f({x})) as an integer.",
                        answer,
                        {"x": x, "a": a, "b": b, "c": c, "d": d, "idx": idx},
                    )
                )
        return tasks


__all__ = ["HiddenEvalManifest", "HiddenEvalPack", "HiddenEvalResult"]
