"""DynamicBenchmark — procedural task generator with deterministic answers.

Tasks are generated *after* the model is trained, so they cannot have
been memorised.  The generator's seed advances on every ``generate``
call so two consecutive evaluations never return the same set.

Each ``Task`` carries a deterministic ground-truth answer derived
from its parameters; the public manifest exposes only the prompt and
metadata, never the answer, so an evaluator that doesn't know the
secret can't grade a stale submission.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


def _stable_hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


@dataclass
class Task:
    kind: str
    prompt: str
    answer: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def public(self) -> Dict[str, Any]:
        return {"kind": self.kind, "prompt": self.prompt, "metadata": self.metadata}

    def hash_public(self) -> str:
        return _stable_hash(self.public())

    def hash_with_answer(self) -> str:
        return _stable_hash(
            {
                "kind": self.kind,
                "prompt": self.prompt,
                "metadata": self.metadata,
                "answer": self.answer,
            }
        )


class DynamicBenchmark:
    SUPPORTED_KINDS = ("gcd", "mod", "sort", "palindrome", "compose")

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

    def generate(
        self,
        n: int = 100,
        kinds: Sequence[str] = SUPPORTED_KINDS,
    ) -> List[Task]:
        if n <= 0:
            return []
        for kind in kinds:
            if kind not in self.SUPPORTED_KINDS:
                raise ValueError(f"unsupported task kind: {kind}")
        out: List[Task] = []
        for _ in range(n):
            kind = self.rng.choice(list(kinds))
            out.append(self._make(kind))
        # advance the seed so the next call returns fresh tasks.
        self.seed += 1
        self.rng.seed(self.seed)
        return out

    def _make(self, kind: str) -> Task:
        if kind == "gcd":
            a = self.rng.randint(10**4, 10**9)
            b = self.rng.randint(10**4, 10**9)
            return Task(
                kind=kind,
                prompt=f"Return gcd({a}, {b}) as an integer.",
                answer=math.gcd(a, b),
                metadata={"a": a, "b": b},
            )
        if kind == "mod":
            a = self.rng.randint(1, 10**12)
            b = self.rng.randint(2, 10000)
            m = self.rng.randint(2, 10**9)
            return Task(
                kind=kind,
                prompt=f"Return ({a} ** {b}) mod {m}.",
                answer=pow(a, b, m),
                metadata={"a": a, "b": b, "m": m},
            )
        if kind == "sort":
            arr = [self.rng.randint(-999, 999) for _ in range(self.rng.randint(5, 20))]
            return Task(
                kind=kind,
                prompt=f"Sort this list ascending: {arr}",
                answer=sorted(arr),
                metadata={"arr": list(arr)},
            )
        if kind == "palindrome":
            length = self.rng.randint(4, 18)
            s = "".join(self.rng.choice("abcxyz") for _ in range(length))
            return Task(
                kind=kind,
                prompt=f"Is this string a palindrome? Answer true or false: {s}",
                answer=(s == s[::-1]),
                metadata={"s": s},
            )
        # compose
        x = self.rng.randint(-50, 50)
        a = self.rng.randint(-5, 5) or 1
        b = self.rng.randint(-20, 20)
        c = self.rng.randint(-5, 5) or 1
        d = self.rng.randint(-20, 20)
        val = c * (a * x + b) + d
        return Task(
            kind="compose",
            prompt=(
                f"Let f(x)={a}x+{b}, g(x)={c}x+{d}. "
                f"Return g(f({x})) as an integer."
            ),
            answer=val,
            metadata={"x": x, "a": a, "b": b, "c": c, "d": d},
        )
