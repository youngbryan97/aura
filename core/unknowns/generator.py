"""UnknownUnknownGenerator — seed mutation + novelty filtering.

For each kind in ``DynamicBenchmark.SUPPORTED_KINDS`` the generator
knows a mutation strategy that pushes the task's parameters toward
edges the seed didn't sample (huge common factors, near-zero
moduli, lists with extreme values, near-palindromic strings, etc.).

Mutations land in the ``NoveltyArchive`` only if they're sufficiently
distant from every previous prompt — otherwise the loop just
re-samples.  When integrated with F9 curriculum, this generator's
output becomes the curriculum's next-iteration seed_tasks.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence

from core.promotion.dynamic_benchmark import Task
from core.unknowns.novelty_archive import NoveltyArchive


class UnknownUnknownGenerator:
    def __init__(
        self,
        seed: int = 0,
        *,
        archive: Optional[NoveltyArchive] = None,
    ):
        self.rng = random.Random(seed)
        # ``or`` would treat an empty NoveltyArchive (falsy via __len__)
        # as missing and replace it with a fresh instance — silently
        # decoupling the caller's archive from the generator.  Use an
        # explicit None check.
        self.archive = archive if archive is not None else NoveltyArchive()

    # ------------------------------------------------------------------
    def mutate(self, task: Task) -> Task:
        kind = task.kind
        meta = dict(task.metadata)
        rng = self.rng
        if kind == "gcd":
            a = int(meta.get("a", 1))
            b = int(meta.get("b", 1))
            factor = rng.choice([2, 3, 5, 7, 11, 97, 997])
            a2 = abs(a * factor + rng.choice([-1, 0, 1]))
            b2 = abs(b * factor)
            return Task(
                kind="gcd",
                prompt=f"Return gcd({a2}, {b2}) as an integer.",
                answer=math.gcd(a2, b2) if (a2 or b2) else 0,
                metadata={"a": a2, "b": b2, "parent": task.hash_public()},
            )
        if kind == "mod":
            a = int(meta.get("a", 1))
            b = int(meta.get("b", 1))
            m = int(meta.get("m", 7))
            b2 = max(1, b * rng.randint(2, 50))
            m2 = max(2, m + rng.choice([-1, 1]) * rng.randint(1, 1000))
            return Task(
                kind="mod",
                prompt=f"Return ({a} ** {b2}) mod {m2}.",
                answer=pow(a, b2, m2),
                metadata={"a": a, "b": b2, "m": m2, "parent": task.hash_public()},
            )
        if kind == "sort":
            arr = list(meta.get("arr", []))
            arr += [rng.choice(arr) if arr else 0, -10**9, 10**9]
            rng.shuffle(arr)
            return Task(
                kind="sort",
                prompt=f"Sort this list ascending: {arr}",
                answer=sorted(arr),
                metadata={"arr": arr, "parent": task.hash_public()},
            )
        if kind == "palindrome":
            s = str(meta.get("s", ""))
            tail = rng.choice([s[::-1], "z", "", s[: max(1, len(s) // 2)]])
            s2 = s + tail
            return Task(
                kind="palindrome",
                prompt=f"Is this string a palindrome? Answer true or false: {s2}",
                answer=(s2 == s2[::-1]),
                metadata={"s": s2, "parent": task.hash_public()},
            )
        if kind == "compose":
            x = int(meta.get("x", 0))
            a = int(meta.get("a", 1))
            b = int(meta.get("b", 0))
            c = int(meta.get("c", 1))
            d = int(meta.get("d", 0))
            x2 = x + rng.randint(-10, 10)
            a2 = a + rng.randint(-2, 2) or 1
            return Task(
                kind="compose",
                prompt=(
                    f"Let f(x)={a2}x+{b}, g(x)={c}x+{d}. "
                    f"Return g(f({x2})) as an integer."
                ),
                answer=c * (a2 * x2 + b) + d,
                metadata={
                    "x": x2, "a": a2, "b": b, "c": c, "d": d,
                    "parent": task.hash_public(),
                },
            )
        # Fallback: rewrap with novelty hint.
        return Task(
            kind=task.kind,
            prompt=task.prompt + " Explain any edge cases.",
            answer=task.answer,
            metadata={**meta, "parent": task.hash_public()},
        )

    # ------------------------------------------------------------------
    def generate(
        self,
        seed_tasks: Sequence[Task],
        n: int = 50,
        *,
        max_attempts_multiplier: int = 30,
    ) -> List[Task]:
        if not seed_tasks:
            return []
        if n <= 0:
            return []
        out: List[Task] = []
        attempts = 0
        budget = n * max_attempts_multiplier
        while len(out) < n and attempts < budget:
            attempts += 1
            seed = self.rng.choice(list(seed_tasks))
            task = self.mutate(seed)
            if self.archive.add_if_novel(task.prompt, metadata={"kind": task.kind}):
                out.append(task)
        return out
