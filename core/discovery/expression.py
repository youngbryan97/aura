"""SafeExpression — small AST over (a, b) for bounded algorithm search.

The expression tree supports a closed set of integer ops:
  add, sub, mul, mod, abs, min, max
plus integer constants and variables ``a``, ``b``, ``one``, ``zero``.

Trees compile down to bounded-cost Python and never call user code,
so the evolver can score thousands of candidates without subprocess
overhead.  When discovery moves to free-form Python, callers switch
to ``SafeCodeEvaluator``.
"""
from __future__ import annotations

import random
from typing import Any, Tuple


_OPS_BINARY = ("add", "sub", "mul", "mod", "min", "max")
_OPS_UNARY = ("abs",)
_VARS = ("a", "b", "one", "zero")
_VALUE_CAP = 10**12  # clamp to keep evaluation bounded


class SafeExpression:
    """Immutable tagged-tuple AST node."""

    __slots__ = ("tree",)

    def __init__(self, tree: Any):
        self.tree = tree

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def random(cls, rng: random.Random, depth: int = 3) -> "SafeExpression":
        return cls(cls._random_tree(rng, depth))

    @classmethod
    def _random_tree(cls, rng: random.Random, depth: int) -> Any:
        if depth <= 0 or rng.random() < 0.25:
            return rng.choice(_VARS + (rng.randint(-10, 10),))
        op = rng.choice(_OPS_BINARY + _OPS_UNARY)
        if op in _OPS_UNARY:
            return (op, cls._random_tree(rng, depth - 1))
        return (op, cls._random_tree(rng, depth - 1), cls._random_tree(rng, depth - 1))

    # ------------------------------------------------------------------
    # mutation
    # ------------------------------------------------------------------
    def mutate(self, rng: random.Random, p: float = 0.25) -> "SafeExpression":
        return SafeExpression(self._mutate(self.tree, rng, p, 0))

    @classmethod
    def _mutate(cls, node: Any, rng: random.Random, p: float, depth: int) -> Any:
        if rng.random() < p:
            return cls._random_tree(rng, max(1, 3 - depth))
        if isinstance(node, tuple):
            return tuple([node[0], *[cls._mutate(c, rng, p, depth + 1) for c in node[1:]]])
        return node

    @classmethod
    def crossover(
        cls,
        a: "SafeExpression",
        b: "SafeExpression",
        rng: random.Random,
    ) -> Tuple["SafeExpression", "SafeExpression"]:
        a_paths = list(cls._paths(a.tree))
        b_paths = list(cls._paths(b.tree))
        if not a_paths or not b_paths:
            return a, b
        a_path = rng.choice(a_paths)
        b_path = rng.choice(b_paths)
        a_sub = cls._get(a.tree, a_path)
        b_sub = cls._get(b.tree, b_path)
        c1 = cls._set(a.tree, a_path, b_sub)
        c2 = cls._set(b.tree, b_path, a_sub)
        return SafeExpression(c1), SafeExpression(c2)

    @classmethod
    def _paths(cls, node: Any, prefix=()):
        yield prefix
        if isinstance(node, tuple):
            for i, child in enumerate(node[1:], start=1):
                yield from cls._paths(child, prefix + (i,))

    @classmethod
    def _get(cls, node: Any, path):
        for p in path:
            node = node[p]
        return node

    @classmethod
    def _set(cls, node: Any, path, replacement):
        if not path:
            return replacement
        head, *rest = path
        if isinstance(node, tuple):
            new_children = list(node)
            new_children[head] = cls._set(node[head], tuple(rest), replacement)
            return tuple(new_children)
        return replacement

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def eval(self, a: int, b: int) -> int:
        env = {"a": int(a), "b": int(b), "one": 1, "zero": 0}
        return self._clamp(self._eval(self.tree, env))

    @classmethod
    def _eval(cls, node: Any, env: dict) -> int:
        if isinstance(node, int):
            return int(node)
        if isinstance(node, str):
            return int(env[node])
        op = node[0]
        if op == "abs":
            return abs(cls._eval(node[1], env))
        x = cls._eval(node[1], env)
        y = cls._eval(node[2], env)
        if op == "add":
            return x + y
        if op == "sub":
            return x - y
        if op == "mul":
            return x * y
        if op == "mod":
            return x % y if y != 0 else 0
        if op == "min":
            return min(x, y)
        if op == "max":
            return max(x, y)
        raise ValueError(f"unknown op: {op}")

    @staticmethod
    def _clamp(value: int) -> int:
        return max(min(value, _VALUE_CAP), -_VALUE_CAP)

    # ------------------------------------------------------------------
    # display
    # ------------------------------------------------------------------
    def size(self) -> int:
        return self._size(self.tree)

    @classmethod
    def _size(cls, node: Any) -> int:
        if isinstance(node, tuple):
            return 1 + sum(cls._size(c) for c in node[1:])
        return 1

    def __str__(self) -> str:
        return self._render(self.tree)

    @classmethod
    def _render(cls, node: Any) -> str:
        if not isinstance(node, tuple):
            return str(node)
        op = node[0]
        if op == "abs":
            return f"abs({cls._render(node[1])})"
        if op in ("min", "max"):
            return f"{op}({cls._render(node[1])}, {cls._render(node[2])})"
        sym = {"add": "+", "sub": "-", "mul": "*", "mod": "%"}[op]
        return f"({cls._render(node[1])} {sym} {cls._render(node[2])})"
