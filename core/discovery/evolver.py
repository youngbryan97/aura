"""ExpressionEvolver — bounded GA over SafeExpression trees.

The evolver:
  * scores each individual against an example set,
  * keeps an elite pool,
  * mutates + crosses over to fill the population,
  * tracks the best-ever individual across generations,
  * emits a discovery receipt on F1 audit chain when it finds an
    individual that perfectly matches every example.

Score = -mean_abs_error - 0.001 * size, so it favours both
correctness and parsimony.  No floating-point quirks: integer-only
domain.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.discovery.expression import SafeExpression


Example = Tuple[int, int, int]


@dataclass
class EvolverResult:
    best: SafeExpression
    score: float
    generation: int
    perfect: bool
    history: List[float] = field(default_factory=list)
    receipt_id: Optional[str] = None
    best_str: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best": str(self.best),
            "score": self.score,
            "generation": self.generation,
            "perfect": self.perfect,
            "history": list(self.history),
            "receipt_id": self.receipt_id,
            "best_str": self.best_str,
        }


class ExpressionEvolver:
    def __init__(
        self,
        seed: int = 0,
        *,
        population_size: int = 64,
        elite_size: int = 8,
        mutation_p: float = 0.35,
        crossover_p: float = 0.4,
        emit_receipts: bool = True,
    ):
        if elite_size < 1 or elite_size > population_size:
            raise ValueError("elite_size must be in [1, population_size]")
        if not 0.0 <= mutation_p <= 1.0:
            raise ValueError("mutation_p must be in [0, 1]")
        if not 0.0 <= crossover_p <= 1.0:
            raise ValueError("crossover_p must be in [0, 1]")
        self.rng = random.Random(seed)
        self.population_size = int(population_size)
        self.elite_size = int(elite_size)
        self.mutation_p = float(mutation_p)
        self.crossover_p = float(crossover_p)
        self.emit_receipts = bool(emit_receipts)

    def evolve(
        self,
        examples: Sequence[Example],
        generations: int = 50,
        *,
        target_label: str = "discovery_target",
    ) -> EvolverResult:
        if generations < 1:
            raise ValueError("generations must be >= 1")
        if not examples:
            raise ValueError("examples must be non-empty")

        population: List[SafeExpression] = [
            SafeExpression.random(self.rng, depth=4) for _ in range(self.population_size)
        ]

        def score(expr: SafeExpression) -> float:
            err = 0.0
            for a, b, target in examples:
                try:
                    pred = expr.eval(a, b)
                except Exception:
                    err += 10**6
                    continue
                err += min(abs(pred - target), 10**6)
            mean_err = err / max(1, len(examples))
            return -mean_err - 0.001 * expr.size()

        history: List[float] = []
        best_score = float("-inf")
        best: Optional[SafeExpression] = None
        best_gen = 0

        for gen in range(generations):
            ranked = sorted(((score(e), e) for e in population), key=lambda x: x[0], reverse=True)
            top_score, top_expr = ranked[0]
            if top_score > best_score:
                best_score = top_score
                best = top_expr
                best_gen = gen
            history.append(top_score)

            elites = [e for _, e in ranked[: self.elite_size]]
            new_pop: List[SafeExpression] = list(elites)
            while len(new_pop) < self.population_size:
                if self.rng.random() < self.crossover_p and len(elites) >= 2:
                    a, b = self.rng.sample(elites, 2)
                    c1, c2 = SafeExpression.crossover(a, b, self.rng)
                    new_pop.append(c1.mutate(self.rng, self.mutation_p))
                    if len(new_pop) < self.population_size:
                        new_pop.append(c2.mutate(self.rng, self.mutation_p))
                else:
                    parent = self.rng.choice(elites)
                    new_pop.append(parent.mutate(self.rng, self.mutation_p))
            population = new_pop[: self.population_size]

        assert best is not None  # generations >= 1 guarantees at least one round
        perfect = self._is_perfect(best, examples)
        receipt_id = None
        if self.emit_receipts and perfect:
            receipt_id = self._emit_discovery_receipt(best, target_label, best_score)

        return EvolverResult(
            best=best,
            score=best_score,
            generation=best_gen,
            perfect=perfect,
            history=history,
            receipt_id=receipt_id,
            best_str=str(best),
        )

    @staticmethod
    def _is_perfect(expr: SafeExpression, examples: Sequence[Example]) -> bool:
        for a, b, target in examples:
            try:
                if expr.eval(a, b) != target:
                    return False
            except Exception:
                return False
        return True

    def _emit_discovery_receipt(
        self,
        expr: SafeExpression,
        target_label: str,
        score: float,
    ) -> Optional[str]:
        try:
            from core.runtime.receipts import StateMutationReceipt, get_receipt_store

            store = get_receipt_store()
            receipt = store.emit(
                StateMutationReceipt(
                    cause="evolver.discovery",
                    domain="algorithm_discovery",
                    key=target_label,
                    metadata={
                        "expression": str(expr),
                        "size": expr.size(),
                        "score": float(score),
                    },
                )
            )
            return receipt.receipt_id
        except Exception:
            return None
