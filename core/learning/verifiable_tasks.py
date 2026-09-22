"""Reasoning tasks a program can grade (CP228).

Everything measured this week ran on ``khop``/``modular``/``register_trace``
-- synthetic algorithmic puzzles that are self-contained by construction.
Retrieval cannot help on them, breadth cannot help on them, and a gain
there was never going to transfer. They were the wrong instrument for the
thesis in Anima Rationis, which targets frontier reasoning across math,
code, logic and science with tools and memory in the loop.

This module supplies the shared substrate for both halves of the program:

* **Evaluation** needs tasks where a bare 32B fails and an integrated one
  might not, so the integration hypothesis becomes falsifiable.
* **RLVR** needs a programmatic reward. Anima Rationis line 511 cites
  QwQ-32B reaching R1-comparable reasoning through exactly this -- RL over
  a 32B with correctness verifiers for math and execution feedback for
  code. A verifier that can be fooled is worse than none, because policy
  gradient optimizes whatever the grader actually measures.

**Relationship to ``core/learning/heldout_battery.py``** (read this before
adding generators): that module is the SEALED, contamination-resistant
battery -- eight mature generators, answer fingerprints, and
``text_collides_with_battery`` for leak detection. It has no concept of
task DEPTH. This module is the depth-stratified scaling instrument: it
exists because Anima Rationis makes d(accuracy)/d(depth) the central
success criterion, and a battery with no depth axis cannot measure it.
Use heldout_battery for promotion gating and contamination checks; use this
for the scaling curve. Do not duplicate its generators here.

Design commitments, each answering a way this could produce fake progress:

* **Graders are programs, not string matches.** ``2/4`` and ``0.5`` are the
  same answer; ``x = 5`` and ``5`` are the same answer.
* **Every task carries the reasoning depth it requires**, so the
  compute-scaling curve (Anima Rationis' central success criterion,
  d accuracy / d steps > 0) can be measured against difficulty rather than
  assumed.
* **Held-out generation is seed-separated**, and a task knows whether it
  is answerable from parametric knowledge alone or needs retrieval. Mixing
  those silently is how an integration claim gets made from a task that
  never required integration.
"""
from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

VERIFIABLE_TASK_SCHEMA = "aura.verifiable_tasks.v1"

# A task that needs no external knowledge cannot demonstrate that external
# knowledge helped. Kept explicit so an integration claim has to name which
# tasks could possibly have shown it.
KNOWLEDGE_FREE = "parametric"
NEEDS_RETRIEVAL = "retrieval"


@dataclass(frozen=True)
class VerifiableTask:
    """A problem whose answer a program can check."""

    task_id: str
    prompt: str
    domain: str
    depth: int
    knowledge: str
    grader: str
    expected: Any
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("task prompt must not be empty")
        if type(self.depth) is not int or self.depth < 1:
            raise ValueError("depth must be a positive integer")
        if self.knowledge not in (KNOWLEDGE_FREE, NEEDS_RETRIEVAL):
            raise ValueError("knowledge must be parametric or retrieval")
        if self.grader not in GRADERS:
            raise ValueError(f"unknown grader: {self.grader}")

    def grade(self, response: str) -> dict[str, Any]:
        """Programmatic verdict. Never raises on bad model output."""
        return GRADERS[self.grader](response, self.expected, self.metadata)


# ── Graders: programs, not string comparison ────────────────────────────


def _extract_final(response: str) -> str:
    """Pull the answer out of whatever the model actually emitted.

    Tolerant on FORM, strict on VALUE. A model that reasons correctly and
    formats loosely should score correct; a model that formats perfectly
    and answers wrongly must not.
    """
    text = str(response or "")
    # Strip markdown emphasis so **FINAL_ANSWER**: parses like FINAL_ANSWER:
    plain = re.sub(r"[*`#]", "", text)
    for pattern in (
        r"FINAL_ANSWER\s*:\s*(.+?)(?:\n|$)",
        r"(?:the )?answer(?: is)?\s*[:=]\s*(.+?)(?:\n|$)",
        r"\\boxed\{([^}]*)\}",
    ):
        found = re.findall(pattern, plain, flags=re.IGNORECASE | re.DOTALL)
        if found:
            return found[-1].strip()
    lines = [line.strip() for line in plain.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _as_number(value: Any) -> float | None:
    """Parse a number the way a person would read it."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value or "").strip()
    text = text.replace(",", "").replace("$", "").rstrip(".")
    text = re.sub(r"^[^\d\-+.]*", "", text)
    text = re.sub(r"[^\d\-+./eE]*$", "", text)
    if not text:
        return None
    if "/" in text:  # 2/4 and 0.5 are the same answer
        parts = text.split("/")
        if len(parts) == 2:
            try:
                denominator = float(parts[1])
                return float(parts[0]) / denominator if denominator else None
            # not a failure: a value that is not a number is not one this can read.
            except ValueError:
                return None
    try:
        return float(text)
    # not a failure: a value that is not a number is not one this can read.
    except ValueError:
        return None


def grade_numeric(response: str, expected: Any, metadata: dict) -> dict[str, Any]:
    """Numeric equality with tolerance -- not string equality."""
    produced = _as_number(_extract_final(response))
    target = _as_number(expected)
    if produced is None or target is None:
        return {"correct": False, "parsed": produced, "reason": "unparseable"}
    tolerance = float(metadata.get("tolerance", 1e-6))
    close = math.isclose(produced, target, rel_tol=tolerance, abs_tol=tolerance)
    return {"correct": bool(close), "parsed": produced, "expected": target}


def grade_exact_set(response: str, expected: Any, metadata: dict) -> dict[str, Any]:
    """Unordered set equality -- order is not part of the answer."""
    raw = _extract_final(response)
    tokens = {t.strip().lower() for t in re.split(r"[,\s]+", raw) if t.strip()}
    target = {str(t).strip().lower() for t in expected}
    return {
        "correct": bool(tokens == target),
        "parsed": sorted(tokens),
        "expected": sorted(target),
    }


def grade_boolean(response: str, expected: Any, metadata: dict) -> dict[str, Any]:
    raw = _extract_final(response).strip().lower()
    truthy = {"true", "yes", "1", "t"}
    falsy = {"false", "no", "0", "f"}
    if raw in truthy:
        produced: bool | None = True
    elif raw in falsy:
        produced = False
    else:
        produced = None
    if produced is None:
        return {"correct": False, "parsed": None, "reason": "unparseable"}
    return {"correct": bool(produced == bool(expected)), "parsed": produced}


def grade_json(response: str, expected: Any, metadata: dict) -> dict[str, Any]:
    """Structural equality on JSON, so key order and spacing do not count."""
    raw = _extract_final(response)
    start = raw.find("{")
    if start >= 0:
        depth = 0
        for index in range(start, len(raw)):
            if raw[index] == "{":
                depth += 1
            elif raw[index] == "}":
                depth -= 1
                if depth == 0:
                    raw = raw[start : index + 1]
                    break
    try:
        produced = json.loads(raw)
    except (ValueError, TypeError):
        return {"correct": False, "parsed": None, "reason": "unparseable"}
    return {"correct": bool(produced == expected), "parsed": produced}


def grade_ordered(response: str, expected: Any, metadata: dict) -> dict[str, Any]:
    """Sequence equality where ORDER is the answer.

    Grading an ordering task with set equality marks every permutation
    correct -- which under policy gradient teaches the model that ordering
    does not matter. The metadata said "ordered" while the grader ignored
    it; the metadata was not the thing being enforced.
    """
    raw = _extract_final(response)
    tokens = [t.strip().lower() for t in re.split(r"[,\s]+", raw) if t.strip()]
    target = [str(t).strip().lower() for t in expected]
    return {
        "correct": bool(tokens == target),
        "parsed": tokens,
        "expected": target,
    }


GRADERS: dict[str, Callable[[str, Any, dict], dict[str, Any]]] = {
    "numeric": grade_numeric,
    "ordered": grade_ordered,
    "exact_set": grade_exact_set,
    "boolean": grade_boolean,
    "json": grade_json,
}


# ── Generators: depth-parameterized, so scaling can be measured ─────────


def _arithmetic_chain(rng: random.Random, depth: int, index: int) -> VerifiableTask:
    """Multi-step arithmetic. Depth = number of dependent operations."""
    value = rng.randint(2, 20)
    steps = [f"Start with {value}."]
    for _ in range(depth):
        operation = rng.choice(("add", "subtract", "multiply"))
        operand = rng.randint(2, 12)
        if operation == "add":
            value += operand
            steps.append(f"Add {operand}.")
        elif operation == "subtract":
            value -= operand
            steps.append(f"Subtract {operand}.")
        else:
            value *= operand
            steps.append(f"Multiply by {operand}.")
    prompt = (
        " ".join(steps)
        + "\nWhat is the final value? Reply with FINAL_ANSWER: <number>"
    )
    return VerifiableTask(
        task_id=f"arith-d{depth}-{index}",
        prompt=prompt,
        domain="math",
        depth=depth,
        knowledge=KNOWLEDGE_FREE,
        grader="numeric",
        expected=value,
        metadata={"tolerance": 1e-9},
    )


def _constraint_satisfaction(
    rng: random.Random, depth: int, index: int
) -> VerifiableTask:
    """Ordering under constraints. Depth = number of constraints."""
    names = ["Ana", "Ben", "Cara", "Dev", "Eli", "Fay", "Gus"][: depth + 2]
    order = names[:]
    rng.shuffle(order)
    constraints = []
    for position in range(len(order) - 1):
        constraints.append(f"{order[position]} is before {order[position + 1]}.")
    rng.shuffle(constraints)
    prompt = (
        "Put these people in order given the constraints.\n"
        + " ".join(constraints)
        + "\nReply with FINAL_ANSWER: <names separated by commas>"
    )
    return VerifiableTask(
        task_id=f"constraint-d{depth}-{index}",
        prompt=prompt,
        domain="logic",
        depth=depth,
        knowledge=KNOWLEDGE_FREE,
        grader="ordered",
        expected=order,
        metadata={},
    )


def _program_trace(rng: random.Random, depth: int, index: int) -> VerifiableTask:
    """Execute a small program mentally. Depth = loop iterations."""
    start = rng.randint(1, 40)
    increment = rng.randint(2, 25)
    value = start
    for _ in range(depth):
        value = value * 2 + increment
    prompt = (
        f"x = {start}\n"
        f"repeat {depth} times:\n    x = x * 2 + {increment}\n"
        "What is x at the end? Reply with FINAL_ANSWER: <number>"
    )
    return VerifiableTask(
        task_id=f"trace-d{depth}-{index}",
        prompt=prompt,
        domain="code",
        depth=depth,
        knowledge=KNOWLEDGE_FREE,
        grader="numeric",
        expected=value,
        metadata={"tolerance": 1e-9},
    )


GENERATORS: dict[str, Callable[[random.Random, int, int], VerifiableTask]] = {
    "arithmetic_chain": _arithmetic_chain,
    "constraint_order": _constraint_satisfaction,
    "program_trace": _program_trace,
}


def build_task_set(
    *,
    domains: list[str],
    depths: list[int],
    per_cell: int,
    seed: int,
) -> list[VerifiableTask]:
    """Generate a depth-stratified, deterministic task set."""
    unknown = [d for d in domains if d not in GENERATORS]
    if unknown:
        raise ValueError(f"unknown generators: {unknown}")
    if not depths or any(type(d) is not int or d < 1 for d in depths):
        raise ValueError("depths must be positive integers")
    if type(per_cell) is not int or per_cell < 1:
        raise ValueError("per_cell must be a positive integer")
    tasks: list[VerifiableTask] = []
    for domain in domains:
        for depth in depths:
            rng = random.Random(f"{seed}:{domain}:{depth}")
            seen: set[str] = set()
            cell: list[VerifiableTask] = []
            # Rejection-sample to distinct prompts. A generator's parameter
            # space is finite (program_trace has 45 combinations at a given
            # depth), so asking for more tasks than exist would silently
            # return duplicates and inflate any score computed over them.
            attempts = 0
            budget = max(200, per_cell * 80)
            while len(cell) < per_cell and attempts < budget:
                attempts += 1
                candidate = GENERATORS[domain](rng, depth, len(cell))
                if candidate.prompt in seen:
                    continue
                candidate = replace(
                    candidate,
                    task_id=f"{candidate.task_id}-s{seed}",
                )
                seen.add(candidate.prompt)
                cell.append(candidate)
            if len(cell) < per_cell:
                raise ValueError(
                    f"{domain} at depth {depth} cannot produce {per_cell} "
                    f"distinct tasks (found {len(cell)} in {attempts} draws); "
                    "reduce per_cell or widen the generator"
                )
            tasks.extend(cell)
    prompts = {task.prompt for task in tasks}
    if len(prompts) != len(tasks):
        raise RuntimeError("generation produced duplicate prompts")
    return tasks


def disjoint_split(
    *,
    domains: list[str],
    depths: list[int],
    train_per_cell: int,
    holdout_per_cell: int,
    seed: int,
) -> tuple[list[VerifiableTask], list[VerifiableTask]]:
    """Train and held-out sets with PROVEN zero prompt overlap.

    Verified rather than assumed. A held-out set that leaks is how a
    training-set number gets reported as generalization, and it is not
    detectable after the fact from the score alone.
    """
    train = build_task_set(
        domains=domains, depths=depths, per_cell=train_per_cell, seed=seed
    )
    holdout = build_task_set(
        domains=domains, depths=depths, per_cell=holdout_per_cell,
        seed=seed + 7919,
    )
    train_prompts = {task.prompt for task in train}
    holdout = [task for task in holdout if task.prompt not in train_prompts]
    if not holdout:
        raise RuntimeError("held-out split is empty after de-overlap")
    task_ids = [task.task_id for task in (*train, *holdout)]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("train and held-out task identities overlap")
    return train, holdout


def scaling_report(
    results: list[tuple[VerifiableTask, bool]],
) -> dict[str, Any]:
    """Accuracy by depth -- the curve Anima Rationis makes the criterion.

    A flat curve means extra computation is not buying reasoning, whatever
    the aggregate score says. Reported per depth so 'it got better' cannot
    be claimed from an average that improved only on easy items.
    """
    by_depth: dict[int, list[bool]] = {}
    for task, correct in results:
        by_depth.setdefault(task.depth, []).append(bool(correct))
    if not by_depth:
        raise ValueError("no results to report")
    depths = sorted(by_depth)
    accuracy = {d: sum(by_depth[d]) / len(by_depth[d]) for d in depths}
    deepest, shallowest = depths[-1], depths[0]
    return {
        "schema": VERIFIABLE_TASK_SCHEMA,
        "depths": depths,
        "accuracy_by_depth": {d: round(accuracy[d], 4) for d in depths},
        "n_by_depth": {d: len(by_depth[d]) for d in depths},
        "overall": round(
            sum(sum(v) for v in by_depth.values())
            / sum(len(v) for v in by_depth.values()),
            4,
        ),
        # A model that holds accuracy as required depth grows is doing the
        # composition; one that falls off is pattern-matching shallow cases.
        "depth_falloff": round(accuracy[shallowest] - accuracy[deepest], 4),
    }


__all__ = [
    "GENERATORS",
    "GRADERS",
    "KNOWLEDGE_FREE",
    "NEEDS_RETRIEVAL",
    "VERIFIABLE_TASK_SCHEMA",
    "VerifiableTask",
    "build_task_set",
    "disjoint_split",
    "grade_boolean",
    "grade_exact_set",
    "grade_json",
    "grade_numeric",
    "grade_ordered",
    "scaling_report",
]
