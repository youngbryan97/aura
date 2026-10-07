#!/usr/bin/env python3
"""Check G05's result without the code that produced it.

Shares none of the grading path: the committed tasks are regenerated from the
published seeds and checked against the plan's commitments; each expected
answer is computed by this file's own interpreter (tools/verify_g04_independently
._run, six primitives); her final answer is read from each row's public text
by this file's own parser, not parse_integral_numeric_claim; and the primary
test, the translation claim and both secondary tests are recomputed from those
readings.

The parser is deliberately plain: the last run of digits in the public text,
with thousands separators (",", " ", thin spaces) between groups of three and
an optional minus sign. Where it and the runner's reading differ, the row is
listed, not resolved.

Usage:
    verify_g05_independently.py --protocol DIR --plan FILE --run DIR --output FILE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.verify_g04_independently import _run, binomial_tail_at_least  # noqa: E402

_NUMBER = re.compile(r"[-−]?\d{1,3}(?:[,   ]\d{3})+(?!\d)|[-−]?\d+")


def last_integer(text: str) -> int | None:
    """The last integer written in ``text``, thousands separators allowed."""
    found = _NUMBER.findall(str(text or ""))
    if not found:
        return None
    raw = found[-1].replace("−", "-")
    return int(re.sub(r"[,   ]", "", raw))


def two_sided(first_only: int, second_only: int) -> float:
    n = first_only + second_only
    if n == 0:
        return 1.0
    k = max(first_only, second_only)
    return min(1.0, 2 * binomial_tail_at_least(k, n))


def _rows(directory: Path) -> dict[str, dict]:
    out = {}
    for path in directory.glob("*.json"):
        row = json.loads(path.read_text())
        out[row["task_id"]] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from core.learning.semantic_g05_long_sequence_corpus import build_g05_long_sequence_corpus
    from core.learning.semantic_program_corpus_replication import (
        build_semantic_program_natural_weave_replication_corpus,
    )

    spec = json.loads((args.protocol / "spec.json").read_text())
    plan = json.loads(args.plan.read_text())
    findings: dict = {"checks": {}}
    design = plan["parameters"]["design"]
    findings["checks"]["plan_matches_spec"] = (design["domains"] == spec["domains"]
                                               and design["artifacts"] == spec["artifacts"])
    canonical = json.dumps(spec["domains"], sort_keys=True, separators=(",", ":")).encode()
    findings["checks"]["task_commitment_recomputed"] = (
        hashlib.sha256(canonical).hexdigest() == plan["parameters"]["task_commitment_sha256"])
    features = spec["features"]
    rebuilt = {
        "composition": build_semantic_program_natural_weave_replication_corpus(
            seed=features["composition"]["seed"],
            examples_per_schema_domain=features["composition"]["examples_per_operation_pair"]),
        "lists": build_g05_long_sequence_corpus(
            seed=features["lists"]["seed"], tasks=features["lists"]["examples_per_operation_pair"]),
    }
    committed = {"composition": spec["domains"]["composition"], "lists": spec["secondary_domains"]["lists"]}
    by_id = {name: {row.example_id: row for row in rows} for name, rows in rebuilt.items()}
    findings["checks"]["tasks_regenerate_from_seed"] = all(
        {task["id"] for task in tasks} == set(by_id[name])
        and all(hashlib.sha256(by_id[name][task["id"]].source_text.encode()).hexdigest() == task["source_sha256"]
                for task in tasks)
        for name, tasks in committed.items())
    consumed = set(spec["consumed"]["source_sha256s"])
    findings["checks"]["no_task_text_was_consumed"] = not any(
        task["source_sha256"] in consumed for tasks in committed.values() for task in tasks)

    readings: dict[str, dict[str, dict[str, dict]]] = {}
    disagreements = []
    for name, tasks in committed.items():
        readings[name] = {}
        for arm in spec["arms_by_stratum"][name]:
            rows = _rows(args.run / name / "rows" / arm)
            readings[name][arm] = {}
            for task in tasks:
                example = by_id[name][task["id"]]
                steps = [(i.instruction.op, tuple(i.instruction.args)) for i in example.instructions]
                expected = _run(steps, tuple(example.inputs))
                row = rows.get(task["id"])
                if row is None:
                    readings[name][arm][task["id"]] = {"missing": True}
                    continue
                stopped = row["termination"] != "stop"
                mine = last_integer(row["public_text"])
                exact = (not stopped) and mine == expected
                if exact != bool(row["answer_exact"]):
                    disagreements.append({"stratum": name, "arm": arm, "task": task["id"],
                                          "runner": bool(row["answer_exact"]), "independent": exact,
                                          "independent_reading": mine, "expected": expected})
                readings[name][arm][task["id"]] = {
                    "exact": exact, "expected_matches_row": expected == row["expected"],
                    "reader_right": row.get("reader_answer") == expected if arm == "assisted" else None,
                }
    findings["checks"]["expected_answers_agree_with_independent_interpreter"] = all(
        reading.get("expected_matches_row", True)
        for arms in readings.values() for rows in arms.values() for reading in rows.values()
        if not reading.get("missing"))
    missing = sum(1 for arms in readings.values() for rows in arms.values()
                  for reading in rows.values() if reading.get("missing"))
    findings["missing_rows"] = missing
    findings["reading_disagreements"] = disagreements

    def exact(name: str, arm: str) -> dict[str, bool]:
        return {task: reading["exact"] for task, reading in readings[name][arm].items() if not reading.get("missing")}

    a, o = exact("composition", "assisted"), exact("composition", "ordinary")
    shared = set(a) & set(o)
    wins = sum(a[t] and not o[t] for t in shared)
    losses = sum(o[t] and not a[t] for t in shared)
    p_value = binomial_tail_at_least(wins, wins + losses) if wins + losses else 1.0
    alpha = plan["parameters"]["per_comparison_alpha"]
    findings["primary"] = {"assisted_exact": sum(a.values()), "ordinary_exact": sum(o.values()),
                           "assisted_only": wins, "ordinary_only": losses, "exact_one_sided_p": p_value,
                           "rejects": missing == 0 and p_value <= alpha}
    reader_right = translated = 0
    for name in committed:
        for task, reading in readings[name]["assisted"].items():
            if reading.get("missing") or not reading["reader_right"]:
                continue
            reader_right += 1
            translated += reading["exact"]
    findings["translation"] = {"reader_right": reader_right, "translated": translated,
                               "holds": missing == 0 and reader_right > 0 and translated == reader_right}
    findings["secondary"] = {}
    for comparison in spec["secondary"]["comparisons"]:
        first, second = exact(comparison["stratum"], comparison["first"]), exact(comparison["stratum"], comparison["second"])
        both = set(first) & set(second)
        f_only = sum(first[t] and not second[t] for t in both)
        s_only = sum(second[t] and not first[t] for t in both)
        p = (binomial_tail_at_least(f_only, f_only + s_only) if f_only + s_only else 1.0) \
            if comparison["alternative"] == "greater" else two_sided(f_only, s_only)
        findings["secondary"][f"{comparison['first']}_vs_{comparison['second']}:{comparison['stratum']}"] = {
            "first_exact": sum(first.values()), "second_exact": sum(second.values()),
            "first_only": f_only, "second_only": s_only, "p": p,
            "rejects": missing == 0 and p <= spec["secondary"]["per_test_alpha"]}
    findings["g05_closure_rule_holds"] = findings["primary"]["rejects"] and findings["translation"]["holds"]
    findings["checks_pass"] = all(value is True for value in findings["checks"].values())
    args.output.write_text(json.dumps(findings, indent=1, sort_keys=True))
    print(json.dumps(findings, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
