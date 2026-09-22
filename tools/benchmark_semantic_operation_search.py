#!/usr/bin/env python3
"""Compare exact search arithmetic with its retained pre-change implementation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    if args.seeds < 1 or args.seeds > 100 or args.output.exists():
        parser.error("use 1..100 seeds and a new output")
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    configure_refit_environment(args.output)
    from core.learning.semantic_operation_search import OperationChartSearch
    from core.learning.semantic_program_ir import TokenSpan
    from core.learning.semantic_program_transducer_fitting import _OperationNode
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent

    path = "core/learning/semantic_operation_search.py"
    revision = subprocess.check_output(
        ["git", "-c", "core.fsmonitor=false", "rev-parse", "--verify", args.reference_revision + "^{commit}"],
        cwd=ROOT, text=True).strip()
    reference_source = subprocess.check_output(
        ["git", "-c", "core.fsmonitor=false", "show", revision + ":" + path], cwd=ROOT, text=True)
    namespace = {"__name__": "search_arithmetic_reference"}
    tree = ast.parse(reference_source)
    # The retained invariant would register the same name twice. Its test
    # function is not part of inference; leave all search definitions intact.
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.FunctionDef) and node.name == "_operation_search_coverage")]
    exec(compile(tree, revision + ":" + path, "exec"), namespace)
    reference = namespace["OperationChartSearch"]
    plan = {"schema": "aura.exact_search_arithmetic_benchmark.v1", "reference_revision": revision,
            "reference_sha256": hashlib.sha256(reference_source.encode()).hexdigest(),
            "reference_excluded_registration": "_operation_search_coverage",
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                              (path, "core/learning/dyadic_scores.py", str(Path(__file__).relative_to(ROOT)))},
            "seeds": args.seeds, "nodes": 480, "max_steps": 6, "charts": 256,
            "max_expansions": 100_000, "claim": "search_arithmetic_only_not_end_to_end_latency"}
    if not atomic_write_bytes_if_absent(args.output.with_suffix(".plan.json"),
                                       json.dumps(plan, sort_keys=True).encode()):
        raise ValueError("benchmark plan already exists")
    rows = []
    for seed in range(args.seeds):
        rng = random.Random(seed)
        nodes = tuple(_OperationNode(TokenSpan(i // 2, i // 2 + 1), "add" if i % 2 else "sub",
                                     rng.uniform(-8, 8), 0., 0.) for i in range(plan["nodes"]))
        outcomes = {}
        arms = [("rational", reference), ("integer", OperationChartSearch)]
        for name, cls in arms[seed % 2:] + arms[:seed % 2]:
            started = time.monotonic()
            search = cls(nodes, max_steps=plan["max_steps"], length_penalty=.1,
                         max_expansions=plan["max_expansions"])
            built = time.monotonic()
            charts = [tuple((n.span.start, n.span.end, n.operation) for n in chart)
                      for chart in itertools.islice(search, plan["charts"])]
            ended = time.monotonic()
            outcomes[name] = {"construction_s": built - started, "enumeration_s": ended - built,
                "total_s": ended - started, "expanded": search.expanded, "yielded": search.yielded,
                "remaining_bound": search.remaining_operation_score_upper_bound,
                "charts_sha256": hashlib.sha256(json.dumps(charts).encode()).hexdigest()}
        for field in ("expanded", "yielded", "remaining_bound", "charts_sha256"):
            if outcomes["rational"][field] != outcomes["integer"][field]:
                raise ValueError(f"exact search changed {field} for seed {seed}")
        rows.append({"seed": seed, "arms": outcomes})
    for name, digest in plan["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError("benchmark implementation changed")
    summary = {"median_total_s": {name: median(row["arms"][name]["total_s"] for row in rows)
                                  for name in ("rational", "integer")}, "identical_search": True}
    summary["median_paired_speed_ratio"] = median(
        row["arms"]["rational"]["total_s"] / row["arms"]["integer"]["total_s"] for row in rows)
    if not atomic_write_bytes_if_absent(args.output, json.dumps(
            {"plan": plan, "summary": summary, "rows": rows}, sort_keys=True).encode()):
        raise ValueError("benchmark output already exists")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
