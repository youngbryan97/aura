#!/usr/bin/env python3
"""G12's numbers: her against her base model, request by request, on each benchmark.

MATH-500 and BIG-Bench Hard rows are paired by request and compared with an
exact two-sided McNemar test; IFBench is read from AllenAI's grader output as
the prompt-level loose accuracy its authors report, with strict beside it,
and paired the same way on whether every instruction was followed.

Usage:
    g12_analysis.py --her-math DIR --base-math DIR --her-bbh DIR --base-bbh DIR
        --her-ifbench DIR --base-ifbench DIR --output FILE
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def two_sided_mcnemar(first_only: int, second_only: int) -> float:
    n = first_only + second_only
    if n == 0:
        return 1.0
    k = max(first_only, second_only)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n
    return min(1.0, 2 * tail)


def wilson(correct: int, n: int) -> list[float]:
    """95% Wilson interval for a proportion."""
    if n == 0:
        return [0.0, 0.0]
    z, p = 1.959963984540054, correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def rows(directory: Path, key: str) -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted(directory.expanduser().glob("**/*.json")):
        if path.parent.name not in {"rows", "ordinary"}:
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        out[str(row[key])] = row
    return out


def paired(her: dict[str, dict], base: dict[str, dict], field: str, group: str | None = None) -> dict[str, Any]:
    shared = sorted(set(her) & set(base))
    her_only = sum(bool(her[k][field]) and not bool(base[k][field]) for k in shared)
    base_only = sum(bool(base[k][field]) and not bool(her[k][field]) for k in shared)
    her_correct = sum(bool(her[k][field]) for k in shared)
    base_correct = sum(bool(base[k][field]) for k in shared)
    result = {"requests": len(shared), "her": her_correct, "base": base_correct,
              "her_interval": wilson(her_correct, len(shared)), "base_interval": wilson(base_correct, len(shared)),
              "her_only": her_only, "base_only": base_only, "two_sided_p": two_sided_mcnemar(her_only, base_only),
              "missing": {"her": len(set(base) - set(her)), "base": len(set(her) - set(base))}}
    if group:
        by: dict[str, list[str]] = defaultdict(list)
        for k in shared:
            by[str(her[k][group])].append(k)
        result["by_" + group] = {name: {"n": len(keys), "her": sum(bool(her[k][field]) for k in keys),
                                        "base": sum(bool(base[k][field]) for k in keys)}
                                 for name, keys in sorted(by.items())}
    return result


def ifbench(directory: Path) -> dict[str, dict[str, Any]]:
    """Per prompt, whether every instruction was followed, loose and strict."""
    out: dict[str, dict[str, Any]] = {}
    for mode in ("loose", "strict"):
        found = sorted(directory.expanduser().glob(f"**/*eval_results_{mode}.jsonl"))
        if not found:
            raise SystemExit(f"no IFBench {mode} results under {directory}")
        for line in found[-1].read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                out.setdefault(record["prompt"], {})[mode] = bool(record["follow_all_instructions"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("her-math", "base-math", "her-bbh", "base-bbh", "her-ifbench", "base-ifbench"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any] = {}
    report["math500"] = paired(rows(args.her_math, "unique_id"), rows(args.base_math, "unique_id"), "correct", "level")
    report["bbh"] = paired(rows(args.her_bbh, "id"), rows(args.base_bbh, "id"), "correct", "task")
    her_if, base_if = ifbench(args.her_ifbench), ifbench(args.base_ifbench)
    report["ifbench"] = {mode: paired({k: v for k, v in her_if.items()}, {k: v for k, v in base_if.items()}, mode)
                         for mode in ("loose", "strict")}
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in value.items() if not k.startswith("by_")} if "her" in value else value
                      for name, value in report.items()}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
