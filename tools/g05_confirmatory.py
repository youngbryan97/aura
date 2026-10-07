#!/usr/bin/env python3
"""G05's confirmatory run: check the frozen identities before it, analyse the rows after it.

``check`` refuses unless every artifact the plan froze (reader, model,
generators, scorer, runtime files) still hashes to what the plan committed and
each feature bundle holds exactly the committed tasks. The answers themselves
come from ``run_g05_public_answers.py``, once per stratum, with the plan's arms
and, on the primary stratum, its counterbalanced order. ``analyse`` computes
the plan's analysis from the rows alone: the primary exact McNemar, the
translation claim, the two secondary tests, and compute per arm.

Usage:
    g05_confirmatory.py check --protocol DIR --features-composition DIR --features-lists DIR
    g05_confirmatory.py analyse --protocol DIR --plan FILE --run DIR
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def exact_test(first_only: int, second_only: int, alternative: str) -> float:
    from scipy.stats import binomtest

    discordant = first_only + second_only
    if not discordant:
        return 1.0
    return float(binomtest(first_only, discordant, 0.5, alternative=alternative).pvalue)


def _rows(run: Path, stratum: str, arm: str) -> dict[str, dict]:
    return {json.loads(p.read_text())["task_id"]: json.loads(p.read_text())
            for p in sorted((run / stratum / "rows" / arm).glob("*.json"))}


def analyse(spec: dict, plan: dict, run: Path) -> dict:
    primary_stratum = next(iter(spec["domains"]))
    strata = {primary_stratum: spec["domains"][primary_stratum], **spec["secondary_domains"]}
    rows = {name: {arm: _rows(run, name, arm) for arm in spec["arms_by_stratum"][name]} for name in strata}
    report: dict = {"missing": {}, "arms": {}}
    for name, tasks in strata.items():
        ids = {task["id"] for task in tasks}
        report["missing"][name] = {arm: len(ids - set(found)) for arm, found in rows[name].items()}
        report["arms"][name] = {
            arm: {
                "exact": sum(bool(row["answer_exact"]) for row in found.values()),
                "decodes": len(found),
                "median_generated_tokens": statistics.median(r["generated_tokens"] for r in found.values()) if found else None,
                "median_seconds": statistics.median(r["seconds"] for r in found.values()) if found else None,
                "budget_stops": sum(r["termination"] != "stop" for r in found.values()),
                "thinking": sorted({r.get("thinking") for r in found.values()}),
            }
            for arm, found in rows[name].items()
        }
    complete = not any(any(count for count in missing.values()) for missing in report["missing"].values())
    alpha = plan["parameters"]["per_comparison_alpha"]
    assisted, ordinary = rows[primary_stratum]["assisted"], rows[primary_stratum]["ordinary"]
    paired = sorted(set(assisted) & set(ordinary))
    wins = sum(assisted[t]["answer_exact"] and not ordinary[t]["answer_exact"] for t in paired)
    losses = sum(ordinary[t]["answer_exact"] and not assisted[t]["answer_exact"] for t in paired)
    p_value = exact_test(wins, losses, "greater")
    report["primary"] = {"stratum": primary_stratum, "tasks": len(paired), "assisted_only": wins,
                         "ordinary_only": losses, "exact_one_sided_p": p_value,
                         "rejects": bool(complete and p_value <= alpha)}
    reader_right = translated = 0
    for name in strata:
        for row in rows[name]["assisted"].values():
            if row["reader_answer"] == row["expected"]:
                reader_right += 1
                translated += bool(row["answer_exact"])
    report["translation"] = {"reader_right": reader_right, "assisted_exact_where_reader_right": translated,
                             "holds": bool(complete and reader_right and translated == reader_right)}
    report["secondary"] = {}
    for comparison in spec["secondary"]["comparisons"]:
        first, second = rows[comparison["stratum"]][comparison["first"]], rows[comparison["stratum"]][comparison["second"]]
        shared = sorted(set(first) & set(second))
        only_first = sum(first[t]["answer_exact"] and not second[t]["answer_exact"] for t in shared)
        only_second = sum(second[t]["answer_exact"] and not first[t]["answer_exact"] for t in shared)
        p_value = exact_test(only_first, only_second, comparison["alternative"])
        report["secondary"][f"{comparison['first']}_vs_{comparison['second']}:{comparison['stratum']}"] = {
            "tasks": len(shared), "first_only": only_first, "second_only": only_second,
            "alternative": comparison["alternative"], "exact_p": p_value,
            "rejects": bool(complete and p_value <= spec["secondary"]["per_test_alpha"]),
        }
    report["complete"] = complete
    report["g05_closure_rule_holds"] = report["primary"]["rejects"] and report["translation"]["holds"]
    return report


def check(args: argparse.Namespace) -> int:
    from core.learning.semantic_program_feature_materialization import load_standard_semantic_feature_bundle
    from tools import g05_public_answer_protocol as protocol

    spec = json.loads((args.protocol / "spec.json").read_text(encoding="utf-8"))
    observed = {
        "candidate": protocol._files_sha256([protocol.READER]),
        "resident_model": protocol._files_sha256(sorted(p for p in protocol.MODEL.iterdir() if p.is_file())),
        "task_generator": protocol._files_sha256([ROOT / p for p in protocol.GENERATOR_FILES]),
        "scorer": protocol._files_sha256([ROOT / p for p in protocol.SCORER_FILES]),
        "runtime": protocol._files_sha256([ROOT / p for p in protocol.RUNTIME_FILES]),
    }
    drift = sorted(name for name, value in spec["artifacts"].items() if observed.get(name) != value)
    if drift:
        raise SystemExit(f"frozen artifacts changed since the plan: {drift}")
    bundles = {"composition": args.features_composition, "lists": args.features_lists}
    committed = {"composition": spec["domains"]["composition"], "lists": spec["secondary_domains"]["lists"]}
    for name, directory in bundles.items():
        bundle = load_standard_semantic_feature_bundle(directory.expanduser())
        held = {(str(example.metadata["example_id"]), str(example.metadata["source_text_sha256"]))
                for example in bundle.examples}
        if held != {(task["id"], task["source_sha256"]) for task in committed[name]}:
            raise SystemExit(f"the {name} bundle does not hold exactly the committed tasks")
    print(json.dumps({"artifacts": "unchanged", "bundles": "exactly the committed tasks"}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    checking = sub.add_parser("check")
    checking.add_argument("--protocol", type=Path, required=True)
    checking.add_argument("--features-composition", type=Path, required=True)
    checking.add_argument("--features-lists", type=Path, required=True)
    analysing = sub.add_parser("analyse")
    analysing.add_argument("--protocol", type=Path, required=True)
    analysing.add_argument("--plan", type=Path, required=True)
    analysing.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check":
        return check(args)
    spec = json.loads((args.protocol / "spec.json").read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    report = analyse(spec, plan, args.run.expanduser())
    (args.run.expanduser() / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
