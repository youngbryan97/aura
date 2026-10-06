#!/usr/bin/env python3
"""Freeze G04's transfer protocol before any fresh request is decoded.

The frozen candidate is G03's v12 and the control is the incumbent it was
built on. Four strata of fresh requests (core/learning/
semantic_g04_transfer_corpus.py) are generated here, checked against the
consumed inventory (tools/build_g04_consumed_inventory.py) for the novelty
each claims, sized for power, and committed with every artifact identity
through the paired-replication plan store. Nothing is decoded and no
feature is read: the plan has to exist, and be published, first.

The planned effect is not chosen. It is the conservative bound of the most
transfer-like evidence there is, the two composition bundles: v12 and the
incumbent disagreed on 34 of 96 requests and v12 was right on all 34. Those
bundles were exposed while G03 was developed, so the plan uses the lower
95% bounds (Clopper-Pearson for the disagreement rate, the one-sided bound
for 34 wins of 34), not the point values.

Usage:
    g04_transfer_protocol.py --inventory FILE --store DIR --output DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVIDENCE = Path("~/.aura/rlc-evidence").expanduser()
CANDIDATE = EVIDENCE / "semantic-peak-antecedent-v12-20261006/candidate.json"
INCUMBENT = EVIDENCE / "semantic-literal-identity-pilot-20260921/incumbent_literal_identity.json"
MODEL = Path("~/.aura/models/Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15").expanduser()

#: The evidence the planned effect is bounded from, and how.
EFFECT_EVIDENCE = {
    "source": "G03 v12 composition bundles weave_v1 and weave_v2, against the same incumbent",
    "requests": 96,
    "discordant": 34,
    "candidate_wins": 34,
    "bound": "lower 95%: Clopper-Pearson (two-sided) for discordance, one-sided for the win share",
    "why_bounds": "the bundles were exposed while G03 was developed",
}

GENERATOR_FILES = ("core/learning/semantic_g04_transfer_corpus.py",)
SCORER_FILES = (
    "core/learning/semantic_graph_counterexamples.py",
    "core/learning/semantic_program_execution.py",
    "tools/run_g04_transfer.py",
)
RUNTIME_FILES = (
    "core/learning/semantic_operation_peaks.py",
    "core/learning/semantic_program_compositional_transducer.py",
    "core/learning/semantic_argument_antecedent.py",
    "core/learning/semantic_argument_ownership.py",
    "core/learning/semantic_argument_optimization.py",
    "core/learning/semantic_program_transducer_fitting.py",
    "core/learning/semantic_program_feature_materialization.py",
    "tools/materialize_semantic_program_features.py",
    "tools/run_g04_transfer.py",
)


def _files_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.name).encode("utf-8") + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 22), b""):
                digest.update(chunk)
    return digest.hexdigest()


def planned_effect() -> tuple[float, float]:
    from scipy.stats import beta

    k, n = EFFECT_EVIDENCE["discordant"], EFFECT_EVIDENCE["requests"]
    wins = EFFECT_EVIDENCE["candidate_wins"]
    discordance = float(beta.ppf(0.025, k, n - k + 1))
    # One-sided 95% lower bound on a share with ``wins`` of ``k`` (here all).
    win_share = float(beta.ppf(0.05, wins, k - wins + 1)) if wins < k else float(0.05 ** (1 / k))
    return discordance, win_share


def tasks_needed(*, alpha: float, target: float, discordance: float, win_share: float) -> int:
    from core.evaluation.paired_power import prospective_mcnemar_power

    tasks = 2
    while prospective_mcnemar_power(tasks, alpha=alpha, discordance=discordance,
                                    win_share=win_share) < target:
        tasks += 2
    return tasks


def novelty_report(strata: dict, inventory: dict, step_limit) -> dict:
    """Each stratum's claim, checked against every consumed request."""
    from core.learning.semantic_g04_transfer_corpus import novelty_signatures

    consumed_texts = [row["source_text"].lower() for row in inventory["examples"]]
    consumed_sha = {row["source_sha256"] for row in inventory["examples"]}
    report: dict = {}
    for name, rows in strata.items():
        found = {
            sig: sum(sig in text for text in consumed_texts)
            for sig in novelty_signatures(name)
        }
        repeats = sum(
            hashlib.sha256(row.source_text.encode("utf-8")).hexdigest() in consumed_sha
            for row in rows
        )
        entry = {"consumed_texts_with_signature": found, "tasks_repeating_a_consumed_text": repeats}
        if name == "depth":
            depths = sorted({len(row.instructions) for row in rows})
            entry["depths"] = depths
            entry["consumed_max_depth"] = inventory["max_depth"]
            entry["within_candidate_envelope"] = all(
                step_limit(len(row.inputs)) is not None
                and len(row.instructions) <= step_limit(len(row.inputs))
                for row in rows
            )
        if name == "family":
            entry["consumed_sequence_operations_with_computed_arguments"] = inventory[
                "sequence_operations_with_computed_arguments"
            ]
        report[name] = entry
    failures = [
        name for name, entry in report.items()
        if any(entry["consumed_texts_with_signature"].values())
        or entry["tasks_repeating_a_consumed_text"]
        or (name == "depth" and (min(entry["depths"]) <= entry["consumed_max_depth"]
                                 or not entry["within_candidate_envelope"]))
        or (name == "family" and entry["consumed_sequence_operations_with_computed_arguments"])
    ]
    return {"strata": report, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--order-seed", type=int, default=20261007)
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument("--target-power", type=float, default=0.9)
    args = parser.parse_args()

    from core.evaluation.paired_replication import build_paired_replication_plan
    from core.governance_context import local_internal_governed_scope
    from core.learning.semantic_g04_transfer_corpus import G04_STRATA, build_g04_transfer_corpus
    from core.learning.semantic_operation_peaks import peak_recognition_transducer_from_dict
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict as restore,
    )

    inventory = json.loads(args.inventory.expanduser().read_text(encoding="utf-8"))
    if args.seed in inventory["seeds"]:
        raise SystemExit("the corpus seed was already consumed")
    discordance, win_share = planned_effect()
    per_comparison_alpha = args.familywise_alpha / len(G04_STRATA)
    tasks = tasks_needed(alpha=per_comparison_alpha, target=args.target_power,
                         discordance=discordance, win_share=win_share)
    strata = build_g04_transfer_corpus(seed=args.seed, tasks_per_stratum=tasks)
    candidate = peak_recognition_transducer_from_dict(
        json.loads(CANDIDATE.read_text(encoding="utf-8")), restore_base=restore
    )
    novelty = novelty_report(strata, inventory, candidate.inference_step_limit)
    if novelty["failures"]:
        raise SystemExit(f"novelty fails for {novelty['failures']}: {json.dumps(novelty, indent=1)}")

    consumed_seeds = set(inventory["seeds"])
    domains = {}
    task_seed = args.seed * 1000
    for name in G04_STRATA:
        committed = []
        for row in strata[name]:
            task_seed += 1
            while task_seed in consumed_seeds:
                task_seed += 1
            committed.append({
                "id": row.example_id,
                "source_sha256": hashlib.sha256(row.source_text.encode("utf-8")).hexdigest(),
                "seed": task_seed,
            })
        domains[name] = committed
    head = subprocess.run(["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    artifacts = {
        "candidate": _files_sha256([CANDIDATE]),
        "incumbent": _files_sha256([INCUMBENT]),
        "resident_model": _files_sha256(sorted(path for path in MODEL.iterdir() if path.is_file())),
        "task_generator": _files_sha256([ROOT / path for path in GENERATOR_FILES]),
        "scorer": _files_sha256([ROOT / path for path in SCORER_FILES]),
        "runtime": _files_sha256([ROOT / path for path in RUNTIME_FILES]),
    }
    spec = {
        "campaign": "g04-transfer-v1",
        "hypothesis": (
            "On fresh requests that differ from everything G03's candidate was fitted on, "
            "selected with or tuned against in construction, vocabulary, depth or family, "
            "the frozen candidate (v12) produces a program with the reference program's "
            "meaning more often than the incumbent it was built on."
        ),
        "sampling_unit": "independent_task",
        "sampling_description": (
            f"{tasks} requests per stratum from build_g04_transfer_corpus(seed={args.seed}); "
            "one decode per task and arm; constructions within a stratum taken in turn."
        ),
        "artifacts": artifacts,
        "source_commit": head,
        "arms": ["incumbent", "candidate"],
        "order_seed": args.order_seed,
        "comparisons": [{
            "treatment": "candidate",
            "control": "incumbent",
            "assumptions": {name: {"discordance": discordance, "win_share": win_share}
                            for name in G04_STRATA},
        }],
        "effect_evidence": EFFECT_EVIDENCE,
        "domains": domains,
        "familywise_alpha": args.familywise_alpha,
        "target_power": args.target_power,
        "consumed": {
            "task_ids": [row["example_id"] for row in inventory["examples"]],
            "source_sha256s": [row["source_sha256"] for row in inventory["examples"]],
            "seeds": sorted(consumed_seeds),
            "inventory_manifest_sha256s": inventory["manifest_sha256s"],
        },
        "primary_outcome": (
            "the decoded program and the reference program have the same meaning: "
            "compare_program_meanings over the public inputs and 32 counterfactual "
            "probes returns 'equivalent'. 'unknown', 'different', a refusal and any "
            "runtime failure count as not equivalent, for both arms alike."
        ),
        "secondary_outcomes": [
            "the decoded program's executed answer on the public inputs equals the reference's",
            "the decode succeeded with program execution unavailable (no executable-system "
            "assistance in choosing the program)",
        ],
        "novelty": novelty,
        "features": {
            "corpus_kind": "g04_transfer_v1",
            "seed": args.seed,
            "examples_per_operation_pair": tasks,
            "max_examples": tasks * len(G04_STRATA),
            "representation": "lexical_mid_final_v1",
        },
    }
    plan = build_paired_replication_plan(spec)
    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    (output / "spec.json").write_text(json.dumps(spec, indent=1, sort_keys=True), encoding="utf-8")
    with local_internal_governed_scope("evaluation.paired_replication", domain="file_write"):
        path = plan.write(args.store.expanduser())
    summary = {
        "plan_hash": plan.plan_hash,
        "plan_path": str(path),
        "tasks_per_stratum": tasks,
        "planned_discordance": discordance,
        "planned_win_share": win_share,
        "power": plan.parameters["power"],
        "artifacts": artifacts,
    }
    (output / "plan_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True),
                                              encoding="utf-8")
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
