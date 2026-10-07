#!/usr/bin/env python3
"""Freeze G04's second transfer protocol before any of its requests is decoded.

The first run (6 October, plan `4989717df596`) rejected on family only. Its
vocabulary and depth failures were read row by row, which consumed them, and
led to candidate v17: G03's v12 with a recognizer that also renames a chosen
chart's operations where their phrases close (each readout one vote), and an
antecedent that counts recency gated by which way a mention points, with
construction families weighed alike. 600 breadth requests (wordings settled
late in the phrase; chains of three to five steps) teach the tagger, the
phrase readouts and the antecedent, not what a verb names. Four earlier
candidates were not frozen (v13 to v16); the development record says why.
v17 holds train at 764 of 764, loses no validation request the incumbent had
right, and reads validation 498 of 500 where v12 read 500: two "after
removing" requests whose renamed phrase carries the next operation's
lead-in. That is recorded as open, not hidden. This protocol commits v17
against the same incumbent on fresh requests from ``g04_transfer_v2``, whose
vocabulary table was committed before the breadth wordings were written
(``f1d2ba1f0``).

Claims, fixed here:

* primary: v17 produces a program with the reference meaning more often than
  the incumbent, exact one-sided McNemar per stratum, Bonferroni over the
  three strata where the first run showed the incumbent failing (vocabulary,
  depth, family);
* construction: every committed construction request decodes to the
  reference meaning. The incumbent reads new scaffolds too (61 of 62 in the
  first run), so a paired test there has nothing to count; what transfers has
  to be all of it;
* G04 is closed only if both hold.

Secondary, for G06, a separate family at alpha 0.05 over two tests: v17 against
itself with the phrase readouts removed, on vocabulary; and with the recency
features zeroed, on depth. v12 is decoded on every task and reported.

Novelty: no committed request repeats any consumed text (2,732, including the
first run's 248 and the 600 breadth requests), and no request uses a word of
the v2 table that any consumed text has. Depth (six and seven steps) is
checked against what was fitted or tuned on (at most five), not against the
first run's test requests, whose depth rows were inspected after that run
and informed the recency fix; the record says so.

Usage:
    g04_transfer_protocol_v2.py --inventory FILE --store DIR --output DIR
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

from tools.g04_transfer_protocol import (  # noqa: E402
    EFFECT_EVIDENCE,
    INCUMBENT,
    MODEL,
    _files_sha256,
    planned_effect,
    tasks_needed,
)

EVIDENCE = Path("~/.aura/rlc-evidence").expanduser()
CANDIDATE = EVIDENCE / "semantic-peak-naming-v17-20261007/candidate.json"
PREVIOUS = EVIDENCE / "semantic-peak-antecedent-v12-20261006/candidate.json"
CORPUS_KIND = "g04_transfer_v2"
PAIRED_STRATA = ("vocabulary", "depth", "family")
SECONDARY = (
    {"treatment": "candidate", "control": "no_phrase_reading", "stratum": "vocabulary"},
    {"treatment": "candidate", "control": "no_recency", "stratum": "depth"},
)
SECONDARY_ALPHA = 0.05

GENERATOR_FILES = ("core/learning/semantic_g04_transfer_corpus.py",)
SCORER_FILES = (
    "core/learning/semantic_graph_counterexamples.py",
    "core/learning/semantic_program_execution.py",
    "tools/run_g04_transfer.py",
    "tools/run_g04_transfer_v2.py",
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
    "tools/run_g04_transfer_v2.py",
)
#: The first run's test requests: consumed, but never fitted or tuned on.
FIRST_RUN_PREFIX = "g04_"


def novelty_report(strata: dict, inventory: dict, step_limit) -> dict:
    """Vocabulary against every consumed text; scaffolds and structure against what was fitted or tuned on.

    The first run's test requests are consumed but were never fitted or tuned
    on. Its construction scaffolds passed there (62 of 62) and were not
    diagnosed, so they are reported where they occur, not counted against.
    """
    from core.learning.semantic_g04_transfer_corpus import novelty_signatures

    everything = [row for row in inventory["examples"]]
    fitted = [row for row in everything if not row["construction_id"].startswith(FIRST_RUN_PREFIX)]
    consumed_sha = {row["source_sha256"] for row in everything}
    fitted_depth = max(row["depth"] for row in fitted)
    fitted_computed = sum(row["sequence_operations_with_computed_arguments"] for row in fitted)
    report: dict = {}
    for name, rows in strata.items():
        against = everything if name == "vocabulary" else fitted
        texts = [row["source_text"].lower() for row in against]
        first_run = [row["source_text"].lower() for row in everything
                     if row["construction_id"].startswith(FIRST_RUN_PREFIX)]
        signatures = novelty_signatures(name, version=2)
        entry = {
            "checked_against": "every consumed request" if name == "vocabulary" else "every fitted or tuned request",
            "texts_with_signature": {sig: sum(sig in text for text in texts) for sig in signatures},
            "first_run_texts_with_signature": {sig: sum(sig in text for text in first_run) for sig in signatures},
            "tasks_repeating_a_consumed_text": sum(
                hashlib.sha256(row.source_text.encode("utf-8")).hexdigest() in consumed_sha
                for row in rows
            ),
        }
        if name == "depth":
            entry["depths"] = sorted({len(row.instructions) for row in rows})
            entry["fitted_max_depth"] = fitted_depth
            entry["within_candidate_envelope"] = all(
                step_limit(len(row.inputs)) is not None
                and len(row.instructions) <= step_limit(len(row.inputs))
                for row in rows
            )
        if name == "family":
            entry["fitted_sequence_operations_with_computed_arguments"] = fitted_computed
        report[name] = entry
    failures = [
        name for name, entry in report.items()
        if any(entry["texts_with_signature"].values())
        or entry["tasks_repeating_a_consumed_text"]
        or (name == "depth" and (min(entry["depths"]) <= entry["fitted_max_depth"]
                                 or not entry["within_candidate_envelope"]))
        or (name == "family" and entry["fitted_sequence_operations_with_computed_arguments"])
    ]
    return {"strata": report, "failures": failures}


def lesions(candidate):
    """The secondary arms: the frozen candidate with one fitted part taken out."""
    from dataclasses import replace

    from core.learning.semantic_operation_peaks import PeakRecognitionTransducer

    recognizer = candidate.recognizer
    no_phrase = replace(recognizer, close_labeler=None, phrase_labeler=None, chart_weights=(0.0, 0.0, 0.0, 0.0),
                        sentence_ends=frozenset(), punctuation=frozenset())
    from core.learning.semantic_argument_antecedent import FEATURES

    antecedent = candidate.antecedent
    kept = len(FEATURES)
    no_recency = replace(antecedent, weight=(*antecedent.weight[:kept],
                                             *(0.0 for _ in antecedent.weight[kept:])))
    return {
        "no_phrase_reading": PeakRecognitionTransducer(candidate.base, no_phrase, candidate.ownership, antecedent),
        "no_recency": PeakRecognitionTransducer(candidate.base, recognizer, candidate.ownership, no_recency),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--order-seed", type=int, default=20261009)
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
    per_comparison_alpha = args.familywise_alpha / len(PAIRED_STRATA)
    tasks = tasks_needed(alpha=per_comparison_alpha, target=args.target_power,
                         discordance=discordance, win_share=win_share)
    strata = build_g04_transfer_corpus(seed=args.seed, tasks_per_stratum=tasks, version=2)
    candidate = peak_recognition_transducer_from_dict(
        json.loads(CANDIDATE.read_text(encoding="utf-8")), restore_base=restore)
    incumbent = restore(json.loads(INCUMBENT.read_text(encoding="utf-8")))
    previous = peak_recognition_transducer_from_dict(
        json.loads(PREVIOUS.read_text(encoding="utf-8")), restore_base=restore)
    if candidate.base.receipt_sha256 != incumbent.receipt_sha256 or previous.base.receipt_sha256 != incumbent.receipt_sha256:
        raise SystemExit("every arm must be built on the incumbent it is compared with")
    novelty = novelty_report(strata, inventory, candidate.inference_step_limit)
    if novelty["failures"]:
        raise SystemExit(f"novelty fails for {novelty['failures']}: {json.dumps(novelty, indent=1)}")

    consumed_seeds = set(inventory["seeds"])
    committed: dict[str, list] = {}
    task_seed = args.seed * 1000
    for name in G04_STRATA:
        rows = []
        for row in strata[name]:
            task_seed += 1
            while task_seed in consumed_seeds:
                task_seed += 1
            rows.append({"id": row.example_id,
                         "source_sha256": hashlib.sha256(row.source_text.encode("utf-8")).hexdigest(),
                         "seed": task_seed})
        committed[name] = rows
    head = subprocess.run(["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    arms_identity = {"candidate": candidate.receipt_sha256, "incumbent": incumbent.receipt_sha256,
                     "v12": previous.receipt_sha256,
                     **{name: model.receipt_sha256 for name, model in lesions(candidate).items()}}
    artifacts = {
        "candidate": _files_sha256([CANDIDATE]),
        "incumbent": _files_sha256([INCUMBENT]),
        "v12": _files_sha256([PREVIOUS]),
        "resident_model": _files_sha256(sorted(path for path in MODEL.iterdir() if path.is_file())),
        "task_generator": _files_sha256([ROOT / path for path in GENERATOR_FILES]),
        "scorer": _files_sha256([ROOT / path for path in SCORER_FILES]),
        "runtime": _files_sha256([ROOT / path for path in RUNTIME_FILES]),
    }
    spec = {
        "campaign": "g04-transfer-v2",
        "hypothesis": (
            "On fresh requests that differ from everything candidate v17 was fitted on or tuned "
            "against in vocabulary, depth or family, v17 produces a program with the reference "
            "program's meaning more often than the incumbent it was built on; and on fresh "
            "requests in scaffolds it never met, it produces the reference meaning every time."
        ),
        "sampling_unit": "independent_task",
        "sampling_description": (
            f"{tasks} requests per stratum from build_g04_transfer_corpus(seed={args.seed}, version=2); "
            "one decode per task and arm; constructions within a stratum taken in turn."
        ),
        "artifacts": artifacts,
        "arm_receipts": arms_identity,
        "source_commit": head,
        "arms": ["incumbent", "candidate", "v12", "no_phrase_reading", "no_recency"],
        "order_seed": args.order_seed,
        "comparisons": [{
            "treatment": "candidate",
            "control": "incumbent",
            "assumptions": {name: {"discordance": discordance, "win_share": win_share}
                            for name in PAIRED_STRATA},
        }],
        "effect_evidence": EFFECT_EVIDENCE,
        "domains": {name: committed[name] for name in PAIRED_STRATA},
        "absolute_domains": {
            "construction": {
                "tasks": committed["construction"],
                "claim": "every committed request decodes to the reference meaning (candidate arm)",
            },
        },
        "secondary": {
            "family_alpha": SECONDARY_ALPHA,
            "per_test_alpha": SECONDARY_ALPHA / len(SECONDARY),
            "test": "exact_one_sided_mcnemar",
            "comparisons": list(SECONDARY),
            "descriptive": "v12 on every task; executed-answer accuracy; decoding with execution unavailable",
        },
        "closure_rule": "G04 closes only if every primary comparison rejects and the construction claim holds",
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
            "runtime failure count as not equivalent, for every arm alike."
        ),
        "secondary_outcomes": [
            "the decoded program's executed answer on the public inputs equals the reference's",
            "the decode succeeded with program execution unavailable (no executable-system "
            "assistance in choosing the program)",
        ],
        "novelty": novelty,
        "first_run_disclosure": (
            "The first run's 248 requests are consumed and none is repeated. Its vocabulary and "
            "depth failures were read row by row; the depth diagnosis led to the recency features, "
            "and v17's decodes of those consumed rows are in the development record. Its vocabulary "
            "table is replaced, not reused; its construction scaffolds and family shapes are reused "
            "with fresh requests, and were not diagnosed."
        ),
        "features": {
            "corpus_kind": CORPUS_KIND,
            "seed": args.seed,
            "examples_per_operation_pair": tasks,
            "max_examples": tasks * len(G04_STRATA),
            "representation": "lexical_mid_final_v1",
        },
    }
    for task in committed["construction"]:
        if task["source_sha256"] in set(spec["consumed"]["source_sha256s"]) or task["seed"] in consumed_seeds:
            raise SystemExit("a construction task repeats consumed development")
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
        "power": [dict(row) for row in plan.parameters["power"]],
        "artifacts": artifacts,
        "arm_receipts": arms_identity,
    }
    (output / "plan_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str),
                                              encoding="utf-8")
    print(json.dumps(summary, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
