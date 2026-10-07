#!/usr/bin/env python3
"""Freeze G05's public-answer comparison before any of its requests is answered.

G05 asks whether an internal gain reaches the answer a person reads. The
internal gain is G04's frozen reader. Her runtime gives a settled answer to
a render with the private channel closed (chat_format.
thinking_enabled_for_generation), so that is where the reader's evidence
reaches a person. On consumed requests (6 October pilots), her closed-channel
answers slipped where the reader did not: long multiplications carried wrong,
entries misread. With the channel open she answered those requests right, at
twenty to forty times the time.

Arms, each one greedy decode through her own template:

* ``ordinary``: the request alone, channel closed;
* ``assisted``: the request, then the reader's program and values as runtime
  evidence, channel closed;
* ``sham``: another request's reading in the same form, channel closed;
* ``ordinary_open``: the request alone, channel open at her serving effort.

Primary: ``assisted`` is exact more often than ``ordinary``, exact one-sided
McNemar in each stratum (composition: five-step requests with large numbers;
lists: a count or lookup over 40 to 64 entries), Bonferroni over the two.
Translation: wherever the reader's answer is right, the assisted public
answer is exact. G05 closes only if both hold.

Secondary, for G06, its own family at alpha 0.05 over two tests: ``assisted``
against ``ordinary_open`` (exact two-sided McNemar: does the cheap path lose
anything to the expensive one), and ``sham`` against ``ordinary`` (exact
two-sided McNemar: what wrong evidence costs). Tokens and seconds per arm are
reported for every arm.

The planned effect is not chosen: it is the conservative bound of the
closed-channel pilot in each stratum, the lower 95% Clopper-Pearson bound on
how often the two arms disagree and the one-sided lower bound on the
assisted arm's share of those disagreements.

Usage:
    g05_public_answer_protocol.py --pilot DIR --inventory FILE --store DIR --output DIR
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

from tools.g04_transfer_protocol import MODEL, _files_sha256, tasks_needed  # noqa: E402

EVIDENCE = Path("~/.aura/rlc-evidence").expanduser()
READER = EVIDENCE / "semantic-peak-direction-v14-20261006/candidate.json"
STRATA = ("composition", "lists")
ARMS = ("ordinary", "assisted", "sham", "ordinary_open")
SECONDARY = (
    {"first": "assisted", "second": "ordinary_open", "alternative": "two-sided"},
    {"first": "sham", "second": "ordinary", "alternative": "two-sided"},
)
SECONDARY_ALPHA = 0.05
GENERATOR_FILES = (
    "core/learning/semantic_program_corpus_replication.py",
    "core/learning/semantic_g05_long_sequence_corpus.py",
)
SCORER_FILES = (
    "core/learning/semantic_program_ordinary_baseline.py",
    "core/brain/llm/latent_cortex/experiment_tasks.py",
    "tools/run_g05_public_answers.py",
)
RUNTIME_FILES = (
    "core/brain/llm/chat_format.py",
    "core/utils/injected_blocks.py",
    "tools/run_g05_public_answers.py",
    "tools/run_g04_transfer.py",
    "core/learning/semantic_operation_peaks.py",
    "core/learning/semantic_argument_antecedent.py",
    "core/learning/semantic_program_transducer_fitting.py",
)


def pilot_effect(pilot: Path, stratum: str) -> dict:
    """Discordance and win share bounds from the closed-channel pilot's paired rows."""
    from scipy.stats import beta

    rows: dict[str, dict[str, bool]] = {}
    for arm in ("ordinary", "assisted"):
        for path in (pilot / stratum / "rows" / arm).glob("*.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("thinking") != "closed":
                raise SystemExit(f"{path} was not decoded with the channel closed")
            rows.setdefault(row["task_id"], {})[arm] = bool(row["answer_exact"])
    paired = [row for row in rows.values() if len(row) == 2]
    wins = sum(row["assisted"] and not row["ordinary"] for row in paired)
    losses = sum(row["ordinary"] and not row["assisted"] for row in paired)
    k, n = wins + losses, len(paired)
    if k == 0 or wins <= losses:
        raise SystemExit(f"the {stratum} pilot shows no paired gain to plan for")
    discordance = float(beta.ppf(0.025, k, n - k + 1))
    win_share = float(beta.ppf(0.05, wins, k - wins + 1)) if wins < k else float(0.05 ** (1 / k))
    return {"pilot_pairs": n, "assisted_only": wins, "ordinary_only": losses,
            "discordance": discordance, "win_share": win_share}


def build_strata(seed: int, tasks: dict[str, int]) -> tuple[dict[str, tuple], int]:
    """Fresh requests in each stratum, and how many composition requests each generator cell gave."""
    from core.learning.semantic_g05_long_sequence_corpus import build_g05_long_sequence_corpus
    from core.learning.semantic_program_corpus_replication import (
        build_semantic_program_natural_weave_replication_corpus,
    )

    # One request per schema and domain for each unit of examples_per_schema_domain.
    cells = len(build_semantic_program_natural_weave_replication_corpus(seed=seed, examples_per_schema_domain=1))
    per_cell = -(-tasks["composition"] // cells)
    composition = build_semantic_program_natural_weave_replication_corpus(
        seed=seed, examples_per_schema_domain=per_cell)
    lists = build_g05_long_sequence_corpus(seed=seed + 1, tasks=tasks["lists"])
    return {"composition": composition, "lists": lists}, per_cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--order-seed", type=int, default=20261013)
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument("--target-power", type=float, default=0.9)
    args = parser.parse_args()

    from core.evaluation.paired_replication import build_paired_replication_plan
    from core.governance_context import local_internal_governed_scope

    inventory = json.loads(args.inventory.expanduser().read_text(encoding="utf-8"))
    consumed_seeds = set(inventory["seeds"])
    if {args.seed, args.seed + 1} & consumed_seeds:
        raise SystemExit("a corpus seed was already consumed")
    pilot = args.pilot.expanduser()
    effects = {stratum: pilot_effect(pilot, stratum) for stratum in STRATA}
    alpha = args.familywise_alpha / len(STRATA)
    tasks = {stratum: tasks_needed(alpha=alpha, target=args.target_power,
                                   discordance=effects[stratum]["discordance"],
                                   win_share=effects[stratum]["win_share"]) for stratum in STRATA}
    strata, per_cell = build_strata(args.seed, tasks)
    consumed_sha = {row["source_sha256"] for row in inventory["examples"]}
    committed: dict[str, list] = {}
    task_seed = args.seed * 1000
    for stratum, rows in strata.items():
        entries = []
        for row in rows:
            source = hashlib.sha256(row.source_text.encode("utf-8")).hexdigest()
            if source in consumed_sha:
                raise SystemExit(f"a {stratum} request repeats a consumed text")
            task_seed += 1
            while task_seed in consumed_seeds:
                task_seed += 1
            entries.append({"id": row.example_id, "source_sha256": source, "seed": task_seed})
        committed[stratum] = entries
    head = subprocess.run(["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    artifacts = {
        "candidate": _files_sha256([READER]),
        "resident_model": _files_sha256(sorted(path for path in MODEL.iterdir() if path.is_file())),
        "task_generator": _files_sha256([ROOT / path for path in GENERATOR_FILES]),
        "scorer": _files_sha256([ROOT / path for path in SCORER_FILES]),
        "runtime": _files_sha256([ROOT / path for path in RUNTIME_FILES]),
    }
    spec = {
        "campaign": "g05-public-answers-v1",
        "hypothesis": (
            "Rendered as her runtime renders a settled answer (private channel closed), her freely "
            "decoded public answer is exact more often with the frozen reader's reading as runtime "
            "evidence than without it; and wherever the reader is right, the assisted answer is exact."
        ),
        "sampling_unit": "independent_task",
        "sampling_description": (
            f"composition: build_semantic_program_natural_weave_replication_corpus(seed={args.seed}, "
            f"examples_per_schema_domain={per_cell}); lists: "
            f"build_g05_long_sequence_corpus(seed={args.seed + 1}, tasks={tasks['lists']}); one decode per "
            "task and arm."
        ),
        "artifacts": artifacts,
        "source_commit": head,
        "arms": list(ARMS),
        "order_seed": args.order_seed,
        "comparisons": [{
            "treatment": "assisted", "control": "ordinary",
            "assumptions": {stratum: {"discordance": effects[stratum]["discordance"],
                                      "win_share": effects[stratum]["win_share"]} for stratum in STRATA},
        }],
        "effect_evidence": {"source": "closed-channel pilot on consumed requests, 6 October",
                            "bound": "lower 95%: Clopper-Pearson (two-sided) for discordance, "
                                     "one-sided for the assisted share", **{s: effects[s] for s in STRATA}},
        "domains": committed,
        "translation_claim": "wherever the reader's answer equals the reference, the assisted public answer is exact",
        "secondary": {"family_alpha": SECONDARY_ALPHA, "per_test_alpha": SECONDARY_ALPHA / len(SECONDARY),
                      "test": "exact_mcnemar", "comparisons": list(SECONDARY),
                      "descriptive": "generated tokens and seconds per arm; terminations; reader accuracy"},
        "closure_rule": "G05 closes only if both primary comparisons reject and the translation claim holds",
        "answer_rule": "the last exact integer in the public reply (parse_integral_numeric_claim); "
                       "the private channel is never read; a budget stop counts as not exact",
        "familywise_alpha": args.familywise_alpha,
        "target_power": args.target_power,
        "consumed": {
            "task_ids": [row["example_id"] for row in inventory["examples"]],
            "source_sha256s": [row["source_sha256"] for row in inventory["examples"]],
            "seeds": sorted(consumed_seeds),
            "inventory_manifest_sha256s": inventory["manifest_sha256s"],
        },
        "features": {
            "composition": {"corpus_kind": "natural_weave_replication_6x5", "seed": args.seed,
                            "examples_per_operation_pair": per_cell},
            "lists": {"corpus_kind": "g05_long_sequence_v1", "seed": args.seed + 1,
                      "examples_per_operation_pair": tasks["lists"]},
            "representation": "lexical_mid_final_v1",
        },
    }
    plan = build_paired_replication_plan(spec)
    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    (output / "spec.json").write_text(json.dumps(spec, indent=1, sort_keys=True), encoding="utf-8")
    with local_internal_governed_scope("evaluation.paired_replication", domain="file_write"):
        path = plan.write(args.store.expanduser())
    summary = {"plan_hash": plan.plan_hash, "plan_path": str(path), "tasks": {s: len(committed[s]) for s in STRATA},
               "effects": effects, "power": [dict(row) for row in plan.parameters["power"]],
               "artifacts": artifacts}
    (output / "plan_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str),
                                              encoding="utf-8")
    arm_order = {task: order for task, order in plan.parameters["arm_order_by_task"].items()}
    (output / "arm_order.json").write_text(json.dumps(arm_order, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
