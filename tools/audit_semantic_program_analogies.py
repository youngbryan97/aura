#!/usr/bin/env python3
"""Audit source-fit program analogies on untouched construction groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.learning.procedure_induction import Instruction, Program  # noqa: E402
from core.learning.semantic_program_analogy import find_program_analogues  # noqa: E402
from tools.calibrate_semantic_native_choices import (  # noqa: E402
    partition_source_rows,
    source_observations,
)
from tools.evaluate_semantic_candidate_ranker import _read_bank  # noqa: E402
from tools.evaluate_semantic_native_checkpoint import digest, verified_document  # noqa: E402
from tools.probe_semantic_proposer_crossfit import _save_if_absent  # noqa: E402


def _programs(bank):
    result = {}
    for candidate in bank["bank"]["candidates"]:
        payload = candidate["program"]
        program = Program(len(bank["bank"]["input_spans"]), tuple(
            Instruction(operation, tuple(arguments))
            for operation, arguments in payload["instructions"]))
        if program.sha() != candidate["program_sha256"] or payload["sha"] != program.sha():
            raise ValueError("candidate program identity differs from source bank")
        result[program.sha()] = program
    return result


def audit(rows, banks, *, excluded, seed, top_k, trials, max_seconds=300.):
    if not 0 < max_seconds <= 3600:
        raise ValueError("analogy audit needs a finite time bound")
    started = time.monotonic()
    fit, _tune, admission = partition_source_rows(rows, excluded_ids=excluded, seed=seed)
    fit_groups = {row["construction"] for row in fit}
    precedents = {}
    for row in fit:
        for sha, program in banks[row["source"]].items():
            if row["labels"][sha] is True:
                precedents[sha] = program
    if not precedents:
        raise ValueError("analogy audit has no verified fit precedents")

    def evaluate(population):
        observations = []
        unsupported = 0
        for row in population:
            if row["construction"] in fit_groups:
                raise ValueError("analogy audit must keep construction groups disjoint")
            for sha, program in sorted(banks[row["source"]].items()):
                if time.monotonic() - started > max_seconds:
                    raise TimeoutError("analogy audit exceeded its declared bound")
                if type(row["labels"][sha]) is not bool:
                    continue
                if program.n_inputs + len(program.instructions) > 7:
                    unsupported += 1
                    continue
                result = find_program_analogues(
                    program, tuple(precedents.values()), (), top_k=top_k, trials=trials)
                retrieved = result["retrieved"]
                observations.append({"source": row["source"], "construction": row["construction"],
                                     "program_sha256": sha, "correct": row["labels"][sha],
                                     "exact_fit_program": sha in precedents,
                                     "analogy_holds": any(item["analogy"]["holds"] for item in retrieved),
                                     "floor_equivalent": any(item["equivalent"] for item in retrieved),
                                     "retrieved": len(retrieved)})
        counts = Counter((row["correct"], row["analogy_holds"], row["floor_equivalent"])
                         for row in observations)
        return {"sources": len(population), "candidates": len(observations),
                "unsupported_graphs": unsupported,
                "counts": [{"correct": correct, "analogy_holds": holds,
                            "floor_equivalent": equivalent, "count": count}
                           for (correct, holds, equivalent), count in sorted(counts.items())],
                "rows_sha256": digest(observations)}

    return {"fit_sources": len(fit), "fit_groups": sorted(fit_groups),
            "fit_verified_programs": len(precedents),
            "admission": evaluate(admission),
            "fit_labels_used_for_precedent_selection": True,
            "admission_labels_used_for_retrieval": False,
            "serving_authority": False, "qualification_evidence": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-bank", type=Path, required=True)
    parser.add_argument("--origin-bank", type=Path, required=True)
    parser.add_argument("--native-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=300.)
    args = parser.parse_args()
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    configure_refit_environment(args.output)
    rows, excluded, bank_report, reports = source_observations(
        args.calibration_bank, (args.native_calibration,), origin_directory=args.origin_bank)
    bank_plan = verified_document(args.calibration_bank / "plan.json", "plan_sha256")
    banks = {}
    for row in rows:
        source = row["source"]
        bank = _read_bank(args.calibration_bank / "rows" / f"{source}.json", source=source,
                          plan_sha=bank_plan["plan_sha256"],
                          model_receipt=bank_report["candidate_receipt_sha256"],
                          expected_receipt=bank_report["row_receipts"][source])
        banks[source] = _programs(bank)
    result = audit(rows, banks, excluded=excluded, seed=args.seed,
                   top_k=args.top_k, trials=args.trials, max_seconds=args.max_seconds)
    basis = {"schema": "aura.semantic_program_analogy_audit.v1", **result,
             "bank_receipt_sha256": bank_report["receipt_sha256"],
             "native_receipt_sha256": reports[0]["receipt_sha256"],
             "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "top_k": args.top_k, "trials": args.trials, "seed": args.seed}
    _save_if_absent(args.output, {**basis, "receipt_sha256": digest(basis)})
    print(json.dumps({"stage": "complete", "fit_sources": result["fit_sources"],
                      "admission": result["admission"]}), flush=True)


if __name__ == "__main__":
    main()
