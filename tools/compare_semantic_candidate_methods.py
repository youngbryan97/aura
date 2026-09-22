#!/usr/bin/env python3
"""Compare independent learned program selectors on one held-out source fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _digest(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _direct_choice(model, features, spans, kinds, programs) -> int:
    """Score complete alternatives without their post-generation labels."""
    import torch

    with torch.no_grad():
        scores = torch.stack([model.score(features, spans, kinds, program)
                              for program in programs])
    if not torch.isfinite(scores).all().item():
        raise ValueError("direct program evidence is nonfinite")
    return int(scores.argmax().item())


def _portfolio_comparison(*, programs, keys, labels, incumbent_present: bool,
                          ranker_index: int, direct_index: int, public_inputs,
                          source: str, transducer_receipt: str,
                          ranker_receipt: str, direct_receipt: str) -> dict:
    from core.learning.semantic_program_portfolio import select_semantic_program_portfolio

    if (not programs or len(programs) != len(keys) or len(keys) != len(labels)
            or not 0 <= ranker_index < len(programs)
            or not 0 <= direct_index < len(programs)):
        raise ValueError("portfolio comparison needs aligned candidate choices")
    portfolio = select_semantic_program_portfolio(
        proposals={"incumbent": programs[0] if incumbent_present else None,
                   "ranker": programs[ranker_index], "direct": programs[direct_index]},
        provenance={"incumbent": transducer_receipt, "ranker": ranker_receipt,
                    "direct": direct_receipt},
        public_inputs=tuple(public_inputs), observation_sha256=source,
        incumbent="incumbent", composition_budget=0,
    )
    selection_sha = (portfolio.selected_program.sha()
                     if portfolio.selected_program is not None else None)
    return {"portfolio_selected": portfolio.decision.selected,
            "portfolio_correct": (bool(labels[keys.index(selection_sha)])
                                  if selection_sha in keys else False),
            "portfolio_disagreements": sum(relation[2]["status"] == "different"
                                           for relation in portfolio.relations),
            "portfolio_inquiries": len(portfolio.plan_inquiries())}


def _load_direct(report: dict, checkpoint_path: Path, *, fold: int,
                 folds: dict, source_report: dict):
    import torch

    from core.learning.semantic_program_decoder import ProgramDecoderConfig, SemanticProgramDecoder

    plan = report["plan"]
    checkpoint_raw = checkpoint_path.read_bytes()
    expected_train = {source for source, assigned in folds["assignments"].items()
                      if assigned != fold}
    expected_held = {source for source, assigned in folds["assignments"].items()
                     if assigned == fold}
    if (plan["schema"] != "aura.semantic_direct_program_source_probe.v1"
            or plan["fold"] != fold or plan["fold_receipt"] != folds["receipt_sha256"]
            or plan["parent_receipt"] != source_report["transducer_receipt_sha256"]
            or set(plan["training_ids"]) != expected_train
            or set(plan["heldout_ids"]) != expected_held
            or plan["evaluation_controls_training"] is not False
            or report["test_used"] is not False
            or hashlib.sha256(checkpoint_raw).hexdigest()
            != report["history"][-1]["checkpoint_sha256"]):
        raise ValueError("direct decoder checkpoint is not bound to this source fold")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (_digest(checkpoint["plan"]) != _digest(plan)
            or checkpoint["epoch"] != report["history"][-1]["epoch"]):
        raise ValueError("direct decoder weights differ from their published plan")
    model = SemanticProgramDecoder(ProgramDecoderConfig(**plan["config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval(), hashlib.sha256(checkpoint_raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("transducer", "source-report", "candidate-report", "feature-root",
                 "bank-directory", "folds", "ranker-report", "direct-report",
                 "direct-checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )

    configure_refit_environment(args.output)
    import torch

    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.learning.semantic_span_pointer import _hidden_array
    from tools.evaluate_semantic_candidate_ranker import _rankable, _read_bank, _verify_sources

    source_report = json.loads(args.source_report.read_bytes())
    candidate_report = json.loads(args.candidate_report.read_bytes())
    bank_report = json.loads((args.bank_directory / "report.json").read_bytes())
    folds = json.loads(args.folds.read_bytes())
    ranker_report = json.loads(args.ranker_report.read_bytes())
    direct_report = json.loads(args.direct_report.read_bytes())
    transducer = compositional_semantic_program_transducer_from_dict(
        json.loads(args.transducer.read_bytes()))
    bundles = [name + "=" + str(args.feature_root / name) for name in
               source_report["representation_compatibility"]["source_feature_manifest_sha256s"]]
    examples = load_source_examples(transducer, source_report, bundles)
    plan, ids = _verify_sources(source_report, candidate_report, transducer,
                                bank_report, folds, examples)
    held = {source for source in ids if folds["assignments"][source] == args.fold}
    evaluation = ranker_report["evaluation"]
    body = {key: value for key, value in ranker_report.items() if key != "receipt_sha256"}
    if (ranker_report["receipt_sha256"] != _digest(body)
            or ranker_report["fold"] != args.fold
            or ranker_report["source_bank_receipt_sha256"] != bank_report["receipt_sha256"]
            or ranker_report["model_receipt_sha256"] != transducer.receipt_sha256
            or {row["source"] for row in evaluation["rows"]} != held):
        raise ValueError("ranker comparison is not the same held-out bank")
    direct, direct_sha = _load_direct(direct_report, args.direct_checkpoint,
                                       fold=args.fold, folds=folds,
                                       source_report=source_report)
    items = {item.ir.source_text_sha256: item for item in examples if item.split == "train"}
    ranker_rows = {row["source"]: row for row in evaluation["rows"]}
    results = []
    for source in sorted(held):
        row = _read_bank(args.bank_directory / "rows" / f"{source}.json",
                         source=source, plan_sha=plan["plan_sha256"],
                         model_receipt=transducer.receipt_sha256,
                         expected_receipt=bank_report["row_receipts"][source])
        (programs, labels, keys), spans, kinds, _anchors = _rankable(items[source], row)
        features = torch.from_numpy(_hidden_array(items[source].hidden_states)).float()
        chosen = _direct_choice(direct, features, spans, kinds, programs)
        ranker_row = ranker_rows[source]
        if (ranker_row["chosen_index"] >= len(keys)
                or ranker_row["chosen_program_sha256"] != keys[ranker_row["chosen_index"]]
                or ranker_row["selected_correct"] != labels[ranker_row["chosen_index"]]):
            raise ValueError("ranker comparison selected a different candidate bank")
        portfolio = _portfolio_comparison(
            programs=programs, keys=keys, labels=labels,
            incumbent_present=row["bank"]["selected_program_sha256"] is not None,
            ranker_index=ranker_row["chosen_index"], direct_index=chosen,
            public_inputs=items[source].public_inputs, source=source,
            transducer_receipt=transducer.receipt_sha256,
            ranker_receipt=ranker_report["receipt_sha256"], direct_receipt=direct_sha)
        results.append({"source": source, "incumbent_correct": ranker_row["incumbent_correct"],
                        "ranker_correct": ranker_row["selected_correct"],
                        "direct_correct": labels[chosen], "direct_index": chosen,
                        "direct_program_sha256": keys[chosen],
                        "ranker_index": ranker_row["chosen_index"],
                        **portfolio})
    body = {"schema": "aura.semantic_candidate_methods_source_fold.v1",
            "pilot_only": bank_report["pilot_only"], "serving_authority": False,
            "fold": args.fold, "source_bank_receipt_sha256": bank_report["receipt_sha256"],
            "ranker_receipt_sha256": ranker_report["receipt_sha256"],
            "direct_checkpoint_sha256": direct_sha,
            "population": len(results),
            "incumbent_correct": sum(row["incumbent_correct"] for row in results),
            "ranker_correct": sum(row["ranker_correct"] for row in results),
            "direct_correct": sum(row["direct_correct"] for row in results),
            "either_learned_correct": sum(row["ranker_correct"] or row["direct_correct"]
                                          for row in results),
            "any_method_correct": sum(row["incumbent_correct"] or row["ranker_correct"]
                                      or row["direct_correct"] for row in results),
            "both_learned_correct": sum(row["ranker_correct"] and row["direct_correct"]
                                        for row in results),
            "portfolio_correct": sum(row["portfolio_correct"] for row in results),
            "portfolio_inquiries": sum(row["portfolio_inquiries"] for row in results),
            "rows": results}
    payload = json.dumps({**body, "receipt_sha256": _digest(body)}, sort_keys=True).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != payload:
        raise ValueError("candidate method comparison already differs")
    args.output.write_bytes(payload)
    print(json.dumps({key: body[key] for key in (
        "population", "incumbent_correct", "ranker_correct", "direct_correct",
        "either_learned_correct", "both_learned_correct", "any_method_correct",
        "portfolio_correct", "portfolio_inquiries")}), flush=True)


if __name__ == "__main__":
    main()
