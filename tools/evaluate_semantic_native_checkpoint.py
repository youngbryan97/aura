#!/usr/bin/env python3
"""Resume a source-selected native checkpoint on its frozen candidate bank."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verified_document(path, field="receipt_sha256"):
    value = json.loads(path.read_bytes())
    if value.get(field) != digest({key: item for key, item in value.items() if key != field}):
        raise ValueError(f"native evidence digest differs: {path.name}")
    return value


def selected_checkpoint(directory):
    """Select only from a complete declared training schedule's calibration receipts."""
    plan = verified_document(directory / "plan.json", "plan_sha256")
    from core.learning.semantic_native_codec import register_encoding_from_plan
    register_encoding_from_plan(plan)
    if (plan.get("schema") != "aura.semantic_native_fit_plan.v1"
            or plan.get("held_labels_used_for_fit_or_selection") is not False
            or plan.get("serving_authority") is not False
            or plan.get("qualification_evidence") is not False):
        raise ValueError("native fit lacks source-only research authority")
    steps, interval = plan["steps"], plan["save_every"]
    if type(steps) is not int or type(interval) is not int or min(steps, interval) < 1 or steps % interval:
        raise ValueError("native fit checkpoint schedule differs")
    expected = set(range(interval, steps + 1, interval))
    if plan.get("unfitted_checkpoint_eligible") is True:
        expected.add(0)
    paths = list(directory.glob("checkpoint-*.json"))
    rows = [verified_document(path) for path in paths]
    if ({row["step"] for row in rows} != expected or len(rows) != len(expected)
            or any(type(row["step"]) is not int or row["plan_sha256"] != plan["plan_sha256"]
                   or not math.isfinite(row["calibration_loss"]) for row in rows)):
        raise ValueError("native fit has incomplete or mismatched checkpoints")
    for row in rows:
        weights = directory / f"checkpoint-{row['step']}.safetensors"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != row["weights_sha256"]:
            raise ValueError("native checkpoint weight digest differs")
    fit, cal, held = (set(plan[key]) for key in ("fit_ids", "calibration_ids", "held_ids"))
    if not fit or not cal or not held or fit & cal or fit & held or cal & held:
        raise ValueError("native fit source partitions overlap or are empty")
    selected = min(rows, key=lambda row: (row["calibration_loss"], row["step"]))
    report_path = directory / "report.json"
    if report_path.exists():
        report = verified_document(report_path)
        if (report["plan_sha256"] != plan["plan_sha256"]
                or report["selected_step"] != selected["step"]
                or report["selected_calibration_loss"] != selected["calibration_loss"]):
            raise ValueError("native fit report differs from source calibration")
    return plan, selected


def observed_program_reach(labels):
    if any(value is True for value in labels):
        return True
    return None if any(value is None for value in labels) else False


def verify_replay_row(row, *, source, plan_sha256, programs=None, labels=None,
                      incumbent_available=True):
    if (row.get("receipt_sha256") != digest({key: value for key, value in row.items()
                                             if key != "receipt_sha256"})
            or row.get("source") != source or row.get("plan_sha256") != plan_sha256
            or row.get("labels_available_to_scorer") is not False):
        raise ValueError("native replay row identity differs")
    if row.get("scored") is False:
        if (programs is not None or row.get("unrankable_reason") != "ordinary_decode_unavailable"
                or any(row.get(key) != [] for key in
                       ("program_sha256s", "scores", "pretrained_scores"))
                or any(row.get(key) is not None for key in
                       ("chosen_program_sha256", "pretrained_program_sha256", "selected_correct",
                        "pretrained_correct", "bank_reachable"))
                or row.get("incumbent_correct") is not False
                or row.get("incumbent_available") is not False):
            raise ValueError("unscored native calibration row claims unavailable evidence")
        return row
    keys, scores, controls = (row[key] for key in
                             ("program_sha256s", "scores", "pretrained_scores"))
    if (not keys or len(set(keys)) != len(keys) or len(keys) != len(scores)
            or len(keys) != len(controls)
            or any(not math.isfinite(value) for value in (*scores, *controls))
            or keys[max(range(len(scores)), key=scores.__getitem__)] != row["chosen_program_sha256"]
            or keys[max(range(len(controls)), key=controls.__getitem__)]
               != row["pretrained_program_sha256"]):
        raise ValueError("native replay scores differ from selected programs")
    if programs is not None:
        statuses = dict(zip(programs, labels, strict=True))
        expected_incumbent = labels[0] if incumbent_available else False
        if (list(keys) != list(programs) or row["incumbent_correct"] != expected_incumbent
                or row.get("incumbent_available", True) != incumbent_available
                or row["bank_reachable"] != observed_program_reach(labels)
                or row["selected_correct"] != statuses[row["chosen_program_sha256"]]
                or row["pretrained_correct"] != statuses[row["pretrained_program_sha256"]]):
            raise ValueError("native replay outcome differs from independent bank grading")
    return row


def source_calibration_labels(bank):
    """Only witnessed differences become negative calibration observations."""
    values = {"equivalent": True, "different": False, "unknown": None}
    result = {}
    for row in bank["diagnosis"]["comparisons"]:
        key, status = row["program_sha256"], row["status"]
        if key in result or status not in values:
            raise ValueError("source calibration comparison is duplicated or unrecognized")
        result[key] = values[status]
    return result


def verified_calibration_replay_basis(directory, *, origin_directory, origin_plan, origin_report):
    from tools.probe_semantic_proposer_crossfit import verify_source_calibration_bank

    plan = verified_document(directory / "plan.json", "plan_sha256")
    report = verified_document(directory / "report.json")
    candidate_sha = hashlib.sha256((origin_directory / "candidate.json").read_bytes()).hexdigest()
    verify_source_calibration_bank(plan, report, origin_plan=origin_plan,
                                   origin_report=origin_report, candidate_sha256=candidate_sha)
    return plan, report, tuple(plan["evaluated_ids"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-directory", "parent", "source-report", "folds", "bank", "directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True)
    parser.add_argument("--held-per-construction", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=1800.)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--source-calibration-bank", type=Path,
                        help="score a complete source calibration bank, never held requests")
    args = parser.parse_args()
    if args.held_per_construction < 1 or not 0 < args.max_seconds <= 14400:
        parser.error("positive source sampling and finite replay bound required")
    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
        source_bundle_arguments,
    )
    configure_refit_environment(args.directory / "report.json")
    from core.brain.llm.model_registry import get_active_cortex_spec
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.train_nested_semantic_ranker import _verified_pair
    from tools.train_semantic_atom_ranker import validate_atom_partition
    from tools.train_semantic_native_program import construction_subset

    training, selected = selected_checkpoint(args.training_directory)
    from core.learning.semantic_native_codec import (
        NATIVE_CODEC_IMPLEMENTATION_PATHS,
        register_encoding_from_plan,
    )
    outer, bank_report = _verified_pair(args.bank)
    source_raw, parent_raw, folds_raw = (path.read_bytes() for path in
                                        (args.source_report, args.parent, args.folds))
    parent = compositional_semantic_program_transducer_from_dict(json.loads(parent_raw))
    if (hashlib.sha256(source_raw).hexdigest() != training["source_report_sha256"]
            or outer["source_report_sha256"] != training["source_report_sha256"]
            or hashlib.sha256(folds_raw).hexdigest() != outer["folds_sha256"]
            or parent.receipt_sha256 != outer["parent_receipt_sha256"]
            or outer["plan_sha256"] != training["bank_plan_sha256"]
            or bank_report["receipt_sha256"] != training["bank_receipt_sha256"]
            or outer["fit_ids"] != training["fit_ids"]
            or not set(training["calibration_ids"]) <= set(outer["calibration_ids"])):
        raise ValueError("native replay source or frozen bank differs from training")
    examples = load_source_examples(parent, json.loads(source_raw), source_bundle_arguments(
        json.loads(source_raw), bundles=args.bundle))
    validate_atom_partition(examples, outer, json.loads(folds_raw))
    identities = construction_subset(examples, outer["held_ids"],
                                      per_construction=args.held_per_construction)
    scoring_directory, scoring_plan, scoring_report = args.bank, outer, bank_report
    if args.source_calibration_bank is not None:
        scoring_directory = args.source_calibration_bank
        scoring_plan, scoring_report, identities = verified_calibration_replay_basis(
            scoring_directory, origin_directory=args.bank,
            origin_plan=outer, origin_report=bank_report)
    forbidden = set(training["fit_ids"])
    if args.source_calibration_bank is None:
        forbidden |= set(training["calibration_ids"])
    else:
        forbidden |= set(outer["held_ids"])
    if set(identities) & forbidden:
        raise ValueError("native replay crosses a source fit or calibration boundary")
    spec = get_active_cortex_spec(force_refresh=True)
    if (spec is None or not spec.exact_identity
            or spec.descriptor_sha256 != training["model_descriptor_sha256"]
            or spec.pointer_sha256 != training["pointer_sha256"]
            or spec.model_path.resolve() != Path(training["model_path"]).resolve()):
        raise ValueError("native replay resident identity differs")
    paths = [ROOT / name for name in (
        "tools/evaluate_semantic_native_checkpoint.py", "tools/train_semantic_native_program.py",
        "core/learning/frozen_decoder_prefix.py", "core/learning/semantic_native_program.py",
        "core/brain/llm/decoder_topology.py", "tools/evaluate_semantic_candidate_ranker.py",
        "core/learning/semantic_program_feature_materialization.py", *NATIVE_CODEC_IMPLEMENTATION_PATHS)]
    if args.source_calibration_bank is not None:
        paths.append(ROOT / "tools/probe_semantic_proposer_crossfit.py")
    implementation = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in paths}
    body = {"schema": "aura.semantic_native_replay_plan.v1",
        "training_plan_sha256": training["plan_sha256"],
        "checkpoint_receipt_sha256": selected["receipt_sha256"],
        "weights_sha256": selected["weights_sha256"], "selected_step": selected["step"],
        "bank_receipt_sha256": bank_report["receipt_sha256"], "held_ids": identities,
        "heldout_axis": outer["heldout_axis"], "loss_scope": training["loss_scope"],
        "semantic_decision_basis": training.get("semantic_decision_basis", "program_atoms_v1"),
        "register_encoding": register_encoding_from_plan(training),
        "model_descriptor_sha256": spec.descriptor_sha256, "pointer_sha256": spec.pointer_sha256,
        "implementation": implementation, "max_seconds": args.max_seconds,
        "fit_updates": 0, "held_labels_used_for_fit_or_selection": False,
        "serving_authority": False, "qualification_evidence": False}
    if args.source_calibration_bank is not None:
        del body["held_ids"]
        body.update({"schema": "aura.semantic_native_source_calibration_plan.v1",
                     "evaluated_ids": identities, "held_rows_evaluated": False,
                     "source_calibration_plan_sha256": scoring_plan["plan_sha256"],
                     "source_calibration_receipt_sha256": scoring_report["receipt_sha256"],
                     "checkpoint_calibration_ids": training["calibration_ids"]})
    plan = {**body, "plan_sha256": digest(body)}
    _save_if_absent(args.directory / "plan.json", plan)
    if args.plan_only:
        print(json.dumps({"stage": "plan_only", "population": len(identities),
                          "selected_step": selected["step"], "plan_sha256": plan["plan_sha256"]}),
              flush=True)
        return
    from types import SimpleNamespace

    import mlx.core as mx
    from mlx.utils import tree_map
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from core.learning.frozen_decoder_prefix import FrozenDecoderPrefix, NativeDecoderSuffix
    from core.learning.semantic_native_codec import (
        native_sequence_for_encoding,
        register_encoding_from_plan,
    )
    from core.learning.semantic_native_program import source_text_from_tokens
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane
    from tools.evaluate_semantic_candidate_ranker import _rankable_or_none, _read_bank
    from tools.train_semantic_native_program import native_loss

    started, rows = time.monotonic(), []
    def bound():
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError("native replay bound reached; rows remain resumable")
    with (standalone_model_lane(owner_id=f"semantic-native-replay:{args.directory.name}",
            model_path=str(spec.model_path), purpose="evaluation", preemptible=False,
            metadata={"production_effect": False}), mlx_memory_envelope(fraction=.80)):
        model, tokenizer = load(str(spec.model_path))
        model.freeze()
        model.eval()
        split = len(model.layers) - training["suffix_layers"]
        prefix, suffix = FrozenDecoderPrefix(model, split_at=split), NativeDecoderSuffix(model, split_at=split)
        mx.random.seed(training["seed"])
        linear_to_lora_layers(model, training["suffix_layers"], {
            "rank": training["rank"], "scale": 16., "dropout": 0., "keys": training["adapter_keys"]})
        baseline = tree_map(lambda value: mx.array(value), suffix.trainable_parameters())
        mx.eval(baseline)
        model.load_weights(str(args.training_directory / f"checkpoint-{selected['step']}.safetensors"),
                           strict=False)
        adapted = tree_map(lambda value: mx.array(value), suffix.trainable_parameters())
        mx.eval(adapted)
        items = {item.ir.source_text_sha256: item for item in examples if item.split == "train"}
        del examples, parent
        gc.collect()
        for identity in identities:
            bound()
            path = args.directory / "rows" / f"{identity}.json"
            bank = _read_bank(scoring_directory / "rows" / f"{identity}.json", source=identity,
                plan_sha=scoring_plan["plan_sha256"],
                model_receipt=scoring_report["candidate_receipt_sha256"],
                expected_receipt=scoring_report["row_receipts"][identity])
            view = _rankable_or_none(SimpleNamespace(public_inputs=items[identity].public_inputs), bank)
            if view is None:
                if args.source_calibration_bank is None:
                    raise ValueError("native replay bank has no complete typed candidates")
                body = {"source": identity, "construction": items[identity].construction_id,
                    "plan_sha256": plan["plan_sha256"], "program_sha256s": [], "scores": [],
                    "pretrained_scores": [], "chosen_program_sha256": None,
                    "pretrained_program_sha256": None, "incumbent_correct": False,
                    "incumbent_available": False, "selected_correct": None,
                    "pretrained_correct": None, "bank_reachable": None,
                    "labels_available_to_scorer": False, "scored": False,
                    "unrankable_reason": "ordinary_decode_unavailable"}
                row = {**body, "receipt_sha256": digest(body)}
                verify_replay_row(row, source=identity, plan_sha256=plan["plan_sha256"])
                _save_if_absent(path, row)
                rows.append(row)
                print(json.dumps({"stage": "unrankable", "observed": len(rows),
                                  "population": len(identities)}), flush=True)
                continue
            (programs, labels, keys), *_rest = view
            if args.source_calibration_bank is not None:
                outcomes = source_calibration_labels(bank)
                labels = tuple(outcomes[key] for key in keys)
            incumbent_available = bank["bank"]["selected_program_sha256"] is not None
            if path.exists():
                rows.append(verify_replay_row(json.loads(path.read_bytes()), source=identity,
                    plan_sha256=plan["plan_sha256"], programs=keys, labels=labels,
                    incumbent_available=incumbent_available))
                continue
            source = source_text_from_tokens(items[identity], tokenizer)
            scores, controls = [], []
            for program in programs:
                bound()
                sequence = native_sequence_for_encoding(source, program, tokenizer,
                    register_encoding=register_encoding_from_plan(training),
                    max_tokens=training["max_sequence_tokens"],
                    decision_basis=plan["semantic_decision_basis"])
                hidden = prefix.capture(mx.array([sequence.tokens[:-1]], dtype=mx.int32))
                scores.append(-native_loss(suffix, hidden, sequence, summed=True,
                                           scope=training["loss_scope"]).item())
                suffix.update(baseline)
                try:
                    controls.append(-native_loss(suffix, hidden, sequence, summed=True,
                                                 scope=training["loss_scope"]).item())
                finally:
                    suffix.update(adapted)
            chosen = max(range(len(scores)), key=scores.__getitem__)
            control = max(range(len(controls)), key=controls.__getitem__)
            row = {"source": identity, "construction": items[identity].construction_id,
                "plan_sha256": plan["plan_sha256"], "program_sha256s": keys,
                "scores": scores, "pretrained_scores": controls,
                "chosen_program_sha256": keys[chosen], "pretrained_program_sha256": keys[control],
                "incumbent_correct": labels[0] if incumbent_available else False,
                "selected_correct": labels[chosen],
                "pretrained_correct": labels[control],
                "bank_reachable": observed_program_reach(labels),
                "labels_available_to_scorer": False}
            if args.source_calibration_bank is not None or not incumbent_available:
                row["incumbent_available"] = incumbent_available
            row = {**row, "receipt_sha256": digest(row)}
            verify_replay_row(row, source=identity, plan_sha256=plan["plan_sha256"],
                              programs=keys, labels=labels, incumbent_available=incumbent_available)
            _save_if_absent(path, row)
            rows.append(row)
            print(json.dumps({"stage": "held", "observed": len(rows), "population": len(identities)}),
                  flush=True)
        current = get_active_cortex_spec(force_refresh=True)
        if (current is None or current.descriptor_sha256 != spec.descriptor_sha256
                or current.pointer_sha256 != spec.pointer_sha256
                or any(path.read_bytes() != raw for path, raw in (
                    (args.source_report, source_raw), (args.parent, parent_raw), (args.folds, folds_raw)))
                or any(hashlib.sha256(path.read_bytes()).hexdigest() != implementation[
                       str(path.relative_to(ROOT))] for path in paths)):
            raise ValueError("native replay implementation or model drifted")
        current_training, current_selected = selected_checkpoint(args.training_directory)
        if current_training != training or current_selected != selected:
            raise ValueError("native replay training basis changed")
        if args.source_calibration_bank is not None:
            current_basis = verified_calibration_replay_basis(
                scoring_directory, origin_directory=args.bank,
                origin_plan=outer, origin_report=bank_report)
            if current_basis != (scoring_plan, scoring_report, identities):
                raise ValueError("native source calibration bank changed")
        result = {"schema": "aura.semantic_native_replay.v1", "plan_sha256": plan["plan_sha256"],
            "population": len(rows), "rows": rows, "fit_updates": 0,
            "incumbent_correct": sum(row["incumbent_correct"] is True for row in rows),
            "native_correct": sum(row["selected_correct"] is True for row in rows),
            "pretrained_correct": sum(row["pretrained_correct"] is True for row in rows),
            "bank_reachable": sum(row["bank_reachable"] is True for row in rows),
            "gains": sum(row["selected_correct"] is True and row["incumbent_correct"] is False
                         for row in rows),
            "regressions": sum(row["incumbent_correct"] is True and row["selected_correct"] is False
                               for row in rows),
            "elapsed_seconds": time.monotonic() - started,
            "serving_authority": False, "qualification_evidence": False}
        if args.source_calibration_bank is not None:
            result.update({"schema": "aura.semantic_native_source_calibration.v1",
                           "held_rows_evaluated": False,
                           "scored_population": sum(row.get("scored", True) is True
                                                    for row in rows),
                           "measured_pairs": sum(type(row["incumbent_correct"]) is bool
                               and type(row["selected_correct"]) is bool for row in rows),
                           "checkpoint_calibration_ids": training["calibration_ids"]})
        _save_if_absent(args.directory / "report.json", {**result, "receipt_sha256": digest(result)})
        print(json.dumps({key: value for key, value in result.items() if key != "rows"}), flush=True)


if __name__ == "__main__":
    main()
