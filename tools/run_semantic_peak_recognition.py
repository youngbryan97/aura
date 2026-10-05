#!/usr/bin/env python3
"""Fit peak operation recognition on training sources and measure it on every development cohort.

Writes, under ``--output``:

* ``recognizer.json`` and ``candidate.json``: the recognizer fitted on every
  training source, and the incumbent with it in place of operation proposal;
* ``development/``: the cohort audit of every train and validation source;
* ``folds/<k>/``: each frozen construction fold audited by a recognizer fitted
  without it, so the training rows are also scored on constructions it never saw;
* ``composition/<name>/<arm>/``: both transducers on an exposed composition bundle;
* ``report.json``: paired comparisons against the incumbent's own cohort report.

The incumbent's grounding and argument heads are used unchanged in every arm.
No test row of the source cohorts is read. Every per-source result is written
once and reused on a rerun, as the cohort audit does.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_EVIDENCE = Path.home() / ".aura" / "rlc-evidence"
SCHEMA = "aura.semantic_peak_recognition_run.v1"


def family_of(construction_id: str) -> str:
    """The source family a bound construction id names."""
    family, separator, _ = construction_id.partition(":")
    if not separator or not family:
        raise ValueError(f"construction id names no family: {construction_id}")
    return family


def _equivalent(row: Mapping[str, Any]) -> bool:
    return row["observation"]["semantic_status"] == "equivalent"


def paired_development_comparison(
    incumbent_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Per split and family: each arm's equivalent count, and the sources that changed."""
    incumbent = {row["source_text_sha256"]: row for row in incumbent_rows}
    candidate = {row["source_text_sha256"]: row for row in candidate_rows}
    if set(incumbent) != set(candidate) or len(incumbent) != len(incumbent_rows):
        raise ValueError("paired comparison needs the same sources on both arms, once each")
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for source in sorted(candidate):
        row = candidate[source]
        if (incumbent[source]["split"], incumbent[source]["construction_id"]) != (
            row["split"],
            row["construction_id"],
        ):
            raise ValueError(f"the arms disagree about a source's split or construction: {source}")
        group = groups.setdefault(
            (row["split"], family_of(row["construction_id"])),
            {"count": 0, "incumbent": 0, "candidate": 0, "gains": [], "losses": []},
        )
        before, after = _equivalent(incumbent[source]), _equivalent(row)
        group["count"] += 1
        group["incumbent"] += before
        group["candidate"] += after
        if after and not before:
            group["gains"].append(source)
        if before and not after:
            group["losses"].append(source)
    splits: dict[str, Any] = {}
    for (split, family), group in sorted(groups.items()):
        entry = splits.setdefault(
            split,
            {"count": 0, "incumbent": 0, "candidate": 0, "gains": 0, "losses": 0, "families": {}},
        )
        entry["families"][family] = group
        for key in ("count", "incumbent", "candidate"):
            entry[key] += group[key]
        entry["gains"] += len(group["gains"])
        entry["losses"] += len(group["losses"])
    return splits


def representation_differences(first: Mapping[str, Any], second: Mapping[str, Any]) -> list[str]:
    """Dotted paths where two worker representation bases disagree."""

    def flatten(value: Any, prefix: str) -> Iterable[tuple[str, Any]]:
        if isinstance(value, Mapping):
            for key in value:
                yield from flatten(value[key], f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from flatten(item, f"{prefix}[{index}]")
        else:
            yield prefix, value

    left, right = dict(flatten(first, "")), dict(flatten(second, ""))
    return sorted(
        key for key in set(left) | set(right) if left.get(key, ...) != right.get(key, ...)
    )


def _write_once(path: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    """Write a document, or confirm the one already there is the same."""
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent

    payload = (json.dumps(document, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not atomic_write_bytes_if_absent(path, payload, mode=0o400) and path.read_bytes() != payload:
        raise FileExistsError(f"{path} holds a different result")
    return dict(document)


def _representation(manifest: Mapping[str, Any]) -> dict[str, Any]:
    from core.brain.llm.latent_cortex.runtime_identity import worker_representation_basis
    from core.learning.semantic_program_basis import _manifest_bases

    bases = list(_manifest_bases(manifest).values())
    representations = [worker_representation_basis(receipt) for receipt in bases]
    if any(item != representations[0] for item in representations[1:]):
        raise ValueError("a feature bundle mixes neural representations")
    return representations[0]


def _answer(ir: Any, inputs: Any, expected: Any) -> bool:
    from core.learning.semantic_program_execution import execute_semantic_program

    try:
        return execute_semantic_program(ir, inputs).result == expected
    except (ArithmeticError, RuntimeError, TypeError, ValueError):
        return False


class _Remembering:
    """A transducer that keeps the outcome of its last decode, so grading needs one decode."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.outcome: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def decode(self, **kwargs: Any) -> Any:
        self.outcome = self.model.decode(**kwargs)
        return self.outcome


def score_composition(
    models: Mapping[str, Any], examples: Sequence[Any], directory: Path
) -> dict[str, Any]:
    """Each arm's observation and executed answer on every composition request."""
    from core.learning.semantic_graph_trial import _observe

    arms: dict[str, Any] = {}
    for arm, model in models.items():
        rows = []
        for item in examples:
            path = directory / arm / f"{item.ir.source_text_sha256}.json"
            if path.exists():
                rows.append(json.loads(path.read_text()))
                continue
            remembering = _Remembering(model)
            observation = _observe(remembering, item, solve_time_limit_s=20.0)
            outcome = remembering.outcome
            expected = item.ir.to_program().run(item.public_inputs)
            rows.append(
                _write_once(
                    path,
                    {
                        "source_text_sha256": item.ir.source_text_sha256,
                        "split": item.split,
                        "construction_id": item.construction_id,
                        "semantic_status": observation["semantic_status"],
                        "refusal": observation["refusal"],
                        "answer_correct": outcome.ir is not None
                        and _answer(outcome.ir, item.public_inputs, expected),
                    },
                )
            )
        arms[arm] = {
            "count": len(rows),
            "equivalent": sum(row["semantic_status"] == "equivalent" for row in rows),
            "answer_correct": sum(row["answer_correct"] for row in rows),
            "statuses": dict(Counter(row["semantic_status"] for row in rows)),
            "rows": {row["source_text_sha256"]: row for row in rows},
        }
    names = list(arms)
    if len(names) == 2:
        first, second = (arms[name]["rows"] for name in names)
        arms["paired_answers"] = {
            f"{names[1]}_only": sorted(
                s for s in first if second[s]["answer_correct"] and not first[s]["answer_correct"]
            ),
            f"{names[0]}_only": sorted(
                s for s in first if first[s]["answer_correct"] and not second[s]["answer_correct"]
            ),
        }
    return arms


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--incumbent",
        type=Path,
        default=_EVIDENCE
        / "semantic-literal-identity-pilot-20260921/incumbent_literal_identity.json",
    )
    parser.add_argument(
        "--incumbent-cohort",
        type=Path,
        default=_EVIDENCE / "semantic-literal-identity-cohort-20260921/report.json",
    )
    parser.add_argument(
        "--source-candidate",
        type=Path,
        default=_EVIDENCE / "semantic-source-fit-20260915/candidate.json",
    )
    parser.add_argument(
        "--source-report", type=Path, default=_EVIDENCE / "semantic-source-fit-20260915/report.json"
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=_EVIDENCE / "semantic-source-reacquisition-20260915/features",
    )
    parser.add_argument(
        "--folds",
        type=Path,
        default=_EVIDENCE / "semantic-architecture-source-folds-20260921/folds.json",
    )
    parser.add_argument(
        "--composition",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="an exposed composition feature bundle; repeatable",
    )
    parser.add_argument("--skip-folds", action="store_true")
    parser.add_argument(
        "--argument-ownership",
        action="store_true",
        help="also fit where-a-mention-stands ownership and add it to argument scores",
    )
    parser.add_argument(
        "--argument-antecedent",
        action="store_true",
        help="also fit where-each-name-was-given antecedents and add them to argument scores",
    )
    parser.add_argument(
        "--antecedent-scoring",
        choices=("absolute", "relative"),
        default="absolute",
        help="how an antecedent enters an argument's score (see semantic_argument_antecedent.py)",
    )
    parser.add_argument(
        "--own-result-is-not-an-input",
        action="store_true",
        help="keep a mention the antecedent says names an operation's own result out of its arguments",
    )
    parser.add_argument(
        "--named-inputs-are-used-by-name",
        action="store_true",
        help="do not offer an input's literal declaration when a mention names the input",
    )
    parser.add_argument(
        "--antecedent-fit",
        choices=("pairwise", "conditional"),
        default="pairwise",
        help="fit the antecedent pair by pair, or as each mention's distribution over registers",
    )
    parser.add_argument(
        "--antecedent-stretches",
        choices=("declaration", "sentence"),
        default="declaration",
        help="what text an operation's register owns: from its word, or its whole sentence",
    )
    parser.add_argument(
        "--arguments-within-sentence",
        action="store_true",
        help="no argument option starts before its operation's sentence (needs --antecedent-stretches sentence)",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("~/.aura/models/Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15/tokenizer.json"),
        help="the tokenizer whose sentence-ending tokens --antecedent-stretches sentence binds",
    )
    args = parser.parse_args()

    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )

    output = args.output.expanduser().absolute()
    configure_refit_environment(output / "report.json")

    from core.learning.semantic_argument_antecedent import fit_argument_antecedent
    from core.learning.semantic_argument_ownership import fit_argument_ownership
    from core.learning.semantic_cohort_diagnosis import audit_semantic_cohort
    from core.learning.semantic_operation_peaks import (
        PeakRecognitionTransducer,
        fit_peak_operation_recognizer,
    )
    from core.learning.semantic_program_campaign import training_examples_from_feature_bundle
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict as restore,
    )
    from core.learning.semantic_program_feature_materialization import (
        load_standard_semantic_feature_bundle,
    )

    incumbent = restore(json.loads(args.incumbent.read_text()))
    parent = restore(json.loads(args.source_candidate.read_text()))
    source_report = json.loads(args.source_report.read_text())
    families = source_report["representation_compatibility"]["source_feature_manifest_sha256s"]
    examples = load_source_examples(
        parent, source_report, [f"{name}={args.feature_root / name}" for name in families]
    )
    development = tuple(item for item in examples if item.split in ("train", "validation"))
    training = tuple(item for item in development if item.split == "train")
    print(
        f"development sources: {len(training)} train, {len(development) - len(training)} validation",
        flush=True,
    )

    sentence_ends: tuple[int, ...] = ()
    if args.antecedent_stretches == "sentence":
        from tokenizers import Tokenizer

        from core.learning.semantic_argument_antecedent import sentence_end_token_ids

        sentence_ends = sentence_end_token_ids(Tokenizer.from_file(str(args.tokenizer.expanduser())))
        print(f"sentence-ending tokens: {len(sentence_ends)}", flush=True)

    recognizer = fit_peak_operation_recognizer(training)
    ownership = fit_argument_ownership(training) if args.argument_ownership else None
    antecedent = (
        replace(fit_argument_antecedent(training, objective=args.antecedent_fit, sentence_end_token_ids=sentence_ends), scoring=args.antecedent_scoring,
                own_result_is_not_an_input=args.own_result_is_not_an_input,
                named_inputs_are_used_by_name=args.named_inputs_are_used_by_name,
                arguments_within_sentence=args.arguments_within_sentence)
        if args.argument_antecedent else None
    )
    candidate = PeakRecognitionTransducer(incumbent, recognizer, ownership, antecedent)
    _write_once(output / "recognizer.json", recognizer.to_dict())
    _write_once(output / "candidate.json", candidate.to_dict())

    def progress(event: Mapping[str, Any]) -> None:
        if "completed" in event and event["completed"] % 100 == 0 and "failure_stage" in event:
            print(f"  {event['completed']} of {event['total']}", flush=True)

    audit = audit_semantic_cohort(
        candidate,
        development,
        directory=output / "development",
        diagnose_failures=False,
        solve_time_limit_s=20.0,
        progress=progress,
    )
    incumbent_report = json.loads(args.incumbent_cohort.read_text())
    if incumbent_report["candidate"] != incumbent.receipt_sha256:
        raise ValueError("the incumbent cohort report is for a different transducer")
    comparison = paired_development_comparison(incumbent_report["rows"], audit["rows"])
    for split, entry in comparison.items():
        print(
            f"{split}: incumbent {entry['incumbent']}, peaks {entry['candidate']} of {entry['count']}"
            f" (+{entry['gains']} -{entry['losses']})",
            flush=True,
        )

    folds: dict[str, Any] = {}
    if not args.skip_folds:
        assignments = json.loads(args.folds.read_text())["assignments"]
        if any(item.ir.source_text_sha256 not in assignments for item in training):
            raise ValueError("the frozen folds do not assign every training source")
        for fold in sorted({assignments[item.ir.source_text_sha256] for item in training}):
            kept = tuple(
                item for item in training if assignments[item.ir.source_text_sha256] != fold
            )
            held = tuple(
                item for item in training if assignments[item.ir.source_text_sha256] == fold
            )
            fold_candidate = PeakRecognitionTransducer(
                incumbent,
                fit_peak_operation_recognizer(kept),
                fit_argument_ownership(kept) if args.argument_ownership else None,
                replace(fit_argument_antecedent(kept, objective=args.antecedent_fit, sentence_end_token_ids=sentence_ends), scoring=args.antecedent_scoring,
                        own_result_is_not_an_input=args.own_result_is_not_an_input,
                        named_inputs_are_used_by_name=args.named_inputs_are_used_by_name,
                arguments_within_sentence=args.arguments_within_sentence)
                if args.argument_antecedent else None,
            )
            fold_audit = audit_semantic_cohort(
                fold_candidate,
                held,
                directory=output / "folds" / str(fold),
                diagnose_failures=False,
                solve_time_limit_s=20.0,
            )
            folds[str(fold)] = {
                "fitted_on": len(kept),
                "held_out": len(held),
                "held_out_constructions": len({item.construction_id for item in held}),
                "constructions_shared_with_fit": len(
                    {item.construction_id for item in held}
                    & {item.construction_id for item in kept}
                ),
                "equivalent": sum(_equivalent(row) for row in fold_audit["rows"]),
                "audit_receipt_sha256": fold_audit["receipt_sha256"],
            }
            print(
                f"fold {fold}: {folds[str(fold)]['equivalent']} of {len(held)} on held-out constructions",
                flush=True,
            )

    composition: dict[str, Any] = {}
    source_manifest = load_standard_semantic_feature_bundle(
        args.feature_root / "natural_source"
    ).manifest
    for value in args.composition:
        name, separator, path = value.partition("=")
        if not separator:
            raise ValueError("--composition takes NAME=PATH")
        bundle = load_standard_semantic_feature_bundle(Path(path).expanduser())
        items = training_examples_from_feature_bundle(
            bundle, required_splits=frozenset({"validation", "test"})
        )
        # The bundle's own session basis is rebound to the incumbent's after
        # recording how its representation differs from the training sources'.
        differences = representation_differences(
            _representation(source_manifest), _representation(bundle.manifest)
        )
        items = tuple(
            replace(
                item, ir=replace(item.ir, model_basis_receipt_sha256=incumbent.model_basis_sha256)
            )
            for item in items
        )
        scored = score_composition(
            {"incumbent": incumbent, "peaks": candidate}, items, output / "composition" / name
        )
        composition[name] = {
            "manifest_sha256": bundle.manifest["manifest_sha256"],
            "representation_differences_from_training": differences,
            **{
                arm: {key: value for key, value in result.items() if key != "rows"}
                if arm != "paired_answers"
                else result
                for arm, result in scored.items()
            },
        }
        print(
            f"{name}: "
            + ", ".join(
                f"{arm} {scored[arm]['answer_correct']} answers" for arm in ("incumbent", "peaks")
            )
            + f" of {len(items)}",
            flush=True,
        )

    _write_once(
        output / "report.json",
        {
            "schema": SCHEMA,
            "incumbent": incumbent.receipt_sha256,
            "candidate": candidate.receipt_sha256,
            "recognizer": recognizer.identity_sha256,
            "recognizer_fit": dict(recognizer.fit_receipt),
            "argument_ownership": None if ownership is None else ownership.to_dict(),
            "argument_antecedent": None if antecedent is None else antecedent.to_dict(),
            "development_audit_receipt_sha256": audit["receipt_sha256"],
            "incumbent_cohort_receipt_sha256": incumbent_report["receipt_sha256"],
            "development": comparison,
            "construction_folds": folds,
            "composition": composition,
            "test_examples_used": 0,
            "serving_authority": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
