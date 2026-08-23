#!/usr/bin/env python3
"""Plan and verify candidate-bound adaptive LoRA training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.candidate_cortex_training import (  # noqa: E402
    JOURNAL_FILE,
    CandidateCortexTrainingError,
    StagePolicy,
    TrainingConfig,
    execution_admission,
    load_and_verify_plan,
    next_stage_plan,
    prepare_training_run,
    read_authenticated_journal,
)


def _json_list(value: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("expected a JSON string array") from exc
    if not isinstance(parsed, list) or not parsed or any(not isinstance(item, str) for item in parsed):
        raise argparse.ArgumentTypeError("expected a non-empty JSON string array")
    return tuple(parsed)


def _key(path: Path) -> bytes:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise CandidateCortexTrainingError("journal_key_invalid")
    payload = resolved.read_bytes()
    if len(payload) < 32:
        raise CandidateCortexTrainingError("journal_key_too_short")
    return payload


def _config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        rank=args.rank,
        scale=args.scale,
        dropout=args.dropout,
        num_layers=args.num_layers,
        targets=tuple(args.targets),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        save_every=args.save_every,
        eval_every=args.eval_every,
        report_every=args.report_every,
        val_batches=args.val_batches,
        seed=args.seed,
    )


def _policy(args: argparse.Namespace) -> StagePolicy:
    return StagePolicy(
        initial_iterations=args.initial_iterations,
        growth_factor=args.growth_factor,
        max_stages=args.max_stages,
        min_stages=args.min_stages,
        patience=args.patience,
        min_loss_improvement=args.min_loss_improvement,
        max_loss_regression_fraction=args.max_loss_regression_fraction,
        persona_floor=args.persona_floor,
        retention_floor=args.retention_floor,
        no_regression_floor=args.no_regression_floor,
        min_eval_samples=args.min_eval_samples,
    )


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = TrainingConfig()
    stages = StagePolicy()
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--descriptor-sha256", required=True)
    parser.add_argument("--dataset-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--admission-command", type=_json_list, required=True)
    parser.add_argument("--rank", type=int, default=defaults.rank)
    parser.add_argument("--scale", type=float, default=defaults.scale)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--num-layers", type=int, default=defaults.num_layers)
    parser.add_argument("--targets", type=_json_list, default=defaults.targets)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=defaults.gradient_accumulation_steps,
    )
    parser.add_argument("--max-seq-length", type=int, default=defaults.max_seq_length)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--save-every", type=int, default=defaults.save_every)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--report-every", type=int, default=defaults.report_every)
    parser.add_argument("--val-batches", type=int, default=defaults.val_batches)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--initial-iterations", type=int, default=stages.initial_iterations)
    parser.add_argument("--growth-factor", type=int, default=stages.growth_factor)
    parser.add_argument("--max-stages", type=int, default=stages.max_stages)
    parser.add_argument("--min-stages", type=int, default=stages.min_stages)
    parser.add_argument("--patience", type=int, default=stages.patience)
    parser.add_argument(
        "--min-loss-improvement", type=float, default=stages.min_loss_improvement
    )
    parser.add_argument(
        "--max-loss-regression-fraction",
        type=float,
        default=stages.max_loss_regression_fraction,
    )
    parser.add_argument("--persona-floor", type=float, default=stages.persona_floor)
    parser.add_argument("--retention-floor", type=float, default=stages.retention_floor)
    parser.add_argument(
        "--no-regression-floor", type=float, default=stages.no_regression_floor
    )
    parser.add_argument("--min-eval-samples", type=int, default=stages.min_eval_samples)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    plan = commands.add_parser("plan", help="publish an immutable dry-run plan")
    _add_plan_arguments(plan)
    plan.add_argument("--execute", action="store_true")

    verify = commands.add_parser("verify", help="verify plan, inputs, identity, and journal")
    verify.add_argument("--run-root", type=Path, required=True)
    verify.add_argument("--journal-key", type=Path)

    resume = commands.add_parser("resume", help="emit the exact next bounded stage")
    resume.add_argument("--run-root", type=Path, required=True)
    resume.add_argument("--journal-key", type=Path, required=True)
    resume.add_argument("--execute", action="store_true")
    return parser


def _journal_evidence(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    admissions: list[dict[str, Any]] = []
    for event in events:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        if event.get("event_type") == "stage_observed":
            observations.append(payload)
        elif event.get("event_type") == "stage_admitted":
            admissions.append(payload)
    return observations, admissions


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "plan":
            plan = prepare_training_run(
                descriptor_path=args.descriptor,
                expected_descriptor_sha256=args.descriptor_sha256,
                dataset_receipt_path=args.dataset_receipt,
                output_root=args.output_root,
                python_executable=args.python,
                admission_command=args.admission_command,
                config=_config(args),
                policy=_policy(args),
                verify_full_model=True,
            )
            stage = next_stage_plan(plan, observations=[], admissions=[])
            execution = execution_admission(plan, execute=args.execute)
            payload = {"plan": plan, "next_stage": stage, "execution": execution}
        else:
            plan = load_and_verify_plan(args.run_root, verify_full_model=True)
            events = (
                read_authenticated_journal(
                    Path(str(plan["paths"]["run_root"])) / JOURNAL_FILE,
                    key=_key(args.journal_key),
                )
                if args.journal_key
                else []
            )
            observations, admissions = _journal_evidence(events)
            payload = {
                "plan_sha256": plan["plan_sha256"],
                "run_id": plan["run_id"],
                "journal_events": len(events),
                "next_stage": next_stage_plan(
                    plan,
                    observations=observations,
                    admissions=admissions,
                ),
            }
            if args.action == "resume":
                payload["execution"] = execution_admission(plan, execute=args.execute)
    except (
        CandidateCortexTrainingError,
        FileNotFoundError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
