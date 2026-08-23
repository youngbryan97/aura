#!/usr/bin/env python3
"""Plan and verify candidate-bound adaptive LoRA training."""

from __future__ import annotations

import argparse
import io
import json
import secrets
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.governance_context import local_internal_governed_scope  # noqa: E402
from core.learning.candidate_cortex_training import (  # noqa: E402
    JOURNAL_FILE,
    CanaryPolicy,
    CandidateCortexTrainingError,
    OptimizerConfig,
    StagePolicy,
    TrainingConfig,
    adjudicate_canary,
    execution_admission,
    effective_stage_evidence,
    load_and_verify_plan,
    next_stage_plan,
    prepare_training_run,
    read_authenticated_journal,
    stage_detached_root,
)
from core.runtime.file_write_gateway import get_file_write_gateway  # noqa: E402
from tools import run_detached_step as detached  # noqa: E402


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


def _optimizer(args: argparse.Namespace) -> OptimizerConfig:
    return OptimizerConfig(name=args.optimizer)


def _canary_policy(args: argparse.Namespace) -> CanaryPolicy:
    return CanaryPolicy(
        optimizer_steps=args.canary_optimizer_steps,
        validation_batches=args.canary_validation_batches,
        validation_interval_optimizer_steps=args.canary_validation_interval,
        timeout_seconds=args.canary_timeout_seconds,
        min_host_available_gb=args.canary_min_host_available_gb,
        max_peak_mlx_gb=args.canary_max_peak_mlx_gb,
        max_validation_loss_ratio=args.canary_max_validation_loss_ratio,
    )


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = TrainingConfig()
    stages = StagePolicy()
    canary = CanaryPolicy()
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
    parser.add_argument(
        "--optimizer",
        choices=("adafactor", "adam"),
        default=OptimizerConfig().name,
    )
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
    parser.add_argument(
        "--canary-optimizer-steps", type=int, default=canary.optimizer_steps
    )
    parser.add_argument(
        "--canary-validation-batches", type=int, default=canary.validation_batches
    )
    parser.add_argument(
        "--canary-validation-interval",
        type=int,
        default=canary.validation_interval_optimizer_steps,
    )
    parser.add_argument(
        "--canary-timeout-seconds", type=int, default=canary.timeout_seconds
    )
    parser.add_argument(
        "--canary-min-host-available-gb",
        type=float,
        default=canary.min_host_available_gb,
    )
    parser.add_argument(
        "--canary-max-peak-mlx-gb",
        type=float,
        default=canary.max_peak_mlx_gb,
    )
    parser.add_argument(
        "--canary-max-validation-loss-ratio",
        type=float,
        default=canary.max_validation_loss_ratio,
    )


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
    key = commands.add_parser("init-key", help="create one private journal key")
    key.add_argument("--path", type=Path, required=True)
    launch = commands.add_parser(
        "launch-canary", help="launch the exact canary under detached custody"
    )
    launch.add_argument("--run-root", type=Path, required=True)
    status = commands.add_parser("status-canary", help="inspect the detached canary")
    status.add_argument("--run-root", type=Path, required=True)
    adjudicate = commands.add_parser(
        "adjudicate-canary", help="verify and publish the terminal canary evidence"
    )
    adjudicate.add_argument("--run-root", type=Path, required=True)
    adjudicate.add_argument("--journal-key", type=Path, required=True)
    adaptive = commands.add_parser(
        "launch-adaptive",
        help="launch all admitted adaptive stages under resumable detached custody",
    )
    adaptive.add_argument("--run-root", type=Path, required=True)
    adaptive.add_argument("--journal-key", type=Path, required=True)
    adaptive.add_argument("--resume", action="store_true")
    adaptive.add_argument(
        "--execution-id",
        default="primary",
        help="immutable detached execution generation (default: primary)",
    )
    adaptive_status = commands.add_parser(
        "status-adaptive", help="inspect the detached adaptive campaign"
    )
    adaptive_status.add_argument("--run-root", type=Path, required=True)
    adaptive_status.add_argument("--execution-id", default="primary")
    return parser


def _target_command(plan: dict[str, Any]) -> list[str]:
    return [
        str(plan["python"]),
        str((REPO_ROOT / "tools" / "run_candidate_cortex_canary_target.py").resolve()),
        "--run-root",
        str(plan["paths"]["run_root"]),
    ]


def _plan_source_root(plan: dict[str, Any]) -> Path:
    inputs = plan.get("admission", {}).get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise CandidateCortexTrainingError("plan_source_binding_missing")
    roots: set[Path] = set()
    for raw in inputs:
        if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
            raise CandidateCortexTrainingError("plan_source_binding_invalid")
        path = Path(raw["path"]).expanduser().resolve(strict=True)
        if path.parent.name != "tools":
            raise CandidateCortexTrainingError("plan_source_binding_invalid")
        roots.add(path.parent.parent)
    if len(roots) != 1:
        raise CandidateCortexTrainingError("plan_source_binding_ambiguous")
    root = roots.pop()
    if not (root / ".git").exists():
        raise CandidateCortexTrainingError("plan_source_not_git_bound")
    return root


def _adaptive_target_command(
    plan: dict[str, Any], journal_key: Path, *, execution_id: str
) -> list[str]:
    return [
        str(plan["python"]),
        str((REPO_ROOT / "tools" / "run_candidate_cortex_adaptive_target.py").resolve()),
        "--run-root",
        str(plan["paths"]["run_root"]),
        "--journal-key",
        str(journal_key.expanduser().resolve(strict=True)),
        "--execution-id",
        execution_id,
    ]


def _adaptive_resume_verifier_command(
    plan: dict[str, Any], journal_key: Path
) -> list[str]:
    return [
        str(plan["python"]),
        str((REPO_ROOT / "tools" / "verify_candidate_cortex_adaptive_resume.py").resolve()),
        "--run-root",
        str(plan["paths"]["run_root"]),
        "--journal-key",
        str(journal_key.expanduser().resolve(strict=True)),
    ]


def _init_key(path: Path) -> dict[str, Any]:
    absolute = path.expanduser().resolve(strict=False)
    with local_internal_governed_scope(
        "candidate_cortex_training.init_key", domain="file_write"
    ):
        get_file_write_gateway().ensure_directory(
            absolute.parent,
            source="candidate_cortex_training.init_key",
        )
        created = get_file_write_gateway().write_bytes_if_absent(
            absolute,
            secrets.token_bytes(64),
            mode=0o600,
            source="candidate_cortex_training.init_key",
        )
    if not created:
        _key(absolute)
    return {"path": str(absolute), "created": created}


def _launch_canary(plan: dict[str, Any]) -> dict[str, Any]:
    execution = execution_admission(plan, execute=True)
    if execution.get("canary_launch_authorized") is not True:
        raise CandidateCortexTrainingError("canary_launch_not_authorized")
    policy = CanaryPolicy(**dict(plan["canary"]))
    launch_args = [
        "launch",
        "--run-dir",
        str(plan["paths"]["canary_detached_root"]),
        "--name",
        f"candidate-cortex-canary-{plan['run_id']}",
        "--cwd",
        str(REPO_ROOT),
        "--timeout",
        str(policy.timeout_seconds),
        "--execution-output-root",
        str(plan["paths"]["canary_execution_root"]),
        "--",
        *_target_command(plan),
    ]
    captured = io.StringIO()
    with redirect_stdout(captured):
        return_code = detached.main(launch_args)
    if return_code != 0:
        detail = captured.getvalue().strip()
        raise CandidateCortexTrainingError(f"canary_detached_launch_failed:{detail[:500]}")
    return detached._status(Path(str(plan["paths"]["canary_detached_root"])))


def _status_canary(plan: dict[str, Any]) -> dict[str, Any]:
    detached_root = Path(str(plan["paths"]["canary_detached_root"]))
    if not detached_root.exists():
        return {"state": "not_launched", "terminal": False}
    return detached._status(detached_root)


def _launch_adaptive(
    plan: dict[str, Any],
    journal_key: Path,
    *,
    resume: bool,
    execution_id: str = "primary",
) -> dict[str, Any]:
    key_path = journal_key.expanduser().resolve(strict=True)
    key = _key(key_path)
    events = read_authenticated_journal(
        Path(str(plan["paths"]["run_root"])) / JOURNAL_FILE,
        key=key,
    )
    execution = execution_admission(
        plan,
        execute=True,
        authenticated_events=events,
    )
    if execution.get("execution_authorized") is not True:
        raise CandidateCortexTrainingError("adaptive_launch_not_authorized")
    detached_root = stage_detached_root(plan, execution_id=execution_id)
    launch_args = [
        "launch",
        "--run-dir",
        str(detached_root),
        "--name",
        f"candidate-cortex-adaptive-{plan['run_id']}-{execution_id}",
        "--cwd",
        str(_plan_source_root(plan)),
        "--timeout",
        "86400",
        "--resume-contract",
        "target_checkpoint",
        "--resume-verifier-json",
        json.dumps(_adaptive_resume_verifier_command(plan, key_path)),
        "--execution-output-root",
        str(plan["paths"]["run_root"]),
    ]
    if resume:
        launch_args.append("--resume")
    launch_args.extend(
        ("--", *_adaptive_target_command(plan, key_path, execution_id=execution_id))
    )
    captured = io.StringIO()
    with redirect_stdout(captured):
        return_code = detached.main(launch_args)
    if return_code != 0:
        detail = captured.getvalue().strip()
        raise CandidateCortexTrainingError(
            f"adaptive_detached_launch_failed:{detail[:500]}"
        )
    return detached._status(detached_root)


def _status_adaptive(
    plan: dict[str, Any], *, execution_id: str = "primary"
) -> dict[str, Any]:
    detached_root = stage_detached_root(plan, execution_id=execution_id)
    if not detached_root.exists():
        return {"state": "not_launched", "terminal": False}
    return detached._status(detached_root)


def _adjudicate_canary(plan: dict[str, Any], journal_key: Path) -> dict[str, Any]:
    status = _status_canary(plan)
    if status.get("terminal") is not True or not isinstance(status.get("receipt"), dict):
        raise CandidateCortexTrainingError("canary_not_terminal")
    return adjudicate_canary(
        plan,
        detached_receipt=status["receipt"],
        expected_target_command=_target_command(plan),
        detached_log_path=Path(str(plan["paths"]["canary_detached_root"]))
        / detached.LOG_FILE,
        host_metrics_path=Path(str(plan["paths"]["canary_host_metrics"])),
        journal_key=_key(journal_key),
    )


def _journal_evidence(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return effective_stage_evidence(events)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "init-key":
            payload = _init_key(args.path)
        elif args.action == "plan":
            plan = prepare_training_run(
                descriptor_path=args.descriptor,
                expected_descriptor_sha256=args.descriptor_sha256,
                dataset_receipt_path=args.dataset_receipt,
                output_root=args.output_root,
                python_executable=args.python,
                admission_command=args.admission_command,
                config=_config(args),
                optimizer=_optimizer(args),
                policy=_policy(args),
                canary_policy=_canary_policy(args),
                verify_full_model=True,
            )
            stage = next_stage_plan(plan, observations=[], admissions=[])
            execution = execution_admission(plan, execute=args.execute)
            payload = {"plan": plan, "next_stage": stage, "execution": execution}
        else:
            plan = load_and_verify_plan(args.run_root, verify_full_model=True)
            if args.action == "launch-canary":
                payload = _launch_canary(plan)
            elif args.action == "status-canary":
                payload = _status_canary(plan)
            elif args.action == "adjudicate-canary":
                payload = _adjudicate_canary(plan, args.journal_key)
            elif args.action == "launch-adaptive":
                payload = _launch_adaptive(
                    plan,
                    args.journal_key,
                    resume=args.resume,
                    execution_id=args.execution_id,
                )
            elif args.action == "status-adaptive":
                payload = _status_adaptive(plan, execution_id=args.execution_id)
            else:
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
                    payload["execution"] = execution_admission(
                        plan,
                        execute=args.execute,
                        authenticated_events=events,
                    )
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
