#!/usr/bin/env python3
"""Plan, launch, and verify detached candidate-cortex fusion."""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.governance_context import local_internal_governed_scope  # noqa: E402
from core.learning.candidate_cortex_fusion import (  # noqa: E402
    FUSION_PLAN_FILE,
    CandidateCortexFusionError,
    load_and_validate_fusion_plan,
    prepare_fusion_plan,
    publish_json,
)
from core.runtime.file_write_gateway import get_file_write_gateway  # noqa: E402
from tools import run_detached_step as detached  # noqa: E402

TARGET = REPO_ROOT / "tools" / "run_candidate_cortex_fusion_target.py"
VERIFIER = REPO_ROOT / "tools" / "verify_candidate_cortex_fusion_resume.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    plan = actions.add_parser("plan")
    plan.add_argument("--training-run", type=Path, required=True)
    plan.add_argument("--journal-key", type=Path, required=True)
    plan.add_argument("--fusion-root", type=Path, required=True)
    plan.add_argument(
        "--output-root",
        type=Path,
        default=Path.home() / ".aura" / "models",
    )
    plan.add_argument("--skip-full-model-verify", action="store_true")
    for name in ("verify", "launch", "status"):
        command = actions.add_parser(name)
        command.add_argument("--fusion-root", type=Path, required=True)
        if name != "status":
            command.add_argument("--journal-key", type=Path, required=True)
        if name == "launch":
            command.add_argument("--resume", action="store_true")
    return parser


def _ensure_directory(path: Path, *, source: str) -> Path:
    absolute = path.expanduser().resolve(strict=False)
    with local_internal_governed_scope(source, domain="file_write"):
        return Path(
            get_file_write_gateway().ensure_directory(
                absolute,
                source=source,
            )
        )


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    fusion_root = _ensure_directory(
        args.fusion_root,
        source="candidate_cortex_fusion.plan_root",
    )
    output_root = _ensure_directory(
        args.output_root,
        source="candidate_cortex_fusion.output_root",
    )
    plan = prepare_fusion_plan(
        run_root=args.training_run,
        journal_key_path=args.journal_key,
        fusion_root=fusion_root,
        output_root=output_root,
        target_source=TARGET,
        verifier_source=VERIFIER,
        verify_full_model=not args.skip_full_model_verify,
    )
    plan_path = fusion_root / FUSION_PLAN_FILE
    if plan_path.exists():
        current = json.loads(plan_path.read_text(encoding="utf-8"))
        if current != plan:
            raise CandidateCortexFusionError("existing_fusion_plan_mismatch")
    else:
        publish_json(
            plan_path,
            plan,
            source="candidate_cortex_fusion.plan",
        )
    return plan


def _load(fusion_root: Path, journal_key: Path, *, full: bool) -> dict[str, Any]:
    return load_and_validate_fusion_plan(
        fusion_root.expanduser().resolve(strict=True) / FUSION_PLAN_FILE,
        journal_key_path=journal_key,
        verify_full_model=full,
    )


def _target_command(plan: dict[str, Any], journal_key: Path) -> list[str]:
    return [
        str(plan["python"]),
        str(TARGET.resolve(strict=True)),
        "--plan",
        str(Path(str(plan["output"]["fusion_root"])) / FUSION_PLAN_FILE),
        "--journal-key",
        str(journal_key.expanduser().resolve(strict=True)),
    ]


def _verifier_command(plan: dict[str, Any], journal_key: Path) -> list[str]:
    return [
        str(plan["python"]),
        str(VERIFIER.resolve(strict=True)),
        "--plan",
        str(Path(str(plan["output"]["fusion_root"])) / FUSION_PLAN_FILE),
        "--journal-key",
        str(journal_key.expanduser().resolve(strict=True)),
    ]


def _detached_root(plan: dict[str, Any]) -> Path:
    return Path(str(plan["output"]["fusion_root"])) / "detached"


def _status_root(fusion_root: Path) -> dict[str, Any]:
    root = fusion_root.expanduser().resolve(strict=True) / "detached"
    if not root.exists():
        return {"state": "not_launched", "terminal": False}
    return detached._status(root)  # noqa: SLF001


def _launch(plan: dict[str, Any], journal_key: Path, *, resume: bool) -> dict[str, Any]:
    fusion_root = Path(str(plan["output"]["fusion_root"]))
    output_root = Path(str(plan["output"]["root"]))
    launch_args = [
        "launch",
        "--run-dir",
        str(_detached_root(plan)),
        "--name",
        f"candidate-cortex-fusion-{plan['output']['generation_id']}",
        "--cwd",
        str((Path.home() / ".aura").resolve(strict=True)),
        "--timeout",
        "7200",
        "--resume-contract",
        "target_checkpoint",
        "--resume-verifier-json",
        json.dumps(_verifier_command(plan, journal_key)),
        "--execution-output-root",
        str(fusion_root),
        "--execution-output-root",
        str(output_root),
    ]
    if resume:
        launch_args.append("--resume")
    launch_args.extend(("--", *_target_command(plan, journal_key)))
    captured = io.StringIO()
    with redirect_stdout(captured):
        return_code = detached.main(launch_args)
    if return_code != 0:
        raise CandidateCortexFusionError(
            f"fusion_detached_launch_failed:{captured.getvalue().strip()[:500]}"
        )
    return detached._status(_detached_root(plan))  # noqa: SLF001


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.action == "plan":
        result = _plan(args)
    elif args.action == "status":
        result = _status_root(args.fusion_root)
    else:
        plan = _load(args.fusion_root, args.journal_key, full=False)
        result = (
            plan
            if args.action == "verify"
            else _launch(plan, args.journal_key, resume=args.resume)
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
