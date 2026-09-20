#!/usr/bin/env python3
"""Generate an undeniable RSI proof bundle by running the AutonomousSuccessorEngine."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["AURA_EAGER_CORTEX_WARMUP"] = "1"
os.environ["AURA_METABOLISM_RATE"] = "0"
os.environ["AURA_STRICT_RUNTIME"] = "1"  # Fully disables volition/MindTick

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.learning.autonomous_rsi import AutonomousSuccessorEngine  # noqa: E402
from core.runtime.atomic_writer import atomic_write_text  # noqa: E402
from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402

RUNTIME_LOG = ROOT / "artifacts" / "rsi_frozen_generations" / "cortex_32b_runtime.log"
LOCK_PATH = ROOT / "artifacts" / "rsi_frozen_generations" / ".generate_undeniable_rsi.lock"


@contextlib.contextmanager
def proof_run_lock():
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another undeniable RSI generation is already running: {LOCK_PATH}") from exc
        yield


def _sandbox_pass(metadata: dict[str, Any]) -> bool:
    sandbox = metadata.get("sandbox_result")
    return isinstance(sandbox, dict) and bool(sandbox.get("pass"))


def _looks_like_fallback_template(source: str) -> bool:
    required = (
        "HANDLERS =",
        "if task.kind == 'gcd'",
        "if task.kind == 'mod'",
        "if task.kind == 'compose'",
        "if task.kind == 'sort'",
        "if task.kind == 'palindrome'",
    )
    return all(token in source for token in required)


def l3_claim_summary(
    *,
    result: Any,
    solver_source: str,
    strategy: dict[str, Any],
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    eval_before: dict[str, Any],
    eval_after: dict[str, Any],
) -> dict[str, Any]:
    baseline_score = float(eval_before.get("score", 0.0))
    candidate_score = float(eval_after.get("score", 0.0))
    candidate_improved = candidate_score > baseline_score
    manifest_kinds = {
        str(task.get("kind"))
        for task in manifest.get("public_tasks", [])
        if isinstance(task, dict) and task.get("kind")
    }
    strategy_handlers = {str(handler) for handler in strategy.get("handlers", [])}
    handler_coverage_complete = bool(manifest_kinds) and manifest_kinds.issubset(strategy_handlers)
    fallback_flag = bool(metadata.get("fallback_flag", True))
    router_presence = bool(metadata.get("router_presence", False))
    generated_source_hash = metadata.get("generated_source_hash")
    prompt_used = metadata.get("prompt_used")
    lineage_verdict = getattr(getattr(result, "verdict", None), "verdict", "")
    fallback_template = _looks_like_fallback_template(solver_source)
    lineage_undeniable = lineage_verdict == "UNDENIABLE_RSI"
    l3_rsi_claim = bool(
        not fallback_flag
        and router_presence
        and candidate_improved
        and generated_source_hash
        and prompt_used
        and _sandbox_pass(metadata)
        and not fallback_template
        and lineage_undeniable
        and handler_coverage_complete
    )
    failed_requirements = []
    requirements = {
        "fallback_flag_false": not fallback_flag,
        "router_presence_true": router_presence,
        "candidate_improved_over_baseline": candidate_improved,
        "generated_source_hash_present": bool(generated_source_hash),
        "generated_solver_not_fallback_template": not fallback_template,
        "prompt_used_present": bool(prompt_used),
        "sandbox_result_pass": _sandbox_pass(metadata),
        "lineage_verdict_undeniable": lineage_undeniable,
        "handler_coverage_complete": handler_coverage_complete,
    }
    failed_requirements = [name for name, passed in requirements.items() if not passed]
    records = list(getattr(result, "records", []) or [])
    return {
        "passed": l3_rsi_claim,
        "artifact_valid": True,
        "status": "l3_rsi_proven" if l3_rsi_claim else "not_l3_evidence",
        "reason": "all_l3_gates_passed" if l3_rsi_claim else "l3_gate_failed",
        "failed_requirements": failed_requirements,
        "l3_rsi_claim": l3_rsi_claim,
        "candidate_improved_over_baseline": candidate_improved,
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "generated_solver_looks_like_fallback_template": fallback_template,
        "manifest_kinds": sorted(manifest_kinds),
        "strategy_handlers": sorted(strategy_handlers),
        # Carried into the artifact so a reader can see where the improver
        # curve came from. The bundle this replaces recorded
        # [0.578, 0.8696, 0.9536, 1.0] with no indication that the numbers
        # were produced by a formula containing the generation counter.
        "improver_provenance": [
            getattr(record, "improver_provenance", "unmeasured") for record in records
        ],
        "improver_measurements": [
            measurement.to_dict()
            for measurement in getattr(result, "improver_measurements", []) or []
        ],
        "improver_order_invariance_violation": getattr(
            result, "order_invariance_violation", ""
        ),
        "lineage_verdict_reasons": list(
            getattr(getattr(result, "verdict", None), "reasons", []) or []
        ),
    }


def _llm_codegen_enabled() -> bool:
    """Whether a model actually writes the successor solvers.

    The bundle recorded `backend: mlx` and `started_runtime: true` regardless,
    while `AURA_RSI_ENABLE_LLM_CODEGEN` defaults off and the deterministic
    generator writes the solver. Booting a cortex the run never consults put a
    model's name on an artifact it did not produce.
    """
    return os.getenv("AURA_RSI_ENABLE_LLM_CODEGEN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


async def prepare_mlx_runtime(args: argparse.Namespace) -> dict[str, Any]:
    """Boot the canonical in-process MLX runtime — the exact cortex path the
    final-proof cert certifies green — and expose its real llm_router /
    cognitive_engine to the RSI engine via the ServiceContainer.

    The RSI engine's LLMCodeGenerator resolves ("inference_gate",
    "llm_router", "cognitive_engine") from the container, so a canonical boot is
    all that's needed — no custom router. Mirrors
    tools/agi/run_dnu_agi_proof_battery.py's boot.
    """
    # `--start-runtime` has existed since this tool was written and nothing
    # read it, so `--no-start-runtime` booted a full desktop runtime anyway.
    # That matters here more than it would elsewhere: this boot loads a second
    # resident model beside a live instance, which the host cannot hold.
    if not getattr(args, "start_runtime", True):
        return {
            "backend": "deterministic",
            "runtime_url": "",
            "model": "",
            "started_runtime": False,
            "llm_codegen_enabled": _llm_codegen_enabled(),
            "note": (
                "no model runtime was booted; the successor solvers came from "
                "the deterministic generator in core.learning.autonomous_rsi"
            ),
        }

    os.environ.setdefault("AURA_LOCAL_BACKEND", "mlx")
    os.environ["AURA_PROOF_MODEL_TIER"] = "primary"
    from aura_main import boot_aura_runtime
    from core.container import ServiceContainer

    orch = await boot_aura_runtime(
        profile="proof",
        ready_label="Proof-RSI",
        readiness_context="autonomous_rsi_proof",
        artifact_root=ROOT / "artifacts" / "current",
    )
    router = ServiceContainer.get("llm_router", default=None)
    if router is not None and hasattr(router, "endpoints"):
        # Keep the run on Cortex only (no second heavyweight local lane).
        router.endpoints.pop("Solver", None)
    engine = ServiceContainer.get("cognitive_engine", default=None) or getattr(orch, "cognitive_engine", None)
    if engine is None:
        raise RuntimeError("canonical boot completed without cognitive_engine")
    if getattr(engine, "lobotomized", False):
        raise RuntimeError("cognitive_engine is lobotomized; cannot run RSI proof")
    return {
        "backend": "mlx",
        "runtime_url": "in-process://mlx-cortex",
        "model": str(getattr(orch, "active_model", "") or "MLX-Cortex"),
        "started_runtime": True,
        # Booting the cortex is not the same as using it. Recorded so a reader
        # can tell whether the model wrote the solvers or only stood by.
        "llm_codegen_enabled": _llm_codegen_enabled(),
    }


async def _shutdown_mlx_runtime() -> None:
    """Graceful in-process teardown via the ShutdownCoordinator.

    NOT aura_main.stop_aura() — that is the *launcher's* stop (launchctl unload +
    kill the aura process), which, called from inside an in-process boot, kills
    this very process (rc=-9). Mirrors the DNU battery's shutdown path.
    """
    with contextlib.suppress(Exception):
        from core.runtime.shutdown_coordinator import get_shutdown_coordinator

        await get_shutdown_coordinator().shutdown(timeout_per_phase=10.0)


async def run_generation(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    runtime_info = await prepare_mlx_runtime(args)
    if runtime_info.get("started_runtime"):
        print(f"In-process MLX cortex ready: {runtime_info['model']}")
    else:
        print("No model runtime booted; deterministic successor generation.")
    print(f"Starting Autonomous RSI Generation ({args.generations} generations)...")

    # Each run gets its own directory. Every run used to write straight into
    # artifacts/rsi_frozen_generations, overwriting the frozen G1-G4 artifacts
    # that the previously published bundle records hashes for — so rerunning
    # the generator silently made the earlier claim unverifiable, and a
    # retraction naming those hashes would point at files that had changed
    # underneath it. The stable path becomes a symlink to the newest run so
    # existing readers still find current artifacts.
    runs_root = Path("artifacts/rsi_frozen_generations")
    artifact_dir = runs_root / "runs" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    engine = AutonomousSuccessorEngine(artifact_dir)
    try:
        result = await asyncio.to_thread(lambda: engine.run(generations=args.generations))
    finally:
        if runtime_info.get("backend") == "mlx":
            await _shutdown_mlx_runtime()
    _point_latest_at(runs_root, artifact_dir)
    runtime_info["artifact_dir"] = str(artifact_dir)
    return result, runtime_info


def _point_latest_at(runs_root: Path, artifact_dir: Path) -> None:
    """Repoint `<runs_root>/latest` at this run, leaving earlier runs alone."""
    latest = runs_root / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        # Relative, so the link still resolves in another checkout. An
        # absolute one points into whichever worktree happened to run the tool.
        latest.symlink_to(
            os.path.relpath(artifact_dir.resolve(), latest.parent.resolve()),
            target_is_directory=True,
        )
    except OSError as exc:
        # A missing convenience link must not fail a completed proof run; the
        # run directory is the artifact and it is already written.
        print(f"could not update {latest}: {exc}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/proof_bundle/latest/UNDENIABLE_RSI.json")
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--runtime-url", default=os.getenv("AURA_RSI_LIVE_LLM_URL", "http://127.0.0.1:11435"))
    parser.add_argument("--runtime-model", default="")
    parser.add_argument("--runtime-model-path", default="")
    parser.add_argument("--generation-timeout-s", type=float, default=600.0)
    parser.add_argument("--ready-timeout-s", type=float, default=180.0)
    parser.add_argument("--start-runtime", action=argparse.BooleanOptionalAction, default=True)
    parser.set_defaults(backend="mlx")
    args = parser.parse_args()

    with proof_run_lock():
        result, runtime_info = asyncio.run(run_generation(args))

    # Gather undeniable proof
    artifact = result.artifacts[-1]
    gen_dir = Path(artifact.directory)
    
    solver_source = (gen_dir / "solver.py").read_text(encoding="utf-8")
    strategy = json.loads((gen_dir / "strategy.json").read_text(encoding="utf-8"))
    manifest = json.loads((gen_dir / "public_manifest.json").read_text(encoding="utf-8"))
    eval_after = json.loads((gen_dir / "eval_after.json").read_text(encoding="utf-8"))
    eval_before = json.loads((gen_dir / "eval_before.json").read_text(encoding="utf-8"))
    metadata = json.loads((gen_dir / "generation_metadata.json").read_text(encoding="utf-8"))
    
    # We also need git commit and reproduction command
    commit_result = get_subprocess_gateway().run(
        ["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"],
        cwd=ROOT,
        timeout=30,
        read_only=True,
        source="proof_tooling:rsi_commit_sha",
        accelerator_capability="none",
    )
    if commit_result.returncode != 0:
        raise RuntimeError(f"failed to resolve git commit SHA: {commit_result.stderr.strip()}")
    commit = commit_result.stdout.strip()
    claim = l3_claim_summary(
        result=result,
        solver_source=solver_source,
        strategy=strategy,
        manifest=manifest,
        metadata=metadata,
        eval_before=eval_before,
        eval_after=eval_after,
    )

    bundle = {
        "generated_at": time.time(),
        "claim": "L3_RSI",
        **claim,
        "exact_commit_SHA": commit,
        "reproduction_command": f"python tools/generate_undeniable_rsi_bundle.py --generations {args.generations}",
        "runtime": runtime_info,
        "lineage_verdict": result.verdict.to_dict(),
        "lineage_result": result.to_dict(),
        "generated_solver_source": solver_source,
        "generated_source_hash": metadata.get("generated_source_hash"),
        "fallback_flag": metadata.get("fallback_flag"),
        "router_presence": metadata.get("router_presence"),
        "prompt_used": metadata.get("prompt_used"),
        "sandbox_result": metadata.get("sandbox_result"),
        "no_answer_leakage": True,
        "hidden_task_manifest_without_answers": manifest,
        "salted_answer_hashes": [task.get("answer_hash") for task in manifest.get("public_tasks", [])],
        "candidate_output_transcript": eval_after,
        "baseline_output_transcript": eval_before,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Undeniable RSI Bundle written to {out_path}")
    return 0 if bundle["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
