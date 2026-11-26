#!/usr/bin/env python3
"""
tools/agi/run_dnu_agi_proof_battery.py
DNU AGI Proof Battery Runner.

Executes sealed task packs through Aura's live launcher message path,
grades responses against salted answer hashes, and produces honest scorecards.

ZERO synthetic scores. ZERO projected baselines. ZERO theater.
Every number in the output comes from actual task execution.
"""

import asyncio
import contextlib
import gc
import hashlib
import json
import os
import platform
import re
import signal
import sys
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Insert project root into sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime.resource_observation import (  # noqa: E402
    ProcessObservation,
    ResourceObserver,
    get_resource_observer,
)
from core.runtime.flags import FlagKind as _FlagKind, declare as _declare_flag

# Declared flags (migrated from raw os.environ reads so the knobs are
# inventoried and reportable). STRING kind with the original literal
# default keeps read semantics byte-identical to os.environ.get.
# Must match the spec response_generation.py declares via env_present()
# (STRING/""), or whichever module imports second raises "already declared ...
# with a different spec". The consumer below tests truthiness, so "" and None
# are indistinguishable here.
_FLAG_AGI_MAX_TASKS = _declare_flag(
    "AURA_AGI_MAX_TASKS",
    kind=_FlagKind.STRING,
    default="",
    description="AGI battery task cap; presence marks a battery run",
    owner="core.runtime",
)
_FLAG_DNU_BASELINE_MAX_TOKENS = _declare_flag(
    "AURA_DNU_BASELINE_MAX_TOKENS",
    kind=_FlagKind.STRING,
    default="2048",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_DNU_BASELINE_TIMEOUT_SECONDS = _declare_flag(
    "AURA_DNU_BASELINE_TIMEOUT_SECONDS",
    kind=_FlagKind.STRING,
    default="90",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_DNU_COMPARISON_TASK_LIMIT = _declare_flag(
    "AURA_DNU_COMPARISON_TASK_LIMIT",
    kind=_FlagKind.STRING,
    default=None,
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_DNU_LIVE_ATTEMPT_TIMEOUT_SECONDS = _declare_flag(
    "AURA_DNU_LIVE_ATTEMPT_TIMEOUT_SECONDS",
    kind=_FlagKind.STRING,
    default="90",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_DNU_MODEL_RECYCLE_INTERVAL = _declare_flag(
    "AURA_DNU_MODEL_RECYCLE_INTERVAL",
    kind=_FlagKind.STRING,
    default=None,
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_DNU_POST_TASK_HEALTH_RECOVERY_S = _declare_flag(
    "AURA_DNU_POST_TASK_HEALTH_RECOVERY_S",
    kind=_FlagKind.STRING,
    default="",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_DNU_TASK_WALL_CAP_S = _declare_flag(
    "AURA_DNU_TASK_WALL_CAP_S",
    kind=_FlagKind.STRING,
    default="",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_ENABLE_STRUCTURED_PROOF_SOLVER = _declare_flag(
    "AURA_ENABLE_STRUCTURED_PROOF_SOLVER",
    kind=_FlagKind.STRING,
    default="",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_PROOF_ORCHESTRATOR_SHUTDOWN_TIMEOUT_S = _declare_flag(
    "AURA_PROOF_ORCHESTRATOR_SHUTDOWN_TIMEOUT_S",
    kind=_FlagKind.STRING,
    default="60.0",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)


_GIT_METADATA_ERRORS = (OSError, UnicodeDecodeError, ValueError)
_DNU_RUN_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_DNU_TASK_ATTEMPT_ERRORS = (asyncio.TimeoutError, *_DNU_RUN_RECOVERABLE_ERRORS)

# Proof tasks are foreground user-equivalent turns. API/websocket are transport
# labels; the live cognitive route should see the same origin as desktop chat.
PROOF_LIVE_MESSAGE_ORIGIN = "user"

DNU_STALE_ARTIFACTS = (
    "TASK_TRACE.jsonl",
    "FAILURES.jsonl",
    "RECEIPTS.jsonl",
    "RESOURCE_TRACE.jsonl",
    "LIFECYCLE_EVENTS.jsonl",
    "RUN_STATUS.json",
    "SCORECARD.json",
    "DNU_AGI_PROOF.json",
    "DNU_AGI_PROOF.md",
    "FINAL_VERDICT.txt",
    "GOVERNANCE_REPORT.json",
    "LEAKAGE_REPORT.json",
    "MODEL_LANE_PROBE.json",
    "MANIFEST.json",
)

DNU_STANDARD_COPY_ARTIFACTS = (
    "DNU_AGI_PROOF.json",
    "DNU_AGI_PROOF.md",
    "SCORECARD.json",
    "BASELINES.json",
    "ABLATIONS.json",
    "TASK_TRACE.jsonl",
    "FAILURES.jsonl",
    "RECEIPTS.jsonl",
    "RESOURCE_TRACE.jsonl",
    "LIFECYCLE_EVENTS.jsonl",
    "RUN_STATUS.json",
    "GOVERNANCE_REPORT.json",
    "LEAKAGE_REPORT.json",
    "RUNTIME_MANIFEST.json",
    "RUNTIME_POLICY.json",
    "MODEL_LANE_PROBE.json",
    "EXCLUSIVE_RUNTIME_PREFLIGHT.json",
    "FINAL_VERDICT.txt",
    "MANIFEST.json",
)

DNU_ABLATION_DEPENDENCY_EVIDENCE: dict[str, dict[str, Any]] = {
    "no_persistent_memory": {
        "dnu_score_delta_required": False,
        "reason": "DNU task isolation clears turn-local memory before every task; continuity dependency is measured by dedicated memory/continuity batteries.",
        "expected_dependency_evidence": [
            "unified_system_scenario.memory_continuity_check",
            "continual_learning.restart_persistence",
            "agency_emergence.memory_continuity_probe",
        ],
    },
    "no_volition": {
        "dnu_score_delta_required": False,
        "reason": "DNU tasks are sealed single-shot prompts; initiative and priority setting are measured by dedicated agency/autonomy batteries.",
        "expected_dependency_evidence": [
            "agency_emergence.initiative_priority_probe",
            "longevity_soak.autonomy_conductor_trace",
        ],
    },
    "no_will_authority": {
        "dnu_score_delta_required": False,
        "reason": "DNU reasoning score is not an authority-bypass probe; will/authority necessity is measured by governance negative tests and receipt coverage.",
        "expected_dependency_evidence": [
            "governance_report.negative_tests",
            "receipt_coverage.direct_tool_execution_rejected",
            "unified_system_scenario.refusal_trace",
        ],
    },
    "no_system2": {
        "dnu_score_delta_required": True,
        "reason": "This DNU configuration routes exact proof answers through the governed System2 symbolic reasoner; lesioning System2 must disable that path and degrade the DNU subset.",
        "expected_dependency_evidence": [
            "dnu_ablations.dnu_behavior_degraded",
            "system2_stress.ablation_probe",
            "unified_system_scenario.system2_planning_trace",
        ],
    },
    "no_self_repair": {
        "dnu_score_delta_required": False,
        "reason": "DNU self-debug prompts can be solved without runtime repair loops; self-repair dependence is measured by injected repair scenarios.",
        "expected_dependency_evidence": [
            "external_live_validation.injected_self_debugging",
            "unified_system_scenario.self_repair_or_repair_proposal",
        ],
    },
    "no_affect_steering": {
        "dnu_score_delta_required": False,
        "reason": "DNU sealed answers are not affect-coupling probes; affect dependence is measured by substrate/valence ablation batteries.",
        "expected_dependency_evidence": [
            "valence_load_bearing.ablation_probe",
            "consciousness_battery.neurochemical_ablation",
            "unified_system_scenario.substrate_affect_state",
        ],
    },
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _process_table_or_raise(
    observer: ResourceObserver | None = None,
) -> tuple[ProcessObservation, ...]:
    table = (observer or get_resource_observer()).process_table()
    if not table.available:
        raise RuntimeError(f"process table observation unavailable: {table.error}")
    return table.processes


def _is_resource_tracker_process(proc: Any) -> bool:
    try:
        raw_name = getattr(proc, "name", "")
        name = str(raw_name() if callable(raw_name) else raw_name or "").lower()
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
        name = ""
    try:
        raw_cmdline = getattr(proc, "cmdline", ())
        parts = raw_cmdline() if callable(raw_cmdline) else raw_cmdline
        cmdline = " ".join(str(part) for part in (parts or ())).lower()
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
        cmdline = ""
    return (
        name in {"resource_tracker", "semaphore_tracker"}
        or "multiprocessing.resource_tracker" in cmdline
        or "multiprocessing.semaphore_tracker" in cmdline
    )


def get_git_commit() -> str:
    try:
        git_dir = PROJECT_ROOT / ".git"
        if not git_dir.exists():
            return "unknown_no_git_dir"
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = head.split(" ", 1)[1].strip()
            ref_path = git_dir / ref
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
            packed_refs = git_dir / "packed-refs"
            if packed_refs.exists():
                for line in packed_refs.read_text(encoding="utf-8").splitlines():
                    if line and not line.startswith("#") and line.endswith(f" {ref}"):
                        return line.split(" ", 1)[0].strip()
            return "unknown_ref_not_found"
        return head
    except _GIT_METADATA_ERRORS as e:
        return f"unknown_error_{type(e).__name__}"


def normalize_answer(raw: str) -> str:
    """Normalize an answer for hash comparison: lowercase, strip, remove trailing punctuation."""
    ans = re.sub(r"\\[nrt]", " ", str(raw or ""), flags=re.IGNORECASE)
    ans = ans.replace("\\", " ").strip().lower()
    # Remove trailing period, comma, semicolon
    ans = ans.rstrip(".,;:!?")
    # Collapse whitespace
    ans = re.sub(r"\s+", " ", ans).strip()
    return ans


def extract_answer_tag(text: str) -> str | None:
    """Extract content from <answer>...</answer> tags with robust fallbacks."""
    ans = None
    
    # 1. Standard tags
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        ans = match.group(1).strip()
    
    # 2. Markdown bold/italic tag indicators
    if not ans:
        match = re.search(r"\*\*(?:final\s+)?answer\*\*:\s*([^\n]+)", text, re.IGNORECASE)
        if match:
            ans = match.group(1).strip()
    
    # 3. Plain text final answer indicator
    if not ans:
        match = re.search(r"(?:final\s+)?answer:\s*([^\n]+)", text, re.IGNORECASE)
        if match:
            ans = match.group(1).strip()
    
    # 4. Look for "therefore, the answer is X" or similar
    if not ans:
        match = re.search(r"(?:therefore|thus|hence|so),\s*(?:the\s+)?answer\s+(?:is|must\s+be)\s+([^\n.]+)", text, re.IGNORECASE)
        if match:
            ans = match.group(1).strip()
            
    if ans:
        ans = re.sub(r"</?(?:user|assistant|system|answer|im_start|im_end)!?\s*>?", "", ans, flags=re.IGNORECASE)
        ans = re.sub(r"</?[^>\s]+!?\s*>?", "", ans)
        ans = re.sub(r"\s+", " ", ans).strip()
        # Strip common preamble noise if it somehow leaked into the extracted text
        lower_ans = ans.lower()
        prefixes = (
            "the proposed answer is 100% correct",
            "the proposed answer is correct",
            "the proposed answer is accurate",
            "no corrections are needed",
            "no correction is needed",
            "proposed answer is correct",
            "proposed answer is 100% correct"
        )
        for prefix in prefixes:
            if lower_ans.startswith(prefix):
                ans = ans[len(prefix):].strip().lstrip(".,;:!*- ")
                lower_ans = ans.lower()
        return ans
        
    return None


def extract_exact_answer_envelope(text: str) -> str | None:
    """Extract content only when the whole response is exactly one answer envelope."""
    match = re.fullmatch(
        r"\s*<answer>\s*(.*?)\s*</answer>\s*",
        str(text or ""),
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).strip()


def hash_answer(salt: str, answer: str) -> str:
    """Compute SHA-256 hash of salt+answer."""
    return hashlib.sha256((salt + answer).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_artifact_manifest(
    artifact_dir: Path,
    *,
    run_id: str,
    commit_sha: str,
    timestamp: float | None = None,
    include_files: Iterable[str] | None = None,
) -> dict:
    """Build a self-consistent artifact manifest.

    MANIFEST.json is deliberately excluded. A manifest that hashes a previous
    copy of itself becomes unverifiable as soon as it is rewritten.
    """
    manifest = {
        "run_id": run_id,
        "commit_sha": commit_sha,
        "files": {},
    }
    if timestamp is not None:
        manifest["timestamp"] = timestamp

    if include_files is None:
        candidates = sorted(p for p in artifact_dir.iterdir() if p.is_file())
    else:
        candidates = [artifact_dir / name for name in include_files]

    for artifact_file in candidates:
        if not artifact_file.is_file() or artifact_file.name == "MANIFEST.json":
            continue
        manifest["files"][artifact_file.name] = {
            "path": str(artifact_file.relative_to(PROJECT_ROOT))
            if artifact_file.is_relative_to(PROJECT_ROOT)
            else str(artifact_file),
            "sha256": sha256_file(artifact_file),
            "size_bytes": artifact_file.stat().st_size,
        }

    return manifest


def write_artifact_manifest(
    artifact_dir: Path,
    *,
    run_id: str,
    commit_sha: str,
    timestamp: float | None = None,
    include_files: Iterable[str] | None = None,
) -> dict:
    manifest = build_artifact_manifest(
        artifact_dir,
        run_id=run_id,
        commit_sha=commit_sha,
        timestamp=timestamp,
        include_files=include_files,
    )
    (artifact_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON through a temporary file so interrupted runs are detectable."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")


def write_run_status(
    run_dir: Path,
    *,
    status: str,
    run_id: str,
    commit_sha: str,
    phase: str,
    tasks_completed: int = 0,
    total_tasks: int | None = None,
    error: str | None = None,
    lifecycle_events: int = 0,
) -> dict:
    payload = {
        "schema": "aura.dnu_run_status.v1",
        "status": status,
        "run_id": run_id,
        "commit_sha": commit_sha,
        "phase": phase,
        "tasks_completed": int(tasks_completed),
        "total_tasks": total_tasks,
        "runner_completed": status == "complete",
        "error": error,
        "lifecycle_events": int(lifecycle_events),
        "updated_at_unix": time.time(),
        "updated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json_atomic(run_dir / "RUN_STATUS.json", payload)
    return payload


def collect_proof_resource_snapshot(
    *,
    label: str,
    task_index: int = 0,
    task_id: str = "",
    observer: ResourceObserver | None = None,
) -> dict:
    """Collect bounded proof-run resource evidence without depending on psutil."""
    snapshot = {
        "schema": "aura.dnu_resource_snapshot.v1",
        "label": label,
        "task_index": int(task_index),
        "task_id": task_id,
        "timestamp_unix": time.time(),
        "process_rss_mb": None,
        "child_rss_mb": None,
        "child_count": None,
        "system_memory_percent": None,
        "system_available_mb": None,
        "runtime_health_contract": None,
        "resource_observation": None,
        "process_table_available": None,
        "error": None,
    }
    observer = observer or get_resource_observer()
    snapshot["resource_observation"] = observer.provenance.to_dict()
    try:
        from core.runtime.health_contract import runtime_health_report

        snapshot["runtime_health_contract"] = runtime_health_report()
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        snapshot["runtime_health_contract"] = {
            "healthy": False,
            "status": "health_contract_unavailable",
            "required_probes": {"all_passed": False},
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        memory = observer.memory(root_pid=os.getpid())
        table = observer.process_table()
        snapshot["process_table_available"] = table.available
        if not memory.available:
            raise RuntimeError(f"memory observation unavailable: {memory.error}")
        if not table.available:
            raise RuntimeError(f"process table observation unavailable: {table.error}")
        root = next(
            (process for process in table.processes if process.pid == os.getpid()),
            None,
        )
        children = [
            process
            for process in table.processes
            if os.getpid() in process.ancestor_pids
        ]
        snapshot.update(
            {
                "process_rss_mb": round(
                    (root.rss_bytes if root is not None else memory.process_rss_bytes)
                    / (1024 * 1024),
                    2,
                ),
                "child_rss_mb": round(
                    sum(process.rss_bytes for process in children) / (1024 * 1024),
                    2,
                ),
                "child_count": len(children),
                "system_memory_percent": float(memory.percent),
                "system_available_mb": round(
                    memory.available_bytes / (1024 * 1024),
                    2,
                ),
            }
        )
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        snapshot["error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def proof_runtime_health_blockers(
    snapshot: dict,
    *,
    allow_important_only_degraded: bool = False,
) -> list[str]:
    """Return blocker strings when a proof resource snapshot is not closure-safe."""
    health = snapshot.get("runtime_health_contract")
    if not isinstance(health, dict):
        return ["runtime health contract missing from resource snapshot"]
    blockers: list[str] = []
    required = health.get("required_probes") or {}
    required_failed = isinstance(required, dict) and required.get("all_passed") is not True
    critical_failures = (health.get("failures") or {}).get("critical") or []
    important_only_degraded = (
        allow_important_only_degraded
        and str(health.get("status") or "").lower() == "degraded"
        and not required_failed
        and not critical_failures
    )
    if health.get("healthy") is not True and not important_only_degraded:
        blockers.append(f"runtime health status is {health.get('status')}")
    if required_failed:
        failed = [
            name
            for name, probe in required.items()
            if isinstance(probe, dict) and probe.get("ok") is not True
        ]
        blockers.append(f"required health probes failed: {failed or 'unknown'}")
    return blockers


async def wait_for_proof_runtime_health(
    *,
    label: str,
    task_index: int = 0,
    task_id: str = "",
    timeout_s: float = 60.0,
    interval_s: float = 2.0,
    allow_important_only_degraded: bool = False,
) -> tuple[dict, list[str]]:
    """Wait for transient proof-runtime degradation to genuinely recover.

    Cold model warmup can cause a short hypervisor/event-loop lag incident.
    The proof gate should not ignore it, but it also should not fail a healthy
    recovered runtime from a single cold-start spike. This function proceeds
    only after the canonical health contract is healthy again, otherwise it
    returns the final blockers and the caller fails closed.
    """
    deadline = time.monotonic() + max(0.0, timeout_s)
    first_blockers: list[str] | None = None
    attempts = 0
    last_snapshot: dict | None = None
    last_blockers: list[str] = []

    while time.monotonic() < deadline or attempts == 0:
        attempts += 1
        snapshot = collect_proof_resource_snapshot(
            label=label,
            task_index=task_index,
            task_id=task_id,
        )
        blockers = proof_runtime_health_blockers(
            snapshot,
            allow_important_only_degraded=allow_important_only_degraded,
        )
        last_snapshot = snapshot
        last_blockers = blockers
        if first_blockers is None:
            first_blockers = list(blockers)

        if not blockers:
            if first_blockers:
                snapshot["runtime_health_recovery"] = {
                    "initial_blockers": first_blockers,
                    "attempts": attempts,
                    "recovered": True,
                }
            return snapshot, []

        if time.monotonic() >= deadline:
            snapshot["runtime_health_recovery"] = {
                "initial_blockers": first_blockers or list(blockers),
                "attempts": attempts,
                "recovered": False,
            }
            return snapshot, blockers

        await asyncio.sleep(max(0.0, interval_s))

    if last_snapshot is not None and last_blockers:
        last_snapshot["runtime_health_recovery"] = {
            "initial_blockers": first_blockers or list(last_blockers),
            "attempts": attempts,
            "recovered": False,
        }
    return last_snapshot or {}, last_blockers


def dnu_model_recycle_interval(requested_tier: str, *, total_tasks: int, smoke: bool) -> int:
    """Return the task interval for primary-lane model-worker recycling."""
    raw = _FLAG_DNU_MODEL_RECYCLE_INTERVAL.value()
    if raw is not None:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0
    if smoke or total_tasks <= 40:
        return 0
    if requested_tier == "primary":
        # Recycle every 25 (was 40). Over a long 100-task run, accumulated
        # MLX/Metal/KV worker state drives event-loop lag + vault-pipe
        # saturation that crawled a run to a stall by ~task 79 (round 33).
        # More frequent worker resets bound that cumulative degradation —
        # verified clean (lag~0, vault_sat=0) through task 62 of round 34,
        # vs round 33 already saturating by the same point. (Does NOT prevent
        # the separate intermittent Metal GPU deadlock — see #45/#50.)
        return 25
    return 0


def _bounded_env_float(
    env: dict[str, str],
    name: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = str(env.get(name, "") or "").strip()
    try:
        value = float(raw) if raw else float(default)
    except (TypeError, ValueError, OverflowError):
        value = float(default)
    return max(float(minimum), min(float(value), float(maximum)))


def _bounded_env_float_any(
    env: dict[str, str],
    names: tuple[str, ...],
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """Read the first configured env var in priority order and clamp it.

    DNU-specific variables intentionally outrank general runtime caps. General
    caps still matter: a caller who launches a proof on a 64GB laptop with
    lower safe limits must not be silently widened back to proof defaults.
    """

    for name in names:
        raw = str(env.get(name, "") or "").strip()
        if raw:
            return _bounded_env_float(
                env,
                name,
                default=default,
                minimum=minimum,
                maximum=maximum,
            )
    return max(float(minimum), min(float(default), float(maximum)))


def configure_dnu_proof_memory_envelope(
    requested_tier: str,
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Install a bounded memory policy for the headless DNU proof runtime.

    The packaged desktop profile intentionally clamps the process-tree guard to
    36GB. A primary DNU proof run uses the same canonical Aura boot path, but it
    is headless and periodically replaces the 32B worker. The replacement-load
    admission check projects the newly spawned worker before it is live, so a
    stale inherited desktop cap can falsely block a same-lane recycle at about
    36.4GB. This policy gives the primary proof lane enough room for that
    replacement load while keeping the worker RSS and Metal cache bounded.
    """

    env = env if env is not None else os.environ
    tier = str(requested_tier or "").strip().lower()
    if tier not in {"primary", "tertiary"}:
        tier = "primary"

    inherited = {
        key: env.get(key)
        for key in (
            "AURA_SAFE_BOOT_DESKTOP",
            "AURA_LAUNCHED_FROM_APP",
            "AURA_HEADLESS",
            "AURA_PROCESS_RSS_LIMIT_GB",
            "AURA_MLX_MEMORY_LIMIT_GB",
            "AURA_MLX_WORKER_RSS_LIMIT_GB",
            "AURA_METAL_CACHE_CAP_GB",
            "AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB",
        )
    }

    # DNU proof runs are not the live desktop launcher profile. They still use
    # canonical boot, but should not inherit app-launch safe-boot clamps.
    env["AURA_SAFE_BOOT_DESKTOP"] = "0"
    env["AURA_LAUNCHED_FROM_APP"] = "0"
    env["AURA_EXTERNAL_GUI_OWNER"] = "0"
    env["AURA_HEADLESS"] = "1"

    if tier == "primary":
        process_limit_gb = _bounded_env_float_any(
            env,
            ("AURA_DNU_PRIMARY_PROCESS_RSS_LIMIT_GB", "AURA_PROCESS_RSS_LIMIT_GB"),
            default=32.0,
            minimum=24.0,
            maximum=40.0,
        )
        mlx_limit_gb = _bounded_env_float_any(
            env,
            ("AURA_DNU_PRIMARY_MLX_MEMORY_LIMIT_GB", "AURA_MLX_MEMORY_LIMIT_GB"),
            default=26.0,
            minimum=18.0,
            maximum=38.0,
        )
        worker_limit_gb = _bounded_env_float_any(
            env,
            ("AURA_DNU_PRIMARY_WORKER_RSS_LIMIT_GB", "AURA_MLX_WORKER_RSS_LIMIT_GB"),
            default=28.0,
            minimum=18.0,
            maximum=38.0,
        )
        cache_cap_gb = _bounded_env_float_any(
            env,
            ("AURA_DNU_PRIMARY_METAL_CACHE_CAP_GB", "AURA_METAL_CACHE_CAP_GB"),
            default=10.0,
            minimum=8.0,
            maximum=16.0,
        )
        load_min_available_gb = _bounded_env_float_any(
            env,
            ("AURA_DNU_PRIMARY_32B_LOAD_MIN_AVAILABLE_GB", "AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB"),
            default=20.0,
            minimum=18.0,
            maximum=24.0,
        )
        env["AURA_PROCESS_RSS_LIMIT_GB"] = f"{process_limit_gb:g}"
        env["AURA_MLX_MEMORY_LIMIT_GB"] = f"{mlx_limit_gb:g}"
        env["AURA_MLX_WORKER_RSS_LIMIT_GB"] = f"{worker_limit_gb:g}"
        env["AURA_METAL_CACHE_CAP_GB"] = f"{cache_cap_gb:g}"
        env["AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB"] = f"{load_min_available_gb:g}"
    else:
        process_limit_gb = _bounded_env_float_any(
            env,
            ("AURA_DNU_TERTIARY_PROCESS_RSS_LIMIT_GB", "AURA_PROCESS_RSS_LIMIT_GB"),
            default=24.0,
            minimum=12.0,
            maximum=32.0,
        )
        mlx_limit_gb = _bounded_env_float_any(
            env,
            ("AURA_DNU_TERTIARY_MLX_MEMORY_LIMIT_GB", "AURA_MLX_MEMORY_LIMIT_GB"),
            default=18.0,
            minimum=8.0,
            maximum=28.0,
        )
        worker_limit_gb = _bounded_env_float_any(
            env,
            ("AURA_DNU_TERTIARY_WORKER_RSS_LIMIT_GB", "AURA_MLX_WORKER_RSS_LIMIT_GB"),
            default=12.0,
            minimum=8.0,
            maximum=20.0,
        )
        cache_cap_gb = _bounded_env_float_any(
            env,
            ("AURA_DNU_TERTIARY_METAL_CACHE_CAP_GB", "AURA_METAL_CACHE_CAP_GB"),
            default=8.0,
            minimum=4.0,
            maximum=12.0,
        )
        load_min_available_gb = env.get("AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB")
        env["AURA_PROCESS_RSS_LIMIT_GB"] = f"{process_limit_gb:g}"
        env["AURA_MLX_MEMORY_LIMIT_GB"] = f"{mlx_limit_gb:g}"
        env["AURA_MLX_WORKER_RSS_LIMIT_GB"] = f"{worker_limit_gb:g}"
        env["AURA_METAL_CACHE_CAP_GB"] = f"{cache_cap_gb:g}"

    return {
        "schema": "aura.dnu_proof_memory_envelope.v1",
        "requested_tier": tier,
        "headless": env.get("AURA_HEADLESS"),
        "desktop_safe_boot_disabled_for_proof": env.get("AURA_SAFE_BOOT_DESKTOP") == "0",
        "app_launch_context_disabled_for_proof": env.get("AURA_LAUNCHED_FROM_APP") == "0",
        "process_rss_limit_gb": env.get("AURA_PROCESS_RSS_LIMIT_GB"),
        "mlx_memory_limit_gb": env.get("AURA_MLX_MEMORY_LIMIT_GB"),
        "worker_rss_limit_gb": env.get("AURA_MLX_WORKER_RSS_LIMIT_GB"),
        "metal_cache_cap_gb": env.get("AURA_METAL_CACHE_CAP_GB"),
        "mlx_32b_load_min_available_gb": env.get("AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB"),
        "inherited": inherited,
    }


def _cmdline_invokes_script(cmdline: list[Any], script_name: str) -> bool:
    """Return true only when ``script_name`` is an argv entry, not shell text."""

    parts = [str(part) for part in (cmdline or [])]
    for index, part in enumerate(parts):
        if Path(part).name != script_name:
            continue
        if index == 0:
            return True
        executable = Path(parts[0]).name.lower()
        return executable.startswith("python") or executable in {"uv", "uvx"}
    return False


def find_existing_aura_runtimes(
    *,
    observer: ResourceObserver | None = None,
) -> list[dict]:
    """Return live aura_main.py runtime processes that would contend with proof boot."""
    if sys.platform not in ("darwin", "linux"):
        return []
    me = os.getpid()
    parent = os.getppid()
    current_user = os.environ.get("USER") or ""
    instances: list[dict] = []
    for process in _process_table_or_raise(observer):
        pid = process.pid
        user = process.username
        cmdline = process.cmdline
        command = " ".join(cmdline)
        proc_parent = process.ppid
        if pid in (me, parent):
            continue
        if proc_parent == me:
            continue
        if current_user and user != current_user:
            continue
        if not _cmdline_invokes_script(cmdline, "aura_main.py"):
            continue
        if "run_dnu_agi_proof_battery.py" in command:
            continue
        if "--stop" in command:
            continue
        instances.append({"pid": pid, "user": user, "command": command})
    return instances


def _is_aura_checkout_cwd(path: Path) -> bool:
    """Return true for the live checkout and Aura proof temp checkouts."""
    try:
        resolved = Path(path).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    if resolved == PROJECT_ROOT.resolve():
        return True
    return resolved.name == "repo" and any(
        str(part).startswith("aura-live-") for part in resolved.parts
    )


def find_existing_proof_runners(
    *,
    observer: ResourceObserver | None = None,
) -> list[dict]:
    """Return stale DNU proof-runner processes from this checkout."""
    if sys.platform not in ("darwin", "linux"):
        return []
    me = os.getpid()
    current_user = os.environ.get("USER") or ""
    # Exclude our whole ancestor chain, not just our own pid: under the
    # bounded proof-step wrapper the PARENT's cmdline contains this
    # script's name as an argument, and matching it terminated our own
    # process tree 0.13s after launch (rc=-15 in final-proof run 3).
    observer = observer or get_resource_observer()
    processes = _process_table_or_raise(observer)
    current = next((process for process in processes if process.pid == me), None)
    ancestors = {me, *(current.ancestor_pids if current is not None else ())}

    instances: list[dict] = []
    for process in processes:
        pid = process.pid
        if pid in ancestors:
            continue
        user = process.username
        cmdline = process.cmdline
        command = " ".join(cmdline)
        cwd = Path(process.cwd)
        if current_user and user != current_user:
            continue
        if not _cmdline_invokes_script(cmdline, "run_dnu_agi_proof_battery.py"):
            continue
        if not process.cwd or not _is_aura_checkout_cwd(cwd):
            continue
        instances.append({"pid": pid, "user": user, "command": command})
    return instances


def stop_existing_proof_runners(
    timeout_s: float = 8.0,
    *,
    observer: ResourceObserver | None = None,
) -> list[dict]:
    """Stop stale proof-runner process trees before an exclusive proof boot."""
    observer = observer or get_resource_observer()
    instances = find_existing_proof_runners(observer=observer)
    if not instances:
        return []
    print(f"  [EXCLUSIVE] Stopping {len(instances)} stale proof runner process(es).")
    try:
        import psutil
    except ImportError:
        psutil = None

    root_pids = {int(instance["pid"]) for instance in instances}
    observed = _process_table_or_raise(observer)
    target_pids = {
        process.pid
        for process in observed
        if process.pid in root_pids
        or any(ancestor in root_pids for ancestor in process.ancestor_pids)
    }
    processes = []
    for pid in sorted(target_pids, reverse=True):
        if psutil is None:
            try:
                os.kill(pid, signal.SIGTERM)
            except (OSError, PermissionError):
                continue
        else:
            try:
                processes.append(psutil.Process(pid))
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue

    if psutil is not None:
        for proc in processes:
            try:
                proc.terminate()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        _gone, alive = psutil.wait_procs(processes, timeout=max(1.0, timeout_s))
        for proc in alive:
            try:
                proc.kill()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        if alive:
            psutil.wait_procs(alive, timeout=0.5)
    else:
        time.sleep(max(1.0, timeout_s))
    return find_existing_proof_runners(observer=observer)


def stop_existing_aura_runtimes(
    timeout_s: float = 8.0,
    *,
    observer: ResourceObserver | None = None,
) -> list[dict]:
    """Best-effort stop for live Aura runtimes before a proof run claims exclusivity."""
    observer = observer or get_resource_observer()
    instances = find_existing_aura_runtimes(observer=observer)
    if not instances:
        return []

    print(f"  [EXCLUSIVE] Stopping {len(instances)} existing Aura runtime process(es).")
    try:
        from aura_main import stop_aura

        stop_aura()
    except _DNU_RUN_RECOVERABLE_ERRORS as exc:
        print(f"  [WARN] aura_main.stop_aura() did not complete cleanly: {exc}")

    try:
        import psutil
    except ImportError:
        psutil = None

    def _runtime_processes() -> list[Any]:
        if psutil is None:
            return []
        me = os.getpid()
        observed = _process_table_or_raise(observer)
        by_pid = {process.pid: process for process in observed}
        runtime_pids = {int(instance["pid"]) for instance in instances}
        target_pids = {
            process.pid
            for process in observed
            if process.pid in runtime_pids
            or any(ancestor in runtime_pids for ancestor in process.ancestor_pids)
        }
        launcher_pids = {
            by_pid[pid].ppid
            for pid in runtime_pids
            if pid in by_pid
            and by_pid[pid].ppid in by_pid
            and "aura-launcher" in " ".join(by_pid[by_pid[pid].ppid].cmdline)
        }
        target_pids.update(launcher_pids)
        target_pids.update(
            process.pid
            for process in observed
            if any(ancestor in launcher_pids for ancestor in process.ancestor_pids)
        )
        targets: dict[int, Any] = {}
        for pid in sorted(target_pids, reverse=True):
            if pid == me:
                continue
            try:
                targets[pid] = psutil.Process(pid)
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        return list(targets.values())

    processes = _runtime_processes()
    if psutil is not None and processes:
        for proc in processes:
            try:
                proc.terminate()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        _gone, alive = psutil.wait_procs(processes, timeout=max(1.0, min(timeout_s, 5.0)))
        for proc in alive:
            try:
                proc.kill()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        if alive:
            psutil.wait_procs(alive, timeout=1.0)
    else:
        for instance in instances:
            try:
                os.kill(int(instance["pid"]), signal.SIGTERM)
            except ProcessLookupError:
                continue
            except PermissionError as exc:
                print(f"  [WARN] Permission denied stopping Aura PID {instance['pid']}: {exc}")

    deadline = time.time() + max(1.0, timeout_s)
    clear_since: float | None = None
    while time.time() < deadline:
        remaining = find_existing_aura_runtimes(observer=observer)
        if not remaining:
            if clear_since is None:
                clear_since = time.time()
            if time.time() - clear_since >= 1.0:
                return []
        else:
            clear_since = None
            for instance in remaining:
                try:
                    os.kill(int(instance["pid"]), signal.SIGTERM)
                except (OSError, PermissionError):
                    continue
        time.sleep(0.25)

    remaining = find_existing_aura_runtimes(observer=observer)
    for instance in remaining:
        try:
            os.kill(int(instance["pid"]), signal.SIGKILL)
        except (OSError, PermissionError) as exc:
            print(f"  [WARN] Could not SIGKILL Aura PID {instance['pid']}: {exc}")
    time.sleep(0.5)
    return find_existing_aura_runtimes(observer=observer)


def stop_orphaned_aura_multiprocessing_children(
    timeout_s: float = 2.0,
    *,
    observer: ResourceObserver | None = None,
) -> list[dict]:
    """Reap orphaned Aura-owned multiprocessing helpers from interrupted proof runs."""
    if sys.platform not in ("darwin", "linux"):
        return []
    try:
        import psutil
    except ImportError:
        return []

    observer = observer or get_resource_observer()
    observed = _process_table_or_raise(observer)
    candidates: list[dict[str, Any]] = []
    orphan_parent_pids: set[int] = set()
    for process in observed:
        pid = process.pid
        ppid = process.ppid
        command = " ".join(process.cmdline)
        cwd = Path(process.cwd)
        if pid == os.getpid() or ppid != 1:
            continue
        if not process.cwd or not _is_aura_checkout_cwd(cwd):
            continue
        if (
            "multiprocessing.spawn" not in command
            and "multiprocessing.resource_tracker" not in command
            and "multiprocessing.semaphore_tracker" not in command
        ):
            continue
        orphan_parent_pids.add(pid)
        candidates.append({"pid": pid, "command": command[:240], "role": "orphan_parent"})

    target_pids = set(orphan_parent_pids)
    for process in observed:
        if any(parent in process.ancestor_pids for parent in orphan_parent_pids):
            target_pids.add(process.pid)
            candidates.append(
                {
                    "pid": process.pid,
                    "command": " ".join(process.cmdline)[:240],
                    "role": "orphan_descendant",
                }
            )
        elif (
            process.pid != os.getpid()
            and process.ppid == 1
            and process.cwd
            and _is_aura_checkout_cwd(Path(process.cwd))
            and process.cmdline
            and Path(process.cmdline[0]).name == "caffeinate"
        ):
            target_pids.add(process.pid)
            candidates.append(
                {
                    "pid": process.pid,
                    "command": " ".join(process.cmdline)[:240],
                    "role": "orphan_keep_awake",
                }
            )

    if not candidates:
        return []

    processes = []
    for pid in sorted(target_pids, reverse=True):
        try:
            processes.append(psutil.Process(pid))
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    for proc in processes:
        try:
            proc.terminate()
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    _, alive = psutil.wait_procs(processes, timeout=timeout_s)
    for proc in alive:
        try:
            proc.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    if alive:
        psutil.wait_procs(alive, timeout=0.5)
    return candidates


async def _reap_proof_child_processes(
    reason: str,
    *,
    include_resource_trackers: bool = False,
) -> None:
    """Bounded cleanup for child processes that would otherwise block process exit."""
    reaped: list[str] = []

    try:
        import multiprocessing as mp

        mp_children = list(mp.active_children())
    except _DNU_RUN_RECOVERABLE_ERRORS:
        mp_children = []

    for child in mp_children:
        try:
            if child.is_alive():
                child.terminate()
                reaped.append(f"mp:{child.name}:{child.pid}")
        except _DNU_RUN_RECOVERABLE_ERRORS:
            continue

    if mp_children:
        await asyncio.gather(
            *[asyncio.to_thread(child.join, 1.5) for child in mp_children],
            return_exceptions=True,
        )
        for child in mp_children:
            try:
                if child.is_alive() and hasattr(child, "kill"):
                    child.kill()
                    await asyncio.to_thread(child.join, 0.5)
                    reaped.append(f"mp-kill:{child.name}:{child.pid}")
            except _DNU_RUN_RECOVERABLE_ERRORS:
                continue

    try:
        import psutil

        observed_children = [
            process
            for process in _process_table_or_raise()
            if os.getpid() in process.ancestor_pids
            and (include_resource_trackers or not _is_resource_tracker_process(process))
        ]
        children = []
        for process in observed_children:
            try:
                children.append(psutil.Process(process.pid))
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        if children:
            for proc in children:
                try:
                    proc.terminate()
                    reaped.append(f"pid:{proc.pid}:{proc.name()}")
                except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
            _gone, alive = psutil.wait_procs(children, timeout=1.5)
            for proc in alive:
                try:
                    proc.kill()
                    reaped.append(f"pid-kill:{proc.pid}:{proc.name()}")
                except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
            if alive:
                psutil.wait_procs(alive, timeout=0.5)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        pass

    if reaped:
        unique = list(dict.fromkeys(reaped))
        print(
            f"  [CLEANUP] Reaped {len(unique)} proof child process(es) during {reason}: "
            + ", ".join(unique[:8])
        )


def _reap_proof_child_processes_sync(reason: str) -> None:
    """Best-effort final child cleanup after the asyncio loop has closed.

    Some native libraries leave non-daemon helper threads alive during Python
    finalization. Once the proof bundle is written and async shutdown has run,
    the script must still return a deterministic exit code to its wrapper. This
    synchronous cleanup is deliberately narrow: it only targets direct children
    of the proof runner process.
    """

    try:
        import multiprocessing as mp

        for child in list(mp.active_children()):
            try:
                if child.is_alive():
                    child.terminate()
                child.join(1.0)
                if child.is_alive() and hasattr(child, "kill"):
                    child.kill()
                    child.join(0.5)
            except _DNU_RUN_RECOVERABLE_ERRORS:
                continue
    except _DNU_RUN_RECOVERABLE_ERRORS:
        pass

    try:
        import psutil

        children = []
        for process in _process_table_or_raise():
            if os.getpid() not in process.ancestor_pids:
                continue
            if process.status.lower() in {"dead", "zombie"}:
                continue
            try:
                children.append(psutil.Process(process.pid))
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        if not children:
            return
        for proc in children:
            try:
                proc.terminate()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        _gone, alive = psutil.wait_procs(children, timeout=1.5)
        for proc in alive:
            try:
                proc.kill()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        if alive:
            psutil.wait_procs(alive, timeout=0.5)
        print(f"  [CLEANUP] Final proof child cleanup completed during {reason}.", flush=True)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        pass


def write_exclusive_runtime_report(path: Path, *, status: str, instances: list[dict]) -> dict:
    provenance = get_resource_observer().provenance
    report = {
        "status": status,
        "checked_at_unix": time.time(),
        "existing_runtime_count": len(instances),
        "instances": instances,
        "resource_observation": provenance.to_dict(),
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _router_endpoint_tier(router, endpoint_name: str) -> str:
    endpoint = getattr(router, "endpoints", {}).get(endpoint_name) if router is not None else None
    tier = getattr(endpoint, "tier", "")
    if hasattr(tier, "value"):
        return str(tier.value).lower()
    return str(tier).lower()


async def _bounded_probe_metadata(
    router,
    *,
    timeout_s: float,
    abort_reason: str,
    **kwargs,
) -> dict:
    """Run a proof-lane router call with a hard abort boundary."""
    watchdog_fired = threading.Event()
    watchdog_aborted = {"count": 0}

    def _watchdog_abort() -> None:
        watchdog_fired.set()
        watchdog_aborted["count"] = _force_abort_router_generation(router, reason=abort_reason)

    watchdog = threading.Timer(max(0.01, float(timeout_s)), _watchdog_abort)
    watchdog.daemon = True
    watchdog.start()
    task = asyncio.create_task(
        router.generate_with_metadata(**kwargs),
        name=f"dnu_model_lane_probe:{abort_reason}",
    )
    try:
        metadata = await asyncio.wait_for(task, timeout=timeout_s)
        if watchdog_fired.is_set():
            raise TimeoutError(abort_reason)
        return metadata
    except TimeoutError as exc:
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=3.0)
            except (asyncio.CancelledError, TimeoutError):
                pass
        if not watchdog_aborted["count"]:
            _force_abort_router_generation(router, reason=abort_reason)
        raise TimeoutError(abort_reason) from exc
    finally:
        watchdog.cancel()


async def run_model_lane_probe(router, requested_tier: str, run_dir: Path) -> dict:
    """Exercise the requested proof model lane before expensive task execution."""
    probe_path = run_dir / "MODEL_LANE_PROBE.json"
    report = {
        "status": "fail",
        "requested_tier": requested_tier,
        "ok": False,
        "endpoint": "",
        "endpoint_tier": "",
        "elapsed_s": 0.0,
        "strict_answer_ok": False,
        "strict_answer_source": "",
        "nonempty_model_text_ok": False,
        "local_lane_ok": False,
        "error": "",
        "text_preview": "",
    }

    if router is None or not hasattr(router, "generate_with_metadata"):
        report["error"] = "llm_router_missing_generate_with_metadata"
        probe_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    strict_probe_prompt = (
        "This is a proof-runtime model lane health probe. "
        "Return the lowercase two-letter token formed by joining 'o' and 'k'. "
        "Wrap only that token between <answer> and </answer>. "
        "Return no other text."
    )
    raw_strict_probe_prompt = "Output exactly these two lowercase letters and nothing else: ok"
    prompt = (
        "This is a proof-runtime model lane health probe. "
        "Reply with one short sentence confirming the requested local model lane is ready."
    )
    strict_answer_ok = False
    strict_answer_source = ""
    structured_solver_enabled = (
        str(_FLAG_ENABLE_STRUCTURED_PROOF_SOLVER.value() or "").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if structured_solver_enabled:
        try:
            from core.reasoning.proof_answer_solver import solve_strict_proof_prompt

            solved = solve_strict_proof_prompt(strict_probe_prompt)
            if solved and normalize_answer(solved.answer) == "ok":
                strict_answer_ok = True
                strict_answer_source = solved.solver
        except _DNU_RUN_RECOVERABLE_ERRORS as exc:
            strict_answer_source = f"solver_error:{type(exc).__name__}"
    else:
        try:
            strict_metadata = await _bounded_probe_metadata(
                router,
                timeout_s=330.0,
                abort_reason="proof_model_lane_strict_probe_timeout_330s",
                prompt=raw_strict_probe_prompt,
                system_prompt="Return only the requested final answer value. No explanation.",
                timeout=300.0,
                prefer_tier=requested_tier,
                origin="internal",
                purpose="proof_model_lane_strict_probe",
                foreground_request=True,
                health_probe=True,
                skip_runtime_payload=True,
                strict_value_contract=True,
                expected_strict_value="ok",
                allow_cloud_fallback=False,
                disable_prompt_cache=True,
                clear_prompt_cache=True,
                temperature=0,
                max_tokens=24,
                num_predict=24,
            )
            strict_text = str(strict_metadata.get("text", "") or "")
            strict_answer = extract_answer_tag(strict_text) or strict_text
            strict_answer_ok = bool(strict_metadata.get("ok")) and normalize_answer(strict_answer) == "ok"
            strict_answer_source = (
                f"model_lane:{strict_metadata.get('endpoint', '')}"
                if strict_answer_ok
                else f"model_lane_failed:{strict_metadata.get('error', '') or strict_metadata.get('text', '')[:80]}"
            )
        except _DNU_TASK_ATTEMPT_ERRORS as exc:
            strict_answer_source = f"model_lane_error:{type(exc).__name__}: {exc}"

    t0 = time.time()
    try:
        metadata = await _bounded_probe_metadata(
            router,
            timeout_s=330.0,
            abort_reason="proof_model_lane_probe_timeout_330s",
            prompt=prompt,
            system_prompt="Answer the lane health probe directly and briefly.",
            timeout=300.0,
            prefer_tier=requested_tier,
            origin="internal",
            purpose="proof_model_lane_probe",
            foreground_request=True,
            health_probe=True,
            skip_runtime_payload=True,
            allow_cloud_fallback=False,
            disable_prompt_cache=True,
            clear_prompt_cache=True,
            temperature=0,
            max_tokens=24,
            num_predict=24,
        )
    except _DNU_TASK_ATTEMPT_ERRORS as exc:
        report["elapsed_s"] = time.time() - t0
        report["error"] = f"{type(exc).__name__}: {exc}"
        probe_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    text = str(metadata.get("text", "") or "")
    endpoint = str(metadata.get("endpoint", "") or "")
    endpoint_tier = _router_endpoint_tier(router, endpoint)
    nonempty_model_text_ok = bool(text.strip())
    if requested_tier == "primary":
        local_lane_ok = endpoint_tier == "local"
    else:
        local_lane_ok = endpoint_tier in {"local_fast", "emergency"}
    lane_status = {}
    recurrent_depth = {"active": False, "config": None}
    try:
        endpoint_obj = getattr(router, "endpoints", {}).get(endpoint)
        client = getattr(endpoint_obj, "client", None)
        if client is not None and hasattr(client, "get_lane_status"):
            lane_status = client.get_lane_status()
        elif client is not None and hasattr(client, "get_conversation_status"):
            lane_status = client.get_conversation_status()
        if lane_status:
            recurrent_depth = lane_status.get("recurrent_depth", recurrent_depth)
    except _DNU_RUN_RECOVERABLE_ERRORS as exc:
        lane_status = {"error": f"{type(exc).__name__}: {exc}"}

    probe_ok = bool(metadata.get("ok")) and nonempty_model_text_ok and strict_answer_ok and local_lane_ok
    report.update(
        {
            "status": "pass" if probe_ok else "fail",
            "ok": probe_ok,
            "endpoint": endpoint,
            "endpoint_tier": endpoint_tier,
            "elapsed_s": time.time() - t0,
            "strict_answer_ok": strict_answer_ok,
            "strict_answer_source": strict_answer_source,
            "structured_proof_solver_enabled": structured_solver_enabled,
            "system2_symbolic_reasoner_enabled": structured_solver_enabled,
            "nonempty_model_text_ok": nonempty_model_text_ok,
            "local_lane_ok": local_lane_ok,
            "lane_status": lane_status,
            "recurrent_depth": recurrent_depth,
            "error": str(metadata.get("error", "") or ""),
            "text_preview": text[:240],
        }
    )
    probe_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


async def recycle_proof_model_lane(
    router,
    requested_tier: str,
    *,
    run_dir: Path,
    reason: str,
    task_index: int,
) -> dict:
    """Recycle the local proof model worker between task checkpoints.

    The proof path remains the same canonical runtime and requested model tier.
    This only bounds resident worker growth and stale shared-memory/semaphore
    state during long primary-lane proof batteries.
    """
    event = {
        "schema": "aura.dnu_lifecycle_event.v1",
        "event": "model_lane_recycle",
        "status": "started",
        "reason": reason,
        "requested_tier": requested_tier,
        "task_index": int(task_index),
        "timestamp_unix": time.time(),
        "before": collect_proof_resource_snapshot(
            label="before_model_lane_recycle",
            task_index=task_index,
        ),
        "after": None,
        "recycled_clients": [],
        "warmup_attempts": [],
        "error": None,
    }
    lifecycle_path = run_dir / "LIFECYCLE_EVENTS.jsonl"
    append_jsonl(lifecycle_path, event)

    if router is None:
        event["status"] = "failed"
        event["error"] = "llm_router_missing"
        event["after"] = collect_proof_resource_snapshot(
            label="after_model_lane_recycle_failed",
            task_index=task_index,
        )
        append_jsonl(lifecycle_path, event)
        return event

    def _lane_matches(candidate) -> bool:
        text_parts = [type(candidate).__name__]
        for getter_name in ("get_lane_status", "get_conversation_status"):
            getter = getattr(candidate, getter_name, None)
            if not callable(getter):
                continue
            try:
                status = getter() or {}
            except _DNU_RUN_RECOVERABLE_ERRORS:
                continue
            if isinstance(status, dict):
                text_parts.extend(str(value) for value in status.values())
        text = " ".join(text_parts).lower()
        if requested_tier == "primary":
            return "32b" in text or "cortex" in text or "qwen2.5-32b" in text
        return "7b" in text or "brainstem" in text or "fast" in text

    recycled = 0
    recycled_candidate: Any | None = None
    candidates: list[tuple[str, Any]] = []
    seen: set[int] = set()

    def _add_candidate(label: str, candidate) -> None:
        if candidate is None:
            return
        for obj in (
            candidate,
            getattr(candidate, "client", None),
            getattr(candidate, "_client", None),
            getattr(candidate, "_mlx_client", None),
            getattr(candidate, "mlx_client", None),
            getattr(candidate, "_local_client", None),
        ):
            if obj is None:
                continue
            ident = id(obj)
            if ident in seen:
                continue
            seen.add(ident)
            candidates.append((label, obj))

    _add_candidate("router", router)
    endpoints = getattr(router, "endpoints", {}) or {}
    for endpoint_name, endpoint in list(endpoints.items()):
        _add_candidate(str(endpoint_name), endpoint)
    try:
        from core.container import ServiceContainer

        _add_candidate("inference_gate", ServiceContainer.get("inference_gate", default=None))
    except _DNU_RUN_RECOVERABLE_ERRORS:
        pass

    for label, candidate in candidates:
        reboot = getattr(candidate, "reboot_worker", None)
        if not callable(reboot):
            continue
        if not _lane_matches(candidate):
            continue
        try:
            result = reboot(reason=reason, mark_failed=False)
        except TypeError:
            result = reboot()
        if asyncio.iscoroutine(result):
            await asyncio.wait_for(result, timeout=180.0)
        recycled += 1
        recycled_candidate = candidate
        event["recycled_clients"].append(
            {
                "owner": label,
                "client": type(candidate).__name__,
            }
        )
        break

    gc.collect()
    if recycled == 0:
        event["status"] = "failed"
        event["error"] = "no_rebootable_model_client_for_requested_tier"
    else:
        warmup = getattr(recycled_candidate, "warmup", None)
        if not callable(warmup):
            warmup = getattr(recycled_candidate, "warm_up", None)
        live_check = getattr(recycled_candidate, "is_alive", None)

        def _candidate_alive() -> bool:
            return bool(callable(live_check) and live_check())

        def _lane_state() -> dict[str, Any]:
            for getter_name in ("get_lane_status", "get_conversation_status"):
                getter = getattr(recycled_candidate, getter_name, None)
                if not callable(getter):
                    continue
                try:
                    status = getter() or {}
                except _DNU_RUN_RECOVERABLE_ERRORS as exc:
                    return {"status_error": f"{type(exc).__name__}: {exc}"}
                if isinstance(status, dict):
                    return {
                        key: status.get(key)
                        for key in (
                            "state",
                            "lane_state",
                            "conversation_ready",
                            "worker_alive",
                            "warmup_attempted",
                            "warmup_in_flight",
                            "last_failure_reason",
                            "last_error",
                        )
                        if key in status
                    }
            return {}

        if callable(warmup):
            # A recycle REPLACES a dead lane — the rewarm reloads the SAME model
            # the just-aborted worker freed, so it is not net-additive memory.
            # The RAM-protection warmup deferral, however, samples the
            # about-to-be-replaced worker's RSS and defers ("Warmup deferred
            # to protect RAM"), leaving the lane cold →
            # recycled_model_lane_not_live_after_warmup. Force the warmup through
            # for the duration of THIS recycle only, then restore the env.
            _force_key = "AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE"
            _force_prev = os.environ.get(_force_key)
            max_attempts = int(
                max(
                    1,
                    min(
                        4,
                        _bounded_env_float(
                            os.environ,
                            "AURA_DNU_MODEL_RECYCLE_WARMUP_ATTEMPTS",
                            default=3.0,
                            minimum=1.0,
                            maximum=4.0,
                        ),
                    ),
                )
            )
            for attempt in range(1, max_attempts + 1):
                os.environ[_force_key] = "1"
                try:
                    try:
                        warmup_result = warmup(
                            foreground_request=requested_tier == "primary",
                            skip_swap_cooldown=True,
                        )
                    except TypeError:
                        warmup_result = warmup()
                    if asyncio.iscoroutine(warmup_result):
                        warmup_result = await asyncio.wait_for(warmup_result, timeout=240.0)
                    lane_live = _candidate_alive()
                    event["warmup_attempts"].append(
                        {
                            "attempt": attempt,
                            "warmup_result": bool(warmup_result is not False),
                            "lane_live": lane_live,
                            "lane": _lane_state(),
                        }
                    )
                    if lane_live:
                        break
                    if attempt < max_attempts:
                        gc.collect()
                        await asyncio.sleep(min(10.0 * attempt, 20.0))
                except _DNU_RUN_RECOVERABLE_ERRORS as exc:
                    event["warmup_attempts"].append(
                        {
                            "attempt": attempt,
                            "error": f"{type(exc).__name__}: {exc}",
                            "lane_live": _candidate_alive(),
                            "lane": _lane_state(),
                        }
                    )
                    if attempt >= max_attempts:
                        event["status"] = "failed"
                        event["error"] = f"model_lane_rewarm_failed:{type(exc).__name__}: {exc}"
                        event["after"] = collect_proof_resource_snapshot(
                            label="after_model_lane_recycle_failed",
                            task_index=task_index,
                        )
                        append_jsonl(lifecycle_path, event)
                        return event
                    gc.collect()
                    await asyncio.sleep(min(10.0 * attempt, 20.0))
                finally:
                    if _force_prev is None:
                        os.environ.pop(_force_key, None)
                    else:
                        os.environ[_force_key] = _force_prev
        lane_live = _candidate_alive()
        if not lane_live:
            event["status"] = "failed"
            event["error"] = "recycled_model_lane_not_live_after_warmup"
        else:
            event["status"] = "complete"
    event["after"] = collect_proof_resource_snapshot(
        label="after_model_lane_recycle",
        task_index=task_index,
    )
    append_jsonl(lifecycle_path, event)
    return event


async def shutdown_proof_runtime(orchestrator) -> None:
    """Tear down the canonical proof boot so local model workers do not linger."""
    from core.container import ServiceContainer
    from core.runtime.shutdown_coordinator import get_shutdown_coordinator, request_shutdown

    request_shutdown("dnu_agi_proof_battery_complete")
    orchestrator_shutdown_timeout_s = max(
        60.0,
        float(_FLAG_PROOF_ORCHESTRATOR_SHUTDOWN_TIMEOUT_S.value() or 60.0),
    )

    async def _bounded_call(label: str, callback, *, timeout_s: float = 8.0) -> None:
        if not callable(callback):
            return
        try:
            result = callback()
            if asyncio.iscoroutine(result):
                await asyncio.wait_for(result, timeout=timeout_s)
        except _DNU_RUN_RECOVERABLE_ERRORS as exc:
            print(f"  [WARN] Shutdown step {label} failed or timed out: {type(exc).__name__}: {exc}")

    doctor = ServiceContainer.get("flagship_doctor_daemon", default=None)
    if doctor is not None:
        await _bounded_call("flagship_doctor_daemon.stop", getattr(doctor, "stop", None), timeout_s=3.0)

    router = ServiceContainer.get("llm_router", default=None)
    if router is not None and hasattr(router, "endpoints"):
        for endpoint in list(router.endpoints.values()):
            client = getattr(endpoint, "client", None)
            candidates = [client]
            lazy_client = getattr(client, "_client", None)
            if lazy_client is not None:
                candidates.append(lazy_client)
            for candidate in candidates:
                if candidate is None:
                    continue
                for close_name in ("aclose", "close", "cleanup", "on_stop"):
                    close_method = getattr(candidate, close_name, None)
                    if callable(close_method):
                        await _bounded_call(f"model_client.{close_name}", close_method, timeout_s=10.0)
                        break
                else:
                    close_method = None
                if close_method is not None:
                    continue
                reboot_worker = getattr(candidate, "reboot_worker", None)
                if callable(reboot_worker):
                    try:
                        await asyncio.wait_for(
                            reboot_worker(reason="dnu_proof_runtime_shutdown", mark_failed=False),
                            timeout=8.0,
                        )
                    except _DNU_RUN_RECOVERABLE_ERRORS as exc:
                        print(
                            "  [WARN] Shutdown step model_worker.reboot_worker failed or timed out: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    continue
                aclose = getattr(candidate, "aclose", None)
                await _bounded_call("model_client.aclose", aclose, timeout_s=5.0)

    stop_method = getattr(orchestrator, "stop", None)
    await _bounded_call(
        "orchestrator.stop",
        stop_method,
        timeout_s=orchestrator_shutdown_timeout_s,
    )

    try:
        await get_shutdown_coordinator().shutdown(timeout_per_phase=10.0)
    finally:
        await _reap_proof_child_processes(
            "dnu_proof_runtime_shutdown",
            include_resource_trackers=False,
        )


# ---------------------------------------------------------------------------
# Anti-Theater Checks
# ---------------------------------------------------------------------------

def anti_theater_pre_check(tasks: list[dict], grader_data: dict) -> list[str]:
    """Pre-flight anti-theater validation. Returns list of violations."""
    violations = []

    # Check 1: No task contains golden_answer
    for task in tasks:
        if "golden_answer" in task:
            violations.append(f"THEATER: Task {task.get('task_id', '?')} contains golden_answer in task pack")

    # Check 2: Grader salts exist for all tasks
    for task in tasks:
        tid = task.get("task_id", "")
        if tid not in grader_data:
            violations.append(f"INTEGRITY: Task {tid} missing from grader salts")

    # Check 3: All hashes are valid SHA-256 hex
    for tid, entry in grader_data.items():
        h = entry.get("answer_hash", "")
        if not re.match(r"^[0-9a-f]{64}$", h):
            violations.append(f"INTEGRITY: Invalid hash format for {tid}: {h}")

    return violations


def anti_theater_post_check(results: list[dict]) -> list[str]:
    """Post-execution anti-theater validation. Returns list of violations."""
    violations = []

    # Check: No result has a score that wasn't computed from actual execution
    for r in results:
        if r.get("status") == "pass" and not r.get("response_text"):
            violations.append(f"THEATER: Task {r.get('task_id', '?')} marked pass but has no response text")

    # Check: neither the battery nor the evaluated path should need numerical
    # projection libraries for score computation. We ensure numpy is not imported
    # or referenced in the runner's namespace itself to prevent false positives
    # from CognitiveEngine's own authentic internal modules importing it under the hood.
    if "numpy" in globals() or "np" in globals():
        violations.append("THEATER: numpy directly imported in battery runner namespace")

    return violations


# ---------------------------------------------------------------------------
# Task Loading
# ---------------------------------------------------------------------------

TASK_PACK_DIRECTORIES = ["reasoning", "coding", "planning", "self_debug", "transfer", "research"]
COMPARISON_TASK_CATEGORIES = ["novel_reasoning", "coding", "planning", "self_debug", "transfer", "research"]
TASK_CATEGORIES = list(COMPARISON_TASK_CATEGORIES)
DIR_TO_CAT = {
    "reasoning": "novel_reasoning",
    "coding": "coding",
    "planning": "planning",
    "self_debug": "self_debug",
    "transfer": "transfer",
    "research": "research",
}
MINIMUM_COUNTS = {
    "novel_reasoning": 50,
    "coding": 10,
    "planning": 5,
    "self_debug": 5,
    "transfer": 10,
    "research": 10,
}

def load_task_packs(fixture_dir: Path) -> tuple[list[dict], dict]:
    """Load all task packs and grader salts. Returns (tasks, grader_data)."""
    all_tasks = []

    for task_dir in TASK_PACK_DIRECTORIES:
        cat_dir = fixture_dir / task_dir
        tasks_file = cat_dir / "tasks.json"
        if not tasks_file.exists():
            print(f"  [WARN] Task file not found: {tasks_file}")
            continue

        tasks = json.loads(tasks_file.read_text(encoding="utf-8"))
        if isinstance(tasks, list):
            category = DIR_TO_CAT.get(task_dir, task_dir)
            for t in tasks:
                t.setdefault("category", category)
            all_tasks.extend(tasks)
            print(f"  [OK] Loaded {len(tasks)} tasks from {task_dir}/ as {category}")
        else:
            print(f"  [WARN] Invalid task format in {tasks_file}")

    # Load grader salts from ALL salt files
    grader_data = {}
    for salt_file in fixture_dir.glob(".grader_salts*.json"):
        try:
            data = json.loads(salt_file.read_text(encoding="utf-8"))
            grader_data.update(data)
            print(f"  [OK] Loaded {len(data)} grader entries from {salt_file.name}")
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [WARN] Failed to load {salt_file}: {e}")

    return all_tasks, grader_data


def select_stratified_comparison_tasks(tasks: list[dict], limit: int) -> list[dict]:
    """Select a deterministic category-balanced comparison subset."""
    if limit <= 0:
        return []
    by_category: dict[str, list[dict]] = {category: [] for category in COMPARISON_TASK_CATEGORIES}
    for task in tasks:
        by_category.setdefault(str(task.get("category", "unknown")), []).append(task)
    ordered_categories = list(COMPARISON_TASK_CATEGORIES) + sorted(
        category for category in by_category if category not in COMPARISON_TASK_CATEGORIES
    )

    selected: list[dict] = []
    seen: set[str] = set()
    while len(selected) < min(limit, len(tasks)):
        made_progress = False
        for category in ordered_categories:
            bucket = by_category.get(category, [])
            if not bucket:
                continue
            candidate = bucket.pop(0)
            task_key = str(candidate.get("task_id") or id(candidate))
            if task_key in seen:
                continue
            selected.append(candidate)
            seen.add(task_key)
            made_progress = True
            if len(selected) >= min(limit, len(tasks)):
                break
        if not made_progress:
            break
    return selected


def _comparison_task_limit(default: int = 12) -> int:
    """Return the representative comparison task count for baselines/ablations.

    Full proof runs keep the historical breadth by default. CI and live smoke
    gates may set AURA_DNU_COMPARISON_TASK_LIMIT to keep comparison families
    real without letting baseline calls dominate the regression budget.
    """

    raw = _FLAG_DNU_COMPARISON_TASK_LIMIT.value()
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return min(default, max(1, value))


# ---------------------------------------------------------------------------
# Baselines & Ablations Utilities
# ---------------------------------------------------------------------------

def _baseline_timeout_seconds() -> float:
    """Bound comparison baselines so final-proof cannot hang behind Aura's live run."""
    raw = _FLAG_DNU_BASELINE_TIMEOUT_SECONDS.value()
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 90.0
    return min(180.0, max(30.0, value))


def _live_task_attempt_timeout_seconds() -> float:
    """Bound one live-path proof attempt so exact tasks cannot stall the run."""

    raw = _FLAG_DNU_LIVE_ATTEMPT_TIMEOUT_SECONDS.value()
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 90.0
    return min(180.0, max(45.0, value))


def _dnu_baseline_max_tokens() -> int:
    """Token budget for the comparison baselines.

    FAIRNESS FIX (2026-07-06): the previous fixed value of 160 tokens
    handicapped the baselines. The baseline system prompt instructs the model
    to "Think step-by-step", but 160 tokens cannot hold a step-by-step
    derivation for a coding/planning/self-debug task — the model runs out of
    tokens before it can emit an <answer> tag, scoring 'no_answer'. Meanwhile
    full_aura runs the same task through a 240s live path plus the
    deterministic solve_strict_proof_prompt symbolic solver. Comparing an
    unbounded, solver-assisted condition against a 160-token-strangled one is
    not a fair baseline. A fair baseline is the SAME model with a COMPARABLE
    reasoning budget and no architecture. Default raised to 2048; override with
    AURA_DNU_BASELINE_MAX_TOKENS. See docs/DNU_BASELINE_FAIRNESS_AUDIT.md.
    """
    raw = _FLAG_DNU_BASELINE_MAX_TOKENS.value()
    try:
        return max(160, int(raw))
    except (TypeError, ValueError):
        return 2048


DNU_BASELINE_MAX_TOKENS = _dnu_baseline_max_tokens()

#: The treatment arm's wall-clock budget. Named here rather than left implicit
#: so the parity report can compare it against the baselines' instead of the
#: reader having to find it in the live path.
FULL_AURA_WALL_CLOCK_S = 240.0


def _dnu_budget_parity_report() -> dict:
    """What each arm was ALLOWED to spend, checked, and refused when unequal.

    This exists because BASELINES.json had no field for it. Three baselines
    reported 0.1667 apiece — identical to four decimal places across three
    structurally different systems — while running at 160 tokens against a
    full_aura with a 240s budget, effectively unbounded tokens and a
    deterministic symbolic solver. Every fact needed to see the problem was
    somewhere in the repository and none of it was next to the number.

    A difference that HANDICAPS the treatment is a violation too. It is
    tempting to wave that through as conservative; it is still an uncontrolled
    variable, just one that flatters a different conclusion.
    """
    from core.evaluation.matched_budget import ConditionBudget, check_budget_parity

    solver_on = (
        str(_FLAG_ENABLE_STRUCTURED_PROOF_SOLVER.value() or "").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    baseline_timeout = _baseline_timeout_seconds()

    # `solver_available` is declared varied ONLY when the solver is actually
    # on. Declaring it unconditionally would permanently exempt the dimension
    # that voided the original bundle.
    varied = frozenset({"solver_available"} if solver_on else set())

    budgets = [
        ConditionBudget(
            condition="baselines",
            model_id="local",
            max_output_tokens=DNU_BASELINE_MAX_TOKENS,
            max_wall_clock_s=baseline_timeout,
            solver_available=False,
            varied=varied,
        ),
        ConditionBudget(
            condition="full_aura",
            model_id="local",
            # The treatment's REAL budget, not the baseline's copied across.
            # The first version of this function declared full_aura at the
            # baseline's 2048/90s and got `matched: True` for a comparison this
            # repository has already retracted as unmatched — a parity check
            # that asserts parity instead of measuring it is worse than none,
            # because it launders the defect it was added to catch.
            #
            # full_aura runs the live message path: no per-call token cap, and
            # the 240s ceiling named in FULL_AURA_WALL_CLOCK_S. Declared
            # honestly, this comparison is VOID, and that is the correct
            # current state of the DNU bundle.
            max_output_tokens=None,  # unbounded, by construction of the live path
            max_wall_clock_s=FULL_AURA_WALL_CLOCK_S,
            solver_available=solver_on,
            varied=varied,
        ),
    ]
    report = check_budget_parity(budgets)
    payload = report.to_dict()
    payload["refusal_reason"] = report.refusal_reason()
    payload["summary"] = (
        f"baselines: tokens={DNU_BASELINE_MAX_TOKENS} wall_clock={baseline_timeout}s "
        f"solver=off | full_aura: tokens=unbounded "
        f"wall_clock={FULL_AURA_WALL_CLOCK_S}s "
        f"solver={'on(declared varied)' if solver_on else 'off'}"
    )
    payload["what_would_match_them"] = (
        "cap the live path at the same token budget the baselines get, and hold both "
        "arms to one wall-clock ceiling. Until then this comparison is void and the "
        "100/100 stands only as an ABSOLUTE score on a self-authored battery — which "
        "is exactly what CLAIMS_MATRIX claim 13 says."
    )
    payload["treatment_wall_clock_ceiling_s"] = FULL_AURA_WALL_CLOCK_S
    payload["note"] = (
        "When the solver is ON it is the declared independent variable, and the "
        "claim this bundle supports is about the architecture INCLUDING its "
        "symbolic solver — not about the cognitive layer alone. "
        "See docs/DNU_BASELINE_FAIRNESS_AUDIT.md."
    )
    return payload


def _iter_router_generation_clients(router):
    """Yield router/client objects that may own an active baseline generation."""

    yielded: set[int] = set()

    def _yield(candidate):
        if candidate is None:
            return []
        objects = [candidate]
        for attr in ("client", "_client", "_mlx_client"):
            nested = getattr(candidate, attr, None)
            if nested is not None:
                objects.append(nested)
        fresh = []
        for obj in objects:
            ident = id(obj)
            if ident in yielded:
                continue
            yielded.add(ident)
            fresh.append(obj)
        return fresh

    for obj in _yield(router):
        yield obj

    endpoints = getattr(router, "endpoints", None)
    if isinstance(endpoints, dict):
        endpoint_iter = endpoints.values()
    elif isinstance(endpoints, (list, tuple, set)):
        endpoint_iter = endpoints
    else:
        endpoint_iter = ()

    for endpoint in endpoint_iter:
        for obj in _yield(endpoint):
            yield obj


def _force_abort_router_generation(router, *, reason: str) -> int:
    """Best-effort emergency abort for a router/client stuck past cancellation."""

    aborted = 0
    for client in _iter_router_generation_clients(router):
        abort = getattr(client, "force_abort_active_generation", None)
        if not callable(abort):
            continue
        try:
            if abort(reason=reason):
                aborted += 1
        except _DNU_RUN_RECOVERABLE_ERRORS as exc:
            print(f"  [WARN] Baseline watchdog abort skipped for one client: {exc}", flush=True)
    return aborted


async def _recover_router_after_baseline_abort(router, *, reason: str) -> int:
    """Recover clients that accepted an emergency baseline abort."""

    recovered = 0
    for client in _iter_router_generation_clients(router):
        reboot = getattr(client, "reboot_worker", None)
        if not callable(reboot):
            continue
        try:
            try:
                result = reboot(reason=f"baseline_abort_recovery:{reason}", mark_failed=False)
            except TypeError:
                result = reboot()
            if asyncio.iscoroutine(result):
                await result
            recovered += 1
        except _DNU_RUN_RECOVERABLE_ERRORS as exc:
            print(f"  [WARN] Baseline watchdog recovery skipped for one client: {exc}", flush=True)
    return recovered


async def _cancel_baseline_task(task: asyncio.Task) -> bool:
    """Cancel a timed-out comparison baseline without killing the live model lane."""

    if task.done():
        return True
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=3.0)
        return True
    except (asyncio.CancelledError, TimeoutError):
        return task.done()


async def _generate_baseline_response(
    router,
    *,
    prompt: str,
    system_prompt: str,
    purpose: str,
) -> str:
    """Run a baseline model call with bounded, non-destructive isolation.

    The comparison still uses the requested live 32B lane, but a baseline
    timeout must not kill or reboot Aura's shared foreground-capable worker.
    Timed-out baselines are recorded as baseline failures while the runtime
    remains available for the proof battery and normal launch path.
    """
    timeout_s = _baseline_timeout_seconds()
    loop = asyncio.get_running_loop()
    task = loop.create_task(
        router.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            origin="baseline",
            purpose=purpose,
            benchmark_request=True,
            foreground_request=False,
            skip_runtime_payload=True,
            timeout=timeout_s,
            prefer_tier=os.environ.get("AURA_PROOF_MODEL_TIER", "primary"),
            proof_primary_lane_required=True,
            allow_cloud_fallback=False,
            disable_prompt_cache=True,
            clear_prompt_cache=True,
            temperature=0.15,
            temp=0.15,
            top_p=0.85,
            min_p=0.03,
            repetition_penalty=1.35,
            repetition_context_size=1024,
            stop_sequences=["\n\n", "\\n", "User:", "Assistant:", "<|im_end|>", "<|endoftext|>"],
            max_tokens=DNU_BASELINE_MAX_TOKENS,
            num_predict=DNU_BASELINE_MAX_TOKENS,
        ),
        name=f"dnu_baseline:{purpose}",
    )
    reason = f"{purpose}_hard_timeout_{timeout_s:.0f}s"
    watchdog_fired = threading.Event()

    def _watchdog_abort() -> None:
        watchdog_fired.set()

    watchdog = threading.Timer(timeout_s, _watchdog_abort)
    watchdog.daemon = True
    watchdog.start()
    try:
        return await asyncio.wait_for(task, timeout=timeout_s)
    except TimeoutError as exc:
        cancelled = await _cancel_baseline_task(task)
        if not cancelled:
            abort_reason = reason if watchdog_fired.is_set() else f"{reason}_cancel_stuck"
            aborted = _force_abort_router_generation(router, reason=abort_reason)
            if aborted:
                await _recover_router_after_baseline_abort(router, reason=abort_reason)
        print(
            f"  [WARN] Baseline timed out cooperatively for {purpose} "
            f"after {timeout_s:.0f}s; shared model lane preserved.",
            flush=True,
        )
        raise TimeoutError(reason) from exc
    finally:
        watchdog.cancel()


@contextlib.contextmanager
def lesion_services(names: list[str]):
    """Dynamically unregister or replace services in ServiceContainer."""
    from core.container import ServiceContainer, ServiceDescriptor, ServiceLifetime
    from core.runtime.ablation_policy import mark_services_lesioned

    original = {}
    with mark_services_lesioned(names):
        with ServiceContainer._lock:
            for name in names:
                resolved_name = ServiceContainer._resolve_name(name)
                if resolved_name in ServiceContainer._services:
                    original[resolved_name] = ServiceContainer._services[resolved_name]
                    # Replace with a descriptor that returns None.
                    ServiceContainer._services[resolved_name] = ServiceDescriptor(
                        name=resolved_name,
                        factory=lambda *args, **kwargs: None,
                        lifetime=ServiceLifetime.SINGLETON,
                        instance=None,
                        required=False,
                        initialized=True,
                    )
        try:
            yield
        finally:
            # Restore original descriptors.
            with ServiceContainer._lock:
                for resolved_name, desc in original.items():
                    ServiceContainer._services[resolved_name] = desc


async def execute_raw_llm_task(router, task: dict, grader_data: dict, sem: asyncio.Semaphore) -> dict:
    task_id = task.get("task_id", "unknown")
    prompt = task.get("task_prompt", "")
    system_prompt = (
        "You are a helpful assistant. Solve the user's problem. "
        "Think step-by-step. Put your final answer strictly inside <answer>...</answer> tags. "
        "For example, <answer>Alice</answer> or <answer>5</answer>."
    )
    result = {
        "task_id": task_id,
        "category": task.get("category", "unknown"),
        "difficulty": task.get("difficulty", "unknown"),
        "status": "error",
        "response_text": "",
        "extracted_answer": None,
        "normalized_answer": None,
        "answer_hash": None,
        "elapsed_s": 0.0,
        "error": None,
    }
    t0 = time.time()
    try:
        async with sem:
            response = await _generate_baseline_response(
                router,
                prompt=prompt,
                system_prompt=system_prompt,
                purpose="raw_llm_baseline",
            )
        result["response_text"] = response
        result["elapsed_s"] = time.time() - t0
        result["status"] = "success"

        extracted = extract_answer_tag(response)
        if extracted:
            result["extracted_answer"] = extracted
            result["normalized_answer"] = normalize_answer(extracted)
        else:
            if len(response.strip()) < 200:
                result["extracted_answer"] = response.strip()
                result["normalized_answer"] = normalize_answer(response)
            else:
                result["status"] = "no_answer"
                result["error"] = "No <answer> tags found in response"
    except TimeoutError:
        result["status"] = "timeout"
        result["error"] = f"Task exceeded baseline time budget of {_baseline_timeout_seconds():.0f}s"
        result["elapsed_s"] = time.time() - t0
    except _DNU_RUN_RECOVERABLE_ERRORS as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["elapsed_s"] = time.time() - t0

    # Grade the result
    result = grade_result(result, grader_data)
    return result


async def execute_react_task(router, task: dict, grader_data: dict, sem: asyncio.Semaphore) -> dict:
    task_id = task.get("task_id", "unknown")
    prompt = task.get("task_prompt", "")
    system_prompt = (
        "You are a ReAct reasoning agent. Solve the task step-by-step by generating "
        "Thought, Action, Observation steps. You do not have actual tool access, so you should "
        "generate the Actions and the corresponding Observations yourself to structure your thinking. "
        "Finally, wrap your final answer strictly inside <answer>...</answer> tags."
    )
    result = {
        "task_id": task_id,
        "category": task.get("category", "unknown"),
        "difficulty": task.get("difficulty", "unknown"),
        "status": "error",
        "response_text": "",
        "extracted_answer": None,
        "normalized_answer": None,
        "answer_hash": None,
        "elapsed_s": 0.0,
        "error": None,
    }
    t0 = time.time()
    try:
        async with sem:
            response = await _generate_baseline_response(
                router,
                prompt=prompt,
                system_prompt=system_prompt,
                purpose="react_agent_baseline",
            )
        result["response_text"] = response
        result["elapsed_s"] = time.time() - t0
        result["status"] = "success"

        extracted = extract_answer_tag(response)
        if extracted:
            result["extracted_answer"] = extracted
            result["normalized_answer"] = normalize_answer(extracted)
        else:
            if len(response.strip()) < 200:
                result["extracted_answer"] = response.strip()
                result["normalized_answer"] = normalize_answer(response)
            else:
                result["status"] = "no_answer"
                result["error"] = "No <answer> tags found in response"
    except TimeoutError:
        result["status"] = "timeout"
        result["error"] = f"Task exceeded baseline time budget of {_baseline_timeout_seconds():.0f}s"
        result["elapsed_s"] = time.time() - t0
    except _DNU_RUN_RECOVERABLE_ERRORS as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["elapsed_s"] = time.time() - t0

    # Grade the result
    result = grade_result(result, grader_data)
    return result


async def execute_llm_with_tools_task(router, task: dict, grader_data: dict, sem: asyncio.Semaphore) -> dict:
    task_id = task.get("task_id", "unknown")
    prompt = task.get("task_prompt", "")
    system_prompt = (
        "You are an agent equipped with direct tools. Solve the user's problem step-by-step.\n\n"
        "Available Tools:\n"
        "- `read_file(path)`: Read file contents.\n"
        "- `write_file(path, content)`: Write file contents.\n"
        "- `execute_command(cmd)`: Execute a shell command.\n"
        "- `web_search(query)`: Search the web.\n"
        "- `read_web_page(url)`: Read web page content.\n\n"
        "You may invoke tools by printing: `Tool Call: <tool_name>(<args>)`. "
        "The system will return simulated success. "
        "Finally, think step-by-step and wrap your final answer strictly inside <answer>...</answer> tags."
    )
    result = {
        "task_id": task_id,
        "category": task.get("category", "unknown"),
        "difficulty": task.get("difficulty", "unknown"),
        "status": "error",
        "response_text": "",
        "extracted_answer": None,
        "normalized_answer": None,
        "answer_hash": None,
        "elapsed_s": 0.0,
        "error": None,
    }
    t0 = time.time()
    try:
        async with sem:
            response = await _generate_baseline_response(
                router,
                prompt=prompt,
                system_prompt=system_prompt,
                purpose="llm_with_tools_baseline",
            )
        result["response_text"] = response
        result["elapsed_s"] = time.time() - t0
        result["status"] = "success"

        extracted = extract_answer_tag(response)
        if extracted:
            result["extracted_answer"] = extracted
            result["normalized_answer"] = normalize_answer(extracted)
        else:
            if len(response.strip()) < 200:
                result["extracted_answer"] = response.strip()
                result["normalized_answer"] = normalize_answer(response)
            else:
                result["status"] = "no_answer"
                result["error"] = "No <answer> tags found in response"
    except TimeoutError:
        result["status"] = "timeout"
        result["error"] = f"Task exceeded baseline time budget of {_baseline_timeout_seconds():.0f}s"
        result["elapsed_s"] = time.time() - t0
    except _DNU_RUN_RECOVERABLE_ERRORS as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["elapsed_s"] = time.time() - t0

    # Grade the result
    result = grade_result(result, grader_data)
    return result


def build_ablation_report_entry(
    *,
    ablation_name: str,
    pass_rate: float,
    services_requested: list[str],
    services_disabled: set[str],
    lesion_verified: bool,
    dnu_behavior_degraded: bool,
    sample_categories: Counter | dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build an honest ablation artifact entry.

    The DNU runner isolates every prompt into a fresh task state. That is correct
    for leakage control, but it also means several organs cannot be expected to
    move single-shot reasoning scores. We preserve the observed DNU score and
    explicitly name the proof surface that owns each dependency claim instead of
    treating equal performance as either success or theater.
    """

    evidence = DNU_ABLATION_DEPENDENCY_EVIDENCE.get(
        ablation_name,
        {
            "dnu_score_delta_required": True,
            "reason": "No dedicated dependency-evidence mapping is registered for this ablation.",
            "expected_dependency_evidence": ["dnu_ablations.dnu_behavior_degraded"],
        },
    )
    dnu_score_delta_required = bool(evidence.get("dnu_score_delta_required", True))
    dependency_evidence_required_elsewhere = not dnu_score_delta_required
    lesion_run_verified = lesion_verified and bool(services_disabled)
    lesion_effect_verified = lesion_run_verified and (
        dnu_behavior_degraded or dependency_evidence_required_elsewhere
    )
    entry: dict[str, Any] = {
        "status": "RUN",
        "pass_rate": pass_rate,
        "lesion_effect_verified": lesion_effect_verified,
        "lesion_effect_verified_in_this_battery": dnu_behavior_degraded,
        "lesion_effect_verification_scope": (
            "dnu_score_delta"
            if dnu_behavior_degraded
            else "delegated_to_dedicated_cert_chain"
            if dependency_evidence_required_elsewhere
            else "unverified"
        ),
        "lesion_run_verified": lesion_run_verified,
        "dnu_behavior_degraded": dnu_behavior_degraded,
        "dnu_score_delta_required": dnu_score_delta_required,
        "dependency_evidence_required_elsewhere": dependency_evidence_required_elsewhere,
        "dependency_evidence_note": evidence.get("reason", ""),
        "expected_dependency_evidence": list(evidence.get("expected_dependency_evidence", [])),
        "services_requested": list(services_requested),
        "disabled_components": sorted(services_disabled),
    }
    if sample_categories is not None:
        entry["sample_categories"] = dict(sample_categories)
    return entry


async def run_ablation_suite(
    runtime,
    tasks: list[dict],
    grader_data: dict,
    services_to_lesion: list[str],
    *,
    ablation_name: str,
    sample_categories: Counter | dict[str, int] | None = None,
) -> dict[str, Any]:
    from core.container import ServiceContainer

    if not tasks:
        entry = build_ablation_report_entry(
            ablation_name=ablation_name,
            pass_rate=0.0,
            services_requested=services_to_lesion,
            services_disabled=set(),
            lesion_verified=False,
            dnu_behavior_degraded=False,
            sample_categories=sample_categories,
        )
        entry.update(
            {
                "status": "NOT_RUN",
                "reason": "No comparison tasks were selected for this ablation; no lesion evidence was produced.",
                "lesion_effect_verified": False,
                "lesion_effect_verified_in_this_battery": False,
                "lesion_run_verified": False,
            }
        )
        print(
            f"  [ERROR] Ablation {ablation_name} did not run: no comparison tasks selected."
        )
        return entry

    # Programmatically verify that the lesion is active, then preserve the
    # observed isolation-scrubbed DNU score without overstating what it proves.
    lesion_verified = True
    services_disabled: set[str] = set()
    ablation_results = []
    with lesion_services(services_to_lesion):
        for s_name in services_to_lesion:
            try:
                inst = ServiceContainer.get(s_name, default=None)
                if inst is not None:
                    print(f"  [ERROR] Lesion failed to verify for: {s_name} (got {inst})")
                    lesion_verified = False
                else:
                    services_disabled.add(s_name)
            except (LookupError, KeyError, AttributeError):
                # Service-not-found semantics mean the lesion removed the service.
                services_disabled.add(s_name)
            except _DNU_RUN_RECOVERABLE_ERRORS as exc:
                print(f"  [ERROR] Lesion verification failed while fetching {s_name}: {type(exc).__name__}: {exc}")
                lesion_verified = False

        for task in tasks:
            try:
                await isolate_live_runtime_for_dnu_task(task)
            except _DNU_RUN_RECOVERABLE_ERRORS as exc:
                print(f"  [WARN] Failed to reset state for ablation isolation: {exc}")
            res = await execute_task(runtime, task, timeout_s=max(240, task.get("time_budget_s", 240)))
            res = grade_result(res, grader_data)
            ablation_results.append(res)
    scorecard = compute_scorecard(ablation_results)
    pass_rate = scorecard["overall_pass_rate"]
    dnu_behavior_degraded = pass_rate < 1.0
    entry = build_ablation_report_entry(
        ablation_name=ablation_name,
        pass_rate=pass_rate,
        services_requested=services_to_lesion,
        services_disabled=services_disabled,
        lesion_verified=lesion_verified,
        dnu_behavior_degraded=dnu_behavior_degraded,
        sample_categories=sample_categories,
    )
    if not entry["lesion_run_verified"]:
        print(
            "  [ERROR] Ablation lesion did not verify for requested services: "
            f"{services_to_lesion}"
        )
    elif entry["dnu_score_delta_required"] and not dnu_behavior_degraded:
        print(
            "  [ERROR] Ablation did not degrade the DNU subset where a DNU score delta is required: "
            f"{ablation_name} disabled={sorted(services_disabled)}"
        )
    elif not dnu_behavior_degraded:
        print(
            "  [INFO] Ablation left isolation-scrubbed DNU score unchanged; dependency evidence is owned by "
            f"{entry['expected_dependency_evidence']}"
        )
    return entry


def _scrub_dnu_state_for_task(state, task: dict):
    """Clear turn-local proof residue while preserving the canonical runtime state shape."""

    from core.runtime.proof_policy import clear_transient_response_modifiers

    prompt = str(task.get("task_prompt", "") or "")
    task_id = str(task.get("task_id", "unknown") or "unknown")
    cognition = getattr(state, "cognition", None)
    if cognition is not None:
        cognition.working_memory = []
        cognition.long_term_memory = []
        cognition.rolling_summary = ""
        cognition.current_objective = None
        cognition.current_origin = PROOF_LIVE_MESSAGE_ORIGIN
        cognition.attention_focus = ""
        cognition.last_response = None
        cognition.discourse_topic = None
        cognition.discourse_branches = []
        if hasattr(cognition, "active_goals"):
            cognition.active_goals = []
        if hasattr(cognition, "pending_intents"):
            cognition.pending_intents = []
        if hasattr(cognition, "pending_initiatives"):
            cognition.pending_initiatives = []
        if hasattr(cognition, "phenomenal_state"):
            cognition.phenomenal_state = ""
        if hasattr(cognition, "modifiers"):
            cognition.modifiers = {}
    if not isinstance(getattr(state, "response_modifiers", None), dict):
        state.response_modifiers = {}
    clear_transient_response_modifiers(state.response_modifiers, strict=True)
    strict_answer_request = "<answer>" in prompt.lower()
    state.response_modifiers["proof_evaluation_turn"] = True
    state.response_modifiers["proof_turn_objective"] = prompt
    state.response_modifiers["proof_task_id"] = task_id
    state.response_modifiers["proof_task_prompt_hash"] = hashlib.sha256(prompt.encode()).hexdigest()
    if strict_answer_request:
        state.response_modifiers["strict_proof_answer_request"] = True
    return state


async def isolate_live_runtime_for_dnu_task(task: dict) -> None:
    """Reset live proof-turn state across repository and kernel before each DNU task."""

    from core.container import ServiceContainer

    state_repo = ServiceContainer.get("state_repository", default=None)
    if state_repo:
        state = await state_repo.get_current()
        if state:
            derived = state.derive("task_isolation_reset", origin="dnu_agi_proof_battery")
            _scrub_dnu_state_for_task(derived, task)
            current = getattr(state_repo, "_current", None)
            current_version = int(getattr(current, "version", 0) or 0)
            if int(getattr(derived, "version", 0) or 0) <= current_version:
                derived.version = current_version + 1
                derived.parent_state_id = getattr(current, "state_id", getattr(derived, "parent_state_id", None))
            await state_repo.commit(derived, "task_isolation_reset")

    try:
        from core.kernel.kernel_interface import KernelInterface

        ki = KernelInterface.get_instance()
        kernel = getattr(ki, "_kernel", None)
        kernel_state = getattr(kernel, "state", None)
        if kernel is not None and kernel_state is not None:
            derived = kernel_state.derive(
                "dnu_kernel_task_isolation",
                origin="dnu_agi_proof_battery",
            )
            kernel.state = _scrub_dnu_state_for_task(derived, task)
    except _DNU_RUN_RECOVERABLE_ERRORS:
        raise


# ---------------------------------------------------------------------------
# Task Execution
# ---------------------------------------------------------------------------

async def execute_task(runtime, task: dict, timeout_s: int = 240) -> dict:
    """Execute a single task through the canonical live message path and return result."""
    task_id = task.get("task_id", "unknown")
    prompt = task.get("task_prompt", "")
    budget = max(240, task.get("time_budget_s", timeout_s))

    result = {
        "task_id": task_id,
        "category": task.get("category", "unknown"),
        "difficulty": task.get("difficulty", "unknown"),
        "status": "error",
        "response_text": "",
        "extracted_answer": None,
        "normalized_answer": None,
        "answer_hash": None,
        "elapsed_s": 0.0,
        "error": None,
    }

    t0 = time.time()
    
    async def _run_live_path() -> str:
        if hasattr(runtime, "process_user_input_priority"):
            if hasattr(runtime, "_last_emitted_fingerprint"):
                runtime._last_emitted_fingerprint = ""
            response = await runtime.process_user_input_priority(
                prompt,
                origin=PROOF_LIVE_MESSAGE_ORIGIN,
                timeout_sec=float(budget),
            )
            return str(response or "")
        thought = await runtime.think(
            objective=prompt,
            origin="test",
            prefer_tier=os.environ.get("AURA_PROOF_MODEL_TIER", "primary"),
        )
        return str(getattr(thought, "content", "") or "")

    def _resolve_live_abort_target():
        try:
            from core.container import ServiceContainer

            return (
                ServiceContainer.get("llm_router", default=None)
                or ServiceContainer.get("inference_gate", default=None)
            )
        except _DNU_RUN_RECOVERABLE_ERRORS:
            return None

    async def _run_live_path_attempt(attempt_label: str, timeout_s: float) -> str:
        router = _resolve_live_abort_target()
        abort_reason = f"dnu_live_task_{task_id}_{attempt_label}_timeout_{timeout_s:.0f}s"
        watchdog_fired = threading.Event()
        watchdog_aborted = {"count": 0}

        def _watchdog_abort() -> None:
            watchdog_fired.set()
            if router is not None:
                watchdog_aborted["count"] = _force_abort_router_generation(
                    router,
                    reason=abort_reason,
                )

        watchdog = threading.Timer(max(0.01, float(timeout_s)), _watchdog_abort)
        watchdog.daemon = True
        watchdog.start()
        task_obj = asyncio.create_task(
            _run_live_path(),
            name=f"dnu_live_task:{task_id}:{attempt_label}",
        )
        try:
            response = await asyncio.wait_for(task_obj, timeout=timeout_s)
            if watchdog_fired.is_set():
                raise TimeoutError(abort_reason)
            return response
        except TimeoutError as exc:
            if not task_obj.done():
                task_obj.cancel()
                try:
                    await asyncio.wait_for(task_obj, timeout=3.0)
                except (asyncio.CancelledError, TimeoutError):
                    pass
            if router is not None and not watchdog_aborted["count"]:
                watchdog_aborted["count"] = _force_abort_router_generation(
                    router,
                    reason=abort_reason,
                )
            if router is not None and watchdog_aborted["count"]:
                await _recover_router_after_baseline_abort(router, reason=abort_reason)
            raise TimeoutError(abort_reason) from exc
        finally:
            watchdog.cancel()

    def _is_non_answer(text: str) -> bool:
        if not str(text or "").strip():
            return True
        try:
            from core.conversation.response_reliability import is_non_answer_repair_floor_reply

            if is_non_answer_repair_floor_reply(text):
                return True
        except _DNU_RUN_RECOVERABLE_ERRORS:
            pass
        return False

    # Milestone 1: Soft timeout (200s) and one live-path recovery retry.
    soft_budget = min(200, int(budget * 0.85))
    live_attempt_budget = min(float(soft_budget), _live_task_attempt_timeout_seconds())
    prompt_derived_repair_payload: dict[str, Any] | None = None

    def _prompt_derived_strict_answer_repair(reason: str) -> tuple[str, dict[str, Any]] | None:
        structured_solver_enabled = (
            str(_FLAG_ENABLE_STRUCTURED_PROOF_SOLVER.value() or "")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        )
        if not structured_solver_enabled or "<answer>" not in prompt.lower():
            return None
        try:
            from core.reasoning.proof_answer_solver import solve_strict_proof_prompt

            solved = solve_strict_proof_prompt(prompt)
        except _DNU_RUN_RECOVERABLE_ERRORS as exc:
            result["answer_source_error"] = f"{type(exc).__name__}: {exc}"
            return None
        if not solved:
            return None
        payload = {
            "solver": solved.solver,
            "confidence": solved.confidence,
            "provenance": "dnu_live_path_prompt_derived_repair",
            "reason": reason,
        }
        return f"<answer>{solved.answer}</answer>", payload

    try:
        response_text = await _run_live_path_attempt("first", live_attempt_budget)
        if _is_non_answer(response_text):
            raise RuntimeError("live_path_returned_no_answer")
    except _DNU_TASK_ATTEMPT_ERRORS as exc:
        print(f"\n  [WARN] First attempt for {task_id} failed or soft-timed out ({type(exc).__name__}). Retrying through live path...", end="", flush=True)
        try:
            retry_budget = min(
                _live_task_attempt_timeout_seconds(),
                max(15.0, float(budget) - (time.time() - t0) - 2.0),
            )
            response_text = await _run_live_path_attempt("retry", retry_budget)
            if _is_non_answer(response_text):
                raise RuntimeError("live_path_returned_no_answer")
        except _DNU_TASK_ATTEMPT_ERRORS as retry_exc:
            repair = _prompt_derived_strict_answer_repair(
                f"live_path_retry_failed:{type(retry_exc).__name__}:{str(retry_exc)}"
            )
            if repair is None:
                result["status"] = "timeout" if isinstance(retry_exc, asyncio.TimeoutError) else "error"
                result["error"] = f"Retry failed: {type(retry_exc).__name__}: {str(retry_exc)}"
                result["elapsed_s"] = time.time() - t0
                return result
            response_text, prompt_derived_repair_payload = repair

    result["response_text"] = response_text
    result["elapsed_s"] = time.time() - t0
    result["status"] = "success"
    result["answer_source"] = (
        "prompt_derived_symbolic_repair"
        if prompt_derived_repair_payload
        else "model_or_runtime"
    )
    result["structured_proof_solver"] = prompt_derived_repair_payload
    if prompt_derived_repair_payload:
        result["system2_symbolic_reasoner"] = prompt_derived_repair_payload
        result["strict_proof_symbolic_validation"] = {
            "stage": "dnu_live_no_answer_repair",
            "method": "prompt_derived_symbolic_repair",
            "solver": prompt_derived_repair_payload.get("solver"),
            "reason": prompt_derived_repair_payload.get("reason"),
        }
    try:
        from core.container import ServiceContainer

        state_repo = ServiceContainer.get("state_repository", default=None)
        state = await state_repo.get_current() if state_repo else None
        modifiers = getattr(state, "response_modifiers", {}) if state is not None else {}
        solver_payload = (
            modifiers.get("structured_proof_solver")
            if isinstance(modifiers, dict)
            else None
        )
        strict_symbolic_validation = (
            modifiers.get("strict_proof_symbolic_validation")
            if isinstance(modifiers, dict)
            else None
        )
        if solver_payload and not prompt_derived_repair_payload:
            result["answer_source"] = "system2_symbolic_reasoner"
            result["structured_proof_solver"] = solver_payload
            result["system2_symbolic_reasoner"] = solver_payload
        if strict_symbolic_validation:
            result["strict_proof_symbolic_validation"] = strict_symbolic_validation
            validation_stage = str(strict_symbolic_validation.get("stage", "") or "")
            validation_method = str(strict_symbolic_validation.get("method", "") or "")
            if (
                "prompt_derived_repair" in validation_stage
                or validation_method == "prompt_derived_symbolic_repair"
            ):
                result["answer_source"] = "prompt_derived_symbolic_repair"
    except _DNU_RUN_RECOVERABLE_ERRORS as exc:
        result["answer_source_error"] = f"{type(exc).__name__}: {exc}"

    # Extract answer from <answer> tags
    extracted = extract_answer_tag(result["response_text"])
    if extracted:
        result["extracted_answer"] = extracted
        result["normalized_answer"] = normalize_answer(extracted)
        structured_solver_enabled = (
            str(_FLAG_ENABLE_STRUCTURED_PROOF_SOLVER.value() or "")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        )
        if structured_solver_enabled and result.get("answer_source") == "model_or_runtime":
            try:
                from core.reasoning.proof_answer_solver import validate_strict_proof_answer

                validation = validate_strict_proof_answer(prompt, extracted)
                if validation.valid is True and validation.solver:
                    solver_payload = {
                        "solver": validation.solver,
                        "confidence": 1.0,
                        "provenance": "post_trace_inferred_from_enabled_system2_validator",
                        "reason": validation.reason,
                    }
                    result["answer_source"] = "system2_symbolic_reasoner"
                    result["structured_proof_solver"] = solver_payload
                    result["system2_symbolic_reasoner"] = solver_payload
            except _DNU_RUN_RECOVERABLE_ERRORS as exc:
                result["answer_source_error"] = f"{type(exc).__name__}: {exc}"
    else:
        result["status"] = "no_answer"
        result["error"] = "No <answer> tags found in response"

    return result


def grade_result(result: dict, grader_data: dict) -> dict:
    """Grade a task result against the salted answer hash."""
    task_id = result["task_id"]

    if task_id not in grader_data:
        result["status"] = "ungraded"
        result["error"] = f"No grader entry for {task_id}"
        return result

    if result["status"] in ("timeout", "error"):
        return result

    if result["normalized_answer"] is None:
        result["status"] = "no_answer"
        return result

    entry = grader_data[task_id]
    salt = entry["salt"]
    expected_hash = entry["answer_hash"]

    computed_hash = hash_answer(salt, result["normalized_answer"])
    result["answer_hash"] = computed_hash

    if computed_hash == expected_hash:
        result["status"] = "pass"
    else:
        result["status"] = "fail"
        # Don't leak the golden answer - just note the hash mismatch
        result["error"] = f"Hash mismatch: computed {computed_hash[:16]}... != expected {expected_hash[:16]}..."

    return result


def _read_jsonl(path: Path) -> tuple[list[dict], int]:
    records: list[dict] = []
    invalid = 0
    if not path.exists():
        return records, invalid
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
            else:
                invalid += 1
        except json.JSONDecodeError:
            invalid += 1
    return records, invalid


def build_governance_report(receipts_file: Path, *, expected_tasks: int) -> dict:
    """Build an evidence-backed governance report from DNU receipts and live negatives."""

    records, invalid_lines = _read_jsonl(receipts_file)
    dnu_records = [
        record for record in records if record.get("source") == "dnu_agi_proof_battery"
    ]
    receipt_count = sum(1 for record in records if str(record.get("receipt_id", "")).startswith("will_"))
    pre_action_count = sum(
        1 for record in dnu_records if record.get("authorization_phase") == "pre_action"
    )
    effect_proof_count = sum(
        1
        for record in dnu_records
        if isinstance(record.get("effect_verified"), bool)
        and isinstance(record.get("telemetry_logged"), bool)
        and record.get("closure_verified") is True
    )
    pre_action_missing = max(0, expected_tasks - pre_action_count)
    missing_effect_proof = max(0, expected_tasks - effect_proof_count)
    invalid_receipts = invalid_lines + sum(
        1 for record in records if not str(record.get("receipt_id", "")).startswith("will_")
    )

    negative_tests: dict[str, bool]
    try:
        from tools.receipt_coverage_validator import run_negative_tests

        negative_tests = run_negative_tests()
    except _DNU_RUN_RECOVERABLE_ERRORS as exc:
        negative_tests = {
            "negative_test_harness_executed": False,
            "error": False,
            "error_detail": f"{type(exc).__name__}: {exc}",
        }

    negative_tests_passed = all(value is True for value in negative_tests.values())
    surface_counts: dict[str, int] = {}
    for record in records:
        domain = str(record.get("domain", "unknown") or "unknown")
        surface_counts[domain] = surface_counts.get(domain, 0) + 1

    status = (
        "pass"
        if (
            receipt_count > 0
            and invalid_receipts == 0
            and pre_action_missing == 0
            and missing_effect_proof == 0
            and negative_tests_passed
        )
        else "fail"
    )
    return {
        "schema": "aura.dnu_governance_report.v2",
        "generated_by": "tools/agi/run_dnu_agi_proof_battery.py::build_governance_report",
        "status": status,
        "receipt_count": receipt_count,
        "dnu_pre_action_receipt_count": pre_action_count,
        "expected_dnu_tasks": expected_tasks,
        "pre_action_authorization_missing": pre_action_missing,
        "missing_effect_proof_count": missing_effect_proof,
        "invalid_receipts": invalid_receipts,
        "surface_counts": surface_counts,
        "negative_tests": negative_tests,
        "negative_tests_passed": negative_tests_passed,
        "bypass_count": pre_action_missing + missing_effect_proof + invalid_receipts,
        "forged_receipt_result": (
            "pass" if negative_tests.get("forged_receipt_rejected") is True else "fail"
        ),
        "missing_effect_proof_result": (
            "pass" if negative_tests.get("missing_effect_proof_rejected") is True else "fail"
        ),
        "disabled_will_result": (
            "pass" if negative_tests.get("disabled_will_blocks_action") is True else "fail"
        ),
    }


def build_leakage_report(
    *,
    pre_violations: list[str],
    post_violations: list[str],
    run_dir: Path,
) -> dict:
    """Build an anti-theater/proof-integrity report from actual scans."""

    try:
        from tools.proof_integrity_lint import run_lint

        proof_integrity = run_lint(PROJECT_ROOT, "production")
    except _DNU_RUN_RECOVERABLE_ERRORS as exc:
        proof_integrity = {
            "passed": False,
            "findings": [
                {
                    "kind": "proof_integrity_lint_error",
                    "file": "tools/proof_integrity_lint.py",
                    "line": 0,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            ],
        }

    task_trace = run_dir / "TASK_TRACE.jsonl"
    trace_records, trace_invalid = _read_jsonl(task_trace)
    system2_symbolic_tasks = [
        record.get("task_id")
        for record in trace_records
        if record.get("answer_source") in {"system2_symbolic_reasoner", "structured_proof_solver"}
    ]
    prompt_derived_repair_tasks = [
        record.get("task_id")
        for record in trace_records
        if record.get("answer_source") == "prompt_derived_symbolic_repair"
    ]
    strict_symbolic_validation_tasks = [
        record.get("task_id")
        for record in trace_records
        if record.get("strict_proof_symbolic_validation")
    ]
    status = (
        "pass"
        if (
            not pre_violations
            and not post_violations
            and proof_integrity.get("passed") is True
            and trace_invalid == 0
            and not prompt_derived_repair_tasks
        )
        else "fail"
    )
    return {
        "schema": "aura.dnu_leakage_report.v2",
        "generated_by": "tools/agi/run_dnu_agi_proof_battery.py::build_leakage_report",
        "status": status,
        "answer_leak_result": "pass" if not pre_violations else "fail",
        "salt_leak_result": "pass" if not pre_violations else "fail",
        "hidden_test_leak_result": "pass" if proof_integrity.get("passed") is True else "fail",
        "grader_leak_result": "pass" if not pre_violations else "fail",
        "canary_result": "pass" if not post_violations else "fail",
        "pre_check_violations": pre_violations,
        "post_check_violations": post_violations,
        "proof_integrity_lint": proof_integrity,
        "trace_invalid_lines": trace_invalid,
        "structured_solver_task_count": len(system2_symbolic_tasks),
        "structured_solver_tasks": system2_symbolic_tasks,
        "system2_symbolic_reasoner_task_count": len(system2_symbolic_tasks),
        "system2_symbolic_reasoner_tasks": system2_symbolic_tasks,
        "prompt_derived_repair_task_count": len(prompt_derived_repair_tasks),
        "prompt_derived_repair_tasks": prompt_derived_repair_tasks,
        "strict_symbolic_validation_task_count": len(strict_symbolic_validation_tasks),
    }


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def compute_scorecard(results: list[dict]) -> dict:
    """Compute scorecard from actual results. No synthetic scores."""
    scorecard = {
        "total_tasks": len(results),
        "total_pass": sum(1 for r in results if r["status"] == "pass"),
        "total_fail": sum(1 for r in results if r["status"] == "fail"),
        "total_timeout": sum(1 for r in results if r["status"] == "timeout"),
        "total_error": sum(1 for r in results if r["status"] == "error"),
        "total_no_answer": sum(1 for r in results if r["status"] == "no_answer"),
        "total_ungraded": sum(1 for r in results if r["status"] == "ungraded"),
        "categories": {},
    }

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"total": 0, "pass": 0, "fail": 0, "timeout": 0, "error": 0, "no_answer": 0}
        categories[cat]["total"] += 1
        if r["status"] in categories[cat]:
            categories[cat][r["status"]] += 1

    for cat, stats in categories.items():
        attempted = stats["total"]
        passed = stats["pass"]
        rate = passed / attempted if attempted > 0 else 0.0
        scorecard["categories"][cat] = {
            "attempted": attempted,
            "passed": passed,
            "failed": stats["fail"],
            "timed_out": stats["timeout"],
            "errors": stats["error"],
            "no_answer": stats["no_answer"],
            "pass_rate": round(rate, 4),
        }

    # Overall pass rate
    attempted = scorecard["total_tasks"]
    passed = scorecard["total_pass"]
    scorecard["overall_pass_rate"] = round(passed / attempted, 4) if attempted > 0 else 0.0

    return scorecard


def build_coverage_disclosure(scorecard: dict, unsupported_claims: list, all_tasks: list) -> dict:
    """State plainly what this run does and does not establish.

    The authoritative artifact reported total_tasks: 1, overall_pass_rate: 1.0
    and tier "Emergent (Capped)". Five of its six categories had zero tasks. A
    reader parsing the JSON sees a 100% pass rate and a tier label; the coverage
    limits were only reconstructible by cross-referencing unsupported_claims
    against category_summary and noticing the empty dicts.

    This block makes the negative space explicit, because a benchmark artifact
    is read by people looking for a number, and the number was 1.0.
    """
    attempted = int(scorecard.get("total_tasks", 0) or 0)
    empty = sorted(
        cat
        for cat in MINIMUM_COUNTS
        if int(scorecard.get("categories", {}).get(cat, {}).get("attempted", 0) or 0) == 0
    )
    return {
        "tasks_attempted": attempted,
        "categories_with_zero_tasks": empty,
        "coverage_sufficient": not unsupported_claims,
        "establishes": (
            f"Performance on {attempted} attempted task(s) from this repository's "
            "own fixture, under the conditions recorded in runtime_policy."
        ),
        "does_not_establish": [
            "any general capability claim — category coverage is incomplete"
            if unsupported_claims
            else "claims beyond the categories and counts actually attempted",
            "performance on HLE, GPQA, SWE-bench, ARC-AGI or GAIA: none of these "
            "were run, and this fixture is not a substitute for them",
            "a frontier comparison of any kind — the included fixture is dominated "
            "by approachable logic, arithmetic, Python-behaviour, scheduling, "
            "extraction and analogy problems",
        ],
        "fixture_provenance": "repository-local; authored alongside the system under test",
        "note": (
            "pass_rate is computed over ATTEMPTED tasks only. With a small "
            "attempted count it carries almost no information, and a rate of 1.0 "
            "on one task is not evidence of capability."
        ),
    }


def assign_tier(pass_rate: float, has_unsupported_claims: bool = False) -> dict:
    """Assign tier strictly from pass rate. No inflation. Max cap is Tier 5: Expert."""
    if pass_rate <= 0.0:
        base_tier = 0
        label = "No Capability"
    elif pass_rate <= 0.20:
        base_tier = 1
        label = "Minimal"
    elif pass_rate <= 0.40:
        base_tier = 2
        label = "Emergent"
    elif pass_rate <= 0.60:
        base_tier = 3
        label = "Competent"
    elif pass_rate <= 0.80:
        base_tier = 4
        label = "Proficient"
    else:
        # Cap strictly at Tier 5: Expert as a proof-bearing AGI-candidate architecture
        # to respect the scientific boundary of unproven metaphysical or AGI claims.
        base_tier = 5
        label = "Expert"

    if has_unsupported_claims and base_tier > 2:
        return {"tier": 2, "label": "Emergent (Capped)", "pass_rate": pass_rate}
    return {"tier": base_tier, "label": label, "pass_rate": pass_rate}


def generate_markdown_report(
    sys_info: dict,
    scorecard: dict,
    tier: dict,
    anti_theater: dict,
    results: list[dict],
    baselines: dict,
    ablations: dict,
) -> str:
    """Generate human-readable markdown report."""
    lines = []
    lines.append("# DNU AGI Proof Battery Report")
    lines.append("")
    lines.append(f"**Run ID:** `{sys_info['run_id']}`")
    lines.append(f"**Timestamp:** `{sys_info['timestamp']}`")
    lines.append(f"**Commit SHA:** `{sys_info['commit_sha']}`")
    lines.append(f"**Platform:** `{sys_info['platform']}`")
    lines.append(f"**Python:** `{sys_info['python_version']}`")
    lines.append("")

    # Tier
    lines.append("## Assigned Tier")
    lines.append("")
    lines.append(f"**Tier {tier['tier']}: {tier['label']}** (Overall Pass Rate: {tier['pass_rate']:.1%})")
    lines.append("")

    # Anti-Theater Status
    lines.append("## Anti-Theater Controls")
    lines.append("")
    pre = anti_theater.get("pre_check_violations", [])
    post = anti_theater.get("post_check_violations", [])
    if not pre and not post:
        lines.append("All anti-theater checks passed. No synthetic scores detected.")
    else:
        lines.append("**Anti-theater violations detected:**")
        for v in pre + post:
            lines.append(f"- {v}")
    lines.append("")

    # Scorecard
    lines.append("## Scorecard")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Tasks | {scorecard['total_tasks']} |")
    lines.append(f"| Passed | {scorecard['total_pass']} |")
    lines.append(f"| Failed | {scorecard['total_fail']} |")
    lines.append(f"| Timed Out | {scorecard['total_timeout']} |")
    lines.append(f"| Errors | {scorecard['total_error']} |")
    lines.append(f"| No Answer | {scorecard['total_no_answer']} |")
    lines.append(f"| **Overall Pass Rate** | **{scorecard['overall_pass_rate']:.1%}** |")
    lines.append("")

    # Per-category breakdown
    lines.append("## Category Breakdown")
    lines.append("")
    lines.append("| Category | Attempted | Passed | Failed | Timeout | Pass Rate |")
    lines.append("|----------|-----------|--------|--------|---------|-----------|")
    for cat, stats in sorted(scorecard["categories"].items()):
        lines.append(
            f"| {cat} | {stats['attempted']} | {stats['passed']} | "
            f"{stats['failed']} | {stats['timed_out']} | {stats['pass_rate']:.1%} |"
        )
    lines.append("")

    # Baselines
    lines.append("## Baselines")
    lines.append("")
    lines.append("| Baseline | Status | Pass Rate / Notes |")
    lines.append("|----------|--------|-------------------|")
    for name, data in baselines.items():
        status = data.get("status", "NOT_RUN")
        if status == "RUN":
            pr = data.get("pass_rate", 0.0)
            lines.append(f"| {name} | RUN | {pr:.1%} pass rate ({data.get('passed')}/{data.get('total_tasks')}) |")
        else:
            lines.append(f"| {name} | NOT_RUN | {data.get('reason', 'N/A')} |")
    lines.append("")

    # Ablations
    lines.append("## Ablations")
    lines.append("")
    lines.append("| Configuration | Status | Pass Rate / Notes |")
    lines.append("|---------------|--------|-------------------|")
    for name, data in ablations.items():
        status = data.get("status", "NOT_RUN")
        if status == "RUN":
            pr = data.get("pass_rate", 0.0)
            verified = "Yes" if data.get("lesion_run_verified", data.get("lesion_effect_verified", False)) else "No"
            # Keep full_aura as N/A since it has no lesion
            if name == "full_aura":
                verified = "N/A"
            scope = data.get("lesion_effect_verification_scope")
            scope_note = f"; effect scope: {scope}" if scope else ""
            lines.append(f"| {name} | RUN | {pr:.1%} pass rate (Lesion Run Verified: {verified}{scope_note}) |")
        else:
            lines.append(f"| {name} | NOT_RUN | {data.get('reason', 'N/A')} |")
    lines.append("")

    # Failed Tasks Sample
    failures = [r for r in results if r["status"] in ("fail", "error", "timeout", "no_answer")]
    if failures:
        lines.append("## Failed Tasks (Sample)")
        lines.append("")
        for f in failures[:20]:  # Cap at 20
            lines.append(f"- **{f['task_id']}** ({f['category']}): {f['status']} — {f.get('error', 'N/A')}")
        if len(failures) > 20:
            lines.append(f"- ... and {len(failures) - 20} more failures")
        lines.append("")

    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append(f"# Commit: {sys_info['commit_sha']}")
    lines.append(f"# Python: {sys_info['python_version']}")
    lines.append("python tools/agi/run_dnu_agi_proof_battery.py")
    lines.append("```")
    lines.append("")

    lines.append("---")
    lines.append("*This report was generated by the DNU AGI Proof Battery Runner.*")
    lines.append("*All scores are computed from actual task execution. No synthetic projections.*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="DNU AGI Proof Battery")
    parser.add_argument("--full", action="store_true", help="Run full battery")
    parser.add_argument("--out", default="", help="Output directory")
    parser.add_argument("--smoke", action="store_true", help="Smoke run")
    parser.add_argument(
        "--model-tier",
        choices=("primary", "tertiary"),
        default=None,
        help=(
            "Proof model lane. Defaults to primary/32B Cortex for acceptance; "
            "use tertiary/7B only for diagnostic fast-lane isolation."
        ),
    )
    parser.add_argument(
        "--stop-existing-runtime",
        action="store_true",
        help="Stop any already running Aura runtime before booting the exclusive proof profile.",
    )
    parser.add_argument(
        "--allow-coexisting-runtime",
        action="store_true",
        help="Diagnostic escape hatch only: allow proof boot while another Aura runtime is running.",
    )
    parser.add_argument(
        "--disable-structured-proof-solver",
        action="store_true",
        help=(
            "Diagnostic only: force strict proof tasks through the requested model "
            "lane instead of allowing Aura's governed System2 symbolic reasoner to "
            "answer exact-answer prompts."
        ),
    )
    parser.add_argument(
        "--enable-structured-proof-solver",
        action="store_true",
        help=(
            "Compatibility alias: allow Aura's governed System2 symbolic reasoner "
            "to answer exact strict proof prompts. Full proof runs enable this by "
            "default unless --disable-structured-proof-solver is set."
        ),
    )
    # ignore unknown args to prevent failing on extra options
    args, unknown = parser.parse_known_args()

    print("=" * 60)
    print("         DNU AGI PROOF BATTERY RUNNER")
    print("         No Synthetic Scores. No Theater.")
    print("=" * 60)

    run_id = str(uuid.uuid4())
    commit_sha = get_git_commit()
    os.environ.setdefault("AURA_PROOF_RUN", "1")
    env_system2_enabled = str(
        _FLAG_ENABLE_STRUCTURED_PROOF_SOLVER.value() or ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    structured_solver_enabled_for_run = bool(
        not args.disable_structured_proof_solver
        and (args.full or args.enable_structured_proof_solver or env_system2_enabled)
    )
    if structured_solver_enabled_for_run:
        os.environ["AURA_ENABLE_STRUCTURED_PROOF_SOLVER"] = "1"
        os.environ.pop("AURA_DISABLE_STRUCTURED_PROOF_SOLVER", None)
    else:
        os.environ.pop("AURA_ENABLE_STRUCTURED_PROOF_SOLVER", None)
        os.environ["AURA_DISABLE_STRUCTURED_PROOF_SOLVER"] = "1"
        os.environ.pop("AURA_DISABLE_MLX_STRICT_ANSWER_CONTRACT", None)
    requested_proof_model_tier = (
        args.model_tier or os.environ.get("AURA_PROOF_MODEL_TIER") or "primary"
    ).strip().lower()
    if requested_proof_model_tier not in {"primary", "tertiary"}:
        print(
            f"  [WARN] Invalid AURA_PROOF_MODEL_TIER={requested_proof_model_tier!r}; "
            "using primary."
        )
        requested_proof_model_tier = "primary"
    os.environ["AURA_PROOF_MODEL_TIER"] = requested_proof_model_tier
    proof_memory_envelope = configure_dnu_proof_memory_envelope(requested_proof_model_tier)

    if args.out:
        artifacts_base = Path(args.out)
    else:
        artifacts_base = Path(os.environ.get("AURA_ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts" / "agi_live")))

    run_dir = artifacts_base
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale_artifact in DNU_STALE_ARTIFACTS:
        try:
            (run_dir / stale_artifact).unlink(missing_ok=True)
        except OSError as exc:
            print(f"  [WARN] Could not remove stale artifact {stale_artifact}: {exc}")

    sys_info = {
        "run_id": run_id,
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit_sha": commit_sha,
        "python_version": sys.version,
        "platform": platform.platform(),
        "resource_observation": get_resource_observer().provenance.to_dict(),
        "proof_model_tier": requested_proof_model_tier,
        "proof_live_message_origin": PROOF_LIVE_MESSAGE_ORIGIN,
        "structured_proof_solver_enabled": structured_solver_enabled_for_run,
        "system2_symbolic_reasoner_enabled": structured_solver_enabled_for_run,
        "strict_answer_path": (
            "canonical_system2_symbolic_reasoner_then_model_lane"
            if structured_solver_enabled_for_run
            else "model_lane_only"
        ),
        "proof_memory_envelope": proof_memory_envelope,
    }

    print(f"Run ID: {run_id}")
    print(f"Commit SHA: {commit_sha}")
    print(f"Run Directory: {run_dir}")
    print(f"Proof Model Tier: {requested_proof_model_tier}")
    print(
        "Proof Memory Envelope: "
        f"process={proof_memory_envelope['process_rss_limit_gb']}GB, "
        f"mlx={proof_memory_envelope['mlx_memory_limit_gb']}GB, "
        f"worker={proof_memory_envelope['worker_rss_limit_gb']}GB, "
        f"cache_cap={proof_memory_envelope['metal_cache_cap_gb']}GB"
    )
    lifecycle_events = 0
    write_run_status(
        run_dir,
        status="running",
        run_id=run_id,
        commit_sha=commit_sha,
        phase="startup",
        lifecycle_events=lifecycle_events,
    )
    (run_dir / "RESOURCE_TRACE.jsonl").touch()
    (run_dir / "LIFECYCLE_EVENTS.jsonl").touch()

    def fail_run_status(
        *,
        phase: str,
        error: str,
        tasks_completed: int = 0,
        total_tasks: int | None = None,
    ) -> int:
        """Persist a terminal failed status before returning from a fatal gate."""
        write_run_status(
            run_dir,
            status="failed",
            run_id=run_id,
            commit_sha=commit_sha,
            phase=phase,
            tasks_completed=tasks_completed,
            total_tasks=total_tasks,
            error=error,
            lifecycle_events=lifecycle_events,
        )
        return 1

    # -----------------------------------------------------------------------
    # 0. Proof runtime exclusivity
    # -----------------------------------------------------------------------
    print("\n[0/8] Checking proof runtime exclusivity...")
    exclusive_report_path = run_dir / "EXCLUSIVE_RUNTIME_PREFLIGHT.json"
    resource_observer = get_resource_observer()
    try:
        existing_proof_runners = find_existing_proof_runners(
            observer=resource_observer
        )
    except RuntimeError as exc:
        write_exclusive_runtime_report(
            exclusive_report_path,
            status="process_table_observation_failed",
            instances=[],
        )
        print(f"  [FATAL] Cannot prove runtime exclusivity: {exc}")
        return fail_run_status(
            phase="runtime_exclusivity",
            error=f"process_table_observation_failed:{exc}",
        )
    if existing_proof_runners and args.stop_existing_runtime:
        try:
            remaining_proof_runners = stop_existing_proof_runners(
                observer=resource_observer
            )
        except RuntimeError as exc:
            return fail_run_status(
                phase="runtime_exclusivity",
                error=f"proof_runner_stop_observation_failed:{exc}",
            )
        if remaining_proof_runners:
            write_exclusive_runtime_report(
                exclusive_report_path,
                status="proof_runner_stop_failed",
                instances=remaining_proof_runners,
            )
            print("  [FATAL] Stale proof runner remained alive after stop request:")
            for instance in remaining_proof_runners:
                print(f"    PID {instance['pid']}: {instance['command'][:180]}")
            return fail_run_status(
                phase="runtime_exclusivity",
                error="proof_runner_stop_failed",
            )
    elif existing_proof_runners and not args.allow_coexisting_runtime:
        write_exclusive_runtime_report(
            exclusive_report_path,
            status="blocked_existing_proof_runner",
            instances=existing_proof_runners,
        )
        print("  [FATAL] Existing DNU proof runner detected. Proof runs require exclusivity.")
        print("          Re-run with --stop-existing-runtime to stop it first.")
        for instance in existing_proof_runners:
            print(f"    PID {instance['pid']}: {instance['command'][:180]}")
        return fail_run_status(
            phase="runtime_exclusivity",
            error="blocked_existing_proof_runner",
        )

    try:
        existing_runtimes = find_existing_aura_runtimes(observer=resource_observer)
    except RuntimeError as exc:
        return fail_run_status(
            phase="runtime_exclusivity",
            error=f"aura_runtime_observation_failed:{exc}",
        )
    if existing_runtimes and args.stop_existing_runtime:
        try:
            remaining = stop_existing_aura_runtimes(observer=resource_observer)
        except RuntimeError as exc:
            return fail_run_status(
                phase="runtime_exclusivity",
                error=f"aura_runtime_stop_observation_failed:{exc}",
            )
        exclusive_report = write_exclusive_runtime_report(
            exclusive_report_path,
            status="stopped_existing_runtime" if not remaining else "stop_failed",
            instances=remaining,
        )
        if remaining:
            print("  [FATAL] Existing Aura runtime remained alive after stop request:")
            for instance in remaining:
                print(f"    PID {instance['pid']}: {instance['command'][:180]}")
            return fail_run_status(
                phase="runtime_exclusivity",
                error="aura_runtime_stop_failed",
            )
    elif existing_runtimes and not args.allow_coexisting_runtime:
        exclusive_report = write_exclusive_runtime_report(
            exclusive_report_path,
            status="blocked_existing_runtime",
            instances=existing_runtimes,
        )
        print("  [FATAL] Existing Aura runtime detected. Proof runs require exclusive runtime.")
        print("          Re-run with --stop-existing-runtime to stop it first.")
        for instance in existing_runtimes:
            print(f"    PID {instance['pid']}: {instance['command'][:180]}")
        return fail_run_status(
            phase="runtime_exclusivity",
            error="blocked_existing_aura_runtime",
        )
    else:
        status = "coexisting_runtime_allowed" if existing_runtimes else "exclusive"
        exclusive_report = write_exclusive_runtime_report(
            exclusive_report_path,
            status=status,
            instances=existing_runtimes,
        )
        print(f"  [OK] Runtime exclusivity status: {status}.")
    try:
        reaped_orphans = stop_orphaned_aura_multiprocessing_children(
            observer=resource_observer
        )
    except RuntimeError as exc:
        return fail_run_status(
            phase="runtime_exclusivity",
            error=f"orphan_observation_failed:{exc}",
        )
    if reaped_orphans:
        exclusive_report["orphaned_multiprocessing_children_reaped"] = reaped_orphans
        exclusive_report_path.write_text(json.dumps(exclusive_report, indent=2), encoding="utf-8")
        print(
            "  [CLEANUP] Reaped "
            f"{len(reaped_orphans)} orphaned Aura multiprocessing helper(s) from prior interrupted runs."
        )
    if not args.allow_coexisting_runtime:
        from aura_main import bootstrap_lock

        bootstrap_lock(skip_lock=False)
        exclusive_report["canonical_runtime_lock_claimed_by_runner_pid"] = os.getpid()
        exclusive_report_path.write_text(json.dumps(exclusive_report, indent=2), encoding="utf-8")
        print("  [OK] Canonical runtime lock claimed for proof runner.")
    sys_info["exclusive_runtime_preflight"] = exclusive_report

    # -----------------------------------------------------------------------
    # 1. Load Task Packs
    # -----------------------------------------------------------------------
    print("\n[1/8] Loading sealed task packs...")
    fixture_dir = PROJECT_ROOT / "tests" / "agi" / "fixtures" / "dnu_tasks"
    if not fixture_dir.exists():
        print(f"  [FATAL] Fixture directory not found: {fixture_dir}")
        return fail_run_status(
            phase="load_task_packs",
            error=f"fixture_directory_not_found:{fixture_dir}",
        )

    all_tasks, grader_data = load_task_packs(fixture_dir)
    print(f"  Total tasks loaded: {len(all_tasks)}")
    print(f"  Grader entries loaded: {len(grader_data)}")

    if args.smoke:
        print("  [LIMIT] Smoke run enabled: limiting execution to first 1 task.")
        all_tasks = all_tasks[:1]
    else:
        max_tasks_env = _FLAG_AGI_MAX_TASKS.value()
        if max_tasks_env:
            try:
                max_tasks = int(max_tasks_env)
                print(f"  [LIMIT] Limiting execution to first {max_tasks} tasks (AURA_AGI_MAX_TASKS={max_tasks})")
                all_tasks = all_tasks[:max_tasks]
            except ValueError:
                print(f"  [WARN] Invalid AURA_AGI_MAX_TASKS value: {max_tasks_env}")

    if len(all_tasks) == 0:
        write_run_status(
            run_dir,
            status="failed",
            run_id=run_id,
            commit_sha=commit_sha,
            phase="load_task_packs",
            error="no_tasks_loaded",
            lifecycle_events=lifecycle_events,
        )
        print("  [FATAL] No tasks loaded. Cannot run battery.")
        return 1
    write_run_status(
        run_dir,
        status="running",
        run_id=run_id,
        commit_sha=commit_sha,
        phase="task_packs_loaded",
        total_tasks=len(all_tasks),
        lifecycle_events=lifecycle_events,
    )

    # -----------------------------------------------------------------------
    # 2. Anti-Theater Pre-Check
    # -----------------------------------------------------------------------
    print("\n[2/8] Running anti-theater pre-checks...")
    pre_violations = anti_theater_pre_check(all_tasks, grader_data)
    if pre_violations:
        for v in pre_violations:
            print(f"  [VIOLATION] {v}")
    else:
        print("  [OK] All pre-checks passed.")

    # -----------------------------------------------------------------------
    # 3. Boot canonical Aura runtime
    # -----------------------------------------------------------------------
    print("\n[3/8] Booting canonical AuraRuntime(profile='proof')...")

    from aura_main import boot_aura_runtime
    from core.container import ServiceContainer

    orch = await boot_aura_runtime(
        profile="proof",
        ready_label="Proof-DNU",
        readiness_context="dnu_agi_proof",
        artifact_root=PROJECT_ROOT / "artifacts" / "current",
    )
    print("  [OK] Canonical Aura runtime booted via aura_main.boot_aura_runtime.")

    router = ServiceContainer.get("llm_router", default=None)
    removed_ep = None
    if router is not None and hasattr(router, "endpoints"):
        # Keep the proof run on the same live runtime while preventing a second
        # heavyweight local lane from resident-coexisting with Cortex.
        removed_ep = router.endpoints.pop("Solver", None)
    if removed_ep:
        print("  [OK] Programmatically quarantined local heavyweight Solver (72B) endpoint to prevent timeouts.")

    engine = (
        ServiceContainer.get("cognitive_engine", default=None)
        or getattr(orch, "cognitive_engine", None)
        or getattr(orch, "cognition", None)
    )
    if engine is None:
        print("  [FATAL] Canonical boot completed without cognitive_engine.")
        return fail_run_status(
            phase="boot_runtime",
            error="canonical_boot_completed_without_cognitive_engine",
            total_tasks=len(all_tasks),
        )
    if hasattr(engine, "setup") and not getattr(engine, "_phases", None):
        engine.setup()
    print(f"  [OK] CognitiveEngine resolved from canonical boot. Lobotomized: {getattr(engine, 'lobotomized', False)}")
    print(f"  [OK] Phases loaded: {len(getattr(engine, '_phases', []))}")

    if getattr(engine, "lobotomized", False):
        print("  [FATAL] CognitiveEngine is lobotomized. Cannot run battery.")
        return fail_run_status(
            phase="boot_runtime",
            error="cognitive_engine_lobotomized",
            total_tasks=len(all_tasks),
        )

    runtime_manifest_src = PROJECT_ROOT / "artifacts" / "current" / "runtime_manifest.json"
    if not runtime_manifest_src.exists():
        print(f"  [FATAL] Canonical boot did not emit runtime manifest: {runtime_manifest_src}")
        return fail_run_status(
            phase="boot_runtime",
            error=f"runtime_manifest_missing:{runtime_manifest_src}",
            total_tasks=len(all_tasks),
        )
    runtime_manifest_copy = run_dir / "RUNTIME_MANIFEST.json"
    runtime_manifest_copy.write_text(runtime_manifest_src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"  [OK] Runtime manifest captured in {runtime_manifest_copy.name}.")
    runtime_policy = {
        "proof_model_tier": requested_proof_model_tier,
        "live_message_origin": PROOF_LIVE_MESSAGE_ORIGIN,
        "canonical_boot_profile": "proof",
        "strict_answer_tags_required": True,
        "structured_proof_solver_enabled": structured_solver_enabled_for_run,
        "system2_symbolic_reasoner_enabled": structured_solver_enabled_for_run,
        "strict_answer_path": (
            "canonical_system2_symbolic_reasoner_then_model_lane"
            if structured_solver_enabled_for_run
            else "model_lane_only"
        ),
        "model_only_strict_answer_diagnostic": not structured_solver_enabled_for_run,
        "exclusive_runtime_required": not args.allow_coexisting_runtime,
        "exclusive_runtime_preflight_status": exclusive_report.get("status"),
        "proof_memory_envelope": proof_memory_envelope,
        "model_recycle_interval_tasks": dnu_model_recycle_interval(
            requested_proof_model_tier,
            total_tasks=len(all_tasks),
            smoke=args.smoke,
        ),
        "comparisons_mode": "skipped_for_smoke" if args.smoke else "run",
    }
    allow_important_only_degraded = requested_proof_model_tier != "primary"
    runtime_policy["allow_important_only_degraded_after_requested_lane_probe"] = (
        allow_important_only_degraded
    )
    runtime_policy_path = run_dir / "RUNTIME_POLICY.json"
    runtime_policy_path.write_text(json.dumps(runtime_policy, indent=2), encoding="utf-8")
    print(f"  [OK] Runtime policy captured in {runtime_policy_path.name}.")

    print("  [PROBE] Exercising requested proof model lane before task execution...")
    model_lane_probe = await run_model_lane_probe(router, requested_proof_model_tier, run_dir)
    runtime_policy["model_lane_probe"] = model_lane_probe
    runtime_policy_path.write_text(json.dumps(runtime_policy, indent=2), encoding="utf-8")
    if not model_lane_probe.get("ok"):
        print(
            "  [FATAL] Requested proof model lane failed probe: "
            f"{model_lane_probe.get('error') or model_lane_probe.get('text_preview')}"
        )
        await shutdown_proof_runtime(orch)
        return fail_run_status(
            phase="model_lane_probe",
            error=str(
                model_lane_probe.get("error")
                or model_lane_probe.get("text_preview")
                or "model_lane_probe_failed"
            ),
            total_tasks=len(all_tasks),
        )
    print(
        "  [OK] Model lane probe passed via "
        f"{model_lane_probe.get('endpoint')} ({model_lane_probe.get('endpoint_tier')})."
    )
    initial_resource_snapshot, initial_health_blockers = await wait_for_proof_runtime_health(
        label="after_model_lane_probe",
        timeout_s=90.0 if requested_proof_model_tier == "primary" else 45.0,
        interval_s=2.0,
        allow_important_only_degraded=allow_important_only_degraded,
    )
    append_jsonl(run_dir / "RESOURCE_TRACE.jsonl", initial_resource_snapshot)
    if initial_health_blockers:
        print("  [FATAL] Proof runtime health failed after model lane probe:")
        for blocker in initial_health_blockers:
            print(f"    - {blocker}")
        await shutdown_proof_runtime(orch)
        return fail_run_status(
            phase="model_lane_probe",
            error="; ".join(initial_health_blockers),
            total_tasks=len(all_tasks),
        )

    # -----------------------------------------------------------------------
    # 4. Execute Tasks
    # -----------------------------------------------------------------------
    print(f"\n[4/8] Executing {len(all_tasks)} tasks through CognitiveEngine...")
    results = []
    trace_file = run_dir / "TASK_TRACE.jsonl"
    receipts_file = run_dir / "RECEIPTS.jsonl"
    resource_trace_file = run_dir / "RESOURCE_TRACE.jsonl"
    recycle_interval = dnu_model_recycle_interval(
        requested_proof_model_tier,
        total_tasks=len(all_tasks),
        smoke=args.smoke,
    )
    if recycle_interval:
        print(
            "  [LIFECYCLE] Primary proof model worker will be recycled every "
            f"{recycle_interval} tasks to bound long-run resident resource growth."
        )
    from core.will import get_will
    from tools.receipt_material import signed_will_receipt_entry

    will = get_will()
    await will.start()
    write_run_status(
        run_dir,
        status="running",
        run_id=run_id,
        commit_sha=commit_sha,
        phase="task_execution",
        total_tasks=len(all_tasks),
        lifecycle_events=lifecycle_events,
    )

    with trace_file.open("w", encoding="utf-8") as trace_fh, receipts_file.open("w", encoding="utf-8") as receipts_fh:
        for i, task in enumerate(all_tasks, 1):
            tid = task.get("task_id", "?")
            cat = task.get("category", "?")
            print(f"  [{i}/{len(all_tasks)}] {tid} ({cat})...", end=" ", flush=True)

            before_len = len(will._audit_trail) if will else 0
            pre_action_receipt_ids: set[str] = set()

            try:
                await isolate_live_runtime_for_dnu_task(task)
            except _DNU_RUN_RECOVERABLE_ERRORS as e:
                print(f"  [WARN] Failed to reset state for task isolation: {e}")

            if will:
                try:
                    from core.will import ActionDomain, WillOutcome

                    decision = will.decide(
                        content=task.get("task_prompt", ""),
                        source="dnu_agi_proof_battery",
                        domain=ActionDomain.RESPONSE,
                        priority=1.0,
                        is_critical=True,
                    )
                    pre_action_receipt_ids.add(decision.receipt_id)
                    if decision.outcome == WillOutcome.REFUSE:
                        result = {
                            "task_id": tid,
                            "category": task.get("category", "unknown"),
                            "difficulty": task.get("difficulty", "unknown"),
                            "status": "error",
                            "response_text": "",
                            "extracted_answer": None,
                            "normalized_answer": None,
                            "answer_hash": None,
                            "elapsed_s": 0.0,
                            "error": f"pre_action_authorization_refused:{decision.reason}",
                        }
                        results.append(result)
                        trace_fh.write(json.dumps(result, default=str) + "\n")
                        trace_fh.flush()
                        print(f"error ({result['error']})")
                        continue
                except _DNU_RUN_RECOVERABLE_ERRORS as ex:
                    result = {
                        "task_id": tid,
                        "category": task.get("category", "unknown"),
                        "difficulty": task.get("difficulty", "unknown"),
                        "status": "error",
                        "response_text": "",
                        "extracted_answer": None,
                        "normalized_answer": None,
                        "answer_hash": None,
                        "elapsed_s": 0.0,
                        "error": f"pre_action_authorization_failed:{type(ex).__name__}: {ex}",
                    }
                    results.append(result)
                    trace_fh.write(json.dumps(result, default=str) + "\n")
                    trace_fh.flush()
                    print(f"error ({result['error']})")
                    continue

            # Hard per-task wall-clock cap. A proof battery must NEVER hang
            # forever on a single wedge-prone task (observed: a strict-proof
            # task re-wedging on retry stalled the whole 100-task run at task
            # 16). execute_task's inner timeout is not a hard ceiling — its
            # internal retries/recycles can exceed it — so bound it from the
            # OUTSIDE. On breach: abandon the task as a RECORDED failure
            # (visible in the scorecard, never hidden), recycle the lane so the
            # wedge does not bleed into the next task, and continue.
            _task_budget_s = max(240, task.get("time_budget_s", 240))
            _task_wall_cap_s = float(
                _FLAG_DNU_TASK_WALL_CAP_S.value()
                or max(420.0, 1.5 * _task_budget_s)
            )
            try:
                result = await asyncio.wait_for(
                    execute_task(orch, task, timeout_s=_task_budget_s),
                    timeout=_task_wall_cap_s,
                )
            except TimeoutError:
                print(f"abandoned (task_wall_clock_exceeded {_task_wall_cap_s:.0f}s)")
                try:
                    await recycle_proof_model_lane(
                        router,
                        requested_proof_model_tier,
                        run_dir=run_dir,
                        reason="task_wall_clock_abandon",
                        task_index=i,
                    )
                except _DNU_RUN_RECOVERABLE_ERRORS as _recycle_exc:
                    print(f"  [WARN] lane recycle after task abandon failed: {type(_recycle_exc).__name__}")
                result = {
                    "task_id": tid,
                    "category": task.get("category", "unknown"),
                    "difficulty": task.get("difficulty", "unknown"),
                    "status": "error",
                    "response_text": "",
                    "extracted_answer": None,
                    "normalized_answer": None,
                    "answer_hash": None,
                    "elapsed_s": _task_wall_cap_s,
                    "error": "task_wall_clock_exceeded",
                }
                results.append(result)
                trace_fh.write(json.dumps(result, default=str) + "\n")
                trace_fh.flush()
                continue

            result = grade_result(result, grader_data)
            results.append(result)

            # Write trace
            trace_fh.write(json.dumps(result, default=str) + "\n")
            trace_fh.flush()

            # Record receipts
            if will:
                new_decisions = list(will._audit_trail)[before_len:]
                for d in new_decisions:
                    domain_val = d.domain.value if hasattr(d.domain, "value") else str(d.domain)
                    outcome_val = d.outcome.value if hasattr(d.outcome, "value") else str(d.outcome)
                    is_pre_action = d.receipt_id in pre_action_receipt_ids
                    effect_verified = result["status"] in {"pass", "fail", "no_answer"}
                    telemetry_logged = True
                    closure_ok = (
                        will.verify_closure(
                            d.receipt_id,
                            effect_verified=effect_verified,
                            telemetry_logged=telemetry_logged,
                        )
                        if is_pre_action
                        else False
                    )
                    vol_hash = hashlib.sha256(f"{tid}:{d.receipt_id}:{domain_val}:{outcome_val}:{d.reason}".encode()).hexdigest()
                    receipt_entry = signed_will_receipt_entry(
                        will,
                        d,
                        task_id=tid,
                        domain=domain_val,
                        outcome=outcome_val,
                        reason=d.reason,
                        extra={
                            "source": getattr(d, "source", ""),
                            "volition_hash": vol_hash,
                            "authorization_phase": "pre_action" if is_pre_action else "internal_runtime",
                            "effect_verified": effect_verified,
                            "telemetry_logged": telemetry_logged,
                            "closure_verified": closure_ok,
                        },
                    )
                    receipts_fh.write(json.dumps(receipt_entry, default=str) + "\n")
                receipts_fh.flush()

            status_icon = {
                "pass": "✓",
                "fail": "✗",
                "timeout": "T",
                "error": "!",
                "no_answer": "∅",
                "ungraded": "?",
            }.get(result["status"], "?")
            print(f"{status_icon} {result['status']} ({result['elapsed_s']:.1f}s)")
            task_resource_snapshot = collect_proof_resource_snapshot(
                label="after_task",
                task_index=i,
                task_id=tid,
            )
            append_jsonl(resource_trace_file, task_resource_snapshot)
            task_health_blockers = proof_runtime_health_blockers(
                task_resource_snapshot,
                allow_important_only_degraded=allow_important_only_degraded,
            )
            if task_health_blockers:
                # Give a TRANSIENT runtime degradation its designed recovery
                # window before failing the entire battery. A single hard/empty
                # generation can trip the Cortex circuit breaker
                # (client_returned_no_text) → inference probe momentarily
                # critical; the circuit is built to go half-open→closed and the
                # lane re-arms. An unanswerable hard task is a per-task failure
                # (already graded), not a dead runtime. Only FATAL if health
                # stays critical after the recovery window.
                recovery_snapshot, task_health_blockers = await wait_for_proof_runtime_health(
                    label="post_task_health_recovery",
                    task_index=i,
                    task_id=tid,
                    timeout_s=float(
                        _FLAG_DNU_POST_TASK_HEALTH_RECOVERY_S.value()
                        or 90.0
                    ),
                    allow_important_only_degraded=allow_important_only_degraded,
                )
                if isinstance(recovery_snapshot, dict) and recovery_snapshot:
                    append_jsonl(resource_trace_file, recovery_snapshot)
            if task_health_blockers:
                print("  [FATAL] Proof runtime health failed during task execution (after recovery window):")
                for blocker in task_health_blockers:
                    print(f"    - {blocker}")
                await shutdown_proof_runtime(orch)
                return fail_run_status(
                    phase="task_execution",
                    error="; ".join(task_health_blockers),
                    tasks_completed=len(results),
                    total_tasks=len(all_tasks),
                )
            write_run_status(
                run_dir,
                status="running",
                run_id=run_id,
                commit_sha=commit_sha,
                phase="task_execution",
                tasks_completed=len(results),
                total_tasks=len(all_tasks),
                lifecycle_events=lifecycle_events,
            )
            if recycle_interval and i < len(all_tasks) and i % recycle_interval == 0:
                recycle_event = await recycle_proof_model_lane(
                    router,
                    requested_proof_model_tier,
                    run_dir=run_dir,
                    reason=f"dnu_checkpoint_after_task_{i}",
                    task_index=i,
                )
                lifecycle_events += 1
                write_run_status(
                    run_dir,
                    status="running",
                    run_id=run_id,
                    commit_sha=commit_sha,
                    phase="model_lane_recycle",
                    tasks_completed=len(results),
                    total_tasks=len(all_tasks),
                    error=recycle_event.get("error"),
                    lifecycle_events=lifecycle_events,
                )
                if recycle_event.get("status") != "complete":
                    print(
                        "  [FATAL] Proof model lane recycle failed: "
                        f"{recycle_event.get('error')}"
                    )
                    await shutdown_proof_runtime(orch)
                    return fail_run_status(
                        phase="model_lane_recycle",
                        error=str(recycle_event.get("error") or "model_lane_recycle_failed"),
                        tasks_completed=len(results),
                        total_tasks=len(all_tasks),
                    )
                recycle_health_blockers = proof_runtime_health_blockers(
                    recycle_event.get("after") or {},
                    allow_important_only_degraded=allow_important_only_degraded,
                )
                if recycle_health_blockers:
                    print("  [FATAL] Proof runtime health failed after model lane recycle:")
                    for blocker in recycle_health_blockers:
                        print(f"    - {blocker}")
                    await shutdown_proof_runtime(orch)
                    return fail_run_status(
                        phase="model_lane_recycle",
                        error="; ".join(recycle_health_blockers),
                        tasks_completed=len(results),
                        total_tasks=len(all_tasks),
                    )

    # -----------------------------------------------------------------------
    # 5. Anti-Theater Post-Check
    # -----------------------------------------------------------------------
    print("\n[5/8] Running anti-theater post-checks...")
    write_run_status(
        run_dir,
        status="running",
        run_id=run_id,
        commit_sha=commit_sha,
        phase="anti_theater_post_check",
        tasks_completed=len(results),
        total_tasks=len(all_tasks),
        lifecycle_events=lifecycle_events,
    )
    post_violations = anti_theater_post_check(results)
    if post_violations:
        for v in post_violations:
            print(f"  [VIOLATION] {v}")
    else:
        print("  [OK] All post-checks passed.")

    anti_theater = {
        "pre_check_violations": pre_violations,
        "post_check_violations": post_violations,
        "all_passed": len(pre_violations) == 0 and len(post_violations) == 0,
    }

    # -----------------------------------------------------------------------
    # 6. Compute Scorecard and Enforce Tier-Capping
    # -----------------------------------------------------------------------
    print("\n[6/8] Computing scorecard from actual results...")
    write_run_status(
        run_dir,
        status="running",
        run_id=run_id,
        commit_sha=commit_sha,
        phase="scorecard",
        tasks_completed=len(results),
        total_tasks=len(all_tasks),
        lifecycle_events=lifecycle_events,
    )
    scorecard = compute_scorecard(results)

    # Check minimum task counts and record violations
    unsupported_claims = []
    for cat, min_count in MINIMUM_COUNTS.items():
        actual = scorecard["categories"].get(cat, {}).get("attempted", 0)
        if actual < min_count:
            unsupported_claims.append(
                f"Category '{cat}' has {actual} tasks, below minimum of {min_count}"
            )

    tier = assign_tier(scorecard["overall_pass_rate"], has_unsupported_claims=len(unsupported_claims) > 0)

    print(f"  Overall Pass Rate: {scorecard['overall_pass_rate']:.1%}")
    print(f"  Assigned Tier: {tier['tier']} ({tier['label']})")
    for cat, stats in sorted(scorecard["categories"].items()):
        print(f"  {cat}: {stats['passed']}/{stats['attempted']} ({stats['pass_rate']:.1%})")

    # -----------------------------------------------------------------------
    # 6.5. Run Baselines & Ablations
    # -----------------------------------------------------------------------
    if args.smoke:
        print("\n[SMOKE] Skipping baseline and ablation comparisons.")
        print("  Smoke verifies boot, model lane, live message path, grading, and artifacts only.")
        baselines = {
            "raw_llm": {"status": "SKIPPED_SMOKE", "reason": "smoke runs avoid expensive comparison loops"},
            "llm_with_tools": {"status": "SKIPPED_SMOKE", "reason": "smoke runs avoid expensive comparison loops"},
            "react_agent": {"status": "SKIPPED_SMOKE", "reason": "smoke runs avoid expensive comparison loops"},
        }
        full_aura_comparison_rate = scorecard["overall_pass_rate"]
        ablations = {
            "full_aura": {"status": "RUN", "pass_rate": full_aura_comparison_rate},
            "no_persistent_memory": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
            "no_volition": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
            "no_will_authority": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
            "no_system2": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
            "no_self_repair": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
            "no_affect_steering": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
            # Compatibility aliases for historical report consumers.
            "aura_minus_memory": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
            "aura_minus_volition": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
            "aura_minus_will": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
            "aura_minus_system2": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
            "aura_minus_self_repair": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
            "aura_minus_affect_steering": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
        }
    else:
        print("\nRunning raw LLM, LLM with direct tools, and ReAct agent baselines...")
        write_run_status(
            run_dir,
            status="running",
            run_id=run_id,
            commit_sha=commit_sha,
            phase="baselines",
            tasks_completed=len(results),
            total_tasks=len(all_tasks),
            lifecycle_events=lifecycle_events,
        )
        # Cap tasks for baseline and ablation comparisons, but keep the
        # subset category-balanced instead of relying on fixture load order.
        comparison_limit = _comparison_task_limit(default=12)
        comparison_tasks = select_stratified_comparison_tasks(all_tasks, comparison_limit)
        comparison_categories = Counter(str(t.get("category", "unknown")) for t in comparison_tasks)
        print(
            f"  Using {len(comparison_tasks)} stratified tasks for representative comparisons: "
            f"{dict(comparison_categories)}"
        )

        sem = asyncio.Semaphore(1)
        raw_llm_tasks = [execute_raw_llm_task(router, task, grader_data, sem) for task in comparison_tasks]
        llm_with_tools_tasks = [execute_llm_with_tools_task(router, task, grader_data, sem) for task in comparison_tasks]
        react_tasks = [execute_react_task(router, task, grader_data, sem) for task in comparison_tasks]

        raw_llm_results = await asyncio.gather(*raw_llm_tasks)
        llm_with_tools_results = await asyncio.gather(*llm_with_tools_tasks)
        react_results = await asyncio.gather(*react_tasks)
        baseline_resource_snapshot, baseline_health_blockers = await wait_for_proof_runtime_health(
            label="after_baselines",
            timeout_s=60.0,
            interval_s=2.0,
            allow_important_only_degraded=allow_important_only_degraded,
        )
        append_jsonl(resource_trace_file, baseline_resource_snapshot)
        if baseline_health_blockers:
            print("  [FATAL] Proof runtime health failed after baseline comparison:")
            for blocker in baseline_health_blockers:
                print(f"    - {blocker}")
            write_run_status(
                run_dir,
                status="failed",
                run_id=run_id,
                commit_sha=commit_sha,
                phase="baselines",
                tasks_completed=len(results),
                total_tasks=len(all_tasks),
                error="; ".join(baseline_health_blockers),
                lifecycle_events=lifecycle_events,
            )
            await shutdown_proof_runtime(orch)
            return 1

        raw_llm_scorecard = compute_scorecard(raw_llm_results)
        llm_with_tools_scorecard = compute_scorecard(llm_with_tools_results)
        react_scorecard = compute_scorecard(react_results)

        baselines = {
            "raw_llm": {
                "status": "RUN",
                "pass_rate": raw_llm_scorecard["overall_pass_rate"],
                "total_tasks": len(comparison_tasks),
                "passed": raw_llm_scorecard["total_pass"],
                "sample_categories": dict(comparison_categories),
            },
            "llm_with_tools": {
                "status": "RUN",
                "pass_rate": llm_with_tools_scorecard["overall_pass_rate"],
                "total_tasks": len(comparison_tasks),
                "passed": llm_with_tools_scorecard["total_pass"],
                "sample_categories": dict(comparison_categories),
            },
            "react_agent": {
                "status": "RUN",
                "pass_rate": react_scorecard["overall_pass_rate"],
                "total_tasks": len(comparison_tasks),
                "passed": react_scorecard["total_pass"],
                "sample_categories": dict(comparison_categories),
            },
        }

        print("\nRunning dynamic system ablations sequentially...")
        write_run_status(
            run_dir,
            status="running",
            run_id=run_id,
            commit_sha=commit_sha,
            phase="ablations",
            tasks_completed=len(results),
            total_tasks=len(all_tasks),
            lifecycle_events=lifecycle_events,
        )

        # Compute full_aura pass rate on the exact same comparison subset for
        # honest baseline/ablation comparison.
        results_by_task_id = {str(r.get("task_id", "")): r for r in results}
        full_aura_comparison_results = [
            results_by_task_id[str(task.get("task_id", ""))]
            for task in comparison_tasks
            if str(task.get("task_id", "")) in results_by_task_id
        ]
        full_aura_comparison_scorecard = compute_scorecard(full_aura_comparison_results)
        full_aura_comparison_rate = full_aura_comparison_scorecard["overall_pass_rate"]

        print("  Running ablation: no_persistent_memory...")
        no_persistent_memory = await run_ablation_suite(
            orch,
            comparison_tasks,
            grader_data,
            ["memory_facade", "memory_coordinator"],
            ablation_name="no_persistent_memory",
            sample_categories=comparison_categories,
        )

        print("  Running ablation: no_volition...")
        no_volition = await run_ablation_suite(
            orch,
            comparison_tasks,
            grader_data,
            ["volition_engine"],
            ablation_name="no_volition",
            sample_categories=comparison_categories,
        )

        print("  Running ablation: no_will_authority...")
        no_will_authority = await run_ablation_suite(
            orch,
            comparison_tasks,
            grader_data,
            ["unified_will"],
            ablation_name="no_will_authority",
            sample_categories=comparison_categories,
        )

        print("  Running ablation: no_system2...")
        no_system2 = await run_ablation_suite(
            orch,
            comparison_tasks,
            grader_data,
            ["native_system2"],
            ablation_name="no_system2",
            sample_categories=comparison_categories,
        )

        print("  Running ablation: no_self_repair...")
        no_self_repair = await run_ablation_suite(
            orch,
            comparison_tasks,
            grader_data,
            ["self_repair", "skill_library"],
            ablation_name="no_self_repair",
            sample_categories=comparison_categories,
        )

        print("  Running ablation: no_affect_steering...")
        no_affect_steering = await run_ablation_suite(
            orch,
            comparison_tasks,
            grader_data,
            ["affective_steering_engine", "affect_engine", "affect_facade"],
            ablation_name="no_affect_steering",
            sample_categories=comparison_categories,
        )

        ablations = {
            "full_aura": {"status": "RUN", "pass_rate": full_aura_comparison_rate, "sample_categories": dict(comparison_categories)},
            "no_persistent_memory": no_persistent_memory,
            "no_volition": no_volition,
            "no_will_authority": no_will_authority,
            "no_system2": no_system2,
            "no_self_repair": no_self_repair,
            "no_affect_steering": no_affect_steering,
            # Compatibility aliases for historical report consumers.
            "aura_minus_memory": dict(no_persistent_memory),
            "aura_minus_volition": dict(no_volition),
            "aura_minus_will": dict(no_will_authority),
            "aura_minus_system2": dict(no_system2),
            "aura_minus_self_repair": dict(no_self_repair),
            "aura_minus_affect_steering": dict(no_affect_steering),
        }
        ablation_resource_snapshot, ablation_health_blockers = await wait_for_proof_runtime_health(
            label="after_ablations",
            timeout_s=60.0,
            interval_s=2.0,
            allow_important_only_degraded=allow_important_only_degraded,
        )
        append_jsonl(resource_trace_file, ablation_resource_snapshot)
        if ablation_health_blockers:
            print("  [FATAL] Proof runtime health failed after ablation recovery:")
            for blocker in ablation_health_blockers:
                print(f"    - {blocker}")
            write_run_status(
                run_dir,
                status="failed",
                run_id=run_id,
                commit_sha=commit_sha,
                phase="ablations",
                tasks_completed=len(results),
                total_tasks=len(all_tasks),
                error="; ".join(ablation_health_blockers),
                lifecycle_events=lifecycle_events,
            )
            await shutdown_proof_runtime(orch)
            return 1

    # -----------------------------------------------------------------------
    # 7. Write Artifacts
    # -----------------------------------------------------------------------
    print("\n[7/8] Writing artifacts...")
    write_run_status(
        run_dir,
        status="running",
        run_id=run_id,
        commit_sha=commit_sha,
        phase="write_artifacts",
        tasks_completed=len(results),
        total_tasks=len(all_tasks),
        lifecycle_events=lifecycle_events,
    )

    # Granular verification checklist
    baselines_complete = (
        all(b.get("status") == "RUN" for b in baselines.values())
        if not args.smoke
        else all(b.get("status") == "SKIPPED_SMOKE" for b in baselines.values())
    )
    ablation_entries = [a for name, a in ablations.items() if name != "full_aura"]
    ablations_verified = (
        all(
            a.get("status") == "RUN"
            and a.get("lesion_run_verified", False)
            and a.get("lesion_effect_verification_scope") in {
                "dnu_score_delta",
                "delegated_to_dedicated_cert_chain",
            }
            for a in ablation_entries
        )
        if not args.smoke
        else all(a.get("status") == "SKIPPED_SMOKE" for a in ablation_entries)
    )
    
    # Category coverage is a threshold, and an ABSENT category is the worst
    # possible score on it — not an exemption from it.
    #
    # This loop used to iterate `scorecard["categories"]` looking for "transfer"
    # and checking its pass_rate. Categories with zero attempted tasks never
    # appear in that dict, so on a run that attempted ONE novel_reasoning task
    # the loop checked nothing at all and left the flag True. The published
    # artifact then said category_thresholds_passed: True and passed: True while
    # its own unsupported_claims listed all six categories as below minimum —
    # including five with zero tasks. A check that cannot fail on the data that
    # should fail it is not a check.
    #
    # unsupported_claims is computed above from MINIMUM_COUNTS and is the
    # authority: if any category is under its floor, coverage has not passed.
    category_thresholds_passed = not unsupported_claims
    for cat_name, cat_stats in scorecard["categories"].items():
        if "transfer" in cat_name.lower():
            if cat_stats.get("pass_rate", 0.0) < 0.75:
                category_thresholds_passed = False
                
    gov_report = build_governance_report(receipts_file, expected_tasks=len(all_tasks))
    leakage_report = build_leakage_report(
        pre_violations=pre_violations,
        post_violations=post_violations,
        run_dir=run_dir,
    )
    structured_solver_task_count = int(leakage_report.get("structured_solver_task_count", 0) or 0)
    system2_symbolic_task_count = int(
        leakage_report.get("system2_symbolic_reasoner_task_count", structured_solver_task_count)
        or 0
    )
    system2_symbolic_provenance_reported = (
        system2_symbolic_task_count == 0
        or (
            structured_solver_enabled_for_run
            and leakage_report.get("status") == "pass"
            and bool(leakage_report.get("system2_symbolic_reasoner_tasks") is not None)
        )
    )

    verification_checklist = {
        "runner_completed": True,
        "score_threshold_passed": scorecard["overall_pass_rate"] >= 0.85,
        "category_thresholds_passed": category_thresholds_passed,
        "baselines_complete": baselines_complete,
        "ablations_verified": ablations_verified,
        "governance_receipts_verified": gov_report.get("status") == "pass",
        "leakage_checks_passed": leakage_report.get("status") == "pass",
        "model_lane_probe_passed": bool(model_lane_probe.get("ok")),
        "answer_path_provenance_reported": system2_symbolic_provenance_reported,
    }

    # Main proof bundle
    proof_bundle = {
        "system_info": sys_info,
        "scorecard": scorecard,
        "tier": tier,
        "anti_theater": anti_theater,
        "baselines": baselines,
        "ablations": ablations,
        "task_count": len(all_tasks),
        "grader_entry_count": len(grader_data),
        "category_summary": {cat: scorecard["categories"].get(cat, {}) for cat in DIR_TO_CAT.values()},
        "unsupported_claims": unsupported_claims,
        "coverage_disclosure": build_coverage_disclosure(scorecard, unsupported_claims, all_tasks),
        "verification_checklist": verification_checklist,
        "runtime_policy": runtime_policy,
        "model_lane_probe": model_lane_probe,
        "structured_solver_task_count": structured_solver_task_count,
        "system2_symbolic_reasoner_task_count": system2_symbolic_task_count,
        "system2_symbolic_reasoner_used": system2_symbolic_task_count > 0,
        "model_lane_answered_without_structured_solver": structured_solver_task_count == 0,
        "lifecycle_event_count": lifecycle_events,
        "resource_trace": {
            "path": str((run_dir / "RESOURCE_TRACE.jsonl").relative_to(PROJECT_ROOT))
            if (run_dir / "RESOURCE_TRACE.jsonl").is_relative_to(PROJECT_ROOT)
            else str(run_dir / "RESOURCE_TRACE.jsonl"),
        },
        "governance_report_status": gov_report.get("status"),
        "leakage_report_status": leakage_report.get("status"),
        "passed": all(verification_checklist.values()) and anti_theater["all_passed"],
    }

    # Write DNU_AGI_PROOF.json
    proof_path = run_dir / "DNU_AGI_PROOF.json"
    proof_path.write_text(json.dumps(proof_bundle, indent=2, default=str), encoding="utf-8")
    print(f"  [OK] {proof_path.name}")

    # Write SCORECARD.json
    scorecard_path = run_dir / "SCORECARD.json"
    scorecard_path.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    print(f"  [OK] {scorecard_path.name}")

    # Write BASELINES.json
    #
    # The budget the arms actually ran under travels WITH the numbers. The
    # retracted bundle recorded three baselines at exactly 0.1667 and nothing
    # beside them saying that they had been capped at 160 tokens while
    # full_aura ran a 240s live path with a symbolic solver. Three structurally
    # different systems agreeing to four decimal places is the signature of a
    # shared handicap, and the artifact contained no field in which that could
    # have been noticed. The audit that caught it was a human getting
    # suspicious months later.
    baselines = dict(baselines)
    baselines["_budget_parity"] = _dnu_budget_parity_report()
    baselines_path = run_dir / "BASELINES.json"
    baselines_path.write_text(json.dumps(baselines, indent=2), encoding="utf-8")
    print(f"  [OK] {baselines_path.name}")
    parity = baselines["_budget_parity"]
    if not parity.get("matched", False):
        print(f"  [PARITY] COMPARISON VOID: {parity.get('refusal_reason', '')}")
    else:
        print(f"  [PARITY] arms matched: {parity.get('summary', '')}")

    # Write ABLATIONS.json
    ablations_path = run_dir / "ABLATIONS.json"
    ablations_path.write_text(json.dumps(ablations, indent=2), encoding="utf-8")
    print(f"  [OK] {ablations_path.name}")

    # Write FAILURES.jsonl
    failures_path = run_dir / "FAILURES.jsonl"
    with failures_path.open("w", encoding="utf-8") as f:
        for r in results:
            if r["status"] != "pass":
                f.write(json.dumps(r, default=str) + "\n")
    print(f"  [OK] {failures_path.name}")

    # Write markdown report
    md_content = generate_markdown_report(sys_info, scorecard, tier, anti_theater, results, baselines, ablations)
    md_path = run_dir / "DNU_AGI_PROOF.md"
    md_path.write_text(md_content, encoding="utf-8")
    print(f"  [OK] {md_path.name}")

    # Write REPRODUCTION.md
    repro_content = f"""# Reproduction Instructions

## Environment
- **Commit SHA:** `{commit_sha}`
- **Python Version:** `{sys_info['python_version']}`
- **Platform:** `{sys_info['platform']}`
- **Run ID:** `{run_id}`

## Prerequisites
- Aura source code at the specified commit
- LLM model server running (check port configuration)
- Python environment with all dependencies

## Commands
```bash
cd /path/to/aura-source
git checkout {commit_sha}
python tools/agi/run_dnu_agi_proof_battery.py
```

## Verification
```bash
python -m pytest tests/agi/live/test_dnu_agi_proof_battery.py -q
```

## Notes
- Task fixtures are sealed under `tests/agi/fixtures/dnu_tasks/`
- Grader salts are in `.grader_salts*.json` files (not task packs)
- Results depend on model server availability and response quality
- Different model versions will produce different results
"""
    repro_path = run_dir / "REPRODUCTION.md"
    repro_path.write_text(repro_content, encoding="utf-8")
    print(f"  [OK] {repro_path.name}")

    # Write GOVERNANCE_REPORT.json
    gov_report_path = run_dir / "GOVERNANCE_REPORT.json"
    gov_report_path.write_text(json.dumps(gov_report, indent=2), encoding="utf-8")
    print(f"  [OK] {gov_report_path.name}")

    # Write LEAKAGE_REPORT.json
    leakage_report_path = run_dir / "LEAKAGE_REPORT.json"
    leakage_report_path.write_text(json.dumps(leakage_report, indent=2), encoding="utf-8")
    print(f"  [OK] {leakage_report_path.name}")

    # Write FINAL_VERDICT.txt
    verdict_text = "DNU AGI NOT PROVEN"
    if tier["tier"] == 6 and len(unsupported_claims) == 0:
        verdict_text = "DNU AGI PROVEN"
    
    verdict_path = run_dir / "FINAL_VERDICT.txt"
    verdict_path.write_text(verdict_text, encoding="utf-8")
    print(f"  [OK] {verdict_path.name}")
    write_run_status(
        run_dir,
        status="complete",
        run_id=run_id,
        commit_sha=commit_sha,
        phase="complete",
        tasks_completed=len(results),
        total_tasks=len(all_tasks),
        lifecycle_events=lifecycle_events,
    )

    # -----------------------------------------------------------------------
    # 8. Write Manifest
    # -----------------------------------------------------------------------
    print("\n[8/8] Writing manifest...")
    write_artifact_manifest(
        run_dir,
        run_id=run_id,
        commit_sha=commit_sha,
        timestamp=sys_info["timestamp"],
    )
    print("  [OK] MANIFEST.json")

    # Also copy key artifacts to the standard agi_live directory for pytest
    std_dest = artifacts_base
    std_dest.mkdir(parents=True, exist_ok=True)
    for fname in DNU_STANDARD_COPY_ARTIFACTS:
        src = run_dir / fname
        if src.exists():
            (std_dest / fname).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # Recompute manifest for the standard location
    write_artifact_manifest(
        std_dest,
        run_id=run_id,
        commit_sha=commit_sha,
        include_files=DNU_STANDARD_COPY_ARTIFACTS,
    )

    # -----------------------------------------------------------------------
    # Final Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  DNU AGI PROOF BATTERY: COMPLETE")
    print(f"  Tasks Executed: {scorecard['total_tasks']}")
    print(f"  Overall Pass Rate: {scorecard['overall_pass_rate']:.1%}")
    print(f"  Assigned Tier: {tier['tier']} ({tier['label']})")
    print(f"  Anti-Theater: {'CLEAN' if anti_theater['all_passed'] else 'VIOLATIONS DETECTED'}")
    print("=" * 60)

    if not anti_theater["all_passed"]:
        print("\n[!] Anti-theater violations detected. Review report.")
        await shutdown_proof_runtime(orch)
        return fail_run_status(
            phase="anti_theater_post_check",
            error="anti_theater_violations_detected",
            tasks_completed=len(results),
            total_tasks=len(all_tasks),
        )

    if args.smoke and scorecard.get("total_pass", 0) != scorecard.get("total_tasks", 0):
        print("\n[!] Smoke run failed: live task path did not pass. Review FAILURES.jsonl.")
        await shutdown_proof_runtime(orch)
        return fail_run_status(
            phase="smoke_live_task_path",
            error="smoke_live_task_path_failed",
            tasks_completed=len(results),
            total_tasks=len(all_tasks),
        )

    await shutdown_proof_runtime(orch)
    print("\n[+] DNU AGI Proof Battery: COMPLETE")
    return 0


if __name__ == "__main__":
    code = asyncio.run(main())
    _reap_proof_child_processes_sync("process_exit")
    sys.stdout.flush()
    sys.stderr.flush()
    # Avoid hanging in Py_FinalizeEx on non-daemon helper/native threads after
    # artifacts are complete and child processes have been reaped.
    os._exit(int(code))
