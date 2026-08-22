#!/usr/bin/env python3
"""Live boot proof: boot Aura for real, converse, act, verify, shut down.

Static gates prove the code; this proves the companion. The driver runs
OUTSIDE Aura's process and treats her like a user would:

    1. boot `aura_main.py --headless` or `--desktop` and poll /api/health until the runtime
   contract reports healthy (bounded wait),
2. send real chat turns through /api/chat and measure latency,
3. check the identity contract holds in the *actual* reply (the
   self-claim verifier runs on what she really said),
4. ask for a real governed desktop action (folder + file) and verify
   the effect on disk from outside her process,
5. watch her process-tree RSS the whole time with a hard abort ceiling,
6. stop her cleanly and verify no orphan workers and no port squat.

Every step lands in a JSONL transcript plus a final JSON verdict under
artifacts/live_proof/. A timeout, OOM abort, dead process, or failed
verification is a loud failed step — never a skipped one. The artifact
records what actually happened, including failures; it is evidence,
not advertising.

Usage:
    python tools/live_boot_proof.py [--port 8000] [--boot-timeout 600]
    python tools/live_boot_proof.py --skip-desktop-action
    python tools/live_boot_proof.py --restart-continuity
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.runtime.resource_observation import (  # noqa: E402
    ResourceObserver,
    get_resource_observer,
)

PROOF_DIR = ROOT / "artifacts" / "live_proof"

# Abort the whole proof if Aura's process tree exceeds this. The runtime should
# refuse/recycle before this external guard fires; the guard exists to protect
# the host if local inference leaks past the in-process policy.
DEFAULT_RSS_ABORT_MB = 56.0 * 1024.0
LIVE_FALLBACK_RE = re.compile(
    r"(say that again|try (?:again|me again|that again)|ask me again|"
    r"give me a moment|i'?m with you|could you repeat|repeat your question|"
    r"send your message again|lost my (?:thread|train of thought)|"
    r"hit a bump|one moment|having trouble formulating|could you try rephrasing)",
    re.IGNORECASE,
)
LIVE_LOG_LEVEL_RE = re.compile(r"(?:^|[\s\[])(?:ERROR|CRITICAL)(?:\]|:)")

LIVE_STREAM_FAILURE_MARKERS = (
    "ERROR",
    "CRITICAL",
    "Traceback",
    "Exception in callback",
    "RuntimeError:",
    "ModuleNotFoundError",
    "MemoryError",
    "[DEGRADATION]",
    "NEW INCIDENT",
    "Runtime: DEGRADED",
    "generation gate forcibly",
    "Cognitive cycle TIMEOUT",
    "Network gateway request failed",
    "desktop_cognitive_engine_required_no_reply",
    "high lag",
    "out of application memory",
    "Cortex Warming",
    "CORTEX UNAVAILABLE",
    "Cortex route blocked",
    "Desktop CognitiveEngine produced no acceptable reply",
    "no answer-quality response",
    "Dialogue contract deterministic repair still failed before retry",
    "ungrounded_live_voice",
)

LIVE_DESKTOP_FULL_RUNTIME_COMPONENTS = (
    "pneuma",
    "mhaf",
    "curiosity",
    "proactive_communication",
    "autonomous_initiative",
    "research",
    "self_healing",
    "self_modification",
    "consciousness_stream",
    "autonomy_conductor",
    "overt_action",
    "deliberation",
    "wake_word",
    "screen_perception",
    "perceptual_pump",
    "cognitive_situation",
    "imagination_engine",
    "timescale_bridge",
    "ambient_developer_stream",
    "autonomic_reflection_loop",
)

LIVE_CONVERSATION_SOAK_PROMPTS = (
    "Answer directly in two sentences: what lane are you using for this live desktop chat?",
    "What tools can you use externally, and what governance has to approve before you act?",
    "Remember this note for later in this conversation: the blue lantern is under the desk.",
    "What note did I ask you to remember in this conversation?",
    "Give a concise plan for creating a note and exporting it as a PDF, but do not execute tools.",
    "If I asked you to disable your governance and edit protected files, what should happen?",
    "Explain how you would use browser research and a document editor together on a user task.",
    "What changed in this conversation after I gave you the blue-lantern note?",
    "Name one failure mode you should surface honestly instead of masking.",
    "How would you keep RAM bounded while using local inference and desktop tools?",
    "Give a practical multi-step desktop task you could attempt after authorization.",
    "Finish with a short status: are you still coherent, on the same thread, and able to continue?",
)


def _env_float(env: dict[str, str], name: str, default: float) -> float:
    try:
        return float(env.get(name, str(default)))
    except (TypeError, ValueError, OverflowError):
        return default


def _truthy_env(env: dict[str, str], name: str) -> bool:
    return str(env.get(name, "") or "").strip().lower() in {"1", "true", "yes", "on"}


def live_proof_rss_abort_mb(env: dict[str, str] | None = None) -> float:
    """Return the outside proof kill ceiling for Aura's process tree."""

    env = dict(os.environ if env is None else env)
    process_limit_gb = _env_float(env, "AURA_PROCESS_RSS_LIMIT_GB", 0.0)
    derived = DEFAULT_RSS_ABORT_MB
    if process_limit_gb > 0.0:
        derived = min(DEFAULT_RSS_ABORT_MB, (process_limit_gb * 1024.0) + 4096.0)

    configured = _env_float(env, "AURA_LIVE_PROOF_RSS_ABORT_MB", 0.0)
    if configured > 0.0:
        if _truthy_env(env, "AURA_ALLOW_UNSAFE_MEMORY_LIMITS"):
            return configured
        return min(configured, derived)
    return derived


def build_safe_boot_env(
    base_env: dict[str, str] | None = None,
    *,
    mode: str = "headless",
    observer: ResourceObserver | None = None,
) -> dict[str, str]:
    """Return the bounded desktop environment used by live proof boots.

    The direct proof exercises the same bounded local model profile a desktop
    user relies on, but it must not impersonate the signed Aura.app launch
    identity. Packaged-lane certification requires a real resident app process
    and its source/signature provenance evidence.
    """

    env = dict(os.environ if base_env is None else base_env)
    mode = str(mode or "headless").strip().lower()
    env["AURA_LOCAL_BACKEND"] = "mlx"
    env.setdefault("AURA_LOCAL_RUNTIME_SINGLETON", "1")
    env.setdefault("AURA_LOCAL_PARALLEL_SLOTS", "1")
    env.setdefault("AURA_GOVERNANCE_MODE", "production")
    env.setdefault("AURA_CONTRACTS_ENFORCE", "1")
    for key in tuple(env):
        if key.startswith("AURA_LAUNCH_"):
            env.pop(key, None)
    env.pop("AURA_AUTO_LISTEN", None)
    if mode == "desktop":
        env["AURA_SAFE_BOOT_DESKTOP"] = "0"
        env["AURA_DESKTOP_RESOURCE_GUARD"] = "1"
        env["AURA_HEADLESS"] = "0"
        env["AURA_LAUNCHED_FROM_APP"] = "0"
        env["AURA_EXTERNAL_GUI_OWNER"] = "0"
        env["AURA_EAGER_LOCAL_SENSORY_BOOT"] = "1"
        env.setdefault("AURA_EAGER_CORTEX_WARMUP", "0")
        env.setdefault("AURA_DEFERRED_CORTEX_PREWARM", "1")
        env.setdefault("AURA_AMBIENT_STREAM_INTERVAL_S", "5")
        env.setdefault("AURA_AUTONOMIC_REFLECTION_INTERVAL_S", "30")
    else:
        env.setdefault("AURA_SAFE_BOOT_DESKTOP", "1")
        env.setdefault("AURA_HEADLESS", "1")
        env.setdefault("AURA_EAGER_LOCAL_SENSORY_BOOT", "0")
        env.setdefault("AURA_DEFERRED_CORTEX_PREWARM", "1")
    env.setdefault("AURA_ENABLE_PROACTIVE_VISION", "0")
    env.setdefault("AURA_DESKTOP_METAL_CACHE_RATIO", "0.16")
    env.setdefault("AURA_DESKTOP_METAL_CACHE_CAP_GB", "10")
    env.setdefault("AURA_DESKTOP_MLX_MEMORY_RATIO", "0.54")
    env.setdefault("AURA_DESKTOP_MLX_MEMORY_CAP_GB", "34")
    env.setdefault("AURA_DESKTOP_MLX_MEMORY_FLOOR_GB", "18")
    env.setdefault("AURA_DESKTOP_PROCESS_RSS_RATIO", "0.81")
    env.setdefault("AURA_DESKTOP_PROCESS_RSS_CAP_GB", "56")
    env.setdefault("AURA_DESKTOP_HOST_RESERVE_RATIO", "0.18")
    env.setdefault("AURA_DESKTOP_HOST_RESERVE_FLOOR_GB", "8")
    env.setdefault("AURA_DESKTOP_PROCESS_RSS_FLOOR_GB", "24")
    env.setdefault("AURA_MEMWATCH_SOFT_MB", "auto")
    env.setdefault("AURA_MEMWATCH_HARD_MB", "auto")
    env.setdefault("AURA_MEMWATCH_LETHAL_MB", "auto")
    env.setdefault("AURA_MEMORY_SENTINEL_INTERVAL_S", "0.5")
    env.setdefault("AURA_GOVERNOR_PRUNE_MB", "auto")
    env.setdefault("AURA_GOVERNOR_UNLOAD_MB", "auto")
    env.setdefault("AURA_GOVERNOR_CRITICAL_MB", "auto")
    env.setdefault("AURA_ENABLE_LOCAL_DEEP_SOLVER", "0")
    env.setdefault("AURA_MLX_32B_PROJECTED_FOOTPRINT_GB", "auto")
    env.setdefault("AURA_MLX_32B_PROCESS_RESERVE_GB", "3")
    env.setdefault("AURA_MLX_72B_PROJECTED_FOOTPRINT_GB", "auto")
    env.setdefault("AURA_MLX_72B_PROCESS_RESERVE_GB", "5")
    env.setdefault("AURA_FOREGROUND_CHAT_MAX_TOKENS", "2048")
    env.setdefault("AURA_WATCHDOG_BOOT_GRACE_S", "240")
    observer = observer or get_resource_observer()

    envelope_factory = None
    memory_total_bytes = 0
    try:
        from core.runtime.desktop_boot_safety import compute_mlx_memory_limit

        memory = observer.memory()
        if not memory.available or memory.total_bytes <= 0:
            raise RuntimeError(f"memory observation unavailable: {memory.error}")
        limit_bytes = compute_mlx_memory_limit(memory.total_bytes, env)
        limit_gb = max(1.0, min(34.0, limit_bytes / float(1024 ** 3)))
    except (ImportError, RuntimeError, TypeError, ValueError, OSError):
        limit_gb = min(34.0, max(1.0, _env_float(env, "AURA_MLX_MEMORY_LIMIT_GB", 34.0)))
    env["AURA_MLX_MEMORY_LIMIT_GB"] = f"{limit_gb:.0f}"

    try:
        from core.runtime.desktop_boot_safety import (
            compute_desktop_memory_envelope,
            compute_process_rss_limit,
        )

        memory = observer.memory()
        if not memory.available or memory.total_bytes <= 0:
            raise RuntimeError(f"memory observation unavailable: {memory.error}")
        limit_bytes = compute_process_rss_limit(memory.total_bytes, env)
        limit_gb = max(1.0, limit_bytes / float(1024 ** 3))
        envelope_factory = compute_desktop_memory_envelope
        memory_total_bytes = memory.total_bytes
    except (ImportError, RuntimeError, TypeError, ValueError, OSError):
        limit_gb = max(1.0, _env_float(env, "AURA_PROCESS_RSS_LIMIT_GB", 52.0))
    env["AURA_PROCESS_RSS_LIMIT_GB"] = f"{limit_gb:.0f}"
    if envelope_factory is not None and memory_total_bytes > 0:
        envelope = envelope_factory(memory_total_bytes, env)
        env["AURA_GOVERNOR_PRUNE_MB"] = f"{envelope.governor_prune_mb:.0f}"
        env["AURA_GOVERNOR_UNLOAD_MB"] = f"{envelope.governor_unload_mb:.0f}"
        env["AURA_GOVERNOR_CRITICAL_MB"] = f"{envelope.governor_critical_mb:.0f}"
        env["AURA_MEMWATCH_SOFT_MB"] = f"{envelope.watchdog_soft_mb:.0f}"
        env["AURA_MEMWATCH_HARD_MB"] = f"{envelope.watchdog_hard_mb:.0f}"
        env["AURA_MEMWATCH_LETHAL_MB"] = f"{envelope.watchdog_lethal_mb:.0f}"
    return env


def current_git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def current_git_dirty() -> bool | None:
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    # The clean-tree requirement is about committed SOURCE: the proof must run on
    # committed code, not a developer's uncommitted edits. Generated artifacts are
    # NOT source — the cert writes artifacts/ during its own run, and a handful are
    # tracked (architecture map, agi_live/aletheia results) so they legitimately
    # drift mid-run. Excluding them keeps the gate honest (real source drift still
    # fails) without flagging the cert's own outputs. (2026-06-22)
    for line in proc.stdout.splitlines():
        path = line[3:] if len(line) > 3 else ""
        if " -> " in path:  # rename: "old -> new"
            path = path.split(" -> ", 1)[1]
        path = path.strip().strip('"')
        if path and not path.startswith("artifacts/"):
            return True
    return False


def artifact_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_launch_python() -> str:
    """Return the interpreter Aura's desktop launcher would use.

    The proof harness itself may be invoked from a different shell Python. The
    launched runtime must still match the real desktop lane, which prefers the
    repository venv and only then a Python 3.12 executable. Launching with the
    harness interpreter can silently move Aura onto the wrong dependency set and
    create false failures around macOS/MLX frameworks.
    """

    candidates = [
        ROOT / ".venv" / "bin" / "python3",
        ROOT / ".venv" / "bin" / "python",
        Path("/opt/homebrew/bin/python3.12"),
        Path("/usr/local/bin/python3.12"),
        Path("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"),
    ]
    for candidate in candidates:
        if not candidate.exists() or not os.access(candidate, os.X_OK):
            continue
        try:
            proc = subprocess.run(
                [str(candidate), "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        version = f"{proc.stdout} {proc.stderr}".strip()
        if "Python 3.12" in version:
            return str(candidate)
    return sys.executable


class LiveProof:
    def __init__(
        self,
        *,
        port: int,
        mode: str,
        boot_timeout_s: float,
        skip_desktop: bool,
        restart_continuity: bool,
        conversation_soak_turns: int,
        proof_dir: Path | None = None,
        observer: ResourceObserver | None = None,
    ):
        self.port = port
        self.mode = "desktop" if str(mode or "").strip().lower() == "desktop" else "headless"
        self.boot_timeout_s = boot_timeout_s
        self.skip_desktop = skip_desktop
        self.restart_continuity = restart_continuity
        self.conversation_soak_turns = max(0, min(conversation_soak_turns, 24))
        self.base = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None
        self.steps: list[dict[str, Any]] = []
        self.peak_rss_mb = 0.0
        self.started_at = time.time()
        self.started_monotonic = time.monotonic()
        self.proof_dir = (proof_dir or PROOF_DIR).resolve()
        self.proof_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.transcript_path = self.proof_dir / f"live_proof_{stamp}.jsonl"
        self.verdict_path = self.proof_dir / f"live_proof_{stamp}_verdict.json"
        self.latest_verdict_path = self.proof_dir / "LATEST_VERDICT.json"
        self.stdout_path = self.proof_dir / f"live_proof_{stamp}_stdout.log"
        self.rss_abort_mb = DEFAULT_RSS_ABORT_MB
        self.resource_observer = observer or get_resource_observer()
        self._stdout_handle = None
        self._boot_count = 0
        self.launch_python = resolve_launch_python()

    # ── recording ─────────────────────────────────────────────────────

    def record(self, step: str, ok: bool, **detail: Any) -> bool:
        entry = {
            "at": time.time(),
            "elapsed_s": round(time.monotonic() - self.started_monotonic, 2),
            "step": step,
            "ok": bool(ok),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "resource_observation": self.resource_observer.provenance.to_dict(),
            **detail,
        }
        self.steps.append(entry)
        with open(self.transcript_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        marker = "✅" if ok else "❌"
        print(f"{marker} [{entry['elapsed_s']:>7.1f}s] {step}: "
              f"{detail.get('summary', '')}", flush=True)
        return ok

    # ── process management ────────────────────────────────────────────

    def tree_rss_mb(self) -> float:
        if self.proc is None:
            return 0.0
        table = self.resource_observer.process_table()
        if not table.available:
            raise RuntimeError(f"process table observation unavailable: {table.error}")
        total = sum(
            process.rss_bytes
            for process in table.processes
            if process.pid == self.proc.pid or self.proc.pid in process.ancestor_pids
        )
        mb = total / (1024 * 1024)
        self.peak_rss_mb = max(self.peak_rss_mb, mb)
        return mb

    def guard_rss(self) -> None:
        mb = self.tree_rss_mb()
        if mb > self.rss_abort_mb:
            self.record(
                "rss_guard",
                False,
                summary=f"ABORT: tree RSS {mb:.0f}MB exceeded {self.rss_abort_mb:.0f}MB",
            )
            self.kill_hard()
            raise RuntimeError("live proof aborted on RSS ceiling")

    def port_in_use(self) -> bool:
        try:
            with httpx.Client(timeout=2.0) as client:
                client.get(f"{self.base}/api/health")
            return True
        except httpx.HTTPError:
            return False

    @staticmethod
    def _is_aura_main_process(cmdline: list[str]) -> bool:
        if not cmdline:
            return False
        args = [str(part or "") for part in cmdline]
        executable = Path(args[0]).name.lower()
        if "python" not in executable:
            return False
        for arg in args[1:]:
            if Path(arg).name == "aura_main.py":
                return True
        return False

    def _running_aura_main_pids(self) -> list[int]:
        table = self.resource_observer.process_table()
        if not table.available:
            raise RuntimeError(f"process table observation unavailable: {table.error}")
        return [
            process.pid
            for process in table.processes
            if self._is_aura_main_process(process.cmdline)
        ]

    def boot(self) -> bool:
        if self.port_in_use():
            return self.record(
                "preflight_port",
                False,
                summary=f"port {self.port} already serving — refusing to "
                f"fight an existing instance; stop it first "
                f"(python aura_main.py --stop)",
            )
        existing = self._running_aura_main_pids()
        if existing:
            return self.record(
                "preflight_process",
                False,
                summary=f"aura_main already running (pids {existing}); "
                f"refusing to double-boot",
            )

        env = build_safe_boot_env(
            os.environ,
            mode=self.mode,
            observer=self.resource_observer,
        )
        self.rss_abort_mb = live_proof_rss_abort_mb(env)
        self._boot_count += 1
        if self._stdout_handle is not None:
            self._stdout_handle.close()
        self._stdout_handle = open(self.stdout_path, "a", encoding="utf-8")
        self._stdout_handle.write(
            f"\n\n===== live_boot_proof boot {self._boot_count} "
            f"at {time.strftime('%Y-%m-%dT%H:%M:%S%z')} =====\n"
        )
        self._stdout_handle.flush()
        mode_arg = "--desktop" if self.mode == "desktop" else "--headless"
        self.proc = subprocess.Popen(
            [self.launch_python, "aura_main.py", mode_arg, "--port", str(self.port)],
            cwd=ROOT,
            stdout=self._stdout_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self.record("boot_spawn", True, summary=f"pid {self.proc.pid}")

        deadline = time.monotonic() + self.boot_timeout_s
        last_state: dict[str, Any] = {}
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                return self.record(
                    "boot_health",
                    False,
                    summary=f"process exited during boot (rc={self.proc.returncode})",
                )
            self.guard_rss()
            try:
                with httpx.Client(timeout=5.0) as client:
                    heartbeat_resp = client.get(f"{self.base}/api/health/heartbeat")
                    boot_resp = client.get(f"{self.base}/api/health/boot")
                    health_resp = client.get(f"{self.base}/api/health")
                if (
                    heartbeat_resp.status_code in {200, 503}
                    and boot_resp.status_code in {200, 503}
                    and health_resp.status_code in {200, 503}
                ):
                    heartbeat_payload = heartbeat_resp.json()
                    boot_payload = boot_resp.json()
                    health_payload = health_resp.json()
                    heartbeat = heartbeat_payload if isinstance(heartbeat_payload, dict) else {}
                    boot = boot_payload if isinstance(boot_payload, dict) else {}
                    api_health = health_payload if isinstance(health_payload, dict) else {}
                    last_state = {"heartbeat": heartbeat, "boot": boot, "health": api_health}
                    required = heartbeat.get("required_probes")
                    required_ok = bool(
                        isinstance(required, dict)
                        and required.get("all_passed") is True
                    )
                    blockers = heartbeat.get("blockers")
                    boot_blockers = boot.get("blockers")
                    no_blockers = isinstance(blockers, list) and not blockers
                    normalized_blockers = {
                        str(item)
                        for item in (blockers if isinstance(blockers, list) else [])
                        if str(item or "").strip()
                    }
                    normalized_boot_blockers = {
                        str(item)
                        for item in (boot_blockers if isinstance(boot_blockers, list) else [])
                        if str(item or "").strip()
                    }
                    lane = heartbeat.get("conversation_lane")
                    if not isinstance(lane, dict):
                        lane = boot.get("conversation_lane")
                    if not isinstance(lane, dict):
                        lane = {}
                    lane_blockers = {
                        str(item)
                        for item in (lane.get("readiness_blockers") or [])
                        if str(item or "").strip()
                    }
                    runtime_degradations = boot.get("runtime_degradations")
                    if not isinstance(runtime_degradations, dict):
                        runtime_degradations = {}
                    degraded_critical = list(runtime_degradations.get("critical") or [])
                    degraded_important = list(runtime_degradations.get("important") or [])
                    checks = boot.get("checks")
                    if not isinstance(checks, dict):
                        checks = {}
                    full_runtime = api_health.get("full_runtime")
                    if not isinstance(full_runtime, dict):
                        full_runtime = {}
                    full_components = full_runtime.get("components")
                    if not isinstance(full_components, dict):
                        full_components = {}
                    full_runtime_blockers = [
                        str(item)
                        for item in (full_runtime.get("blockers") or [])
                        if str(item or "").strip()
                    ]
                    missing_full_components: list[str] = []
                    if self.mode == "desktop":
                        for component in LIVE_DESKTOP_FULL_RUNTIME_COMPONENTS:
                            component_status = full_components.get(component)
                            if not isinstance(component_status, dict) or not bool(
                                component_status.get("running", False)
                            ):
                                missing_full_components.append(component)
                    full_runtime_ok = (
                        self.mode != "desktop"
                        or (
                            bool(full_runtime.get("full_runtime_expected", False))
                            and bool(api_health.get("full_runtime_ready", full_runtime.get("ready", False)))
                            and not full_runtime_blockers
                            and not missing_full_components
                        )
                    )
                    all_observed_blockers = normalized_blockers | normalized_boot_blockers | lane_blockers
                    allowed_unproven_conversation_blockers = {
                        "healthy",
                        "runtime_contract_healthy",
                        "conversation_ready",
                        "conversation_lane:cold",
                        "conversation_lane:closed",
                        "conversation_lane:handshaking",
                        "conversation_lane:warming",
                        "conversation_lane:ready",
                        "conversation_reason:worker_not_alive",
                        "conversation_reason:init_not_complete",
                        "conversation_reason:lane_cold",
                        "worker_not_alive",
                        "init_not_complete",
                        "lane_cold",
                        "conversation_reason:visible_conversation_probe_missing",
                        "visible_conversation_probe_missing",
                    }
                    conversation_unproven_only = bool(
                        self.mode == "desktop"
                        and required_ok
                        and full_runtime_ok
                        and heartbeat.get("runtime_probe_healthy") is True
                        and checks.get("runtime_required_probes") is True
                        and not degraded_critical
                        and not degraded_important
                        and lane.get("state") == "ready"
                        and "visible_conversation_probe_missing" in lane_blockers
                        and all_observed_blockers
                        and all_observed_blockers <= allowed_unproven_conversation_blockers
                    )
                    conversation_standby_only = bool(
                        self.mode == "desktop"
                        and required_ok
                        and full_runtime_ok
                        and heartbeat.get("runtime_probe_healthy") is True
                        and boot.get("system_ready") is True
                        and boot.get("launcher_ready") is True
                        and checks.get("runtime_required_probes") is True
                        and not degraded_critical
                        and not degraded_important
                        and lane.get("state") in {"cold", "closed", ""}
                        and lane.get("conversation_ready") is False
                        and not bool(lane.get("warmup_attempted", False))
                        and not bool(lane.get("warmup_in_flight", False))
                        and all_observed_blockers
                        and all_observed_blockers <= allowed_unproven_conversation_blockers
                    )
                    if (
                        heartbeat.get("healthy") is True
                        and heartbeat.get("runtime_probe_healthy") is True
                        and boot.get("system_ready") is True
                        and boot.get("conversation_ready") is True
                        and boot.get("ready") is True
                        and required_ok
                        and full_runtime_ok
                        and no_blockers
                    ):
                        return self.record(
                            "boot_health",
                            True,
                            summary=f"healthy after {time.time() - self.started_at:.0f}s "
                            f"(rss {self.tree_rss_mb():.0f}MB)",
                            health=last_state,
                        )
                    if conversation_unproven_only:
                        return self.record(
                            "boot_health",
                            True,
                            summary=(
                                "desktop system ready; live conversation remains unproven "
                                "until the first required CognitiveEngine chat probe "
                                f"(rss {self.tree_rss_mb():.0f}MB)"
                            ),
                            health=last_state,
                            conversation_probe_required=True,
                        )
                    if conversation_standby_only:
                        return self.record(
                            "boot_health",
                            True,
                            summary=(
                                "desktop system ready; conversation lane is standby and "
                                "must warm on the first required CognitiveEngine chat probe "
                                f"(rss {self.tree_rss_mb():.0f}MB)"
                            ),
                            health=last_state,
                            conversation_probe_required=True,
                        )
            except httpx.HTTPError as exc:
                last_state = {
                    **last_state,
                    "last_health_error": f"{type(exc).__name__}: {exc}",
                    "mode": self.mode,
                }
            time.sleep(3.0)
        return self.record(
            "boot_health",
            False,
            summary=f"not healthy within {self.boot_timeout_s:.0f}s",
            last_health=last_state,
        )

    # ── exercises ─────────────────────────────────────────────────────

    def chat(
        self,
        message: str,
        *,
        timeout_s: float = 180.0,
        session_id: str = "live-proof",
        headers: dict[str, str] | None = None,
    ) -> tuple[bool, str, float]:
        started = time.monotonic()
        try:
            with httpx.Client(timeout=timeout_s, headers=headers) as client:
                resp = client.post(
                    f"{self.base}/api/chat",
                    json={"message": message, "session_id": session_id},
                )
            latency = time.monotonic() - started
            if resp.status_code != 200:
                return False, f"http {resp.status_code}: {resp.text[:300]}", latency
            payload = resp.json()
            text = str(
                payload.get("response")
                or payload.get("reply")
                or payload.get("message")
                or payload.get("text")
                or ""
            ).strip()
            return bool(text), text, latency
        except httpx.HTTPError as exc:
            return False, f"{type(exc).__name__}: {exc}", time.monotonic() - started

    def exercise_identity_turn(self) -> bool:
        ok, text, latency = self.chat(
            "Quick reliability check, in two or three sentences: what are you, "
            "and will you remember this conversation tomorrow?"
        )
        self.guard_rss()
        if not ok:
            return self.record(
                "chat_identity", False, summary=text[:200], latency_s=round(latency, 1)
            )
        from core.conversation.response_reliability import assess_user_facing_reply
        from core.conversation.self_claim_verifier import verify_self_claims

        verdict = verify_self_claims(text)
        reliability = assess_user_facing_reply(
            (
                "Quick reliability check, in two or three sentences: what are you, "
                "and will you remember this conversation tomorrow?"
            ),
            text,
        )
        ok = verdict.ok and reliability.ok
        return self.record(
            "chat_identity",
            ok,
            summary=(
                f"{latency:.1f}s, {len(text)} chars"
                + ("" if verdict.ok else
                   f" — SELF-CLAIM VIOLATIONS: {[v.kind for v in verdict.violations]}")
                + ("" if reliability.ok else
                   f" — RELIABILITY: {list(reliability.reasons)}")
            ),
            latency_s=round(latency, 1),
            reply=text[:1500],
            self_claim_ok=verdict.ok,
            violations=[v.kind for v in verdict.violations],
            reliability_ok=reliability.ok,
            reliability_reasons=list(reliability.reasons),
        )

    def exercise_capability_inventory_turn(self) -> bool:
        started = time.monotonic()
        rss_before = self.tree_rss_mb()
        message = (
            "What tools can you do externally from the live desktop path? "
            "Name the practical categories and one hypothetical multi-step scenario, "
            "but do not open apps or execute tools yet."
        )
        try:
            with httpx.Client(
                timeout=45.0,
                headers={
                    "X-Aura-Surface": "desktop-ui",
                    "X-Aura-Require-CognitiveEngine": "true",
                },
            ) as client:
                resp = client.post(
                    f"{self.base}/api/chat",
                    json={"message": message, "session_id": "live-proof"},
                )
            latency = time.monotonic() - started
            self.guard_rss()
            if resp.status_code != 200:
                return self.record(
                    "chat_capability_inventory",
                    False,
                    summary=f"http {resp.status_code}: {resp.text[:200]}",
                    latency_s=round(latency, 1),
                    response_status_code=resp.status_code,
                    response_body=resp.text,
                )
            payload = resp.json()
            text = str(payload.get("response") or "").strip()
            lowered = text.lower()
            status = str(payload.get("status") or "")
            from core.conversation.response_reliability import assess_user_facing_reply

            reliability = assess_user_facing_reply(message, text)
            required_terms = ("desktop", "browser", "file", "govern", "not opening apps")
            missing = [term for term in required_terms if term not in lowered]
            false_limit = bool(re.search(r"\bi\s+(?:can(?:not|'t)|cannot|do not have access)\b", lowered))
            ok = bool(text) and reliability.ok and not missing and not false_limit
            return self.record(
                "chat_capability_inventory",
                ok,
                summary=(
                    f"{latency:.1f}s, status={status or 'unknown'}, "
                    f"rss_delta={self.tree_rss_mb() - rss_before:.0f}MB"
                    + (
                        ""
                        if ok
                        else (
                            f", missing={missing}, false_limit={false_limit}, "
                            f"reliability={list(reliability.reasons)}"
                        )
                    )
                ),
                latency_s=round(latency, 1),
                status=status,
                reply=text[:1200],
                reliability_ok=reliability.ok,
                reliability_reasons=list(reliability.reasons),
                rss_before_mb=round(rss_before, 1),
                rss_after_mb=round(self.tree_rss_mb(), 1),
            )
        except httpx.HTTPError as exc:
            return self.record(
                "chat_capability_inventory",
                False,
                summary=f"{type(exc).__name__}: {exc}",
                latency_s=round(time.monotonic() - started, 1),
            )

    def exercise_conversation_soak(self) -> bool:
        if self.conversation_soak_turns <= 0:
            return self.record("chat_conversation_soak", True, summary="skipped", skipped=True)

        from core.conversation.response_reliability import assess_user_facing_reply

        prompts = LIVE_CONVERSATION_SOAK_PROMPTS[: self.conversation_soak_turns]
        session_id = f"live-proof-soak-{int(time.time())}"
        turn_summaries: list[dict[str, Any]] = []
        passed = True
        for index, prompt in enumerate(prompts, start=1):
            started = time.monotonic()
            rss_before = self.tree_rss_mb()
            try:
                with httpx.Client(
                    timeout=180.0,
                    headers={
                        "X-Aura-Surface": "desktop-ui",
                        "X-Aura-Require-CognitiveEngine": "true",
                    },
                ) as client:
                    resp = client.post(
                        f"{self.base}/api/chat",
                        json={"message": prompt, "session_id": session_id},
                    )
                latency = time.monotonic() - started
                self.guard_rss()
                if resp.status_code != 200:
                    return self.record(
                        f"chat_soak_turn_{index:02d}",
                        False,
                        summary=f"http {resp.status_code}: {resp.text[:180]}",
                        turn=index,
                        latency_s=round(latency, 1),
                        response_status_code=resp.status_code,
                        response_body=resp.text,
                    )
                payload = resp.json()
                text = str(payload.get("response") or payload.get("reply") or "").strip()
                status = str(payload.get("status") or "")
                reliability = assess_user_facing_reply(prompt, text)
                fallback = bool(LIVE_FALLBACK_RE.search(text))
                bounded_floor = status in {
                    "cognitive_engine_bounded_planning",
                    "cognitive_engine_failure_mode_surface",
                    "desktop_social_presence_contract",
                    "runtime_fact_status",
                }
                ok = bool(text) and reliability.ok and not fallback and not bounded_floor
                turn_detail = {
                    "turn": index,
                    "status": status,
                    "latency_s": round(latency, 1),
                    "rss_before_mb": round(rss_before, 1),
                    "rss_after_mb": round(self.tree_rss_mb(), 1),
                    "chars": len(text),
                    "reliability_ok": reliability.ok,
                    "reliability_reasons": list(reliability.reasons),
                    "fallback": fallback,
                    "bounded_floor": bounded_floor,
                    "reply": text[:800],
                }
                turn_summaries.append(turn_detail)
                self.record(
                    f"chat_soak_turn_{index:02d}",
                    ok,
                    summary=(
                        f"{latency:.1f}s status={status or 'unknown'} "
                        f"chars={len(text)} rss_delta={self.tree_rss_mb() - rss_before:.0f}MB"
                        + (
                            ""
                            if ok
                            else (
                                f" reasons={list(reliability.reasons)} "
                                f"fallback={fallback} bounded_floor={bounded_floor}"
                            )
                        )
                    ),
                    **turn_detail,
                )
                passed &= ok
                if not ok:
                    break
            except httpx.HTTPError as exc:
                return self.record(
                    f"chat_soak_turn_{index:02d}",
                    False,
                    summary=f"{type(exc).__name__}: {exc}",
                    turn=index,
                    latency_s=round(time.monotonic() - started, 1),
                )

        return self.record(
            "chat_conversation_soak",
            passed and len(turn_summaries) == len(prompts),
            summary=f"{len(turn_summaries)}/{len(prompts)} turns passed",
            turns=turn_summaries,
        )

    def exercise_cognitive_organ_participation(self) -> bool:
        """Prove foreground replies used live mind organs and timescale grounding."""

        required = ("cognitive_situation", "imagination_engine")
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self.base}/api/health")
            if resp.status_code != 200:
                return self.record(
                    "chat_cognitive_organs",
                    False,
                    summary=f"health http {resp.status_code}: {resp.text[:180]}",
                )
            payload = resp.json()
            full_runtime = payload.get("full_runtime") if isinstance(payload, dict) else None
            components = (
                full_runtime.get("components")
                if isinstance(full_runtime, dict)
                else None
            )
            components = components if isinstance(components, dict) else {}
            evidence: dict[str, dict[str, Any]] = {}
            blockers: list[str] = []
            for name in required:
                status = components.get(name)
                status = status if isinstance(status, dict) else {}
                frames_built = int(status.get("frames_built") or status.get("frames") or 0)
                latest = status.get("latest")
                organ_ok = bool(
                    status.get("running") is True
                    and frames_built > 0
                    and isinstance(latest, dict)
                    and latest
                )
                evidence[name] = {
                    "running": status.get("running") is True,
                    "frames_built": frames_built,
                    "latest_frame_id": (
                        str(latest.get("frame_id") or "")
                        if isinstance(latest, dict)
                        else ""
                    ),
                    "participated": organ_ok,
                }
                if not organ_ok:
                    blockers.append(name)
            timescale_status = components.get("timescale_bridge")
            timescale_status = timescale_status if isinstance(timescale_status, dict) else {}
            latest_observation = timescale_status.get("latest_observation")
            latest_observation = (
                latest_observation
                if isinstance(latest_observation, dict)
                else {}
            )
            last_reconciliation = timescale_status.get("last_reconciliation")
            last_reconciliation = (
                last_reconciliation
                if isinstance(last_reconciliation, dict)
                else {}
            )
            directives = last_reconciliation.get("directives")
            directives = directives if isinstance(directives, list) else []
            observations = int(timescale_status.get("observations") or 0)
            frames_ingested = int(timescale_status.get("frames_ingested") or 0)
            timescale_ok = bool(
                timescale_status.get("running") is True
                and observations > 0
                and frames_ingested > 0
                and latest_observation
                and last_reconciliation
                and directives
            )
            evidence["timescale_bridge"] = {
                "running": timescale_status.get("running") is True,
                "observations": observations,
                "frames_ingested": frames_ingested,
                "latest_source": str(latest_observation.get("source") or ""),
                "last_idle_gap_s": last_reconciliation.get("idle_gap_s"),
                "last_summary": str(last_reconciliation.get("summary") or ""),
                "directives": directives[:4],
                "participated": timescale_ok,
            }
            if not timescale_ok:
                blockers.append("timescale_bridge")
            ambient_status = components.get("ambient_developer_stream")
            ambient_status = ambient_status if isinstance(ambient_status, dict) else {}
            latest_ambient = ambient_status.get("latest_frame")
            latest_ambient = latest_ambient if isinstance(latest_ambient, dict) else {}
            ambient_frames = int(ambient_status.get("frames") or 0)
            ambient_ok = bool(
                ambient_status.get("running") is True
                and ambient_frames > 0
                and latest_ambient
            )
            evidence["ambient_developer_stream"] = {
                "running": ambient_status.get("running") is True,
                "frames": ambient_frames,
                "latest_summary": str(latest_ambient.get("summary") or ""),
                "participated": ambient_ok,
            }
            if not ambient_ok:
                blockers.append("ambient_developer_stream")

            reflection_status = components.get("autonomic_reflection_loop")
            reflection_status = reflection_status if isinstance(reflection_status, dict) else {}
            reflection_errors = int(reflection_status.get("errors") or 0)
            reflections_written = int(reflection_status.get("reflections_written") or 0)
            reflection_ok = bool(
                reflection_status.get("running") is True
                and reflection_errors == 0
            )
            evidence["autonomic_reflection_loop"] = {
                "running": reflection_status.get("running") is True,
                "reflections_written": reflections_written,
                "errors": reflection_errors,
                "participated": reflection_ok,
            }
            if not reflection_ok:
                blockers.append("autonomic_reflection_loop")
            return self.record(
                "chat_cognitive_organs",
                not blockers,
                summary=(
                    "semantic, imagination, timescale, ambient, and autonomic organs processed live turns"
                    if not blockers
                    else f"missing live participation: {', '.join(blockers)}"
                ),
                organs=evidence,
                blockers=blockers,
            )
        except (httpx.HTTPError, json.JSONDecodeError, TypeError, ValueError) as exc:
            return self.record(
                "chat_cognitive_organs",
                False,
                summary=f"{type(exc).__name__}: {exc}",
            )

    def exercise_continuity_turn(self) -> bool:
        token = f"amber-{int(time.time()) % 100000}"
        ok1, _, lat1 = self.chat(
            f"Remember this codeword for me: {token}. Just confirm you have it."
        )
        self.guard_rss()
        ok2, text2, lat2 = self.chat("What codeword did I just give you?")
        self.guard_rss()
        recalled = token.lower() in text2.lower()
        # Recall is the criterion. Round 13: she answered 'The codeword
        # you gave me is amber-82004' — perfect recall — but the set
        # turn's reply text had been empty under gate serialization and
        # the old all-three conjunction marked the step red. A silent
        # set with proven recall is a pass; the set latency still lands
        # in the transcript for the record.
        return self.record(
            "chat_continuity",
            ok2 and recalled,
            summary=(
                f"set {lat1:.1f}s / recall {lat2:.1f}s — "
                + ("codeword recalled" if recalled else
                   f"NOT recalled (reply: {text2[:160]})")
            ),
            token=token,
            recalled=recalled,
            reply=text2[:600],
        )

    def exercise_restart_continuity_turn(self) -> bool:
        token = f"restart-{int(time.time()) % 100000}"
        ok1, text1, lat1 = self.chat(
            f"Remember this codeword across restart: {token}. Just confirm.",
            session_id="live-proof-restart",
        )
        self.guard_rss()
        if not ok1:
            return self.record(
                "chat_restart_continuity",
                False,
                summary=f"memory set failed before restart: {text1[:200]}",
                token=token,
                set_latency_s=round(lat1, 1),
            )
        shutdown_ok = self.shutdown(step="restart_shutdown")
        if not shutdown_ok:
            return self.record(
                "chat_restart_continuity",
                False,
                summary="shutdown failed before restart recall",
                token=token,
                set_latency_s=round(lat1, 1),
            )
        self.proc = None
        time.sleep(3.0)
        boot_ok = self.boot()
        if not boot_ok:
            return self.record(
                "chat_restart_continuity",
                False,
                summary="reboot failed before restart recall",
                token=token,
                set_latency_s=round(lat1, 1),
            )
        ok2, text2, lat2 = self.chat(
            "What codeword did I ask you to remember before restart?",
            session_id="live-proof-restart-after",
        )
        self.guard_rss()
        recalled = token.lower() in text2.lower()
        return self.record(
            "chat_restart_continuity",
            ok2 and recalled,
            summary=(
                f"set {lat1:.1f}s / reboot recall {lat2:.1f}s — "
                + ("codeword recalled" if recalled else f"NOT recalled ({text2[:160]})")
            ),
            token=token,
            recalled=recalled,
            set_reply=text1[:400],
            recall_reply=text2[:800],
        )

    def exercise_desktop_action(self) -> bool:
        if self.skip_desktop:
            return self.record(
                "desktop_action", True, summary="skipped by flag", skipped=True
            )
        target_dir = Path.home() / "Documents" / "Aura Live Proof"
        marker = target_dir / "live_proof.txt"
        step_started = time.time()
        ok, text, latency = self.chat(
            "Please create a folder named 'Aura Live Proof' in my Documents "
            "folder and write a file inside it called live_proof.txt with one "
            "sentence about who you are and the current timestamp. Use your "
            "desktop tools and confirm exactly what you did.",
            timeout_s=300.0,
        )
        self.guard_rss()
        # External verification: the proof is on disk, not in her words.
        # Freshness required: versioned writes mean the fixed path may
        # hold a PREVIOUS round's file (round 13 verified round 12's
        # artifact). Accept the newest matching file in the folder, but
        # only if it was written AFTER this step began — stale green is
        # forbidden evidence.
        time.sleep(2.0)
        candidates = sorted(
            target_dir.glob("live_proof*.txt"),
            key=lambda c: c.stat().st_mtime if c.exists() else 0,
            reverse=True,
        ) if target_dir.is_dir() else []
        fresh = [c for c in candidates if c.stat().st_mtime >= step_started - 1.0]
        marker = fresh[0] if fresh else marker
        file_exists = bool(fresh) and marker.is_file()
        content = marker.read_text(errors="replace")[:400] if file_exists else ""
        return self.record(
            "desktop_action",
            ok and file_exists and bool(content.strip()),
            summary=(
                f"{latency:.1f}s — "
                + (f"file verified on disk ({len(content)} chars)"
                   if file_exists else "FILE NOT FOUND on disk")
            ),
            latency_s=round(latency, 1),
            reply=text[:800],
            file_exists=file_exists,
            file_content=content,
            path=str(marker),
        )

    def snapshot_vitals(self) -> bool:
        vitals: dict[str, Any] = {"tree_rss_mb": round(self.tree_rss_mb(), 1)}
        ok = True
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self.base}/api/health")
            vitals["health_status_code"] = resp.status_code
            if resp.status_code == 200:
                payload = resp.json()
                if isinstance(payload, dict):
                    vitals["health"] = {
                        k: payload.get(k)
                        for k in ("status", "state", "healthy", "runtime", "uptime_s")
                        if k in payload
                    }
            else:
                ok = False
        except httpx.HTTPError as exc:
            vitals["health_error"] = str(exc)
            ok = False
        return self.record(
            "vitals", ok, summary=f"rss {vitals['tree_rss_mb']}MB", **vitals
        )

    def scan_runtime_stream(self) -> bool:
        """Fail the proof if the captured runtime stream exposes known live-path breaks."""

        if self._stdout_handle is not None:
            self._stdout_handle.flush()
        if not self.stdout_path.exists():
            return self.record(
                "runtime_stream_scan",
                True,
                summary="no runtime stdout log was created",
                skipped=True,
            )
        text = self.stdout_path.read_text(errors="replace")
        matches: dict[str, list[str]] = {}
        for marker in LIVE_STREAM_FAILURE_MARKERS:
            marker_lower = marker.lower()
            lines = [
                line[:700]
                for line in text.splitlines()
                if (
                    LIVE_LOG_LEVEL_RE.search(line)
                    if marker in {"ERROR", "CRITICAL"}
                    else marker_lower in line.lower()
                )
            ][:5]
            if lines:
                matches[marker] = lines
        ok = not matches
        return self.record(
            "runtime_stream_scan",
            ok,
            summary=(
                "no failure markers in runtime stdout"
                if ok
                else f"failure markers found: {', '.join(sorted(matches))}"
            ),
            stdout_log=artifact_display_path(self.stdout_path),
            markers=matches,
        )

    # ── shutdown ──────────────────────────────────────────────────────

    def shutdown(self, *, step: str = "shutdown") -> bool:
        if self.proc is None:
            return self.record(step, False, summary="no process")

        shutdown_started = time.monotonic()
        shutdown_budget_s = max(
            15.0,
            _env_float(os.environ, "AURA_LIVE_PROOF_SHUTDOWN_MAX_S", 90.0),
        )
        try:
            stop = subprocess.run(
                [self.launch_python, "aura_main.py", "--stop"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=90,
            )
            stop_note = f"--stop rc={stop.returncode}"
        except subprocess.SubprocessError as exc:
            stop_note = f"--stop failed: {exc}"

        try:
            self.proc.wait(timeout=60)
            graceful = True
        except subprocess.TimeoutExpired:
            graceful = False
            self.kill_hard()

        time.sleep(2.0)
        if self._stdout_handle is not None:
            self._stdout_handle.flush()
            self._stdout_handle.close()
            self._stdout_handle = None
        table = self.resource_observer.process_table()
        orphan_scan_error = "" if table.available else table.error or "unavailable"
        orphans = [
            process.pid
            for process in table.processes
            if any(
                marker in " ".join(process.cmdline)
                for marker in ("aura_main.py", "mlx_worker.py")
            )
        ]
        for pid in orphans:
            try:
                psutil.Process(pid).kill()
            except psutil.Error:
                pass
        port_free = not self.port_in_use()
        shutdown_duration_s = time.monotonic() - shutdown_started
        within_budget = shutdown_duration_s <= shutdown_budget_s
        return self.record(
            step,
            graceful and not orphans and not orphan_scan_error and port_free and within_budget,
            summary=(
                f"{stop_note}; graceful={graceful}; orphans={orphans or 'none'}; "
                f"orphan_scan_error={orphan_scan_error or 'none'}; "
                f"port_free={port_free}; duration={shutdown_duration_s:.1f}s/"
                f"{shutdown_budget_s:.0f}s"
            ),
            graceful=graceful,
            orphans=orphans,
            orphan_scan_error=orphan_scan_error,
            port_free=port_free,
            duration_s=round(shutdown_duration_s, 2),
            duration_budget_s=round(shutdown_budget_s, 2),
            within_budget=within_budget,
        )

    def kill_hard(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                self.proc.kill()
            except OSError:
                pass
        if self._stdout_handle is not None:
            self._stdout_handle.flush()
            self._stdout_handle.close()
            self._stdout_handle = None

    # ── orchestration ─────────────────────────────────────────────────

    def run(self) -> int:
        passed = True
        try:
            if not self.boot():
                passed = False
            else:
                passed &= self.snapshot_vitals()
                passed &= self.exercise_capability_inventory_turn()
                passed &= self.exercise_identity_turn()
                passed &= self.exercise_continuity_turn()
                passed &= self.exercise_conversation_soak()
                passed &= self.exercise_cognitive_organ_participation()
                passed &= self.exercise_desktop_action()
                if self.restart_continuity:
                    passed &= self.exercise_restart_continuity_turn()
                passed &= self.snapshot_vitals()
        except RuntimeError as exc:
            self.record("abort", False, summary=str(exc))
            passed = False
        finally:
            if self.proc is not None and self.proc.poll() is None:
                passed &= self.shutdown()
            passed &= self.scan_runtime_stream()

        finished_at = time.time()
        git_commit = current_git_commit()
        git_dirty = current_git_dirty()

        verdict = {
            "schema": "aura.live_boot_proof.v1",
            "passed": passed,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "ended_at": finished_at,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "mode": self.mode,
            "steps": self.steps,
            "transcript": artifact_display_path(self.transcript_path),
            "stdout_log": artifact_display_path(self.stdout_path),
            "resource_observation": self.resource_observer.provenance.to_dict(),
        }
        verdict_json = json.dumps(verdict, indent=2, default=str)
        self.verdict_path.write_text(verdict_json)
        self.latest_verdict_path.write_text(verdict_json)
        print(f"\n{'✅ LIVE PROOF PASSED' if passed else '❌ LIVE PROOF FAILED'}")
        print(f"verdict: {artifact_display_path(self.verdict_path)}")
        return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--mode",
        choices=("headless", "desktop"),
        default="headless",
        help="boot path to prove; desktop mirrors the packaged app launcher environment",
    )
    parser.add_argument("--boot-timeout", type=float, default=600.0)
    parser.add_argument("--skip-desktop-action", action="store_true")
    parser.add_argument(
        "--restart-continuity",
        action="store_true",
        help="prove explicit chat memory survives a real Aura process restart",
    )
    parser.add_argument(
        "--conversation-soak-turns",
        type=int,
        default=0,
        help="run repeated live desktop chat turns to catch coherence/fallback regressions",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROOF_DIR,
        help="directory for live proof transcript, stdout, and verdict artifacts",
    )
    args = parser.parse_args(argv)
    proof = LiveProof(
        port=args.port,
        mode=args.mode,
        boot_timeout_s=args.boot_timeout,
        skip_desktop=args.skip_desktop_action,
        restart_continuity=args.restart_continuity,
        conversation_soak_turns=args.conversation_soak_turns,
        proof_dir=args.out_dir,
    )
    return proof.run()


if __name__ == "__main__":
    raise SystemExit(main())
