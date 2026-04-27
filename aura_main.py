#!/usr/bin/env python3
"""
Aura Main Entry Point
---------------------
Standardized, single-entry launcher for CLI, Server, Desktop, and Watchdog modes.
Replaces: aura_launcher.py, aura_desktop.py, run_aura.py, run_aura_loop.py, and reboot.py.
"""

import argparse
import asyncio

# Install global task supervision before subsystems spawn background tasks.
try:
    import core.utils.asyncio_patch  # noqa: F401
except Exception:
    pass

import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional, List, Any, Dict
import multiprocessing

# Phase 31: Native M1 Pro Resilience Fixes
# 0. Force 'spawn' on macOS to prevent Cocoa/XPC deadlocks in child actors
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # PyAV's bundled libavdevice and OpenCV's bundled libavdevice both register
    # the Objective-C classes AVFFrameReceiver / AVFAudioReceiver.  The objc
    # runtime unconditionally prints a "Class … is implemented in both …"
    # warning to stderr when this happens, which floods our logs on every boot
    # (this is the source of the recurring "homebrew" error the user reported).
    # The conflict is benign in practice — objc keeps the first registration,
    # and nothing in Aura uses the AVFoundation capture classes — so we
    # eagerly load both libraries here with stderr muted, which absorbs the
    # one-shot duplicate-class notice and leaves subsequent transitive imports
    # silent.
    try:
        import contextlib as _contextlib
        _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        _saved_stderr = os.dup(2)
        try:
            os.dup2(_devnull_fd, 2)
            import av as _av  # noqa: F401  (ordering matters — av first)
            import cv2 as _cv2  # noqa: F401  (eager load to absorb dup warning)
        finally:
            os.dup2(_saved_stderr, 2)
            os.close(_devnull_fd)
            os.close(_saved_stderr)
    except ImportError:
        pass
    except Exception:
        # Never let the dylib suppression block boot.
        pass

# Early .env loading — ensures AURA_LOCAL_BACKEND and other env vars are
# available BEFORE module-level code in model_registry.py reads os.getenv().
# Without this, pydantic's env_file loading happens too late.
try:
    from dotenv import load_dotenv as _load_dotenv
    _env_path = Path(__file__).resolve().parent / ".env"
    if _env_path.exists():
        _load_dotenv(_env_path, override=False)
except ImportError:
    pass

# 1. Path Resolution & Environment Locking (Radical Fix)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# QUAL-07: Define logger early so venv injection logging works
logger = logging.getLogger("Aura.Main")

# [STABILITY] Force the execution context to the absolute path of the current venv
# This prevents the "ModuleNotFoundError" when pip is in the venv but the script runs elsewhere.
VENV_PATH = PROJECT_ROOT / ".venv"
if not VENV_PATH.exists():
    VENV_PATH = PROJECT_ROOT / ".venv_aura"

if VENV_PATH.exists():
    # Scan for any python3.x directory to handle version mismatches (e.g. venv is 3.12, system is 3.14)
    lib_dir = VENV_PATH / "lib"
    if lib_dir.exists():
        curr_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        for py_dir in lib_dir.glob("python3.*"):
            if py_dir.name != curr_ver:
                logger.debug("⏭️  Skipping venv injection for mismatched version: %s (Current: %s)", py_dir.name, curr_ver)
                continue
                
            site_packages = py_dir / "site-packages"
            if site_packages.exists() and str(site_packages) not in sys.path:
                sys.path.insert(0, str(site_packages))
                import site
                site.addsitedir(str(site_packages))
                logger.info("📍 Injected venv site-packages: %s", site_packages)

# Desktop-safe boot should keep the main Aura process off the in-process
# MLX/Metal path. The managed LLM runtimes use their own subprocesses.
try:
    from core.runtime.desktop_boot_safety import configure_inprocess_mlx_runtime

    _mlx_runtime = configure_inprocess_mlx_runtime()
    if _mlx_runtime.get("device") == "cpu":
        logger.info(
            "🛡️ In-process MLX pinned to CPU (%s).",
            _mlx_runtime.get("reason", "guard"),
        )
    elif _mlx_runtime.get("device") == "metal":
        logger.info(
            "⚡ In-process MLX Metal retained (%s).",
            _mlx_runtime.get("reason", "enabled"),
        )
except Exception as _mlx_guard_exc:
    logger.debug("In-process MLX boot guard unavailable: %s", _mlx_guard_exc)

# Phase 31: Native M1 Pro Resilience Fixes
# 1. Address AVFFrameReceiver conflict (cv2 vs av/PyAV) on macOS
# This prevents the "AVFFrameReceiver: ... is already established" crash
if sys.platform == "darwin":
    os.environ["OPENCV_VIDEOIO_AVFOUNDATION_USE_FRAME_RECEIVER"] = "0"
    os.environ["PYAV_SKIP_AVF_FRAME_RECEIVER"] = "1"

# Strip PyInstaller matplotlib bloat in frozen builds
if getattr(sys, 'frozen', False):
    os.environ.pop("MPLBACKEND", None)

# 2. Bootstrap Configuration & Logging
try:
    from core.config import config
    from core.logging_config import setup_logging
    # Centralized logging setup - always include log_dir for persistence
    setup_logging(log_dir=config.paths.log_dir)
    logger = logging.getLogger("Aura.Main")
except Exception as e:
    # Minimal fallback logging if core is broken
    import traceback
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("Aura.Main")
    logger.error("❌ BOOTSTRAP FAILURE: Could not load core configuration.")
    logger.error(traceback.format_exc())
    config = None # Ensure NameError is avoided

# Category 11: Reliability Hardening
_supervisor_tree: Optional[Any] = None

def get_supervisor_tree() -> Any:
    global _supervisor_tree
    if _supervisor_tree is None:
        from core.supervisor.tree import SupervisionTree
        _supervisor_tree = SupervisionTree()
    return _supervisor_tree

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def validate_security_config():
    """Verify that we aren't exposing a public API without authentication."""
    if config is None:
        logger.warning("⚠️ Config unavailable — skipping security validation (bootstrap failure).")
        return
    internal_only = getattr(config.security, "internal_only_mode", False)
    api_token = config.api_token
    
    # If host is NOT localhost and no token is set, we are in a dangerous state
    # Note: We check this even if the user passed --host 127.0.0.1 because
    # the server.py might override it or be proxied.
    if not internal_only and not api_token:
        from core.exceptions import SecurityConfigError
        logger.critical("🚨 SECURITY VIOLATION: Public API access enabled but AURA_API_TOKEN is unset.")
        logger.critical("   To fix this: Set AURA_API_TOKEN in .env or run with AURA_INTERNAL_ONLY=1")
        raise SecurityConfigError("Public API access enabled without AURA_API_TOKEN")

def check_environment():
    """Verify system readiness."""
    logger.info("🔍 Verifying Environment Integrity...")
    logger.info("📍 RUNTIME PATH Diagnostic:")
    logger.info("   • __file__: %s", __file__)
    logger.info("   • sys.executable: %s", sys.executable)
    logger.info("   • sys.path: %s", sys.path)
    try:
        import core
        logger.info("   • core.__file__: %s", core.__file__)
    except Exception as e:
        logger.error("   • core import failed: %s", e)
    
    if sys.version_info < (3, 12):
        logger.error("Aura requires Python 3.12+")
        sys.exit(1)
        
    if config is None:
        logger.error("❌ Environment check aborted: Configuration not loaded.")
        raise RuntimeError("Configuration not loaded")

    # Perplexity Audit Fix: Fail-closed security validation
    validate_security_config()

    # Validate any pending patches generated by the Autonomous Code Mod system
    patch_file = PROJECT_ROOT / "core" / "patches" / "pending_patch.py"
    if patch_file.exists():
        logger.info("🛠️  Pending patch detected. Validating syntax...")
        try:
            content = patch_file.read_text()
            import ast
            ast.parse(content)
            logger.warning("pending_patch.py passed syntax check. Run patch_applicator.py to apply.")
        except SyntaxError as e:
            logger.error("❌ Pending patch has syntax errors. Quarantining patch file. Error: %s", e)
            patch_file.rename(patch_file.with_suffix(".py.quarantined"))

    # Ensure home directory exists
    config.paths.create_directories()

def kill_port(port: int, pattern: str = "aura"):
    """Force kill any process on a specific port matching a pattern.
    
    [PHASE 51] HARDENING: If port is 8000 or 10003, we kill WHATEVER is on it
    to ensure Aura is not blocked by stale processes from any source.
    """
    try:
        import psutil
    except ImportError:
        logger.warning("psutil missing - skipping advanced port cleanup.")
        return

    # Critical ports that MUST be freed regardless of pattern
    critical_ports = {8000, 10003}
    force_all = port in critical_ports

    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.net_connections(kind='inet'):
                if conn.laddr.port == port:
                    pid = proc.pid
                    name = proc.info.get("name") or ""
                    cmd_str = ""
                    if not force_all:
                        try:
                            cmd_str = " ".join(proc.cmdline() or []).lower()
                        except (psutil.Error, PermissionError, SystemError, OSError) as exc:
                            logger.debug("Skipping cmdline inspection for PID %s during port cleanup: %s", pid, exc)

                    should_kill = force_all or (pattern in cmd_str or pattern in name.lower())
                    
                    if should_kill:
                        logger.info("Terminating process %s (%s) on port %s...", pid, name, port)
                        try:
                            proc.terminate()
                            proc.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            logger.warning("Process %s resistant to SIGTERM. Sending SIGKILL.", pid)
                            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, PermissionError, SystemError, OSError):
            pass

def clean_artifacts():
    """Purge stale bytecode and temporary caches."""
    logger.info("🧹 Purging runtime artifacts...")
    for p in PROJECT_ROOT.rglob("__pycache__"):
        try: shutil.rmtree(p)
        except Exception: pass
    for p in PROJECT_ROOT.rglob("*.pyc"):
        try: p.unlink()
        except Exception: pass


def _select_preferred_launcher_python(current_executable: Optional[str] = None) -> Optional[Path]:
    """Prefer a stable Homebrew Python 3.12 launcher over shimmed venv paths."""
    if sys.platform != "darwin":
        return None

    current_raw = Path(current_executable or sys.executable)
    current_raw_str = str(current_raw)
    if "/.venv/" not in current_raw_str and "/.venv_aura/" not in current_raw_str:
        return None

    candidates: list[Path] = []
    explicit = os.environ.get("AURA_PREFERRED_PYTHON")
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            Path("/opt/homebrew/opt/python@3.12/bin/python3.12"),
            Path("/opt/homebrew/bin/python3.12"),
        ]
    )

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            current_resolved = current_raw.resolve()
        except FileNotFoundError:
            continue
        if resolved.exists() and resolved != current_resolved:
            return candidate
    return None


def _launcher_python_executable() -> str:
    preferred = os.environ.get("AURA_PREFERRED_PYTHON", "").strip()
    if preferred and Path(preferred).exists():
        return preferred
    return sys.executable


REAPER_MANIFEST_ENV = "AURA_REAPER_MANIFEST"
DEFAULT_REAPER_MANIFEST = Path("/tmp/aura_reaper_manifest.json")


def _ensure_reaper_manifest_env() -> Path:
    """Force every launcher/process surface to share one manifest path."""
    raw_path = os.environ.get(REAPER_MANIFEST_ENV, "").strip()
    if raw_path:
        manifest_path = Path(raw_path).expanduser()
    else:
        manifest_path = DEFAULT_REAPER_MANIFEST
        os.environ[REAPER_MANIFEST_ENV] = str(manifest_path)
    return manifest_path


def _maybe_relaunch_with_preferred_python():
    if os.environ.get("AURA_SKIP_PREFERRED_PYTHON_RELAUNCH") == "1":
        return

    preferred = _select_preferred_launcher_python()
    if not preferred:
        return

    logger.warning("🔁 Relaunching Aura with preferred interpreter: %s", preferred)
    env = os.environ.copy()
    env["AURA_SKIP_PREFERRED_PYTHON_RELAUNCH"] = "1"
    env["AURA_PREFERRED_PYTHON"] = str(preferred)
    env["AURA_LOCAL_BACKEND"] = "llama_cpp"
    os.execve(str(preferred), [str(preferred), *sys.argv], env)

# ---------------------------------------------------------------------------
# Shims & Compatibility
# ---------------------------------------------------------------------------
try:
    from core.cognitive_integration import CognitiveIntegrationLayer
    CognitiveIntegration = CognitiveIntegrationLayer # Legacy Alias shim
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

async def bootstrap_aura(orchestrator: Any):
    """Initialize background services using the Resilient Boot sequence."""
    from core.ops.resilient_boot import ResilientBoot
    from core.container import ServiceContainer
    from core.bus.actor_bus import create_actor_bus
    from core.utils.task_tracker import get_task_tracker
    
    # Register core services early to satisfy boot dependencies
    supervisor = get_supervisor_tree()
    ServiceContainer.register_instance("supervisor", supervisor)
    
    actor_bus = create_actor_bus() # Main bus for orchestrator
    ServiceContainer.register_instance("actor_bus", actor_bus)

    # Guarded stage-based ignition
    # Explicitly link internal refs to ensure property lookups match initialized instances
    orchestrator._actor_bus = actor_bus
    orchestrator._supervisor_tree = supervisor
    tracker = get_task_tracker()
    tracker.install_loop_hygiene(asyncio.get_running_loop())
    
    boot = ResilientBoot(orchestrator)
    # [STABILITY] Wait for ignition to complete before proceeding
    # This ensures all core services and state repository are ready.
    status = await boot.ignite()
    logger.info("🛡️ [BOOT] Resilient Ignition finished with status: %s", status)
    
    # Final interface check
    if hasattr(orchestrator, "kernel_interface") and orchestrator.kernel_interface:
        for _ in range(5):
            if orchestrator.kernel_interface.is_ready():
                break
            await asyncio.sleep(1.0)
    
    # Register supervisor tree in container (Redundant but safe)
    # ServiceContainer.register_instance("supervisor", supervisor)
    
    # Post-boot background tasks
    from core.utils.memory_monitor import AppleSiliconMemoryMonitor
    mem_monitor = AppleSiliconMemoryMonitor()
    try:
        ServiceContainer.register_instance("memory_monitor", mem_monitor, required=False)
    except Exception as exc:
        logger.debug("Memory monitor registration skipped: %s", exc)
    tracker.create_task(mem_monitor.start(), name="memory_monitor.start")
    
    logger.info("🛡️  Task Supervisor active (Memory monitoring enabled).")

    # Hot-Swap Bridge
    runtime_loop = asyncio.get_running_loop()

    def _on_actor_restart(name: str, new_pipe: Any):
        logger.info("🔄 [HOTSWAP] Detected restart of %s. Re-binding IPC...", name)

        def _schedule_rebind():
            actor_bus = ServiceContainer.get("actor_bus", default=None)
            if actor_bus:
                tracker.create_task(
                    actor_bus.update_actor(name, new_pipe),
                    name=f"actor_bus.update_actor.{name}",
                )

        runtime_loop.call_soon_threadsafe(_schedule_rebind)
                
    supervisor.set_restart_callback(_on_actor_restart)

    # Joy & Social Integration
    try:
        from skills.joy_social_integration import integrate_joy_social
        # We integrate without explicit config to use MockAdapters by default
        # unless user has set environment variables.
        integrate_joy_social(orchestrator)
        logger.info("🌟 Joy & Social systems integrated into startup sequence.")
    except ImportError:
        logger.warning("⚠️ JoySocial skills not found — skipping integration.")
    except Exception as exc:
        logger.error("❌ Failed to integrate JoySocial: %s", exc)

    # Apply Consciousness, Response, and SafeMode Genesis Patches
    try:
        from core.consciousness.apply_patches import apply_consciousness_patches
        from core.apply_response_patches import apply_response_patches
        from core.safe_mode import apply_orchestrator_patches

        apply_consciousness_patches(orchestrator)
        apply_response_patches()
        # Activate the dynamic autonomy bridge
        apply_orchestrator_patches(orchestrator)
        logger.info("🛡️ [GENESIS] Autonomy bridge and stability patches active.")
    except Exception as exc:
        logger.error("❌ Failed to apply gap-closing patches: %s", exc)


async def _boot_runtime_orchestrator(
    *,
    ready_label: str,
    readiness_context: Optional[str] = None,
):
    """Canonical runtime boot path shared by CLI/server/desktop surfaces."""
    from core.orchestrator import create_orchestrator
    from core.container import ServiceContainer

    orchestrator = create_orchestrator()
    await bootstrap_aura(orchestrator)

    # ── Morphogenetic self-organization runtime ───────────────────────
    # Starts bounded cell ecology, metabolism, and organ stabilizer.
    # Must boot after ServiceContainer is populated (bootstrap_aura)
    # and before orchestrator enters long-running loops.
    try:
        from core.morphogenesis.integration import start_morphogenesis_runtime
        await start_morphogenesis_runtime()
        logger.info("🧬 Morphogenetic self-organization runtime online.")
    except Exception as morph_exc:
        # Never block boot. Morphogenesis is an adaptive layer, not the boot root.
        logger.warning("Morphogenetic runtime startup skipped/degraded: %s", morph_exc)

    await orchestrator.start()

    if readiness_context and hasattr(orchestrator, "_ensure_inference_gate_ready"):
        await orchestrator._ensure_inference_gate_ready(context=readiness_context)

    # Register the runtime singletons that the ServiceManifest names as
    # canonical owners but that did not previously live in ServiceContainer.
    # Done before lock_registration so they show up to the manifest check.
    _register_runtime_singletons(orchestrator)

    ServiceContainer.lock_registration()
    _enforce_service_manifest(ready_label)
    await _enforce_boot_probes(ready_label)
    logger.info("🛡️ Registry Locked. Aura Ready (%s).", ready_label)

    # ── Wire viability + self-healing + stem cells + boot phases ───────
    # All four subsystems live in core/ and are independent of the
    # orchestrator's existing lifecycle. We start them here so they are
    # active for the same lifetime as the orchestrator.
    try:
        from core.organism.viability import get_viability
        await get_viability().start(interval=5.0)
        ServiceContainer.register_instance("viability", get_viability(), required=False)
    except Exception as exc:
        logger.warning("viability engine start failed: %s", exc)

    try:
        from core.runtime.self_healing import get_healer
        healer = get_healer()
        # Watch the orchestrator main loop; the orchestrator is expected
        # to call `healer.heartbeat("orchestrator")` on every tick. If
        # the heartbeat goes stale by 2.5x its expected interval, the
        # healer asks the orchestrator to restart_async() (no-op if
        # the method isn't defined — falls back to ServiceContainer).
        healer.watch("orchestrator", expected_interval_s=15.0, container_key="orchestrator")
        healer.watch("agency_bus",   expected_interval_s=30.0, container_key="agency_bus")
        healer.watch("phi_core",     expected_interval_s=30.0, container_key="phi_core")
        healer.watch("memory_facade", expected_interval_s=60.0, container_key="memory_facade")
        await healer.start(interval=5.0)
        ServiceContainer.register_instance("self_healing", healer, required=False)
    except Exception as exc:
        logger.warning("self-healing watcher start failed: %s", exc)

    try:
        from core.runtime.boot_phases import get_boot_phases
        bp = get_boot_phases()
        # Reflect the post-boot ready state into the boot panel.
        bp.update_organ("core", "ready")
        bp.update_organ("memory", "ready")
        bp.update_organ("cortex", "ready")
        bp.update_organ("voice", "waiting")
        bp.update_organ("autonomy", "ready")
        ServiceContainer.register_instance("boot_phases", bp, required=False)
    except Exception as exc:
        logger.debug("boot phases hook skipped: %s", exc)

    try:
        from core.runtime.performance_guard import get_guard
        await get_guard().start(interval=5.0)
        ServiceContainer.register_instance("performance_guard", get_guard(), required=False)
    except Exception as exc:
        logger.debug("performance guard start skipped: %s", exc)

    # Capture stem-cell snapshots for the load-bearing organs so the
    # immune layer has something to revert to if a future mutation
    # corrupts them. Snapshots are HMAC-signed in
    # ~/.aura/data/stem_cells/.
    try:
        from core.resilience.stem_cell import get_registry
        reg = get_registry()
        will = ServiceContainer.get("unified_will", default=None) or ServiceContainer.get("will", default=None)
        if will is not None:
            reg.register("unified_will")
            reg.capture("unified_will", will, schema_version="1")
        from core.agency.agency_orchestrator import get_orchestrator as _get_ao
        ao = _get_ao()
        reg.register("agency_orchestrator")
        reg.capture("agency_orchestrator", ao, schema_version="1")
        from core.identity.self_object import get_self
        reg.register("self_object")
        reg.capture("self_object", get_self().snapshot().continuity_hash, schema_version="1")
        ServiceContainer.register_instance("stem_cell_registry", reg, required=False)
    except Exception as exc:
        logger.warning("stem-cell capture at boot failed: %s", exc)

    return orchestrator


def _register_runtime_singletons(orchestrator: Any) -> None:
    """Register module-level singletons + orchestrator-attached components
    with ServiceContainer so the manifest verification finds canonical owners.

    Surfaces registered:
      - task_tracker / task_supervisor (alias)
      - shutdown_coordinator
      - output_gate (orchestrator attribute -> registry)
      - aura_runtime / orchestrator (alias)
    """
    try:
        from core.container import ServiceContainer
    except Exception:
        return

    try:
        from core.utils.task_tracker import get_task_tracker

        tracker = get_task_tracker()
        if not ServiceContainer.has("task_tracker"):
            ServiceContainer.register_instance("task_tracker", tracker, required=False)
        if not ServiceContainer.has("task_supervisor"):
            ServiceContainer.register_instance("task_supervisor", tracker, required=False)
    except Exception as exc:
        logger.debug("task_tracker registration skipped: %s", exc)

    try:
        from core.runtime.shutdown_coordinator import get_shutdown_coordinator

        coord = get_shutdown_coordinator()
        if not ServiceContainer.has("shutdown_coordinator"):
            ServiceContainer.register_instance("shutdown_coordinator", coord, required=False)
    except Exception as exc:
        logger.debug("shutdown_coordinator registration skipped: %s", exc)

    output_gate = getattr(orchestrator, "output_gate", None)
    if output_gate is not None:
        try:
            if not ServiceContainer.has("output_gate"):
                ServiceContainer.register_instance("output_gate", output_gate, required=False)
        except Exception as exc:
            logger.debug("output_gate registration skipped: %s", exc)

    try:
        if not ServiceContainer.has("orchestrator"):
            ServiceContainer.register_instance("orchestrator", orchestrator, required=False)
        if not ServiceContainer.has("aura_runtime"):
            ServiceContainer.register_instance("aura_runtime", orchestrator, required=False)
    except Exception as exc:
        logger.debug("orchestrator registration skipped: %s", exc)


async def _enforce_boot_probes(ready_label: str) -> None:
    """Run behavioral boot probes. In strict mode any failure aborts boot.

    The probe set covers the audit's required surfaces: memory write/read,
    state mutate/read, governance approve/deny, output gate, event bus, and
    actor supervisor. Surface-level probes (output_gate, event_bus,
    actor_supervisor) verify the contract is wired but do not require live
    backends.
    """
    try:
        from core.runtime.boot_probes import run_boot_probes
    except Exception as exc:
        logger.debug("boot_probes module unavailable: %s", exc)
        return
    strict_mode = os.environ.get("AURA_STRICT_RUNTIME") == "1"
    try:
        report = await run_boot_probes(strict=strict_mode)
        for r in report.results:
            if not r.ok:
                logger.warning(
                    "Boot probe %s failed during %s boot: %s",
                    r.name, ready_label, r.detail,
                )
    except Exception as exc:
        if strict_mode:
            raise
        logger.warning(
            "Boot probes raised in non-strict mode (%s); continuing degraded: %s",
            ready_label, exc,
        )


def _enforce_service_manifest(ready_label: str) -> None:
    """Verify ServiceManifest invariants once the registry is locked.

    In strict runtime, a critical violation aborts boot. Otherwise the
    violation is logged so operators can see the drift without forcing
    a desktop crash.
    """
    try:
        from core.runtime.service_manifest import (
            SERVICE_MANIFEST,
            critical_violations,
            verify_manifest,
        )
        from core.container import ServiceContainer
    except Exception as exc:  # pragma: no cover - defensive import
        logger.debug("ServiceManifest unavailable during boot: %s", exc)
        return

    snapshot: dict = {}
    for role in SERVICE_MANIFEST.values():
        for candidate in (role.canonical_owner, *role.aliases):
            if ServiceContainer.has(candidate):
                instance = ServiceContainer.get(candidate, default=None)
                if instance is not None:
                    snapshot[candidate] = instance

    violations = verify_manifest(snapshot)
    crit = critical_violations(violations)
    if violations:
        for v in violations:
            logger.warning(
                "ServiceManifest %s violation [%s]: %s",
                v.severity,
                v.role,
                v.reason,
            )
    strict_mode = os.environ.get("AURA_STRICT_RUNTIME", "0") == "1"
    if crit and strict_mode:
        raise RuntimeError(
            f"AURA_STRICT_RUNTIME: ServiceManifest critical violations during {ready_label} boot: "
            + "; ".join(f"{v.role}: {v.reason}" for v in crit)
        )

async def run_console():
    """Interactive CLI Mode"""
    orchestrator = await _boot_runtime_orchestrator(ready_label="CLI")

    from core.main import conversation_loop
    await conversation_loop(orchestrator=orchestrator)

async def run_server_async(host: str, port: int):
    """API Server Mode (Unified Loop)"""
    import uvicorn
    from interface.server import app as fastapi_app
    logger.info("🚀 Starting API Server on %s:%s", host, port)
    
    server_config = uvicorn.Config(
        fastapi_app, host=host, port=port, log_level="info"
    )
    server = uvicorn.Server(server_config)
    await server.serve()

import httpx # Added import for httpx

async def _wait_for_server_http(url: str, timeout: float = 60.0) -> bool:
    """Wait for internal API server to return 200 OK and report ready status."""
    start = time.time()
    
    logger.info("📡 Waiting for API Server health check: %s", url)
    count = 0
    while time.time() - start < timeout:
        count += 1
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "").lower()
                    ready = bool(data.get("ready"))
                    logger.info("📡 API Health status received: '%s'", status)
                    if ready or status in ("online", "operational", "healthy", "ok", "ready"):
                        logger.info("✅ API Server is ONLINE and HEALTHY after %ds.", int(time.time() - start))
                        return True
                    else:
                        logger.warning("📡 API Server status is '%s', not yet 'online'. Full data: %s", status, data)
                else:
                    logger.warning("📡 API Server returned HTTP %d", response.status_code)
        except httpx.ConnectError:
            if count % 10 == 0:
                 logger.info("📡 API Server not yet listening (Attempt %d)...", count)
        except Exception as e:
            logger.error("📡 API Health check probe FAILURE: %s", e)

        await asyncio.sleep(1.0)
        
    logger.error("❌ API Server health check TIMEOUT after %.1fs", timeout)
    return False

async def run_desktop(port: int):
    """GUI Mode (Managed Actor Process)"""
    from core.container import ServiceContainer
    from core.supervisor.tree import ActorSpec
    from interface.gui_actor import gui_actor_entry
    from core.utils.task_tracker import get_task_tracker
    
    supervisor = get_supervisor_tree()
    tracker = get_task_tracker()
    
    # 1. Start API Server (v21: Server now runs in Kernel)
    # [STABILITY] Start API immediately so port 8000 binds while brain thaws.
    import uvicorn
    from interface.server import app as fastapi_app
    host = "127.0.0.1" if config.security.internal_only_mode else "0.0.0.0"

    def _serve_api_sync():
        """Synchronous wrapper for uvicorn to run in a thread."""
        try:
            # Re-import inside thread to avoid loop issues
            import uvicorn
            from interface.server import app as _app
            server_config = uvicorn.Config(_app, host=host, port=port, log_level="info", loop="asyncio")
            server_config.handle_signals = False
            server = uvicorn.Server(server_config)
            logger.info("🚀 API Server (Kernel Thread) starting on port %s...", port)
            server.run()
            logger.info("📡 API Server thread has exited.")
        except Exception as e:
            logger.critical("🛑 API THREAD CRITICAL FAILURE: %s", e, exc_info=True)

    async def _run_api_server():
        logger.info("📡 API Server task starting (offloading to thread for M1 stability)...")
        await asyncio.to_thread(_serve_api_sync)

    async def _main_loop():
        from core.container import ServiceContainer
        
        # 1. Initialize Orchestrator and wait for boot
        logger.info("🧠 Orchestrator boot beginning...")
        orchestrator = await _boot_runtime_orchestrator(
            ready_label="Desktop",
            readiness_context="server_boot",
        )
        tracker.create_task(orchestrator.run(), name="OrchestratorMainLoop")

        # 2. Start API Server (v21: Server now runs in Kernel)
        # [STABILITY] Start API after brain is ready to ensure correct ServiceContainer lookups.
        logger.info("🎬 [DEBUG] Pre-starting API server mission...")
        api_task = tracker.create_task(_run_api_server(), name="api_server")
        logger.info("🎬 [DEBUG] API server task created successfully.")
        
        # Wait for API server to be TRULY ready (HTTP 200)
        # This prevents the GUI from launching too early and hitting "Connection Refused".
        health_url = f"http://127.0.0.1:{port}/api/health/boot"
        logger.info("⏳ Waiting for API health check on port %s...", port)
        if await _wait_for_server_http(health_url, 30.0):
            logger.info("✅ API Server is HEALTHY. Proceeding to GUI launch.")
        else:
            logger.warning("⚠️ API Server health check timed out after 30s. GUI launch may be degraded.")
        
        # 3. Start GUI Actor (WebView Only)
        # On macOS, we use subprocess.Popen instead of multiprocessing to avoid XPC/Cocoa deadlocks
        if sys.platform == "darwin":
            logger.info("🎨 Launching GUI via SUBPROCESS for macOS stability...")
            
            async def _gui_reaper_loop():
                """Re-implements supervision for the subprocess-based macOS GUI."""
                max_restarts = 5
                restart_count = 0
                while restart_count < max_restarts:
                    proc = await asyncio.create_subprocess_exec(
                        _launcher_python_executable(), "interface/gui_actor.py", str(port),
                        cwd=str(PROJECT_ROOT),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        start_new_session=True
                    )
                    logger.info("🎨 GUI Process Started (PID: %s)", proc.pid)
                    
                    async def _stream_logger(stream, level):
                        content = []
                        while True:
                            line = await stream.readline()
                            if not line: break
                            decoded = line.decode('utf-8', errors='replace').rstrip()
                            if decoded:
                                if level == "ERROR": logger.error(f"[GUI] {decoded}")
                                else: logger.debug(f"[GUI] {decoded}")
                                content.append(decoded)
                        return "\n".join(content)

                    out_task = tracker.create_task(
                        _stream_logger(proc.stdout, "DEBUG"),
                        name="gui.stdout_stream",
                    )
                    err_task = tracker.create_task(
                        _stream_logger(proc.stderr, "ERROR"),
                        name="gui.stderr_stream",
                    )
                    
                    # Watch for exit
                    while proc.returncode is None:
                        # Check for system-wide shutdown
                        if not getattr(supervisor, "_is_running", True):
                            try:
                                proc.terminate()
                            except ProcessLookupError:
                                pass
                            return
                        
                        try:
                            # Wait with timeout to allow checking shutdown flag
                            await asyncio.wait_for(proc.wait(), timeout=2.0)
                        except asyncio.TimeoutError:
                            pass
                    
                    # Ensure stream reading completes
                    await out_task
                    stderr_output = await err_task
                    
                    if proc.returncode == 0:
                        # User closed the window cleanly — treat this as "quit
                        # Aura", not "restart the GUI in the background".
                        # Otherwise the orchestrator stays alive pinned to
                        # MLX workers while the user believes they've quit,
                        # which is how "multiple versions in the background"
                        # happens.
                        logger.info("🎨 GUI closed by user — initiating full shutdown.")
                        try:
                            supervisor._is_running = False
                        except Exception:
                            pass
                        # Signal the main process so the outer event loop
                        # cancels its pending tasks and runs shutdown hooks.
                        try:
                            os.kill(os.getpid(), signal.SIGTERM)
                        except Exception:
                            pass
                        return

                    restart_count += 1
                    logger.critical(f"🛑 GUI Process crashed (code: {proc.returncode}). Reason:\n{stderr_output}")
                    logger.warning(f"🎨 Restarting GUI in 5s... (Attempt {restart_count}/{max_restarts})")
                    await asyncio.sleep(5.0)

            tracker.create_task(_gui_reaper_loop(), name="gui_reaper")
            pipe = None # Subprocess doesn't use the actor pipe
        else:
            # Linux/Others can still use the supervised actor
            spec = ActorSpec(
                name="desktop_gui",
                entry_point=gui_actor_entry,
                args=(port,),
                restart_policy="always"
            )
            supervisor.add_actor(spec)
            pipe = supervisor.start_actor("desktop_gui")
        
        # 4. Register GUI in ActorBus
        actor_bus = ServiceContainer.get("actor_bus", default=None)
        if actor_bus:
            actor_bus.add_actor("desktop_gui", pipe, is_child=True)
            
        logger.info("🎨 Desktop GUI Actor launched and supervised (WebView-only mode).")
        
        # Wait on long-lived tasks to prevent premature exit
        # We also wait on the orchestrator's main loop if we created one
        orchestrator_loop_task = [t for t in asyncio.all_tasks() if t.get_name() == "OrchestratorMainLoop"]
        tasks_to_wait = [api_task]
        if orchestrator_loop_task: tasks_to_wait.append(orchestrator_loop_task[0])
        
        # Actually wait forever for the supervisor
        await supervisor.wait_forever()

    await _main_loop()

async def run_watchdog():
    """Stability Watchdog Loop (Async)."""
    logger.info("🛡️ Watchdog supervisor active (supervision-only mode).")

    # Write watchdog PID so it can be stopped
    lock_file = Path.home() / ".aura" / "locks" / "watchdog.lock"
    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_file, "w") as f:
            f.write(str(os.getpid()))
    except Exception:
        pass
    try:
        restart_count = 0
        while restart_count < 10:
            logger.info("🛡️  Watchdog: Launching Aura (Attempt %s)", restart_count+1)
            start_time = time.time()
            
            # Use asyncio.create_subprocess_exec for non-blocking wait
            proc = await asyncio.create_subprocess_exec(_launcher_python_executable(), __file__, "--cli")
            await proc.wait()
            
            # Perplexity Audit Fix: Detect deterministic config errors (Exit 1)
            if proc.returncode == 1:
                logger.error("🛡️  Watchdog: Fatal configuration or security error (Code 1). Aborting restart loop.")
                break

            if proc.returncode == 0:
                logger.info("Clean shutdown detected. Watchdog exiting.")
                break

            # If ran for > 10 mins, reset counter
            if time.time() - start_time > 600:
                restart_count = 0
                
            restart_count += 1
            # Exponential backoff: 5, 10, 20, 40, 60...
            delay = min(60, 5 * (2 ** max(0, restart_count - 1)))
            logger.warning("Crash detected (Code: %s). Restarting in %ss...", proc.returncode, delay)
            await asyncio.sleep(delay)
    finally:
        try:
            lock_file.unlink(missing_ok=True)
        except Exception:
            pass

# ---------------------------------------------------------------------------
from core.utils.singleton import acquire_instance_lock, release_instance_lock

def _reap_orphaned_aura_processes() -> int:
    """Kill any stale Aura main processes owned by this user before taking the
    singleton lock.

    Why: the singleton lock file is cleared on clean exit, but if a previous
    launch hard-crashed (SIGKILL, power cut, WebView hang that left the
    supervisor orphaned) the orchestrator process can live on as a headless
    background job consuming 3–6 GB of RAM and pinning MLX worker subprocesses.
    Users then relaunch, see the new window, close it, and end up with two (or
    three, or five) full Aura stacks running simultaneously — which is exactly
    what the "multiple versions eating up resources" report describes.

    We match processes conservatively: only Python interpreters whose argv
    contains aura_main.py and whose PID is NOT ours or our parent's, and only
    processes owned by the current user. TERM first, KILL after a short grace
    window.
    """
    if sys.platform not in ("darwin", "linux"):
        return 0
    me = os.getpid()
    parent = os.getppid()
    killed = 0
    try:
        out = subprocess.check_output(
            ["ps", "-axo", "pid=,user=,command="], text=True, timeout=5
        )
    except Exception:
        return 0
    current_user = os.environ.get("USER") or ""
    stale_pids: list[int] = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        user = parts[1]
        cmd = parts[2]
        if pid in (me, parent):
            continue
        if current_user and user != current_user:
            continue
        if "aura_main.py" not in cmd:
            continue
        # Skip launcher/reaper children; we only want the top-level orchestrator.
        if "gui_actor.py" in cmd or "reaper" in cmd.lower():
            continue
        stale_pids.append(pid)
    for pid in stale_pids:
        try:
            os.kill(pid, signal.SIGTERM)
            killed += 1
        except ProcessLookupError:
            continue
        except PermissionError:
            continue
    if stale_pids:
        time.sleep(1.5)
        for pid in stale_pids:
            try:
                os.kill(pid, 0)  # still alive?
            except OSError:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        logger.warning(
            "🧹 Reaped %d orphaned Aura process(es) before boot: %s",
            killed, stale_pids,
        )
    return killed


def bootstrap_lock(skip_lock: bool = False):
    """Bridge to the shared singleton utility."""
    # Clean up orphaned stacks from prior hard-crashes before grabbing the lock.
    _reap_orphaned_aura_processes()
    acquire_instance_lock(lock_name="orchestrator", skip_lock=skip_lock)

def stop_aura():
    """Reads PID from lock file and sends SIGTERM. Also unloads launchd agent."""
    lock_file = Path.home() / ".aura" / "locks" / "orchestrator.lock"
    
    # 1. Unload Launchd Agent (Prevents auto-revival on macOS)
    if sys.platform == "darwin":
        plist_path = Path.home() / "Library/LaunchAgents/com.aura.daemon.plist"
        if plist_path.exists():
            logger.info("Unloading launchd daemon to prevent auto-revival...")
            try:
                subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True, timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("launchctl unload timed out.")
            except Exception as e:
                logger.warning("launchctl unload failed: %s", e)
            
    if not lock_file.exists():
        print("Aura does not appear to be running (no lock file found).")
        return

    try:
        with open(lock_file, "r") as f:
            pid_str = f.read().strip()
            if not pid_str:
                print("Lock file found but no PID recorded.")
                return
            pid = int(pid_str)
        
        # Perplexity Audit Fix: Verify the PID actually belongs to Aura
        try:
            import psutil
            proc = psutil.Process(pid)
            cmdline = " ".join(proc.cmdline()).lower()
            if "aura" not in cmdline and "python" not in cmdline:
                print(f"⚠️  Lock file PID {pid} does not appear to be an Aura process. Cleaning stale lock.")
                lock_file.unlink(missing_ok=True)
                return
        except (ImportError, psutil.NoSuchProcess):
            # If psutil is missing, we fall back to signal 0 check below
            pass
        except Exception as e:
            logger.debug("PID verification error: %s", e)

        print(f"Stopping Aura (PID: {pid})...")
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            print("Process already dead or inaccessible.")
            if lock_file.exists(): lock_file.unlink()
            return

        # Wait for cleanup
        for _ in range(10):
            try:
                os.kill(pid, 0) # Check if alive
            except (ProcessLookupError, PermissionError):
                break
            time.sleep(0.5)
        else:
            print("Aura is stubborn. Sending SIGKILL...")
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            
        # Force remove lock if still there
        if lock_file.exists():
            try:
                lock_file.unlink()
            except Exception:
                pass
            
        print("✅ Aura stopped successfully.")
    except Exception as e:
        print(f"Failed to stop Aura: {e}")

# ---------------------------------------------------------------------------
# Main Entry
# ---------------------------------------------------------------------------

def main():
    _maybe_relaunch_with_preferred_python()
    reaper_manifest_path = _ensure_reaper_manifest_env()

    parser = argparse.ArgumentParser(description="Aura Unified Entry Point")
    parser.add_argument("--cli", action="store_true", help="Interactive Console Mode")
    parser.add_argument("--server", action="store_true", help="API Server Mode")
    parser.add_argument("--headless", action="store_true", help="Headless local mode (API server only, no desktop GUI)")
    parser.add_argument("--desktop", action="store_true", help="Desktop GUI Mode")
    parser.add_argument("--gui-window", action="store_true", help="Open a desktop GUI window attached to an existing Aura server")
    parser.add_argument("--watchdog", action="store_true", help="Watchdog / Keep-alive Mode")
    parser.add_argument("--stop", action="store_true", help="Stop any running Aura instance")
    parser.add_argument("--reboot", action="store_true", help="Force cleanup and restart (Standardize)")
    parser.add_argument("--port", type=int, default=8000, help="Port for Server/GUI")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for Server")
    parser.add_argument("--skeletal", action="store_true", help="Skeletal Mode: Bypass heavy subsystems")
    
    args = parser.parse_args()
    
    # Standardize: Reboot behavior
    if args.stop:
        stop_aura()
        sys.exit(0)

    if args.skeletal:
        logger.info("💀 SKELETAL MODE ACTIVATED")
        config.skeletal_mode = True
        # Force-disable heavy components in config
        if hasattr(config, "soma"):
            config.soma.enabled = False
        config.features.mycelium_visualizer = False
        config.features.autonomous_impulses = False

    if args.reboot:
        logger.info("🔄 REBOOT SEQUENCE ACTIVATED")
        clean_artifacts()
        stop_aura()
        time.sleep(1.0)
        # Default to desktop if no other mode specified
        if not (args.cli or args.server or args.desktop):
            args.desktop = True

    if args.headless:
        logger.info("🖥️ HEADLESS MODE ACTIVATED")
        args.server = True
        args.desktop = False
        args.gui_window = False
        # Headless demo mode should stay local even when public API mode is enabled.
        args.host = "127.0.0.1"

    if not args.gui_window:
        check_environment()
    
    # uvloop activation is opt-in on macOS because native media stacks have
    # repeatedly crashed under the libuv event loop during desktop boot.
    from core.runtime.boot_safety import uvloop_allowed

    if not args.gui_window and uvloop_allowed():
        try:
            import uvloop
            # We'll set the policy inside the setup phase if needed,
            # but asyncio.run handles loop creation itself in 3.11+.
            # Setting policy here is safe as long as no loop exists.
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            logger.info("⚡ uvloop activated for maximum concurrency performance.")
        except ImportError:
            logger.debug("uvloop not installed. Using standard asyncio event loop.")
        except Exception as e:
            logger.warning("Could not set uvloop policy: %s", e)
    else:
        logger.info(
            "🛡️ uvloop disabled for this runtime profile. "
            "Set AURA_ENABLE_UVLOOP=1 to force-enable it."
        )
    
    # Do not acquire lock for watchdog, since the watchdog itself runs the child process
    if not args.gui_window:
        bootstrap_lock(skip_lock=args.watchdog)

    # Only the lock owner is allowed to reclaim ports or spawn the reaper.
    # This prevents a second boot attempt from disrupting a healthy live Aura
    # instance before the singleton fence has a chance to reject it.
    if not args.cli and not args.gui_window and not args.watchdog:
        logger.info("🧹 Pre-clearing known ports...")
        kill_port(args.port)
        kill_port(10003, pattern="aura")

    # SIGKILL Reaper Initialization
    if not args.gui_window and not args.watchdog:
        try:
            from core.reaper import reaper_loop
            reaper_proc = multiprocessing.Process(
                target=reaper_loop, 
                args=(os.getpid(), reaper_manifest_path),
                daemon=True,
                name="AuraReaper"
            )
            reaper_proc.start()
            logger.info("🛡️  REAPER ACTIVE (Survives SIGKILL). Monitoring Kernel PID: %s", os.getpid())
        except (ImportError, Exception) as e:
            logger.error("⚠️ Reaper initialization skipped or failed: %s", e)

    # Perplexity Audit Fix: Use asyncio.run for cleaner entry points
    try:
        if args.server:
            # Dynamic host selection if default was used
            host = args.host
            if (
                host == "127.0.0.1"
                and not args.headless
                and not getattr(config.security, "internal_only_mode", False)
            ):
                host = "0.0.0.0"
            async def _run_server_with_bootstrap():
                orchestrator = await _boot_runtime_orchestrator(
                    ready_label="Server",
                    readiness_context="server_boot",
                )
                from core.utils.task_tracker import get_task_tracker

                get_task_tracker().create_task(orchestrator.run(), name="OrchestratorMainLoop")
                await run_server_async(host, args.port)
            asyncio.run(_run_server_with_bootstrap())
        elif args.desktop:
            # For desktop, we'll need a way to bootstrap the loop if uvicorn starts it
            # But Desktop mode in aura_main runs uvicorn in a thread.
            # We should probably bootstrap the main thread for the GUI if it needs it.
            asyncio.run(run_desktop(args.port))
        elif args.gui_window:
            from interface.gui_actor import gui_actor_entry

            logger.info("🪟 Opening Aura desktop window on port %s...", args.port)
            gui_actor_entry(args.port)
        elif args.watchdog:
            asyncio.run(run_watchdog())
        elif args.cli:
            
            asyncio.run(run_console())
        else:
            # Default fallback: Desktop if double-clicked, else CLI if terminal
            if sys.stdin and sys.stdin.isatty():
                asyncio.run(run_console())
            else:
                logger.info("Initializing in Desktop/Autonomy mode...")
                asyncio.run(run_desktop(args.port))
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user.")
    except Exception as e:
        logger.critical("FATAL BOOT ERROR: %s", e, exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
