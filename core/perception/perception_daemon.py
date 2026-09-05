"""Always-on PerceptionDaemon for Aura.

Maintains continuous environmental context (screen, window focus, clipboard, audio, entity tracking)
and updates a rolling perceptual memory.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import subprocess
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from core.container import ServiceContainer
from core.runtime.service_access import optional_service
from core.event_bus import EventPriority, get_event_bus
from core.governance_context import local_internal_governed_scope
from core.perception.frontmost_app import frontmost_app_name_fast
from core.perception.multimodal_sync import (
    Calibration,
    Modality,
    MultimodalSynchronizer,
    PerceptualClaim,
    PerceptualEvent,
    PrivacyClass,
    PrivacyPolicy,
)
from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind, declare
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.utils.task_tracker import get_task_tracker
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.PerceptionDaemon")

# The desktop perception loop polls the GUI (window focus, clipboard, browser
# tabs, `ps`) every check_interval by shelling out — ~4 subprocesses per cycle,
# ~7k/hour at a 2s cadence. On macOS that fork/exec churn grows the parent's
# malloc arena and the OS does not reclaim it, which 2026-07-13 tracemalloc
# attribution isolated as the dominant IDLE RSS leak (Python heap flat at
# ~18MB/h while RSS climbed ~350MB/h; every top native site routed through
# Popen). When the user has been idle past this threshold the loop backs off
# to the dormant interval; any detected activity snaps the cadence back.
_IDLE_BACKOFF_AFTER_FLAG = declare(
    "AURA_PERCEPTION_IDLE_BACKOFF_AFTER_S",
    kind=FlagKind.FLOAT,
    default=120.0,
    description=(
        "Seconds of user inactivity after which the desktop perception loop "
        "backs off to the dormant interval, ending the idle subprocess storm."
    ),
    owner="core/perception/perception_daemon.py",
)

_PERCEPTION_DAEMON_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    subprocess.SubprocessError,
)


class PerceptionDaemon:
    """Always-on perception loop and rolling short/medium term memory."""

    _instance: PerceptionDaemon | None = None
    _lock = asyncio.Lock()

    def __init__(self, *, check_interval_s: float = 2.0):
        self.check_interval = check_interval_s
        # Idle cadence when there is no GUI surface to perceive (headless/proof
        # runtime): the loop stays alive but skips the subprocess-spawning probes.
        self._dormant_interval_s = 45.0
        # Desktop idle backoff: once the user has been inactive this long, the
        # loop polls at the dormant cadence instead of check_interval so the
        # per-cycle subprocess storm stops while nothing is happening.
        self._idle_backoff_after_s = float(_IDLE_BACKOFF_AFTER_FLAG.value())
        self.running = False
        self._tasks: list[asyncio.Task] = []

        # Rolling buffers (thread-safe deques)
        self._short_term_buffer: deque[dict[str, Any]] = deque(maxlen=200)   # last ~5-10 mins
        self._medium_term_buffer: deque[dict[str, Any]] = deque(maxlen=2000) # last ~24 hours

        # Entity tracking
        self._entities: dict[str, dict[str, Any]] = {}
        self._entity_aliases: dict[str, str] = {}

        # Attention state
        self.user_focus = "unknown"
        self.aura_focus = "unknown"
        self.joint_attention_score = 0.5
        self.attention_lock = asyncio.Lock()

        # Telemetry & States
        self.user_active = True
        self.last_user_activity = time.time()
        self.last_clipboard_hash = ""
        self.last_active_window = ""
        self._last_screen_hash = ""
        self._synchronizer_sequence = 0

        # Privacy configs
        self.privacy_mode = False
        self.redacted_patterns = ["password", "token", "key", "secret", "private"]
        self._file_scan_root = Path(
            os.getenv("AURA_PERCEPTION_FILE_SCAN_ROOT", str(state_root()))
        )

        logger.info("📡 PerceptionDaemon initialized.")

    @classmethod
    async def get(cls) -> PerceptionDaemon:
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def get_sync(cls) -> PerceptionDaemon:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        logger.info("📡 PerceptionDaemon starting background sensory loops...")

        self._tasks.append(
            get_task_tracker().create_task(
                self._main_perceptual_loop(),
                name="perception_daemon.main",
            )
        )
        self._tasks.append(
            get_task_tracker().create_task(
                self._attention_alignment_loop(),
                name="perception_daemon.attention",
            )
        )

        logger.info("📡 PerceptionDaemon is ONLINE.")

    async def stop(self) -> None:
        self.running = False
        pending: list[asyncio.Task] = []
        for task in self._tasks:
            if not task.done():
                task.cancel()
                pending.append(task)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._tasks.clear()
        logger.info("📡 PerceptionDaemon is OFFLINE.")

    def register_moment(self, source: str, content: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Insert a perceptual moment, apply privacy filters, and publish to the EventBus."""
        now = time.time()
        meta = dict(metadata or {})
        
        # Privacy redact
        filtered_content = content
        if self.privacy_mode or any(p in content.lower() for p in self.redacted_patterns):
            filtered_content = "<redacted: privacy policy>"
            meta["redacted"] = True

        moment = {
            "moment_id": f"pmom-{uuid.uuid4().hex[:8]}",
            "timestamp": now,
            "source": source,
            "content": filtered_content,
            "metadata": meta,
        }

        # Add to buffers
        self._short_term_buffer.append(moment)
        self._medium_term_buffer.append(moment)
        try:
            self._publish_synchronized_moment(moment)
        except _PERCEPTION_DAEMON_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "perception_daemon.multimodal_sync",
                exc,
                severity="warning",
                action="retained daemon moment after canonical fusion bridge rejected it",
                enforce_failure_policy=False,
            )

        # Publish autonomic sensory event
        try:
            get_event_bus().publish_threadsafe(
                topic="aura/perception/moment",
                data=moment,
                priority=EventPriority.AUTONOMIC,
            )
        except _PERCEPTION_DAEMON_RECOVERABLE_ERRORS as e:
            record_degradation("perception_daemon.event_bus_publish", e)
            logger.debug("Daemon failed to publish sensory moment to EventBus: %s", e)

        return moment

    def _publish_synchronized_moment(self, moment: dict[str, Any]) -> None:
        """Bridge semantic daemon moments into the canonical evidence ledger."""

        synchronizer = optional_service("multimodal_synchronizer")
        if not isinstance(synchronizer, MultimodalSynchronizer):
            return
        source = str(moment.get("source") or "daemon")[:80]
        metadata = moment.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        content = str(moment.get("content") or "")
        content_digest = hashlib.sha256(
            content.encode("utf-8", errors="ignore")
        ).hexdigest()[:24]
        confidence = 0.65
        modality = Modality.TEXT
        claims: list[PerceptualClaim] = [
            PerceptualClaim(f"moment.{source}.digest", content_digest, 0.70),
        ]
        quality_flags = ["semantic_daemon_moment", "raw_content_not_retained"]

        if source == "window_focus":
            modality = Modality.SPATIAL
            app_name = str(metadata.get("app_name") or "")[:160]
            if app_name:
                claims.append(PerceptualClaim("spatial.foreground_app", app_name, 0.90))
            confidence = 0.82
        elif source == "clipboard":
            claims.append(
                PerceptualClaim(
                    "text.clipboard_char_count",
                    max(0, int(metadata.get("char_count", 0) or 0)),
                    0.95,
                )
            )
            quality_flags.append("clipboard_content_redacted_to_digest")
        elif source == "browser":
            tabs = metadata.get("tabs")
            claims.append(
                PerceptualClaim(
                    "browser.open_tab_count",
                    len(tabs) if isinstance(tabs, list) else 0,
                    0.80,
                )
            )
            quality_flags.append("browser_titles_and_urls_not_retained")
        elif source == "terminal":
            shells = metadata.get("shells")
            claims.append(
                PerceptualClaim(
                    "device.active_shell_count",
                    len(shells) if isinstance(shells, list) else 0,
                    0.80,
                )
            )
        elif source == "file_system":
            files = metadata.get("modified_files")
            claims.append(
                PerceptualClaim(
                    "text.recent_file_mutation_count",
                    len(files) if isinstance(files, list) else 0,
                    0.80,
                )
            )
            quality_flags.append("file_paths_not_retained")
        elif source == "microphone":
            modality = Modality.AUDIO
            claims.append(PerceptualClaim("audio.sensor_active", True, 0.70))
            confidence = 0.45
            quality_flags.append("sensor_status_not_audio_observation")
        elif source == "user_presence":
            modality = Modality.BODY
            idle_seconds = max(0.0, float(metadata.get("idle_seconds", 0.0) or 0.0))
            claims.extend(
                (
                    PerceptualClaim("user.idle_seconds", round(idle_seconds, 3), 0.70),
                    PerceptualClaim("user.present", idle_seconds <= 120.0, 0.65),
                )
            )
            confidence = 0.60
        if metadata.get("redacted"):
            quality_flags.append("source_content_redacted")

        self._synchronizer_sequence += 1
        observed_monotonic_ns = time.monotonic_ns()
        synchronizer.ingest(
            PerceptualEvent(
                event_id=f"daemon:{moment.get('moment_id', self._synchronizer_sequence)}",
                modality=modality,
                source=f"perception_daemon:{source}",
                sequence=self._synchronizer_sequence,
                observed_at=float(moment.get("timestamp") or time.time()),
                observed_monotonic_ns=observed_monotonic_ns,
                summary=f"redacted semantic moment from {source}"[:320],
                confidence=confidence,
                claims=tuple(claims[:16]),
                calibration=Calibration(
                    f"runtime:perception_daemon:{source}",
                    status="unknown",
                    reliability=0.75,
                ),
                provenance=("core.perception.perception_daemon", source),
                privacy=PrivacyPolicy(
                    classification=PrivacyClass.SENSITIVE,
                    retention="none",
                    redacted=True,
                ),
                quality_flags=tuple(quality_flags[:8]),
            )
        )

    async def _main_perceptual_loop(self) -> None:
        """Poll clipboard, active window, terminal, browser, and file changes continuously."""
        # A headless/proof runtime has no GUI surface to perceive. Running the
        # window/clipboard/browser/ps probes there spawns a subprocess storm
        # (osascript/pbpaste/ps) on every interval — the dominant source of RSS
        # churn and event-loop lag in the headless longevity soak — for zero
        # signal. Resolve once and stay dormant.
        try:
            from core.runtime.proof_policy import proof_headless_run

            headless = proof_headless_run()
        except (ImportError, RuntimeError, AttributeError):
            headless = False
        if headless:
            logger.info("📡 PerceptionDaemon: headless runtime — GUI perception dormant.")
        while self.running:
            try:
                if headless:
                    await asyncio.sleep(self._dormant_interval_s)
                    continue
                # Adaptive cadence: full speed while the user is active, dormant
                # once idle past the threshold. This is the fix for the measured
                # idle-RSS leak — the subprocess storm only runs when there is
                # plausibly something to perceive. Worst case on the user's
                # return is one dormant interval (~45s) of latency before Aura
                # notices, which is acceptable for an ambient companion.
                idle_time = time.time() - self.last_user_activity
                effective_interval = (
                    self._dormant_interval_s
                    if idle_time > self._idle_backoff_after_s
                    else self.check_interval
                )
                await asyncio.sleep(effective_interval)

                # 1. Active Window Focus Check (macOS)
                window = await self._check_active_window()
                if window and window != self.last_active_window:
                    self.last_active_window = window
                    self.register_moment(
                        source="window_focus",
                        content=f"User switched focus application to: {window}",
                        metadata={"app_name": window}
                    )
                    self.last_user_activity = time.time()
                    self.user_active = True

                # 2. Clipboard Change Check (macOS)
                clipboard = await self._check_clipboard()
                if clipboard:
                    clip_hash = hashlib.sha256(clipboard.encode("utf-8")).hexdigest()
                    if clip_hash != self.last_clipboard_hash:
                        self.last_clipboard_hash = clip_hash
                        snippet = clipboard[:200] + ("..." if len(clipboard) > 200 else "")
                        self.register_moment(
                            source="clipboard",
                            content=f"Clipboard changed: {snippet}",
                            metadata={"char_count": len(clipboard)}
                        )
                        self.last_user_activity = time.time()
                        self.user_active = True

                # 3. Browser Tab State Check
                try:
                    from core.capabilities.browser_controller import get_browser_controller
                    bc = get_browser_controller()
                    if bc and getattr(bc, "_started", False):
                        tabs = await bc.get_open_tabs()
                        if tabs:
                            tab_summary = ", ".join(f"{t.get('title')} ({t.get('url')})" for t in tabs[:3])
                            self.register_moment(
                                source="browser",
                                content=f"Active browser tabs: {tab_summary}",
                                metadata={"tabs": tabs}
                            )
                except _PERCEPTION_DAEMON_RECOVERABLE_ERRORS as e:
                    record_degradation("perception_daemon.browser_status", e)
                    logger.debug("Browser status check failed: %s", e)

                # 4. Terminal / Process State Check
                try:
                    with local_internal_governed_scope("perception_daemon.terminal_process", domain="tool_execution"):
                        proc = await get_subprocess_gateway().run_async(
                            ["ps", "-A", "-o", "comm"],
                            read_only=True,
                            timeout=1.0,
                            source="perception_daemon.terminal_process",
                            accelerator_capability="none",
                        )
                    if proc.returncode == 0:
                        lines = proc.stdout.splitlines()
                        running_shells = [
                            line for line in lines if any(shell in line for shell in ("zsh", "bash", "sh"))
                        ]
                        if running_shells:
                            self.register_moment(
                                source="terminal",
                                content=f"Active terminal shell processes: {len(running_shells)} running",
                                metadata={"shells": running_shells}
                            )
                except _PERCEPTION_DAEMON_RECOVERABLE_ERRORS as e:
                    record_degradation("perception_daemon.terminal_process", e)
                    logger.debug("Terminal process check failed: %s", e)

                # 5. File System Activity Watcher
                try:
                    recent_files = await asyncio.to_thread(
                        self._scan_recent_file_mutations,
                        self._file_scan_root,
                        effective_interval,
                    )
                    if recent_files:
                        self.register_moment(
                            source="file_system",
                            content=f"Detected local file mutations: {', '.join(recent_files[:3])}",
                            metadata={"modified_files": recent_files}
                        )
                except _PERCEPTION_DAEMON_RECOVERABLE_ERRORS as e:
                    record_degradation("perception_daemon.file_system", e)
                    logger.debug("File system check failed: %s", e)

                # 6. Ambient Microphone Status Check
                try:
                    ears = optional_service("ears")
                    if ears:
                        self.register_moment(
                            source="microphone",
                            content="Microphone engine is active & listening",
                            metadata={"ears_configured": True}
                        )
                except _PERCEPTION_DAEMON_RECOVERABLE_ERRORS as e:
                    record_degradation("perception_daemon.microphone_status", e)
                    logger.debug("Microphone loop failed: %s", e)

                # 7. User Idle State Assessment
                idle_time = time.time() - self.last_user_activity
                if idle_time > 120.0 and self.user_active:
                    self.user_active = False
                    self.register_moment(
                        source="user_presence",
                        content="User has become idle (inactive for >2 mins)",
                        metadata={"idle_seconds": idle_time}
                    )
                elif idle_time <= 10.0 and not self.user_active:
                    self.user_active = True
                    self.register_moment(
                        source="user_presence",
                        content="User has resumed activity",
                        metadata={"idle_seconds": idle_time}
                    )

            except asyncio.CancelledError:
                break
            except _PERCEPTION_DAEMON_RECOVERABLE_ERRORS as e:
                record_degradation("perception_daemon.main_loop", e)
                logger.debug("Error in PerceptionDaemon main loop: %s", e)
                await asyncio.sleep(self.check_interval * 2)

    async def _attention_alignment_loop(self) -> None:
        """Analyze rolling sensory moment topics to estimate shared attention."""
        while self.running:
            try:
                await asyncio.sleep(15.0)

                # Look at recent moments in last 30s to update joint attention
                recent = self.get_recent_moments(duration_seconds=30.0)
                async with self.attention_lock:
                    if recent:
                        sources = {m["source"] for m in recent}
                        if any(s in sources for s in ("window_focus", "clipboard", "screen_ocr")):
                            self.user_focus = self.last_active_window or "desktop"
                            self.joint_attention_score = min(1.0, self.joint_attention_score + 0.1)
                        else:
                            self.joint_attention_score = max(0.2, self.joint_attention_score - 0.05)
                    else:
                        self.joint_attention_score = max(0.1, self.joint_attention_score - 0.1)

            except asyncio.CancelledError:
                break
            except _PERCEPTION_DAEMON_RECOVERABLE_ERRORS as e:
                record_degradation("perception_daemon.attention_loop", e)
                logger.debug("Error in PerceptionDaemon attention loop: %s", e)
                await asyncio.sleep(15.0)

    def _scan_recent_file_mutations(self, workspace: Path, interval_s: float) -> list[str]:
        """Return recent file mutations without blocking the asyncio loop.

        The live desktop daemon used to run an unbounded ``os.walk`` directly
        on the event loop. On a real Aura install, ``~/.aura`` can contain the
        source tree, virtualenvs, artifacts, caches, and model metadata. This
        scanner is intentionally bounded so perception stays useful without
        becoming a foreground latency source.
        """
        try:
            max_files = max(50, int(os.getenv("AURA_PERCEPTION_FILE_SCAN_MAX_FILES", "1500") or 1500))
        except (TypeError, ValueError):
            max_files = 1500
        try:
            max_seconds = max(0.05, float(os.getenv("AURA_PERCEPTION_FILE_SCAN_MAX_SECONDS", "0.5") or 0.5))
        except (TypeError, ValueError):
            max_seconds = 0.5

        excluded_dirs = {
            ".cache",
            ".git",
            ".hg",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".tox",
            ".venv",
            "__pycache__",
            "cache",
            "Caches",
            "dist",
            "logs",
            "model_cache",
            "models",
            "node_modules",
            "venv",
        }
        if os.getenv("AURA_PERCEPTION_SCAN_ARTIFACTS", "").strip().lower() not in {"1", "true", "yes"}:
            excluded_dirs.add("artifacts")

        workspace = Path(workspace).expanduser()
        if not workspace.exists():
            return []

        started = time.monotonic()
        now = time.time()
        cutoff = now - max(float(interval_s or self.check_interval), 0.1)
        scanned = 0
        recent_files: list[str] = []

        for root, dirs, files in os.walk(workspace, topdown=True, followlinks=False):
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".") and d not in excluded_dirs
            ]
            for file in files:
                if file.startswith("."):
                    continue
                scanned += 1
                if scanned > max_files or (time.monotonic() - started) > max_seconds:
                    return recent_files

                fp = Path(root) / file
                try:
                    mtime = fp.stat(follow_symlinks=False).st_mtime
                except FileNotFoundError:
                    continue
                except OSError as e:
                    record_degradation("perception_daemon.file_stat", e)
                    logger.debug("File stat check failed for %s: %s", fp, e)
                    continue

                if mtime >= cutoff:
                    recent_files.append(str(fp))

        return recent_files

    @staticmethod
    def _describe_file_status(path: Path) -> str | None:
        if not path.exists():
            return None
        stat = path.stat()
        return f"File {path.name} exists, size={stat.st_size} bytes, modified={stat.st_mtime}"

    async def _check_clipboard(self) -> str | None:
        try:
            with local_internal_governed_scope("perception_daemon.clipboard", domain="tool_execution"):
                proc = await get_subprocess_gateway().run_async(
                    ["pbpaste"],
                    read_only=True,
                    timeout=1.0,
                    source="perception_daemon.clipboard",
                    accelerator_capability="none",
                )
            if proc.returncode == 0:
                return proc.stdout.strip()
            return None
        except _PERCEPTION_DAEMON_RECOVERABLE_ERRORS as e:
            logger.debug("Clipboard check failed: %s", e)
            return None

    async def _check_active_window(self) -> str | None:
        # Fast path: in-process NSWorkspace lookup (no fork). This runs on a ~500ms
        # cadence, so avoiding an osascript subprocess per poll matters.
        fast = frontmost_app_name_fast()
        if fast:
            return fast
        try:
            with local_internal_governed_scope("perception_daemon.active_window", domain="tool_execution"):
                cmd = ["osascript", "-e", 'tell application "System Events" to get name of first application process whose frontmost is true']
                proc = await get_subprocess_gateway().run_async(
                    cmd,
                    read_only=True,
                    timeout=1.5,
                    source="perception_daemon.active_window",
                    accelerator_capability="auto",
                )
            if proc.returncode == 0:
                return proc.stdout.strip()
            return None
        except _PERCEPTION_DAEMON_RECOVERABLE_ERRORS as e:
            logger.debug("Active window check failed: %s", e)
            return None

    # --- Public API surface ------------------------------------------------

    def get_recent_moments(self, source: str | None = None, duration_seconds: float = 300.0) -> list[dict[str, Any]]:
        now = time.time()
        cutoff = now - duration_seconds
        res = [m for m in self._short_term_buffer if m["timestamp"] >= cutoff]
        if not res and duration_seconds > 300.0:
            res = [m for m in self._medium_term_buffer if m["timestamp"] >= cutoff]
        
        if source:
            res = [m for m in res if m["source"] == source]
        return res

    def track_entity(self, entity_type: str, name: str, metadata: dict[str, Any] | None = None) -> str:
        """Track/retrieve stable ID for files, browser tabs, tasks, users, etc."""
        alias_key = f"{entity_type}::{name}".lower()
        if alias_key in self._entity_aliases:
            entity_id = self._entity_aliases[alias_key]
            self._entities[entity_id]["last_seen"] = time.time()
            if metadata:
                self._entities[entity_id]["metadata"].update(metadata)
            return entity_id

        entity_id = f"ent-{uuid.uuid4().hex[:8]}"
        self._entity_aliases[alias_key] = entity_id
        self._entities[entity_id] = {
            "entity_id": entity_id,
            "type": entity_type,
            "name": name,
            "created_at": time.time(),
            "last_seen": time.time(),
            "metadata": dict(metadata or {}),
        }
        logger.info("🆕 Tracking entity: %s (type=%s, ID=%s)", name, entity_type, entity_id)
        return entity_id

    async def active_perceive(self, probe_type: str, query: str | None = None) -> dict[str, Any]:
        """Force a perception probe like a manual screen capture or file verification."""
        logger.info("🔍 Active perception triggered: probe_type=%s, query=%s", probe_type, query)
        
        if probe_type == "screen_ocr":
            vision = optional_service("vision_engine")
            if vision and hasattr(vision, "analyze_moment"):
                try:
                    desc = await vision.analyze_moment(prompt=query or "Describe current text contents.")
                    self.register_moment(source="active_ocr", content=desc)
                    return {"ok": True, "result": desc}
                except _PERCEPTION_DAEMON_RECOVERABLE_ERRORS as e:
                    record_degradation("perception_daemon.active_ocr", e)
                    return {"ok": False, "error": str(e)}
            return {"ok": False, "error": "vision_engine_unavailable"}
            
        elif probe_type == "file_status":
            if not query:
                return {"ok": False, "error": "missing file path query"}
            p = Path(query)
            desc = await asyncio.to_thread(self._describe_file_status, p)
            if desc:
                self.register_moment(source="active_file_check", content=desc, metadata={"path": str(p)})
                return {"ok": True, "result": desc}
            return {"ok": False, "error": "file_not_found"}

        return {"ok": False, "error": f"unsupported probe type: {probe_type}"}


def get_perception_daemon() -> PerceptionDaemon:
    return PerceptionDaemon.get_sync()
