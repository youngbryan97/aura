"""StallWatchdog: Async Event Loop Monitoring + Active Recovery
Part of Aura's Neural Neuro-Surgeon (Phase 29).

Design notes:
- The watchdog runs in its own daemon thread so it survives even if the
  asyncio loop is wedged.
- A heartbeat is scheduled via call_soon_threadsafe every second. If the
  loop fails to run that callback within `threshold` seconds, we record a
  stall.
- On a long stall we now do more than log: we (a) dump task state, (b)
  cancel asyncio tasks that look hung, and (c) signal subsystems to
  recycle. This is what turns "we noticed the freeze" into "we ended
  the freeze."
"""

import asyncio
import io
import logging
import os
import sys
import threading
import time
import traceback
from importlib import import_module
from pathlib import Path

from core.governance_context import local_internal_governed_scope
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.task_ownership import create_tracked_task
from core.runtime.flags import FlagKind as _FlagKind, declare as _declare_flag

# Declared flags (migrated from raw os.environ reads so the knobs are
# inventoried and reportable). STRING kind with the original literal
# default keeps read semantics byte-identical to os.environ.get.
_FLAG_EXTERNAL_GUI_OWNER = _declare_flag(
    "AURA_EXTERNAL_GUI_OWNER",
    kind=_FlagKind.STRING,
    default=None,
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_LIVENESS_HEARTBEAT_FILE = _declare_flag(
    "AURA_LIVENESS_HEARTBEAT_FILE",
    kind=_FlagKind.STRING,
    default="data/runtime/liveness_heartbeat.json",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
# AURA_LOG_DIR is declared once, in core.runtime.flags.aura_log_dir_override(),
# and read through forensics_root() below. The declaration that used to sit here
# survived after this module stopped reading it, and two declarations of one
# name with different specs make declare() raise — correctly: a knob must have
# exactly one meaning, and the flag registry is the thing that enforces it.
_FLAG_SAFE_BOOT_DESKTOP = _declare_flag(
    "AURA_SAFE_BOOT_DESKTOP",
    kind=_FlagKind.STRING,
    default=None,
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_WATCHDOG_CANCEL_HUNG_TASKS = _declare_flag(
    "AURA_WATCHDOG_CANCEL_HUNG_TASKS",
    kind=_FlagKind.STRING,
    default="",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_WATCHDOG_FOREGROUND_GRACE_S = _declare_flag(
    "AURA_WATCHDOG_FOREGROUND_GRACE_S",
    kind=_FlagKind.STRING,
    default="75",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_WATCHDOG_HARD_EXIT_BOOT_GRACE_S = _declare_flag(
    "AURA_WATCHDOG_HARD_EXIT_BOOT_GRACE_S",
    kind=_FlagKind.STRING,
    default=None,
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_WATCHDOG_SERVICE_PROOF_GRACE_S = _declare_flag(
    "AURA_WATCHDOG_SERVICE_PROOF_GRACE_S",
    kind=_FlagKind.STRING,
    default="240",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_WATCHDOG_SUPPRESSION_LOG_INTERVAL_S = _declare_flag(
    "AURA_WATCHDOG_SUPPRESSION_LOG_INTERVAL_S",
    kind=_FlagKind.STRING,
    default="60",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)


logger = logging.getLogger("Aura.Resilience.Watchdog")

_STALL_WATCHDOG_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _forensics_root() -> Path:
    """Where stall/wedge evidence lands.

    Live convention: repo-relative data/error_logs (crash_triage, the
    narrator, and operators all read there). Test runs set AURA_LOG_DIR —
    honoring it keeps hermetic runs from salting the real forensic record
    (58 test-driver dumps polluted a triage ranking on 2026-07-10).
    """
    # Anchored, not cwd-relative, and the AURA_LOG_DIR override now lives in
    # one place: forensics_root() applies it for every writer, so this lane
    # cannot honour the switch differently from the lane that reads it back.
    from core.utils.paths import forensics_root

    return forensics_root()


def _record_watchdog_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict | None = None,
) -> None:
    record_degradation(
        "stall_watchdog",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )

# How long an asyncio task can be pending (not done) before the watchdog
# considers it suspect during a stall and cancels it. Conservative — only
# fires after a confirmed stall, not as routine cleanup.
_TASK_HUNG_SECONDS = 90.0

# Minimum stall length to trigger active recovery. Below this, we just log.
_ACTIVE_RECOVERY_THRESHOLD = 30.0

# Absolute ceiling on CONTINUOUS event-loop unresponsiveness, immune to all
# stall suppression. A healthy runtime runs the watchdog heartbeat every
# second even during long generations (those run off the loop), so the loop
# only goes silent this long when it is genuinely wedged. Observed live: a
# steered 32B generation triggered a Metal GPU command-buffer hang that
# deadlocked every thread (all parked on a Metal semaphore) — the event loop
# died, in-process "active recovery" could not run (it schedules onto the dead
# loop), and even SIGTERM could not shut down. In-process recovery of a shared
# GPU-context deadlock is impossible, so the out-of-band watchdog thread — the
# only code still running — force-exits the process. The launchd KeepAlive
# supervisor (tools/install_supervisor.sh) restarts within ~15s with continuity
# from the periodic state-vault snapshots. 0 disables the ceiling.
_LOOP_WEDGE_HARD_EXIT_S = 150.0
# Distinct from memory_watchdog's categorized exit (70). Non-zero ⇒ the
# supervisor's KeepAlive/SuccessfulExit=false restarts the runtime.
_LOOP_WEDGE_EXIT_CODE = 71
_ACTIVE_WATCHDOG: "StallWatchdog | None" = None


def mark_runtime_service_progress(source: str = "runtime") -> None:
    """Record proof that the live runtime service is actively responding.

    The event-loop heartbeat remains the primary wedge detector. This service
    proof exists for real desktop launches where the watched loop can be stale
    during boot/runtime handoff while the HTTP/UI lane is demonstrably alive.
    """
    dog = _ACTIVE_WATCHDOG
    if dog is not None:
        dog.mark_runtime_service_progress(source)


class StallWatchdog(threading.Thread):
    """Monitor thread that tracks event loop responsiveness."""

    def __init__(self, loop: asyncio.AbstractEventLoop, threshold: float = 5.0):
        super().__init__(daemon=True, name="AuraStallWatchdog")
        self.loop = loop
        self.threshold = threshold
        self._last_heartbeat = time.time()
        self._running = False
        self._stop_event = threading.Event()
        self._task_birth: dict[int, float] = {}
        self._consecutive_long_stalls: int = 0
        self._started_at: float = time.time()
        # Source of truth for "the loop actually ran a callback", updated ONLY
        # by _heartbeat (never by stall suppression). The suppression paths
        # reset _last_heartbeat to silence expected warmup/foreground stalls,
        # which would otherwise mask a genuinely wedged loop forever — so the
        # absolute hard-exit ceiling keys off this independent timestamp.
        self._last_loop_run: float = time.time()
        self._diagnostic_only_notice_logged: bool = False
        self._last_boot_suppression_log_at: float = 0.0
        self._last_foreground_suppression_log_at: float = 0.0
        self._last_service_suppression_log_at: float = 0.0
        self._last_runtime_service_progress: float = time.time()
        self._last_runtime_service_source: str = "init"
        # Out-of-process liveness beacon. This daemon thread writes _last_loop_run
        # to a heartbeat file each tick; the external liveness_sentinel watches
        # it and kills+restarts the tree when it goes stale. This covers the case
        # the in-process hard-exit below CANNOT: a Metal GPU deadlock that holds
        # the GIL so no Python thread (this one included) can run — the file then
        # simply stops updating and the out-of-process sentinel acts.
        self._heartbeat_file: Path | None = self._resolve_heartbeat_file()
        self._last_heartbeat_file_write: float = 0.0
        # The event loop's OS thread id, learned from the heartbeat callback
        # (which runs ON the loop). Stall dumps stamp this thread so the
        # triage parser attributes the stall to the loop's actual frame
        # instead of guessing — a guess once blamed a sleeping daemon thread
        # for 19 stalls whose real culprit was on-loop SQLite.
        self._loop_thread_id: int | None = None

        # Dump-rate state: composing a dump is GIL-expensive, so during a
        # sustained wedge we keep the first and suppress its near-duplicates.
        self._last_stall_dump_at: float = 0.0
        self._last_stall_dump_path: str = ""
        self._suppressed_stall_dumps: int = 0

        # Boot-grace wedge detection. Boot grace is a time budget, so a boot
        # that is progressing slowly and a boot parked on one frame look
        # identical to it. They are not: a progressing boot moves. These track
        # the loop's innermost frame so a motionless one can be told apart.
        self._boot_frame_signature: str = ""
        self._boot_frame_signature_since: float = time.time()

    @staticmethod
    def _resolve_heartbeat_file() -> Path | None:
        raw = _FLAG_LIVENESS_HEARTBEAT_FILE.value()
        if not str(raw).strip():
            return None
        try:
            path = Path(raw)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        except OSError:
            return None

    def _write_liveness_heartbeat(self, *, loop_state: str = "alive", force: bool = False) -> None:
        """Append-free atomic write of the loop-liveness beacon (best-effort)."""
        if self._heartbeat_file is None:
            return
        now = time.time()
        # Throttle to ~1s; the run loop already ticks at 1s but guard anyway.
        if not force and now - self._last_heartbeat_file_write < 1.0:
            return
        self._last_heartbeat_file_write = now
        try:
            payload = (
                '{"pid": %d, "last_loop_run": %.3f, "last_runtime_service_progress": %.3f, '
                '"runtime_service_source": "%s", "written_at": %.3f, "loop_state": "%s"}'
                % (
                    os.getpid(),
                    self._last_loop_run,
                    self._last_runtime_service_progress,
                    self._json_escape(self._last_runtime_service_source),
                    now,
                    loop_state,
                )
            )
            atomic_write_text(self._heartbeat_file, payload, encoding="utf-8")
        except (OSError, RuntimeError) as exc:
            self._log_suppression(
                "_last_heartbeat_write_error_log_at",
                "StallWatchdog liveness beacon write failed: %s",
                exc,
            )

    @staticmethod
    def _suppression_log_interval_s() -> float:
        try:
            return max(5.0, float(_FLAG_WATCHDOG_SUPPRESSION_LOG_INTERVAL_S.value() or 60))
        except (TypeError, ValueError):
            return 60.0

    def _log_suppression(self, attr_name: str, message: str, *args: object) -> None:
        now = time.time()
        last = float(getattr(self, attr_name, 0.0) or 0.0)
        if now - last >= self._suppression_log_interval_s():
            logger.info(message, *args)
            setattr(self, attr_name, now)
        else:
            logger.debug(message, *args)

    @staticmethod
    def _json_escape(value: str) -> str:
        return str(value).replace("\\", "\\\\").replace('"', '\\"')[:120]

    def mark_runtime_service_progress(self, source: str = "runtime") -> None:
        self._last_runtime_service_progress = time.time()
        self._last_runtime_service_source = str(source or "runtime")[:120]

    def run(self):
        logger.info("🛡️ StallWatchdog: Monitoring loop (Threshold: %.1fs)", self.threshold)
        self._running = True
        now = time.time()
        self._started_at = now
        self._last_heartbeat = now
        self._last_loop_run = now
        self._write_liveness_heartbeat(loop_state="starting", force=True)
        # Drain the historical dump backlog at startup (budgeted): a healthy
        # instance that never stalls again must still shed old dumps, and the
        # per-stall batch alone would take ~90 boots to clear a storm backlog.
        self._drain_stall_dump_backlog(_forensics_root() / "stalls")

        while not self._stop_event.is_set():
            # Schedule a heartbeat on the loop
            try:
                if self.loop.is_closed():
                    logger.debug("StallWatchdog: event loop closed, exiting.")
                    break
                is_running = getattr(self.loop, "is_running", None)
                if callable(is_running) and not is_running():
                    logger.info(
                        "StallWatchdog: watched loop is no longer running; retiring monitor "
                        "instead of treating loop handoff as a wedge."
                    )
                    self._write_liveness_heartbeat(loop_state="retired", force=True)
                    break
                self.loop.call_soon_threadsafe(self._heartbeat)
            except RuntimeError:
                # Event loop closed during shutdown — exit silently
                break
            except (AttributeError, TypeError, ValueError) as e:
                _record_watchdog_degradation(
                    e,
                    action="skipped watchdog heartbeat schedule and kept monitoring thread alive",
                    severity="warning",
                    extra={"stage": "heartbeat_schedule"},
                )
                logger.debug("Watchdog heartbeat schedule issue: %s", e)

            time.sleep(1.0)  # Check every second

            # Refresh the out-of-process liveness beacon every tick. When the
            # loop wedges, _last_loop_run stops advancing; when the GIL is held
            # by a Metal deadlock, this write stops entirely — either way the
            # external liveness_sentinel sees staleness and restarts the tree.
            self._write_liveness_heartbeat()

            # Absolute loop-liveness ceiling — checked FIRST and immune to all
            # stall suppression. _last_loop_run is advanced only when the loop
            # actually executes the heartbeat callback, so this measures true
            # loop death (e.g. a Metal GPU deadlock) that suppression would
            # otherwise hide. If the loop is wedged beyond any in-process
            # recovery, hard-exit so the supervisor restarts the runtime.
            loop_silence = time.time() - self._last_loop_run
            if self._should_force_exit(loop_silence):
                self._force_exit_for_restart(loop_silence)

            # Check for stall
            elapsed = time.time() - self._last_heartbeat
            if elapsed > self.threshold:
                if self._should_suppress_stall(elapsed):
                    self._last_heartbeat = time.time()
                    self._consecutive_long_stalls = 0
                    continue
                self._report_stall(elapsed)
                if elapsed >= _ACTIVE_RECOVERY_THRESHOLD:
                    self._consecutive_long_stalls += 1
                    self._attempt_active_recovery(elapsed)
                else:
                    self._consecutive_long_stalls = 0
                # Reset so the next stall measurement is fresh.
                self._last_heartbeat = time.time()
            else:
                self._consecutive_long_stalls = 0

    def stop(self):
        self._stop_event.set()

    def _heartbeat(self):
        now = time.time()
        self._last_heartbeat = now
        # Independent liveness proof for the hard-exit ceiling: this only runs
        # when the loop is genuinely alive, and nothing else writes it.
        self._last_loop_run = now
        self._loop_thread_id = threading.get_ident()
        # Track task ages so a future stall can pick out which ones look hung.
        # This runs on the loop thread — cheap and safe.
        try:
            now = time.time()
            tasks = asyncio.all_tasks(self.loop)
            seen = set()
            for task in tasks:
                tid = id(task)
                seen.add(tid)
                if tid not in self._task_birth:
                    self._task_birth[tid] = now
            # Drop dead bookkeeping
            for tid in list(self._task_birth.keys()):
                if tid not in seen:
                    self._task_birth.pop(tid, None)
        except _STALL_WATCHDOG_ERRORS as exc:
            _record_watchdog_degradation(
                exc,
                action="kept watchdog alive after task-age bookkeeping failed",
                severity="warning",
                extra={"stage": "task_age_bookkeeping"},
            )
            logger.debug("Task age bookkeeping failed: %s", exc)

    @staticmethod
    def _hard_exit_ceiling_s() -> float:
        try:
            return float(
                os.getenv("AURA_WATCHDOG_HARD_EXIT_S", str(_LOOP_WEDGE_HARD_EXIT_S))
                or _LOOP_WEDGE_HARD_EXIT_S
            )
        except (TypeError, ValueError):
            return _LOOP_WEDGE_HARD_EXIT_S

    @staticmethod
    def _hard_exit_code() -> int:
        try:
            return int(
                os.getenv("AURA_WATCHDOG_HARD_EXIT_CODE", str(_LOOP_WEDGE_EXIT_CODE))
                or _LOOP_WEDGE_EXIT_CODE
            )
        except (TypeError, ValueError):
            return _LOOP_WEDGE_EXIT_CODE

    @staticmethod
    def _hard_exit_boot_grace_s() -> float:
        """Boot grace for the out-of-band hard-exit ceiling.

        The hard-exit ceiling protects against true loop wedges, but a fresh
        desktop boot can spend several minutes in synchronous import/model/UI
        startup before the watched loop has a stable heartbeat. A stale first
        sample must not kill the process and create the white-screen boot loop.
        """
        explicit = _FLAG_WATCHDOG_HARD_EXIT_BOOT_GRACE_S.value()
        if explicit is not None:
            try:
                return max(0.0, float(explicit or 0.0))
            except (TypeError, ValueError):
                return 0.0
        if "AURA_WATCHDOG_BOOT_GRACE_S" in os.environ:
            try:
                inherited = max(0.0, float(os.getenv("AURA_WATCHDOG_BOOT_GRACE_S") or 0.0))
            except (TypeError, ValueError):
                inherited = 0.0
            if _FLAG_SAFE_BOOT_DESKTOP.value() == "1" or _FLAG_EXTERNAL_GUI_OWNER.value() == "1":
                return max(inherited, 1200.0)
            return inherited
        return 1200.0

    @staticmethod
    def _runtime_service_progress_grace_s() -> float:
        try:
            return max(
                0.0,
                float(_FLAG_WATCHDOG_SERVICE_PROOF_GRACE_S.value() or 240),
            )
        except (TypeError, ValueError):
            return 240.0

    def _loop_frame_signature(self) -> str:
        """Cheap identity of the loop's innermost frame — file:line:function.

        Deliberately not a formatted traceback: this runs every watchdog tick,
        and composing full stacks is the GIL burst that made stall dumps cause
        the next stall.
        """

        thread_id = self._loop_thread_id
        if thread_id is None:
            return ""
        try:
            frame = sys._current_frames().get(thread_id)
            if frame is None:
                return ""
            code = frame.f_code
            return f"{os.path.basename(code.co_filename)}:{frame.f_lineno}:{code.co_name}"
        except (AttributeError, KeyError, RuntimeError, ValueError):
            return ""

    def _boot_frame_stuck_for(self) -> tuple[float, str]:
        """How long the loop's innermost frame has been unchanged, and which.

        Returns ``(0.0, frame)`` whenever the frame moves — progress resets the
        clock, so only a genuinely motionless loop accumulates time.
        """

        signature = self._loop_frame_signature()
        now = time.time()
        if not signature:
            # No readable frame is not evidence of a wedge; the loop thread id
            # is only learned once the heartbeat has run on the loop.
            self._boot_frame_signature = ""
            self._boot_frame_signature_since = now
            return 0.0, ""
        if signature != self._boot_frame_signature:
            self._boot_frame_signature = signature
            self._boot_frame_signature_since = now
            return 0.0, signature
        return now - self._boot_frame_signature_since, signature

    def _should_force_exit(self, loop_silence: float) -> bool:
        """True when the loop is wedged beyond any in-process recovery.

        Immune to stall suppression by design: this is the last-resort ceiling
        for a genuinely dead loop (e.g. a Metal GPU deadlock). The only thing
        we honor is an intentional shutdown, when the loop stops on purpose.
        """
        ceiling = self._hard_exit_ceiling_s()
        if ceiling <= 0 or loop_silence < ceiling:
            return False
        boot_grace = self._hard_exit_boot_grace_s()
        if boot_grace > 0 and (time.time() - self._started_at) < boot_grace:
            motionless_for, frame = self._boot_frame_stuck_for()
            if motionless_for >= ceiling:
                # BOUNDED, like the foreground-lane suppression above. On
                # 2026-08-03 the boot loop parked in the skill catalog behind
                # an ABBA deadlock; the watchdog dumped 33 times in three
                # minutes and suppressed every escalation, because 1200s of
                # boot grace had not elapsed. The reaper killed the kernel
                # first and the launcher restarted into the same race. A boot
                # whose loop has not moved off one frame for longer than the
                # wedge ceiling is not booting slowly — it is wedged, and boot
                # grace must not protect it.
                logger.critical(
                    "🛑 [WATCHDOG] Boot loop wedged: silent %.0fs and parked on %s for "
                    "%.0fs (≥ %.0fs ceiling) — boot grace no longer applies.",
                    loop_silence,
                    frame or "an unreadable frame",
                    motionless_for,
                    ceiling,
                )
                return True
            self._log_suppression(
                "_last_hard_exit_boot_suppression_log_at",
                "StallWatchdog: suppressing %.1fs hard-exit loop silence during boot grace "
                "(grace=%.1fs, loop frame moving, at=%s).",
                loop_silence,
                boot_grace,
                frame or "unknown",
            )
            return False
        service_grace = self._runtime_service_progress_grace_s()
        service_age = time.time() - self._last_runtime_service_progress
        if service_grace > 0 and service_age < service_grace:
            self._log_suppression(
                "_last_service_suppression_log_at",
                "StallWatchdog: suppressing %.1fs hard-exit loop silence because "
                "runtime service progress is fresh (age=%.1fs source=%s grace=%.1fs).",
                loop_silence,
                service_age,
                self._last_runtime_service_source,
                service_grace,
            )
            return False
        try:
            from core.runtime.shutdown_coordinator import is_shutdown_requested

            if is_shutdown_requested():
                return False
        except (ImportError, RuntimeError):
            pass
        return True

    def _force_exit_for_restart(self, loop_silence: float) -> None:
        """Hard-exit a wedged process so the supervisor restarts it.

        Runs on the out-of-band watchdog thread — the only code still alive
        when the loop is deadlocked. Dumps thread stacks via low-level
        faulthandler to a raw fd (never through app locks/gateways the wedge
        may be holding), then exits immediately with a non-zero code. No
        logging.shutdown()/handler flush: those can block under a wedge (the
        lesson from the 115GB crash where the lethal path never reached exit).
        """
        logger.critical(
            "🛑 [WATCHDOG] Event loop unresponsive %.0fs ≥ hard ceiling — wedged beyond "
            "in-process recovery; forcing process exit %d for supervisor restart.",
            loop_silence,
            self._hard_exit_code(),
        )
        try:
            import faulthandler

            crash_dir = _forensics_root() / "crash"
            crash_dir.mkdir(parents=True, exist_ok=True)
            with open(crash_dir / "loop_wedge_stacks.log", "a") as fh:
                fh.write(
                    f"\n===== LOOP WEDGE pid={os.getpid()} "
                    f"silence={loop_silence:.1f}s at={time.time()} =====\n"
                )
                fh.flush()
                faulthandler.dump_traceback(file=fh, all_threads=True)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.debug("Loop-wedge stack dump failed: %s", exc)
        os._exit(self._hard_exit_code())

    def _should_suppress_stall(self, elapsed: float) -> bool:
        """Suppress expected launch/shutdown stalls without hiding live hangs."""
        try:
            from core.runtime.shutdown_coordinator import is_shutdown_requested

            if is_shutdown_requested():
                logger.debug("StallWatchdog: suppressing %.1fs stall during shutdown.", elapsed)
                return True
        except (ImportError, RuntimeError) as exc:
            logger.debug("StallWatchdog shutdown probe skipped: %s", exc)

        try:
            boot_grace = float(os.getenv("AURA_WATCHDOG_BOOT_GRACE_S", "120") or 120)
        except (TypeError, ValueError):
            boot_grace = 120.0
        if boot_grace > 0 and (time.time() - self._started_at) < boot_grace:
            self._log_suppression(
                "_last_boot_suppression_log_at",
                "StallWatchdog: suppressing %.1fs launch stall during boot grace.",
                elapsed,
            )
            return True
        try:
            foreground_grace = float(_FLAG_WATCHDOG_FOREGROUND_GRACE_S.value() or 75)
        except (TypeError, ValueError):
            foreground_grace = 75.0
        if foreground_grace > 0 and elapsed <= foreground_grace:
            try:
                from core.container import ServiceContainer

                gate = ServiceContainer.get("inference_gate", default=None)
                lane = gate.get_conversation_status() if gate and hasattr(gate, "get_conversation_status") else {}
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                logger.debug("StallWatchdog foreground lane probe unavailable: %s", exc)
                lane = {}
            lane_state = str(lane.get("state") or "").lower()
            foreground_active = bool(
                lane.get("foreground_owned")
                or int(lane.get("active_generations", 0) or 0) > 0
                or lane.get("warmup_in_flight")
                or lane_state in {"spawning", "handshaking", "warming", "recovering"}
            )
            if foreground_active:
                # BOUNDED suppression. During the Jul 9 48%-wedge the lane
                # sat perpetually 'warming', so every 5-10s loop stall was
                # suppressed and the forensic record went silent for the
                # exact window that mattered (hours-old last dump while the
                # health contract failed on 8s lags). A lane continuously
                # busy past the warmup deadline is a wedge, not a warmup —
                # stalls dump again.
                now = time.time()
                since = float(getattr(self, "_foreground_suppression_started_at", 0.0) or 0.0)
                if since <= 0.0:
                    since = now
                    self._foreground_suppression_started_at = now
                if (now - since) > 300.0:
                    logger.warning(
                        "StallWatchdog: foreground lane continuously busy %.0fs — "
                        "suppression deadline passed, resuming stall dumps.",
                        now - since,
                    )
                    return False
                self._log_suppression(
                    "_last_foreground_suppression_log_at",
                    "StallWatchdog: suppressing %.1fs foreground inference stall "
                    "(state=%s active=%s warmup=%s owner=%s).",
                    elapsed,
                    lane.get("state"),
                    lane.get("active_generations", 0),
                    lane.get("warmup_in_flight"),
                    lane.get("foreground_owner", ""),
                )
                return True
            # Lane went quiet — reset the continuous-busy anchor.
            if getattr(self, "_foreground_suppression_started_at", 0.0):
                self._foreground_suppression_started_at = 0.0
        return False

    _STALL_DUMP_KEEP = 500
    _STALL_DUMP_PRUNE_BATCH = 200

    #: Minimum gap between full all-thread dumps.
    #:
    #: Composing one walks sys._current_frames() and formats a stack for every
    #: thread — ~80 of them live, 85KB of output — as a pure-Python burst that
    #: holds the GIL. Under an ongoing stall that starves the loop further, so
    #: the dump taken to explain the freeze helps cause the next one. On
    #: 2026-07-29 three fired inside three minutes as the stalls escalated
    #: 5.6s → 17.6s → 103.8s, and back-to-back dumps of the same wedge are
    #: near-identical anyway: the second one buys pressure, not evidence.
    #: This is the same defect record_degradation had (fixed 2026-07-29,
    #: 00c65528) — the act of recording a freeze cost enough to cause the next.
    _STALL_DUMP_MIN_INTERVAL_S = 60.0

    def _drain_stall_dump_backlog(self, dump_dir: Path, *, budget_s: float = 3.0) -> None:
        """Fully drain the stall-dump backlog to the retention target, bounded
        by a wall-clock budget. Runs on the watchdog thread at startup."""
        try:
            if not dump_dir.exists():
                return
            deadline = time.monotonic() + max(0.1, budget_s)
            while time.monotonic() < deadline:
                names = sorted(
                    entry.name
                    for entry in os.scandir(dump_dir)
                    if entry.name.startswith("stall_") and entry.name.endswith(".txt")
                )
                excess = len(names) - self._STALL_DUMP_KEEP
                if excess <= 0:
                    return
                for name in names[: min(excess, 2000)]:
                    try:
                        (dump_dir / name).unlink()
                    except OSError:
                        continue
        except OSError as exc:
            logger.debug("Startup stall-dump drain skipped: %s", exc)

    def _prune_stall_dumps(self, dump_dir: Path) -> None:
        """Bounded retention: keep the newest dumps, drain backlog gradually.

        Live incidents have accumulated tens of thousands of stall reports;
        deleting at most one batch per stall keeps this call cheap while the
        backlog shrinks toward the retention target.
        """
        try:
            names = sorted(
                entry.name
                for entry in os.scandir(dump_dir)
                if entry.name.startswith("stall_") and entry.name.endswith(".txt")
            )
            excess = len(names) - self._STALL_DUMP_KEEP
            if excess <= 0:
                return
            for name in names[: min(excess, self._STALL_DUMP_PRUNE_BATCH)]:
                try:
                    (dump_dir / name).unlink()
                except OSError:
                    continue
        except OSError as exc:
            logger.debug("Stall dump pruning skipped: %s", exc)

    def _compose_dump_text(self, elapsed: float) -> str:
        """All-thread traceback snapshot with the event-loop thread STAMPED
        (header line + section marker), so the attribution parser
        (core/observability/stall_dump.py) names the loop's actual frame
        instead of guessing among 70 innocent parked threads."""
        buffer = io.StringIO()
        try:
            buffer.write(f"STALL DETECTED: {elapsed:.1f}s\n")
            loop_thread_id = self._loop_thread_id
            if loop_thread_id is not None:
                buffer.write(f"LOOP THREAD: {loop_thread_id}\n")
            buffer.write("=" * 40 + "\n")
            for thread_id, frame in sys._current_frames().items():
                marker = "  [EVENT LOOP]" if thread_id == loop_thread_id else ""
                buffer.write(f"\nThread ID: {thread_id}{marker}\n")
                traceback.print_stack(frame, file=buffer)
            return buffer.getvalue()
        finally:
            buffer.close()

    def _report_stall(self, elapsed: float):
        logger.error("🚨 [WATCHDOG] EVENT LOOP STALL DETECTED! (Elapsed: %.1fs)", elapsed)

        # A dump costs real GIL time; during an ongoing stall that is time
        # taken from the loop we are trying to rescue. Keep the first dump of
        # a wedge — that one carries the evidence — and skip its near-identical
        # successors, saying so rather than dropping them silently.
        now = time.monotonic()
        since_last = now - self._last_stall_dump_at
        if self._last_stall_dump_at > 0.0 and since_last < self._STALL_DUMP_MIN_INTERVAL_S:
            self._suppressed_stall_dumps += 1
            logger.warning(
                "[WATCHDOG] Stall dump suppressed (%.0fs since the last of this "
                "wedge, %d suppressed): composing one costs the GIL the stalled "
                "loop needs. Evidence is in %s",
                since_last,
                self._suppressed_stall_dumps,
                self._last_stall_dump_path or "the previous dump",
            )
            self._notify_diagnostics(elapsed)
            return

        # Dump tracebacks of all threads
        dump_dir = _forensics_root() / "stalls"
        dump_dir.mkdir(parents=True, exist_ok=True)
        self._prune_stall_dumps(dump_dir)
        dump_file = dump_dir / f"stall_{int(time.time())}.txt"
        self._last_stall_dump_at = now
        self._last_stall_dump_path = str(dump_file)

        dump_text = self._compose_dump_text(elapsed)
        try:
            with local_internal_governed_scope(
                "resilience.stall_watchdog.traceback_dump",
                domain="file_write",
            ):
                get_file_write_gateway().write_text(
                    dump_file,
                    dump_text,
                    source="resilience.stall_watchdog.traceback_dump",
                )
        except _STALL_WATCHDOG_ERRORS as exc:
            _record_watchdog_degradation(
                exc,
                action="continued stall handling after traceback dump write failed",
                severity="warning",
                extra={"stage": "traceback_dump", "elapsed_s": elapsed},
            )

        logger.info("💉 [IMMUNE] Stall traceback dumped to: %s", dump_file)

        self._notify_diagnostics(elapsed)

    def _notify_diagnostics(self, elapsed: float) -> None:
        """Proactively trigger Neuro-Surgeon analysis.

        Runs for a suppressed dump too: the stall is just as real, only its
        traceback is redundant.
        """
        try:
            from core.resilience.diagnostic_hub import get_diagnostic_hub
            get_diagnostic_hub()
            # Future: trigger auto-repair or circuit break
        except (ImportError, AttributeError, RuntimeError) as _e:
            record_degradation(
                "stall_watchdog",
                _e,
                severity="debug",
                action="continued after optional stall diagnostic hub was unavailable",
            )
            logger.debug('Ignored Exception in stall_watchdog.py: %s', _e)

    def _attempt_active_recovery(self, elapsed: float) -> None:
        """Don't just log a stall — try to break it.

        We schedule a recovery coroutine onto the (possibly wedged) loop.
        If the loop is truly frozen, the coroutine will queue and run when
        the loop wakes up — and it will then prevent the next stall by
        cancelling the hung tasks. If the loop is partially responsive, the
        coroutine runs immediately.
        """
        def _schedule_recovery() -> None:
            try:
                create_tracked_task(
                    self._recover_on_loop(elapsed),
                    name="stall_watchdog.active_recovery",
                )
            except _STALL_WATCHDOG_ERRORS as exc:
                _record_watchdog_degradation(
                    exc,
                    action="continued watchdog reporting after active recovery task ownership failed",
                    severity="warning",
                    extra={"stage": "active_recovery_task", "elapsed_s": elapsed},
                )

        try:
            self.loop.call_soon_threadsafe(_schedule_recovery)
        except RuntimeError:
            return
        except (AttributeError, TypeError, ValueError) as exc:
            _record_watchdog_degradation(
                exc,
                action="continued watchdog reporting after active recovery scheduling failed",
                severity="warning",
                extra={"stage": "active_recovery_schedule", "elapsed_s": elapsed},
            )
            logger.debug("Stall recovery scheduling failed: %s", exc)

    async def _recover_on_loop(self, elapsed: float) -> None:
        """Cancel hung tasks and ask known subsystems to recycle.

        Runs on the asyncio loop. Conservative — we only cancel tasks that
        have been alive for far longer than this stall, and we never touch
        the kernel main loop / watchdog / orchestrator coordinator tasks.
        """
        protected_substrings = (
            "AuraKernel",
            "OrchestratorMainLoop",
            "AuraStallWatchdog",
            "ConsciousnessLoopMonitor",
            "Server.Chat",
            "uvicorn",
            "server.ws",
        )
        cutoff = time.time() - max(_TASK_HUNG_SECONDS, elapsed * 1.5)
        cancelled = 0
        try:
            tasks = asyncio.all_tasks(self.loop)
        except RuntimeError:
            return

        aggressive_cancel = _FLAG_WATCHDOG_CANCEL_HUNG_TASKS.value().strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if aggressive_cancel:
            for task in tasks:
                if task.done():
                    continue
                name = getattr(task, "get_name", lambda: "")() or repr(task)
                if any(p in name for p in protected_substrings):
                    continue
                birth = self._task_birth.get(id(task))
                if birth is None or birth >= cutoff:
                    continue
                try:
                    task.cancel()
                    cancelled += 1
                except RuntimeError as exc:
                    _record_watchdog_degradation(
                        exc,
                        action="continued stall recovery after hung-task cancellation failed",
                        severity="warning",
                        extra={"stage": "cancel_hung_task", "task_name": name},
                    )
                    logger.debug("Stall recovery: failed to cancel %s: %s", name, exc)
        else:
            message = (
                "💉 [IMMUNE] Stall recovery is diagnostic-only; not bulk-cancelling asyncio tasks. "
                "Set AURA_WATCHDOG_CANCEL_HUNG_TASKS=1 to enable aggressive cancellation."
            )
            if self._diagnostic_only_notice_logged:
                logger.debug(message)
            else:
                logger.info(message)
                self._diagnostic_only_notice_logged = True

        if cancelled:
            logger.warning(
                "💉 [IMMUNE] Stall recovery cancelled %d hung asyncio tasks (stall=%.0fs).",
                cancelled,
                elapsed,
            )

        # Ask the brainstem and cortex MLX clients to self-check; the
        # stale-handshake path in mlx_client._ensure_worker_alive will
        # recycle anyone that's been wedged.
        try:
            mlx_client_module = import_module("core.brain.llm.mlx_client")
            live_mlx_clients = getattr(mlx_client_module, "_LIVE_MLX_CLIENTS", None)
        except (ImportError, AttributeError, RuntimeError):
            live_mlx_clients = None

        if live_mlx_clients:
            for client in list(live_mlx_clients):
                try:
                    if hasattr(client, "_lane_state") and client._lane_state == "handshaking":
                        # Schedule a no-op alive probe so the stale-handshake
                        # branch fires on next entry.
                        self.loop.call_soon(client._mark_progress)
                except (OSError, ConnectionError, TimeoutError) as exc:
                    _record_watchdog_degradation(
                        exc,
                        action="continued stall recovery after MLX liveness poke failed",
                        severity="warning",
                        extra={"stage": "mlx_liveness_poke"},
                    )
                    logger.debug("Stall recovery MLX poke failed: %s", exc)

        # If we've taken many long stalls in a row, ask the orchestrator's
        # state vault to flush so we don't lose continuity.
        if self._consecutive_long_stalls >= 3:
            try:
                from core.container import ServiceContainer
                state_repo = ServiceContainer.get("state_repository", default=None)
                if state_repo and hasattr(state_repo, "request_flush"):
                    state_repo.request_flush()
                    logger.info("💉 [IMMUNE] Requested state vault flush after %d consecutive stalls.", self._consecutive_long_stalls)
            except (ImportError, AttributeError, RuntimeError) as exc:
                _record_watchdog_degradation(
                    exc,
                    action="continued stall recovery after state flush request failed",
                    severity="warning",
                    extra={"stage": "state_flush_request"},
                )
                logger.debug("Stall recovery state-flush request failed: %s", exc)

def start_watchdog(loop: asyncio.AbstractEventLoop | None = None, threshold: float = 5.0):
    """Convenience helper to start the watchdog."""
    global _ACTIVE_WATCHDOG
    try:
        target_loop = loop or asyncio.get_running_loop()
    except RuntimeError:
        target_loop = asyncio.new_event_loop()
    dog = StallWatchdog(target_loop, threshold=threshold)
    _ACTIVE_WATCHDOG = dog
    try:
        from core.container import ServiceContainer

        ServiceContainer.register_instance("stall_watchdog", dog, required=False)
    except (ImportError, RuntimeError, TypeError, ValueError, AttributeError):
        pass
    dog.start()
    return dog
