"""core/conversation/terminal_chat.py
TerminalFallbackChat — Emergency Last-Resort Communication Channel.

Aura can AUTONOMOUSLY open terminal chat when she has something to say and
the main UI is confirmed gone. This is strictly a last resort.

Hard rules (never relaxed):
  1. Will NOT open if the main UI/websocket server is detectable.
  2. Will NOT open if not running in a real tty.
  3. Will NOT spam: enforces minimum interval between output lines.
  4. Will NOT randomly switch: once open, stays until main UI returns.
  5. Pending message queue has a hard cap — old messages are discarded,
     not accumulated indefinitely.
  6. The autonomous monitor loop (TerminalWatchdog) only fires once the
     UI has been confirmed gone for UI_GONE_CONFIRMATION_SECS seconds
     (prevents false-positive flapping on a brief WebSocket drop).

Autonomous activation triggers (checked by TerminalWatchdog):
  - Proactive presence queued a message that couldn't be delivered
  - Dream journal or sleep cycle produced insight Aura wants to share
  - Emergency mode (critical battery / memory / thermal)
  - Any subsystem calls terminal_fallback.queue_autonomous_message(text)

Deactivation:
  - Main UI comes back (detected by watchdog)
  - User types 'exit' / 'quit'
  - No activity for IDLE_TIMEOUT_SECS and no pending messages
"""

import asyncio
import codecs
import collections
import logging
import os
import subprocess
import sys
import time
from typing import Any

from core.runtime.errors import FallbackClassification, record_degradation
from core.runtime.lockdep import LockRank, checked_async_lock
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.TerminalFallback")

# How long the UI must be gone before autonomous activation is allowed
UI_GONE_CONFIRMATION_SECS: float = 30.0
# How often the watchdog checks for UI state + pending messages
WATCHDOG_INTERVAL_SECS: float = 10.0
# Max unread messages queued before oldest are discarded
MAX_PENDING_MESSAGES: int = 5
# Minimum seconds between any terminal output (spam guard)
MIN_OUTPUT_INTERVAL: float = 2.0
# If no input and no pending messages for this long, close terminal session
IDLE_TIMEOUT_SECS: float = 120.0
SHELL_COMMAND_TIMEOUT_SECS: float = 30.0
SHELL_KILL_GRACE_SECS: float = 2.0
SHELL_COMMENT_TIMEOUT_SECS: float = 15.0
SHELL_OUTPUT_LINE_CAP: int = 200
SHELL_OUTPUT_CHAR_CAP: int = 24_000

INPUT_PREFIX = "[Aura] You: "
OUTPUT_PREFIX = "[Aura] "
BANNER = (
    "\n╔══════════════════════════════════════════════════╗\n"
    "║  AURA — TERMINAL MODE  (main window unavailable) ║\n"
    "║  Chat normally, or prefix with ! to run a        ║\n"
    "║  shell command (e.g. !ls -la).                   ║\n"
    "║  Type 'exit' to end. Auto-closes when app opens. ║\n"
    "╚══════════════════════════════════════════════════╝\n"
)


def _record_terminal_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
):
    return record_degradation(
        "terminal_chat",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


class TerminalFallbackChat:
    """Emergency last-resort terminal chat.

    Two activation paths:
      • explicit:    await activate(orchestrator, force=True)
      • autonomous:  TerminalWatchdog detects UI gone + pending messages
    """

    def __init__(self):
        self._active: bool = False
        self._chat_task: asyncio.Task | None = None
        self._last_output_at: float = 0.0
        self._last_activity_at: float = time.time()
        self._orch = None
        self._output_lock = checked_async_lock(
            f"terminal_chat.output.{id(self)}", rank=LockRank.LEAF
        )
        self._stdin_loop: asyncio.AbstractEventLoop | None = None
        self._stdin_fd: int | None = None
        self._stdin_reader_installed = False
        self._stdin_lines: asyncio.Queue[str | None] = asyncio.Queue()
        self._stdin_buffer = ""
        self._stdin_decoder: codecs.IncrementalDecoder | None = None

        # Pending messages Aura wants to deliver autonomously
        self._pending: collections.deque[str] = collections.deque(maxlen=MAX_PENDING_MESSAGES)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self._active

    def queue_autonomous_message(self, text: str) -> bool:
        """Any subsystem calls this when it has something to say.

        The TerminalWatchdog will deliver it if/when terminal mode opens.
        If terminal is already active, it's written immediately.
        """
        if not text or not text.strip():
            return False
        constitutional_runtime_live = False
        try:
            from core.constitution import get_constitutional_core
            from core.container import ServiceContainer

            constitutional_runtime_live = (
                ServiceContainer.has("executive_core")
                or ServiceContainer.has("aura_kernel")
                or ServiceContainer.has("kernel_interface")
                or bool(getattr(ServiceContainer, "_registration_locked", False))
            )
            approved, reason = get_constitutional_core().approve_expression_sync(
                text.strip(),
                source="background",
                urgency=0.55,
            )
            if not approved:
                if constitutional_runtime_live and any(
                    marker in str(reason or "")
                    for marker in ("gate_failed", "required", "unavailable")
                ):
                    try:
                        from core.health.degraded_events import record_degraded_event

                        record_degraded_event(
                            "terminal_fallback",
                            "executive_gate_unavailable",
                            detail=text[:120],
                            severity="warning",
                            classification="background_degraded",
                            context={"reason": reason},
                        )
                    except (ImportError, AttributeError, RuntimeError) as exc:
                        _record_terminal_degradation(
                            exc,
                            action=(
                                "suppressed autonomous terminal message after degraded-event receipt "
                                "emission failed"
                            ),
                            extra={
                                "message_preview": text[:120],
                                "reason": str(reason)[:160],
                            },
                        )
                        logger.debug("Suppressed Exception: %s", exc)
                    logger.debug(
                        "TerminalFallback: constitutional gate unavailable, suppressing autonomous message: %s",
                        reason,
                    )
                    return False
                logger.debug(
                    "TerminalFallback: constitutional gate suppressed queued autonomous message: %s",
                    reason,
                )
                return False
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_terminal_degradation(
                exc,
                action="evaluated autonomous terminal message with executive gate degraded",
                extra={"message_preview": text[:120]},
            )
            if constitutional_runtime_live:
                try:
                    from core.health.degraded_events import record_degraded_event

                    record_degraded_event(
                        "terminal_fallback",
                        "executive_gate_unavailable",
                        detail=text[:120],
                        severity="warning",
                        classification="background_degraded",
                        context={"error": type(exc).__name__},
                        exc=exc,
                    )
                except (ImportError, AttributeError, RuntimeError) as receipt_exc:
                    _record_terminal_degradation(
                        receipt_exc,
                        action=(
                            "suppressed autonomous terminal message after degraded-event receipt "
                            "emission failed"
                        ),
                        extra={"message_preview": text[:120]},
                    )
                    logger.debug("Suppressed Exception: %s", receipt_exc)
                logger.debug(
                    "TerminalFallback: executive gate unavailable, suppressing autonomous message: %s",
                    exc,
                )
                return False
            logger.debug(
                "TerminalFallback: executive gate unavailable, proceeding degraded: %s", exc
            )
        self._pending.append(text.strip())
        logger.debug("TerminalFallback: queued message (%d pending)", len(self._pending))

        if self._active:
            # Already in terminal mode — flush now
            get_task_tracker().track(self._flush_pending())
        return True

    async def activate(self, orchestrator=None, force: bool = False) -> bool:
        """Activate terminal fallback mode.

        Returns True if activation succeeded.
        force=True bypasses tty + resource checks (headless / no-GUI devices).
        """
        if self._active:
            return True

        if not self._can_activate(force=force):
            return False

        self._active = True
        self._orch = orchestrator
        self._last_activity_at = time.time()
        self._chat_task = get_task_tracker().create_task(
            self._chat_loop(), name="TerminalFallback.chat"
        )
        logger.warning("📟 TerminalFallback ACTIVE — communicating via terminal stdin/stdout.")
        return True

    async def deactivate(self, reason: str = ""):
        """Shut down terminal mode."""
        if not self._active:
            return
        self._active = False
        self._stop_stdin_reader()
        current_task = asyncio.current_task()
        owned_task = self._chat_task
        if (
            owned_task
            and owned_task is not current_task
            and not owned_task.done()
        ):
            owned_task.cancel()
            await asyncio.gather(owned_task, return_exceptions=True)
        self._chat_task = None
        self._orch = None
        if reason:
            logger.info("TerminalFallback deactivated: %s", reason)

    # ── Activation guard ──────────────────────────────────────────────────────

    def _can_activate(self, force: bool = False) -> bool:
        """Check all hard safety conditions for activation."""
        # Must have a real tty unless forced
        if not force and not sys.stdin.isatty():
            logger.debug("TerminalFallback: not a tty, skipping")
            return False
        # Must confirm UI is gone unless forced
        if not force and self._is_main_ui_open():
            logger.debug("TerminalFallback: main UI still open, skipping")
            return False
        return True

    # ── Chat loop ─────────────────────────────────────────────────────────────

    def _on_stdin_ready(self) -> None:
        """Drain available bytes without ever blocking on a partial line."""

        try:
            chunk = os.read(self._stdin_fd, 4_096)
        # not a failure: a non-blocking read with nothing ready is the ordinary case this
        # loop is written around.
        except BlockingIOError:
            return
        except (OSError, TypeError, ValueError):
            chunk = b""
        if not chunk:
            if self._stdin_decoder is not None:
                self._stdin_buffer += self._stdin_decoder.decode(b"", final=True)
            self._stdin_loop.remove_reader(self._stdin_fd)
            self._stdin_reader_installed = False
            if self._stdin_buffer:
                self._stdin_lines.put_nowait(self._stdin_buffer)
                self._stdin_buffer = ""
            self._stdin_lines.put_nowait(None)
            return

        if self._stdin_decoder is None:
            self._stdin_decoder = codecs.getincrementaldecoder("utf-8")(
                errors="replace"
            )
        self._stdin_buffer += self._stdin_decoder.decode(chunk)
        while "\n" in self._stdin_buffer:
            line, self._stdin_buffer = self._stdin_buffer.split("\n", 1)
            self._stdin_lines.put_nowait(f"{line}\n")

    def _start_stdin_reader(self) -> None:
        """Install exactly one event-loop reader for the active terminal session."""

        if self._stdin_reader_installed:
            return
        loop = asyncio.get_running_loop()
        try:
            fd = int(sys.stdin.fileno())
            loop.add_reader(fd, self._on_stdin_ready)
        except (AttributeError, NotImplementedError, OSError, TypeError, ValueError) as exc:
            raise RuntimeError("terminal stdin does not support async fd readiness") from exc
        self._stdin_loop = loop
        self._stdin_fd = fd
        encoding = str(getattr(sys.stdin, "encoding", "") or "utf-8")
        try:
            decoder_type = codecs.getincrementaldecoder(encoding)
        except LookupError:
            decoder_type = codecs.getincrementaldecoder("utf-8")
        self._stdin_decoder = decoder_type(errors="replace")
        self._stdin_buffer = ""
        self._stdin_reader_installed = True

    def _stop_stdin_reader(self) -> None:
        """Remove the owned reader synchronously; no blocked executor survives."""

        if self._stdin_reader_installed and self._stdin_loop is not None:
            try:
                self._stdin_loop.remove_reader(self._stdin_fd)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                _record_terminal_degradation(
                    exc,
                    action="removed terminal stdin readiness reader during deactivation",
                    severity="warning",
                )
        self._stdin_reader_installed = False
        self._stdin_loop = None
        self._stdin_fd = None
        self._stdin_decoder = None
        self._stdin_buffer = ""
        while not self._stdin_lines.empty():
            try:
                self._stdin_lines.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _read_stdin_line(self, *, wait_seconds: float = 20.0) -> str | None:
        """Wait for one line without spawning or abandoning reader threads."""

        self._start_stdin_reader()
        try:
            line = await asyncio.wait_for(
                self._stdin_lines.get(), timeout=wait_seconds
            )
        # not a failure: no line within the wait is no line, which None reports.
        except TimeoutError:
            return None
        if line is None:
            raise EOFError
        return line

    async def _chat_loop(self):
        """Main terminal I/O loop: flush pending, then accept input."""
        try:
            sys.stdout.write(BANNER)
            sys.stdout.flush()

            # Flush any pending autonomous messages first
            await self._flush_pending()

            while self._active:
                # Watch for main UI returning
                if self._is_main_ui_open():
                    sys.stdout.write("\n[Aura] Main app is back online — switching over. Bye!\n\n")
                    sys.stdout.flush()
                    await self.deactivate("main UI returned")
                    return

                # Flush any messages that arrived while waiting
                await self._flush_pending()

                # Idle timeout: close if nothing happening
                if time.time() - self._last_activity_at > IDLE_TIMEOUT_SECS:
                    sys.stdout.write(
                        "\n[Aura] No activity for a while — closing terminal session.\n\n"
                    )
                    sys.stdout.flush()
                    await self.deactivate("idle timeout")
                    return

                # Prompt for input
                sys.stdout.write(INPUT_PREFIX)
                sys.stdout.flush()

                try:
                    user_input = await self._read_stdin_line(wait_seconds=20.0)
                # not a failure: end of input is how a terminal session ends.
                except EOFError:
                    sys.stdout.write("\n[Aura] Terminal session closed.\n\n")
                    sys.stdout.flush()
                    await self.deactivate("EOF")
                    return
                if user_input is None:
                    # No input yet — loop back to check for pending messages / UI return
                    # Erase the dangling prompt
                    sys.stdout.write("\r" + " " * len(INPUT_PREFIX) + "\r")
                    sys.stdout.flush()
                    continue

                user_input = user_input.strip()
                if not user_input:
                    continue

                self._last_activity_at = time.time()

                if user_input.lower() in ("exit", "quit", "bye", ":q"):
                    sys.stdout.write("[Aura] Goodbye.\n\n")
                    sys.stdout.flush()
                    await self.deactivate("user exit")
                    return

                # Shell command passthrough: prefix with ! to run in terminal
                if user_input.startswith("!"):
                    await self._run_shell_command(user_input[1:].strip())
                    continue

                response = await self._get_response(user_input)
                await self._write_output_scheduled(response)

        except asyncio.CancelledError as _exc:
            logger.debug("Suppressed asyncio.CancelledError: %s", _exc)
        except (RuntimeError, TimeoutError, AttributeError) as e:
            _record_terminal_degradation(
                e,
                action="closed terminal fallback chat loop after recoverable runtime failure",
            )
            logger.error("TerminalFallback chat loop error: %s", e)
            self._active = False
        finally:
            self._stop_stdin_reader()
            self._active = False
            if self._chat_task is asyncio.current_task():
                self._chat_task = None
            self._orch = None

    async def _flush_pending(self):
        """Deliver all pending autonomous messages to terminal."""
        while self._pending:
            msg = self._pending.popleft()
            await self._write_output_scheduled(msg)
            self._last_activity_at = time.time()

    def _validate_shell_command(self, cmd: str) -> tuple[bool, str]:
        if len(cmd) > 8_192:
            return False, "command is too long for terminal fallback mode"
        if "\x00" in cmd:
            return False, "command contains an invalid NUL byte"

        try:
            from core.autonomy.behavior_controller import AutonomousBehaviorController

            controller = AutonomousBehaviorController()
            if not controller.validate_action({"type": "terminal", "command": cmd}):
                return False, "that command is on the safety deny-list"
            return True, "approved"
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_terminal_degradation(
                exc,
                action="blocked terminal shell command because safety validation was unavailable",
                severity="degraded",
                extra={"command_preview": cmd[:160]},
            )
            return (
                False,
                "terminal command safety checks are unavailable, so I blocked it fail-closed",
            )

    async def _terminate_shell_process(self, proc, *, cmd: str) -> None:
        try:
            proc.kill()
        # not a failure: a process already gone is the state this is killing it into.
        except ProcessLookupError:
            return
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            _record_terminal_degradation(
                exc,
                action="attempted to kill timed-out terminal shell process",
                severity="degraded",
                extra={"command_preview": cmd[:160]},
            )
            return

        try:
            await asyncio.wait_for(proc.wait(), timeout=SHELL_KILL_GRACE_SECS)
        except TimeoutError as exc:
            _record_terminal_degradation(
                exc,
                action="timed-out terminal shell process did not exit after kill",
                severity="degraded",
                extra={"command_preview": cmd[:160]},
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            _record_terminal_degradation(
                exc,
                action="reaped timed-out terminal shell process after kill",
                severity="degraded",
                extra={"command_preview": cmd[:160]},
            )

    async def _run_shell_command(self, cmd: str):
        """Execute a shell command and stream output to terminal.

        User-initiated only (prefixed with !). Aura does NOT autonomously
        run shell commands without being asked.

        Safety: routes through BehaviorController.validate_action() first.
        Blocked patterns (rm -rf /, mkfs, kill -9 -1, etc.) are rejected.
        """
        if not cmd:
            return

        allowed, reason = self._validate_shell_command(cmd)
        if not allowed:
            await self._write_output_scheduled(f"[Aura] Blocked: {reason}.")
            return

        sys.stdout.write(f"[Aura] Running: {cmd}\n")
        sys.stdout.flush()
        try:
            proc = await get_subprocess_gateway().spawn_shell_async(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                source="tool_execution:terminal_chat.shell",
                accelerator_capability="auto",
            )
            try:
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=SHELL_COMMAND_TIMEOUT_SECS,
                )
            except TimeoutError as exc:
                await self._terminate_shell_process(proc, cmd=cmd)
                _record_terminal_degradation(
                    exc,
                    action="killed timed-out terminal shell command and returned timeout result",
                    severity="degraded",
                    extra={"command_preview": cmd[:160]},
                )
                await self._write_output_scheduled(
                    f"[Aura] Command timed out ({SHELL_COMMAND_TIMEOUT_SECS:.0f}s limit)."
                )
                return

            output = stdout.decode(errors="replace").strip() if stdout else ""
            output_truncated = False
            if len(output) > SHELL_OUTPUT_CHAR_CAP:
                output = output[:SHELL_OUTPUT_CHAR_CAP].rstrip()
                output_truncated = True
            if output:
                # Stream output lines directly (not through _write_output to preserve formatting)
                lines = output.splitlines()
                for line in lines[:SHELL_OUTPUT_LINE_CAP]:
                    sys.stdout.write(f"  {line}\n")
                if output_truncated or len(lines) > SHELL_OUTPUT_LINE_CAP:
                    sys.stdout.write("  ...[terminal output truncated]\n")
                sys.stdout.flush()
            else:
                sys.stdout.write("  (no output)\n")
                sys.stdout.flush()

            # Also feed result to Aura's inference so she can comment on it
            if self._orch and output:
                try:
                    comment = await asyncio.wait_for(
                        self._get_response(f"[Terminal output from '{cmd}']: {output[:400]}"),
                        timeout=SHELL_COMMENT_TIMEOUT_SECS,
                    )
                    if comment and "[Terminal output" not in comment:
                        await self._write_output_scheduled(comment)
                except TimeoutError as exc:
                    _record_terminal_degradation(
                        exc,
                        action="skipped terminal output commentary after response timeout",
                        severity="debug",
                        extra={"command_preview": cmd[:160]},
                    )

        except (RuntimeError, subprocess.SubprocessError, OSError) as e:
            _record_terminal_degradation(
                e,
                action="returned shell error after terminal command launch failed",
                severity="degraded",
                extra={"command_preview": cmd[:160]},
            )
            await self._write_output_scheduled(f"[Aura] Shell error: {e}")

    async def _get_response(self, user_input: str) -> str:
        """Route user input through the orchestrator."""
        if self._orch is None:
            return "Running in minimal mode — orchestrator unavailable."
        try:
            if hasattr(self._orch, "process_user_input"):
                result = await asyncio.wait_for(
                    self._orch.process_user_input(user_input, origin="terminal_fallback"),
                    timeout=30.0,
                )
                if isinstance(result, dict):
                    return result.get("response") or result.get("text") or str(result)
                return str(result) if result else "..."
        except TimeoutError:
            return "Response timed out — I'm running slowly in emergency mode."
        except OSError as e:
            _record_terminal_degradation(
                e,
                action="returned terminal fallback response error to user",
                severity="degraded",
            )
            logger.debug("TerminalFallback response error: %s", e)
            return f"[error: {e}]"
        return "Message received but can't fully respond in this mode."

    def _write_output(self, text: str) -> None:
        """Format and write one output; scheduling belongs to the async caller."""

        lines = str(text).strip().splitlines()
        if lines:
            sys.stdout.write(f"\n{OUTPUT_PREFIX}{lines[0]}\n")
            for line in lines[1:]:
                sys.stdout.write(f"       {line}\n")
        sys.stdout.flush()
        self._last_output_at = time.monotonic()

    async def _write_output_scheduled(self, text: str) -> None:
        """Serialize terminal speech and rate-limit it without blocking the loop."""

        async with self._output_lock:
            since_last = time.monotonic() - self._last_output_at
            if since_last < MIN_OUTPUT_INTERVAL:
                await asyncio.sleep(MIN_OUTPUT_INTERVAL - since_last)
            self._write_output(text)

    # ── Environment detection ─────────────────────────────────────────────────

    def _is_main_ui_open(self) -> bool:
        """True only when this runtime has an authenticated owner UI client."""

        try:
            from interface.websocket_manager import ws_manager

            return ws_manager.owner_count() > 0
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_terminal_degradation(
                exc,
                action="treated main UI as unavailable after owner-client probe failed",
                severity="warning",
            )
            logger.debug("Suppressed Exception: %s", exc)
        return False


class TerminalWatchdog:
    """Background monitor that autonomously opens terminal chat when needed.

    Watches two things independently:
      1. UI presence  — tracks how long the main UI has been gone.
      2. Message queue — Aura has something to say (queue_autonomous_message called).

    Opens terminal only when BOTH are true:
      - UI confirmed gone for >= UI_GONE_CONFIRMATION_SECS
      - At least one message is pending (Aura has something to say)

    This is intentionally conservative: brief WebSocket blips, app relaunches,
    and screen-off states won't trigger a spurious terminal window.
    """

    def __init__(self, chat: TerminalFallbackChat, orchestrator=None):
        self._chat = chat
        self._orch = orchestrator
        self._running = False
        self._task: asyncio.Task | None = None
        self._ui_gone_since: float | None = None  # timestamp UI was last confirmed gone

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = get_task_tracker().create_task(self._watch_loop(), name="TerminalWatchdog")
        logger.info("📟 TerminalWatchdog monitoring UI presence.")

    async def stop(self):
        self._running = False
        task = self._task
        self._task = None
        if task and task is not asyncio.current_task() and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _watch_loop(self):
        while self._running:
            try:
                await asyncio.sleep(WATCHDOG_INTERVAL_SECS)
                await self._tick()
            except asyncio.CancelledError:
                if not self._running or is_shutdown_requested():
                    break
                logger.warning("TerminalWatchdog spuriously cancelled. Ignoring.")
                continue
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                _record_terminal_degradation(
                    e,
                    action="kept terminal watchdog alive after recoverable tick failure",
                    severity="warning",
                )
                logger.debug("TerminalWatchdog tick error: %s", e)

    async def _tick(self):
        ui_open = self._chat._is_main_ui_open()

        if ui_open:
            # UI is back — reset gone-timer, deactivate terminal if it was open
            self._ui_gone_since = None
            if self._chat.is_active:
                sys.stdout.write(
                    "\n[Aura] Main app came back online — closing terminal session.\n\n"
                )
                sys.stdout.flush()
                await self._chat.deactivate("UI returned (watchdog)")
            return

        # UI is gone — start or continue tracking how long
        now = time.time()
        if self._ui_gone_since is None:
            self._ui_gone_since = now
            logger.debug("TerminalWatchdog: UI gone, starting confirmation timer.")
            return

        gone_for = now - self._ui_gone_since

        # Not confirmed gone long enough yet
        if gone_for < UI_GONE_CONFIRMATION_SECS:
            return

        # Has Aura has something to say?
        if not self._chat._pending:
            return

        # All conditions met — autonomously open terminal
        if not self._chat.is_active:
            logger.warning(
                "📟 TerminalWatchdog: UI gone %.0fs, %d pending messages — opening terminal.",
                gone_for,
                len(self._chat._pending),
            )
            await self._chat.activate(orchestrator=self._orch)


# ── Singleton helpers ─────────────────────────────────────────────────────────

_fallback: TerminalFallbackChat | None = None
_watchdog: TerminalWatchdog | None = None


def get_terminal_fallback() -> TerminalFallbackChat:
    global _fallback
    if _fallback is None:
        _fallback = TerminalFallbackChat()
    return _fallback


def get_terminal_watchdog(orchestrator=None) -> TerminalWatchdog:
    global _watchdog
    if _watchdog is None:
        _watchdog = TerminalWatchdog(get_terminal_fallback(), orchestrator=orchestrator)
    return _watchdog
