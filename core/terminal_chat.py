"""core/terminal_chat.py
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

from core.utils.task_tracker import get_task_tracker
import asyncio
import collections
import logging
import sys
import time
from typing import Deque, Optional

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

INPUT_PREFIX  = "[Aura] You: "
OUTPUT_PREFIX = "[Aura] "
BANNER = (
    "\n╔══════════════════════════════════════════════════╗\n"
    "║  AURA — TERMINAL MODE  (main window unavailable) ║\n"
    "║  Chat normally, or prefix with ! to run a        ║\n"
    "║  shell command (e.g. !ls -la).                   ║\n"
    "║  Type 'exit' to end. Auto-closes when app opens. ║\n"
    "╚══════════════════════════════════════════════════╝\n"
)


class TerminalFallbackChat:
    """Emergency last-resort terminal chat.

    Two activation paths:
      • explicit:    await activate(orchestrator, force=True)
      • autonomous:  TerminalWatchdog detects UI gone + pending messages
    """

    def __init__(self):
        self._active: bool = False
        self._chat_task: Optional[asyncio.Task] = None
        self._last_output_at: float = 0.0
        self._last_activity_at: float = time.time()
        self._orch = None

        # Pending messages Aura wants to deliver autonomously
        self._pending: Deque[str] = collections.deque(maxlen=MAX_PENDING_MESSAGES)

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
            from core.container import ServiceContainer
            from core.constitution import get_constitutional_core

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
                    except Exception as _exc:
                        logger.debug("Suppressed Exception: %s", _exc)
                    logger.debug("TerminalFallback: constitutional gate unavailable, suppressing autonomous message: %s", reason)
                    return False
                logger.debug("TerminalFallback: constitutional gate suppressed queued autonomous message: %s", reason)
                return False
        except Exception as exc:
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
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
                logger.debug("TerminalFallback: executive gate unavailable, suppressing autonomous message: %s", exc)
                return False
            logger.debug("TerminalFallback: executive gate unavailable, proceeding degraded: %s", exc)
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
        if self._chat_task and not self._chat_task.done():
            self._chat_task.cancel()
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

    async def _chat_loop(self):
        """Main terminal I/O loop: flush pending, then accept input."""
        try:
            sys.stdout.write(BANNER)
            sys.stdout.flush()

            # Flush any pending autonomous messages first
            await self._flush_pending()

            loop = asyncio.get_running_loop()

            while self._active:
                # Watch for main UI returning
                if self._is_main_ui_open():
                    sys.stdout.write(
                        "\n[Aura] Main app is back online — switching over. Bye!\n\n"
                    )
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
                    user_input = await asyncio.wait_for(
                        loop.run_in_executor(None, sys.stdin.readline),
                        timeout=20.0,
                    )
                except asyncio.TimeoutError:
                    # No input yet — loop back to check for pending messages / UI return
                    # Erase the dangling prompt
                    sys.stdout.write("\r" + " " * len(INPUT_PREFIX) + "\r")
                    sys.stdout.flush()
                    continue
                except (EOFError, KeyboardInterrupt):
                    sys.stdout.write("\n[Aura] Terminal session closed.\n\n")
                    sys.stdout.flush()
                    await self.deactivate("EOF/keyboard interrupt")
                    return

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
                self._write_output(response)

        except asyncio.CancelledError as _exc:
            logger.debug("Suppressed asyncio.CancelledError: %s", _exc)
        except Exception as e:
            logger.error("TerminalFallback chat loop error: %s", e)
            self._active = False

    async def _flush_pending(self):
        """Deliver all pending autonomous messages to terminal."""
        while self._pending:
            msg = self._pending.popleft()
            self._write_output(msg)
            self._last_activity_at = time.time()
            await asyncio.sleep(0)  # yield

    async def _run_shell_command(self, cmd: str):
        """Execute a shell command and stream output to terminal.

        User-initiated only (prefixed with !). Aura does NOT autonomously
        run shell commands without being asked.

        Safety: routes through BehaviorController.validate_action() first.
        Blocked patterns (rm -rf /, mkfs, kill -9 -1, etc.) are rejected.
        """
        if not cmd:
            return

        # Safety check via BehaviorController
        try:
            from core.behavior_controller import AutonomousBehaviorController
            bc = AutonomousBehaviorController()
            if not bc.validate_action({"type": "terminal", "command": cmd}):
                self._write_output(f"[Aura] Blocked: that command is on the safety deny-list.")
                return
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        sys.stdout.write(f"[Aura] Running: {cmd}\n")
        sys.stdout.flush()
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            except asyncio.TimeoutError:
                proc.kill()
                self._write_output("[Aura] Command timed out (30s limit).")
                return

            output = stdout.decode(errors="replace").strip() if stdout else ""
            if output:
                # Stream output lines directly (not through _write_output to preserve formatting)
                for line in output.splitlines()[:200]:   # cap at 200 lines
                    sys.stdout.write(f"  {line}\n")
                sys.stdout.flush()
            else:
                sys.stdout.write("  (no output)\n")
                sys.stdout.flush()

            # Also feed result to Aura's inference so she can comment on it
            if self._orch and output:
                try:
                    comment = await asyncio.wait_for(
                        self._get_response(
                            f"[Terminal output from '{cmd}']: {output[:400]}"
                        ),
                        timeout=15.0,
                    )
                    if comment and "[Terminal output" not in comment:
                        self._write_output(comment)
                except asyncio.TimeoutError as _exc:
                    logger.debug("Suppressed asyncio.TimeoutError: %s", _exc)

        except Exception as e:
            self._write_output(f"[Aura] Shell error: {e}")

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
        except asyncio.TimeoutError:
            return "Response timed out — I'm running slowly in emergency mode."
        except Exception as e:
            logger.debug("TerminalFallback response error: %s", e)
            return f"[error: {e}]"
        return "Message received but can't fully respond in this mode."

    def _write_output(self, text: str):
        """Write to stdout with spam guard."""
        now = time.time()
        since_last = now - self._last_output_at
        if since_last < MIN_OUTPUT_INTERVAL:
            time.sleep(MIN_OUTPUT_INTERVAL - since_last)

        lines = str(text).strip().splitlines()
        if lines:
            sys.stdout.write(f"\n{OUTPUT_PREFIX}{lines[0]}\n")
            for line in lines[1:]:
                sys.stdout.write(f"       {line}\n")
        sys.stdout.flush()
        self._last_output_at = time.time()

    # ── Environment detection ─────────────────────────────────────────────────

    def _is_main_ui_open(self) -> bool:
        """True if the main Aura UI / WebSocket server process is running."""
        try:
            import psutil
            for proc in psutil.process_iter(["cmdline"]):
                try:
                    cmdline = " ".join(proc.info.get("cmdline") or [])
                    if any(kw in cmdline for kw in [
                        "aura_main", "aura.server", "uvicorn", "gunicorn",
                        "Aura.app", "aura_app",
                    ]):
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
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
        self._task: Optional[asyncio.Task] = None
        self._ui_gone_since: Optional[float] = None   # timestamp UI was last confirmed gone

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = get_task_tracker().create_task(self._watch_loop(), name="TerminalWatchdog")
        logger.info("📟 TerminalWatchdog monitoring UI presence.")

    async def stop(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def _watch_loop(self):
        while self._running:
            try:
                await asyncio.sleep(WATCHDOG_INTERVAL_SECS)
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
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

_fallback: Optional[TerminalFallbackChat] = None
_watchdog: Optional[TerminalWatchdog] = None


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
