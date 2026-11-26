"""infrastructure/watchdog.py
────────────────────────
External thread-based monitoring for core system stability.

This watchdog runs in a dedicated background thread (not an asyncio task)
to ensure it can detect and report stalls even if the main asyncio event 
loop is blocked or deadlocked.
"""

import logging
import threading
import time
from collections.abc import Callable

logger = logging.getLogger("Infra.Watchdog")

_WATCHDOG_RECOVERY_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


class SystemWatchdog:
    """Monitors system heartbeats from an external thread.
    
    If a component fails to emit a heartbeat within its defined timeout,
    the watchdog logs a critical error and can trigger a recovery action.
    """
    
    _DEFAULT_ADOPTION_TIMEOUT_S = 60.0

    def __init__(self, check_interval: float = 5.0, default_timeout: float = 60.0):
        self._check_interval = check_interval
        self._default_timeout = float(default_timeout or self._DEFAULT_ADOPTION_TIMEOUT_S)
        self._heartbeats: dict[str, float] = {}
        self._timeouts: dict[str, float] = {}
        self._callbacks: dict[str, Callable] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stalled: set[str] = set()
        # Components seen only via heartbeat(). Their liveness is OBSERVABLE but
        # not ENFORCED, because nobody declared a cadence for them and this
        # class must not invent one. `orchestrator` heartbeats per processed
        # message, not on a timer, so any invented timeout fires on the first
        # quiet minute — which is worse than silence, because false stall
        # reports train everyone to ignore real ones. Measured while fixing
        # this: adopting it at 60s produced "SYSTEM STALL DETECTED: Component
        # 'orchestrator' has not responded for 60.4s!" on an idle, healthy
        # runtime. Registering with an explicit timeout is how a component opts
        # into enforcement.
        self._unenforced: set[str] = set()

    def register_component(
        self, 
        name: str, 
        timeout: float = 60.0, 
        on_stall: Callable | None = None
    ):
        """Register a component to be monitored."""
        with self._lock:
            self._heartbeats[name] = time.time()
            self._timeouts[name] = timeout
            # Registering DECLARES a cadence, which is what opts a component
            # into stall enforcement — including one previously only observed.
            self._unenforced.discard(name)
            if on_stall:
                self._callbacks[name] = on_stall
        logger.info("Watchdog registered component: %s (timeout: %.1fs)", name, timeout)
        # Registering is a request to be WATCHED. Measured live on the desktop
        # runtime: `mind_tick` registered, `orchestrator` heartbeated, and
        # "System Watchdog started" never appeared in the log — the boot path
        # that calls start() had not run, so every component was registered into
        # a dict nobody checked. A monitor that is never started is
        # indistinguishable from a monitor that finds nothing wrong.
        self.start()

    def heartbeat(self, name: str):
        """Record a heartbeat for a component, registering it if it is new.

        An unknown name used to log a warning and then drop the heartbeat, which
        left the component permanently UNWATCHED while producing a line that
        looked like mere noise. Measured live: `orchestrator` — the single most
        important component — heartbeated into that branch, so a wedged
        orchestrator could never have been detected. Adopting the component is
        the safe direction: it starts being monitored, and the adoption is said
        out loud exactly once so a typo'd name is still visible.
        """

        adopted = False
        with self._lock:
            if name not in self._heartbeats:
                self._unenforced.add(name)
                adopted = True
            self._heartbeats[name] = time.time()
        if adopted:
            logger.info(
                "Watchdog is observing %r, which heartbeats without having "
                "registered a cadence: its liveness is visible in status but no "
                "stall timeout is enforced. Call register_component(%r, "
                "timeout=...) to opt into enforcement.",
                name,
                name,
            )
            self.start()

    def start(self):
        """Start the monitoring thread."""
        if self._thread and self._thread.is_alive():
            return
            
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="AuraWatchdog", daemon=True)
        self._thread.start()
        logger.info("System Watchdog started")

    def stop(self):
        """Stop the monitoring thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("System Watchdog stopped")

    def _run(self):
        """Internal loop running in the dedicated thread."""
        while not self._stop_event.is_set():
            now = time.time()
            stalled_components = []
            
            with self._lock:
                for name, last_seen in self._heartbeats.items():
                    if name in self._unenforced:
                        # Observed, not enforced: no declared cadence, so a
                        # quiet period is not evidence of a stall.
                        continue
                    timeout = self._timeouts.get(name, 60.0)
                    if now - last_seen > timeout:
                        if name not in self._stalled:
                            stalled_components.append(name)
                            self._stalled.add(name)
                    else:
                        self._stalled.discard(name)
            
            for name in stalled_components:
                logger.critical(
                    "SYSTEM STALL DETECTED: Component '%s' has not responded for %.1fs!",
                    name, now - self._heartbeats[name]
                )
                
                # Trigger recovery callback if registered
                callback = self._callbacks.get(name)
                if callback:
                    try:
                        logger.warning("Executing recovery callback for %s", name)
                        callback()
                    except _WATCHDOG_RECOVERY_ERRORS as e:
                        logger.error("Recovery callback for %s failed: %s", name, e)
                
                # A critical component stalled. This used to claim it was
                # "Attempting state rollback..." and call
                # SnapshotManager.rollback() — a method that does not exist and
                # never has (the real API is freeze()/thaw()). Every invocation
                # raised AttributeError into the handler below and logged
                # "Watchdog rollback failed", which reads as a remedy that was
                # tried and did not work rather than a remedy that was never
                # there. Observed directly while testing this file.
                #
                # It is NOT replaced with an automatic thaw(). Restoring a
                # snapshot is a consequential, governance-gated action, and a
                # watchdog tick is the wrong authority to take it on its own —
                # a stalled component is not evidence that reverting state is
                # the right repair. So the honest behaviour is to make the
                # stall count as a real degradation and say plainly that no
                # automatic rollback is attempted.
                if name in ("orchestrator", "cognitive_engine", "server"):
                    logger.critical(
                        "CRITICAL COMPONENT STALL: %r. No automatic state "
                        "rollback is attempted — snapshot restore is a governed "
                        "action and is not a watchdog's call. Recorded as a "
                        "degradation for the recovery ladder.",
                        name,
                    )
                    try:
                        from core.observability.degradation import record_degradation

                        record_degradation(
                            "watchdog",
                            RuntimeError(
                                f"critical_component_stall:{name}:"
                                f"{now - self._heartbeats[name]:.1f}s"
                            ),
                            action=(
                                "reported a critical component stall without "
                                "attempting an ungoverned state rollback"
                            ),
                            severity="critical",
                        )
                    except _WATCHDOG_RECOVERY_ERRORS as e:
                        logger.error("Watchdog could not record the critical stall: %s", e)
            self._stop_event.wait(self._check_interval)

_global_watchdog: SystemWatchdog | None = None
_watchdog_lock = threading.Lock()

def get_watchdog() -> SystemWatchdog:
    """Get or create the global system watchdog."""
    global _global_watchdog
    with _watchdog_lock:
        if _global_watchdog is None:
            _global_watchdog = SystemWatchdog()
        return _global_watchdog
