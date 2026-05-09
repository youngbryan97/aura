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
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger("Infra.Watchdog")

class SystemWatchdog:
    """Monitors system heartbeats from an external thread.
    
    If a component fails to emit a heartbeat within its defined timeout,
    the watchdog logs a critical error and can trigger a recovery action.
    """
    
    def __init__(self, check_interval: float = 5.0):
        self._check_interval = check_interval
        self._heartbeats: Dict[str, float] = {}
        self._timeouts: Dict[str, float] = {}
        self._callbacks: Dict[str, Callable] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stalled: set[str] = set()
        
    def register_component(
        self, 
        name: str, 
        timeout: float = 60.0, 
        on_stall: Optional[Callable] = None
    ):
        """Register a component to be monitored."""
        with self._lock:
            self._heartbeats[name] = time.time()
            self._timeouts[name] = timeout
            if on_stall:
                self._callbacks[name] = on_stall
        logger.info("Watchdog registered component: %s (timeout: %.1fs)", name, timeout)

    def heartbeat(self, name: str):
        """Record a heartbeat for a component."""
        with self._lock:
            if name in self._heartbeats:
                self._heartbeats[name] = time.time()
            else:
                logger.warning("Watchdog received heartbeat for unknown component: %s", name)

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
                    timeout = self._timeouts.get(name, 60.0)
                    if now - last_seen > timeout:
                        if name not in self._stalled:
                            stalled_components.append(name)
                            self._stalled.add(name)
                    else:
                        self._stalled.discard(name)
            
            for name in stalled_components:
                logger.critical(
                    "🔥 SYSTEM STALL DETECTED: Component '%s' has not responded for %.1fs!",
                    name, now - self._heartbeats[name]
                )
                
                # Trigger recovery callback if registered
                callback = self._callbacks.get(name)
                if callback:
                    try:
                        logger.warning("Executing recovery callback for %s", name)
                        callback()
                    except Exception as e:
                        logger.error("Recovery callback for %s failed: %s", name, e)
                
                # A+ Hardening: Auto-Rollback on persistent stall
                # If it's a critical component like 'orchestrator' or 'brain'
                if name in ["orchestrator", "cognitive_engine", "server"]:
                    logger.critical("🚨 CRITICAL COMPONENT STALL. Attempting state rollback...")
                    try:
                        from core.resilience.snapshot_manager import SnapshotManager
                        sm = SnapshotManager()
                        if sm.rollback():
                            logger.info("✅ Watchdog-initiated rollback successful. Restarting system might be required.")
                    except Exception as e:
                        logger.error("Watchdog rollback failed: %s", e)

                pass
            self._stop_event.wait(self._check_interval)

_global_watchdog: Optional[SystemWatchdog] = None
_watchdog_lock = threading.Lock()

def get_watchdog() -> SystemWatchdog:
    """Get or create the global system watchdog."""
    global _global_watchdog
    with _watchdog_lock:
        if _global_watchdog is None:
            _global_watchdog = SystemWatchdog()
        return _global_watchdog
