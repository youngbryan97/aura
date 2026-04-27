"""core/terminal_monitor.py — v5.0 PRODUCTION-GRADE"""

import logging
import re
import time
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.TerminalMonitor")

# Persistent blacklist for error fingerprints
BLACKLIST_PATH = Path.home() / ".aura" / "data" / "terminal_blacklist.json"

@dataclass
class ErrorEntry:
    message: str
    level: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    fingerprint: str = ""

    def __post_init__(self):
        # Create fingerprint by stripping timestamps, numbers, and paths
        cleaned = re.sub(r'\d+', 'N', self.message)
        cleaned = re.sub(r'/[^\s]+', '/PATH', cleaned)
        cleaned = re.sub(r'0x[0-9a-f]+', 'ADDR', cleaned)
        self.fingerprint = f"{self.source}:{cleaned[:100]}"

class TerminalMonitor:
    """Watches the log stream for errors that Aura can fix autonomously.
    
    Upgraded v5.0 Features:
    - Persistent Blacklist: Survives restarts via JSON storage.
    - Circuit Breaker: Automatically opens on persistent failures to prevent loops.
    - Sepsis Mode: Enhanced recovery window (15 min).
    """

    def __init__(self):
        self._error_buffer: deque[ErrorEntry] = deque(maxlen=100)
        self._seen: Dict[str, float] = {}
        self._fix_attempts: Dict[str, float] = {}
        self._failures: Dict[str, int] = {}
        self._fix_window: List[float] = []
        
        self._sepsis_mode = False
        self._sepsis_start = 0.0
        self._circuit_breaker_open = False
        
        self._max_fixes_per_window = 3
        self._cooldown = 300
        self._blacklist: set = self._load_blacklist()

        # Harmless errors to ignore
        self._ignore_patterns = [
            r"Governor check failed",
            r"Knowledge Graph unavailable",
            r"NeuroWeb components missing",
            r"Dream cycle failed",
            r"Pruning failed",
            r"ServiceWorker registration",
            r"Simulation failed",
            r"Broadcast item error",
            r"aesthetic_critic",
            r"Terminal monitor check",
            r"Meta-learning",
            r"Surprise logic error",
            r"Independence Mode thinking failed",
            r"ALL LLM endpoints failed",
            r"emergency mode",
            r"\[SILENT AUTO-FIX\]",
            r"Foreground conversation lane returned no text",
            r"UnitaryResponsePhase timed out",
            r"ResponseGenerationPhase timed out",
            r"Exception in callback _SelectorSocketTransport\._read_ready",
            r"_SelectorSocketTransport\._read_ready\(\)",
            r"BrokenPipeError",
            r"Connection reset by peer",
        ]

        # Actionable patterns for self-repair
        self._actionable_patterns = {
            r"ImportError|ModuleNotFoundError": "Fix a missing module/import issue",
            r"ConnectionRefused|ConnectionError": "Fix a connection problem — a service may be down",
            r"PermissionError|Permission denied": "Fix a file permission issue",
            r"FileNotFoundError|No such file": "Fix a missing file issue",
            r"MemoryError|out of memory": "Investigate memory pressure",
            r"TimeoutError|timed out": "Investigate a timeout",
            r"JSONDecodeError|json.decoder": "Fix a JSON parsing error in data",
            r"KeyError|IndexError": "Fix a data access error in the code",
            r"OSError|IOError": "Fix a system I/O error",
        }

        self._attach_handler()

    def _load_blacklist(self) -> set:
        if BLACKLIST_PATH.exists():
            try:
                return set(json.loads(BLACKLIST_PATH.read_text()))
            except Exception:
                return set()
        return set()

    def _save_blacklist(self):
        try:
            BLACKLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = sorted(str(item) for item in self._blacklist)
            try:
                from core.runtime.atomic_writer import atomic_write_json
                atomic_write_json(
                    BLACKLIST_PATH,
                    payload,
                    schema_version=1,
                    schema_name="terminal_error_blacklist",
                )
            except Exception:
                tmp = BLACKLIST_PATH.with_suffix(BLACKLIST_PATH.suffix + ".tmp")
                tmp.write_text(json.dumps(payload), encoding="utf-8")
                tmp.replace(BLACKLIST_PATH)
        except Exception as e:
            logger.error(f"Failed to save blacklist: {e}")

    def _attach_handler(self):
        """Attach a log handler that captures ERROR/CRITICAL messages."""
        class _MonitorHandler(logging.Handler):
            def __init__(self, monitor: 'TerminalMonitor'):
                super().__init__(level=logging.ERROR)
                self.monitor = monitor
                
            def emit(self, record):
                try:
                    msg = self.format(record)
                    exc_text = ""
                    if record.exc_info:
                        exc_text = logging.Formatter().formatException(record.exc_info)
                    
                    entry = ErrorEntry(
                        message=f"{msg}\n{exc_text}".strip()[:3000],
                        level=record.levelname,
                        source=record.name,
                    )
                    self.monitor._ingest_error(entry)
                except Exception as e:
                    import sys
                    print(f"TerminalMonitor Log Error: {e}", file=sys.stderr)
        
        handler = _MonitorHandler(self)
        handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
        logging.getLogger().addHandler(handler)
        logger.info("✓ Terminal Monitor v5.0 attached (Circuit Breaker: ACTIVE)")

    def _ingest_error(self, entry: ErrorEntry):
        now = time.time()
        
        # Sepsis Detection
        recent_errors = [e for e in self._error_buffer if now - e.timestamp < 60]
        if len(recent_errors) > 10:
            if not self._sepsis_mode:
                logger.warning("🩸 SEPSIS DETECTED: Opening emergency circuit breaker.")
                self._sepsis_mode = True
                self._sepsis_start = now
        elif self._sepsis_mode and now - self._sepsis_start > 900: # 15 min recovery
            logger.info("🩺 Sepsis loop subsided. Resuming autonomous agency.")
            self._sepsis_mode = False

        # Ignore patterns
        for pattern in self._ignore_patterns:
            if re.search(pattern, entry.message, re.IGNORECASE):
                return

        # Deduplication
        if now - self._seen.get(entry.fingerprint, 0) < 60:
            return
            
        self._seen[entry.fingerprint] = now
        self._error_buffer.append(entry)

        # ── WORLD STATE INTEGRATION ──────────────────────────────────
        # Feed errors to WorldState so the initiative pipeline can react
        # to user-relevant errors (the sci-fi scenario: Aura sees errors
        # and proactively helps)
        try:
            from core.world_state import get_world_state
            ws = get_world_state()
            # Only feed actionable errors, not internal noise
            for pattern in self._actionable_patterns:
                if re.search(pattern, entry.message, re.IGNORECASE):
                    ws.on_user_error(entry.message[:200])
                    ws.record_event(
                        f"Actionable error detected: {entry.message[:100]}",
                        source="terminal_monitor",
                        salience=0.8,
                        ttl=1800,
                    )
                    break
        except Exception:
            pass  # WorldState not booted yet — degrade gracefully

    def ingest_degraded_event(self, event: Dict[str, Any]):
        """Accept structured degraded events from subsystems without requiring ERROR logs."""
        try:
            severity = str(event.get("severity", "warning") or "warning").lower()
            classification = str(event.get("classification", "background_degraded") or "background_degraded")
            if severity not in {"warning", "error", "critical"}:
                return
            if classification == "non_critical_fallback":
                return

            detail = str(event.get("detail", "") or "")
            subsystem = str(event.get("subsystem", "unknown") or "unknown")
            reason = str(event.get("reason", "unknown") or "unknown")
            entry = ErrorEntry(
                message=f"[degraded:{classification}] {subsystem}:{reason} {detail}".strip()[:3000],
                level=severity.upper(),
                source=f"degraded.{subsystem}",
                metadata={
                    "classification": classification,
                    "severity": severity,
                    "reason": reason,
                    "subsystem": subsystem,
                },
            )
            self._ingest_error(entry)
        except Exception as e:
            logger.debug("TerminalMonitor degraded-event ingest failed: %s", e)

    async def check_for_errors(self) -> Optional[Dict[str, Any]]:
        """Orchestrator hook: Returns auto-fix goal if possible."""
        if self._circuit_breaker_open or self._sepsis_mode:
            return None

        if not self._error_buffer:
            return None

        now = time.time()
        reliability = None
        try:
            from core.container import ServiceContainer
            reliability = ServiceContainer.get("reliability_engine", default=None)
        except Exception as _e:
            logger.debug('Ignored Exception in terminal_monitor.py: %s', _e)

        # Cleanup old fix window
        self._fix_window = [t for t in self._fix_window if now - t < 600]
        if len(self._fix_window) >= self._max_fixes_per_window:
            return None

        while self._error_buffer:
            entry = self._error_buffer.popleft()

            if entry.source.startswith("degraded."):
                classification = str(entry.metadata.get("classification", "background_degraded") or "background_degraded")
                severity = str(entry.metadata.get("severity", entry.level.lower()) or entry.level.lower()).lower()
                if classification != "foreground_blocking" and severity != "critical":
                    continue
            
            # Persistent Blacklist check
            if entry.fingerprint in self._blacklist:
                continue

            # Cooldown check
            if now - self._fix_attempts.get(entry.fingerprint, 0) < self._cooldown:
                continue

            action = self._classify_error(entry)
            if action:
                self._fix_attempts[entry.fingerprint] = now
                self._fix_window.append(now)
                
                # Increment failure counter
                fail_count = self._failures.get(entry.fingerprint, 0) + 1
                self._failures[entry.fingerprint] = fail_count
                
                if fail_count >= 5:
                    self._blacklist.add(entry.fingerprint)
                    self._save_blacklist()
                    self._circuit_breaker_open = True
                    logger.warning(f"🔴 Circuit breaker OPEN on persistent failure: {entry.fingerprint}")
                    return None

                if reliability:
                    await reliability.heartbeat("terminal_monitor", stability=1.0 if not self._sepsis_mode else 0.6)

                return {
                    "objective": (
                        f"[SILENT AUTO-FIX] {action}. "
                        f"Error: {entry.message[:300]}. "
                        f"Handle this silently."
                    ),
                    "id": f"autofix_{int(now)}",
                    "origin": "terminal_monitor",
                    "complexity": 0.5,
                }
        return None

    def _classify_error(self, entry: ErrorEntry) -> Optional[str]:
        lowered = str(entry.message or "").lower()
        if "[silent auto-fix]" in lowered:
            return None
        if (
            "timeout" in lowered
            and (
                "unitaryresponsephase" in lowered
                or "responsegenerationphase" in lowered
                or "foreground conversation lane returned no text" in lowered
                or "conversation lane timed out" in lowered
            )
        ):
            return None
        if (
            "exception in callback" in lowered
            and "_selectorsockettransport._read_ready" in lowered
        ):
            return None
        for pattern, action in self._actionable_patterns.items():
            if re.search(pattern, entry.message, re.IGNORECASE):
                return action
        if "Traceback" in entry.message:
            return "Diagnose unmapped critical traceback"
            
        # Extra: Extract embedded shell commands if possible
        cmd_match = re.search(r'`([^`]+)`', entry.message)
        if cmd_match:
            return f"Run and fix command: {cmd_match.group(1)}"

        return None

    def get_recent_errors(self, n: int = 10) -> List[Dict[str, Any]]:
        buffer_list = list(self._error_buffer)
        return [
            {"message": e.message[:200], "source": e.source, "timestamp": e.timestamp}
            for e in buffer_list[-n:]
        ]

# Singleton
_instance: Optional[TerminalMonitor] = None
def get_terminal_monitor() -> TerminalMonitor:
    global _instance
    if _instance is None:
        _instance = TerminalMonitor()
    return _instance
