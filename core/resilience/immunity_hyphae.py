"""Immunity Hyphae: Deterministic Python-Based Error Healing
-----------------------------------------------------------
Part of Aura's Immune System 2.0. Hardwired exception matching
and repair for known common failure signatures.
"""

import logging
import sys
import os
import traceback
from typing import Dict, Callable, Any, List, Optional, Set
from pathlib import Path
import time

logger = logging.getLogger("Aura.Resilience.Immunity")

class CircuitBreaker:
    """Tracks component failures and 'quarantines' them if they enter a crash-loop."""
    
    def __init__(self, threshold: int = 3, recovery_time: int = 300):
        self.threshold = threshold
        self.recovery_time = recovery_time
        self.failure_counts: Dict[str, int] = {}
        self.quarantined_until: Dict[str, float] = {}
        self.blacklisted: Set[str] = set()

    def report_failure(self, component: str):
        """Register a failure for a component."""
        now = time.time()
        self.failure_counts[component] = self.failure_counts.get(component, 0) + 1
        
        if self.failure_counts[component] >= self.threshold:
            logger.warning("🚨 [CIRCUIT] Component '%s' triggered circuit breaker. Quarantining for %ds.", component, self.recovery_time)
            self.quarantined_until[component] = now + self.recovery_time
            self.blacklisted.add(component)

    def is_quarantined(self, component: str) -> bool:
        """Check if a component is currently quarantined."""
        if component not in self.blacklisted:
            return False
        
        now = time.time()
        if now > self.quarantined_until.get(component, 0):
            # Recovery
            logger.info("🛡️ [CIRCUIT] Component '%s' recovered from quarantine.", component)
            self.blacklisted.remove(component)
            self.failure_counts[component] = 0
            return False
            
        return True

class SignatureRepairRegistry:
    """Registry of known error signatures and their deterministic Python-based repairs."""
    
    def __init__(self):
        self.signatures: List[Dict[str, Any]] = []
        self._load_default_signatures()

    def _load_default_signatures(self):
        """Pre-defined signatures for common Aura failure modes."""
        
        # 1. Stale PID File (Common on crashes/force-stops)
        self.register(
            name="stale_pid_cleanup",
            error_patterns=[
                "PID file already exists", 
                "locked by another process",
                "[Errno 17] File exists: 'aura.pid'"
            ],
            repair_fn=self._repair_pid_cleanup
        )
        
        # 2. Missing Critical Data Directories
        self.register(
            name="data_dir_recovery",
            error_patterns=[
                "No such file or directory:",
                "data/brain",
                "data/vault"
            ],
            repair_fn=self._repair_data_dirs
        )
        
        # 3. Port Conflict (Binding errors)
        self.register(
            name="port_retry",
            error_patterns=[
                "[Errno 48] Address already in use",
                "Binding specifically to port failed",
                "EADDRINUSE"
            ],
            repair_fn=self._repair_port_conflict
        )

        # 4. SQLite / File Lock (Stale locks from crashes)
        self.register(
            name="lock_cleanup",
            error_patterns=[
                "database is locked",
                "sqlite3.OperationalError: database is locked",
                "another process is currently using the file"
            ],
            repair_fn=self._repair_lock_cleanup
        )

    def register(self, name: str, error_patterns: List[str], repair_fn: Callable):
        self.signatures.append({
            "name": name,
            "patterns": error_patterns,
            "fn": repair_fn
        })

    def match_and_repair(self, error_msg: str) -> bool:
        """Attempt to match the error against known signatures and apply the repair."""
        for sig in self.signatures:
            if any(p in error_msg for p in sig["patterns"]):
                logger.info("💉 [IMMUNE] Signature match: %s. Initiating repair...", sig["name"])
                try:
                    sig["fn"]()
                    logger.info("✅ [IMMUNE] Deterministic repair successful: %s", sig["name"])
                    return True
                except Exception as e:
                    logger.error("❌ [IMMUNE] Repair failed for %s: %s", sig["name"], e)
        return False

    # --- Deterministic Repair Realizations ---

    def _repair_pid_cleanup(self):
        pid_file = "aura.pid"
        if os.path.exists(pid_file):
            logger.warning("💉 Removing stale PID file: %s", pid_file)
            os.remove(pid_file)

    def _repair_data_dirs(self):
        """Ensures common data structure exists."""
        try:
            from core.config import config
            paths = [
                config.paths.data_dir / "brain",
                config.paths.data_dir / "vault",
                config.paths.data_dir / "error_logs",
                config.paths.data_dir / "snapshots"
            ]
            for p in paths:
                if not p.exists():
                    logger.warning("💉 Restoring missing system directory: %s", p)
                    p.mkdir(parents=True, exist_ok=True)
        except ImportError:
            logger.error("💉 Could not import config for data dir repair")

    def _repair_port_conflict(self):
        """Generic logic for port conflicts is usually to wait or kill the zombie."""
        logger.warning("💉 Port conflict detected. Recommend process cleanup.")
        # Attempt to find the PID using this port (Mac specific lsof)
        try:
            from core.config import config
            port = getattr(config.server, 'port', 8000)
            import subprocess
            res = subprocess.run(["lsof", f"-ti:{port}"], capture_output=True, text=True)
            if res.stdout.strip():
                zombie_pid = res.stdout.strip().split('\n')[0]
                logger.warning("💉 [IMMUNE] Found zombie process %s on port %s. Cleaning up...", zombie_pid, port)
                subprocess.run(["kill", "-9", zombie_pid])
        except Exception as e:
            logger.debug("Port repair probe failed: %s", e)

    def _repair_lock_cleanup(self):
        """Cleans up stale WAL/SHM files for SQLite if a lock is detected."""
        logger.warning("💉 [IMMUNE] Database lock detected. Checking for stale WAL files...")
        try:
            from core.config import config
            db_path = config.paths.data_dir / "aura.db"
            for suffix in ["-wal", "-shm"]:
                ext_path = db_path.with_name(db_path.name + suffix)
                if ext_path.exists():
                    logger.info("💉 Removing stale SQLite file: %s", ext_path)
                    ext_path.unlink()
        except Exception as e:
            logger.error("💉 Lock cleanup failed: %s", e)

    def log_sieve(self, log_files: List[Path]) -> List[str]:
        """Scan logs for common 'hidden' errors that often get masked by '0 bugs' reports."""
        hidden_bugs = []
        # Patterns that indicate real issues but might not trigger a crash immediately
        critical_patterns = [
            "Connection refused",
            "TimeoutError",
            "BrokenPipeError",
            "MemoryError",
            "Authentication failed",
            "Atomic Guard Reject"
        ]
        
        for log_path in log_files:
            if not log_path.exists(): continue
            try:
                # Read last 100 lines
                with open(log_path, 'r') as f:
                    all_lines = f.readlines()
                    lines = all_lines[max(0, len(all_lines)-100):]
                    for line in lines:
                        if any(p in line for p in critical_patterns):
                            hidden_bugs.append(f"[{log_path.name}] {line.strip()}")
            except Exception as e:
                logger.error("💉 Log Sieve failed for %s: %s", log_path, e)
        
        if hidden_bugs:
             logger.warning("💉 [IMMUNE] Log Sieve detected %d hidden issues!", len(hidden_bugs))
        return hidden_bugs

class ImmunityHyphae:
    """Hardwired error propagator for deterministic repair."""
    
    _instance = None
    
    def __init__(self):
        if not hasattr(self, 'registry'):
            self.registry = SignatureRepairRegistry()
        if not hasattr(self, 'circuit_breaker'):
            self.circuit_breaker = CircuitBreaker()
        if not hasattr(self, '_hooked'):
            self._hooked = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImmunityHyphae, cls).__new__(cls)
        return cls._instance

    @property
    def hooked(self) -> bool:
        return getattr(self, "_hooked", False)

    def hook_system(self):
        """Hardwire into Python's exception handling at the global level."""
        if self.hooked:
            return
        
        original_excepthook = sys.excepthook
        
        def immune_excepthook(exc_type, exc_value, exc_traceback):
            error_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            errors: list[str] = [] # Ensure errors list is correctly typed
            if self.registry.match_and_repair(error_str):
                # Repaired!
                pass
            original_excepthook(exc_type, exc_value, exc_traceback)
            
        sys.excepthook = immune_excepthook
        self._hooked = True
        logger.info("💉 ImmunityHyphae: Global exception hook installed.")

    def audit_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Manual entry point for caught exceptions to be matched against signatures."""
        error_msg = f"{type(error).__name__}: {str(error)}"
        component = context.get("component", "unknown") if context else "unknown"
        
        if self.circuit_breaker.is_quarantined(component):
            logger.debug("🛡️ [CIRCUIT] Silent suppression for quarantined component: %s", component)
            return True # Pretend handled

        try:
            from core.container import ServiceContainer

            adaptive_immune = ServiceContainer.get("adaptive_immune_system", default=None)
            if adaptive_immune and hasattr(adaptive_immune, "observe_error"):
                adaptive_immune.observe_error(error, {
                    **(context or {}),
                    "component": component,
                    "stack_trace": traceback.format_exc(),
                })
        except Exception as exc:
            logger.debug("Adaptive immune escalation skipped: %s", exc)

        repaired = self.registry.match_and_repair(error_msg)
        
        if not repaired and component != "unknown":
            self.circuit_breaker.report_failure(component)
            
        return repaired

# Global Access Hook
_immunity = ImmunityHyphae()

def get_immunity() -> ImmunityHyphae:
    return _immunity
