import multiprocessing
import multiprocessing.connection
import time
import logging
import signal
import os
import sys
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger("Aura.Supervisor")

@dataclass
class ActorSpec:
    name: str
    entry_point: Optional[Callable] = None
    target: Optional[Callable] = None  # Alias for entry_point to match Orchestrator usage
    args: tuple = field(default_factory=tuple)
    restart_policy: str = "always" # always, transient, never
    max_restarts: int = 3
    restart_delay: float = 1.0 # Base delay
    backoff_factor: float = 2.0 # Exponential backoff factor
    window_seconds: int = 60 # Period to track consecutive failures
    health_timeout: float = 30.0 # PIPELINE HARDENING: Generous timeout to prevent false kills
    grace_period: float = 45.0 # PIPELINE HARDENING: Long grace for model loading on M5

    def __post_init__(self):
        if self.target and not self.entry_point:
            self.entry_point = self.target
        if not self.entry_point:
            raise ValueError("ActorSpec requires either entry_point or target")

class ActorHealthGate:
    """
    ZENITH LOCKDOWN: Health gating for actors.
    Provides grace periods and miss thresholds for heartbeats.
    """
    def __init__(self, grace_period: float = 15.0, timeout: float = 10.0):
        self.start_time = time.time()
        self.last_heartbeat = time.time()
        self.grace_period = grace_period
        self.timeout = timeout
        self.miss_count = 0
        self.max_misses = 3

    def record_heartbeat(self):
        self.last_heartbeat = time.time()
        self.miss_count = 0

    def is_healthy(self) -> bool:
        now = time.time()
        # Grace period for boot
        if now - self.start_time < self.grace_period:
            return True
        # Check timeout
        if now - self.last_heartbeat > self.timeout:
            self.miss_count += 1
            return self.miss_count <= self.max_misses
        return True
    
@dataclass
class ManagedActor:
    spec: ActorSpec
    process: Optional[multiprocessing.Process] = None
    pipe: Optional[multiprocessing.connection.Connection] = None
    restarts: int = 0
    consecutive_failures: int = 0
    last_restart: float = 0.0
    next_restart_time: float = 0.0
    is_circuit_broken: bool = False
    health_gate: Optional[ActorHealthGate] = None
    monitor_health: bool = False

class SupervisionTree:
    """
    The 'Immune System' of Aura.
    Hierarchical supervisor for sovereign processes.
    """
    def __init__(self):
        self._actors: Dict[str, ManagedActor] = {}
        self._is_running = False
        self._restart_callback: Optional[Callable[[str, Any], None]] = None

    def set_restart_callback(self, callback: Callable[[str, Any], None]):
        """Set a callback for when an actor is restarted with a new pipe."""
        self._restart_callback = callback

    def add_actor(self, spec: ActorSpec):
        """Register a new actor spec."""
        self._actors[spec.name] = ManagedActor(spec=spec)
        logger.info(f"🛡️ Actor Registered for Supervision: {spec.name}")

    def is_actor_running(self, name: str) -> bool:
        actor = self._actors.get(name)
        return bool(actor and actor.process and actor.process.is_alive())

    def get_actor_pipe(self, name: str):
        actor = self._actors.get(name)
        return actor.pipe if actor else None

    def record_activity(self, name: str):
        """Mark an actor as alive without directly reading from its IPC pipe."""
        actor = self._actors.get(name)
        if not actor:
            return
        if actor.health_gate is None:
            actor.health_gate = ActorHealthGate(
                grace_period=actor.spec.grace_period,
                timeout=actor.spec.health_timeout,
            )
        actor.monitor_health = True
        actor.health_gate.record_heartbeat()

    def start_actor(self, name: str):
        """Spin up a specific actor. Idempotent: returns existing pipe if already running."""
        actor = self._actors.get(name)
        if not actor:
            raise ValueError(f"Unknown actor: {name}")
            
        # If already running, just return existing pipe
        if actor.process and actor.process.is_alive():
            logger.debug(f"🛡️ start_actor: {name} is already alive (PID: {actor.process.pid}). Returning existing pipe.")
            return actor.pipe

        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        
        proc = ctx.Process(
            target=actor.spec.entry_point,
            args=(*actor.spec.args, child_conn),
            name=f"AuraActor:{name}",
            daemon=True
        )
        
        proc.start()
        # PIPELINE HARDENING: Removed time.sleep(1.5) that was blocking the
        # event loop on every actor spawn. The OS handles memory without this.
        actor.process = proc
        actor.pipe = parent_conn
        actor.last_restart = time.time()
        actor.is_circuit_broken = False
        actor.health_gate = ActorHealthGate(
            grace_period=actor.spec.grace_period,
            timeout=actor.spec.health_timeout,
        )
        actor.health_gate.record_heartbeat()
        
        logger.info(f"🚀 Actor Started: {name} (PID: {proc.pid})")
        return parent_conn

    def stop_actor(self, name: str):
        """Gracefully stop an actor.
        
        PIPELINE HARDENING: No process.join() — it blocks the event loop for
        up to 2s. We kill and let the OS reap. The process is daemon=True
        so it will be cleaned up on main process exit regardless.
        """
        actor = self._actors.get(name)
        if actor and actor.process:
            logger.info(f"🛑 Stopping Actor: {name}")
            try:
                actor.process.kill()  # Immediate kill, no graceful shutdown
            except Exception as e:
                logger.debug(f"Error stopping actor {name}: {e}")
            finally:
                actor.process = None
                actor.pipe = None

    async def start(self):
        """Async entry point for Orchestrator bootstrap."""
        self._is_running = True
        logger.info("🛡️ Supervision Tree initialized (Async).")

    async def stop(self):
        """Async lifecycle stop."""
        self.stop_all()

    async def wait_forever(self):
        """Main supervision loop (non-blocking async)."""
        import asyncio
        self._is_running = True
        logger.info("🛡️ Supervision Tree ACTIVE (Async). Monitoring actors...")
        
        try:
            while self._is_running:
                self._poll_health()
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            self.stop_all()

    def run_forever(self):
        """Main supervision loop (blocking)."""
        self._is_running = True
        logger.info("🛡️ Supervision Tree ACTIVE. Monitoring actors...")
        
        try:
            while self._is_running:
                self._poll_health()
                time.sleep(1.0)
        except KeyboardInterrupt:
            self.stop_all()

    def _poll_health(self):
        """Check all actors and restart if needed."""
        now = time.time()
        for name, actor in self._actors.items():
            if actor.is_circuit_broken:
                continue

            if actor.process and not actor.process.is_alive():
                exit_code = actor.process.exitcode
                logger.warning(f"⚠️ Actor CRASHED: {name} (Exit Code: {exit_code})")
                self._handle_failure(name)
            
            elif actor.process and actor.health_gate and actor.monitor_health:
                if not actor.health_gate.is_healthy():
                    logger.error(f"🚨 Actor STALLED (Liveness Failure): {name}")
                    self.stop_actor(name)
                    self._handle_failure(name)
            
            elif actor.process is None and actor.next_restart_time > 0 and now >= actor.next_restart_time:
                logger.info(f"♻️ Restarting Actor {name} after backoff...")
                actor.next_restart_time = 0 # Reset
                self._restart_actor(name)

    def _handle_failure(self, name: str):
        """Apply restart policy with circuit breaker and backoff."""
        actor = self._actors[name]
        
        # Mark process as gone
        actor.process = None
        actor.pipe = None

        # 1. Update Failure Tracking
        now = time.time()
        if now - actor.last_restart < actor.spec.window_seconds:
             actor.consecutive_failures += 1
        else:
             actor.consecutive_failures = 1 # Reset window
             
        if actor.consecutive_failures > actor.spec.max_restarts:
            logger.error(f"🛑 CIRCUIT BROKEN: Actor {name} failed too many times in window.")
            actor.is_circuit_broken = True
            return

        # 2. Calculate Exponential Backoff
        delay = actor.spec.restart_delay * (actor.spec.backoff_factor ** (actor.consecutive_failures - 1))
        delay = min(delay, 60.0) # Cap at 1 minute

        actor.next_restart_time = now + delay
        logger.info(f"⏳ Scheduling Restart for {name} (Attempt {actor.consecutive_failures}/{actor.spec.max_restarts}) in {delay:.1f}s...")

    def _restart_actor(self, name: str):
        """Internal helper to start actor and trigger callback."""
        new_pipe = self.start_actor(name)
        if self._restart_callback and new_pipe:
            try:
                self._restart_callback(name, new_pipe)
            except Exception as e:
                logger.error(f"❌ Restart callback failed for {name}: {e}")

    def stop_all(self):
        """Kill everything."""
        self._is_running = False
        for name in list(self._actors.keys()):
            self.stop_actor(name)
        # Ensure all multiprocess children are reaped (ORPHAN-04)
        import multiprocessing
        for p in multiprocessing.active_children():
            if p.name.startswith("AuraActor:"):
                logger.info(f"🧹 Reaping orphaned actor: {p.name}")
                p.terminate()
                p.join(timeout=1.0)
                if p.is_alive(): p.kill()
        logger.info("🛡️ Supervision Tree Shutdown Complete.")

_tree_instance: Optional[SupervisionTree] = None

def get_tree() -> SupervisionTree:
    global _tree_instance
    if _tree_instance is None:
        _tree_instance = SupervisionTree()
    return _tree_instance


def reset_tree() -> None:
    """Reset the process supervisor singleton to a clean state."""
    global _tree_instance
    if _tree_instance is not None:
        _tree_instance.stop_all()
    _tree_instance = None
