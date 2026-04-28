from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("Aura.HealingSwarm")

class HealingSwarmService:
    """
    [PHASE 8] HEALING SWARM SERVICE
    Monitors SubsystemAudit for STALE or failing components.
    Spawns recovery shards via SovereignSwarm to attempt autonomous repair.
    """
    # Boot grace mirrors SelfModificationEngine._health_watcher_loop — all
    # subsystems need time to register their first heartbeat before the
    # audit's NEVER_SEEN signal means anything.
    _BOOT_GRACE_SECONDS = 300.0

    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self.is_running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._repair_history: Dict[str, float] = {}
        self._started_at: float = 0.0

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._started_at = time.time()
        self._monitor_task = get_task_tracker().create_task(self._monitor_loop())
        logger.info("🛡️ Healing Swarm Service ONLINE.")

    async def _monitor_loop(self):
        # Match SelfModificationEngine's 300s boot grace so we don't spawn
        # repair shards for subsystems that simply haven't checked in yet.
        await asyncio.sleep(self._BOOT_GRACE_SECONDS)
        while self.is_running:
            try:
                await asyncio.sleep(45) # Lower frequency than MetaCognition
                await self.reconcile_subsystems()
            except Exception as e:
                record_degradation('healing_swarm', e)
                logger.error(f"Healing Swarm monitor loop failed: {e}")
                await asyncio.sleep(10)

    async def reconcile_subsystems(self):
        """Check all subsystems and trigger repairs if needed."""
        # SubsystemAudit is registered as 'subsystem_audit'
        audit = getattr(self.orchestrator, 'subsystem_audit', None)
        if not audit:
            return

        health = audit.check_health()
        if health.get("all_ok"):
            return

        for name, info in health.get("subsystems", {}).items():
            status = info.get("status", "UNKNOWN")
            # Only act on subsystems that genuinely went STALE/DEGRADED. A
            # NEVER_SEEN subsystem hasn't booted yet — it's not failing, it
            # just hasn't sent its first heartbeat. Spawning a recovery
            # shard for every NEVER_SEEN entry on every cycle exhausts the
            # SovereignSwarm's capacity (M5 Pro 64GB safeguard) and burns
            # LLM calls retrying ShardResponse generations that can't
            # actually fix anything.
            if status not in {"STALE", "DEGRADED"}:
                continue
            await self.attempt_repair(name, info)

    async def attempt_repair(self, subsystem_name: str, info: Dict[str, Any]):
        """Trigger an autonomous repair shard for a failing subsystem."""
        now = time.time()
        # Cooldown: Don't spam repairs for the same component (5 min)
        last_repair = self._repair_history.get(subsystem_name, 0)
        if now - last_repair < 300:
            return

        status = info.get("status", "UNKNOWN")
        logger.warning("🚨 [HEAL] Attempting autonomous repair for %s (%s)", subsystem_name, status)
        self._repair_history[subsystem_name] = now

        # Use SovereignSwarm to spawn a recovery shard
        # SovereignSwarm is accessible via self.orchestrator.sovereign_swarm
        swarm = getattr(self.orchestrator, 'sovereign_swarm', None)
        if swarm:
            stale_seconds = info.get('stale_seconds', 'N/A')
            goal = f"Identify root cause for {subsystem_name} failure and suggest/trigger fix."
            context = f"Subsystem {subsystem_name} is in status {status}. Stale for {stale_seconds}s."
            success = await swarm.spawn_shard(goal, context)
            if success:
                logger.info("🛡️ [HEAL] Recovery shard spawned for %s.", subsystem_name)
            else:
                logger.error("🛡️ [HEAL] Failed to spawn recovery shard for %s (Capacity reached).", subsystem_name)
        else:
            logger.error("🛡️ [HEAL] SovereignSwarm not available for repair.")

