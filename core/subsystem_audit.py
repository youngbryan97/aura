from core.runtime.errors import record_degradation
import logging
import time
import os
import psutil
from typing import Any, Dict, List

logger = logging.getLogger("Aura.SubsystemAudit")


class SubsystemAudit:
    """Tracks and verifies that all cognitive subsystems are actively running."""
    
    # All known subsystems and their expected pulse interval (seconds)
    SUBSYSTEMS = {
        "personality_engine":   60,
        "liquid_state":         5,
        "liquid_substrate":     5,
        "drive_controller":     60,
        "consciousness":        120,
        "affect_engine":        30,
        "agency_core":          10,
        "capability_engine":    60,
        "identity":             300,
        "cognitive_engine":     120,
        "sovereign_scanner":    10,
        # v47 FIX: Removed 'database_hygiene', 'memory', 'belief_graph', 'memory_manager', 'pulse_manager', 'soma'
        # These either heartbeat infrequently from the metabolic loop, or are obsolete aliases
        # that create false alarm noise in the health dashboard.
    }
    
    def __init__(self):
        self._heartbeats: Dict[str, float] = {}
        self._failures: Dict[str, List[Dict[str, Any]]] = {}
        self._last_health_pulse = time.time()
        self._health_pulse_interval = 15  # TEMPORARY TEST: Emit full health report every 15s
        self._cycle_counts = 0
        self._start_time = time.time()
        logger.info("🫀 SubsystemAudit initialized. Tracking %d subsystems.", len(self.SUBSYSTEMS))
    
    def heartbeat(self, subsystem_name: str):
        """Register a heartbeat from a subsystem."""
        self._heartbeats[subsystem_name] = time.time()

    def report_failure(self, subsystem_name: str, error: str):
        """Record a failure event for a subsystem."""
        if subsystem_name not in self._failures:
            self._failures[subsystem_name] = []
        
        self._failures[subsystem_name].append({
            "timestamp": time.time(),
            "error": error
        })
        # Keep only last 5 failures
        history = self._failures[subsystem_name]
        if len(history) > 5:
            self._failures[subsystem_name] = history[-5:]
        logger.error("🚨 Subsystem [%s] reported failure: %s", subsystem_name, error)

    def get_status(self, subsystem_name: str) -> Dict[str, Any]:
        """Get the current health status of a specific subsystem."""
        now = time.time()
        last_beat = self._heartbeats.get(subsystem_name)
        failures = self._failures.get(subsystem_name, [])
        
        degraded = len(failures) >= 3  # Simple threshold for degradation
        
        stale = (now - last_beat) if last_beat else None
        max_interval = self.SUBSYSTEMS.get(subsystem_name, 300)
        
        is_stale = stale > max_interval * 2 if stale else False
        
        # Derive human-readable status
        if last_beat is None:
            status = "NEVER_SEEN"
        elif is_stale:
            status = "STALE"
        elif degraded:
            status = "DEGRADED"
        else:
            status = "ACTIVE"
        
        return {
            "name": subsystem_name,
            "status": status,
            "active": last_beat is not None and not is_stale,
            "degraded": degraded,
            "stale_seconds": int(stale) if stale else None,
            "failure_count": len(failures),
            "last_error": failures[-1]["error"] if failures else None
        }
    
    def check_health(self) -> Dict[str, Any]:
        """Check all subsystems and return their status."""
        now = time.time()
        report = {}
        all_ok = True
        
        # 1. Standard Subsystem Checks
        for name in self.SUBSYSTEMS:
            status_info = self.get_status(name)
            report[name] = status_info
            if not status_info["active"] or status_info["degraded"]:
                all_ok = False
        
        # 2. AFFECTIVE ESCALATION (Phase 23)
        try:
            from core.container import ServiceContainer
            homeostasis = ServiceContainer.get("homeostasis", default=None)
            if homeostasis:
                status = homeostasis.get_status()
                # If Will to Live/Vitality or Integrity is critically low, escalate
                vitality = status.get("will_to_live", 1.0)
                integrity = status.get("integrity", 1.0)
                
                if integrity < 0.3 or vitality < 0.3:
                    all_ok = False
                    report["homeostasis_escalation"] = {
                        "status": "CRITICAL",
                        "vitality": vitality,
                        "integrity": integrity,
                        "reason": "Affective/Homeostatic collapse imminent"
                    }
                    logger.warning("🚨 [ESC] Homeostatic collapse detected in SubsystemAudit (Vitality: %.2f)", vitality)
        except Exception as e:
            record_degradation('subsystem_audit', e)
            logger.debug("Affective escalation check failed: %s", e)

        return {"all_ok": all_ok, "subsystems": report, "checked_at": now}
    
    def should_emit_pulse(self) -> bool:
        """Check if it's time to emit a full health pulse."""
        return time.time() - self._last_health_pulse > self._health_pulse_interval
    
    def emit_pulse(self) -> str:
        """Generate a human-readable health pulse for the Neural Feed."""
        self._last_health_pulse = time.time()
        health = self.check_health()
        uptime = time.time() - self._start_time
        
        # System Metrics
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        
        lines = ["═══ UNIFIED HEALTH PULSE ═══"]
        lines.append(f"System: CPU {cpu}% | RAM {mem}% | Uptime: {int(uptime)}s")
        
        active_count = 0
        stale_count = 0
        missing_count = 0
        
        for name, info in health["subsystems"].items():
            status = info.get("status", "UNKNOWN")
            if status == "ACTIVE":
                active_count += 1
            elif status == "STALE":
                stale_count += 1
                lines.append(f"  ⚠️ {name}: STALE ({info['stale_seconds']}s)")
            else:
                missing_count += 1
                lines.append(f"  ❌ {name}: NEVER SEEN")
        
        summary = f"Total: {active_count}/{len(self.SUBSYSTEMS)} Subsystems Active"
        if stale_count:
            summary += f" | ⚠️ {stale_count} STALE"
        if missing_count:
            summary += f" | ❌ {missing_count} MISSING"
        
        lines.insert(2, summary)
        lines.append("═══════════════════════════")
        
        return "\n".join(lines)