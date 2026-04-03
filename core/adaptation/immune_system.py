"""core/adaptation/immune_system.py

Protected Enclaves & Cognitive Rollback.
Ensures core identity, kinship data, and lore bibles are immune to memory decay.
Proactively scans for silent errors, dormant services, and broken interfaces.
"""
import asyncio
import logging
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.ImmuneSystem")

# Services that MUST be functional for Aura to operate
CRITICAL_SERVICES = [
    ("llm_router", ["think", "generate"]),
    ("state_repository", ["get_current"]),
    ("event_bus", ["publish", "subscribe"]),
    ("affect_engine", ["decay_tick", "get_state_sync"]),
]

IMPORTANT_SERVICES = [
    ("personality_engine", ["get_personality_prompt"]),
    ("cognitive_integration", ["process_turn"]),
    ("voice_engine", ["synthesize_speech"]),
    ("continuity", ["save", "load"]),
    ("immune_system", ["is_protected"]),
    ("metrics", []),
    ("persistence", ["start_session"]),
    ("dlq", []),
    ("audit", ["record"]),
    ("self_model", []),
]


class ImmuneSystem:
    def __init__(self, data_dir: str = "data/backups"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # IDs that the IntegrityGuard is forbidden from touching
        self.enclaves = ["identity_pillar", "DOME_lore", "kinship", "family_recipe"]
        self.protected_tags = self.enclaves
        self.rollback_active = False
        self._last_scan_results: Dict[str, Any] = {}

    def is_protected(self, metadata: dict) -> bool:
        """Checks if a memory fragment or belief is in a protected enclave."""
        tags = metadata.get("tags", [])
        if any(tag in self.enclaves for tag in tags):
            return True
        return False

    async def verify_integrity(self, memory_fragment: dict) -> bool:
        """Checks if a memory is protected before allowing decay/deletion."""
        metadata = memory_fragment.get("metadata", {}) or memory_fragment
        return self.is_protected(metadata)

    async def scan_system_health(self) -> Dict[str, Any]:
        """Proactively scan all registered services for silent failures.
        
        Detects:
        - Required services that resolve to None
        - Services with broken interfaces (missing expected methods)
        - Unregistered critical services
        
        Returns a health report dict with 'healthy', 'degraded', and 'failed' lists.
        """
        from core.container import ServiceContainer

        report = {
            "timestamp": time.time(),
            "healthy": [],
            "degraded": [],
            "failed": [],
            "warnings": [],
        }

        all_checks = [
            (CRITICAL_SERVICES, "critical"),
            (IMPORTANT_SERVICES, "important"),
        ]

        for service_list, tier in all_checks:
            for service_name, expected_methods in service_list:
                try:
                    instance = ServiceContainer.get(service_name, default=None)
                    if instance is None:
                        entry = {
                            "service": service_name,
                            "tier": tier,
                            "issue": "not_registered_or_none",
                        }
                        if tier == "critical":
                            report["failed"].append(entry)
                            logger.warning(
                                "🛡️ IMMUNE: CRITICAL service '%s' is missing or None", service_name
                            )
                        else:
                            report["degraded"].append(entry)
                            logger.info(
                                "🛡️ IMMUNE: Service '%s' is not available (tier=%s)", service_name, tier
                            )
                        continue

                    # Check interface completeness
                    missing_methods = [
                        m for m in expected_methods if not hasattr(instance, m)
                    ]
                    if missing_methods:
                        entry = {
                            "service": service_name,
                            "tier": tier,
                            "issue": "broken_interface",
                            "missing": missing_methods,
                            "actual_type": type(instance).__name__,
                        }
                        report["degraded"].append(entry)
                        logger.warning(
                            "🛡️ IMMUNE: Service '%s' (%s) missing methods: %s",
                            service_name,
                            type(instance).__name__,
                            missing_methods,
                        )
                    else:
                        report["healthy"].append(service_name)
                except Exception as e:
                    report["failed"].append({
                        "service": service_name,
                        "tier": tier,
                        "issue": "resolution_error",
                        "error": str(e),
                    })
                    logger.error(
                        "🛡️ IMMUNE: Service '%s' resolution raised: %s", service_name, e
                    )

        self._last_scan_results = report
        
        # Surface to event bus if available
        try:
            from core.event_bus import get_event_bus
            bus = get_event_bus()
            if report["failed"]:
                bus.publish_threadsafe(
                    "immune_alert",
                    {
                        "type": "silent_failure_detected",
                        "failed_services": [f["service"] for f in report["failed"]],
                        "degraded_count": len(report["degraded"]),
                        "message": f"🛡️ {len(report['failed'])} critical service(s) failed, "
                                   f"{len(report['degraded'])} degraded",
                    },
                )
        except Exception as _e:
            logger.error("🛡️ IMMUNE: Failed to publish health report to event bus: %s", _e)

        summary = (
            f"🛡️ Immune Scan: {len(report['healthy'])} healthy, "
            f"{len(report['degraded'])} degraded, {len(report['failed'])} failed"
        )
        logger.info(summary)
        return report

    async def post_boot_scan(self, orchestrator=None):
        """Run after boot completes to surface any silent failures.
        Called by the boot sequence after all subsystems are initialized.
        """
        report = await self.scan_system_health()
        
        if report["failed"]:
            logger.critical(
                "🚨 IMMUNE SYSTEM: %d critical service(s) FAILED after boot: %s",
                len(report["failed"]),
                [f["service"] for f in report["failed"]],
            )
        
        if report["degraded"]:
            logger.warning(
                "⚠️ IMMUNE SYSTEM: %d service(s) degraded after boot: %s",
                len(report["degraded"]),
                [d["service"] for d in report["degraded"]],
            )
        
        return report

    def get_last_scan(self) -> Dict[str, Any]:
        """Return the most recent scan results."""
        return self._last_scan_results

    async def initiate_rollback(self, snapshot_path: str):
        """Emergency restoration of core files if self-architecture fails."""
        self.rollback_active = True
        
        try:
            # Audit Fix: Prevent path traversal by resolving absolute paths
            # and checking the common prefix.
            base_dir = self.data_dir.resolve()
            snapshot = Path(snapshot_path).resolve()
            
            if not str(snapshot).startswith(str(base_dir)):
                logger.error(f"🛑 Security violation: Rollback path traversal detected! {snapshot_path}")
                return

            if not snapshot.exists():
                logger.error(f"Rollback failed: Snapshot {snapshot_path} not found.")
                return

            logger.warning(f"🚨 CRITICAL FAILURE: Rolling back to {snapshot_path}")
            target = Path("core/cognitive_kernel.py")
            await asyncio.to_thread(shutil.copy2, snapshot, target)
            logger.info(f"✅ Rollback complete: {target} restored.")
        except Exception as e:
            logger.error(f"Rollback error: {e}")
        finally:
            self.rollback_active = False

# Singleton support
_instance = None

def get_immune_system():
    global _instance
    if _instance is None:
        _instance = ImmuneSystem()
    return _instance

