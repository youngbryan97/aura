"""core/adaptation/immune_system.py

Protected Enclaves & Cognitive Rollback.
Ensures core identity, kinship data, and lore bibles are immune to memory decay.
Proactively scans for silent errors, dormant services, and broken interfaces.
"""
import asyncio
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from core.runtime.artifact_integrity import verify_artifact
from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service

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
        self._last_scan_results: dict[str, Any] = {}

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

    async def scan_system_health(self) -> dict[str, Any]:
        """Proactively scan all registered services for silent failures.
        
        Detects:
        - Required services that resolve to None
        - Services with broken interfaces (missing expected methods)
        - Unregistered critical services
        
        Returns a health report dict with 'healthy', 'degraded', and 'failed' lists.
        """
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
                    instance = get_runtime_service(service_name, default=None)
                    if instance is None:
                        entry = {
                            "service": service_name,
                            "tier": tier,
                            "issue": "not_registered_or_none",
                        }
                        if tier == "critical":
                            report["failed"].append(entry)
                            logger.warning(
                                "IMMUNE: CRITICAL service '%s' is missing or None", service_name
                            )
                        else:
                            report["degraded"].append(entry)
                            logger.info(
                                "IMMUNE: Service '%s' is not available (tier=%s)", service_name, tier
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
                            "IMMUNE: Service '%s' (%s) missing methods: %s",
                            service_name,
                            type(instance).__name__,
                            missing_methods,
                        )
                    else:
                        report["healthy"].append(service_name)
                except (ImportError, AttributeError, RuntimeError) as e:
                    record_degradation('immune_system', e)
                    report["failed"].append({
                        "service": service_name,
                        "tier": tier,
                        "issue": "resolution_error",
                        "error": str(e),
                    })
                    logger.error(
                        "IMMUNE: Service '%s' resolution raised: %s", service_name, e
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
                        "message": f"{len(report['failed'])} critical service(s) failed, "
                                   f"{len(report['degraded'])} degraded",
                    },
                )
        except (ImportError, AttributeError, RuntimeError) as _e:
            record_degradation('immune_system', _e)
            logger.error("IMMUNE: Failed to publish health report to event bus: %s", _e)

        summary = (
            f"Immune Scan: {len(report['healthy'])} healthy, "
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
                "IMMUNE SYSTEM: %d critical service(s) FAILED after boot: %s",
                len(report["failed"]),
                [f["service"] for f in report["failed"]],
            )
        
        if report["degraded"]:
            logger.warning(
                "IMMUNE SYSTEM: %d service(s) degraded after boot: %s",
                len(report["degraded"]),
                [d["service"] for d in report["degraded"]],
            )
        
        return report

    def get_last_scan(self) -> dict[str, Any]:
        """Return the most recent scan results."""
        return self._last_scan_results

    async def initiate_rollback(self, snapshot_path: str) -> bool:
        """Emergency restoration of core files if self-architecture fails.

        This method OVERWRITES EXECUTABLE CORE CODE, so it is gated three ways
        before a byte is copied: containment, integrity, and authority. Returns
        whether the rollback was performed — it used to return None on every
        path, so a caller could not distinguish "restored" from "refused".
        """
        self.rollback_active = True

        try:
            # Path resolution and stat are blocking syscalls; this runs on the
            # event loop, so they are offloaded like every other filesystem
            # touch in this method.
            # CONTAINMENT + INTEGRITY, via the shared gate in
            # core/runtime/artifact_integrity.py. This logic was written here
            # first; it now lives in one place so every path that promotes a
            # file into executable code gets the same checks instead of
            # reinventing weaker ones.
            verdict = await asyncio.to_thread(
                verify_artifact, snapshot_path, within=self.data_dir
            )
            if not verdict.ok:
                logger.error(
                    "Rollback refused: %s (%s)", verdict.failure, verdict.detail
                )
                return False
            snapshot = verdict.path

            # AUTHORITY. Overwriting core code is the most consequential act
            # this module can take and it had no governance decision at all.
            if not await self._rollback_authorized(snapshot):
                logger.error("Rollback refused: no governing approval for %s.",
                             snapshot)
                return False

            logger.warning("CRITICAL FAILURE: Rolling back to %s", snapshot)
            target = await asyncio.to_thread(
                lambda: Path("core/cognition/cognitive_kernel.py").resolve()
            )
            # Keep what we are about to destroy: an emergency restore that
            # cannot itself be undone is a one-way door.
            if await asyncio.to_thread(target.exists):
                undo = target.with_suffix(f".pre_rollback.{int(time.time())}.py")
                await asyncio.to_thread(shutil.copy2, target, undo)
                logger.info("Saved pre-rollback copy: %s", undo)
            await asyncio.to_thread(shutil.copy2, snapshot, target)
            logger.info("Rollback complete: %s restored.", target)
            return True
        except OSError as e:
            record_degradation('immune_system', e)
            logger.error("Rollback error: %s", e)
            return False
        finally:
            self.rollback_active = False

    async def _rollback_authorized(self, snapshot: Path) -> bool:
        """Ask the Will before overwriting executable core code.

        Fails CLOSED: if governance cannot be consulted, the rollback does not
        proceed. An emergency is not authority.
        """
        try:
            from core.runtime.action_executor import ActionExecutor
            from core.will import ActionDomain

            admission = ActionExecutor.authorize_action(
                action_name="immune_system.core_code_rollback",
                params={
                    "snapshot": str(snapshot),
                    "target": "core/cognition/cognitive_kernel.py",
                },
                source="immune_system",
                domain=ActionDomain.SELF_MODIFICATION,
                priority=0.95,
                context={
                    "operation": "core_code_rollback",
                    "snapshot": str(snapshot),
                    "target": "core/cognition/cognitive_kernel.py",
                },
            )
            return admission.approved
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                'immune_system', exc, severity="warning",
                action="refused core-code rollback because governance was unreachable",
            )
            return False

# Singleton support
_instance = None

def get_immune_system():
    global _instance
    if _instance is None:
        _instance = ImmuneSystem()
    return _instance
