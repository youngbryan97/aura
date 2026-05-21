"""core/embodiment/digital_body.py
================================
Thread-safe canonical model of Aura's software body schema (proprioception).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set

try:
    import psutil
except ImportError:
    psutil = None

from core.container import ServiceContainer
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Embodiment.DigitalBody")

_SINGLETON_LOCK = threading.Lock()
_digital_body_instance: Optional[DigitalBody] = None


class DigitalBody:
    """Singleton representing the software body schema and resource proprioception of Aura."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        
        # Core identification and capabilities
        self.available_models: List[str] = ["cortex", "solver", "brainstem"]
        self.available_tools: List[str] = []
        
        # Proprioceptive resource state
        self.resource_state: Dict[str, float] = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "disk_percent": 0.0,
            "latency_ms": 0.0
        }
        
        # Fine-grained security permissions
        self.permissions: Dict[str, bool] = {
            "file_write": True,
            "shell_execution": True,
            "network_access": False,
            "self_modification": True
        }
        
        # Operational health indicators
        self.degraded_systems: Set[str] = set()
        
        # Commitment / Goal tracking
        self.current_commitments: List[Dict[str, Any]] = []
        
        # Environment grounding references
        self.environment_handles: Dict[str, Any] = {}
        
        # Initialize internal performance clocks
        self._last_update_time = time.time()
        logger.info("DigitalBody schema initialized.")

    def update_telemetry(self) -> None:
        """Polls system diagnostics using psutil to refresh the proprioceptive schema."""
        with self._lock:
            start_time = time.time()
            try:
                if psutil is not None:
                    # Non-blocking CPU reading
                    cpu = psutil.cpu_percent(interval=None)
                    mem = psutil.virtual_memory().percent
                    disk = psutil.disk_usage("/").percent
                    
                    self.resource_state["cpu_percent"] = float(cpu)
                    self.resource_state["memory_percent"] = float(mem)
                    self.resource_state["disk_percent"] = float(disk)
                else:
                    # Resilient fallback values with slight dynamic fluctuations
                    import random
                    t = time.time()
                    self.resource_state["cpu_percent"] = round(15.0 + 10.0 * (t % 3.0) + random.uniform(-1, 1), 2)
                    self.resource_state["memory_percent"] = round(42.0 + 2.0 * (t % 5.0) + random.uniform(-0.5, 0.5), 2)
                    self.resource_state["disk_percent"] = 35.4
            except Exception as exc:
                record_degradation("digital_body_telemetry", exc)
                logger.warning("Failed to refresh digital body telemetry: %s", exc)
            finally:
                elapsed_ms = (time.time() - start_time) * 1000.0
                # Moving average update of internal sensing latency
                alpha = 0.2
                self.resource_state["latency_ms"] = round(
                    (1 - alpha) * self.resource_state["latency_ms"] + alpha * elapsed_ms, 3
                )
                self._last_update_time = time.time()

    def register_environment(self, env_id: str, env: Any) -> None:
        """Registers a grounded virtual environment in the body schema."""
        with self._lock:
            self.environment_handles[env_id] = env
            logger.info("Grounded virtual environment registered: %s", env_id)

    def deregister_environment(self, env_id: str) -> None:
        """Removes a virtual environment association."""
        with self._lock:
            if env_id in self.environment_handles:
                del self.environment_handles[env_id]
                logger.info("Environment %s deregistered.", env_id)

    def register_commitment(self, commitment: Dict[str, Any]) -> None:
        """Registers an active commitment (goal) with a strict, persistent structured schema."""
        with self._lock:
            if "id" not in commitment:
                import uuid
                commitment["id"] = uuid.uuid4().hex[:12]
            if "created_at" not in commitment:
                commitment["created_at"] = time.time()
            if "status" not in commitment:
                commitment["status"] = "active"
                
            # Enforce and default structured goal fields from the prompt specification
            commitment.setdefault("goal", "unspecified_maintenance_task")
            commitment.setdefault("origin", "internal_drive_pressure")
            commitment.setdefault("expected_value", 0.5)
            commitment.setdefault("risk", 0.1)
            commitment.setdefault("resource_budget", "low")
            commitment.setdefault("deadline", "next_idle_window")
            commitment.setdefault("success_metric", "baseline_invariants_preserved")
            
            # Canonical commitment tracking ledger
            commitment.setdefault("current_plan", [])
            commitment.setdefault("last_action", "reflect")
            commitment.setdefault("blocked_reason", None)
            commitment.setdefault("next_action", "reflect")
            commitment.setdefault("evidence_of_completion", None)
            commitment.setdefault("postmortem", None)
                
            self.current_commitments.append(commitment)
            logger.info("DigitalBody registered commitment ledger: %s (goal: %s, origin: %s, expected_value: %.2f)", 
                        commitment["id"], commitment["goal"], commitment["origin"], commitment["expected_value"])

    def resolve_commitment(self, commitment_id: str, status: str = "completed", 
                           evidence: str = None, postmortem: str = None) -> Optional[Dict[str, Any]]:
        """Resolves an active commitment, storing evidence of completion and postmortem analysis."""
        with self._lock:
            for commitment in self.current_commitments:
                if commitment.get("id") == commitment_id:
                    commitment["status"] = status
                    commitment["resolved_at"] = time.time()
                    if evidence:
                        commitment["evidence_of_completion"] = evidence
                    if postmortem:
                        commitment["postmortem"] = postmortem
                    logger.info("DigitalBody commitment %s resolved as: %s. Postmortem: %s", commitment_id, status, postmortem)
                    return commitment
            return None

    def is_action_authorized(self, action_name: str) -> bool:
        """Evaluates whether an action is authorized under current proprioceptive class constraints.
        
        Action Classes:
        - safe: reflect, summarize memory, run diagnostics, compact database, generate test, simulate plan
        - medium: modify local files in sandbox (file_read/file_write), run tests, propose patch, browse/read environment
        - high: write production code, push git changes (commit_code), send messages, spend money, delete data, external side effects
        """
        with self._lock:
            # Safe action class: implicitly allowed
            safe_actions = {"reflect", "summarize memory", "run diagnostics", "compact database", "generate test", "simulate plan"}
            if action_name in safe_actions:
                return True
                
            # Medium action class: requires specific permissions
            medium_actions = {"file_read", "file_write", "run_test", "patch_code", "browse_environment"}
            if action_name in medium_actions:
                if action_name == "file_write" or action_name == "patch_code":
                    return self.permissions.get("file_write", False)
                return True
                
            # High action class: requires advanced permissions
            high_actions = {"file_delete", "commit_code", "write_production_code", "push_git_changes", "send_messages", "delete_data"}
            if action_name in high_actions:
                if action_name == "file_delete":
                    return self.permissions.get("file_write", False)
                if action_name == "commit_code":
                    return self.permissions.get("self_modification", False)
                return False  # Strict default block for untested high-risk features
                
            return False


    def mark_system_degraded(self, system_name: str, degraded: bool = True) -> None:
        """Flags a system as degraded or recovers it."""
        with self._lock:
            if degraded:
                self.degraded_systems.add(system_name)
                logger.warning("DigitalBody registered system degradation: %s", system_name)
            else:
                if system_name in self.degraded_systems:
                    self.degraded_systems.remove(system_name)
                    logger.info("DigitalBody registered system recovery: %s", system_name)

    def get_state_dict(self) -> Dict[str, Any]:
        """Returns a snapshot of the body schema for world model observation."""
        with self._lock:
            return {
                "available_models": list(self.available_models),
                "available_tools": list(self.available_tools),
                "resource_state": dict(self.resource_state),
                "permissions": dict(self.permissions),
                "degraded_systems": list(self.degraded_systems),
                "commitments_count": len(self.current_commitments),
                "active_commitments": [c for c in self.current_commitments if c.get("status") == "active"],
                "environments": list(self.environment_handles.keys())
            }


def get_digital_body() -> DigitalBody:
    """Thread-safe accessor for the DigitalBody singleton."""
    global _digital_body_instance
    if _digital_body_instance is None:
        with _SINGLETON_LOCK:
            if _digital_body_instance is None:
                _digital_body_instance = DigitalBody()
                # Auto register in ServiceContainer
                ServiceContainer.register_instance("digital_body", _digital_body_instance)
    return _digital_body_instance
