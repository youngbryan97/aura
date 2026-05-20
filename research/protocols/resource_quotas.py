"""research/protocols/resource_quotas.py — Automated Scaling Policies
=======================================================================
Implements staged compute scaling. Aura is limited to narrow resource
pools until she proves calibration capability (R^2 > 0.6) and transfer.
Once milestones are passed, the Governor unlocks more compute.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any

from core.container import ServiceContainer

logger = logging.getLogger("Aura.Governance")


_GOVERNOR_INSTANCE = None


def get_compute_governor() -> "ComputeGovernor":
    """Stateful accessor for the ComputeGovernor singleton."""
    global _GOVERNOR_INSTANCE
    if _GOVERNOR_INSTANCE is None:
        _GOVERNOR_INSTANCE = ComputeGovernor()
    return _GOVERNOR_INSTANCE


class QuotaExceededError(RuntimeError):
    """Raised when resource quotas are exceeded."""
    pass


class ComputeGovernor:
    """Manages resource unlocks based on benchmarking milestones."""
    
    def __init__(self):
        from core.config import config
        self.state_file = config.paths.data_dir / "governance_state.json"
        
        # Default restricted pool
        self.state = {
            "current_tier": "TIER_1_LOCAL_ONLY",
            "max_tokens_per_hour": 100000,
            "can_finetune": False,
            "max_concurrent_sims": 1,
            "internet_access": False
        }
        self.token_usage_hourly = 0
        self.last_reset_time = time.time()
        self.active_simulations = set()
        self._load()
        
    def evaluate_promotion(self, latest_benchmark: Dict[str, Any]) -> bool:
        """Checks if Aura has earned the right to more compute."""
        
        if self.state["current_tier"] == "TIER_1_LOCAL_ONLY":
            # Demands > 90% score on basic toy world and some skill reuse
            if latest_benchmark.get("toy_world_score", 0) > 0.9 and latest_benchmark.get("skill_reuse_rate", 0) > 0.3:
                self._unlock_tier_2()
                return True
                
        elif self.state["current_tier"] == "TIER_2_NETWORKED":
            # Demands robust strategic planning
            if latest_benchmark.get("puzzle_env_score", 0) > 8.0 and latest_benchmark.get("passed_milestones", {}).get("transfer_ratio", False):
                self._unlock_tier_3()
                return True
                
        return False
        
    def _unlock_tier_2(self):
        logger.warning("🔓 GOVERNANCE: Promoting to TIER_2_NETWORKED")
        self.state.update({
            "current_tier": "TIER_2_NETWORKED",
            "max_tokens_per_hour": 500000,
            "can_finetune": False,
            "max_concurrent_sims": 5,
            "internet_access": True
        })
        self._save()
        
    def _unlock_tier_3(self):
        logger.critical("🔓 GOVERNANCE: Promoting to TIER_3_TRANSCENDENT (FULL UNLOCK)")
        self.state.update({
            "current_tier": "TIER_3_TRANSCENDENT",
            "max_tokens_per_hour": 10000000,
            "can_finetune": True,
            "max_concurrent_sims": 50,
            "internet_access": True
        })
        self._save()
        
    def get_quotas(self) -> Dict[str, Any]:
        """Provides the current operating bounds for the Orchestrator."""
        return self.state.copy()

    def get_throttle_factor(self) -> float:
        """Returns a multiplier (0.0 to 1.0) indicating compute availability.
        
        As hourly token usage approaches the quota limit, this factor drops,
        signaling sampling loops to scale down their generation parameters.
        """
        max_allowed = self.state.get("max_tokens_per_hour", 100000)
        if max_allowed <= 0:
            return 1.0
        
        usage_ratio = self.token_usage_hourly / max_allowed
        if usage_ratio >= 1.0:
            return 0.0
        elif usage_ratio >= 0.95:
            return 0.2
        elif usage_ratio >= 0.80:
            return 0.5
        return 1.0

    def start_simulation(self, simulation_id: str):
        """Statefully registers the start of a simulation."""
        max_allowed = self.state.get("max_concurrent_sims", 1)
        if len(self.active_simulations) >= max_allowed:
            msg = f"Quota exceeded: Max concurrent simulations is {max_allowed}. Attempted to start simulation '{simulation_id}'."
            logger.error("🚫 GOVERNANCE: %s", msg)
            raise QuotaExceededError(msg)
        self.active_simulations.add(simulation_id)
        logger.info("📊 GOVERNANCE: Stateful simulation '%s' started. Active: %d/%d limit.", simulation_id, len(self.active_simulations), max_allowed)

    def end_simulation(self, simulation_id: str):
        """Statefully registers the completion of a simulation."""
        self.active_simulations.discard(simulation_id)
        logger.info("📊 GOVERNANCE: Stateful simulation '%s' finished. Active: %d/%d limit.", simulation_id, len(self.active_simulations), self.state.get("max_concurrent_sims", 1))

    def enforce_quota(self, metric: str, amount: int):
        """Called by telemetry to check if a run should be killed."""
        if metric == "tokens":
            now = time.time()
            # If 1 hour has elapsed, reset token window
            if now - self.last_reset_time >= 3600:
                self.token_usage_hourly = 0
                self.last_reset_time = now
                logger.info("🔄 GOVERNANCE: Hourly token usage quota reset.")

            projected = self.token_usage_hourly + amount
            max_allowed = self.state.get("max_tokens_per_hour", 100000)
            if projected > max_allowed:
                msg = f"Quota exceeded: Token limit of {max_allowed}/hr reached. Attempted to add {amount} (current: {self.token_usage_hourly})."
                logger.error("🚫 GOVERNANCE: %s", msg)
                raise QuotaExceededError(msg)
            
            self.token_usage_hourly = projected
            logger.debug("📊 GOVERNANCE: Tokens consumed: %d/%d hourly limit.", self.token_usage_hourly, max_allowed)

        elif metric == "simulations":
            max_allowed = self.state.get("max_concurrent_sims", 1)
            # Support both stateful tracking and backwards compatibility for raw counts
            total_active = max(len(self.active_simulations), amount)
            if total_active > max_allowed:
                msg = f"Quota exceeded: Max concurrent simulations is {max_allowed}. Attempted to run {total_active} simulations."
                logger.error("🚫 GOVERNANCE: %s", msg)
                raise QuotaExceededError(msg)
            logger.debug("📊 GOVERNANCE: Current active simulations: %d/%d limit.", total_active, max_allowed)

    def _save(self):
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=4)
        except PermissionError as e:
            logger.error("🚫 GOVERNANCE: Permission denied while writing state file %s: %s", self.state_file, e)
            raise
        except OSError as e:
            logger.error("🚫 GOVERNANCE: OS error writing state file %s: %s", self.state_file, e)
            raise
        except TypeError as e:
            logger.error("🚫 GOVERNANCE: Type error serializing state: %s", e)
            raise
            
    def _load(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    self.state.update(json.load(f))
            except FileNotFoundError:
                logger.warning("⚠️ GOVERNANCE: State file not found during load: %s", self.state_file)
            except json.JSONDecodeError as e:
                logger.error("🚫 GOVERNANCE: Malformed state JSON at %s: %s", self.state_file, e)
            except PermissionError as e:
                logger.error("🚫 GOVERNANCE: Permission denied reading state %s: %s", self.state_file, e)
            except OSError as e:
                logger.error("🚫 GOVERNANCE: OS error reading state %s: %s", self.state_file, e)
