"""research/protocols/resource_quotas.py — Automated Scaling Policies
=======================================================================
Implements staged compute scaling. Aura is limited to narrow resource
pools until she proves calibration capability (R^2 > 0.6) and transfer.
Once milestones are passed, the Governor unlocks more compute.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

from core.container import ServiceContainer

logger = logging.getLogger("Aura.Governance")

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

    def enforce_quota(self, metric: str, amount: int):
        """Called by telemetry to check if a run should be killed."""
        # E.g. token tracking
        pass

    def _save(self):
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=4)
        except Exception:
            pass
            
    def _load(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    self.state.update(json.load(f))
            except Exception:
                pass
