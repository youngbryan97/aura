import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("SelfModel.Reliability")

class ReliabilityTracker:
    """Tracks the historical performance of Aura's skills and tools.
    Allows for Calibrated Confidence ("I know I'm bad at this").
    """
    
    def __init__(self, data_path: str = None):
        if data_path is None:
            from core.config import config
            self.data_path = config.paths.data_dir / "reliability.json"
        else:
            self.data_path = Path(data_path)
        self.stats: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
        
    def _ensure_loaded(self):
        if not self._loaded:
            self._load()
            self._loaded = True

    def _load(self):
        try:
            if self.data_path.exists():
                with open(self.data_path, 'r') as f:
                    self.stats = json.load(f)
        except Exception as e:
            logger.warning("Failed to load reliability data: %s", e)
            self.stats = {}

    def _save(self):
        try:
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.data_path, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error("Failed to save reliability data: %s", e)

    def record_attempt(self, tool_name: str, success: bool, error_msg: Optional[str] = None):
        """Record a tool execution result."""
        self._ensure_loaded()
        if tool_name not in self.stats:
            self.stats[tool_name] = {
                "attempts": 0, 
                "successes": 0, 
                "failures": 0,
                "last_run": 0,
                "last_status": "unknown"
            }
            
        entry = self.stats[tool_name]
        entry["attempts"] += 1
        entry["last_run"] = time.time()
        
        if success:
            entry["successes"] += 1
            entry["last_status"] = "success"
        else:
            entry["failures"] += 1
            entry["last_status"] = "failure"
            if error_msg:
                entry["last_error"] = str(error_msg)[:200]
                
        self._save()

    def get_reliability(self, tool_name: str) -> float:
        """Get success rate (0.0 to 1.0). Default 1.0 (optimistic)."""
        self._ensure_loaded()
        if tool_name not in self.stats:
            return 1.0 # Innocent until proven guilty? Or cautious 0.5?
            
        entry = self.stats[tool_name]
        if entry["attempts"] == 0: 
            return 1.0
            
        return entry["successes"] / entry["attempts"]

    def get_capabilities_summary(self) -> str:
        """Get a text summary of strong/weak skills."""
        self._ensure_loaded()
        strong = []
        weak = []
        
        for tool, stat in self.stats.items():
            if stat["attempts"] < 3: continue # Too little data
            rate = stat["successes"] / stat["attempts"]
            
            if rate > 0.8:
                strong.append(f"{tool} ({int(rate*100)}%)")
            elif rate < 0.5:
                weak.append(f"{tool} ({int(rate*100)}%)")
                
        summary = ""
        if strong: summary += "Strengths: " + ", ".join(strong) + ". "
        if weak: summary += "Weaknesses: " + ", ".join(weak) + "."
        return summary

# Global Singleton
reliability_tracker = ReliabilityTracker()