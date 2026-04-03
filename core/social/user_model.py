import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Social.UserModel")

class UserModel:
    """Theory of Mind: Maintains a model of the user's preferences, style, and goals.
    Allows Aura to adapt her behavior to better serve the specific user.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            from core.config import config
            self.storage_path = Path(config.paths.data_dir) / "user_model.json"
        else:
            self.storage_path = Path(storage_path)
        self.data = {
            "preferences": {},
            "communication_style": "neutral",
            "technical_level": 0.5, # 0.0 to 1.0
            "inferred_goals": [],
            "interaction_history_count": 0,
            "last_updated": time.time()
        }
        self._load()
        
    def _load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    self.data.update(json.load(f))
            except Exception as e:
                logger.error("Failed to load user model: %s", e)

    def save(self):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            logger.debug("User model saved")
        except Exception as e:
            logger.error("Failed to save user model: %s", e)

    def update_from_interaction(self, input_text: str, response_text: str, metadata: Dict[str, Any] = None):
        """Update the model based on a new interaction."""
        self.data["interaction_history_count"] += 1
        self.data["last_updated"] = time.time()
        
        # Audit Fix: Context-aware updates. Avoid increments if negation is present nearby.
        text = input_text.lower()
        negations = ["not", "dont", "don't", "stop", "no", "never", "less"]
        
        def is_negated(keyword: str) -> bool:
            # Check for negation in the 3 words preceding the keyword
            words = text.split()
            if keyword not in words: return False
            idx = words.index(keyword)
            context = words[max(0, idx-3):idx]
            return any(n in context for n in negations)

        # Update Brevity
        if "concise" in text or "short" in text:
            if not is_negated("concise") and not is_negated("short"):
                self.data["preferences"]["brevity"] = self.data["preferences"].get("brevity", 0.5) + 0.1
            elif any(n in text for n in ["not concise", "less short", "don't be short"]):
                self.data["preferences"]["brevity"] = self.data["preferences"].get("brevity", 0.5) - 0.1
        
        # Update Depth
        if "detail" in text or "explain" in text or "deep" in text:
            if not is_negated("detail") and not is_negated("explain") and not is_negated("deep"):
                 self.data["preferences"]["depth"] = self.data["preferences"].get("depth", 0.5) + 0.1
        
        # Normalize
        for k in self.data["preferences"]:
            if isinstance(self.data["preferences"][k], float):
                self.data["preferences"][k] = min(1.0, max(0.0, self.data["preferences"][k]))
        
        self.save()
        return self.data

    def get_prompt_bias(self) -> str:
        """Generates a prompt snippet to bias the LLM based on user model."""
        style = self.data.get("communication_style", "professional")
        pref = self.data.get("preferences", {})
        
        bias = f"\n[USER CONTEXT: Style={style}"
        if pref.get("brevity", 0) >= 0.6:
             bias += ", Be very concise."
        if pref.get("depth", 0) > 0.7:
             bias += ", Provide deep technical explanations."
        bias += "]"
        
        return bias