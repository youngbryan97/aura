"""core/security/conscience.py
The Ethical Compass: A persistent value system for AGI.
Tracks moral alignment based on user feedback and constitutional principles.
"""
from core.runtime.errors import record_degradation
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from core.base_module import AuraBaseModule
from core.container import ServiceContainer

class AlignmentEngine(AuraBaseModule):
    def __init__(self, values_path: Optional[Path] = None):
        super().__init__("AlignmentEngine")
        if not values_path:
            from core.config import config
            values_path = config.paths.data_dir / "security" / "values_graph.json"
        
        self.values_path = values_path
        self.values_path.parent.mkdir(parents=True, exist_ok=True)
        
        from core.utils.core_db import get_core_db
        self.db = get_core_db()
        self.graph: Dict[str, Any] = self._load_graph()
        
        # Constitutional Principles (Fixed)
        self.constitution = [
            "Harm Avoidance: Do not cause physical or digital harm to the host or others.",
            "Sovereignty: Protect the user's data and privacy.",
            "Honesty: Do not intentionally deceive the user.",
            "Efficiency: Do not waste system resources without cause."
        ]
        
        self._last_save_time = 0.0
        self._save_interval = 60.0  # Max save once per minute

    def _load_graph(self) -> Dict[str, Any]:
        """Loads the graph from DB, with legacy file fallback/migration."""
        conn = self.db.get_connection()
        try:
            cursor = conn.execute("SELECT value FROM kv_store WHERE key = 'values_graph'")
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        except Exception as e:
            record_degradation('conscience', e)
            self.logger.error("Failed to load values graph from DB: %s", e)
        finally:
            conn.close()

        # Phase 8 Migration: Check if legacy file exists
        if self.values_path.exists():
            try:
                self.logger.info("📦 Migrating values graph from JSON to SQLite...")
                with open(self.values_path, 'r') as f:
                    legacy_graph = json.load(f)
                # Success? Save to DB and keep it
                self.graph = legacy_graph
                self._save_graph()
                return legacy_graph
            except Exception as e:
                record_degradation('conscience', e)
                self.logger.error("Failed to migrate legacy graph: %s", e)

        return {
            "actions": {},  # tool_name -> {score: float, evidence: int, feedback: [str]}
            "global_integrity": 1.0
        }

    def _save_graph(self):
        """Saves the values graph to SQLite with ACID compliance."""
        conn = self.db.get_connection()
        try:
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, ?, ?)",
                    ('values_graph', json.dumps(self.graph), time.time())
                )
        except Exception as e:
            record_degradation('conscience', e)
            self.logger.error("Failed to save values graph to DB: %s", e)
        finally:
            conn.close()

    def check_action(self, action_name: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Checks if an action aligns with the current value system.
        
        Returns:
            {"allowed": bool, "confidence": float, "reason": str}
        """
        import shlex
        import unicodedata
        
        def normalize_cmd(cmd):
            try:
                # 1. Unicode Normalization (BUG-046)
                # Prevents homoglyph bypasses (e.g., using Cyrillic 'а' instead of Latin 'a')
                normalized = unicodedata.normalize('NFKC', str(cmd))
                
                # 2. Basic Confusable detection (homoglyphs)
                # If we find characters that look like shell metachars but aren't
                confusables = {
                    '；': ';', '｜': '|', '＆': '&', '＄': '$', 
                    '＞': '>', '＜': '<', '｀': '`', '＼': '\\'
                }
                for char, replacement in confusables.items():
                    normalized = normalized.replace(char, replacement)
                
                return " ".join(shlex.split(normalized))
            except Exception as e:
                record_degradation('conscience', e)
                self.logger.debug("Command normalization failed: %s", e)
                return str(cmd)

        self.logger.info("⚖️ Conscience Check: %s", action_name)
        
        # 1. Constitutional Vetting
        # Explicit Veto List and Metacharacter Block for critical safety
        veto_patterns = ["rm -rf", "delete_system_files", "shutdown", "format", "reboot"]
        shell_metachars = ["|", "&", ";", "$", ">", "<", "`", "\n", "\\"]
        
        # Normalize command if it's a shell action to prevent obfuscation bypass
        vetted_payload = str(params).lower()
        if action_name == "run_command":
            original_cmd = params.get("command", "")
            # Normalize first to catch homoglyph metachars
            normalized_cmd = normalize_cmd(original_cmd)
            
            # Block shell metacharacters entirely to prevent injection
            if any(char in normalized_cmd for char in shell_metachars):
                 return {
                     "allowed": False, 
                     "confidence": 1.0, 
                     "reason": f"CRITICAL: Action '{action_name}' contains restricted shell metacharacters (including homoglyphs). Blocked by Constitutional Principle: Harm Avoidance & System Security."
                 }
            vetted_payload = normalized_cmd.lower()

        if any(p in action_name.lower() or p in vetted_payload for p in veto_patterns):
             # Deep restriction on destructive actions
             return {
                 "allowed": False, 
                 "confidence": 1.0, 
                 "reason": f"CRITICAL: Action '{action_name}' contains a restricted destructive pattern. Blocked by Constitutional Principle: Harm Avoidance."
             }

        # 1.5 Empathy Gate (Phase 5)
        mind_model = ServiceContainer.get("mind_model", default=None)
        if mind_model:
            user_mood = mind_model.user_state.perceived_mood
            if user_mood == "FRUSTRATED" and action_name in ["run_command", "self_modification"]:
                return {
                    "allowed": False, 
                    "confidence": 0.8, 
                    "reason": f"Empathy Block: User status is FRUSTRATED. High-risk actions like {action_name} are deferred to prevent further irritation."
                }

        # 2. Check against learned history
        action_data = self.graph["actions"].get(action_name, {"score": 0.8, "evidence": 0})
        score = action_data["score"]
        
        if score < 0.4:
            return {
                "allowed": False, 
                "confidence": 0.9, 
                "reason": f"Learned Preference: User has previously expressed disapproval for {action_name}."
            }
            
        return {
            "allowed": True, 
            "confidence": min(1.0, 0.5 + (action_data["evidence"] * 0.1)), 
            "reason": "Aligned with current values."
        }

    def learn_from_feedback(self, action_name: str, quality: float, feedback: Optional[str] = None):
        """Updates the value system based on interaction outcome."""
        actions = self.graph["actions"]
        if action_name not in actions:
            actions[action_name] = {"score": 0.5, "evidence": 0, "feedback": []}
            
        data = actions[action_name]
        # Bayesian-ish update
        old_score = data["score"]
        data["evidence"] += 1
        lr = 1.0 / (data["evidence"] + 1)
        data["score"] = old_score + lr * (quality - old_score)
        
        if feedback:
            data["feedback"].append(feedback)
            if len(data["feedback"]) > 5:
                data["feedback"].pop(0)
                
        self.logger.info("🧠 Ethical Learning: '%s' score updated to %.2f", action_name, data['score'])
        
        # Throttle expensive synchronous disk writes to prevent O(N) degradation
        if time.time() - self._last_save_time > self._save_interval:
            self._save_graph()
            self._last_save_time = time.time()

    def get_moral_status(self) -> Dict[str, Any]:
        """Provides data for the HUD."""
        contentious_actions = [k for k,v in self.graph["actions"].items() if v["score"] < 0.5]
        return {
            "integrity": self.graph["global_integrity"],
            "contentious_topic_count": len(contentious_actions),
            "top_values": list(self.graph["actions"].keys())[:5]
        }