"""core/memory/social_memory.py
Social Memory & Narrative Engine.
Tracks relationship milestones, shared context, and social depth.
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from core.base_module import AuraBaseModule

class RelationshipMilestone:
    def __init__(self, description: str, timestamp: Optional[float] = None, importance: float = 0.5):
        self.description = description
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.importance = importance

    def to_dict(self):
        return {
            "description": self.description,
            "timestamp": self.timestamp,
            "importance": self.importance
        }

class SocialMemory(AuraBaseModule):
    def __init__(self, data_path: Optional[Path] = None):
        super().__init__("SocialMemory")
        if not data_path:
            from core.config import config
            data_path = config.paths.data_dir / "memory" / "social_memory.json"
        
        self.data_path = data_path
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.milestones: List[RelationshipMilestone] = []
        self.relationship_depth = 0.0
        self.shared_context_keys: List[str] = []
        self._load()

    def _load(self):
        if self.data_path.exists():
            try:
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                    self.milestones = [RelationshipMilestone(**m) for m in data.get("milestones", []) if isinstance(m, dict)]
                    self.relationship_depth = data.get("depth", 0.0)
                    self.shared_context_keys = data.get("shared_keys", [])
            except Exception as e:
                if self.logger:
                    self.logger.error("Failed to load social memory: %s", e)

    def save(self):
        try:
            import os
            tmp_path = str(self.data_path) + ".tmp"
            with open(tmp_path, 'w') as f:
                json.dump({
                    "milestones": [m.to_dict() for m in self.milestones],
                    "depth": self.relationship_depth,
                    "shared_keys": self.shared_context_keys
                }, f, indent=2)
            os.replace(tmp_path, self.data_path)
        except Exception as e:
            self.logger.error("Failed to save social memory: %s", e)

    def record_milestone(self, description: str, importance: float = 0.5):
        """Adds a new milestone and increases relationship depth."""
        milestone = RelationshipMilestone(description, importance=importance)
        self.milestones.append(milestone)
        self.relationship_depth = min(1.0, self.relationship_depth + (importance * 0.05))
        self.logger.info("💌 Social Milestone Recorded: %s", description)
        self.save()

    def add_shared_context(self, key: str):
        if key not in self.shared_context_keys:
            self.shared_context_keys.append(key)
            self.relationship_depth = min(1.0, self.relationship_depth + 0.01)
            self.save()

    def get_social_context(self) -> str:
        """Returns a string representing the social relationship for the LLM."""
        depth_label = "Acquaintance"
        if self.relationship_depth > 0.8: depth_label = "Peak Synchrony"
        elif self.relationship_depth > 0.6: depth_label = "Deep Collaborator"
        elif self.relationship_depth > 0.4: depth_label = "Trusted Peer"
        elif self.relationship_depth > 0.2: depth_label = "Friendly Explorer"
        
        m_list = [m.description for m in self.milestones[-3:]]
        milestones_str = ", ".join(m_list) if m_list else "None yet"
        
        return (
            f"[SOCIAL RELATIONSHIP]\n"
            f"- Status: {depth_label}\n"
            f"- Depth: {self.relationship_depth:.2f}\n"
            f"- Recent Milestones: {milestones_str}\n"
            f"- Shared Knowledge Keys: {len(self.shared_context_keys)}"
        )