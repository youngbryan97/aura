"""core/values/preference_conflict.py
Detects and resolves internal preference conflicts.
"""


class PreferenceConflictResolver:
    """Arbitrates conflicting preferences (e.g. speed vs accuracy)."""

    def resolve(self, preferences: dict[str, float]) -> dict[str, float]:
        """Ensures conflicting preferences are balanced to prevent lockups."""
        resolved = preferences.copy()
        
        # Balance speed vs accuracy: if both are high, average them
        if resolved.get("speed", 0.0) > 0.8 and resolved.get("accuracy", 0.0) > 0.8:
            resolved["speed"] = 0.7
            resolved["accuracy"] = 0.7
            
        return resolved
