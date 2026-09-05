"""core/social/user_preference_model.py
Tracks explicit and implicit user communication preferences.
"""
from typing import Any


class UserPreferenceModel:
    """Manages profile preferences (e.g. verbosity levels) for operators."""

    def __init__(self):
        self._profiles: dict[str, dict[str, Any]] = {
            "Bryan": {
                "verbosity": "concise",
                "tts_enabled": False,
                "allow_autonomous_execution": True
            }
        }

    def get_preference(self, person: str, key: str, default: Any = None) -> Any:
        return self._profiles.get(person, {}).get(key, default)

    def set_preference(self, person: str, key: str, value: Any) -> None:
        if person not in self._profiles:
            self._profiles[person] = {}
        self._profiles[person][key] = value
