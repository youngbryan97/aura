"""Legacy compatibility wrapper for the canonical core speak skill."""

from core.skills.speak import SpeakInput, SpeakSkill

# Preserve the historical class name used by older imports.
VoiceSkill = SpeakSkill

__all__ = ["SpeakInput", "SpeakSkill", "VoiceSkill"]
