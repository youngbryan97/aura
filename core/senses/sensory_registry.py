from dataclasses import dataclass

@dataclass
class SensoryCapabilityFlags:
    """
    Set at boot time based on what is actually available in the environment.
    Prevents the cognitive core from crashing when hardware or libs are missing.
    """
    hearing_enabled: bool = False   # SovereignEars / STT
    speech_enabled: bool = False    # VoiceEngine / TTS
    vision_enabled: bool = False    # cv2 / screen capture
    metrics_enabled: bool = False   # prometheus_client

    @classmethod
    def from_boot_status(cls, status: dict[str, bool]) -> "SensoryCapabilityFlags":
        """
        Maps module availability to capability flags.
        """
        return cls(
            hearing_enabled = status.get("speech_recognition", False) or status.get("sounddevice", False),
            speech_enabled  = (
                status.get("pyttsx3", False)
                or status.get("tts", False)
                or status.get("TTS", False)
                or status.get("voice_engine", False)
            ),
            vision_enabled  = (status.get("cv2", False) and status.get("mss", False)) or status.get("opencv-python", False),
            metrics_enabled = status.get("prometheus_client", False),
        )

    @classmethod
    def get_current(cls) -> "SensoryCapabilityFlags":
        """Legacy-compatible accessor for global capabilities."""
        return get_capabilities()

# Global registry instance
_capabilities = SensoryCapabilityFlags()

def get_capabilities() -> SensoryCapabilityFlags:
    return _capabilities

def set_capabilities(flags: SensoryCapabilityFlags):
    global _capabilities
    _capabilities = flags
