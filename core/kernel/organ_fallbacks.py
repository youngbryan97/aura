"""Boot-time fallback implementations for kernel organs.

When a real subsystem (LLM, vision, neural bridge, etc.) isn't available
yet — because the model is still loading, hardware isn't present, or a
dependency failed — the kernel uses these minimal fallbacks so the
cognitive tick can still complete.

These are NOT stubs. They're the offline tier of a graceful degradation
system. Once the real subsystem comes online, it replaces the fallback.
"""

from typing import Any


class FallbackLLM:
    """Returns a minimal response until the real model finishes loading."""

    async def think(self, prompt: str, **kwargs: Any) -> str:
        return "I'm Aura."

    async def classify(self, prompt: str) -> str:
        return "CHAT"


class FallbackVision:
    """No-op vision when no camera or vision model is available."""

    async def capture(self) -> None:
        return None

    async def capture_desktop(self) -> None:
        return None

    async def load(self) -> None:
        pass  # no-op: intentional


class FallbackNeural:
    """Lightweight stand-in when NeuralBridge can't initialize."""

    async def load(self) -> None:
        pass  # no-op: intentional

    def get_status(self) -> dict[str, bool]:
        return {"is_running": False, "lightweight_mode": True}


class FallbackVoice:
    """Silent voice when TTS engine isn't available."""

    async def speak(self, text: str) -> None:
        pass  # no-op: intentional

    async def say(self, text: str) -> None:
        pass  # no-op: intentional

    async def load(self) -> None:
        pass  # no-op: intentional


class FallbackOrgan:
    """Generic fallback for any organ that failed to load."""

    async def load(self) -> None:
        pass  # no-op: intentional
