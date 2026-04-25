"""Aura PerceptionRuntime — governed multimodal sensors.

The audit calls for camera/audio/subtitle ingestion plus a
shared-attention / silence-policy / movie-session-memory stack, all
mediated by capability tokens issued by Unified Will.

This package exposes the *contract* surface only. Real hardware drivers
are intentionally out of scope here so the runtime can boot without a
camera/microphone present; tests assert that the contract holds and that
sensor enablement is governed.
"""

from .perception_runtime import (
    CapabilityToken,
    MovieSessionMemory,
    PerceptionRuntime,
    SharedAttentionState,
    SilencePolicy,
    SceneEvent,
)

__all__ = [
    "CapabilityToken",
    "MovieSessionMemory",
    "PerceptionRuntime",
    "SharedAttentionState",
    "SilencePolicy",
    "SceneEvent",
]
