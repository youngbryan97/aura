"""core/providers/sensory_provider.py — Sensory & Voice Registration
"""

import logging
from core.container import ServiceLifetime

logger = logging.getLogger("Aura.Providers.Sensory")

def register_sensory_services(container):
    # 27. Voice Engine
    def create_voice_engine():
        try:
            from core.senses.voice_engine import SovereignVoiceEngine
            return SovereignVoiceEngine()
        except ImportError:
            return None
    container.register('voice_engine', create_voice_engine, lifetime=ServiceLifetime.SINGLETON, required=False)

    def create_interaction_signals():
        from core.senses.interaction_signals import InteractionSignalsEngine
        return InteractionSignalsEngine()
    container.register('interaction_signals', create_interaction_signals, lifetime=ServiceLifetime.SINGLETON, required=False)

    # 27.1 TTS Stream
    def create_tts_stream():
        try:
            from core.senses.tts_stream import FastMouth
            return FastMouth()
        except ImportError:
            return None
    container.register('tts_stream', create_tts_stream, lifetime=ServiceLifetime.SINGLETON, required=False)

    # 56. Vision System
    def create_vision():
        try:
            from core.senses.vision import VisionSystem
            return VisionSystem()
        except ImportError:
            return None
    container.register('vision', create_vision, lifetime=ServiceLifetime.SINGLETON, required=False)

    # 57. Hearing System
    def create_hearing():
        try:
            from core.senses.hearing import HearingSystem
            return HearingSystem()
        except ImportError:
            return None
    container.register('hearing', create_hearing, lifetime=ServiceLifetime.SINGLETON, required=False)

    # 58. Soma Subsystem (Proprioception Lifecycle)
    def create_soma_subsystem():
        from core.senses.soma_subsystem import SomaSubsystem
        return SomaSubsystem()
    container.register('soma_subsystem', create_soma_subsystem, lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('soma', lambda: container.get("soma_subsystem").soma if container.get("soma_subsystem") else None, lifetime=ServiceLifetime.SINGLETON, required=False)
