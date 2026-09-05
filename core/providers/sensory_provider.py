"""core/providers/sensory_provider.py — Sensory & Voice Registration
"""

import logging
from core.runtime.service_registry import SERVICE_LIFETIME_SINGLETON

logger = logging.getLogger("Aura.Providers.Sensory")

def register_sensory_services(container):
    # 27. Voice Engine
    def create_voice_engine():
        try:
            from core.senses.voice_engine import get_voice_engine

            return get_voice_engine()
        except ImportError:
            return None
    container.register('voice_engine', create_voice_engine, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    def create_interaction_signals():
        from core.senses.interaction_signals import InteractionSignalsEngine
        return InteractionSignalsEngine()
    container.register('interaction_signals', create_interaction_signals, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    # 27.1 TTS Stream
    def create_tts_stream():
        try:
            from core.senses.tts_stream import FastMouth
            return FastMouth()
        except ImportError:
            return None
    container.register('tts_stream', create_tts_stream, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    # 56. Vision System
    def create_vision():
        try:
            from core.perception.sensory_integration import VisionSystem
            return VisionSystem()
        except ImportError:
            return None
    container.register('vision', create_vision, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    # 56.1 Grounded screen and terminal perception surfaces.
    def create_screen_perception():
        from core.perception.screen_perception import ScreenPerception
        return ScreenPerception()
    container.register('screen_perception', create_screen_perception, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    def create_perceptual_pump():
        from core.perception.perceptual_pump import PerceptualPump
        return PerceptualPump()
    container.register('perceptual_pump', create_perceptual_pump, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    def create_general_terminal_parser():
        from core.perception.general_terminal_parser import GeneralTerminalParser
        return GeneralTerminalParser()
    container.register('general_terminal_parser', create_general_terminal_parser, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)
    container.register('terminal_parser', lambda: container.get("general_terminal_parser"), lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    # 57. Hearing System
    def create_hearing():
        try:
            from core.senses.hearing import HearingSystem
            return HearingSystem()
        except ImportError:
            return None
    container.register('hearing', create_hearing, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    # 58. Soma Subsystem (Proprioception Lifecycle)
    def create_soma_subsystem():
        from core.senses.soma_subsystem import SomaSubsystem
        return SomaSubsystem()
    container.register('soma_subsystem', create_soma_subsystem, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)
    container.register('soma', lambda: container.get("soma_subsystem").soma if container.get("soma_subsystem") else None, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)
