from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
from typing import Optional, AsyncGenerator
import asyncio
from core.event_bus import get_event_bus
from core.utils.exceptions import capture_and_log

logger = logging.getLogger("Aura.VoiceBridge")

class VoiceConversationBridge:
    """
    The neural bridge between the Voice Pipeline (Senses) and the 
    Conversation Engine (Cognition).
    
    Handles low-latency streaming of thoughts and speech directly
    to the voice output buffer.
    """
    def __init__(self, orchestrator, conversation_engine):
        self._orch = orchestrator
        self._engine = conversation_engine
        self._bus = get_event_bus()
        self._active_utterance_task: Optional[asyncio.Task] = None
        
    async def process_voice_input(self, text: str):
        """
        Routes transcribed voice input into the cognitive engine.
        Supports real-time interruption of current speaking tasks.
        """
        logger.info(f"🎙️ Voice Bridge: Routing utterance -> {text[:50]}...")
        
        # 1. Interrupt any current TTS or thinking
        if self._active_utterance_task and not self._active_utterance_task.done():
            self._active_utterance_task.cancel()
            
        # 2. Process through the full cognitive pipeline
        self._active_utterance_task = get_task_tracker().create_task(
            self._orch.process_user_input(text, origin="voice")
        )
        
        try:
            response = await self._active_utterance_task
            return response
        except asyncio.CancelledError:
            logger.debug("Voice Bridge: Task cancelled due to barge-in/new input")
            return None
        except Exception as e:
            logger.error("Voice Bridge process input error: %s", e)
            return None

    async def stream_response_to_voice(self, stream: AsyncGenerator[str, None]):
        """
        Feeds tokens/chunks from the LLM directly into the voice pipeline's
        streaming buffer for low-latency response.
        Chunks tokens into speakable clauses before sending to the engine.
        """
        import re
        from core.container import ServiceContainer
        
        voice_engine = ServiceContainer.get("voice_presence", default=None)
        if not voice_engine:
            logger.debug("VoiceBridge: No voice presence engine found. Dropping stream.")
            async for _ in stream:
                pass
            return

        buffer: str = ""
        # Delimiters that constitute a "speakable clause"
        delimiters = re.compile(r'([.!?\n]+|,\s)')
        
        logger.info("🎙️ Voice Bridge: Streaming response chunks to voice engine...")
        try:
            async for token in stream:
                if self._active_utterance_task and self._active_utterance_task.cancelled():
                    logger.debug("VoiceBridge: Stream aborted due to barge-in.")
                    break
                    
                buffer = str(buffer) + str(token)
                
                # If we encounter a clause-ending delimiter, ship it to the TTS engine
                match = delimiters.search(buffer)
                if match:
                    split_idx = match.end()
                    chunk = buffer[:split_idx].strip()
                    buffer = buffer[split_idx:]
                    
                    if chunk:
                        # Fallback for old TTSEngine vs DecoupledVoiceEngine
                        if hasattr(voice_engine, "speak_nonblocking"):
                            voice_engine.speak_nonblocking(chunk)
                        elif hasattr(voice_engine, "speak"):
                            # If it's pure async we schedule it
                            get_task_tracker().create_task(voice_engine.speak(chunk))
                            
            # Flush any remaining text in the buffer
            if buffer.strip():
                if hasattr(voice_engine, "speak_nonblocking"):
                    voice_engine.speak_nonblocking(buffer.strip())
                elif hasattr(voice_engine, "speak"):
                    get_task_tracker().create_task(voice_engine.speak(buffer.strip()))
                    
        except Exception as e:
            logger.error("VoiceBridge Stream Error: %s", e)
