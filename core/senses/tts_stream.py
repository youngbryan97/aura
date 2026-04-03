import logging
import asyncio

# Issue 30: Unused dead imports removed

logger = logging.getLogger("Senses.Mouth")

class FastMouth:
    """Shim for backward compatibility."""
    def __init__(self):
        from core.container import ServiceContainer
        self.engine = ServiceContainer.get("voice_engine", default=None)
        self._speak_task = None
        self._stream_task = None
    
    def speak(self, text: str):
        import asyncio
        if self.engine:
            try:
                loop = asyncio.get_running_loop()
                # Cancel previous task to prevent pile-up
                if self._speak_task and not self._speak_task.done():
                    self._speak_task.cancel()
                self._speak_task = loop.create_task(self.engine.speak(text))
            except RuntimeError as _e:
                logger.debug('Ignored RuntimeError in tts_stream.py: %s', _e)

    def speak_stream(self, text_generator):
        import asyncio
        if self.engine:
            try:
                loop = asyncio.get_running_loop()
                if self._stream_task and not self._stream_task.done():
                    self._stream_task.cancel()
                self._stream_task = loop.create_task(self.engine.speak_stream(text_generator))
            except RuntimeError as _e:
                logger.debug('Ignored RuntimeError in tts_stream.py: %s', _e)

    def stop(self):
        for task in (self._speak_task, self._stream_task):
            if task and not task.done():
                task.cancel()