import logging
import asyncio
import threading
import queue
import time
import subprocess
from enum import Enum, auto
from typing import Optional, List, Callable, Awaitable

logger = logging.getLogger("Aura.VoiceEngine")

class VoiceState(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()

class DecoupledVoiceEngine:
    """
    [PHASE 7] DECOUPLED VOICE ENGINE
    Reduces I/O blocking by moving synthesis to a background thread with its own queue.
    Prevents the '30-message stall' caused by event loop congestion.
    """
    def __init__(self, use_xtts: bool = True):
        self.state = VoiceState.IDLE
        self._speech_queue = queue.Queue()
        self._is_running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._tts_engine = None
        self._use_xtts = use_xtts
        self.interrupt_flag = threading.Event() # Thread-safe flag
        
        # Performance monitoring
        self._last_speech_time: float = 0.0
        self._speech_count = 0
        self._current_proc: Optional[subprocess.Popen] = None

    def start(self):
        if self._is_running:
            return
        self._is_running = True
        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True, name="AuraVoiceWorker")
        self._worker_thread.start()
        logger.info("🎙️ DecoupledVoiceEngine worker started.")

    def stop(self):
        self._is_running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2)

    def speak(self, text: str):
        """Enqueue text for speech. Non-blocking."""
        if not text or not text.strip():
            return
        self._speech_queue.put(text)
        logger.debug(f"Queued speech: {text[:50]}...")

    def interrupt(self):
        """Signal immediate silence."""
        self.interrupt_flag.set()
        # Drain queue
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
            except queue.Empty:
                break
        logger.warning("🎙️ VOICE INTERRUPTED: Queue drained.")

    def _speech_worker(self):
        """Isolated thread for synthesis and playback."""
        # Issue 41: Service lookup is expensive; do it once outside the loop
        try:
            from core.container import ServiceContainer
            self._init_engines()
        except Exception as e:
            logger.error(f"Failed to init speech worker engines: {e}")
            return

        while self._is_running:
            try:
                text = self._speech_queue.get(timeout=1.0)
                self.interrupt_flag.clear()
                self.state = VoiceState.SPEAKING
                
                start_time = time.time()
                self._synthesize_and_play(text)
                
                self._last_speech_time = time.time()
                self._speech_count += 1
                logger.debug(f"Speech finished in {self._last_speech_time - start_time:.2f}s (Total: {self._speech_count})")
                
                self.state = VoiceState.IDLE
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Speech worker error: {e}")
                self.state = VoiceState.IDLE

    def _init_engines(self):
        # We use a simple subprocess call in this thread, so no heavy setup required
        self._current_proc = None

    def _synthesize_and_play(self, text: str):
        """Internal synthesis call. Respects interrupt_flag."""
        import subprocess
        import re
        
        # Clean markdown and URLs before speaking
        clean = re.sub(r"```[\s\S]*?```", "", text)
        clean = re.sub(r"`[^`]+`", "", clean)
        clean = re.sub(r"https?://\S+", "", clean)
        clean = clean.strip()
        
        if not clean:
            return
            
        try:
            self._current_proc = subprocess.Popen(
                ["say", clean],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Poll so we can interrupt it instantly
            while self._current_proc and self._current_proc.poll() is None:
                if self.interrupt_flag.is_set():
                    self._current_proc.terminate()
                    break
                time.sleep(0.05)
                
        except Exception as e:
            logger.error(f"TTS Synthesis error: {e}")
