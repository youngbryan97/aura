import asyncio
import logging
import numpy as np
try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    import webrtcvad
except ImportError:
    webrtcvad = None

import subprocess
import time
from typing import Optional
from core.container import ServiceContainer
from core.utils.task_tracker import fire_and_track
try:
    from core.managers.vram_manager import get_vram_manager
except ImportError:
    def get_vram_manager(): return None

try:
    import mlx_whisper
except ImportError:
    mlx_whisper = None

logger = logging.getLogger("Aura.LocalVoice")

class LocalVoiceCortex:
    """
    Aura's Local Auditory and Speech Cortex.
    Handles low-latency VAD, STT (Whisper), and TTS (say).
    Optimized for Apple Silicon.
    """
    name = "local_voice_cortex"

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator or ServiceContainer.get("orchestrator", default=None)
        self.is_listening = False
        
        # Audio Settings
        if pyaudio:
            self.FORMAT = pyaudio.paInt16
            self.CHANNELS = 1
            self.RATE = 16000
            self.CHUNK = 320        # 20ms VAD frame @ 16kHz (valid WebRTCVAD size)
            self.DMA_BUFFER = 2048  # PyAudio DMA buffer for M1 stability
        else:
            self.FORMAT = self.CHANNELS = self.RATE = self.CHUNK = self.DMA_BUFFER = None
        
        # Model State
        self.stt_model = None
        self.whisper_params = {
            "beam_size": 5,
            "best_of": 5,
            "path_or_hf_repo": "mlx-community/whisper-small.en-mlx"
        }
        self.vad = None
        self.audio_interface = None
        self._loop_task: Optional[asyncio.Task] = None
        self.audio_queue: Optional[asyncio.Queue] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_event: Optional[asyncio.Event] = None
        self._vad_timeout = 2.0  # seconds

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Fires in a separate C-thread by PyAudio."""
        if not self._shutdown_event.is_set():
            self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, in_data)
        return (None, pyaudio.paContinue)

    async def start(self):
        """Initializes hardware and starts the listen loop."""
        try:
            self.loop = asyncio.get_running_loop()
            self.audio_queue = asyncio.Queue(maxsize=500)
            self._shutdown_event = asyncio.Event()
            logger.info("🧠 Loading Local Auditory Cortex (mlx-whisper)...")
            self._shutdown_event.clear()
            if not mlx_whisper:
                logger.warning("mlx-whisper not installed. Voice Cortex will fail transcribing.")

            if not webrtcvad or not pyaudio:
                logger.warning("⚠️ CRITICAL Hardware components (PyAudio/WebRTCVAD) missing. Headless Mode active.")
                self.is_listening = False
                return

            self.vad = webrtcvad.Vad()
            self.vad.set_mode(2) # Intermediate aggressiveness
            
            self.audio_interface = pyaudio.PyAudio()
            self.is_listening = True
            
            self._loop_task = fire_and_track(self.listen_loop(), name="VoiceListenLoop")
            logger.info("✅ Voice Cortex online. Aura is listening locally.")
        except Exception as e:
            logger.error(f"Failed to start Voice Cortex: {e}")

    async def stop(self):
        """Clean shutdown of audio hardware with queue drain."""
        self.is_listening = False
        if self._shutdown_event:
            self._shutdown_event.set()

        if self._loop_task:
            self._loop_task.cancel()
            try:
                await asyncio.wait_for(self._loop_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.debug('Ignored Exception in local_voice_cortex.py: %s', "unknown_error")

        if self.audio_interface:
            self.audio_interface.terminate()

        if self.audio_queue:
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        logger.info("Voice Cortex disengaged.")

    async def speak(self, text: str):
        """Reflexive speech output via Sovereign Voice Engine (High Fidelity)."""
        logger.info(f"🗣️ Aura: {text}")
        try:
            from core.container import ServiceContainer
            voice = ServiceContainer.get("voice_engine", default=None)
            if voice:
                await voice.synthesize_speech(text)
            else:
                # Last resort fallback to system say
                await asyncio.to_thread(subprocess.run, ["say", "-v", "Samantha", text])
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            try:
                await asyncio.to_thread(subprocess.run, ["say", text])
            except Exception as e2:
                logger.error(f"Fallback TTS also failed: {e2}")

    async def listen_loop(self):
        """Main VAD -> STT -> Orchestrator loop with timeout watchdog and auto-restart."""
        # Cache service references once, outside the hot loop
        reliability = ServiceContainer.get("reliability_engine", default=None)
        vram_manager = get_vram_manager()
        retry_delay = 1.0

        while self.is_listening and not self._shutdown_event.is_set():
            try:
                stream = self.audio_interface.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True,
                    frames_per_buffer=self.DMA_BUFFER,
                    stream_callback=self._audio_callback,
                )
                stream.start_stream()
                retry_delay = 1.0  # Reset on successful open
            except Exception as e:
                logger.error(f"Could not open audio stream: {e}. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(60.0, retry_delay * 2)
                continue

            logger.debug("Listening for voice activity...")
            
            try:
                while self.is_listening and not self._shutdown_event.is_set():
                    frames = []
                silence_count = 0
                is_speaking = False
                
                while self.is_listening and not self._shutdown_event.is_set():
                    try:
                        # Watchdog timeout for queue access
                        data = await asyncio.wait_for(self.audio_queue.get(), timeout=self._vad_timeout)
                        
                        # Reliability heartbeat
                        if reliability:
                            vram_usage = vram_manager.usage() if vram_manager else 0.0
                            await reliability.heartbeat(
                                "local_voice_cortex", stability=1.0, pressure=vram_usage
                            )
                            
                        # VAD check
                        # Process audio in proper 320-byte VAD frames (handled by CHUNK=320)
                        if self.vad.is_speech(data, self.RATE):
                            if not is_speaking:
                                logger.debug("Voice detected.")
                            is_speaking = True
                            frames.append(data)
                            silence_count = 0
                        elif is_speaking:
                            frames.append(data)
                            silence_count += 1
                            if silence_count > 50: # ~1 second of silence (50 × 20ms frames)
                                break
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.debug(f"Audio read error: {e}")
                        break

                if frames and self.is_listening:
                    # Transcribe
                    await self._process_audio_segment(frames)
            finally:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
                
            if not self.is_listening or self._shutdown_event.is_set():
                break  # Normal exit
            logger.warning("🎙️ Audio loop exited abnormally, restarting stream...")
            await asyncio.sleep(1.0)

    async def _process_audio_segment(self, frames):
        """Transcribes frames and routes to the orchestrator's Broca area."""
        if not mlx_whisper:
            return
            
        try:
            # Convert buffer to numpy array for Whisper
            audio_data = (
                np.frombuffer(b"".join(frames), np.int16)
                .flatten()
                .astype(np.float32)
                / 32768.0
            )
            
            vram = get_vram_manager()
            
            transcribe_task = asyncio.create_task(
                asyncio.to_thread(mlx_whisper.transcribe, audio_data, **self.whisper_params)
            )
            
            try:
                if vram:
                    async with vram.acquire_session("mlx-whisper"):
                        result = await asyncio.shield(transcribe_task)
                else:
                    result = await asyncio.shield(transcribe_task)
            except asyncio.CancelledError:
                # IMPORTANT: If cancelled, do NOT release the vram lock until the thread 
                # actually finishes, otherwise the worker thread will crash the neural engine 
                # when another model is swapped into VRAM concurrently.
                logger.warning("⚠️ Transcribe cancelled! Waiting for thread to release VRAM cleanly...")
                try:
                    await transcribe_task
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
                raise
            transcript = result.get("text", "").strip()
            
            if transcript and len(transcript) > 2:
                logger.info(f"👂 Heard: \"{transcript}\"")
                
                if self.orchestrator:
                    # Send to Broca's Area (fast local LLM)
                    response = await self.orchestrator.generate_voice_response(transcript)
                    
                    # Reflexive Speech
                    if response:
                        await self.speak(response)
                        
        except Exception as e:
            logger.error(f"Error processing audio segment: {e}")