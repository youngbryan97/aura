"""Sensory Integration System
Gives Aura access to cameras, microphones, speakers, and A/V production tools
"""
from core.runtime.errors import record_degradation
import asyncio
import base64
import logging
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from .container import ServiceContainer, ServiceLifetime
except (ImportError, ValueError):
    from container import ServiceContainer, ServiceLifetime

logger = logging.getLogger("Aura.SensoryIntegration")


class SensoryModality(Enum):
    """Types of sensory input"""

    VISION = "vision"
    HEARING = "hearing"
    TEXT = "text"


class SensorySystem:
    """Manages Aura's sensory perception across multiple modalities.
    
    Enables:
    - Vision (camera)
    - Hearing (microphone)
    - Speech (speakers/TTS)
    - Audio/visual production
    """
    
    def __init__(self):
        self.vision = VisionSystem()
        self.hearing = HearingSystem()
        self.speech = SpeechSystem()
        self.av_production = AVProductionSystem()
        
        # Sensory memory (recent perceptions)
        from collections import deque
        # Issue 43: Use deque(maxlen=) for O(1) memory management
        self.max_memory_items = 100
        self.sensory_memory = deque(maxlen=self.max_memory_items)
        
    async def perceive(self, modality: SensoryModality, **kwargs) -> Dict[str, Any]:
        """Perceive through specified sensory modality (Async)."""
        perception = {
            "timestamp": time.time(),
            "modality": modality.value,
            "data": None,
            "interpretation": None
        }
        
        try:
            if modality == SensoryModality.VISION:
                perception["data"] = await self.vision.capture(**kwargs)
                perception["interpretation"] = await self.vision.analyze(perception["data"])
                
            elif modality == SensoryModality.HEARING:
                perception["data"] = await self.hearing.listen(**kwargs)
                perception["interpretation"] = await self.hearing.transcribe(perception["data"])
                
            elif modality == SensoryModality.TEXT:
                perception["data"] = kwargs.get("text")
                perception["interpretation"] = perception["data"]
            
            # Store observations
            self._store_in_memory(perception)
            return perception
            
        except Exception as e:
            record_degradation('sensory_integration', e)
            logger.error("Perception failed: %s", e)
            perception["error"] = str(e)
            return perception
    
    async def express(self, modality: SensoryModality, content: Any, **kwargs) -> Dict[str, Any]:
        """Express through specified modality (Async)."""
        expression = {
            "timestamp": time.time(),
            "modality": modality.value,
            "content": content,
            "success": False
        }
        
        try:
            if modality == SensoryModality.HEARING:
                result = await self.speech.speak(content, **kwargs)
                expression["success"] = result.get("success", False)
                expression["audio_file"] = result.get("audio_file")
            
            return expression
            
        except Exception as e:
            record_degradation('sensory_integration', e)
            logger.error("Expression failed: %s", e)
            expression["error"] = str(e)
            return expression
    
    def _store_in_memory(self, perception: Dict[str, Any]):
        """Store perception in short-term sensory memory"""
        # Issue 43: deque handle limits automatically
        self.sensory_memory.append(perception)
    
    def get_recent_perceptions(self, modality: Optional[SensoryModality] = None, count: int = 10) -> List[Dict]:
        """Get recent perceptions, optionally filtered by modality"""
        # Issue 43: Convert deque slice to list
        perceptions = list(self.sensory_memory)[-count:]
        
        if modality:
            perceptions = [p for p in perceptions if p["modality"] == modality.value]
        
        return perceptions


class VisionSystem:
    """Camera and visual perception system.
    
    Enables Aura to:
    - Capture images/video from camera
    - Analyze visual content
    - Recognize objects, faces, text
    - Understand scenes
    """
    
    def __init__(self):
        self._camera_checked = False
        self._camera_available = False
        self.last_capture = None

    @property
    def camera_available(self) -> bool:
        """Public access to camera availability."""
        return self._camera_available

    async def _get_camera_available(self) -> bool:
        """Probe hardware asynchronously."""
        if not self._camera_checked:
            self._camera_available = await asyncio.to_thread(self._check_camera)
            self._camera_checked = True
        return self._camera_available
        
    def _check_camera(self) -> bool:
        """Check if camera is available (Must be called in thread)."""
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            available = cap.isOpened()
            cap.release()
            return available
        except ImportError:
            return False
        except Exception:
            return False
    
    async def capture(self, duration: float = 0, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Capture from camera (Async)."""
        if not await self._get_camera_available():
            return {"error": "camera_not_available"}
        
        def _do_capture():
            try:
                import cv2
                cap = cv2.VideoCapture(0)
                if duration == 0:
                    ret, frame = cap.read()
                    cap.release()
                    if not ret: return {"error": "capture_failed"}
                    if save_path: cv2.imwrite(save_path, frame)
                    _, buffer = cv2.imencode('.jpg', frame)
                    return {
                        "type": "image",
                        "data": base64.b64encode(buffer).decode('utf-8'),
                        "path": save_path,
                        "timestamp": time.time()
                    }
                else:
                    path = save_path or f"capture_{int(time.time())}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(path, fourcc, 20.0, (640, 480))
                    start = time.time()
                    while time.time() - start < duration:
                        ret, frame = cap.read()
                        if ret: out.write(frame)
                    cap.release()
                    out.release()
                    return {"type": "video", "path": path, "duration": duration, "timestamp": time.time()}
            except Exception as e:
                record_degradation('sensory_integration', e)
                return {"error": str(e)}

        result = await asyncio.to_thread(_do_capture)
        if "error" not in result:
            self.last_capture = result
        return result
    
    async def analyze(self, capture_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze captured visual data via the cognitive engine."""
        if not capture_data or "error" in capture_data:
            return {"error": "invalid_capture"}

        # Try real vision analysis via screen_vision -> cognitive engine
        try:
            from core.container import ServiceContainer
            brain = ServiceContainer.get("cognitive_engine", default=None)
            if brain is None:
                brain = ServiceContainer.get("brain", default=None)

            if brain is not None and hasattr(brain, "think"):
                import base64
                image_path = capture_data.get("path")
                image_b64 = None

                if image_path:
                    from PIL import Image
                    import io
                    img = Image.open(image_path)
                    img = img.convert("RGB")
                    img.thumbnail((672, 672))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=80)
                    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                elif capture_data.get("base64"):
                    image_b64 = capture_data["base64"]

                if image_b64:
                    description = await brain.think(
                        "Describe what you see in this image concisely. "
                        "List any objects, text, or people visible.",
                        images=[image_b64],
                    )
                    return {
                        "timestamp": time.time(),
                        "scene_description": str(description or "").strip(),
                        "objects_detected": [],
                        "text_detected": [],
                        "faces_detected": 0,
                    }
        except Exception as e:
            record_degradation('sensory_integration', e)
            logger.debug("Vision analysis via brain failed: %s", e)

        # Fallback: no vision model available
        return {
            "timestamp": time.time(),
            "objects_detected": [],
            "scene_description": "No vision model available for analysis.",
            "text_detected": [],
            "faces_detected": 0,
        }


class HearingSystem:
    """Microphone and audio perception system.
    
    Enables Aura to:
    - Record audio from microphone
    - Transcribe speech to text
    - Understand tone and emotion
    - Detect sounds
    """
    
    def __init__(self):
        self._microphone_checked = False
        self._microphone_available = False
        self.last_recording = None

    @property
    def microphone_available(self) -> bool:
        """Public access to microphone availability."""
        return self._microphone_available

    async def _get_microphone_available(self) -> bool:
        """Probe microphone hardware asynchronously."""
        if not self._microphone_checked:
            self._microphone_available = await asyncio.to_thread(self._check_microphone)
            self._microphone_checked = True
        return self._microphone_available
        
    def _check_microphone(self) -> bool:
        """Check if microphone is available (Must be called in thread)."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            # Look for any input device
            return any(d.get('max_input_channels', 0) > 0 for d in devices)
        except Exception:
            return False
    
    async def listen(self, duration: float = 5.0, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Record audio from microphone (Async)."""
        if not await self._get_microphone_available():
            return {"error": "microphone_not_available"}
        
        def _do_listen():
            try:
                import soundfile as sf
                import sounddevice as sd
                import numpy as np
                
                path = save_path or f"recording_{int(time.time())}.wav"
                CHANNELS, RATE = 1, 44100
                
                # Record as numpy array
                recording = sd.rec(int(duration * RATE), samplerate=RATE, channels=CHANNELS)
                sd.wait() # Wait for recording to finish
                
                # Save using soundfile
                sf.write(path, recording, RATE)
                
                return {"type": "audio", "path": path, "duration": duration, "timestamp": time.time()}
            except Exception as e:
                record_degradation('sensory_integration', e)
                return {"error": str(e)}

        result = await asyncio.to_thread(_do_listen)
        if "error" not in result:
            self.last_recording = result
        return result
    
    async def transcribe(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe audio to text (Async)."""
        if not audio_data or "error" in audio_data:
            return {"error": "invalid_audio"}
        
        def _do_transcribe():
            try:
                from core.senses.voice_socket_logic import get_whisper_model
                model = get_whisper_model("tiny")
                if model:
                    segments, info = model.transcribe(audio_data["path"], beam_size=5)
                    text = " ".join([seg.text for seg in segments]).strip()
                    return {"text": text, "confidence": 0.95, "language": "en"}
                else:
                    import speech_recognition as sr
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(audio_data["path"]) as src:
                        audio = recognizer.record(src)
                    text = recognizer.recognize_google(audio)
                    return {"text": text, "confidence": 0.8, "language": "en"}
            except Exception as e:
                record_degradation('sensory_integration', e)
                return {"text": "[Transcription failed]", "error": str(e)}

        result = await asyncio.to_thread(_do_transcribe)
        result["timestamp"] = time.time()
        return result


class SpeechSystem:
    """Text-to-speech and voice synthesis system.
    
    Enables Aura to:
    - Speak text aloud
    - Use different voices/emotions
    - Control speech rate, pitch
    """
    
    def __init__(self):
        # Issue 42: Lazy-init engine and store here
        self._engine = None
        self._lock = threading.Lock()
        self.tts_available = self._check_tts()
    def _check_tts(self) -> bool:
        """Check if TTS is available"""
        try:
            import pyttsx3
            return True
        except ImportError:
            logger.warning("pyttsx3 not installed - TTS unavailable")
            return False
    
    async def speak(self, text: str, rate: int = 150, volume: float = 1.0, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Speak text aloud using TTS (Async)."""
        if not self.tts_available:
            return {"error": "tts_not_available", "success": False}
        
        def _do_speak():
            try:
                import pyttsx3
                # Issue 42: Lazy-init and reuse engine with lock
                with self._lock:
                    if self._engine is None:
                        self._engine = pyttsx3.init()
                    
                    engine = self._engine
                    engine.setProperty('rate', rate)
                    engine.setProperty('volume', volume)
                    if save_path:
                        engine.save_to_file(text, save_path)
                        engine.runAndWait()
                        return {"success": True, "audio_file": save_path}
                    else:
                        engine.say(text)
                        engine.runAndWait()
                        return {"success": True}
            except Exception as e:
                record_degradation('sensory_integration', e)
                return {"success": False, "error": str(e)}

        result = await asyncio.to_thread(_do_speak)
        result["timestamp"] = time.time()
        result["text"] = text
        return result


class AVProductionSystem:
    """Audio/visual production tools.
    
    Enables Aura to:
    - Edit audio/video
    - Create visual content
    - Generate images/animations
    - Mix audio
    """
    
    def __init__(self):
        pass
    
    async def create_image(self, description: str, style: str = "realistic") -> Dict[str, Any]:
        """Generate image via local Stable Diffusion or brain inference."""
        try:
            from core.container import ServiceContainer
            brain = ServiceContainer.get("cognitive_engine", default=None)
            if brain and hasattr(brain, "generate_image"):
                result = await brain.generate_image(description, style=style)
                if result:
                    return {"path": result, "description": description, "timestamp": time.time()}
        except Exception as e:
            record_degradation('sensory_integration', e)
            logger.debug("Image generation via brain failed: %s", e)

        return {"error": "no_image_model", "description": description}

    async def edit_video(self, video_path: str, edits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply edits to video via FFmpeg."""
        import shutil
        if not shutil.which("ffmpeg"):
            return {"error": "ffmpeg_not_installed"}

        import subprocess
        try:
            # Basic trim operation as baseline
            for edit in edits:
                if edit.get("type") == "trim":
                    start = edit.get("start", 0)
                    end = edit.get("end")
                    output = video_path.rsplit(".", 1)[0] + "_edited.mp4"
                    cmd = ["ffmpeg", "-y", "-i", video_path, "-ss", str(start)]
                    if end:
                        cmd.extend(["-to", str(end)])
                    cmd.extend(["-c", "copy", output])
                    proc = await asyncio.to_thread(
                        subprocess.run, cmd, capture_output=True, timeout=60
                    )
                    if proc.returncode == 0:
                        return {"path": output, "edits_applied": len(edits)}
                    return {"error": proc.stderr.decode()[:200]}
            return {"error": "no_supported_edits", "supported": ["trim"]}
        except Exception as e:
            record_degradation('sensory_integration', e)
            return {"error": str(e)}


_sensory_lock = threading.Lock()

def get_sensory_system() -> SensorySystem:
    """Get global sensory system via DI container"""
    try:
        with _sensory_lock:
            if not ServiceContainer.get("sensory_system", None):
                ServiceContainer.register(
                    "sensory_system",
                    factory=lambda: SensorySystem(),
                    lifetime=ServiceLifetime.SINGLETON
                )
            res = ServiceContainer.get("sensory_system", default=None)
            if res is not None:
                return res
            return SensorySystem()
    except Exception as e:
        record_degradation('sensory_integration', e)
        logger.debug("ServiceContainer unavailable or failed: %s. Using transient SensorySystem.", e)
        return SensorySystem()


# Integration helpers
def integrate_sensory_system(orchestrator):
    """Integrate sensory system into orchestrator.
    
    Adds sensory perception as available actions.
    """
    sensory = get_sensory_system()
    
    # Store reference
    orchestrator.sensory_system = sensory
    
    logger.info("✓ Sensory system integrated")
    logger.info("  Camera: %s", 'available' if sensory.vision.camera_available else 'unavailable')
    logger.info("  Microphone: %s", 'available' if sensory.hearing.microphone_available else 'unavailable')
    logger.info("  TTS: %s", 'available' if sensory.speech.tts_available else 'unavailable')