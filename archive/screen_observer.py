"""core/screen_observer.py — v4.3.2 Unified Sensory Observer

Wraps the existing senses/vision_service.py and senses/audio_service.py
into a higher-level observer that:
  1. Manages start/stop of screen capture and audio recording
  2. Reads sensory JSON output files periodically
  3. Feeds observations (especially audio transcripts) into the knowledge graph
  4. Integrates with the agency system for proactive awareness

Limitations (current):
  - Vision: Screen capture works (via mss), but qwen2.5:14b is text-only.
    A vision-capable model (llava, qwen-vl) would be needed for image understanding.
    For now, captures are stored but not interpreted.
  - Audio: Uses local faster-whisper for transcription.
    Transcripts are fed into knowledge graph automatically.
  - macOS permissions required: Screen Recording, Microphone

Dependencies: mss, sounddevice, scipy (install via pip if missing)
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("Aura.ScreenObserver")

# Base directory — relative to project root
_BASE = Path(__file__).resolve().parent.parent


class ScreenObserver:
    """High-level controller for Aura's sensory systems.
    
    Manages background processes for screen capture and audio recording,
    reads their output, and feeds into the knowledge graph.
    """
    
    def __init__(self):
        self._vision_proc: Optional[subprocess.Popen] = None
        self._audio_proc: Optional[subprocess.Popen] = None
        self._vision_active = False
        self._audio_active = False
        self._last_vision_check = 0
        self._last_audio_check = 0
        self._last_transcript = ""
        self._observation_count = 0
        self._kg = None  # Lazy-loaded knowledge graph
    
    async def vision_active(self) -> bool:
        """Check if vision service is alive (Async)."""
        if self._vision_proc and await asyncio.to_thread(self._vision_proc.poll) is None:
            return True
        self._vision_active = False
        return False
    
    async def audio_active(self) -> bool:
        """Check if audio service is alive (Async)."""
        if self._audio_proc and await asyncio.to_thread(self._audio_proc.poll) is None:
            return True
        self._audio_active = False
        return False
    
    async def start_vision(self) -> Dict[str, Any]:
        """Start screen capture service (Async)."""
        if await self.vision_active():
            return {"ok": True, "message": "Vision already active", "pid": self._vision_proc.pid}
        
        script = _BASE / "senses" / "vision_service.py"
        if not script.exists():
            return {"ok": False, "error": f"Vision service not found at {script}"}
        
        try:
            # Popen is fast but technically synchronous; we can wrap it or just use it.
            # Using create_subprocess_exec would be even better but Popen gives more control for background procs.
            self._vision_proc = await asyncio.to_thread(
                subprocess.Popen,
                [sys.executable, str(script)],
                cwd=str(_BASE),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._vision_active = True
            logger.info("👁️ Vision service started (PID: %s)", self._vision_proc.pid)
            return {"ok": True, "pid": self._vision_proc.pid, "message": "Screen capture active"}
        except Exception as e:
            logger.error("Failed to start vision: %s", e)
            return {"ok": False, "error": str(e)}
    
    async def start_audio(self) -> Dict[str, Any]:
        """Start audio capture service (Async)."""
        if await self.audio_active():
            return {"ok": True, "message": "Audio already active", "pid": self._audio_proc.pid}
        
        script = _BASE / "senses" / "audio_service.py"
        if not script.exists():
            return {"ok": False, "error": f"Audio service not found at {script}"}
        
        try:
            self._audio_proc = await asyncio.to_thread(
                subprocess.Popen,
                [sys.executable, str(script)],
                cwd=str(_BASE),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._audio_active = True
            logger.info("👂 Audio service started (PID: %s)", self._audio_proc.pid)
            return {"ok": True, "pid": self._audio_proc.pid, "message": "Audio capture active"}
        except Exception as e:
            logger.error("Failed to start audio: %s", e)
            return {"ok": False, "error": str(e)}
    
    async def stop_vision(self) -> Dict[str, Any]:
        """Stop screen capture (Async)."""
        if self._vision_proc and await asyncio.to_thread(self._vision_proc.poll) is None:
            self._vision_proc.terminate()
            try:
                await asyncio.to_thread(self._vision_proc.wait, timeout=5)
            except subprocess.TimeoutExpired:
                self._vision_proc.kill()
            logger.info("👁️ Vision service stopped")
        self._vision_proc = None
        self._vision_active = False
        return {"ok": True, "message": "Vision stopped"}
    
    async def stop_audio(self) -> Dict[str, Any]:
        """Stop audio capture (Async)."""
        if self._audio_proc and await asyncio.to_thread(self._audio_proc.poll) is None:
            self._audio_proc.terminate()
            try:
                await asyncio.to_thread(self._audio_proc.wait, timeout=5)
            except subprocess.TimeoutExpired:
                self._audio_proc.kill()
            logger.info("👂 Audio service stopped")
        self._audio_proc = None
        self._audio_active = False
        return {"ok": True, "message": "Audio stopped"}
    
    async def stop_all(self):
        """Stop all sensory services (Async)."""
        await self.stop_vision()
        await self.stop_audio()
    
    async def read_vision(self) -> Optional[Dict]:
        """Read latest screen capture data (Async)."""
        path = _BASE / "sensory_vision.json"
        try:
            if not path.exists():
                return None
            mtime = await asyncio.to_thread(path.stat)
            age = time.time() - mtime.st_mtime
            if age > 30:
                return None
                
            def _load():
                with open(path) as f:
                    return json.load(f)
            data = await asyncio.to_thread(_load)
            data["age_seconds"] = round(age, 1)
            data["has_image"] = bool(data.get("image"))
            # Don't return the full base64 image (too large for API)
            if "image" in data:
                data["image_size"] = len(data["image"])
                del data["image"]
            return data
        except Exception as e:
            logger.debug("Vision read error: %s", e)
        return None
    
    async def read_audio(self) -> Optional[Dict]:
        """Read latest audio data with transcript (Async)."""
        path = _BASE / "sensory_audio.json"
        try:
            if not path.exists():
                return None
            mtime = await asyncio.to_thread(path.stat)
            age = time.time() - mtime.st_mtime
            if age > 30:
                return None

            def _load():
                with open(path) as f:
                    return json.load(f)
            data = await asyncio.to_thread(_load)
            data["age_seconds"] = round(age, 1)
            return data
        except Exception as e:
            logger.debug("Audio read error: %s", e)
        return None
    
    async def check_observations(self) -> list:
        """Poll sensory files and return new observations (Async).
        Call this periodically from the agency loop.
        
        Returns list of observation dicts suitable for knowledge graph ingestion.
        """
        observations = []
        now = time.time()
        
        # Check audio transcript (most useful without vision model)
        if now - self._last_audio_check > 10:  # Every 10 seconds
            self._last_audio_check = now
            audio = await self.read_audio()
            if audio and audio.get("transcript"):
                transcript = audio["transcript"].strip()
                if transcript and transcript != self._last_transcript:
                    self._last_transcript = transcript
                    observations.append({
                        "type": "audio_observation",
                        "content": f"Heard: {transcript}",
                        "source": "microphone",
                        "confidence": 0.7,
                        "timestamp": audio.get("timestamp", now)
                    })
        
        # Check vision (log activity but can't interpret without vision model)
        if now - self._last_vision_check > 15:  # Every 15 seconds
            self._last_vision_check = now
            vision = await self.read_vision()
            if vision and vision.get("has_image"):
                self._observation_count += 1
                # Only log periodically to avoid spam
                if self._observation_count % 10 == 1:
                    observations.append({
                        "type": "vision_observation",
                        "content": f"Screen capture active — frame {self._observation_count} captured ({vision.get('image_size', 0)} bytes)",
                        "source": "screen_capture",
                        "confidence": 0.3,  # Low confidence — not actually interpreted
                        "timestamp": vision.get("timestamp", now)
                    })
        
        # Feed observations to knowledge graph
        if observations:
            await self._store_observations(observations)
        
        return observations
    
    async def _store_observations(self, observations: list):
        """Store observations in the knowledge graph (Async)."""
        kg = self._get_kg()
        if not kg:
            return
        
        for obs in observations:
            try:
                # DB I/O must be offloaded to prevent event loop starvation
                await asyncio.to_thread(
                    kg.add_knowledge,
                    content=obs["content"],
                    knowledge_type=obs.get("type", "observation"),
                    source=obs.get("source", "senses"),
                    confidence=obs.get("confidence", 0.5),
                )
            except Exception as e:
                logger.debug("Failed to store observation: %s", e)
    
    def _get_kg(self):
        """Lazy-load knowledge graph."""
        if self._kg is None:
            try:
                from core.memory.knowledge_graph import PersistentKnowledgeGraph
                self._kg = PersistentKnowledgeGraph(str(_BASE / "data" / "knowledge.db"))
            except Exception as exc:
                logger.debug("Suppressed: %%s", exc)

                return self._kg
    
    def get_status(self) -> Dict[str, Any]:
        """Get full status of all sensory systems."""
        vision_data = self.read_vision()
        audio_data = self.read_audio()
        
        return {
            "vision": {
                "active": self.vision_active,
                "pid": self._vision_proc.pid if self._vision_proc and self._vision_proc.poll() is None else None,
                "last_capture": vision_data is not None,
                "frames_captured": self._observation_count,
                "note": "Screen capture active but image interpretation requires a vision-capable model (llava/qwen-vl)"
            },
            "audio": {
                "active": self.audio_active,
                "pid": self._audio_proc.pid if self._audio_proc and self._audio_proc.poll() is None else None,
                "last_transcript": audio_data.get("transcript", "")[:200] if audio_data else None,
                "note": "Audio capture + local transcription"
            },
            "observations_stored": self._observation_count,
            "capabilities": {
                "screen_capture": True,
                "screen_understanding": False,  # Needs vision model
                "audio_capture": True,
                "audio_transcription": True,
                "camera": True,   # via core/sensory_integration.py (OpenCV)
                "microphone": True,
            },
            "requirements": {
                "screen": "macOS Screen Recording permission + pip install mss",
                "audio": "macOS Microphone permission + pip install sounddevice scipy",
                "camera": "macOS Camera permission + pip install opencv-python",
                "transcription": "Local faster-whisper engine",
                "vision_understanding": "Vision-capable LLM (llava, qwen-vl) — current model is text-only"
            }
        }


# Singleton
_instance: Optional[ScreenObserver] = None

def get_screen_observer() -> ScreenObserver:
    global _instance
    if _instance is None:
        _instance = ScreenObserver()
    return _instance