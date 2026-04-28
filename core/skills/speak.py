from core.runtime.errors import record_degradation
import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.Voice")

class SpeakInput(BaseModel):
    text: str = Field(..., description="The text to convert to speech.")
    voice: Optional[str] = Field(None, description="Optional voice name (e.g., 'Samantha').")
    rate: Optional[int] = Field(185, description="Speech rate/speed.")

class SpeakSkill(BaseSkill):
    name = "speak"
    description = "Convert text to audible speech using local system voices."
    input_model = SpeakInput
    
    def __init__(self):
        super().__init__()
        self._voice_engine = None
        self._fallback_engine = None
        if sys.platform != "darwin":
            try:
                import pyttsx3
                self._fallback_engine = pyttsx3.init()
                self._fallback_engine.setProperty('rate', 175) 
            except Exception as e:
                record_degradation('speak', e)
                logger.warning("pyttsx3 init failed: %s", e)
            
    def _get_engine(self):
        """Resolve voice engine from ServiceContainer."""
        if self._voice_engine is None:
            try:
                from core.container import ServiceContainer
                self._voice_engine = ServiceContainer.get("voice_engine")
            except Exception as e:
                record_degradation('speak', e)
                logger.error("Failed to resolve voice_engine: %s", e)
        return self._voice_engine

    async def execute(self, params: SpeakInput, context: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(params, dict):
            try:
                params = SpeakInput(**params)
            except Exception as e:
                record_degradation('speak', e)
                return {"ok": False, "error": f"Invalid input: {e}"}

        text = params.text
        voice = params.voice or "Samantha"
        rate = str(params.rate or 185)
            
        if not text:
            return {"ok": False, "error": "No text provided to speak."}
            
        logger.info("🔊 Speaking: %s...", text[:50])
        
        # Strategy 1: Sovereign Voice Engine (Premium/Standard)
        engine = self._get_engine()
        if engine:
            try:
                await engine.synthesize_speech(text)
                return {"ok": True, "mode": "sovereign", "message": "Spoken via Sovereign Voice Engine."}
            except Exception as e:
                record_degradation('speak', e)
                logger.error("Sovereign synthesis failed: %s", e)

        # Strategy 2: macOS 'say' (High Quality Fallback)
        if sys.platform == "darwin":
            try:
                # Use Samantha for the cool, collected AGI persona
                await asyncio.create_subprocess_exec("say", "-v", voice, "-r", rate, text)
                return {"ok": True, "mode": "macos_say", "message": f"Spoken via macOS ({voice})."}
            except Exception as e:
                record_degradation('speak', e)
                logger.error("macOS say failed: %s", e)

        # Strategy 3: Local pyttsx3 (Generic Fallback)
        if self._fallback_engine:
            try:
                self._fallback_engine.say(text)
                self._fallback_engine.runAndWait()
                return {"ok": True, "mode": "pyttsx3", "message": "Spoken via local engine."}
            except Exception as e:
                record_degradation('speak', e)
                logger.error("pyttsx3 failed: %s", e)
        
        return {"ok": False, "error": "No voice engine available."}