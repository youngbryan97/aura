"""Local LLM Server Management
Sets up and manages local LLM servers for complete autonomy.
Includes failsafes to ensure the "Titan" model is pulled and loaded.
"""
from core.runtime.errors import record_degradation
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("Brain.LocalLLM")

class LocalLLMServer:
    def __init__(self, model_name: str, port: int):
        self.model_name = model_name
        self.port = port
        self.process = None

    def start(self):
        raise NotImplementedError

    async def is_running(self) -> bool:
        try:
            # Most local servers have a health or version endpoint
            import httpx
            async with httpx.AsyncClient() as client:
                url = f"http://localhost:{self.port}/api/tags" if self.port == 11434 else f"http://localhost:{self.port}/v1/models"
                response = await client.get(url, timeout=2)
                return response.status_code == 200
        except Exception:
            return False

class OllamaManager(LocalLLMServer):
    def __init__(self, model_name: str = "llama3.1:70b", port: int = 11434):
        super().__init__(model_name, port)

    def ensure_installed(self):
        """Checks if Ollama is installed, if not, logs instruction."""
        try:
            subprocess.run(["ollama", "--version"], check=True, capture_output=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.error("❌ Ollama not found. Please install from https://ollama.com")
            return False

    def ensure_model(self):
        """Ensures the Titan model is pulled."""
        logger.info("Checking for Titan model: %s", self.model_name)
        try:
            res = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if self.model_name not in res.stdout:
                logger.info("📥 Pulling %s... this may take a while.", self.model_name)
                subprocess.run(["ollama", "pull", self.model_name], check=True)
            return True
        except Exception as e:
            record_degradation('local_llm_setup', e)
            logger.error("Failed to ensure model %s: %s", self.model_name, e)
            return False

    async def start(self):
        if await self.is_running():
            logger.info("✅ Ollama is already running.")
            return self.ensure_model()
        
        logger.info("🚀 Starting Ollama serve...")
        try:
            # We don't usually need to start 'ollama serve' manually if the service is running,
            # but for a self-contained system we attempt it.
            self.process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Poll for readiness instead of blind sleep
            for _ in range(20):  # Up to 10 seconds
                await asyncio.sleep(0.5)
                if await self.is_running():
                    return self.ensure_model()
            logger.error("Ollama did not become ready within 10s")
        except Exception as e:
            record_degradation('local_llm_setup', e)
            logger.error("Failed to start Ollama: %s", e)
        return False

class LocalLLMManager:
    def __init__(self):
        from core.config import config
        # Titan uses the deep reasoning model, fallback uses the fast model
        self.titan = OllamaManager(model_name=config.llm.deep_model)
        self.fallback = OllamaManager(model_name=config.llm.fast_model)

    async def boot_titan(self):
        """Failsafe boot sequence for the primary brain."""
        if not self.titan.ensure_installed():
            return False
        
        if await self.titan.start():
            logger.info("🧠 Titan Brain Phase 1: Loaded.")
            return True
        else:
            logger.warning("⚠️ Titan 70B failed to load. Attempting 8B Fallback...")
            if await self.fallback.start():
                logger.info("🧠 Titan Brain Phase 1: Loaded (Reduced Capacity).")
                return True
        
        logger.error("💀 CRITICAL: All local brains failed to boot.")
        return False

# Global instance for initialization
llm_manager = LocalLLMManager()
