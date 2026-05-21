"""Local LLM Server Management
Sets up and manages local LLM servers for complete autonomy.
Includes failsafes to ensure the "Titan" model is pulled and loaded.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import Any

from core.runtime.errors import FallbackClassification, record_degradation

logger = logging.getLogger("Brain.LocalLLM")


_LOCAL_LLM_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    subprocess.SubprocessError,
    subprocess.TimeoutExpired,
    asyncio.TimeoutError,
)
_VERSION_TIMEOUT_S = 10
_LIST_TIMEOUT_S = 20
_PULL_TIMEOUT_S = 60 * 60
_SERVE_READY_ATTEMPTS = 20
_SERVE_READY_INTERVAL_S = 0.5
_PROCESS_STOP_TIMEOUT_S = 5.0


def _record_local_llm_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "degraded",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "local_llm_setup",
        exc,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


class LocalLLMServer(ABC):
    def __init__(self, model_name: str, port: int):
        self.model_name = model_name
        self.port = port
        self.process = None

    @abstractmethod
    async def start(self) -> bool:
        raise NotImplementedError(
            f"{type(self).__name__}.start must be implemented by a local server adapter"
        )

    async def is_running(self) -> bool:
        try:
            # Most local servers have a health or version endpoint
            import httpx
        except ImportError as exc:
            logger.debug("Local LLM health check unavailable on port %s: %s", self.port, exc)
            return False

        try:
            async with httpx.AsyncClient() as client:
                url = (
                    f"http://localhost:{self.port}/api/tags"
                    if self.port == 11434
                    else f"http://localhost:{self.port}/v1/models"
                )
                response = await client.get(url, timeout=2)
                return response.status_code == 200
        except (httpx.HTTPError, AttributeError, RuntimeError, OSError) as exc:
            logger.debug("Local LLM health check unavailable on port %s: %s", self.port, exc)
            return False


class OllamaManager(LocalLLMServer):
    def __init__(self, model_name: str = "llama3.1:70b", port: int = 11434):
        super().__init__(model_name, port)

    def ensure_installed(self):
        """Checks if Ollama is installed, if not, logs instruction."""
        if shutil.which("ollama") is None:
            logger.error("❌ Ollama not found. Please install from https://ollama.com")
            return False
        try:
            subprocess.run(
                ["ollama", "--version"],
                check=True,
                capture_output=True,
                timeout=_VERSION_TIMEOUT_S,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            _record_local_llm_degradation(
                exc,
                action="reported ollama unavailable before local model boot",
                severity="warning",
            )
            logger.error("❌ Ollama not usable: %s", exc)
            return False

    def ensure_model(self):
        """Ensures the Titan model is pulled."""
        logger.info("Checking for Titan model: %s", self.model_name)
        try:
            res = subprocess.run(
                ["ollama", "list"],
                check=True,
                capture_output=True,
                text=True,
                timeout=_LIST_TIMEOUT_S,
            )
            if self.model_name not in res.stdout:
                logger.info("📥 Pulling %s... this may take a while.", self.model_name)
                subprocess.run(
                    ["ollama", "pull", self.model_name],
                    check=True,
                    timeout=_PULL_TIMEOUT_S,
                )
            return True
        except _LOCAL_LLM_RECOVERABLE_ERRORS as e:
            _record_local_llm_degradation(
                e,
                action="left local model unavailable after bounded ollama model check failed",
                severity="warning",
                extra={"model": self.model_name},
            )
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
            self.process = await asyncio.create_subprocess_exec(
                "ollama",
                "serve",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Poll for readiness instead of blind sleep
            for _ in range(_SERVE_READY_ATTEMPTS):
                await asyncio.sleep(_SERVE_READY_INTERVAL_S)
                if await self.is_running():
                    return self.ensure_model()
            await self._terminate_process("readiness_timeout")
            logger.error(
                "Ollama did not become ready within %.1fs",
                _SERVE_READY_ATTEMPTS * _SERVE_READY_INTERVAL_S,
            )
        except _LOCAL_LLM_RECOVERABLE_ERRORS as e:
            await self._terminate_process("start_failure")
            _record_local_llm_degradation(
                e,
                action="failed closed local LLM start and cleaned up ollama serve process",
                severity="warning",
                extra={"model": self.model_name, "port": self.port},
            )
            logger.error("Failed to start Ollama: %s", e)
        return False

    async def _terminate_process(self, reason: str) -> None:
        process = self.process
        if process is None:
            return
        if process.returncode is not None:
            return
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=_PROCESS_STOP_TIMEOUT_S)
        except (ProcessLookupError, RuntimeError, OSError, TimeoutError) as exc:
            _record_local_llm_degradation(
                exc,
                action="attempted local LLM process cleanup after failed start",
                severity="warning",
                extra={"reason": reason, "model": self.model_name},
            )
            try:
                process.kill()
            except (ProcessLookupError, RuntimeError, OSError):
                logger.debug("Ollama process already gone during cleanup.")


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
