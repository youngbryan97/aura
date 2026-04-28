from core.runtime.errors import record_degradation
import logging
from typing import Any, Dict, List, Optional

from .provider import LLMProvider

logger = logging.getLogger("LLM.Fallback")

class FallbackLLMClient(LLMProvider):
    """A resilient LLM client that chains multiple providers.
    If the primary provider fails (e.g., 429 Quota), it falls back to the next one.
    """

    def __init__(self, providers: List[LLMProvider]):
        self.providers = providers
        if not self.providers:
            raise ValueError("FallbackLLMClient requires at least one provider.")
        logger.info("Fallback LLM Client initialized with %d providers.", len(self.providers))

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None) -> str:
        """Attempt text generation through the chain of providers."""
        last_error = None
        for provider in self.providers:
            try:
                if not provider.check_health():
                    continue
                return provider.generate_text(prompt, system_prompt, model)
            except Exception as e:
                record_degradation('fallback_client', e)
                last_error = e
                logger.warning("Provider %s failed: %s. Trying fallback if available...", provider.__class__.__name__, e)
        
        logger.error("All LLM providers in fallback chain failed.")
        raise last_error or RuntimeError("No providers available")

    def generate_json(self, prompt: str, schema: Dict[str, Any], system_prompt: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """Attempt JSON generation through the chain of providers."""
        last_error = None
        for provider in self.providers:
            try:
                if not provider.check_health():
                    continue
                return provider.generate_json(prompt, schema, system_prompt, model)
            except Exception as e:
                record_degradation('fallback_client', e)
                last_error = e
                logger.warning("Provider %s failed: %s. Trying fallback if available...", provider.__class__.__name__, e)
        
        logger.error("All LLM providers in fallback JSON chain failed.")
        raise last_error or RuntimeError("No providers available")

    async def generate_text_async(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None, **kwargs) -> str:
        """Attempt text generation through the chain of providers (Async)."""
        import asyncio
        last_error = None
        for provider in self.providers:
            try:
                # Check health (async pref)
                is_healthy = False
                if hasattr(provider, "check_health_async"):
                    is_healthy = await provider.check_health_async()
                else:
                    # Run sync health check in thread to avoid blocking
                    is_healthy = await asyncio.to_thread(provider.check_health)
                
                if not is_healthy:
                    continue

                # Generate (async pref)
                if hasattr(provider, "generate_text_async"):
                    return await provider.generate_text_async(prompt, system_prompt, model, **kwargs)
                else:
                    return await asyncio.to_thread(provider.generate_text, prompt, system_prompt, model, **kwargs)
            except Exception as e:
                record_degradation('fallback_client', e)
                last_error = e
                logger.warning("Provider %s failed (async): %s. Trying fallback...", provider.__class__.__name__, e)
        
        logger.error("All LLM providers in fallback chain failed (async).")
        raise last_error or RuntimeError("No providers available")

    async def generate_json_async(self, prompt: str, schema: Dict[str, Any], system_prompt: Optional[str] = None, model: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Attempt JSON generation through the chain of providers (Async)."""
        import asyncio
        last_error = None
        for provider in self.providers:
            try:
                # Check health (async pref)
                is_healthy = False
                if hasattr(provider, "check_health_async"):
                    is_healthy = await provider.check_health_async()
                else:
                    is_healthy = await asyncio.to_thread(provider.check_health)

                if not is_healthy:
                    continue

                # Generate (async pref)
                if hasattr(provider, "generate_json_async"):
                    return await provider.generate_json_async(prompt, schema, system_prompt, model, **kwargs)
                else:
                    return await asyncio.to_thread(provider.generate_json, prompt, schema, system_prompt, model, **kwargs)
            except Exception as e:
                record_degradation('fallback_client', e)
                last_error = e
                logger.warning("Provider %s failed (async): %s. Trying fallback...", provider.__class__.__name__, e)
        
        logger.error("All LLM providers in fallback JSON chain failed (async).")
        raise last_error or RuntimeError("No providers available")

    async def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None, **kwargs):
        """Attempt streaming generation through the chain of providers."""
        import asyncio
        last_error = None
        for provider in self.providers:
            try:
                is_healthy = False
                if hasattr(provider, "check_health_async"):
                    is_healthy = await provider.check_health_async()
                else:
                    is_healthy = await asyncio.to_thread(provider.check_health)

                if not is_healthy:
                    continue

                async for chunk in provider.generate_stream(prompt, system_prompt, model, **kwargs):
                    yield chunk
                return
            except Exception as e:
                record_degradation('fallback_client', e)
                last_error = e
                logger.warning("Provider %s stream failed: %s. Trying fallback...", provider.__class__.__name__, e)

        logger.error("All LLM providers in fallback stream chain failed.")
        if last_error:
            raise last_error
        raise RuntimeError("No providers available")

    def check_health(self) -> bool:
        """Health check returns true if ANY provider is healthy."""
        return any(p.check_health() for p in self.providers)

    async def check_health_async(self) -> bool:
        """Async health check returns true if ANY provider is healthy."""
        import asyncio
        for p in self.providers:
            if hasattr(p, "check_health_async"):
                if await p.check_health_async():
                    return True
            else:
                 if await asyncio.to_thread(p.check_health):
                     return True
        return False