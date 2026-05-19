from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Any

from core.runtime.errors import ModelUnavailable, Severity, record_degradation

from .provider import LLMProvider

logger = logging.getLogger("LLM.Fallback")

_FALLBACK_RECOVERABLE_ERRORS = (
    RuntimeError,
    AttributeError,
    TypeError,
    ValueError,
    OSError,
    ConnectionError,
    TimeoutError,
)


@dataclass
class ProviderAttempt:
    operation: str
    provider: str
    status: str
    action: str
    error: str = ""


def _provider_name(provider: LLMProvider | Any) -> str:
    return getattr(provider, "name", None) or provider.__class__.__name__


def _error_summary(error: BaseException) -> str:
    return f"{type(error).__qualname__}: {error}"[:300]


def _record_fallback_degradation(
    error: BaseException,
    *,
    operation: str,
    provider: str,
    action: str,
    severity: Severity = "warning",
) -> None:
    record_degradation(
        "fallback_client",
        error,
        severity=severity,
        action=action,
        extra={"operation": operation, "provider": provider},
    )


class FallbackLLMClient(LLMProvider):
    """A resilient LLM client that chains multiple providers.

    Each provider attempt is recorded so runtime health can explain which
    model lane failed, which lane recovered, and when the chain fully exhausted.
    """

    def __init__(self, providers: list[LLMProvider]):
        self.providers = list(providers)
        self._last_attempts: list[ProviderAttempt] = []
        self._provider_failures: dict[str, int] = {}
        self._provider_successes: dict[str, int] = {}
        if not self.providers:
            raise ValueError("FallbackLLMClient requires at least one provider.")
        logger.info("Fallback LLM Client initialized with %d providers.", len(self.providers))

    def _reset_attempts(self) -> None:
        self._last_attempts = []

    def _remember(self, attempt: ProviderAttempt) -> None:
        self._last_attempts.append(attempt)
        if attempt.status == "failed":
            self._provider_failures[attempt.provider] = self._provider_failures.get(attempt.provider, 0) + 1
        elif attempt.status == "succeeded":
            self._provider_successes[attempt.provider] = self._provider_successes.get(attempt.provider, 0) + 1

    def get_status(self) -> dict[str, Any]:
        return {
            "provider_count": len(self.providers),
            "last_attempts": [asdict(attempt) for attempt in self._last_attempts],
            "provider_failures": dict(self._provider_failures),
            "provider_successes": dict(self._provider_successes),
        }

    def _check_health_sync(self, provider: LLMProvider, operation: str) -> bool:
        provider_name = _provider_name(provider)
        try:
            healthy = bool(provider.check_health())
        except _FALLBACK_RECOVERABLE_ERRORS as exc:
            action = "Provider health check failed; skipping provider and trying next lane"
            _record_fallback_degradation(
                exc,
                operation=operation,
                provider=provider_name,
                action=action,
                severity="warning",
            )
            self._remember(ProviderAttempt(operation, provider_name, "failed", action, _error_summary(exc)))
            return False
        if not healthy:
            self._remember(
                ProviderAttempt(
                    operation,
                    provider_name,
                    "skipped",
                    "Provider reported unhealthy; skipped before generation",
                )
            )
        return healthy

    async def _check_health_async(self, provider: LLMProvider, operation: str) -> bool:
        provider_name = _provider_name(provider)
        try:
            if hasattr(provider, "check_health_async"):
                healthy = bool(await provider.check_health_async())
            else:
                healthy = bool(await asyncio.to_thread(provider.check_health))
        except _FALLBACK_RECOVERABLE_ERRORS as exc:
            action = "Provider async health check failed; skipping provider and trying next lane"
            _record_fallback_degradation(
                exc,
                operation=operation,
                provider=provider_name,
                action=action,
                severity="warning",
            )
            self._remember(ProviderAttempt(operation, provider_name, "failed", action, _error_summary(exc)))
            return False
        if not healthy:
            self._remember(
                ProviderAttempt(
                    operation,
                    provider_name,
                    "skipped",
                    "Provider reported unhealthy; skipped before generation",
                )
            )
        return healthy

    def _raise_exhausted(self, operation: str, last_error: BaseException | None) -> None:
        error = last_error or ModelUnavailable(f"No healthy LLM providers available for {operation}")
        _record_fallback_degradation(
            error,
            operation=operation,
            provider="fallback_chain",
            action="Exhausted provider chain; failing closed instead of returning empty output",
            severity="critical",
        )
        logger.error("All LLM providers in fallback chain failed for %s.", operation)
        raise error

    def _validate_json_result(self, result: Any, provider_name: str) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise TypeError(f"{provider_name} returned {type(result).__name__}, expected dict")
        return result

    def generate_text(self, prompt: str, system_prompt: str | None = None, model: str | None = None) -> str:
        """Attempt text generation through the chain of providers."""
        self._reset_attempts()
        last_error: BaseException | None = None
        operation = "generate_text"
        for provider in self.providers:
            provider_name = _provider_name(provider)
            if not self._check_health_sync(provider, operation):
                continue
            try:
                result = provider.generate_text(prompt, system_prompt, model)
                self._remember(
                    ProviderAttempt(operation, provider_name, "succeeded", "Generated text with provider")
                )
                return result
            except _FALLBACK_RECOVERABLE_ERRORS as exc:
                action = "Provider text generation failed; trying next LLM lane"
                _record_fallback_degradation(
                    exc,
                    operation=operation,
                    provider=provider_name,
                    action=action,
                    severity="warning",
                )
                self._remember(ProviderAttempt(operation, provider_name, "failed", action, _error_summary(exc)))
                last_error = exc
                logger.warning("Provider %s failed: %s. Trying fallback if available...", provider_name, exc)
        self._raise_exhausted(operation, last_error)

    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Attempt JSON generation through the chain of providers."""
        self._reset_attempts()
        last_error: BaseException | None = None
        operation = "generate_json"
        for provider in self.providers:
            provider_name = _provider_name(provider)
            if not self._check_health_sync(provider, operation):
                continue
            try:
                result = self._validate_json_result(
                    provider.generate_json(prompt, schema, system_prompt, model),
                    provider_name,
                )
                self._remember(ProviderAttempt(operation, provider_name, "succeeded", "Generated JSON"))
                return result
            except _FALLBACK_RECOVERABLE_ERRORS as exc:
                action = "Provider JSON generation failed; trying next LLM lane"
                _record_fallback_degradation(
                    exc,
                    operation=operation,
                    provider=provider_name,
                    action=action,
                    severity="warning",
                )
                self._remember(ProviderAttempt(operation, provider_name, "failed", action, _error_summary(exc)))
                last_error = exc
                logger.warning("Provider %s failed: %s. Trying fallback if available...", provider_name, exc)
        self._raise_exhausted(operation, last_error)

    async def generate_text_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Attempt text generation through the chain of providers."""
        self._reset_attempts()
        last_error: BaseException | None = None
        operation = "generate_text_async"
        for provider in self.providers:
            provider_name = _provider_name(provider)
            if not await self._check_health_async(provider, operation):
                continue
            try:
                if hasattr(provider, "generate_text_async"):
                    result = await provider.generate_text_async(prompt, system_prompt, model, **kwargs)
                else:
                    result = await asyncio.to_thread(
                        provider.generate_text,
                        prompt,
                        system_prompt,
                        model,
                        **kwargs,
                    )
                self._remember(
                    ProviderAttempt(operation, provider_name, "succeeded", "Generated text with provider")
                )
                return result
            except _FALLBACK_RECOVERABLE_ERRORS as exc:
                action = "Provider async text generation failed; trying next LLM lane"
                _record_fallback_degradation(
                    exc,
                    operation=operation,
                    provider=provider_name,
                    action=action,
                    severity="warning",
                )
                self._remember(ProviderAttempt(operation, provider_name, "failed", action, _error_summary(exc)))
                last_error = exc
                logger.warning("Provider %s failed (async): %s. Trying fallback...", provider_name, exc)
        self._raise_exhausted(operation, last_error)

    async def generate_json_async(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_prompt: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Attempt JSON generation through the chain of providers."""
        self._reset_attempts()
        last_error: BaseException | None = None
        operation = "generate_json_async"
        for provider in self.providers:
            provider_name = _provider_name(provider)
            if not await self._check_health_async(provider, operation):
                continue
            try:
                if hasattr(provider, "generate_json_async"):
                    raw = await provider.generate_json_async(prompt, schema, system_prompt, model, **kwargs)
                else:
                    raw = await asyncio.to_thread(
                        provider.generate_json,
                        prompt,
                        schema,
                        system_prompt,
                        model,
                        **kwargs,
                    )
                result = self._validate_json_result(raw, provider_name)
                self._remember(ProviderAttempt(operation, provider_name, "succeeded", "Generated JSON"))
                return result
            except _FALLBACK_RECOVERABLE_ERRORS as exc:
                action = "Provider async JSON generation failed; trying next LLM lane"
                _record_fallback_degradation(
                    exc,
                    operation=operation,
                    provider=provider_name,
                    action=action,
                    severity="warning",
                )
                self._remember(ProviderAttempt(operation, provider_name, "failed", action, _error_summary(exc)))
                last_error = exc
                logger.warning("Provider %s failed (async): %s. Trying fallback...", provider_name, exc)
        self._raise_exhausted(operation, last_error)

    async def _buffer_provider_stream(
        self,
        provider: LLMProvider,
        prompt: str,
        system_prompt: str | None,
        model: str | None,
        **kwargs: Any,
    ) -> list[Any]:
        if hasattr(provider, "generate_stream"):
            stream = provider.generate_stream(prompt, system_prompt, model, **kwargs)
            if hasattr(stream, "__aiter__"):
                return [chunk async for chunk in stream]
            return list(stream)
        if hasattr(provider, "generate_text_async"):
            return [await provider.generate_text_async(prompt, system_prompt, model, **kwargs)]
        return [
            await asyncio.to_thread(
                provider.generate_text,
                prompt,
                system_prompt,
                model,
                **kwargs,
            )
        ]

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Attempt streaming generation through the chain without emitting failed partial output."""
        self._reset_attempts()
        last_error: BaseException | None = None
        operation = "generate_stream"
        for provider in self.providers:
            provider_name = _provider_name(provider)
            if not await self._check_health_async(provider, operation):
                continue
            try:
                chunks = await self._buffer_provider_stream(provider, prompt, system_prompt, model, **kwargs)
                self._remember(
                    ProviderAttempt(
                        operation,
                        provider_name,
                        "succeeded",
                        f"Generated stream with {len(chunks)} buffered chunk(s)",
                    )
                )
                for chunk in chunks:
                    yield chunk
                return
            except _FALLBACK_RECOVERABLE_ERRORS as exc:
                action = "Provider stream failed before emission; trying next LLM lane"
                _record_fallback_degradation(
                    exc,
                    operation=operation,
                    provider=provider_name,
                    action=action,
                    severity="warning",
                )
                self._remember(ProviderAttempt(operation, provider_name, "failed", action, _error_summary(exc)))
                last_error = exc
                logger.warning("Provider %s stream failed: %s. Trying fallback...", provider_name, exc)
        self._raise_exhausted(operation, last_error)

    def check_health(self) -> bool:
        """Health check returns true if any provider is healthy."""
        self._reset_attempts()
        return any(self._check_health_sync(provider, "check_health") for provider in self.providers)

    async def check_health_async(self) -> bool:
        """Async health check returns true if any provider is healthy."""
        self._reset_attempts()
        for provider in self.providers:
            if await self._check_health_async(provider, "check_health_async"):
                return True
        return False
