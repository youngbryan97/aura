"""Production code generation adapter for clean-room reconstruction.

This module is the LLM-layer implementation of the ``CodeGenerator``
protocol used by ``core.self_improvement``.  The reconstruction lab remains
deterministic at the authority boundary: the model only proposes code, while
syntax checks, guardrails, sandbox tests, and the promotion gate decide whether
anything is acceptable.
"""
from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import inspect
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.LLMCodeGenerator")


_CODE_GEN_SYSTEM_PROMPT = (
    "You are Aura's production Python code generator for clean-room module "
    "reconstruction.\n\n"
    "Return only complete, valid Python source code. Do not include markdown, "
    "analysis, prose, shell commands, or test output. Preserve the public "
    "interface described in the specification. Implement real behavior from "
    "the contract; do not hardcode expected test values. Include required "
    "imports and keep side effects minimal. Never use eval(), exec(), compile(), "
    "__import__(), network access, subprocesses, or filesystem reads of the "
    "original implementation."
)


_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(?P<code>.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class GenerationRequest:
    """Normalized request sent to Aura's LLM runtime."""

    prompt: str
    system_prompt: str
    prefer_tier: str
    max_tokens: int
    temperature: float
    origin: str = "reimplementation_lab"
    is_background: bool = True


def _first_pythonish_line(text: str) -> int:
    starters = (
        "from ",
        "import ",
        "class ",
        "def ",
        "async def ",
        "@",
        '"""',
        "'''",
        "#",
        "__all__",
    )
    for idx, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith(starters):
            return idx
    return 0


def extract_python_code(text: str) -> str:
    """Extract Python source from a model response without trusting wrappers."""

    raw = str(text or "").strip()
    if not raw:
        return ""

    fenced = _FENCE_RE.findall(raw)
    if fenced:
        candidates = [candidate.strip() for candidate in fenced if candidate.strip()]
        if candidates:
            return max(candidates, key=len).strip()

    lines = raw.splitlines()
    start = _first_pythonish_line(raw)
    if start:
        raw = "\n".join(lines[start:]).strip()

    # Some models append a short explanatory tail after otherwise valid code.
    # Prefer the full response if it parses; otherwise progressively trim the
    # tail until the candidate is syntactically valid.
    try:
        ast.parse(raw)
        return raw
    except SyntaxError:
        pass

    trimmed = raw.splitlines()
    for end in range(len(trimmed) - 1, 0, -1):
        candidate = "\n".join(trimmed[:end]).rstrip()
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue

    return raw


def _coerce_response_text(response: Any) -> str:
    """Normalize the different LLM client return shapes used in Aura."""

    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, tuple):
        if len(response) >= 2 and isinstance(response[0], bool):
            return str(response[1] or "") if response[0] else ""
        for item in response:
            text = _coerce_response_text(item)
            if text:
                return text
        return ""
    if isinstance(response, dict):
        if response.get("ok") is False:
            return ""
        for key in ("text", "content", "response", "output"):
            if response.get(key):
                return str(response[key])
        message = response.get("message")
        if isinstance(message, dict) and message.get("content"):
            return str(message["content"])
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            return _coerce_response_text(choices[0])
        return ""
    for attr in ("content", "text", "response"):
        value = getattr(response, attr, None)
        if value:
            return str(value)
    return str(response)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pass only supported kwargs unless the callable accepts arbitrary kwargs."""

    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)

    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return dict(kwargs)

    filtered: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in parameters:
            filtered[key] = value

    if "system" in parameters and "system_prompt" in kwargs and "system" not in filtered:
        filtered["system"] = kwargs["system_prompt"]

    if "context" in parameters and "context" not in filtered:
        filtered["context"] = dict(kwargs)

    return filtered


class LLMCodeGenerator:
    """CodeGenerator implementation backed by Aura's LLM runtime.

    ``generate_async`` is the preferred production path.  ``generate`` remains
    for the existing synchronous protocol and for tests that use a simple
    pluggable generator.
    """

    def __init__(
        self,
        *,
        router: Any | None = None,
        service_names: Iterable[str] = ("llm_router", "inference_gate", "cognitive_engine"),
        prefer_tier: str = "primary",
        prefer_endpoint: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.25,
        timeout_s: float = 180.0,
        fallback_to_stub: bool = True,
    ) -> None:
        self._router = router
        self.service_names = tuple(service_names)
        self.prefer_tier = prefer_tier
        self.prefer_endpoint = prefer_endpoint
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.fallback_to_stub = fallback_to_stub

    def generate(self, prompt: str, context: dict[str, Any]) -> str:
        """Synchronous protocol adapter."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.generate_async(prompt, context))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(lambda: asyncio.run(self.generate_async(prompt, context)))
            return future.result(timeout=self.timeout_s + 5.0)

    async def generate_async(self, prompt: str, context: dict[str, Any]) -> str:
        request = GenerationRequest(
            prompt=self._augment_prompt(prompt, context),
            system_prompt=_CODE_GEN_SYSTEM_PROMPT,
            prefer_tier=self.prefer_tier,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        try:
            router = self._resolve_router()
            if router is None:
                raise RuntimeError("no LLM router, inference gate, or cognitive engine is registered")

            response = await self._call_router(router, request)
            code = extract_python_code(_coerce_response_text(response))
            if not code:
                raise RuntimeError("LLM returned no Python source")

            ast.parse(code)
            logger.info(
                "Generated reconstruction candidate for %s (%d chars)",
                context.get("module_path", "<unknown>"),
                len(code),
            )
            return code
        except Exception as exc:
            record_degradation("llm_code_generator", exc)
            logger.warning("LLM code generation failed: %s", exc)
            if self.fallback_to_stub:
                return str(context.get("stub_code") or "# Generation failed\npass\n")
            raise

    def _resolve_router(self) -> Any:
        if self._router is not None:
            return self._router
        try:
            from core.container import ServiceContainer

            for service_name in self.service_names:
                service = ServiceContainer.get(service_name, default=None)
                if service is not None:
                    self._router = service
                    return service
        except Exception as exc:
            record_degradation("llm_code_generator", exc)
            logger.debug("Could not resolve LLM service for code generation: %s", exc)
        return None

    def _augment_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        module_path = context.get("module_path", "<unknown>")
        attempt = context.get("attempt", 1)
        return (
            f"{prompt}\n\n"
            "## Production Constraints\n"
            f"- Target module: {module_path}\n"
            f"- Reconstruction attempt: {attempt}\n"
            "- Generate a complete module, not a patch.\n"
            "- Use only the specification above and the public interface stub.\n"
            "- If behavior is underspecified, choose the safest deterministic implementation.\n"
            "- CRITICAL: If feedback from a previous attempt is provided, analyze the discrepancies and ensure the new implementation addresses the root causes.\n"
        )

    async def _call_router(self, router: Any, request: GenerationRequest) -> Any:
        kwargs: dict[str, Any] = {
            "system_prompt": request.system_prompt,
            "prefer_tier": request.prefer_tier,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "origin": request.origin,
            "is_background": request.is_background,
        }
        if self.prefer_endpoint:
            kwargs["prefer_endpoint"] = self.prefer_endpoint
        if request.prefer_tier.lower() in {"secondary", "deep", "api_deep", "local_deep"}:
            kwargs["deep_handoff"] = True
            kwargs["allow_deep_handoff"] = True

        for method_name in ("think", "generate", "call", "generate_text_async", "generate_text"):
            method = getattr(router, method_name, None)
            if not callable(method):
                continue
            call_kwargs = _filter_kwargs(method, kwargs)
            try:
                return await _maybe_await(method(request.prompt, **call_kwargs))
            except TypeError as exc:
                logger.debug("LLM method %s rejected kwargs: %s", method_name, exc)
                continue

        raise RuntimeError(f"{type(router).__name__} exposes no supported generation method")


__all__ = ["LLMCodeGenerator", "GenerationRequest", "extract_python_code"]
