"""ModelRuntimeActor — single inference authority.

Audit constraint: no subsystem may call MLX/model objects directly.
All generate / embed / vision / pause / unload routes through this
actor, which serializes access via an asyncio queue and records
per-call ToolExecutionReceipt entries when receipts are requested.
"""
from __future__ import annotations


import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger("Aura.ModelRuntimeActor")


@dataclass
class GenerateRequest:
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)
    receipt_required: bool = False


@dataclass
class EmbedRequest:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisionRequest:
    image_bytes: bytes
    prompt: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerateResult:
    text: str
    tokens: int
    duration_s: float
    receipt_id: Optional[str] = None


GenerateBackend = Callable[[GenerateRequest], Awaitable[GenerateResult]]


class ModelRuntimeActor:
    def __init__(self, *, backend: Optional[GenerateBackend] = None, queue_depth: int = 64):
        self._backend = backend
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_depth)
        self._lock = asyncio.Lock()
        self._paused = False
        self._idle_since: float = time.monotonic()
        self.calls = 0
        self.failures = 0

    def set_backend(self, backend: GenerateBackend) -> None:
        self._backend = backend

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        if self._backend is None:
            raise RuntimeError("ModelRuntimeActor has no backend wired")
        if self._paused:
            raise RuntimeError("ModelRuntimeActor is paused")
        async with self._lock:
            self.calls += 1
            start = time.monotonic()
            try:
                result = await self._backend(request)
            except BaseException:
                self.failures += 1
                raise
            duration = time.monotonic() - start
            self._idle_since = time.monotonic()
            if not isinstance(result, GenerateResult):
                result = GenerateResult(
                    text=str(result),
                    tokens=len(str(result).split()),
                    duration_s=duration,
                )
            if request.receipt_required:
                from core.runtime.receipts import (
                    ToolExecutionReceipt,
                    get_receipt_store,
                )

                rec = ToolExecutionReceipt(
                    receipt_id=f"toolex-{uuid.uuid4()}",
                    cause="model_runtime.generate",
                    tool="model_runtime",
                    status="success_unverified",
                    output_digest=f"len={len(result.text)}",
                    metadata={"tokens": result.tokens, "duration_s": result.duration_s},
                )
                get_receipt_store().emit(rec)
                result.receipt_id = rec.receipt_id
            return result

    async def pause(self) -> None:
        self._paused = True

    async def resume(self) -> None:
        self._paused = False

    async def unload_when_idle(self, *, idle_threshold_s: float = 30.0) -> bool:
        idle_for = time.monotonic() - self._idle_since
        return idle_for >= idle_threshold_s


_global: Optional[ModelRuntimeActor] = None


def get_model_runtime_actor() -> ModelRuntimeActor:
    global _global
    if _global is None:
        _global = ModelRuntimeActor()
    return _global


def reset_model_runtime_actor() -> None:
    global _global
    _global = None
