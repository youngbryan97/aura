"""MLX applier for the expert-LoRA library — the live worker seam.

Bridges ``ExpertLoRALibrary`` residency to the resident worker model via
``MLXLocalClient.set_expert_adapter`` (an in-worker wrap of target linears +
adapter weight load; no model reload). The worker holds AT MOST ONE expert
adapter, so this applier serializes swaps and treats "load B" as the atomic
swap "detach A, attach B" the worker already implements.

Honesty contract: ``load``/``unload`` return what the worker actually did.
A busy lane (generation in flight) refuses the swap — callers try again at
the next idle moment; nothing queues silently against a model whose weights
must not change mid-decode.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from core.brain.expert_lora_library import LoRAAdapter

logger = logging.getLogger("Aura.ExpertAdapterApplier")


class MLXExpertAdapterApplier:
    """One-resident async applier over the MLX worker IPC."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._gate = asyncio.Lock()

    async def load(self, adapter: LoRAAdapter) -> bool:
        async with self._gate:
            receipt = await self._client.set_expert_adapter(adapter.path)
            ok = bool(receipt.get("ok"))
            if ok:
                logger.info(
                    "[ExpertLoRA] '%s' live on %s (%d layers).",
                    adapter.name,
                    getattr(self._client, "model_path", "?").rsplit("/", 1)[-1],
                    int(receipt.get("wrapped_layers") or 0),
                )
            else:
                logger.info(
                    "[ExpertLoRA] swap to '%s' refused: %s",
                    adapter.name,
                    receipt.get("reason"),
                )
            return ok

    async def unload(self, adapter: LoRAAdapter) -> bool:
        async with self._gate:
            resident = getattr(self._client, "expert_adapter_resident", None)
            if resident != adapter.path:
                return True  # worker already swapped past it — nothing to undo
            receipt = await self._client.set_expert_adapter(None)
            return bool(receipt.get("ok"))


__all__ = ["MLXExpertAdapterApplier"]
