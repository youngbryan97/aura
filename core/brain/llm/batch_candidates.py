"""Batched candidate generation for verifier-selection reasoning.

Resolves the registry-assigned Cortex MLX client and decodes N sampled
candidates in one batched worker pass. Every failure degrades to None/[] so
callers keep their serial-sampling fallback — this module only ever makes
best-of-N cheaper, never a new failure mode.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from core.brain.generation_provenance import attributed_text

logger = logging.getLogger("Aura.BatchCandidates")


def _batching_enabled() -> bool:
    return str(os.environ.get("AURA_BATCHED_CANDIDATES", "1")).strip().lower() in {
        "1", "true", "yes", "on",
    }


def _resolve_primary_client() -> Any | None:
    try:
        from core.brain.llm.mlx_client import clients_snapshot
    except ImportError:
        return None
    # Snapshot first: iterating the live registry races registration.
    for _path, client in clients_snapshot():
        if not getattr(client, "is_alive", lambda: False)():
            continue
        assignment = getattr(client, "runtime_assignment", None)
        if getattr(assignment, "role", "") == "cortex":
            return client
    return None


async def generate_candidates_batched(
    prompt: str,
    n: int,
    *,
    max_tokens: int = 512,
    temperature: float = 0.8,
    timeout_s: float = 180.0,
) -> list[str] | None:
    """Return N raw candidates from one batched pass, or None to signal
    'use the serial path' (disabled, no live client, or failure)."""
    if not _batching_enabled() or n < 2:
        return None
    client = _resolve_primary_client()
    if client is None or not (
        hasattr(client, "generate_batch_with_metadata_async")
        or hasattr(client, "generate_batch_async")
    ):
        return None
    try:
        structured_generate = getattr(client, "generate_batch_with_metadata_async", None)
        if callable(structured_generate):
            result = await structured_generate(
                prompt,
                n=n,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
            )
            texts = list(result.get("texts") or []) if isinstance(result, dict) else []
            metadata = (
                dict(result.get("generation_metadata") or {})
                if isinstance(result, dict)
                else {}
            )
            candidate_metadata = (
                list(result.get("candidate_generation_metadata") or [])
                if isinstance(result, dict)
                else []
            )
            return [
                attributed_text(
                    text,
                    {
                        **metadata,
                        **(
                            dict(candidate_metadata[index])
                            if index < len(candidate_metadata)
                            and isinstance(candidate_metadata[index], dict)
                            else {}
                        ),
                        "batch_candidate_index": index,
                    },
                )
                for index, text in enumerate(texts)
                if str(text or "").strip()
            ] or None
        texts = await client.generate_batch_async(
            prompt,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )
    except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
        logger.debug("Batched candidate generation unavailable: %s", exc)
        return None
    return texts or None
