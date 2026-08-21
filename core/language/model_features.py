"""Sentences as the resident model represents them.

MEASURED, 2026-08-20. A topical sentence embedder was put behind every one of
this runtime's twenty-five declared matchers. Eight separated their own
examples and none by more than the spread inside their classes: zero usable
boundaries. That is what the feature space is for — an embedder is trained to
place "I saved it as report.csv" and "you could save it as report.csv" near
each other, and who acts and whether it is asserted is the axis these
decisions need.

The resident model does carry that axis; it is what lets it answer at all.
This reads it directly: one causal forward over the sentence, the last
token's hidden state, no sampling. There is no text to write and nothing to
steer, so it is a measurement rather than a prompt.

Off the critical path by construction. When the worker is absent or busy the
call returns nothing, and every caller treats nothing as "no opinion".
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable

from core.runtime.errors import record_degradation

__all__ = ["model_hidden_features"]

_RECOVERABLE = (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError)

#: One sentence per worker lease.  A bulk request performs one full 32B
#: forward per sentence while holding a single non-preemptible lane; measured
#: live, sixteen sentences crossed the deadline after Aura had already
#: advertised conversation readiness and forced a resident-worker recycle.
#: Releasing ownership after each sentence lets a newly arriving foreground
#: owner stop the batch mechanically.  Eight seconds is a per-forward fault
#: ceiling, not a scaled allowance for an uninterruptible batch.
_CHUNK = 1
_SECONDS_PER_SENTENCE = 8.0


def model_hidden_features(sentences: Iterable[str]) -> list[list[float]]:
    """Hidden-state vectors for these sentences, or [] when unavailable."""
    texts = [str(text or "") for text in sentences if str(text or "").strip()]
    if not texts:
        return []
    try:
        from core.brain.llm.mlx_client import get_mlx_client

        client = get_mlx_client()
        if client is None or not hasattr(client, "encode_hidden"):
            return []
        # Called from synchronous matcher code. A running loop means this is
        # already inside a turn, and blocking one to embed a sentence is the
        # trade this module exists to avoid.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # In chunks, because one request holding a hundred sentences is
            # a hundred forwards behind a single deadline: measured live, the
            # whole batch timed out and reported nothing rather than most of
            # it. The deadline scales with what is being asked for.
            vectors: list[list[float]] = []
            for start in range(0, len(texts), _CHUNK):
                chunk = texts[start : start + _CHUNK]
                got = asyncio.run(
                    client.encode_hidden(chunk, timeout_s=_SECONDS_PER_SENTENCE * len(chunk))
                )
                if not got or len(got) != len(chunk):
                    return []
                vectors.extend(got)
            if not vectors:
                logging.getLogger("Aura.LanguageFeatures").info(
                    "🔤 [FEATURES] the worker returned no vectors for %d sentence(s).",
                    len(texts),
                )
            return vectors
        logging.getLogger("Aura.LanguageFeatures").debug(
            "🔤 [FEATURES] declined: a loop is running, so this is inside a turn."
        )
        return []
    except _RECOVERABLE as exc:
        record_degradation(
            "language.model_features",
            exc,
            severity="debug",
            action="left the decision to the declared pattern",
            enforce_failure_policy=False,
        )
        return []
