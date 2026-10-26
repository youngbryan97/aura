"""Where her whole inner state stood when a memory was made, kept with it, and how near she stands to it now.

What was learned in a state comes back more readily in that state, and the
state is more than a feeling. In the context models of recall, every
experience is bound to the slowly drifting inner context it happened in, and
the present context is the cue (Tulving and Thomson, "Encoding specificity",
1973; Howard and Kahana's temporal context model, 2002; Talmi, Lohnas and Daw
for the emotional context, 2019). Her memories carried the emotions she felt
(core/memory/felt_at_encoding.py) and nothing else of her state.

Her unified field is that context: binding, interoception, the substrate, the
neural mesh and her chemistry are integrated there, and it drifts with all of
them. On seed 23 (26 October) a nudge towards feeling good moved the domain
the field is read in more than any other, a third of the shift between percept
classes and most of it in the field's own columns, and what she recalled
followed her internal shift at 0.28 where the structure ground asks 0.30.
Recall had no way to hear the field, and the same domain was the part of her
the carrier could cut away most cheaply.

So each memory is stamped with the field's state when it is written, and recall
is cued by how near her field stands to it now: the cosine between the two once
what every competing memory shares is taken out, which needs no scale of its own.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["CONTEXT_KEY", "context_now", "decode", "distinctive", "encode", "near", "stamp"]

#: The metadata key a memory's context travels under.
CONTEXT_KEY: str = "context"


def context_now() -> np.ndarray | None:
    """Her unified field's state now, or None where there is no field to read."""
    try:
        from core.container import ServiceContainer

        field = ServiceContainer.get("unified_field", default=None)
        if field is None or not hasattr(field, "get_field_state"):
            return None
        state = np.asarray(field.get_field_state(), dtype=np.float64).reshape(-1)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        from core.runtime.errors import record_degradation

        record_degradation("memory_context", exc, action="read no context for this memory or cue")
        return None
    if state.size == 0 or not np.all(np.isfinite(state)) or not np.any(state):
        return None
    return state


def encode(state: np.ndarray) -> str:
    """The state as text a memory store keeps: half precision, base64."""
    return base64.b64encode(np.asarray(state, dtype=np.float16).tobytes()).decode("ascii")


def decode(text: Any) -> np.ndarray | None:
    """The state back out of a stamp; anything unreadable reads as no context."""
    if not isinstance(text, str) or not text:
        return None
    try:
        state = np.frombuffer(base64.b64decode(text.encode("ascii"), validate=True), dtype=np.float16)
    except (ValueError, TypeError):
        return None
    state = state.astype(np.float64)
    return state if state.size and np.all(np.isfinite(state)) else None


def stamp(metadata: dict[str, Any], state: np.ndarray | None = None) -> dict[str, Any]:
    """Keep where her field stands now with the memory being written.

    A stamp already present is left alone, as a feeling stamp is: the writer
    that had the state in hand knows better than a later reading.
    """
    if not isinstance(metadata, dict) or metadata.get(CONTEXT_KEY):
        return metadata
    state = context_now() if state is None else state
    if state is not None:
        metadata[CONTEXT_KEY] = encode(state)
    return metadata


def distinctive(contexts: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """Each candidate's context less what the pool it competes in shares, and that share.

    What every candidate was made in says nothing about which of them this
    moment calls up, as `felt_at_encoding.distinctive` takes the common feeling
    out; the present state is measured from the same centre (`near`). A pool of
    one keeps its context whole and has no centre. Keyed as given; candidates
    with no readable stamp are left out.
    """
    decoded = {key: state for key, value in contexts.items() if (state := decode(value)) is not None}
    sizes = {state.size for state in decoded.values()}
    if len(sizes) != 1:
        # Stamps from fields of different sizes cannot be compared; keep the
        # largest group, which is the field this process runs.
        if not decoded:
            return {}, None
        size = max(sizes, key=lambda s: sum(1 for state in decoded.values() if state.size == s))
        decoded = {key: state for key, state in decoded.items() if state.size == size}
    if len(decoded) < 2:
        return decoded, None
    mean = np.mean(np.stack(list(decoded.values())), axis=0)
    return {key: state - mean for key, state in decoded.items()}, mean


def near(now: np.ndarray | None, then: np.ndarray | None, centre: np.ndarray | None = None) -> float:
    """How near her field stands to where it stood, from 0 to 1.

    The cosine of the two states, both measured from `centre` when one is
    given, with a state facing away counted as not near at all.
    """
    if now is None or then is None or now.size != then.size:
        return 0.0
    a = now - centre if centre is not None and centre.size == now.size else now
    b = then
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(a, b)) / norm))
