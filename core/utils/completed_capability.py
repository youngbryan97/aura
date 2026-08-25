"""Runtime proof that a capability already completed for this turn.

Capability work crosses several owners before a reply is delivered. A route
may collect evidence, a cognitive phase may assemble it, and the inference
gate may offer tools. This receipt is the shared fact that prevents each
owner from doing the same work again.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from core.utils.injected_blocks import (
    is_stamped_runtime_payload,
    stamp_runtime_payload,
)

COMPLETED_CAPABILITY_SCHEMA = "aura.completed_capability_evidence.v1"


def _normalized_names(values: Iterable[Any]) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        values = (values,)
    return frozenset(
        name for value in values if (name := str(value or "").strip())
    )


def make_completed_capability_evidence(
    capabilities: Iterable[Any],
    *,
    ok: bool,
    **evidence: Any,
) -> dict[str, Any]:
    """Create same-process evidence for capability work that already ran."""

    payload = {
        "schema": COMPLETED_CAPABILITY_SCHEMA,
        "ok": bool(ok),
        "completed_capabilities": sorted(_normalized_names(capabilities)),
        **evidence,
    }
    return stamp_runtime_payload(payload)


def completed_capabilities(evidence: Any) -> frozenset[str]:
    """Return completed names only from a valid successful runtime receipt."""

    if not is_stamped_runtime_payload(evidence):
        return frozenset()
    if str(evidence.get("schema") or "") != COMPLETED_CAPABILITY_SCHEMA:
        return frozenset()
    if evidence.get("ok") is not True:
        return frozenset()
    values = evidence.get("completed_capabilities")
    if not isinstance(values, (list, tuple, set, frozenset)):
        return frozenset()
    return _normalized_names(values)


def remaining_capabilities(required: Iterable[Any], evidence: Any) -> list[str]:
    """Keep the caller's order while removing work proved complete."""

    completed = completed_capabilities(evidence)
    if isinstance(required, (str, bytes)):
        required = (required,)
    return [
        name
        for value in required
        if (name := str(value or "").strip()) and name not in completed
    ]


def any_capability_completed(evidence: Any, names: Iterable[Any]) -> bool:
    """Whether any name in a caller-defined capability family completed."""

    return bool(completed_capabilities(evidence) & _normalized_names(names))
