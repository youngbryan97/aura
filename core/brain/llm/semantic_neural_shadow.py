"""Privacy-bounded shadow comparison for qualified semantic neural answers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

SEMANTIC_NEURAL_SHADOW_SCHEMA: Final = "aura.semantic_neural_shadow.v1"
_FINAL_LINE = re.compile(r"^FINAL_ANSWER: (?P<payload>\{.*\})$", re.MULTILINE)
_MAX_TEXT_BYTES: Final = 32_768


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _answer_object(text: str) -> dict[str, str | int] | None:
    if not isinstance(text, str) or not text or len(text.encode("utf-8")) > _MAX_TEXT_BYTES:
        return None
    matches = list(_FINAL_LINE.finditer(text.strip()))
    if not matches:
        return None
    try:
        payload = json.loads(matches[-1].group("payload"))
    except (json.JSONDecodeError, RecursionError):
        return None
    if (
        not isinstance(payload, dict)
        or not payload
        or len(payload) > 16
        or any(
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, (str, int))
            for key, value in payload.items()
        )
    ):
        return None
    return dict(payload)


def build_semantic_shadow_comparison(
    *,
    objective: str,
    qualified_text: str,
    ordinary_text: str,
    admission_receipt: Mapping[str, Any],
    activation_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare final answer objects without retaining either answer or prompt."""

    if (
        not isinstance(objective, str)
        or not objective
        or not isinstance(qualified_text, str)
        or not isinstance(ordinary_text, str)
        or not isinstance(admission_receipt, Mapping)
        or not isinstance(activation_receipt, Mapping)
    ):
        raise ValueError("semantic shadow comparison input is invalid")
    qualified = _answer_object(qualified_text)
    ordinary = _answer_object(ordinary_text)
    if qualified is None:
        raise ValueError("qualified semantic shadow answer is not canonical")
    answer_match = ordinary == qualified
    body = {
        "schema": SEMANTIC_NEURAL_SHADOW_SCHEMA,
        "recorded_at": round(time.time(), 6),
        "objective_sha256": _text_sha(objective),
        "qualified_answer_sha256": _text_sha(qualified_text),
        "ordinary_answer_sha256": _text_sha(ordinary_text),
        "qualified_object_sha256": _sha(qualified),
        "ordinary_object_sha256": _sha(ordinary) if ordinary is not None else None,
        "ordinary_answer_parsed": ordinary is not None,
        "answer_match": answer_match,
        "qualified_gain_candidate": not answer_match,
        "ordinary_success_preserved": answer_match,
        "family": str(admission_receipt.get("family") or ""),
        "parser_id": str(admission_receipt.get("parser_id") or ""),
        "admission_receipt_sha256": str(admission_receipt.get("receipt_sha256") or ""),
        "activation_sha256": str(activation_receipt.get("activation_sha256") or ""),
        "package_id": str(activation_receipt.get("package_id") or ""),
        "promotion_mode": str(activation_receipt.get("promotion_mode") or ""),
        "raw_prompt_retained": False,
        "raw_answers_retained": False,
    }
    return {**body, "receipt_sha256": _sha(body)}


def _default_ledger_path() -> Path:
    return state_root() / "runtime" / "semantic_neural_shadow.jsonl"


async def record_semantic_shadow_comparison(
    *,
    objective: str,
    qualified_text: str,
    ordinary_text: str,
    admission_receipt: Mapping[str, Any],
    activation_receipt: Mapping[str, Any],
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Durably append one comparison off the event loop; failure never alters the reply."""

    comparison = build_semantic_shadow_comparison(
        objective=objective,
        qualified_text=qualified_text,
        ordinary_text=ordinary_text,
        admission_receipt=admission_receipt,
        activation_receipt=activation_receipt,
    )
    destination = Path(ledger_path) if ledger_path is not None else _default_ledger_path()
    line = json.dumps(comparison, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        # Through the gateway rather than straight to the atomic writer. This is
        # a durable ledger, so it is a consequential write and the gateway is
        # where consequential writes are governed and receipted; the async lane
        # keeps the fsync off the event loop, which is what the bare
        # ``to_thread`` was achieving on its own.
        from core.runtime.file_write_gateway import get_file_write_gateway

        await get_file_write_gateway().append_text_async(
            destination, line, source="latent_cortex.semantic_neural_shadow"
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "latent_cortex.semantic_neural_shadow_ledger",
            exc,
            action="retained the ordinary response after shadow telemetry persistence failed",
            severity="warning",
        )
        return {**comparison, "persisted": False, "persistence_error": type(exc).__name__}
    return {**comparison, "persisted": True}


__all__ = [
    "SEMANTIC_NEURAL_SHADOW_SCHEMA",
    "build_semantic_shadow_comparison",
    "record_semantic_shadow_comparison",
]
