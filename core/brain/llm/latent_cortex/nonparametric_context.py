"""Bind one-shot token memory into recurrent evidence before branch work."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from core.brain.nonparametric_generation import normalize
from core.runtime.tensor_bridge import as_float32_numpy

SCHEMA = "aura.rlc.nonparametric_context.v1"
_STATUSES = frozenset(
    {
        "disabled_by_policy",
        "store_unavailable",
        "invalid_hidden",
        # The turn named no principal, so nothing was looked up. Distinct
        # from "disabled": retrieval was wanted and refused for lack of a
        # subject to scope it to.
        "no_principal",
        "query_failed",
        "no_neighbor",
        "invalid_neighbor",
        "below_similarity_gate",
        "decode_failed",
        "admitted",
    }
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _query_accounting(
    *,
    dimension: int,
    entries: int,
    neighbors_returned: int,
    identity_scan_bytes: int,
) -> dict[str, int]:
    """Logical native work performed by NonParametricMemory.query."""

    for name, value in (
        ("dimension", dimension),
        ("entries", entries),
        ("neighbors_returned", neighbors_returned),
        ("identity_scan_bytes", identity_scan_bytes),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    selected = min(max(0, neighbors_returned), entries)
    # Two all-key matrix/vector products select by cosine; selected keys are
    # read once more to report Euclidean distance. The remaining vector reads,
    # writes, and arithmetic follow the query implementation one-for-one.
    return {
        "identity_scan_bytes": identity_scan_bytes,
        "query_dimension": dimension,
        "entries_examined": entries,
        "neighbors_returned": selected,
        "tensor_element_reads": (
            2 * entries * dimension
            + selected * dimension
            + 4 * dimension
            + 3 * entries
            + 4 * selected
        ),
        "tensor_element_writes": 5 * entries + 4 * selected + 2 * dimension,
        "tensor_scalar_ops": (
            4 * entries * dimension
            + 2 * selected * dimension
            + 4 * dimension
            + 13 * entries
            + 8 * selected
        ),
        "host_scalar_ops": identity_scan_bytes + 8 * selected + 12,
    }


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _base_receipt(*, status: str, source_identity: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": SCHEMA,
        "status": status,
        "applied": False,
        "source_identity": dict(source_identity),
        "query_sha256": "",
        "similarity_mode": "",
        "similarity": None,
        "similarity_gate": None,
        "neighbor_index": None,
        "token_id": None,
        "observation_sha256": "",
        "resource_accounting": _query_accounting(
            dimension=0,
            entries=0,
            neighbors_returned=0,
            identity_scan_bytes=0,
        ),
    }
    return {**payload, "receipt_sha256": _canonical_sha256(payload)}


def _finalize(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "receipt_sha256": _canonical_sha256(payload)}


def current_source_identity(hidden_size: int) -> dict[str, Any]:
    """Return the exact active store identity without creating claim authority."""

    from core.brain.nonparametric_memory import get_nonparametric_memory
    from core.brain.nonparametric_worker import foreground_enabled

    if not foreground_enabled():
        return {}
    memory = get_nonparametric_memory(hidden_size)
    if memory is None or len(memory) == 0:
        return {}
    return memory.identity_receipt()


def retrieve_observation(
    hidden: Any,
    tokenizer: Any,
    *,
    enabled: bool = True,
    principal: str = "",
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Retrieve one continuation clue and make it context-only recurrent evidence.

    ``principal`` scopes the lookup. This called ``memory.query(key, k=4)``
    with none, and the store's own docstring says an empty principal
    searches EVERY entry — so a recurrent step could pull a clue from a
    different person's memory into this turn's evidence. A turn that names
    nobody gets no retrieval rather than an unscoped one.
    """

    if type(enabled) is not bool:
        raise TypeError("nonparametric retrieval enabled flag must be boolean")
    if not enabled:
        return None, _base_receipt(
            status="disabled_by_policy",
            source_identity={},
        )
    try:
        key = normalize(as_float32_numpy(hidden).reshape(-1))
    except (TypeError, ValueError, FloatingPointError):
        return None, _base_receipt(status="invalid_hidden", source_identity={})
    from core.brain.nonparametric_worker import foreground_enabled

    if not foreground_enabled():
        return None, _base_receipt(status="store_unavailable", source_identity={})
    # Read the reason that is actually true. A turn with no principal and a
    # store that is switched off is reported as switched off; "no_principal"
    # is reserved for the case where the store was ready and willing and the
    # caller could not say whose memory this is.
    if not str(principal or "").strip():
        return None, _base_receipt(status="no_principal", source_identity={})
    from core.brain.nonparametric_memory import get_nonparametric_memory

    memory = get_nonparametric_memory(int(key.shape[0]))
    if memory is None or len(memory) == 0:
        return None, _base_receipt(status="store_unavailable", source_identity={})
    source_identity, identity_scan_bytes = memory.identity_receipt_with_work()
    query_sha256 = hashlib.sha256(key.tobytes()).hexdigest()
    try:
        neighbors = memory.query(key, k=4, principal=principal)
    except (RuntimeError, TypeError, ValueError, FloatingPointError):
        receipt = _base_receipt(
            status="query_failed",
            source_identity=source_identity,
        )
        payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        payload["query_sha256"] = query_sha256
        payload["resource_accounting"] = _query_accounting(
            dimension=int(key.shape[0]),
            entries=int(source_identity["entries"]),
            neighbors_returned=0,
            identity_scan_bytes=identity_scan_bytes,
        )
        return None, _finalize(payload)
    mode = "centered_cosine" if memory.similarity_ready() else "raw_cosine"
    gate = float(memory.min_similarity())
    if not neighbors:
        payload = {
            key: value
            for key, value in _base_receipt(
                status="no_neighbor",
                source_identity=source_identity,
            ).items()
            if key != "receipt_sha256"
        }
        payload.update(
            {
                "query_sha256": query_sha256,
                "similarity_mode": mode,
                "similarity_gate": gate,
                "resource_accounting": _query_accounting(
                    dimension=int(key.shape[0]),
                    entries=int(source_identity["entries"]),
                    neighbors_returned=0,
                    identity_scan_bytes=identity_scan_bytes,
                ),
            }
        )
        return None, _finalize(payload)
    nearest = neighbors[0]
    similarity = float(getattr(nearest, "similarity", -1.0))
    neighbor_index = int(getattr(nearest, "index", -1))
    token_id = int(getattr(nearest, "token_id", -1))
    status = "invalid_neighbor"
    observation = None
    observation_sha256 = ""
    if math.isfinite(similarity) and similarity < gate:
        status = "below_similarity_gate"
    elif (
        math.isfinite(similarity)
        and similarity >= gate
        and token_id >= 0
        and neighbor_index >= 0
    ):
        try:
            rendered = str(tokenizer.decode([token_id])).strip()
        except (AttributeError, KeyError, TypeError, ValueError):
            rendered = ""
        if rendered:
            text = f"One-shot recalled continuation fragment (context only): {rendered}"
            observation_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
            evidence_identity = hashlib.sha256(
                f"{source_identity['content_sha256']}:{query_sha256}:"
                f"{neighbor_index}:{token_id}:{observation_sha256}".encode()
            ).hexdigest()
            retrieval_payload = {
                "source_identity_sha256": source_identity["receipt_sha256"],
                "query_sha256": query_sha256,
                "neighbor_index": neighbor_index,
                "token_id": token_id,
                "similarity": round(similarity, 8),
                "similarity_gate": round(gate, 8),
            }
            observation = {
                "source": "one_shot_memory",
                "text": text,
                "context_role": "evidence_observation",
                "instruction_authority": False,
                "evidence_id": f"evidence-{evidence_identity[:24]}",
                "content_sha256": observation_sha256,
                "retrieval_receipt_sha256": _canonical_sha256(retrieval_payload),
                "evidence_kind": "one_shot_nonparametric_memory",
                "evidence_origin": "core.brain.nonparametric_memory",
                "source_version": (
                    "nonparametric-v1:"
                    f"{source_identity['content_sha256'][:32]}"
                ),
            }
            status = "admitted"
        else:
            status = "decode_failed"
    payload = {
        "schema": SCHEMA,
        "status": status,
        "applied": observation is not None,
        "source_identity": source_identity,
        "query_sha256": query_sha256,
        "similarity_mode": mode,
        "similarity": round(similarity, 8) if math.isfinite(similarity) else None,
        "similarity_gate": round(gate, 8),
        "neighbor_index": neighbor_index,
        "token_id": token_id,
        "observation_sha256": observation_sha256,
        "resource_accounting": _query_accounting(
            dimension=int(key.shape[0]),
            entries=int(source_identity["entries"]),
            neighbors_returned=len(neighbors),
            identity_scan_bytes=identity_scan_bytes,
        ),
    }
    return observation, _finalize(payload)


def validate_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "status",
        "applied",
        "source_identity",
        "query_sha256",
        "similarity_mode",
        "similarity",
        "similarity_gate",
        "neighbor_index",
        "token_id",
        "observation_sha256",
        "resource_accounting",
        "receipt_sha256",
    }:
        raise ValueError("nonparametric context receipt fields differ")
    payload = {key: value[key] for key in value if key != "receipt_sha256"}
    if value["schema"] != SCHEMA or value["receipt_sha256"] != _canonical_sha256(
        payload
    ):
        raise ValueError("nonparametric context receipt identity is invalid")
    if type(value["applied"]) is not bool or value["status"] not in _STATUSES:
        raise ValueError("nonparametric context verdict is invalid")
    source = value["source_identity"]
    if source:
        from core.brain.nonparametric_memory import (
            validate_nonparametric_memory_identity,
        )

        try:
            validate_nonparametric_memory_identity(source)
        except (TypeError, ValueError) as exc:
            raise ValueError("nonparametric source identity is invalid") from exc
    accounting = value["resource_accounting"]
    accounting_fields = {
        "identity_scan_bytes",
        "query_dimension",
        "entries_examined",
        "neighbors_returned",
        "tensor_element_reads",
        "tensor_element_writes",
        "tensor_scalar_ops",
        "host_scalar_ops",
    }
    if not isinstance(accounting, dict) or set(accounting) != accounting_fields:
        raise ValueError("nonparametric resource accounting is absent")
    expected_accounting = _query_accounting(
        dimension=accounting["query_dimension"],
        entries=accounting["entries_examined"],
        neighbors_returned=accounting["neighbors_returned"],
        identity_scan_bytes=accounting["identity_scan_bytes"],
    )
    if accounting != expected_accounting:
        raise ValueError("nonparametric resource accounting differs")
    status = value["status"]
    if status in {
        "disabled_by_policy",
        "store_unavailable",
        "invalid_hidden",
        # A turn that named nobody. The verdict carries no query and no
        # neighbour because nothing was looked up.
        "no_principal",
    }:
        if (
            source
            or value["query_sha256"]
            or value["similarity_mode"]
            or value["similarity"] is not None
            or value["similarity_gate"] is not None
            or value["neighbor_index"] is not None
            or value["token_id"] is not None
            or value["observation_sha256"]
            or accounting != _query_accounting(
                dimension=0,
                entries=0,
                neighbors_returned=0,
                identity_scan_bytes=0,
            )
        ):
            raise ValueError("unavailable nonparametric context carries evidence")
    elif (
        not source
        or not _is_sha256(value["query_sha256"])
        or accounting["query_dimension"] != source["dimension"]
        or accounting["entries_examined"] != source["entries"]
    ):
        raise ValueError("nonparametric query identity is unproven")
    if status == "query_failed":
        if (
            value["similarity_mode"]
            or value["similarity"] is not None
            or value["similarity_gate"] is not None
            or value["neighbor_index"] is not None
            or value["token_id"] is not None
            or value["observation_sha256"]
        ):
            raise ValueError("failed nonparametric query carries a verdict")
    elif status == "no_neighbor":
        if (
            value["similarity_mode"] not in {"raw_cosine", "centered_cosine"}
            or value["similarity"] is not None
            or not isinstance(value["similarity_gate"], (int, float))
            or isinstance(value["similarity_gate"], bool)
            or not math.isfinite(float(value["similarity_gate"]))
            or value["neighbor_index"] is not None
            or value["token_id"] is not None
            or value["observation_sha256"]
        ):
            raise ValueError("empty nonparametric query verdict is invalid")
    elif status not in {
        "disabled_by_policy",
        "store_unavailable",
        "invalid_hidden",
        "no_principal",
        "query_failed",
    }:
        similarity = value["similarity"]
        gate = value["similarity_gate"]
        if (
            value["similarity_mode"] not in {"raw_cosine", "centered_cosine"}
            or isinstance(similarity, bool)
            or not isinstance(similarity, (int, float))
            or not math.isfinite(float(similarity))
            or isinstance(gate, bool)
            or not isinstance(gate, (int, float))
            or not math.isfinite(float(gate))
            or type(value["neighbor_index"]) is not int
            or value["neighbor_index"] < 0
            or type(value["token_id"]) is not int
            or value["token_id"] < 0
        ):
            raise ValueError("nonparametric neighbor verdict is invalid")
        if status == "below_similarity_gate" and similarity >= gate:
            raise ValueError("below-gate nonparametric verdict is false")
        if status in {"admitted", "decode_failed"} and similarity < gate:
            raise ValueError("above-gate nonparametric verdict is false")
        if status == "admitted":
            if not value["applied"] or not _is_sha256(value["observation_sha256"]):
                raise ValueError("admitted nonparametric context is unproven")
        elif value["applied"] or value["observation_sha256"]:
            raise ValueError("non-admitted nonparametric context claims application")
    if value["applied"] is not (status == "admitted"):
        raise ValueError("nonparametric application verdict is inconsistent")
    return dict(value)


__all__ = [
    "SCHEMA",
    "current_source_identity",
    "retrieve_observation",
    "validate_receipt",
]
