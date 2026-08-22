"""Bounded KV lineage, exact rewind, and rejected-slice isolation.

The recurrent cortex deliberately computes many speculative continuations
against one prompt cache.  A plain ``snapshot -> run -> restore`` discipline
can prove a single call rewound, but it cannot prove lineage across branch
savepoints, verifier rejection, regeneration, and final persistence.

``KVStateTree`` supplies that missing transaction boundary:

* every branch-visible boundary is linked to a parent commitment;
* speculative K/V writes are observed before they are removed;
* removal restores the exact immutable MLX array objects held by the parent;
* a rejected child commitment can never become a later parent;
* regeneration records the verified parent from which it restarted; and
* accepted final lanes become explicit terminal nodes.

The public receipt contains only salted commitments, offsets, and lineage.
It never serializes K/V tensors or private reasoning state.  Exactness is a
worker-side object-identity property: MLX arrays are immutable, and restoring
the same array objects makes rejected writes unreachable without copying a
resident model's potentially multi-gigabyte prompt cache to the host.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.brain.llm.latent_cortex.kv_mutation_transaction import KVMutationTransaction
from core.brain.llm.latent_cortex.recurrence import _cache_matches_snapshot
from core.brain.llm.recurrent_depth import (
    CacheSnapshotError,
    _cache_snapshot_commitment_parts,
    _restore_recurrent_caches,
    _snapshot_recurrent_caches,
)

KV_STATE_TREE_SCHEMA = "aura.rlc.kv_state_tree.v1"
KV_STATE_NODE_SCHEMA = "aura.rlc.kv_state_node.v1"
KV_STATE_EVENT_SCHEMA = "aura.rlc.kv_state_event.v1"

_MAX_LAYERS = 512
_MAX_NODES = 4096
_MAX_EVENTS = 16384
_MAX_LABEL_CHARS = 96
_MAX_PURPOSE_CHARS = 96
_SHA256_LEN = 64


class KVStateTreeError(RuntimeError):
    """The cache lineage contract could not be maintained."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LEN
        and all(char in "0123456789abcdef" for char in value)
    )


def _bounded_text(value: Any, *, field: str, limit: int) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise ValueError(f"{field} must be a non-empty bounded string")
    return value


def _branch_index(value: Any) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0 or value > 4095:
        raise ValueError("branch_index must be null or a bounded non-negative integer")
    return value


def _shape(value: Any) -> list[int] | None:
    raw = getattr(value, "shape", None)
    if raw is None:
        return None
    try:
        shape = [int(item) for item in raw]
    except (TypeError, ValueError) as exc:
        raise KVStateTreeError("cache tensor shape is not integral") from exc
    if any(item < 0 for item in shape):
        raise KVStateTreeError("cache tensor shape contains a negative dimension")
    return shape


def _private_value_descriptor(value: Any, *, salt: bytes) -> Any:
    """Describe storage identity without copying tensor content to the host."""

    shape = _shape(value)
    if shape is not None and hasattr(value, "dtype"):
        storage_token = hashlib.sha256(
            salt
            + b":tensor:"
            + str(id(value)).encode("ascii")
            + b":"
            + str(type(value).__module__).encode("utf-8")
            + b":"
            + str(type(value).__qualname__).encode("utf-8")
        ).hexdigest()
        return {
            "kind": "immutable_tensor",
            "shape": shape,
            "dtype": str(value.dtype),
            "storage_token": storage_token,
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_private_value_descriptor(item, salt=salt) for item in value],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [_private_value_descriptor(item, salt=salt) for item in value],
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise KVStateTreeError("cache metadata mappings must use string keys")
        return {
            "kind": "mapping",
            "items": {
                key: _private_value_descriptor(value[key], salt=salt) for key in sorted(value)
            },
        }
    if isinstance(value, (str, int, float, bool, type(None))):
        return {"kind": type(value).__name__, "value": value}
    raise KVStateTreeError(
        f"unsupported cache metadata value: {type(value).__module__}.{type(value).__qualname__}"
    )


def _snapshot_commitment(snapshots: list, *, salt: bytes) -> str:
    rows: list[Any] = []
    for snapshot in snapshots:
        if snapshot is None:
            rows.append(None)
            continue
        if not isinstance(snapshot, tuple) or not snapshot:
            raise KVStateTreeError("cache snapshot entry is malformed")
        try:
            kind, state_value, metadata_value = _cache_snapshot_commitment_parts(
                snapshot
            )
        except CacheSnapshotError as exc:
            raise KVStateTreeError(str(exc)) from exc
        rows.append(
            {
                "kind": kind,
                "state": _private_value_descriptor(state_value, salt=salt),
                "metadata": _private_value_descriptor(
                    metadata_value,
                    salt=salt,
                ),
            }
        )
    return _canonical_sha256(rows)


def _cache_offsets(cache: Sequence[Any], start: int, end: int) -> list[int]:
    if not 0 <= start < end <= len(cache):
        raise KVStateTreeError("cache offset range is invalid")
    offsets: list[int] = []
    for index, item in enumerate(cache[start:end], start=start):
        if item is None:
            offsets.append(0)
            continue
        # BatchKVCache has per-example offsets because left padding differs,
        # but its scalar _idx is the shared logical K/V cursor. Transactions
        # need one cursor per layer so layer-window escape checks remain exact.
        offset = getattr(item, "_idx", None)
        if type(offset) is not int:
            offset = getattr(item, "offset", None)
        if type(offset) is not int or offset < 0:
            raise KVStateTreeError(f"cache layer {index} has an invalid offset")
        offsets.append(offset)
    return offsets


def _offsets_sha256(offsets: Sequence[int]) -> str:
    return _canonical_sha256(list(offsets))


def _node_payload(
    *,
    ordinal: int,
    parent_sha256: str,
    cache_sha256: str,
    offsets_sha256: str,
    min_offset: int,
    max_offset: int,
    branch_index: int | None,
    label: str,
    authority: str,
    verified: bool,
    final: bool,
    latent_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": KV_STATE_NODE_SCHEMA,
        "ordinal": ordinal,
        "parent_sha256": parent_sha256,
        "cache_sha256": cache_sha256,
        "offsets_sha256": offsets_sha256,
        "min_offset": min_offset,
        "max_offset": max_offset,
        "branch_index": branch_index,
        "label": label,
        "authority": authority,
        "verified": verified,
        "final": final,
        "latent_sha256": latent_sha256,
    }


@dataclass
class _NodeRuntime:
    receipt: dict[str, Any]
    snapshots: list


class KVStateTree:
    """In-memory exact snapshots plus a bounded public lineage receipt."""

    def __init__(
        self,
        cache: Sequence[Any],
        *,
        n_layers: int,
        episode_id: str,
        input_tokens_sha256: str,
        max_nodes: int = _MAX_NODES,
        max_events: int = _MAX_EVENTS,
    ) -> None:
        if type(n_layers) is not int or not 1 <= n_layers <= _MAX_LAYERS:
            raise ValueError("n_layers is outside the KV state-tree bound")
        if not isinstance(cache, (list, tuple)) or len(cache) != n_layers:
            raise ValueError("cache must cover the complete model layer set")
        _bounded_text(episode_id, field="episode_id", limit=160)
        if not _is_sha256(input_tokens_sha256):
            raise ValueError("input_tokens_sha256 is invalid")
        if type(max_nodes) is not int or not 2 <= max_nodes <= _MAX_NODES:
            raise ValueError("max_nodes is outside the supported bound")
        if type(max_events) is not int or not 1 <= max_events <= _MAX_EVENTS:
            raise ValueError("max_events is outside the supported bound")
        self.n_layers = n_layers
        self.episode_id = episode_id
        self.input_tokens_sha256 = input_tokens_sha256
        self.max_nodes = max_nodes
        self.max_events = max_events
        self._salt = secrets.token_bytes(32)
        self._nodes: list[_NodeRuntime] = []
        self._nodes_by_sha256: dict[str, _NodeRuntime] = {}
        self._node_offsets_by_sha256: dict[str, list[int]] = {}
        self._events: list[dict[str, Any]] = []
        # (parent node, branch index, child content hash). A content hash is
        # not an identity: a deterministic decode reaches the same cache
        # contents on every branch, so keeping only the hash made every
        # content-identical boundary look like the resurrection of a pruned
        # one and degraded the episode to a vanilla decode. What must not
        # happen is a mutation REJECTED UNDER A PARENT being handed back under
        # that same parent, and that is a triple.
        self._rejected_child_commitments: set[tuple[str, int | None, str]] = set()
        self._restore_failures = 0

        root = self._capture_node(
            cache,
            parent_sha256="",
            branch_index=None,
            label="prefill_root",
            authority="prompt_prefill",
            verified=True,
            final=False,
            latent_sha256="",
        )
        self.root_sha256 = root
        # Reapply the retained boundary and verify both immutable K/V storage
        # identity and logical cursor metadata before accepting the root.
        self.restore_boundary(cache, root)

    def _node(self, node_sha256: str) -> _NodeRuntime:
        try:
            return self._nodes_by_sha256[node_sha256]
        except KeyError as exc:
            raise KVStateTreeError("KV state-tree parent is unknown") from exc

    def _node_offsets(self, node_sha256: str) -> list[int]:
        try:
            return list(self._node_offsets_by_sha256[node_sha256])
        except KeyError as exc:
            raise KVStateTreeError("KV state-tree parent offsets are unavailable") from exc

    def _capture_node(
        self,
        cache: Sequence[Any],
        *,
        parent_sha256: str,
        branch_index: int | None,
        label: str,
        authority: str,
        verified: bool,
        final: bool,
        latent_sha256: str,
        snapshots: list | None = None,
    ) -> str:
        if len(self._nodes) >= self.max_nodes:
            raise KVStateTreeError("KV state-tree node bound exhausted")
        branch_index = _branch_index(branch_index)
        label = _bounded_text(label, field="label", limit=_MAX_LABEL_CHARS)
        authority = _bounded_text(
            authority,
            field="authority",
            limit=_MAX_LABEL_CHARS,
        )
        if type(verified) is not bool or type(final) is not bool:
            raise ValueError("KV node verified/final flags must be booleans")
        if latent_sha256 and not _is_sha256(latent_sha256):
            raise ValueError("latent_sha256 is invalid")
        if parent_sha256:
            self._node(parent_sha256)
        elif self._nodes:
            raise KVStateTreeError("only the root KV node may omit a parent")
        captured = (
            snapshots
            if snapshots is not None
            else _snapshot_recurrent_caches(cache, 0, self.n_layers)
        )
        cache_sha256 = _snapshot_commitment(captured, salt=self._salt)
        # Reuse means handing back a snapshot that was rejected, not reaching
        # the same content again.
        #
        # A commitment is a content hash and a hash is not an identity. When
        # `snapshots` is None the cache is captured live, so identical content
        # means the model produced it again — which happens the moment two
        # branches decode the same tokens, and is the whole point of a
        # deterministic tie. Refusing that made every content-identical
        # boundary look like the resurrection of a pruned one, and the episode
        # degraded to a vanilla decode.
        #
        # Replay is what is worth refusing, and replay is exactly
        # `snapshots is not None` — a stored snapshot handed back. The lineage
        # is checked with it, so the refusal is "this mutation, rejected under
        # this parent on this branch, is being handed back", which is a
        # statement about an event rather than about bytes.
        #
        # The stronger protection is elsewhere and unchanged: rejecting a
        # transaction restores the parent and verifies its commitment, so a
        # rejected child's state cannot survive in the cache at all.
        if (
            snapshots is not None
            and (parent_sha256, branch_index, cache_sha256)
            in self._rejected_child_commitments
        ):
            raise KVStateTreeError("a rejected KV child was reused as a live boundary")
        offsets = _cache_offsets(cache, 0, self.n_layers)
        payload = _node_payload(
            ordinal=len(self._nodes),
            parent_sha256=parent_sha256,
            cache_sha256=cache_sha256,
            offsets_sha256=_offsets_sha256(offsets),
            min_offset=min(offsets),
            max_offset=max(offsets),
            branch_index=branch_index,
            label=label,
            authority=authority,
            verified=verified,
            final=final,
            latent_sha256=latent_sha256,
        )
        node_sha256 = _canonical_sha256(payload)
        receipt = {**payload, "node_sha256": node_sha256}
        runtime = _NodeRuntime(receipt=receipt, snapshots=captured)
        self._nodes.append(runtime)
        self._nodes_by_sha256[node_sha256] = runtime
        self._node_offsets_by_sha256[node_sha256] = offsets
        return node_sha256

    def capture_boundary(
        self,
        cache: Sequence[Any],
        *,
        parent_sha256: str,
        branch_index: int | None,
        label: str,
        authority: str,
        verified: bool,
        latent_sha256: str,
        final: bool = False,
    ) -> str:
        """Capture a logical savepoint against the exact current cache."""

        parent = self._node(parent_sha256)
        if not _cache_matches_snapshot(
            cache,
            0,
            self.n_layers,
            parent.snapshots,
        ):
            raise KVStateTreeError("KV boundary does not descend from its declared parent")
        # This is a logical branch/checkpoint boundary, not a K/V mutation.
        # Reuse the parent's exact retained storage rather than materializing
        # a new cropped view that would be value-equal but identity-distinct.
        return self._capture_node(
            cache,
            parent_sha256=parent_sha256,
            branch_index=branch_index,
            label=label,
            authority=authority,
            verified=verified,
            final=final,
            latent_sha256=latent_sha256,
            snapshots=parent.snapshots,
        )

    def restore_boundary(self, cache: Sequence[Any], node_sha256: str) -> None:
        """Restore and verify one complete cache boundary."""

        node = self._node(node_sha256)
        _restore_recurrent_caches(
            cache,
            0,
            self.n_layers,
            node.snapshots,
        )
        if not _cache_matches_snapshot(
            cache,
            0,
            self.n_layers,
            node.snapshots,
        ):
            self._restore_failures += 1
            raise KVStateTreeError("exact KV boundary restoration failed")

    def begin_speculation(
        self,
        cache: Sequence[Any],
        *,
        start: int,
        end: int,
        purpose: str,
        branch_index: int | None,
        parent_sha256: str,
        isolated: bool = False,
    ) -> KVMutationTransaction:
        if len(self._events) >= self.max_events:
            raise KVStateTreeError("KV state-tree event bound exhausted")
        if not 0 <= start < end <= self.n_layers:
            raise ValueError("KV transaction range is invalid")
        purpose = _bounded_text(
            purpose,
            field="purpose",
            limit=_MAX_PURPOSE_CHARS,
        )
        branch_index = _branch_index(branch_index)
        parent = self._node(parent_sha256)
        if not _cache_matches_snapshot(
            cache,
            0,
            self.n_layers,
            parent.snapshots,
        ):
            raise KVStateTreeError(
                "speculative KV transaction did not start at its declared parent"
            )
        rejected_here = (
            parent.receipt.get("parent_sha256", ""),
            parent.receipt.get("branch_index"),
            parent.receipt["cache_sha256"],
        )
        if rejected_here in self._rejected_child_commitments:
            raise KVStateTreeError("a rejected KV child became a transaction parent")
        # Opening a fresh transaction on this parent and branch retires the
        # earlier rejection for it. The rejection means "that mutation was
        # discarded and its state must not be handed back"; it cannot mean
        # "this branch may never reach that content again", because a
        # deterministic decode reaches it every time the branch is re-run, and
        # the re-run is a new mutation the transaction will judge on its own.
        self._rejected_child_commitments = {
            entry
            for entry in self._rejected_child_commitments
            if entry[:2] != (parent_sha256, branch_index)
        }
        return KVMutationTransaction(
            self,
            cache=cache,
            start=start,
            end=end,
            purpose=purpose,
            branch_index=branch_index,
            parent_sha256=parent_sha256,
            isolated=isolated,
        )

    def _event_payload(
        self,
        transaction: KVMutationTransaction,
        *,
        disposition: str,
        parent_restored: bool,
        pruned: bool,
        restored_cache_sha256: str,
        restored_offsets_sha256: str,
        result_node_sha256: str,
    ) -> dict[str, Any]:
        parent = self._node(transaction.parent_sha256).receipt
        return {
            "schema": KV_STATE_EVENT_SCHEMA,
            "ordinal": len(self._events),
            "parent_node_sha256": transaction.parent_sha256,
            "parent_cache_sha256": parent["cache_sha256"],
            "parent_offsets_sha256": parent["offsets_sha256"],
            "child_cache_sha256": transaction._child_cache_sha256,
            "child_offsets_sha256": transaction._child_offsets_sha256,
            "restored_cache_sha256": restored_cache_sha256,
            "restored_offsets_sha256": restored_offsets_sha256,
            "result_node_sha256": result_node_sha256,
            "purpose": transaction.purpose,
            "branch_index": transaction.branch_index,
            "start_layer": transaction.start,
            "end_layer": transaction.end,
            "mutation_observed": transaction._mutation_observed,
            "appended_tokens_min": transaction._appended_min,
            "appended_tokens_max": transaction._appended_max,
            "execution_failed": transaction._execution_failed,
            "isolated_cache": transaction.isolated,
            "disposition": disposition,
            "parent_restored": parent_restored,
            "pruned": pruned,
            "rejected_child_reused": False,
        }

    def _append_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        event_sha256 = _canonical_sha256(payload)
        event = {**payload, "event_sha256": event_sha256}
        self._events.append(event)
        return event

    @staticmethod
    def _ensure_mutation_observed(transaction: KVMutationTransaction) -> None:
        if transaction._child_snapshots is None:
            raise KVStateTreeError("KV transaction closed before observing its child")

    def _close_rejected(
        self,
        transaction: KVMutationTransaction,
        *,
        parent_cache: Sequence[Any],
        isolated: bool,
    ) -> dict[str, Any]:
        self._ensure_mutation_observed(transaction)
        if transaction._closed:
            raise KVStateTreeError("KV transaction was already closed")
        parent = self._node(transaction.parent_sha256)
        exact = _cache_matches_snapshot(
            parent_cache,
            0,
            self.n_layers,
            parent.snapshots,
        )
        if not exact:
            self._restore_failures += 1
            raise KVStateTreeError("rejected KV child did not restore its exact parent")
        restored_snapshots = _snapshot_recurrent_caches(
            parent_cache,
            0,
            self.n_layers,
        )
        restored_cache_sha256 = _snapshot_commitment(
            restored_snapshots,
            salt=self._salt,
        )
        restored_offsets = _cache_offsets(parent_cache, 0, self.n_layers)
        if restored_cache_sha256 != parent.receipt["cache_sha256"]:
            self._restore_failures += 1
            raise KVStateTreeError("restored KV parent commitment changed")
        if transaction._mutation_observed:
            if transaction._child_cache_sha256 == parent.receipt["cache_sha256"]:
                raise KVStateTreeError("mutated KV child aliases its parent commitment")
            self._rejected_child_commitments.add(
                (
                    transaction.parent_sha256,
                    transaction.branch_index,
                    transaction._child_cache_sha256,
                )
            )
        payload = self._event_payload(
            transaction,
            disposition=("isolated_discarded" if isolated else "rejected_pruned"),
            parent_restored=True,
            pruned=True,
            restored_cache_sha256=restored_cache_sha256,
            restored_offsets_sha256=_offsets_sha256(restored_offsets),
            result_node_sha256="",
        )
        transaction._closed = True
        return self._append_event(payload)

    def _commit_transaction(
        self,
        transaction: KVMutationTransaction,
        *,
        label: str,
        authority: str,
        latent_sha256: str,
        final: bool,
    ) -> str:
        self._ensure_mutation_observed(transaction)
        if transaction._closed:
            raise KVStateTreeError("KV transaction was already closed")
        if transaction._execution_failed:
            raise KVStateTreeError("failed KV transaction cannot be committed")
        _restore_recurrent_caches(
            transaction._cache,
            0,
            self.n_layers,
            transaction._child_snapshots,
        )
        if not _cache_matches_snapshot(
            transaction._cache,
            0,
            self.n_layers,
            transaction._child_snapshots,
        ):
            self._restore_failures += 1
            raise KVStateTreeError("committed KV child could not be normalized")
        node_sha256 = self._capture_node(
            transaction._cache,
            parent_sha256=transaction.parent_sha256,
            branch_index=transaction.branch_index,
            label=label,
            authority=authority,
            verified=True,
            final=final,
            latent_sha256=latent_sha256,
            snapshots=transaction._child_snapshots,
        )
        payload = self._event_payload(
            transaction,
            disposition="committed",
            parent_restored=False,
            pruned=False,
            restored_cache_sha256="",
            restored_offsets_sha256="",
            result_node_sha256=node_sha256,
        )
        transaction._closed = True
        self._append_event(payload)
        return node_sha256

    def receipt(self) -> dict[str, Any]:
        nodes = [dict(runtime.receipt) for runtime in self._nodes]
        events = [dict(event) for event in self._events]
        rejected = [
            event
            for event in events
            if event["disposition"] in {"rejected_pruned", "isolated_discarded"}
        ]
        committed = [event for event in events if event["disposition"] == "committed"]
        body = {
            "schema": KV_STATE_TREE_SCHEMA,
            "episode_id": self.episode_id,
            "input_tokens_sha256": self.input_tokens_sha256,
            "n_layers": self.n_layers,
            "root_node_sha256": self.root_sha256,
            "node_count": len(nodes),
            "event_count": len(events),
            "verified_boundary_count": sum(node["verified"] for node in nodes),
            "final_node_count": sum(node["final"] for node in nodes),
            "rejected_event_count": len(rejected),
            "committed_event_count": len(committed),
            "regeneration_count": sum(
                event["purpose"] == "regenerate_from_prefix" for event in events
            ),
            "restore_failure_count": self._restore_failures,
            "all_rejected_slices_pruned": bool(rejected)
            and all(
                event["pruned"] and event["parent_restored"] and not event["rejected_child_reused"]
                for event in rejected
            ),
            "exact_parent_restoration": self._restore_failures == 0
            and all(
                event["restored_cache_sha256"] == event["parent_cache_sha256"]
                and event["restored_offsets_sha256"] == event["parent_offsets_sha256"]
                for event in rejected
            ),
            "no_rejected_child_reused": not any(event["rejected_child_reused"] for event in events),
            "nodes": nodes,
            "events": events,
        }
        return {**body, "receipt_sha256": _canonical_sha256(body)}


def validate_kv_state_tree_receipt(
    value: Any,
    *,
    episode_id: str,
    input_tokens_sha256: str,
    n_layers: int,
    expected_n_branches: int,
    require_final: bool,
) -> dict[str, Any]:
    """Independently reconstruct and validate the public lineage receipt."""

    if not isinstance(value, dict):
        raise ValueError("KV state-tree receipt must be a mapping")
    required = {
        "schema",
        "episode_id",
        "input_tokens_sha256",
        "n_layers",
        "root_node_sha256",
        "node_count",
        "event_count",
        "verified_boundary_count",
        "final_node_count",
        "rejected_event_count",
        "committed_event_count",
        "regeneration_count",
        "restore_failure_count",
        "all_rejected_slices_pruned",
        "exact_parent_restoration",
        "no_rejected_child_reused",
        "nodes",
        "events",
        "receipt_sha256",
    }
    if set(value) != required:
        raise ValueError("KV state-tree receipt fields differ")
    if (
        value["schema"] != KV_STATE_TREE_SCHEMA
        or value["episode_id"] != episode_id
        or value["input_tokens_sha256"] != input_tokens_sha256
        or value["n_layers"] != n_layers
    ):
        raise ValueError("KV state-tree identity binding differs")
    if (
        type(n_layers) is not int
        or not 1 <= n_layers <= _MAX_LAYERS
        or type(expected_n_branches) is not int
        or not 1 <= expected_n_branches <= 4096
    ):
        raise ValueError("KV state-tree expected topology is invalid")
    nodes = value["nodes"]
    events = value["events"]
    if (
        not isinstance(nodes, list)
        or not 1 <= len(nodes) <= _MAX_NODES
        or not isinstance(events, list)
        or not 1 <= len(events) <= _MAX_EVENTS
    ):
        raise ValueError("KV state-tree trace cardinality is invalid")
    if value["node_count"] != len(nodes) or value["event_count"] != len(events):
        raise ValueError("KV state-tree trace counts differ")

    node_keys = set(
        _node_payload(
            ordinal=0,
            parent_sha256="",
            cache_sha256="0" * 64,
            offsets_sha256="0" * 64,
            min_offset=0,
            max_offset=0,
            branch_index=None,
            label="x",
            authority="x",
            verified=True,
            final=False,
            latent_sha256="",
        )
    ) | {"node_sha256"}
    by_node: dict[str, dict[str, Any]] = {}
    for ordinal, raw in enumerate(nodes):
        if not isinstance(raw, dict) or set(raw) != node_keys:
            raise ValueError("KV state-tree node fields differ")
        if raw["schema"] != KV_STATE_NODE_SCHEMA or raw["ordinal"] != ordinal:
            raise ValueError("KV state-tree node ordering differs")
        for key in ("cache_sha256", "offsets_sha256", "node_sha256"):
            if not _is_sha256(raw[key]):
                raise ValueError("KV state-tree node commitment is invalid")
        parent = raw["parent_sha256"]
        if ordinal == 0:
            if parent or raw["branch_index"] is not None:
                raise ValueError("KV state-tree root is malformed")
        elif not _is_sha256(parent) or parent not in by_node:
            raise ValueError("KV state-tree node parent is not prior")
        branch = _branch_index(raw["branch_index"])
        if branch is not None and branch >= expected_n_branches:
            raise ValueError("KV state-tree node branch is outside topology")
        _bounded_text(raw["label"], field="label", limit=_MAX_LABEL_CHARS)
        _bounded_text(
            raw["authority"],
            field="authority",
            limit=_MAX_LABEL_CHARS,
        )
        if type(raw["verified"]) is not bool or type(raw["final"]) is not bool:
            raise ValueError("KV state-tree node flags are invalid")
        if raw["final"] and raw["verified"] is not True:
            raise ValueError("KV state-tree final node is not verified")
        if raw["latent_sha256"] and not _is_sha256(raw["latent_sha256"]):
            raise ValueError("KV state-tree latent commitment is invalid")
        if (
            type(raw["min_offset"]) is not int
            or type(raw["max_offset"]) is not int
            or not 0 <= raw["min_offset"] <= raw["max_offset"]
        ):
            raise ValueError("KV state-tree node offsets are invalid")
        payload = {key: raw[key] for key in raw if key != "node_sha256"}
        if _canonical_sha256(payload) != raw["node_sha256"]:
            raise ValueError("KV state-tree node hash differs")
        if raw["node_sha256"] in by_node:
            raise ValueError("KV state-tree node hash is duplicated")
        by_node[raw["node_sha256"]] = raw
    if (
        nodes[0]["node_sha256"] != value["root_node_sha256"]
        or nodes[0]["label"] != "prefill_root"
        or nodes[0]["authority"] != "prompt_prefill"
        or nodes[0]["verified"] is not True
        or nodes[0]["final"] is not False
    ):
        raise ValueError("KV state-tree root binding differs")

    event_template = {
        "schema",
        "ordinal",
        "parent_node_sha256",
        "parent_cache_sha256",
        "parent_offsets_sha256",
        "child_cache_sha256",
        "child_offsets_sha256",
        "restored_cache_sha256",
        "restored_offsets_sha256",
        "result_node_sha256",
        "purpose",
        "branch_index",
        "start_layer",
        "end_layer",
        "mutation_observed",
        "appended_tokens_min",
        "appended_tokens_max",
        "execution_failed",
        "isolated_cache",
        "disposition",
        "parent_restored",
        "pruned",
        "rejected_child_reused",
        "event_sha256",
    }
    # (parent node, branch index, child content hash). Keyed by lineage, not
    # by content alone: a deterministic decode reaches the same cache contents
    # every time a branch is re-run, so a bare hash set marked every
    # content-identical live node as the resurrection of a pruned one.
    rejected_children: set[tuple[str, int | None, str]] = set()
    committed_results: set[str] = set()
    rejected_count = 0
    committed_count = 0
    regeneration_count = 0
    for ordinal, raw in enumerate(events):
        if not isinstance(raw, dict) or set(raw) != event_template:
            raise ValueError("KV state-tree event fields differ")
        if raw["schema"] != KV_STATE_EVENT_SCHEMA or raw["ordinal"] != ordinal:
            raise ValueError("KV state-tree event ordering differs")
        parent_sha256 = raw["parent_node_sha256"]
        if parent_sha256 not in by_node:
            raise ValueError("KV state-tree event parent is unknown")
        parent = by_node[parent_sha256]
        if (
            raw["parent_cache_sha256"] != parent["cache_sha256"]
            or raw["parent_offsets_sha256"] != parent["offsets_sha256"]
        ):
            raise ValueError("KV state-tree event parent binding differs")
        for key in ("child_cache_sha256", "child_offsets_sha256", "event_sha256"):
            if not _is_sha256(raw[key]):
                raise ValueError("KV state-tree event commitment is invalid")
        _bounded_text(
            raw["purpose"],
            field="purpose",
            limit=_MAX_PURPOSE_CHARS,
        )
        branch = _branch_index(raw["branch_index"])
        if branch is not None and branch >= expected_n_branches:
            raise ValueError("KV state-tree event branch is outside topology")
        if (
            type(raw["start_layer"]) is not int
            or type(raw["end_layer"]) is not int
            or not 0 <= raw["start_layer"] < raw["end_layer"] <= n_layers
        ):
            raise ValueError("KV state-tree event window is invalid")
        for key in (
            "mutation_observed",
            "execution_failed",
            "isolated_cache",
            "parent_restored",
            "pruned",
            "rejected_child_reused",
        ):
            if type(raw[key]) is not bool:
                raise ValueError("KV state-tree event flag is invalid")
        if (
            type(raw["appended_tokens_min"]) is not int
            or type(raw["appended_tokens_max"]) is not int
            or not 0 <= raw["appended_tokens_min"] <= raw["appended_tokens_max"]
        ):
            raise ValueError("KV state-tree event append span is invalid")
        if raw["mutation_observed"] is (raw["appended_tokens_max"] == 0):
            raise ValueError("KV state-tree mutation evidence is inconsistent")
        disposition = raw["disposition"]
        if disposition in {"rejected_pruned", "isolated_discarded"}:
            rejected_count += 1
            if (
                raw["isolated_cache"] is (disposition == "rejected_pruned")
                or raw["parent_restored"] is not True
                or raw["pruned"] is not True
                or raw["rejected_child_reused"] is not False
                or raw["restored_cache_sha256"] != raw["parent_cache_sha256"]
                or raw["restored_offsets_sha256"] != raw["parent_offsets_sha256"]
                or raw["result_node_sha256"]
            ):
                raise ValueError("rejected KV child was not exactly pruned")
            if raw["mutation_observed"]:
                rejected_children.add(
                    (parent_sha256, raw["branch_index"], raw["child_cache_sha256"])
                )
        elif disposition == "committed":
            committed_count += 1
            result_node = raw["result_node_sha256"]
            if (
                raw["execution_failed"]
                or raw["parent_restored"]
                or raw["pruned"]
                or raw["restored_cache_sha256"]
                or raw["restored_offsets_sha256"]
                or result_node not in by_node
                or by_node[result_node]["parent_sha256"] != parent_sha256
                or by_node[result_node]["cache_sha256"] != raw["child_cache_sha256"]
                or by_node[result_node]["offsets_sha256"] != raw["child_offsets_sha256"]
                or by_node[result_node]["branch_index"] != raw["branch_index"]
            ):
                raise ValueError("committed KV child binding differs")
            committed_results.add(result_node)
        else:
            raise ValueError("KV state-tree event disposition is invalid")
        if raw["purpose"] == "regenerate_from_prefix":
            regeneration_count += 1
        payload = {key: raw[key] for key in raw if key != "event_sha256"}
        if _canonical_sha256(payload) != raw["event_sha256"]:
            raise ValueError("KV state-tree event hash differs")
    for node in nodes:
        # A node that a committed transaction produced is not a resurrection,
        # whatever it hashes to. Re-running a branch after a rejection reaches
        # the same parent, the same branch index and — with a deterministic
        # decode — the same contents, so lineage alone still cannot tell the
        # retry from the reuse. The commit event can: a resurrection is a node
        # no transaction committed, wearing a pruned child's lineage.
        if node["node_sha256"] in committed_results:
            continue
        lineage = (node["parent_sha256"], node["branch_index"], node["cache_sha256"])
        if lineage in rejected_children:
            raise ValueError("rejected KV child was reused by a live node")
    final_nodes = {node["node_sha256"] for node in nodes if node["final"]}
    if not final_nodes.issubset(committed_results):
        raise ValueError("KV state-tree final node was not transactionally committed")

    verified_count = sum(node["verified"] for node in nodes)
    final_count = sum(node["final"] for node in nodes)
    if (
        value["verified_boundary_count"] != verified_count
        or value["final_node_count"] != final_count
        or value["rejected_event_count"] != rejected_count
        or value["committed_event_count"] != committed_count
        or value["regeneration_count"] != regeneration_count
        or value["restore_failure_count"] != 0
        or value["all_rejected_slices_pruned"] is not True
        or value["exact_parent_restoration"] is not True
        or value["no_rejected_child_reused"] is not True
    ):
        raise ValueError("KV state-tree aggregate verdict differs")
    if require_final and final_count < 1:
        raise ValueError("KV state-tree has no accepted final boundary")
    body = {key: value[key] for key in value if key != "receipt_sha256"}
    if (
        not _is_sha256(value["receipt_sha256"])
        or _canonical_sha256(body) != value["receipt_sha256"]
    ):
        raise ValueError("KV state-tree receipt hash differs")
    return value


__all__ = [
    "KVMutationTransaction",
    "KVStateTree",
    "KVStateTreeError",
    "KV_STATE_EVENT_SCHEMA",
    "KV_STATE_NODE_SCHEMA",
    "KV_STATE_TREE_SCHEMA",
    "validate_kv_state_tree_receipt",
]
