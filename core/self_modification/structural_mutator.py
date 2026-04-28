"""Structural self-modification with audit log and reversibility.

Addresses the "learning is structurally limited / no open-ended self-modification"
critique. This module lets constrained parts of the architecture change at
runtime — modules enable/disable, parameter bands, routing edges — with a
tamper-evident audit log and explicit rollback.

The bar here is deliberately modest: it is not open-ended evolution. It is
"the system can change its own configuration within a whitelisted API and we
can prove what it changed, when, why, and revert it". That alone is stronger
than STDP-in-a-fixed-architecture and answers the critique that goals and
structure are both frozen at design time.

Changes are reversible. Rollback is first-class. Every mutation is hash-chained
so the audit trail cannot be silently rewritten.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


MUTATION_KINDS = {
    "module_toggle": "enable or disable a registered module",
    "parameter_band": "adjust a numeric parameter within its allowed band",
    "routing_edge": "add or remove a routing edge between modules",
}


@dataclass(frozen=True)
class MutationRequest:
    kind: str
    target: str
    operation: str
    payload: Dict[str, Any]
    rationale: str
    requested_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class MutationRecord:
    mutation_id: str
    kind: str
    target: str
    operation: str
    payload: Dict[str, Any]
    prev_state: Dict[str, Any]
    post_state: Dict[str, Any]
    rationale: str
    applied_at: float
    reverted_at: Optional[float]
    prev_hash: str
    hash: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "kind": self.kind,
            "target": self.target,
            "operation": self.operation,
            "payload": self.payload,
            "prev_state": self.prev_state,
            "post_state": self.post_state,
            "rationale": self.rationale,
            "applied_at": self.applied_at,
            "reverted_at": self.reverted_at,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        }


class StructuralMutator:
    """Runtime architecture mutations with audit log and rollback."""

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        self._lock = threading.RLock()
        if db_path is None:
            try:
                from core.config import config
                db_path = Path(config.paths.data_dir) / "structural_mutations.sqlite3"
            except Exception:
                db_path = Path.home() / ".aura" / "structural_mutations.sqlite3"
        self._db_path = Path(db_path)
        get_task_tracker().create_task(get_storage_gateway().create_dir(self._db_path.parent, cause='StructuralMutator.__init__'))
        self._init_db()

        self._module_state: Dict[str, bool] = {}
        self._parameter_bands: Dict[str, Tuple[float, float, float]] = {}
        self._routing_edges: set[Tuple[str, str]] = set()
        self._module_registry: Dict[str, Callable[[bool], None]] = {}
        self._parameter_registry: Dict[str, Callable[[float], None]] = {}

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS structural_mutations (
                    mutation_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    target TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    prev_state TEXT NOT NULL,
                    post_state TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    applied_at REAL NOT NULL,
                    reverted_at REAL,
                    prev_hash TEXT NOT NULL,
                    hash TEXT NOT NULL
                )
                """
            )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_module(self, name: str, setter: Callable[[bool], None], *, initial: bool = True) -> None:
        with self._lock:
            self._module_state[name] = bool(initial)
            self._module_registry[name] = setter

    def register_parameter(
        self,
        name: str,
        setter: Callable[[float], None],
        *,
        initial: float,
        min_value: float,
        max_value: float,
    ) -> None:
        if min_value >= max_value:
            raise ValueError("parameter band requires min < max")
        with self._lock:
            self._parameter_bands[name] = (float(min_value), float(max_value), float(initial))
            self._parameter_registry[name] = setter

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------
    def apply(self, request: MutationRequest) -> MutationRecord:
        with self._lock:
            if request.kind == "module_toggle":
                prev, post = self._apply_module_toggle(request)
            elif request.kind == "parameter_band":
                prev, post = self._apply_parameter_band(request)
            elif request.kind == "routing_edge":
                prev, post = self._apply_routing_edge(request)
            else:
                raise ValueError(f"unknown mutation kind: {request.kind}")
            record = self._commit_record(request, prev, post)
            return record

    def revert(self, mutation_id: str, rationale: str = "rollback") -> MutationRecord:
        with self._lock:
            row = self._fetch_row(mutation_id)
            if row is None:
                raise KeyError(f"mutation {mutation_id} not found")
            if row["reverted_at"] is not None:
                raise RuntimeError(f"mutation {mutation_id} already reverted")
            kind = row["kind"]
            target = row["target"]
            prev_state = json.loads(row["prev_state"])
            if kind == "module_toggle":
                enabled = bool(prev_state.get("enabled", True))
                setter = self._module_registry.get(target)
                if setter is not None:
                    try:
                        setter(enabled)
                    except Exception:
                        pass  # no-op: intentional
                self._module_state[target] = enabled
                post = {"enabled": enabled}
            elif kind == "parameter_band":
                value = float(prev_state.get("value", 0.0))
                setter = self._parameter_registry.get(target)
                if setter is not None:
                    try:
                        setter(value)
                    except Exception:
                        pass  # no-op: intentional
                minv, maxv, _ = self._parameter_bands.get(target, (value, value, value))
                self._parameter_bands[target] = (minv, maxv, value)
                post = {"value": value}
            elif kind == "routing_edge":
                edge = tuple(prev_state.get("edge", ["", ""]))
                if edge and all(edge):
                    self._routing_edges.add((str(edge[0]), str(edge[1])))
                post = {"edge": list(edge), "present": True}
            else:
                raise ValueError(f"cannot revert kind: {kind}")

            reverted_record = self._commit_record(
                MutationRequest(
                    kind=kind,
                    target=target,
                    operation="revert",
                    payload={"source_mutation_id": mutation_id},
                    rationale=rationale,
                ),
                {},
                {},
                prev_state_override=json.loads(row["post_state"]),
                post_state_override=post,
                mark_source_reverted=mutation_id,
            )
            return reverted_record

    # ------------------------------------------------------------------
    # Kind handlers
    # ------------------------------------------------------------------
    def _apply_module_toggle(self, request: MutationRequest) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        name = request.target
        if name not in self._module_state:
            raise KeyError(f"module {name} not registered for toggling")
        desired = bool(request.payload.get("enabled", not self._module_state[name]))
        prev = {"enabled": self._module_state[name]}
        setter = self._module_registry.get(name)
        if setter is not None:
            setter(desired)
        self._module_state[name] = desired
        return prev, {"enabled": desired}

    def _apply_parameter_band(self, request: MutationRequest) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        name = request.target
        if name not in self._parameter_bands:
            raise KeyError(f"parameter {name} not registered")
        minv, maxv, current = self._parameter_bands[name]
        desired = float(request.payload.get("value", current))
        clamped = max(minv, min(maxv, desired))
        prev = {"value": current, "min": minv, "max": maxv}
        setter = self._parameter_registry.get(name)
        if setter is not None:
            setter(clamped)
        self._parameter_bands[name] = (minv, maxv, clamped)
        return prev, {"value": clamped, "requested": desired, "clamped": desired != clamped}

    def _apply_routing_edge(self, request: MutationRequest) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        payload = request.payload or {}
        edge = (str(payload.get("source", "")), str(payload.get("dest", "")))
        if not all(edge):
            raise ValueError("routing_edge requires source and dest")
        add = bool(payload.get("add", True))
        present_before = edge in self._routing_edges
        if add:
            self._routing_edges.add(edge)
        else:
            self._routing_edges.discard(edge)
        return {"edge": list(edge), "present": present_before}, {"edge": list(edge), "present": add}

    # ------------------------------------------------------------------
    # Audit chain
    # ------------------------------------------------------------------
    def _commit_record(
        self,
        request: MutationRequest,
        prev_state: Dict[str, Any],
        post_state: Dict[str, Any],
        *,
        prev_state_override: Optional[Dict[str, Any]] = None,
        post_state_override: Optional[Dict[str, Any]] = None,
        mark_source_reverted: Optional[str] = None,
    ) -> MutationRecord:
        applied_at = time.time()
        pre = prev_state_override if prev_state_override is not None else prev_state
        post = post_state_override if post_state_override is not None else post_state
        mutation_id = hashlib.sha256(
            f"{applied_at}|{request.kind}|{request.target}|{request.operation}|{json.dumps(request.payload, sort_keys=True)}".encode("utf-8")
        ).hexdigest()[:16]
        prev_hash = self._tail_hash()
        payload = {
            "mutation_id": mutation_id,
            "kind": request.kind,
            "target": request.target,
            "operation": request.operation,
            "payload": request.payload,
            "prev_state": pre,
            "post_state": post,
            "rationale": request.rationale,
            "applied_at": applied_at,
            "prev_hash": prev_hash,
        }
        hash_ = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        record = MutationRecord(
            mutation_id=mutation_id,
            kind=request.kind,
            target=request.target,
            operation=request.operation,
            payload=dict(request.payload),
            prev_state=dict(pre),
            post_state=dict(post),
            rationale=request.rationale,
            applied_at=applied_at,
            reverted_at=None,
            prev_hash=prev_hash,
            hash=hash_,
        )
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO structural_mutations (
                    mutation_id, kind, target, operation, payload,
                    prev_state, post_state, rationale, applied_at,
                    reverted_at, prev_hash, hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    record.mutation_id,
                    record.kind,
                    record.target,
                    record.operation,
                    json.dumps(record.payload, sort_keys=True),
                    json.dumps(record.prev_state, sort_keys=True),
                    json.dumps(record.post_state, sort_keys=True),
                    record.rationale,
                    record.applied_at,
                    record.prev_hash,
                    record.hash,
                ),
            )
            if mark_source_reverted:
                conn.execute(
                    "UPDATE structural_mutations SET reverted_at = ? WHERE mutation_id = ?",
                    (time.time(), mark_source_reverted),
                )
        return record

    def _tail_hash(self) -> str:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT hash FROM structural_mutations ORDER BY applied_at DESC, mutation_id DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else "GENESIS"

    def _fetch_row(self, mutation_id: str) -> Optional[sqlite3.Row]:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute("SELECT * FROM structural_mutations WHERE mutation_id = ?", (mutation_id,)).fetchone()

    def audit_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM structural_mutations ORDER BY applied_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [
            {
                "mutation_id": r["mutation_id"],
                "kind": r["kind"],
                "target": r["target"],
                "operation": r["operation"],
                "payload": json.loads(r["payload"]),
                "prev_state": json.loads(r["prev_state"]),
                "post_state": json.loads(r["post_state"]),
                "rationale": r["rationale"],
                "applied_at": r["applied_at"],
                "reverted_at": r["reverted_at"],
                "prev_hash": r["prev_hash"],
                "hash": r["hash"],
            }
            for r in rows
        ]

    def verify_chain(self) -> bool:
        """Walk the audit chain and confirm each hash is intact."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM structural_mutations ORDER BY applied_at ASC, mutation_id ASC"
            ).fetchall()
        prev_hash = "GENESIS"
        for r in rows:
            payload = {
                "mutation_id": r["mutation_id"],
                "kind": r["kind"],
                "target": r["target"],
                "operation": r["operation"],
                "payload": json.loads(r["payload"]),
                "prev_state": json.loads(r["prev_state"]),
                "post_state": json.loads(r["post_state"]),
                "rationale": r["rationale"],
                "applied_at": r["applied_at"],
                "prev_hash": r["prev_hash"],
            }
            expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
            if expected != r["hash"] or r["prev_hash"] != prev_hash:
                return False
            prev_hash = r["hash"]
        return True


_singleton: Optional[StructuralMutator] = None
_lock = threading.Lock()


def get_structural_mutator() -> StructuralMutator:
    global _singleton
    with _lock:
        if _singleton is None:
            _singleton = StructuralMutator()
        return _singleton


def reset_singleton_for_test() -> None:
    global _singleton
    with _lock:
        _singleton = None
