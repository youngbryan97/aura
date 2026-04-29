"""Bounded in-process hot-swap registry.

This is not arbitrary core self-replacement. It is a small, auditable primitive
for swapping approved service objects while preserving exported state. The
caller supplies validation and migration hooks; failures leave the active object
unchanged.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Optional


StateExporter = Callable[[Any], Dict[str, Any]]
StateImporter = Callable[[Any, Dict[str, Any]], Any]
Validator = Callable[[Any], bool]


@dataclass(frozen=True)
class HotSwapTicket:
    ticket_id: str
    service_name: str
    created_at: float
    state_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HotSwapResult:
    ok: bool
    service_name: str
    reason: str
    ticket_id: Optional[str] = None
    old_generation: int = 0
    new_generation: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class _ServiceSlot:
    instance: Any
    generation: int = 0
    exporter: Optional[StateExporter] = None
    importer: Optional[StateImporter] = None


class HotSwapRegistry:
    """Atomic service replacement with validation and state migration."""

    def __init__(self):
        self._lock = threading.RLock()
        self._slots: Dict[str, _ServiceSlot] = {}
        self._pending: Dict[str, tuple[str, Any, Dict[str, Any], Validator]] = {}

    def register(
        self,
        name: str,
        instance: Any,
        *,
        exporter: Optional[StateExporter] = None,
        importer: Optional[StateImporter] = None,
    ) -> None:
        if not name:
            raise ValueError("service name is required")
        with self._lock:
            self._slots[name] = _ServiceSlot(instance=instance, exporter=exporter, importer=importer)

    def get(self, name: str) -> Any:
        with self._lock:
            return self._slots[name].instance

    def generation(self, name: str) -> int:
        with self._lock:
            return self._slots[name].generation

    def prepare(self, name: str, candidate: Any, *, validator: Validator) -> HotSwapTicket:
        with self._lock:
            if name not in self._slots:
                raise KeyError(name)
            if not validator(candidate):
                raise ValueError(f"candidate failed validation for {name}")
            slot = self._slots[name]
            state = slot.exporter(slot.instance) if slot.exporter else {}
            ticket = HotSwapTicket(
                ticket_id=f"hs-{uuid.uuid4().hex[:10]}",
                service_name=name,
                created_at=time.time(),
                state_keys=sorted(state.keys()),
            )
            self._pending[ticket.ticket_id] = (name, candidate, state, validator)
            return ticket

    def promote(self, ticket_id: str) -> HotSwapResult:
        with self._lock:
            pending = self._pending.pop(ticket_id, None)
            if pending is None:
                return HotSwapResult(False, "", "unknown_ticket", ticket_id=ticket_id)
            name, candidate, state, validator = pending
            slot = self._slots[name]
            old_generation = slot.generation
            try:
                migrated = slot.importer(candidate, state) if slot.importer else candidate
                if not validator(migrated):
                    return HotSwapResult(False, name, "post_migration_validation_failed", ticket_id, old_generation, old_generation)
                slot.instance = migrated
                slot.generation = old_generation + 1
                return HotSwapResult(True, name, "promoted", ticket_id, old_generation, slot.generation)
            except Exception as exc:
                return HotSwapResult(False, name, f"migration_error:{type(exc).__name__}", ticket_id, old_generation, old_generation)


__all__ = ["HotSwapRegistry", "HotSwapResult", "HotSwapTicket"]
