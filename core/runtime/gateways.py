"""Abstract gateway contracts for memory and state writes.

The audit demands that *every* memory write goes through a single
``MemoryWriteGateway`` and *every* state mutation through a single
``StateGateway``. Existing modules (memory_facade, state_repository,
state_vault) are wired into these contracts via the ServiceManifest;
this module gives the abstract interface the runtime can demand.

These are intentionally *abstract* — the goal is to have a stable
contract that runtime invariants can assert against without rewriting
the existing implementations all at once.
"""
from __future__ import annotations


from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class MemoryWriteRequest:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    receipt_id: Optional[str] = None
    cause: str = "unspecified"


@dataclass
class MemoryWriteReceipt:
    record_id: str
    receipt_id: str
    bytes_written: int
    schema_version: int


class MemoryWriteGateway(ABC):
    """Single canonical memory write authority."""

    @abstractmethod
    async def write(self, request: MemoryWriteRequest) -> MemoryWriteReceipt:
        ...

    @abstractmethod
    async def quarantine(self, record_id: str, reason: str) -> None:
        ...


@dataclass
class StateMutationRequest:
    key: str
    new_value: Any
    receipt_id: Optional[str] = None
    cause: str = "unspecified"


@dataclass
class StateMutationReceipt:
    key: str
    old_value: Any
    new_value: Any
    receipt_id: str


class StateGateway(ABC):
    """Single canonical state mutation authority."""

    @abstractmethod
    async def mutate(self, request: StateMutationRequest) -> StateMutationReceipt:
        ...

    @abstractmethod
    async def read(self, key: str, default: Any = None) -> Any:
        ...

    @abstractmethod
    async def snapshot(self) -> Dict[str, Any]:
        ...
