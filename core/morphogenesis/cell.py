from __future__ import annotations
from core.runtime.errors import record_degradation


import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence

from .field import MorphogenField
from .types import (
    CellLifecycle,
    CellManifest,
    CellRole,
    CellState,
    MorphogenSignal,
    SignalKind,
    clamp01,
    json_safe,
    stable_digest,
)

logger = logging.getLogger("Aura.Morphogenesis.Cell")

CellHandler = Callable[["MorphogenCell", List[MorphogenSignal], Dict[str, float]], Any]


@dataclass
class CellTickResult:
    cell_id: str
    activated: bool
    actions: List[Dict[str, Any]] = field(default_factory=list)
    emitted_signals: List[MorphogenSignal] = field(default_factory=list)
    success: bool = True
    error: str = ""
    latency_ms: float = 0.0
    activation_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "activated": self.activated,
            "actions": json_safe(self.actions),
            "emitted_signals": [s.to_dict() for s in self.emitted_signals],
            "success": bool(self.success),
            "error": self.error,
            "latency_ms": round(float(self.latency_ms), 3),
            "activation_score": round(float(self.activation_score), 5),
        }


class MorphogenCell:
    """A bounded, local-rule code cell.

    The cell does not edit source.  It observes signals/field gradients,
    decides local actions, can emit repair/growth signals, and delegates
    privileged repair to existing Aura systems.
    """

    def __init__(
        self,
        manifest: CellManifest,
        *,
        state: Optional[CellState] = None,
        handler: Optional[CellHandler] = None,
    ):
        self.manifest = manifest
        self.cell_id = manifest.canonical_id()
        self.state = state or CellState(
            energy=manifest.baseline_energy,
            lineage_id=f"lineage_{stable_digest(manifest.name, manifest.subsystem)}",
        )
        self.handler = handler
        self.neighbours: Dict[str, float] = {}
        self._running_tasks = 0

    @property
    def lifecycle(self) -> CellLifecycle:
        try:
            return CellLifecycle(self.state.lifecycle)
        except Exception:
            return CellLifecycle.ACTIVE

    @property
    def protected(self) -> bool:
        return bool(self.manifest.protected)

    def is_available(self) -> bool:
        lifecycle = self.lifecycle
        if lifecycle in {CellLifecycle.DEAD, CellLifecycle.APOPTOTIC}:
            return False
        if lifecycle == CellLifecycle.QUARANTINED and time.time() < self.state.quarantined_until:
            return False
        if lifecycle == CellLifecycle.QUARANTINED and time.time() >= self.state.quarantined_until:
            self.state.lifecycle = CellLifecycle.DORMANT
        return True

    def perceive(self, signals: Sequence[MorphogenSignal]) -> List[MorphogenSignal]:
        if not self.is_available():
            return []
        consumes = {str(v) for v in self.manifest.consumes}
        caps = {str(v) for v in self.manifest.capabilities}
        relevant: List[MorphogenSignal] = []
        for sig in signals:
            kind = sig.kind.value if isinstance(sig.kind, SignalKind) else str(sig.kind)
            if sig.target_cell_id and sig.target_cell_id != self.cell_id:
                continue
            if sig.subsystem not in {self.manifest.subsystem, "global", "generic"} and self.manifest.subsystem != "global":
                # Let protected/governor cells perceive global threats, but do not flood specialists.
                if not self.protected and "global" not in caps:
                    continue
            if kind in consumes or kind in caps or sig.target_cell_id == self.cell_id:
                relevant.append(sig)
        return relevant

    def activation_score(self, signals: Sequence[MorphogenSignal], field_state: Dict[str, float]) -> float:
        if not signals:
            gradient = max(
                float(field_state.get("danger", 0.0)),
                float(field_state.get("task_pressure", 0.0)),
                float(field_state.get("repair", 0.0)),
                float(field_state.get("growth", 0.0)),
            )
            return clamp01(gradient * 0.5)

        signal_drive = max((sig.intensity for sig in signals), default=0.0)
        field_drive = (
            0.25 * field_state.get("task_pressure", 0.0)
            + 0.25 * field_state.get("danger", 0.0)
            + 0.15 * field_state.get("repair", 0.0)
            + 0.15 * field_state.get("growth", 0.0)
            + 0.10 * field_state.get("novelty", 0.0)
            + 0.10 * field_state.get("curiosity", 0.0)
        )
        pressure = field_state.get("inhibition", 0.0) * 0.35 + field_state.get("resource_pressure", 0.0) * 0.35
        return clamp01((0.65 * signal_drive + 0.45 * field_drive) - pressure)

    async def tick(
        self,
        *,
        signals: Sequence[MorphogenSignal],
        field: MorphogenField,
        global_energy: float = 1.0,
    ) -> CellTickResult:
        t0 = time.monotonic()
        self.state.age_ticks += 1

        if not self.is_available():
            return CellTickResult(cell_id=self.cell_id, activated=False, success=True)

        # Gentle recovery for healthy idle cells.
        if self.lifecycle in {CellLifecycle.ACTIVE, CellLifecycle.DORMANT}:
            self.state.health = clamp01(self.state.health + 0.003)
            self.state.energy = clamp01(self.state.energy + 0.02)

        field_state = field.sample(self.manifest.subsystem)
        relevant = self.perceive(signals)
        score = self.activation_score(relevant, field_state)

        if self.lifecycle == CellLifecycle.HIBERNATING and score < self.manifest.activation_threshold * 1.5:
            return CellTickResult(cell_id=self.cell_id, activated=False, activation_score=score)

        if (
            self.state.energy < self.manifest.hibernation_threshold
            or global_energy < 0.08
            or field.pressure(self.manifest.subsystem) > 0.88
        ):
            self.hibernate(reason="energy_or_pressure")
            return CellTickResult(
                cell_id=self.cell_id,
                activated=False,
                actions=[{"kind": "hibernate", "reason": "energy_or_pressure"}],
                activation_score=score,
            )

        if score < self.manifest.activation_threshold:
            if self.state.age_ticks - self.state.activation_count > 900 and not self.protected:
                self.state.lifecycle = CellLifecycle.DORMANT
            return CellTickResult(cell_id=self.cell_id, activated=False, activation_score=score)

        if self._running_tasks >= max(1, self.manifest.max_parallel_tasks):
            return CellTickResult(
                cell_id=self.cell_id,
                activated=False,
                actions=[{"kind": "backpressure", "running_tasks": self._running_tasks}],
                activation_score=score,
            )

        get_task_tracker().create_task(get_state_gateway().mutate(StateMutationRequest(key='lifecycle', new_value=CellLifecycle.ACTIVE, cause='MorphogenCell.tick')))
        self.state.activation_count += 1
        get_task_tracker().create_task(get_state_gateway().mutate(StateMutationRequest(key='last_activation_at', new_value=time.time(), cause='MorphogenCell.tick')))
        energy_cost = min(0.18, 0.04 + score * 0.12)
        self.state.energy = clamp01(self.state.energy - energy_cost)

        actions: List[Dict[str, Any]] = []
        emitted: List[MorphogenSignal] = []

        # Built-in local rules.
        if field_state.get("danger", 0.0) > 0.55 or any(str(s.kind) in {"SignalKind.ERROR", "SignalKind.EXCEPTION", "error", "exception"} for s in relevant):
            actions.append({"kind": "request_immune_observation", "subsystem": self.manifest.subsystem})
            emitted.append(
                MorphogenSignal(
                    kind=SignalKind.REPAIR,
                    source=self.cell_id,
                    subsystem=self.manifest.subsystem,
                    intensity=max(0.2, field_state.get("danger", 0.0)),
                    payload={"cell": self.cell_id, "reason": "danger_gradient"},
                )
            )

        if score > self.manifest.differentiation_threshold and self.manifest.role == CellRole.STEM:
            actions.append({"kind": "differentiate_candidate", "subsystem": self.manifest.subsystem})
            self.state.specialisation_score = clamp01(self.state.specialisation_score + 0.06)

        if (
            score > self.manifest.replication_threshold
            and self.state.success_count >= 3
            and not self.protected
        ):
            actions.append({"kind": "replication_candidate", "lineage": self.state.lineage_id})

        # Optional handler: compose with existing service, never patch source directly.
        success = True
        error = ""
        if self.handler is not None:
            self._running_tasks += 1
            try:
                result = self.handler(self, list(relevant), field_state)
                if inspect.isawaitable(result):
                    result = await asyncio.wait_for(result, timeout=max(0.2, self.manifest.timeout_s))
                if isinstance(result, dict):
                    actions.extend(result.get("actions", []))
                    for raw_signal in result.get("signals", []):
                        if isinstance(raw_signal, MorphogenSignal):
                            emitted.append(raw_signal)
                elif isinstance(result, list):
                    actions.extend([{"kind": "handler_result", "value": item} for item in result[:8]])
            except asyncio.TimeoutError:
                success = False
                error = "handler_timeout"
                self.state.last_error = error
                actions.append({"kind": "handler_timeout", "timeout_s": self.manifest.timeout_s})
            except Exception as exc:
                record_degradation('cell', exc)
                success = False
                error = f"{type(exc).__name__}: {exc}"
                self.state.last_error = error
                actions.append({"kind": "handler_error", "error": error[:200]})
            finally:
                self._running_tasks = max(0, self._running_tasks - 1)

        self.apply_feedback(success=success)
        latency_ms = (time.monotonic() - t0) * 1000.0
        return CellTickResult(
            cell_id=self.cell_id,
            activated=True,
            actions=actions[:16],
            emitted_signals=emitted[:8],
            success=success,
            error=error,
            latency_ms=latency_ms,
            activation_score=score,
        )

    def apply_feedback(self, *, success: bool) -> None:
        if success:
            self.state.success_count += 1
            get_task_tracker().create_task(get_state_gateway().mutate(StateMutationRequest(key='confidence', new_value=clamp01(self.state.confidence + 0.025), cause='MorphogenCell.apply_feedback')))
            get_task_tracker().create_task(get_state_gateway().mutate(StateMutationRequest(key='health', new_value=clamp01(self.state.health + 0.025), cause='MorphogenCell.apply_feedback')))
        else:
            self.state.failure_count += 1
            self.state.confidence = clamp01(self.state.confidence - 0.06)
            self.state.health = clamp01(self.state.health - 0.09)
            if self.state.failure_count >= 3 and not self.protected:
                self.quarantine(reason="repeated_failures")

    def strengthen(self, neighbour_cell_id: str, amount: float = 0.05) -> None:
        self.neighbours[neighbour_cell_id] = clamp01(self.neighbours.get(neighbour_cell_id, 0.0) + amount)

    def weaken(self, neighbour_cell_id: str, amount: float = 0.05) -> None:
        current = self.neighbours.get(neighbour_cell_id, 0.0)
        new = clamp01(current - amount)
        if new <= 0.001:
            self.neighbours.pop(neighbour_cell_id, None)
        else:
            self.neighbours[neighbour_cell_id] = new

    def hibernate(self, reason: str = "") -> None:
        if self.protected:
            get_task_tracker().create_task(get_state_gateway().mutate(StateMutationRequest(key='lifecycle', new_value=CellLifecycle.DORMANT, cause='MorphogenCell.hibernate')))
        else:
            self.state.lifecycle = CellLifecycle.HIBERNATING
        self.state.last_error = reason

    def quarantine(self, reason: str = "", seconds: float = 300.0) -> None:
        if self.protected:
            get_task_tracker().create_task(get_state_gateway().mutate(StateMutationRequest(key='lifecycle', new_value=CellLifecycle.DORMANT, cause='MorphogenCell.quarantine')))
            get_task_tracker().create_task(get_state_gateway().mutate(StateMutationRequest(key='last_error', new_value=f"protected_quarantine_blocked:{reason}", cause='MorphogenCell.quarantine')))
            return
        self.state.lifecycle = CellLifecycle.QUARANTINED
        self.state.quarantined_until = time.time() + max(1.0, seconds)
        self.state.last_error = reason

    def apoptosis(self, reason: str = "") -> None:
        if self.protected:
            self.state.lifecycle = CellLifecycle.QUARANTINED
            get_task_tracker().create_task(get_state_gateway().mutate(StateMutationRequest(key='quarantined_until', new_value=time.time() + 60.0, cause='MorphogenCell.apoptosis')))
            get_task_tracker().create_task(get_state_gateway().mutate(StateMutationRequest(key='last_error', new_value=f"protected_apoptosis_blocked:{reason}", cause='MorphogenCell.apoptosis')))
            return
        self.state.lifecycle = CellLifecycle.APOPTOTIC
        self.state.last_error = reason

    def clone_manifest(self, *, suffix: str, role: Optional[CellRole] = None) -> CellManifest:
        data = self.manifest.to_dict()
        data["name"] = f"{self.manifest.name}:{suffix}"
        if role is not None:
            data["role"] = role.value
        data["metadata"] = {
            **dict(data.get("metadata") or {}),
            "parent_cell_id": self.cell_id,
            "lineage_id": self.state.lineage_id,
        }
        return CellManifest.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "manifest": self.manifest.to_dict(),
            "state": self.state.to_dict(),
            "neighbours": {str(k): round(float(v), 5) for k, v in self.neighbours.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, handler: Optional[CellHandler] = None) -> "MorphogenCell":
        cell = cls(
            CellManifest.from_dict(data.get("manifest", {})),
            state=CellState.from_dict(data.get("state", {})),
            handler=handler,
        )
        cell.neighbours = {str(k): clamp01(float(v)) for k, v in dict(data.get("neighbours", {})).items()}
        return cell
