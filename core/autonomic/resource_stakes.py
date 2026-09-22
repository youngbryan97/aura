"""Operational resource stakes for Aura.

This is not a claim that software resource pressure is biological metabolism.
It is a concrete bridge from "energy is a decorative float" toward "resource
state constrains what the system can do, persists, and has consequences."  The
ledger records scarcity, applies irreversible degradation until explicit repair,
and produces action envelopes that other subsystems can obey.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from core.runtime.errors import record_degradation
from core.runtime.resource_observation import ResourceObserver, get_resource_observer
from core.runtime.sqlite_support import connecting


@dataclass(frozen=True)
class ResourceSnapshot:
    timestamp: float
    cpu_seconds: float
    memory_rss_mb: float
    free_disk_mb: float
    process_id: int
    observation_source: str = "unavailable"
    observation_scenario_id: str = ""

    @classmethod
    def capture(
        cls,
        path: str | os.PathLike[str] = ".",
        *,
        observer: ResourceObserver | None = None,
    ) -> ResourceSnapshot:
        resource_observer = observer or get_resource_observer()
        disk = resource_observer.disk(path)
        memory = resource_observer.memory()
        process = resource_observer.process(os.getpid())
        provenance = resource_observer.provenance
        return cls(
            timestamp=time.time(),
            cpu_seconds=(
                0.0
                if process is None
                else float(process.cpu_user_seconds + process.cpu_system_seconds)
            ),
            memory_rss_mb=float(memory.process_rss_bytes) / float(1024**2),
            free_disk_mb=float(disk.free_bytes) / float(1024**2),
            process_id=os.getpid(),
            observation_source=provenance.source.value,
            observation_scenario_id=provenance.scenario_id,
        )


@dataclass(frozen=True)
class ViabilityState:
    energy: float
    tool_budget: float
    memory_budget: float
    storage_budget: float
    integrity: float
    suspended_capabilities: tuple[str, ...] = ()
    degradation_events: int = 0

    @property
    def viability(self) -> float:
        parts = [self.energy, self.tool_budget, self.memory_budget, self.storage_budget, self.integrity]
        mean = sum(parts) / len(parts)
        # A plain mean lets four healthy dimensions fully mask an EXHAUSTED
        # critical one (tool/storage/memory/integrity at 0). Cap viability by
        # the worst dimension so a critically-low resource pulls the envelope
        # down instead of being averaged away: at min_dim=0 viability is capped
        # at 0.3; at min_dim=1 there is no cap.
        min_dim = min(parts)
        return max(0.0, min(1.0, mean, 0.3 + 0.7 * min_dim))


@dataclass(frozen=True)
class ActionEnvelope:
    allowed: bool
    reason: str
    max_tokens: int
    effort: str
    disabled_capabilities: tuple[str, ...]
    viability: float

    def as_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "max_tokens": self.max_tokens,
            "effort": self.effort,
            "disabled_capabilities": list(self.disabled_capabilities),
            "viability": round(self.viability, 4),
        }


class ResourceStakesLedger:
    """Persistent budget ledger with non-cosmetic consequences."""

    def __init__(
        self,
        db_path: str | os.PathLike[str] | None = None,
        *,
        initial: ViabilityState | None = None,
        observer: ResourceObserver | None = None,
    ) -> None:
        # Resolve the default path against the config data dir (not the process
        # cwd) so different launch directories share one viability history
        # instead of forking independent ledgers.
        if db_path is None:
            try:
                from core.config import config
                default_path = config.paths.data_dir / "resource_stakes" / "stakes.sqlite3"
            except (ImportError, AttributeError, RuntimeError):
                default_path = Path("data/resource_stakes.sqlite3").resolve()
            self.db_path = Path(default_path)
        else:
            self.db_path = Path(db_path)
        self._observer = observer
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        if initial is not None and self._load_state() is None:
            self._save_state(initial)
        elif self._load_state() is None:
            self._save_state(
                ViabilityState(
                    energy=1.0,
                    tool_budget=1.0,
                    memory_budget=1.0,
                    storage_budget=1.0,
                    integrity=1.0,
                )
            )

    def state(self) -> ViabilityState:
        loaded = self._load_state()
        if loaded is None:
            raise RuntimeError("resource stakes ledger failed to initialize state")
        return loaded

    def observe(self, snapshot: ResourceSnapshot | None = None) -> ViabilityState:
        snapshot = snapshot or ResourceSnapshot.capture(
            self.db_path.parent,
            observer=self._observer,
        )
        state = self.state()
        penalties: dict[str, float] = {}
        if snapshot.free_disk_mb < 512:
            penalties["storage_budget"] = 0.15
        if snapshot.memory_rss_mb > 12_000:
            penalties["memory_budget"] = 0.10
        if penalties:
            state = self.degrade("resource_probe", penalties, suspend=("background_exploration",))
        self._append_event("observe", {"snapshot": snapshot.__dict__, "state": _state_dict(state)})
        return state

    def consume(
        self,
        action: str,
        *,
        energy: float = 0.0,
        tool_budget: float = 0.0,
        memory_budget: float = 0.0,
        storage_budget: float = 0.0,
    ) -> ViabilityState:
        state = self.state()
        new_state = ViabilityState(
            energy=_clamp01(state.energy - max(0.0, energy)),
            tool_budget=_clamp01(state.tool_budget - max(0.0, tool_budget)),
            memory_budget=_clamp01(state.memory_budget - max(0.0, memory_budget)),
            storage_budget=_clamp01(state.storage_budget - max(0.0, storage_budget)),
            integrity=state.integrity,
            suspended_capabilities=state.suspended_capabilities,
            degradation_events=state.degradation_events,
        )
        if new_state.viability < 0.30:
            new_state = self._with_degradation(
                new_state,
                integrity_penalty=0.08,
                suspend=("large_model_cortex", "nonessential_tools"),
            )
        self._save_state(new_state)
        self._append_event(
            "consume",
            {
                "action": action,
                "costs": {
                    "energy": energy,
                    "tool_budget": tool_budget,
                    "memory_budget": memory_budget,
                    "storage_budget": storage_budget,
                },
                "state": _state_dict(new_state),
            },
        )
        return new_state

    def earn(self, reason: str, rewards: Mapping[str, float]) -> ViabilityState:
        """Recover budgets through successful work.

        Integrity is intentionally not restored here.  That requires explicit
        repair, so "damage" is not wiped away by a casual reward.
        """
        state = self.state()
        new_state = ViabilityState(
            energy=_clamp01(state.energy + max(0.0, float(rewards.get("energy", 0.0)))),
            tool_budget=_clamp01(
                state.tool_budget + max(0.0, float(rewards.get("tool_budget", 0.0)))
            ),
            memory_budget=_clamp01(
                state.memory_budget + max(0.0, float(rewards.get("memory_budget", 0.0)))
            ),
            storage_budget=_clamp01(
                state.storage_budget + max(0.0, float(rewards.get("storage_budget", 0.0)))
            ),
            integrity=state.integrity,
            suspended_capabilities=state.suspended_capabilities,
            degradation_events=state.degradation_events,
        )
        self._save_state(new_state)
        self._append_event("earn", {"reason": reason, "rewards": dict(rewards), "state": _state_dict(new_state)})
        return new_state

    def degrade(
        self,
        reason: str,
        penalties: Mapping[str, float],
        *,
        suspend: tuple[str, ...] = (),
    ) -> ViabilityState:
        state = self.state()
        new_state = ViabilityState(
            energy=_clamp01(state.energy - max(0.0, float(penalties.get("energy", 0.0)))),
            tool_budget=_clamp01(
                state.tool_budget - max(0.0, float(penalties.get("tool_budget", 0.0)))
            ),
            memory_budget=_clamp01(
                state.memory_budget - max(0.0, float(penalties.get("memory_budget", 0.0)))
            ),
            storage_budget=_clamp01(
                state.storage_budget - max(0.0, float(penalties.get("storage_budget", 0.0)))
            ),
            integrity=_clamp01(state.integrity - max(0.0, float(penalties.get("integrity", 0.0)))),
            suspended_capabilities=tuple(sorted(set(state.suspended_capabilities).union(suspend))),
            degradation_events=state.degradation_events + 1,
        )
        self._save_state(new_state)
        self._append_event(
            "degrade",
            {"reason": reason, "penalties": dict(penalties), "suspend": list(suspend), "state": _state_dict(new_state)},
        )
        return new_state

    def repair(self, reason: str, *, integrity: float = 0.0, restore: tuple[str, ...] = ()) -> ViabilityState:
        state = self.state()
        suspended = set(state.suspended_capabilities)
        suspended.difference_update(restore)
        new_state = ViabilityState(
            energy=state.energy,
            tool_budget=state.tool_budget,
            memory_budget=state.memory_budget,
            storage_budget=state.storage_budget,
            integrity=_clamp01(state.integrity + max(0.0, integrity)),
            suspended_capabilities=tuple(sorted(suspended)),
            degradation_events=state.degradation_events,
        )
        self._save_state(new_state)
        self._append_event(
            "repair",
            {"reason": reason, "integrity": integrity, "restore": list(restore), "state": _state_dict(new_state)},
        )
        return new_state

    def action_envelope(self, requested_effort: str = "normal") -> ActionEnvelope:
        state = self.state()
        viability = state.viability
        disabled = set(state.suspended_capabilities)
        if viability < 0.18 or state.integrity < 0.20:
            return ActionEnvelope(
                allowed=False,
                reason="viability below survival threshold; repair/rest required before outward action",
                max_tokens=0,
                effort="repair_only",
                disabled_capabilities=tuple(sorted(disabled.union({"llm_generation", "tool_use"}))),
                viability=viability,
            )
        if viability < 0.35:
            disabled.update({"large_model_cortex", "background_exploration", "nonessential_tools"})
            return ActionEnvelope(
                allowed=True,
                reason="scarcity mode: conserve resources and prioritize self-maintenance",
                max_tokens=640, # Increased from 384
                effort="low",
                disabled_capabilities=tuple(sorted(disabled)),
                viability=viability,
            )
        if viability < 0.60:
            disabled.update({"background_exploration"})
            return ActionEnvelope(
                allowed=True,
                reason="constrained mode: bounded output and no optional exploration",
                max_tokens=1024, # Increased from 768 to preserve voice
                effort="normal" if requested_effort != "high" else "normal",
                disabled_capabilities=tuple(sorted(disabled)),
                viability=viability,
            )
        return ActionEnvelope(
            allowed=True,
            reason="viable",
            max_tokens=2048 if requested_effort == "high" else 1536, # Restored full range
            effort=requested_effort,
            disabled_capabilities=tuple(sorted(disabled)),
            viability=viability,
        )

    def events(self, limit: int = 50) -> list[dict[str, object]]:
        # Bound the LIMIT: a negative value forwarded to SQLite removes the
        # limit entirely and can exhaust memory on a large ledger.
        try:
            bounded = max(1, min(1000, int(limit)))
        except (TypeError, ValueError):
            bounded = 50
        with connecting(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT kind, payload, created_at FROM resource_events ORDER BY id DESC LIMIT ?",
                (bounded,),
            ).fetchall()
        return [
            {"kind": kind, "payload": json.loads(payload), "created_at": created_at}
            for kind, payload, created_at in rows
        ]

    def _with_degradation(
        self,
        state: ViabilityState,
        *,
        integrity_penalty: float,
        suspend: tuple[str, ...],
    ) -> ViabilityState:
        return ViabilityState(
            energy=state.energy,
            tool_budget=state.tool_budget,
            memory_budget=state.memory_budget,
            storage_budget=state.storage_budget,
            integrity=_clamp01(state.integrity - integrity_penalty),
            suspended_capabilities=tuple(sorted(set(state.suspended_capabilities).union(suspend))),
            degradation_events=state.degradation_events + 1,
        )

    def _init_db(self) -> None:
        with connecting(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS resource_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS resource_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    def _load_state(self) -> ViabilityState | None:
        try:
            with connecting(sqlite3.connect(self.db_path)) as conn:
                row = conn.execute("SELECT payload FROM resource_state WHERE id = 1").fetchone()
            if row is None:
                return None
            payload = json.loads(row[0])
            if not isinstance(payload, dict):
                raise ValueError("resource_state payload is not a mapping")
            # Every loaded scalar passes through _clamp01 (finite + [0,1]) so a
            # corrupt/hostile row cannot inject NaN/inf/out-of-range budgets.
            return ViabilityState(
                energy=_clamp01(payload.get("energy", 1.0)),
                tool_budget=_clamp01(payload.get("tool_budget", 1.0)),
                memory_budget=_clamp01(payload.get("memory_budget", 1.0)),
                storage_budget=_clamp01(payload.get("storage_budget", 1.0)),
                integrity=_clamp01(payload.get("integrity", 1.0)),
                suspended_capabilities=tuple(
                    str(c)[:80] for c in (payload.get("suspended_capabilities") or [])
                    if isinstance(c, str)
                ),
                degradation_events=max(0, int(payload.get("degradation_events", 0) or 0)),
            )
        except (sqlite3.Error, json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            # Corrupt/incompatible persisted state must not prevent boot —
            # quarantine to a clean default rather than propagating.
            record_degradation("resource_stakes", exc)
            return None

    def _save_state(self, state: ViabilityState) -> None:
        with connecting(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO resource_state (id, payload, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET payload = excluded.payload, updated_at = excluded.updated_at
                """,
                (json.dumps(_state_dict(state), sort_keys=True), time.time()),
            )

    def _append_event(self, kind: str, payload: Mapping[str, object]) -> None:
        with connecting(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "INSERT INTO resource_events (kind, payload, created_at) VALUES (?, ?, ?)",
                (kind, json.dumps(payload, sort_keys=True), time.time()),
            )


def _state_dict(state: ViabilityState) -> dict[str, object]:
    return {
        "energy": state.energy,
        "tool_budget": state.tool_budget,
        "memory_budget": state.memory_budget,
        "storage_budget": state.storage_budget,
        "integrity": state.integrity,
        "suspended_capabilities": list(state.suspended_capabilities),
        "degradation_events": state.degradation_events,
        "viability": state.viability,
    }


def _clamp01(value: float) -> float:
    try:
        v = float(value)
    # not a failure: a value that is not a number is not one this can read.
    except (TypeError, ValueError):
        return 0.0
    # Reject NaN/inf so they cannot create boundary/non-finite state or corrupt
    # the JSON persisted payload.
    if v != v or v in (float("inf"), float("-inf")):
        return 0.0
    return max(0.0, min(1.0, v))
