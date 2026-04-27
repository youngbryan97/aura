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
import resource
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ResourceSnapshot:
    timestamp: float
    cpu_seconds: float
    memory_rss_mb: float
    free_disk_mb: float
    process_id: int

    @classmethod
    def capture(cls, path: str | os.PathLike[str] = ".") -> "ResourceSnapshot":
        usage = resource.getrusage(resource.RUSAGE_SELF)
        disk = shutil.disk_usage(path)
        return cls(
            timestamp=time.time(),
            cpu_seconds=float(usage.ru_utime + usage.ru_stime),
            memory_rss_mb=_rss_to_mb(float(usage.ru_maxrss)),
            free_disk_mb=float(disk.free / (1024 * 1024)),
            process_id=os.getpid(),
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
        return max(0.0, min(1.0, sum(parts) / len(parts)))


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
    ) -> None:
        self.db_path = Path(db_path or "data/resource_stakes.sqlite3")
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
        snapshot = snapshot or ResourceSnapshot.capture(self.db_path.parent)
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
                max_tokens=384,
                effort="low",
                disabled_capabilities=tuple(sorted(disabled)),
                viability=viability,
            )
        if viability < 0.60:
            disabled.update({"background_exploration"})
            return ActionEnvelope(
                allowed=True,
                reason="constrained mode: bounded output and no optional exploration",
                max_tokens=768,
                effort="normal" if requested_effort != "high" else "normal",
                disabled_capabilities=tuple(sorted(disabled)),
                viability=viability,
            )
        return ActionEnvelope(
            allowed=True,
            reason="viable",
            max_tokens=1536 if requested_effort == "high" else 1024,
            effort=requested_effort,
            disabled_capabilities=tuple(sorted(disabled)),
            viability=viability,
        )

    def events(self, limit: int = 50) -> list[dict[str, object]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT kind, payload, created_at FROM resource_events ORDER BY id DESC LIMIT ?",
                (int(limit),),
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
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT payload FROM resource_state WHERE id = 1").fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        return ViabilityState(
            energy=float(payload["energy"]),
            tool_budget=float(payload["tool_budget"]),
            memory_budget=float(payload["memory_budget"]),
            storage_budget=float(payload["storage_budget"]),
            integrity=float(payload["integrity"]),
            suspended_capabilities=tuple(payload.get("suspended_capabilities", [])),
            degradation_events=int(payload.get("degradation_events", 0)),
        )

    def _save_state(self, state: ViabilityState) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO resource_state (id, payload, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET payload = excluded.payload, updated_at = excluded.updated_at
                """,
                (json.dumps(_state_dict(state), sort_keys=True), time.time()),
            )

    def _append_event(self, kind: str, payload: Mapping[str, object]) -> None:
        with sqlite3.connect(self.db_path) as conn:
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


def _rss_to_mb(raw: float) -> float:
    # Linux reports ru_maxrss in KiB; macOS reports bytes.  The threshold below
    # keeps small test processes sensible on both platforms.
    if raw > 10_000_000:
        return raw / (1024 * 1024)
    return raw / 1024


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

