"""AURA_FAULT_INJECTION harness.

The audits call for an abuse gauntlet (2h, 24h, 72h, 7d) that injects
specific fault classes and asserts the invariants hold. This module
provides the programmable fault-injection primitives those stages call.

Fault classes (mapped from the audit prompt):

- hanging_sync_skill
- malformed_tool_result
- actor_crash
- browser_crash
- model_timeout
- event_bus_overflow
- bad_checkpoint_file
- dirty_shutdown
- memory_pressure

The harness is opt-in via the AURA_FAULT_INJECTION=1 env var or by
explicit construction, so chaos cannot fire in normal runs.
"""
from __future__ import annotations


import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("Aura.FaultInjection")


FAULT_CLASSES: Tuple[str, ...] = (
    "hanging_sync_skill",
    "malformed_tool_result",
    "actor_crash",
    "browser_crash",
    "model_timeout",
    "event_bus_overflow",
    "bad_checkpoint_file",
    "dirty_shutdown",
    "memory_pressure",
)


@dataclass
class FaultEvent:
    name: str
    timestamp: float
    severity: str
    payload: Dict[str, Any] = field(default_factory=dict)


# A fault handler is a callable returning the synthetic fault output.
# It may be sync or async.
FaultHandler = Callable[[Dict[str, Any]], Union[Any, Awaitable[Any]]]


class FaultInjector:
    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        rng: Optional[random.Random] = None,
    ):
        if enabled is None:
            enabled = os.environ.get("AURA_FAULT_INJECTION") == "1"
        self.enabled = enabled
        self._handlers: Dict[str, FaultHandler] = {}
        self._events: List[FaultEvent] = []
        self._probabilities: Dict[str, float] = {name: 0.0 for name in FAULT_CLASSES}
        self._rng = rng or random.Random()

    # --- configuration -----------------------------------------------------

    def set_probability(self, name: str, probability: float) -> None:
        if name not in FAULT_CLASSES:
            raise ValueError(f"unknown fault class '{name}'")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0,1]")
        self._probabilities[name] = probability

    def register_handler(self, name: str, handler: FaultHandler) -> None:
        if name not in FAULT_CLASSES:
            raise ValueError(f"unknown fault class '{name}'")
        self._handlers[name] = handler

    def reset(self) -> None:
        self._events.clear()
        for name in self._probabilities:
            self._probabilities[name] = 0.0

    # --- runtime API -------------------------------------------------------

    def maybe_inject(self, name: str, payload: Optional[Dict[str, Any]] = None) -> Optional[FaultEvent]:
        if not self.enabled:
            return None
        prob = self._probabilities.get(name, 0.0)
        if prob <= 0.0 or self._rng.random() > prob:
            return None
        ev = FaultEvent(
            name=name,
            timestamp=time.time(),
            severity=self._severity_for(name),
            payload=dict(payload or {}),
        )
        self._events.append(ev)
        return ev

    async def execute(self, name: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        """Force a fault by class. Used by tests and the abuse runner."""
        ev = FaultEvent(
            name=name,
            timestamp=time.time(),
            severity=self._severity_for(name),
            payload=dict(payload or {}),
        )
        self._events.append(ev)
        handler = self._handlers.get(name) or _DEFAULT_HANDLERS.get(name)
        if handler is None:
            return ev
        result = handler(ev.payload)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    def events(self) -> List[FaultEvent]:
        return list(self._events)

    @staticmethod
    def _severity_for(name: str) -> str:
        if name in {"actor_crash", "dirty_shutdown", "bad_checkpoint_file"}:
            return "critical"
        if name in {"model_timeout", "memory_pressure", "event_bus_overflow"}:
            return "high"
        return "medium"


# ---------------------------------------------------------------------------
# Default fault handlers (deterministic stand-ins used by the abuse runner)
# ---------------------------------------------------------------------------


def _malformed_tool_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"error": payload.get("reason", "synthetic_malformed_tool_result")}


def _model_timeout(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": False, "error": "model_timeout", "timeout_s": payload.get("timeout_s", 30.0)}


def _bad_checkpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": False, "error": "checkpoint_corrupted", "where": payload.get("path", "<unknown>")}


def _memory_pressure(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": False, "error": "memory_pressure", "rss_mb": payload.get("rss_mb", 4000)}


_DEFAULT_HANDLERS: Dict[str, FaultHandler] = {
    "malformed_tool_result": _malformed_tool_result,
    "model_timeout": _model_timeout,
    "bad_checkpoint_file": _bad_checkpoint,
    "memory_pressure": _memory_pressure,
}


# Singleton accessor ---------------------------------------------------------

_global_injector: Optional[FaultInjector] = None


def get_fault_injector() -> FaultInjector:
    global _global_injector
    if _global_injector is None:
        _global_injector = FaultInjector()
    return _global_injector


def reset_fault_injector() -> None:
    global _global_injector
    _global_injector = None


# ---------------------------------------------------------------------------
# Abuse Gauntlet stages
# ---------------------------------------------------------------------------


ABUSE_STAGES: Tuple[Tuple[str, float], ...] = (
    ("stage_1_2h", 2 * 3600.0),
    ("stage_2_24h", 24 * 3600.0),
    ("stage_3_72h", 72 * 3600.0),
    ("stage_4_7d", 7 * 24 * 3600.0),
)


@dataclass
class AbuseGauntletReport:
    stage: str
    duration_s: float
    fired: List[FaultEvent] = field(default_factory=list)
    invariant_violations: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.invariant_violations


async def run_abuse_stage(
    stage: str,
    *,
    invariants_check: Callable[[], Union[bool, Awaitable[bool]]],
    injector: Optional[FaultInjector] = None,
    duration_s: Optional[float] = None,
    interval_s: float = 30.0,
    fault_sequence: Optional[List[str]] = None,
) -> AbuseGauntletReport:
    """Run a single stage of the abuse gauntlet.

    Tests can call this with a tiny duration_s and a deterministic
    fault_sequence to simulate the long-running stage in milliseconds.
    """
    stages = dict(ABUSE_STAGES)
    if stage not in stages:
        raise ValueError(f"unknown abuse stage '{stage}'")
    duration = duration_s if duration_s is not None else stages[stage]
    injector = injector or get_fault_injector()
    if not injector.enabled:
        injector.enabled = True
    report = AbuseGauntletReport(stage=stage, duration_s=duration)
    sequence = list(fault_sequence) if fault_sequence else list(FAULT_CLASSES)
    end = time.monotonic() + duration
    idx = 0
    while time.monotonic() < end:
        name = sequence[idx % len(sequence)]
        idx += 1
        ev = await injector.execute(name)
        if isinstance(ev, FaultEvent):
            report.fired.append(ev)
        else:
            report.fired.append(injector.events()[-1])
        ok = invariants_check()
        if asyncio.iscoroutine(ok):
            ok = await ok
        if not ok:
            report.invariant_violations.append(name)
            break
        await asyncio.sleep(interval_s)
    return report
