"""The register of publishers that run on a cadence, and the runner for them.

A declared channel whose writer nobody calls reads exactly like a channel
nobody ever wrote. The display says "no reading", the organ behind it looks
dead, and every claim bound to that channel decays. Two of these were found on
2026-09-10 and both were correct, complete, and covered:

* ``core.phenomena_wiring.sample`` holds the only ``write`` for all
  twenty-three disposition channels. Nothing called it, so every one of the
  fourteen dispositions reported "declared but never written" in a runtime
  that had been up for hours.
* ``core.conation.wiring.tick`` publishes eight conative channels and delivers
  arousal to the soma. Nothing called it either.

A subsystem that declares channels registers the function that writes them
here, once, next to the declaration. The 1Hz rate group runs the register.
That way wiring a publisher is one line in the place a person is already
looking, rather than a second edit in a file they have no reason to open.

A sampler that raises is recorded and skipped, never allowed to stop the
others: the cadence carries every subsystem's readings, so one bad reader
must not take the rest of the instrument down with it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation

#: What a sampler may raise without it being anyone's emergency. A publisher
#: reads live organs that can be half-started, so absence and shape are both
#: ordinary. Anything outside this set is a real fault and propagates.
_RECOVERABLE = (AttributeError, ImportError, KeyError, TypeError, ValueError, RuntimeError)


@dataclass
class _Sampler:
    """One registered publisher and what its last run did."""

    name: str
    run: Callable[[], Any]
    owner: str
    channels: tuple[str, ...] = ()
    runs: int = 0
    failures: int = 0
    last_run: float = 0.0
    last_error: str = ""
    last_wrote: int | None = None


@dataclass
class SamplerRegister:
    """Every publisher that must run on the cadence, and their outcomes."""

    _lock: threading.RLock = field(default_factory=threading.RLock)
    _samplers: dict[str, _Sampler] = field(default_factory=dict)

    def register(
        self,
        name: str,
        run: Callable[[], Any],
        *,
        owner: str,
        channels: tuple[str, ...] = (),
    ) -> bool:
        """Add a publisher, or replace the one under this name.

        Re-registering is normal: boot functions are idempotent and a second
        boot must leave one sampler, not two writing the same channels twice
        a second.
        """
        if not callable(run):
            raise TypeError(f"sampler {name!r} is not callable")
        with self._lock:
            existing = self._samplers.get(name)
            self._samplers[name] = _Sampler(
                name=name, run=run, owner=owner, channels=tuple(channels)
            )
            return existing is None

    def unregister(self, name: str) -> bool:
        with self._lock:
            return self._samplers.pop(name, None) is not None

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._samplers))

    def run_all(self) -> dict[str, Any]:
        """Run every registered publisher once. Returns what each one did."""
        with self._lock:
            due = list(self._samplers.values())

        outcomes: dict[str, Any] = {}
        for sampler in due:
            started = time.time()
            try:
                result = sampler.run()
            except _RECOVERABLE as exc:
                sampler.failures += 1
                sampler.last_error = f"{type(exc).__name__}: {exc}"
                sampler.last_run = started
                record_degradation(
                    "telemetry_samplers", exc, severity="debug",
                    action=f"{sampler.name} did not publish this cycle",
                )
                outcomes[sampler.name] = {"ok": False, "error": sampler.last_error}
                continue
            sampler.runs += 1
            sampler.last_error = ""
            sampler.last_run = started
            sampler.last_wrote = _how_many_it_wrote(result)
            outcomes[sampler.name] = {"ok": True, "wrote": sampler.last_wrote}
        return outcomes

    def report(self) -> dict[str, Any]:
        with self._lock:
            samplers = list(self._samplers.values())
        return {
            "schema": "aura.telemetry.samplers.v1",
            "registered": len(samplers),
            "never_ran": [s.name for s in samplers if s.runs == 0],
            "failing": [
                {"name": s.name, "failures": s.failures, "error": s.last_error}
                for s in samplers
                if s.last_error
            ],
            "samplers": [
                {
                    "name": s.name,
                    "owner": s.owner,
                    "channels": list(s.channels),
                    "runs": s.runs,
                    "failures": s.failures,
                    "last_run": s.last_run,
                    "wrote": s.last_wrote,
                }
                for s in sorted(samplers, key=lambda s: s.name)
            ],
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._samplers.clear()


def _how_many_it_wrote(result: Any) -> int | None:
    """How many readings the publisher says it took, where it says so.

    Publishers return different things — a dict of what was written, a bool,
    a summary. Read the count where one is legible and leave it unknown
    otherwise rather than reporting a zero nobody measured.
    """
    if isinstance(result, dict):
        published = result.get("published")
        if isinstance(published, bool):
            return 1 if published else 0
        return len(result)
    if isinstance(result, bool):
        return 1 if result else 0
    if isinstance(result, (list, tuple, set)):
        return len(result)
    return None


_REGISTER = SamplerRegister()


def get_sampler_register() -> SamplerRegister:
    return _REGISTER


def register_sampler(
    name: str,
    run: Callable[[], Any],
    *,
    owner: str,
    channels: tuple[str, ...] = (),
) -> bool:
    """Register a publisher to run on the telemetry cadence."""
    return _REGISTER.register(name, run, owner=owner, channels=channels)


def run_registered_samplers() -> dict[str, Any]:
    return _REGISTER.run_all()


def samplers_report() -> dict[str, Any]:
    return _REGISTER.report()


def reset_samplers_for_test() -> None:
    _REGISTER.reset_for_test()


__all__ = [
    "SamplerRegister",
    "get_sampler_register",
    "register_sampler",
    "reset_samplers_for_test",
    "run_registered_samplers",
    "samplers_report",
]
