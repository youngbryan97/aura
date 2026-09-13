"""What the organism keeps at module scope is rewound with everything else.

A service is carried under its name and a phase takes its own attributes with
it. Neither reached the globals behind accessors: the peripheral awareness
engine, the narrative gravity centre, the higher-order thought engine's `_hot`,
the authority audit. Over three conversation arms off one snapshot, thirteen
of them started each arm a record longer than the arm before.
"""

from __future__ import annotations

import asyncio
import sys
import types
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from core.subject.snapshot import (
    _MACHINERY_PACKAGES,
    _in_packages,
    _module_holdings,
    _module_state,
    _restore_module_state,
)

PROBE = "core._fork_probe_holdings"


def _probe_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType(PROBE)

    class Engine:
        def __init__(self) -> None:
            self.history = ["hello"]

        def remember(self, text: str) -> None:
            self.history.append(text)

    Engine.__module__ = PROBE
    module.Engine = Engine
    module._instance = Engine()
    module._seen = {"hello"}
    module._handlers = [module._instance.remember]
    module.LIMIT = 16
    monkeypatch.setitem(sys.modules, PROBE, module)
    return module


def test_an_engine_kept_in_a_module_global_is_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _probe_module(monkeypatch)
    holdings = _module_holdings()
    assert f"{PROBE}:_instance" in holdings
    assert f"{PROBE}:_seen" in holdings
    assert f"{PROBE}:Engine" not in holdings
    assert f"{PROBE}:LIMIT" not in holdings


def test_a_registry_of_callbacks_is_wiring_and_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deep copy of a bound method copies the object it is bound to, so a
    restored registry would call an organ nobody else holds."""
    _probe_module(monkeypatch)
    assert f"{PROBE}:_handlers" not in _module_holdings()


def test_the_machinery_and_the_instrument_are_not_rewound() -> None:
    held = [
        key for key in _module_holdings()
        if _in_packages(key.partition(":")[0], _MACHINERY_PACKAGES)
    ]
    assert not held, held[:5]


def test_a_restore_rewinds_the_engine_and_puts_back_a_cleared_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _probe_module(monkeypatch)
    saved = _module_state({f"{PROBE}:_instance", f"{PROBE}:_seen"})
    original, seen = module._instance, module._seen

    original.remember("the arm spoke")
    seen.add("the arm spoke")
    module._instance = None
    _restore_module_state(saved)

    assert module._instance is original and original.history == ["hello"]
    assert module._seen is seen and seen == {"hello"}


@pytest.mark.slow
def test_no_module_global_grows_from_one_arm_to_the_next() -> None:
    from core.subject.clock import installed_clock
    from core.subject.driver import (
        CONDITIONS,
        build_runtime,
        calibrate_clock,
        quiesce_organism,
        start_organism,
    )

    def sizes() -> dict[str, int]:
        out: dict[str, int] = {}
        for key, value in _module_holdings().items():
            if isinstance(value, (dict, list, set, deque)):
                out[key] = len(value)
                continue
            for name, held in vars(value).items():
                if isinstance(held, (dict, list, set, deque)):
                    out[f"{key}.{name}"] = len(held)
        return out

    async def run() -> list[dict[str, int]]:
        with TemporaryDirectory() as tmp:
            runtime = build_runtime(Path(tmp) / "runtime", seed=11)
            await start_organism(runtime)
            await quiesce_organism(runtime)
            await calibrate_clock(runtime, CONDITIONS, turns=1)
            runtime.freeze_host()
            conversation = CONDITIONS[0]
            await runtime.turn_once(conversation)
            snapshot = runtime.snapshot()
            starts = []
            for _ in range(3):
                runtime.restore(snapshot)
                starts.append(sizes())
                await runtime.turn_once(conversation)
            return starts

    try:
        starts = asyncio.run(run())
    finally:
        clock = installed_clock()
        if clock is not None:
            clock.uninstall()
    grew = sorted(
        {
            key
            for earlier, later in zip(starts, starts[1:], strict=False)
            for key in set(earlier) | set(later)
            if earlier.get(key) != later.get(key)
        }
    )
    assert not grew, f"module state one arm left for the next: {grew[:12]}"
