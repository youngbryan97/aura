"""The host a run declares, held still for the arms of a trial: frozen, released, and reported to the engine that judges the body.

Lifted whole out of `driver`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


class _HeldHost:
    """Lifted whole out of SubjectRuntime; see driver.py."""

    def freeze_host(self, declared: Any = None) -> dict[str, float]:
        """Take the body's current reading and hold it for every arm to come.

        `declared` is another process's held reading, `{"hardware", "latency"}`:
        a carrier worker holds its coordinator's, so every arm of the run is one
        body. Holding its own, each worker held the machine as it found it, and
        a worker's two turns from a coordinator's anchor stood 2.14 apart from
        the coordinator's, first in the body's cpu, vram and temperature
        (26 October).
        """
        if isinstance(declared, dict) and declared.get("hardware") is not None:
            hardware = dict(declared.get("hardware") or {})
            latency = dict(declared.get("latency") or {})
        else:
            hardware = dict(getattr(self.state.soma, "hardware", {}) or {})
            latency = dict(getattr(self.state.soma, "latency", {}) or {})
        self.frozen_host = {
            key: float(hardware.get(key, 0.0) or 0.0)
            for key in ("cpu_usage", "vram_usage", "ram_usage", "temperature")
        }
        # Latency is elapsed wall clock, so it differs between two arms run
        # seconds apart by exactly as much as the machine was busy. Same
        # argument as the hardware readings: it is the environment, and it
        # belongs held still.
        self.frozen_latency = {
            key: float(latency.get(key, 0.0) or 0.0)
            for key in ("last_thought_ms", "perception_lag_ms", "token_velocity")
        }
        self._hold_observer()
        return self.frozen_host

    def _republish_body(self) -> None:
        """Tell the engine that judges the body what the body was just held at.

        The proprioceptive loop reads the machine and reports it to the
        resilience engine mid-phase, and the hold is applied after the phase.
        Without this the engine — and through it homeostasis, and through that
        her will to live — kept reading the real machine while the state was
        held at the displaced value, which is the two-bodies problem again with
        the seam moved. One body: the held reading is the reading.
        """
        from .driver import (
            logger,
        )

        engine = getattr(self.organs, "soma", None)
        report = getattr(engine, "observe_host", None)
        if not callable(report) or self.frozen_host is None:
            return
        try:
            report(
                cpu_percent=self.frozen_host.get("cpu_usage", 0.0),
                ram_percent=self.frozen_host.get(
                    "ram_usage", self.frozen_host.get("vram_usage", 0.0)
                ),
                temperature_c=self.frozen_host.get("temperature"),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            # The frozen host is what keeps an arm from reading the real
            # machine. One that does not install leaves the guards reading
            # live load, which is the defect the freeze exists to prevent.
            logger.warning("The declared host was not installed: %s", exc)
            return

    def thaw_host(self) -> None:
        self.frozen_host = None
        self.frozen_latency = None
        self._release_observer()

    def _hold_observer(self) -> None:
        """Hold the shared host observer still for the arms that follow.

        The state's body readings are held by `freeze_host`, and every layer
        that reads the machine through the shared observer went round that hold
        — embodied interoception samples it once a second of the organism's
        life, so two arms seconds apart read a different machine and the
        difference was in the floor of every edge into the body. This is the
        same act at the observer's own seam: the first reading of each kind
        stands for all three arms.
        """
        from .driver import (
            _HeldObserver,
        )

        try:
            from core.runtime.resource_observation import (
                get_resource_observer,
                set_resource_observer_for_test,
            )
        except (ImportError, AttributeError):
            # not a failure: no observer module, so there is nothing to
            # hold and nothing reading it either.
            return
        if self._held_observer is not None:
            return
        # Over the run's declared host when one is installed, so an arm's held
        # answers are the declared reading rather than the machine's.
        held = _HeldObserver(get_resource_observer())
        self._previous_observer = set_resource_observer_for_test(held)
        self._held_observer = held

    def _release_observer(self) -> None:
        from .driver import (
            logger,
        )

        if self._held_observer is None:
            return
        try:
            from core.runtime.resource_observation import set_resource_observer_for_test

            set_resource_observer_for_test(self._previous_observer)
        except (ImportError, AttributeError) as exc:
            # The held observer is cleared below either way, so a release
            # that did not land leaves the run's observer installed over
            # whatever runs next.
            logger.warning("The previous observer was not put back: %s", exc)
        self._held_observer = None
        self._previous_observer = None

