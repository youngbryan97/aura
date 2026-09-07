"""core/connectome/activity.py — recording the tissue while it works.

A connectome is anatomy. It says which cell can reach which and says nothing
about what any of them did this morning. ZAPBench exists because that gap is
the interesting one: the zebrafish paper records 71,721 neurons for two hours
under nine stimulus conditions and then asks whether anyone can predict the
next thirty seconds.

Aura can be recorded the same way, and better, because the volume and the
recording come from the same body. The zebrafish work has to map a different
fish than the one it imaged. Here the cell that fires is the cell in the graph.

The instrument is :mod:`sys.monitoring`, which Python 3.12 added for exactly
this: per-code-object events with no cost on code nobody is watching. Starting
a recording is starting the light sheet, and it is bounded in frames, in cells
and in wall-clock so that leaving one running cannot eat the host.

Two signals come out, and both are kept:

``spikes``
    Calls per frame. This is ground truth and no microscope can see it.

``calcium``
    The same trace through a first-order kernel with GCaMP's decay, which is
    what a light-sheet recording of that activity would look like. Keeping it
    lets a model trained here be compared against a model trained on ZAPBench
    without the comparison quietly resting on Aura's cleaner data.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
import types
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import stable_id

logger = logging.getLogger("Aura.Connectome.Activity")

__all__ = [
    "ActivityFrame",
    "ActivityTrace",
    "RecorderConfig",
    "ActivityRecorder",
    "get_activity_recorder",
    "reset_activity_recorder_for_test",
    "GCAMP6F_DECAY_PER_FRAME",
    "ZAPBENCH_FRAME_SECONDS",
    "ObservedEdges",
]

#: ZAPBench's volume rate: one whole-brain frame every 914 ms.
ZAPBENCH_FRAME_SECONDS: float = 0.914

#: GCaMP6f decay over one ZAPBench frame. The indicator's single-spike decay
#: time constant is about 400 ms, so a frame retains exp(-914/400) of what came
#: before it. Source: Chen et al., Nature 499:295 (2013), GCaMP6f tau_decay.
GCAMP6F_TAU_SECONDS: float = 0.400
GCAMP6F_DECAY_PER_FRAME: float = math.exp(-ZAPBENCH_FRAME_SECONDS / GCAMP6F_TAU_SECONDS)

#: The monitoring tool slot. Six exist; five are reserved for debuggers,
#: coverage and profilers, and taking one of those would silently break them.
_TOOL_ID: int = 5


@dataclass(frozen=True)
class ActivityFrame:
    """One volume. Sparse, because most cells are silent in any 914 ms."""

    index: int
    started_at: float
    condition: str
    counts: Mapping[str, int]

    def total(self) -> int:
        return sum(self.counts.values())


@dataclass
class ActivityTrace:
    """A recording, in the shape a forecaster wants: cells by time.

    ``uids`` fixes the row order for the life of the trace, so a model trained
    on one trace can be evaluated on another from the same connectome.
    """

    uids: tuple[str, ...]
    conditions: tuple[str, ...]
    spikes: list[list[float]]
    frame_seconds: float = ZAPBENCH_FRAME_SECONDS
    attrs: dict[str, Any] = field(default_factory=dict)
    #: A dense array, when one was built directly. Twenty thousand frames over
    #: thirteen thousand cells is a quarter of a billion numbers, and holding
    #: those as Python floats costs about twenty times what the array does, so a
    #: fast frame rate needs this path rather than the list one.
    array: Any = None

    @property
    def n_cells(self) -> int:
        return len(self.uids)

    @property
    def n_frames(self) -> int:
        return int(self.array.shape[0]) if self.array is not None else len(self.spikes)

    def matrix(self) -> Any:
        """The trace as a dense ``(frames, cells)`` array."""
        import numpy as np

        if self.array is not None:
            return self.array
        if not self.spikes:
            return np.zeros((0, len(self.uids)), dtype=np.float32)
        return np.asarray(self.spikes, dtype=np.float32)

    def calcium(self, decay: float = GCAMP6F_DECAY_PER_FRAME) -> Any:
        """What a light sheet would have seen, as normalised ΔF/F.

        The kernel is first order, which is the standard forward model for a
        genetically encoded calcium indicator at frame rates well below the
        rise time. Normalisation is per cell against its own baseline, matching
        how ZAPBench's traces are prepared, so a silent cell reads zero rather
        than reading as a cell with a small constant fluorescence.
        """
        import numpy as np

        raw = self.matrix()
        if raw.size == 0:
            return raw
        out = np.zeros_like(raw)
        carry = np.zeros(raw.shape[1], dtype=np.float32)
        for t in range(raw.shape[0]):
            carry = carry * decay + raw[t]
            out[t] = carry
        baseline = np.percentile(out, 10, axis=0) if out.shape[0] >= 10 else out.min(axis=0)
        scale = np.maximum(baseline, 1.0)
        return (out - baseline) / scale

    def condition_index(self) -> dict[str, list[int]]:
        index: dict[str, list[int]] = {}
        for t, condition in enumerate(self.conditions):
            index.setdefault(condition, []).append(t)
        return index

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for condition in self.conditions:
            counts[condition] = counts.get(condition, 0) + 1
        matrix = self.matrix()
        active = int((matrix != 0).any(axis=0).sum()) if matrix.size else 0
        return {
            "cells": self.n_cells,
            "frames": self.n_frames,
            "active_cells": active,
            "frame_seconds": self.frame_seconds,
            "conditions": counts,
            "events": int(matrix.sum()) if matrix.size else 0,
        }


@dataclass
class ObservedEdges:
    """Caller-to-callee pairs seen firing, with how often each fired.

    Automated reconstruction has no ground truth to check itself against, which
    is why connectomics spends most of its labour on human proofreading. Aura
    has ground truth: run her and watch. An edge that fired happened, and a
    static reconstruction that missed it has a split error at that pair.
    """

    counts: dict[tuple[str, str], int] = field(default_factory=dict)
    unresolved: int = 0

    def add(self, pre: str, post: str) -> None:
        if pre == post:
            return
        key = (pre, post)
        self.counts[key] = self.counts.get(key, 0) + 1

    def without_self_pairs(self) -> ObservedEdges:
        """A copy with self-pairs dropped, for a recording made before they were.

        Kept so an older recording can be scored the same way a new one is
        rather than being reread as evidence of thousands of missing edges.
        """
        cleaned = ObservedEdges(unresolved=self.unresolved)
        cleaned.counts = {
            pair: count for pair, count in self.counts.items() if pair[0] != pair[1]
        }
        return cleaned

    def pairs(self) -> set[tuple[str, str]]:
        return set(self.counts)

    def summary(self) -> dict[str, Any]:
        return {
            "pairs": len(self.counts),
            "calls": sum(self.counts.values()),
            "unresolved": self.unresolved,
        }


@dataclass(frozen=True)
class RecorderConfig:
    """Every field here exists so a forgotten recording cannot hurt the host."""

    max_frames: int = 8_192
    max_cells: int = 200_000
    frame_seconds: float = ZAPBENCH_FRAME_SECONDS
    max_wall_seconds: float = 3_600.0
    #: Only code under these roots is watched. Anything else is out of volume.
    roots: tuple[str, ...] = ("core", "interface", "skills", "security", "llm", "executors")
    #: Also watch which cell called which. This costs a second monitoring event
    #: per call and is what turns a recording into ground truth for the static
    #: reconstruction, so it is worth paying for during a calibration window.
    capture_edges: bool = False
    max_observed_pairs: int = 400_000


class ActivityRecorder:
    """The light sheet.

    One recorder at a time holds the monitoring slot. ``start`` is idempotent
    for the same condition and refuses a second concurrent recording rather
    than interleaving two experiments into one trace.
    """

    def __init__(self, repo: Path | None = None, config: RecorderConfig | None = None) -> None:
        self.repo = Path(repo) if repo else Path(__file__).resolve().parents[2]
        self.config = config or RecorderConfig()
        self._lock = threading.RLock()
        self._recording = False
        self._owns_monitoring_slot = False
        self._condition = ""
        self._started_at = 0.0
        self._frame_started = 0.0
        self._frame_index = 0
        self._live: dict[str, int] = defaultdict(int)
        self._frames: deque[ActivityFrame] = deque(maxlen=self.config.max_frames)
        self._uid_cache: dict[tuple[str, str], str | None] = {}
        self._dropped_cells = 0
        self._events = 0
        self._callback_failures = 0
        self.observed = ObservedEdges()

    # -- lifecycle ------------------------------------------------------

    @property
    def recording(self) -> bool:
        return self._recording

    def start(self, condition: str = "baseline") -> bool:
        with self._lock:
            if self._recording:
                return self._condition == condition
            monitoring = getattr(sys, "monitoring", None)
            if monitoring is None:
                logger.info("sys.monitoring is unavailable; activity recording is off")
                return False
            try:
                monitoring.use_tool_id(_TOOL_ID, "aura.connectome")
                self._owns_monitoring_slot = True
                monitoring.register_callback(
                    _TOOL_ID, monitoring.events.PY_START, self._on_py_start
                )
                wanted = monitoring.events.PY_START
                if self.config.capture_edges:
                    monitoring.register_callback(
                        _TOOL_ID, monitoring.events.CALL, self._on_call
                    )
                    wanted |= monitoring.events.CALL
                monitoring.set_events(_TOOL_ID, wanted)
            except (ValueError, RuntimeError) as exc:
                logger.info("activity recording could not claim a monitoring slot: %s", exc)
                self._release_tool()
                return False
            self._recording = True
            self._condition = condition
            self._started_at = time.monotonic()
            self._frame_started = self._started_at
            return True

    def stop(self) -> ActivityTrace:
        with self._lock:
            if self._recording:
                self._close_frame()
                self._release_tool()
                self._recording = False
            return self.trace()

    def set_condition(self, condition: str) -> None:
        """Change the stimulus. The current frame is closed under the old one."""
        with self._lock:
            if condition == self._condition:
                return
            if self._recording:
                self._close_frame()
            self._condition = condition

    def _release_tool(self) -> None:
        if not self._owns_monitoring_slot:
            return
        monitoring = getattr(sys, "monitoring", None)
        if monitoring is None:
            return
        try:
            monitoring.set_events(_TOOL_ID, 0)
            monitoring.register_callback(_TOOL_ID, monitoring.events.PY_START, None)
            if self.config.capture_edges:
                monitoring.register_callback(_TOOL_ID, monitoring.events.CALL, None)
            monitoring.free_tool_id(_TOOL_ID)
            self._owns_monitoring_slot = False
        except (ValueError, RuntimeError) as exc:
            logger.debug("monitoring slot release was refused: %s", exc)

    # -- the callback ---------------------------------------------------

    def _on_py_start(self, code: Any, _offset: int) -> Any:
        try:
            uid = self._uid_for(code)
            if uid is None:
                monitoring = getattr(sys, "monitoring", None)
                return monitoring.DISABLE if monitoring else None
            self._live[uid] += 1
            self._events += 1
            now = time.monotonic()
            if now - self._frame_started >= self.config.frame_seconds:
                with self._lock:
                    if now - self._frame_started >= self.config.frame_seconds:
                        self._close_frame(now)
        except BaseException as exc:  # noqa: BLE001 - see _callback_failed
            return self._callback_failed(exc)
        return None

    def _on_call(self, code: Any, _offset: int, callee: Any, _arg0: Any) -> Any:
        """Record one observed caller-to-callee pair.

        The callee arrives as the callable rather than as a code object, so a
        builtin, a C function or a bound method with no Python body has nothing
        to attach to and is skipped. Those are the same calls the static
        reconstruction counts as leaving the volume.

        ``__code__`` is checked for being a code object rather than for being
        present. On a class it resolves to the descriptor, and reading
        ``co_filename`` off that raises inside whatever code happened to be
        running — which is how a recording broke 88 test files at import.
        """
        try:
            target = getattr(callee, "__code__", None)
            if not isinstance(target, types.CodeType):
                inner = getattr(callee, "__func__", None)
                target = getattr(inner, "__code__", None)
            if not isinstance(target, types.CodeType):
                return None
            post = self._uid_for(target)
            if post is None:
                return None
            pre = self._uid_for(code)
            if pre is None:
                self.observed.unresolved += 1
                return None
            if pre == post:
                # A comprehension or a nested function carries its parent's
                # identity, because the static side folds a closure into the
                # cell that contains it. On this side that shows up as a cell
                # calling itself millions of times, which is not a connection.
                return None
            if len(self.observed.counts) >= self.config.max_observed_pairs:
                return None
            self.observed.add(pre, post)
        except BaseException as exc:  # noqa: BLE001 - see _callback_failed
            return self._callback_failed(exc)
        return None

    def _callback_failed(self, exc: BaseException) -> Any:
        """Swallow a callback failure, and stop recording if they keep coming.

        A monitoring callback runs inside arbitrary user code. An exception
        raised here does not fail the recording, it fails whatever was running,
        which is the worst possible way for an observation tool to behave. So
        the first failures are counted and disabled per code object, and a
        recording that keeps failing takes itself off rather than degrading
        every call in the process.
        """
        self._callback_failures += 1
        if self._callback_failures == 1:
            logger.warning("activity recording callback failed: %r", exc, exc_info=False)
        if self._callback_failures >= 64:
            logger.warning("activity recording stopped after repeated callback failures")
            self._release_tool()
            self._recording = False
            return None
        monitoring = getattr(sys, "monitoring", None)
        return monitoring.DISABLE if monitoring else None

    def _uid_for(self, code: Any) -> str | None:
        """Map a code object onto a cell, or say it is out of volume.

        ``DISABLE`` is returned once per code object for anything outside the
        volume, so the cost of ignoring a hot standard-library function is paid
        a single time rather than on every call.
        """
        key = (code.co_filename, getattr(code, "co_qualname", code.co_name))
        cached = self._uid_cache.get(key)
        if cached is not None or key in self._uid_cache:
            return cached
        uid = self._compute_uid(key[0], key[1])
        if len(self._uid_cache) < self.config.max_cells:
            self._uid_cache[key] = uid
        else:
            self._dropped_cells += 1
        return uid

    def _compute_uid(self, filename: str, qualname: str) -> str | None:
        try:
            path = Path(filename).resolve()
            rel = path.relative_to(self.repo)
        except (ValueError, OSError):
            return None
        parts = list(rel.with_suffix("").parts)
        if not parts or parts[0] not in self.config.roots:
            return None
        if parts[-1] == "__init__":
            parts.pop()
        module = ".".join(parts)
        if "<locals>" in qualname:
            qualname = qualname.split(".<locals>", 1)[0]
        if "<" in qualname:
            return None
        return stable_id(module, qualname)

    # -- frames ---------------------------------------------------------

    def _close_frame(self, now: float | None = None) -> None:
        stamp = now if now is not None else time.monotonic()
        counts = dict(self._live)
        self._live.clear()
        self._frames.append(
            ActivityFrame(
                index=self._frame_index,
                started_at=self._frame_started,
                condition=self._condition,
                counts=counts,
            )
        )
        self._frame_index += 1
        self._frame_started = stamp
        if stamp - self._started_at >= self.config.max_wall_seconds and self._recording:
            self._release_tool()
            self._recording = False
            logger.info("activity recording stopped at its wall-clock bound")

    def frames(self) -> list[ActivityFrame]:
        with self._lock:
            return list(self._frames)

    def trace(self, uids: Sequence[str] | None = None) -> ActivityTrace:
        """Assemble the recorded frames into a cells-by-time trace."""
        with self._lock:
            frames = list(self._frames)
        if uids is None:
            seen: dict[str, None] = {}
            for frame in frames:
                for uid in frame.counts:
                    seen.setdefault(uid, None)
            order = tuple(sorted(seen))
        else:
            order = tuple(uids)
        position = {uid: i for i, uid in enumerate(order)}
        array = None
        rows: list[list[float]] = []
        try:
            import numpy as np

            array = np.zeros((len(frames), len(order)), dtype=np.float32)
            for index, frame in enumerate(frames):
                for uid, count in frame.counts.items():
                    column = position.get(uid)
                    if column is not None:
                        array[index, column] = float(count)
        except (ImportError, MemoryError, ValueError) as exc:
            logger.info("dense trace assembly fell back to lists: %s", exc)
            array = None
            for frame in frames:
                row = [0.0] * len(order)
                for uid, count in frame.counts.items():
                    column = position.get(uid)
                    if column is not None:
                        row[column] = float(count)
                rows.append(row)
        trace = ActivityTrace(
            uids=order,
            conditions=tuple(f.condition for f in frames),
            spikes=rows,
            frame_seconds=self.config.frame_seconds,
            array=array,
        )
        trace.attrs.update(
            {
                "events": self._events,
                "dropped_cells": self._dropped_cells,
                "callback_failures": self._callback_failures,
                "watched_code_objects": len(self._uid_cache),
                "observed": self.observed.summary(),
            }
        )
        return trace

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
            self._live.clear()
            self._frame_index = 0
            self._events = 0
            self.observed = ObservedEdges()


_RECORDER: ActivityRecorder | None = None
_RECORDER_LOCK = threading.Lock()


def get_activity_recorder() -> ActivityRecorder:
    global _RECORDER
    with _RECORDER_LOCK:
        if _RECORDER is None:
            _RECORDER = ActivityRecorder()
        return _RECORDER


def reset_activity_recorder_for_test() -> None:
    global _RECORDER
    with _RECORDER_LOCK:
        if _RECORDER is not None and _RECORDER.recording:
            _RECORDER.stop()
        _RECORDER = None
