"""The periphery a frame is compared against is the periphery at that frame.

Closure asks whether anything outside K at frame t predicts K at t + 1 beyond
K at t. Until 22 October both recorders read the periphery in a loop over the
frames a finished turn returned, so every frame of a turn was given the turn's
last periphery, which already holds what the turn did after that frame. A
reading of the future predicts the future; that is a leak of the measurement,
not of the core.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import core.subject.closure as closure
from tools.run_subject_core import _record

REPO = Path(__file__).resolve().parents[1]


class _Frame:
    def __init__(self, tag: str) -> None:
        self.misses: dict[str, str] = {}
        self.tag = tag


class _Runtime:
    """Moves the outside once per frame, and once more after the last."""

    def __init__(self) -> None:
        self.kernel = SimpleNamespace(level=0.0)
        self.organs = None
        self.state = None

    async def turn_once(self, condition, *, on_frame=None):
        frames = []
        for index, tag in enumerate(("open", "phase", "heartbeat", "ontogeny")):
            self.kernel.level = float(index)
            frame = _Frame(tag)
            frames.append(frame)
            if on_frame is not None:
                on_frame(frame)
        self.kernel.level = 99.0
        return frames


def test_each_turn_carries_the_outside_as_it_was_at_its_closing_frame(monkeypatch):
    monkeypatch.setattr(closure, "read_periphery", lambda kernel, organs=None: {"level": kernel.level})
    frames, (matrix, names) = asyncio.run(_record(_Runtime(), ["one", "two"], 1, strict=False))
    assert len(frames) == 8
    assert names == ("level",)
    assert matrix[:, 0].tolist() == [3.0, 3.0]


def test_a_periphery_read_at_every_frame_is_taken_at_the_turn_rows():
    import numpy as np

    recording = SimpleNamespace(
        x=np.zeros((8, 2)),
        turn_rows=lambda marker="ontogeny": np.array([3, 7]),
    )
    per_frame = np.arange(8, dtype=float)[:, None]
    assert closure.periphery_by_turn(per_frame, recording)[:, 0].tolist() == [3.0, 7.0]
    per_turn = np.array([[3.0], [7.0]])
    assert closure.periphery_by_turn(per_turn, recording) is per_turn


def _reads_after_the_turn(path: Path) -> list[int]:
    """Lines where a loop over a finished turn's frames reads the periphery."""
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        source = ast.unparse(node.iter)
        if "turn_once" not in source:
            continue
        if any(
            isinstance(inner, ast.Call) and "read_periphery" in ast.unparse(inner.func)
            for inner in ast.walk(node)
        ):
            found.append(node.lineno)
    return found


def test_no_recorder_reads_the_outside_after_the_turn_returned():
    for name in ("tools/run_subject_core.py", "tools/run_subject_core_v25.py"):
        assert _reads_after_the_turn(REPO / name) == [], name
