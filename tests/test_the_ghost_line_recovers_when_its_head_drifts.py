"""A degradation that had no way back.

The ghost line verifies the frame before the one it is about to append. When
its in-memory sequence drifts ahead of the chain file — a durable append that
did not land, or something rewinding the chain underneath it — that check fails
and keeps failing: every later advance is assigned a sequence the file does not
agree with. One campaign recorded it 1,074 times, and the line stopped
recording identity for the rest of the run.

The message named none of the three things that could have caused it, so the
state had to be reconstructed from the files afterwards. It names them now, and
re-reading the head is the recovery path that was missing.
"""

from __future__ import annotations

import pytest

from core.ghost.ghost_line import (
    GhostAnchorUnavailable,
    GhostLine,
    SelfDigest,
    SubstrateFingerprint,
)


@pytest.fixture()
def line(tmp_path, monkeypatch) -> GhostLine:
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path))
    return GhostLine(root=tmp_path / "ghost") if _takes_root() else GhostLine()


def _takes_root() -> bool:
    import inspect

    return "root" in inspect.signature(GhostLine.__init__).parameters


def _step(line: GhostLine, i: int):
    return line.advance(
        SelfDigest("Aura", f"hash-{i}", f"essence-{i}"),
        SubstrateFingerprint(model_artifact="stub"),
        trigger="tick",
    )


def test_a_drifted_head_recovers_instead_of_failing_forever(line: GhostLine) -> None:
    for i in range(3):
        _step(line, i)
    assert line._chain.last_entry().seq == 2

    # The state the campaign reached: the writer believes it is further along
    # than the file it writes to.
    line._chain._next_seq += 1

    frame = _step(line, 9)
    assert frame.seq == 3, "the append did not land at the chain's own next seq"
    assert line._chain.last_entry().seq == 3

    # And it keeps working afterwards, which is the part that was broken.
    assert _step(line, 10).seq == 4


def test_the_drift_names_what_it_saw(line: GhostLine) -> None:
    """Missing body and drifted head have different causes and different repairs."""
    for i in range(3):
        _step(line, i)
    with pytest.raises(GhostAnchorUnavailable) as caught:
        line._restore_anchor_for_append(99)
    error = caught.value
    assert error.seq == 99
    assert error.head == 2
    assert error.body_present is False
    assert "chain head 2" in str(error)
    assert "missing" in str(error)


def test_a_refresh_returns_the_sequence_the_file_agrees_with(line: GhostLine) -> None:
    for i in range(2):
        _step(line, i)
    line._chain._next_seq = 500
    assert line._chain.refresh_from_disk() == 2
