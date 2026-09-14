"""Closure was passing on a periphery of width zero.

Both campaigns reported `closed: true` with `periphery_width: 0` and every loss
at exactly 1.0. Four hundred numbers were read and every one of them was the
same on every frame, so the comparison the criterion is about never happened —
and its absence read exactly like a pass, because a core with nothing outside
it has a leak of 0.0 and so does a genuinely closed one.

The walk spent its whole budget inside the first subtree it sorted into, which
is why the four hundred were constants: attributes are walked in name order and
the kernel's own configuration fills the cap before a phase, an organ or a
service is reached. The budget is per source now.
"""

from __future__ import annotations

from core.subject.closure import MAX_PERIPHERY, ClosureReport, _numbers, periphery_matrix


def _report(**kwargs) -> ClosureReport:
    base = dict(
        loss_core=1.0,
        loss_core_and_periphery=1.0,
        loss_shuffled_periphery=1.0,
        leak=0.0,
        shuffled_leak=0.0,
        leak_over_shuffle=0.0,
        periphery_width=0,
    )
    base.update(kwargs)
    return ClosureReport(**base)


def test_a_periphery_of_width_zero_is_not_a_closed_core() -> None:
    report = _report()
    assert not report.measured
    assert not report.closed, "closure passed on a comparison that never happened"
    assert "never compared" in report.as_dict()["why_not_measured"]


def test_a_core_the_periphery_cannot_improve_on_is_closed() -> None:
    report = _report(periphery_width=12, leak=0.0)
    assert report.measured
    assert report.closed
    assert report.as_dict()["why_not_measured"] == ""


def test_a_leak_inside_its_own_shuffled_floor_is_closed() -> None:
    report = _report(periphery_width=12, leak=0.01, shuffled_leak=0.02)
    assert report.closed


def test_a_real_leak_is_open() -> None:
    report = _report(periphery_width=12, leak=0.20, shuffled_leak=0.02, floor_high=0.05)
    assert report.measured
    assert not report.closed


class _Deep:
    def __init__(self) -> None:
        for index in range(600):
            setattr(self, f"a{index:04d}", float(index))


class _Other:
    def __init__(self) -> None:
        self.z_moves = 1.0


def test_one_subtree_cannot_spend_the_whole_budget() -> None:
    """The kernel's own constants used to fill four hundred slots alone."""
    out: dict[str, float] = {}
    share = MAX_PERIPHERY // 2
    _numbers(_Deep(), "deep", out, ceiling=share)
    assert len(out) <= share
    _numbers(_Other(), "other", out, ceiling=MAX_PERIPHERY)
    assert "other.z_moves" in out, "the second source never got a slot"


def test_a_missing_key_reads_as_zero_rather_than_dropping_the_frame() -> None:
    matrix, names = periphery_matrix([{"a": 1.0}, {"a": 2.0, "b": 3.0}])
    assert names == ("a", "b")
    assert matrix.shape == (2, 2)
    assert matrix[0, 1] == 0.0
