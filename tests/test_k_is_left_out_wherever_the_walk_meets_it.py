"""What K reads of an organ is not periphery, whichever path the walk takes to it.

The periphery skipped K's own attributes by name, at the top of a service. The
walk reaches an organ through the service's `instance`, or first through a
phase that holds it, and the skip did not go there: the field's state, the
mesh's weights and the phi core's visits were read as periphery under another
projection than K's, and reported as state K does not take in (22 October).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.subject.closure import read_periphery


def _field() -> SimpleNamespace:
    # K reads organ:field.F; _prev_F is the field's own and not K's.
    return SimpleNamespace(F=np.linspace(-1.0, 1.0, 16), _prev_F=np.linspace(0.0, 1.0, 16))


def _columns(seen: dict[str, float], attribute: str) -> list[str]:
    return [name for name in seen if f".{attribute}." in name or name.endswith(f".{attribute}")]


def test_an_organ_reached_through_a_phase_keeps_out_what_k_reads():
    field = _field()
    phase = SimpleNamespace(_field=field)
    kernel = SimpleNamespace(_phases=[phase], organs={})
    seen = read_periphery(kernel, SimpleNamespace(field=field))
    assert _columns(seen, "F") == []
    assert _columns(seen, "_prev_F")


def test_an_organ_reached_through_an_instance_keeps_out_what_k_reads():
    field = _field()
    kernel = SimpleNamespace(_phases=[], organs={"some_wrapper": SimpleNamespace(instance=field)})
    seen = read_periphery(kernel, SimpleNamespace(field=field))
    assert _columns(seen, "F") == []
    assert _columns(seen, "_prev_F")


def test_without_the_organs_the_walk_is_as_it_was():
    field = _field()
    phase = SimpleNamespace(_field=field)
    kernel = SimpleNamespace(_phases=[phase], organs={})
    seen = read_periphery(kernel)
    assert _columns(seen, "F")
