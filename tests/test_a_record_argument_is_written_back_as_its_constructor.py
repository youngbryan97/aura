"""A fault whose argument is a record of plain values can be made again (Master Todo F16, to build).

LIVE 19 November 2026 her log said twice "Not a fault she can make again: value is a PhenomenalField, not a plain
value". A fault's call is written back as source so her loop can make it again, and only literals could be written. A
dataclass whose every field is set by its generated ``__init__``, with no ``__post_init__``, is rebuilt exactly by that
``__init__`` called with its fields, so it is written as that call, importing its own class.
"""
from __future__ import annotations

import dataclasses

from core.runtime.a_call_that_failed import _written_back
from core.self_modification import her_real_calls as real
from core.state.aura_state import PhenomenalField


@dataclasses.dataclass
class _Inner:
    a: int
    b: str = "x"


@dataclasses.dataclass
class _Outer:
    inner: _Inner
    weights: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class _Adjusted:
    a: int

    def __post_init__(self) -> None:
        self.a += 1


@dataclasses.dataclass
class _Derived:
    a: int
    b: int = dataclasses.field(init=False, default=0)


@dataclasses.dataclass
class _HoldsAnObject:
    lock: object


def test_the_record_of_that_morning_is_written_back_and_rebuilt_equal():
    field = PhenomenalField(claim="I feel steady", valence=0.2, latent_snapshot=[0.1, 0.2], updated_at=12.5)
    source = _written_back(field)
    assert source.startswith("__import__('core.state.aura_state', fromlist=['PhenomenalField']).PhenomenalField(")
    assert eval(source) == field  # noqa: S307 - the source is what a test of hers runs


def test_a_record_inside_a_record_is_written_back_too():
    outer = _Outer(_Inner(3), [1.5])
    assert eval(_written_back(outer)) == outer  # noqa: S307


def test_a_record_its_constructor_would_not_rebuild_is_not_written():
    assert _written_back(_Adjusted(1)) is None, "__post_init__ changes what it is given"
    assert _written_back(_Derived(1)) is None, "a field its constructor does not take"
    assert _written_back(_HoldsAnObject(object())) is None, "a field that is not a value"


def test_her_real_calls_still_keep_only_literals():
    """A real call is replayed in a clone from its row; a record written as a constructor is not a literal, so it is not
    kept, as before."""
    row = {"module": "m", "qualname": "f", "file": "m.py", "keywords": [["field", _written_back(_Inner(3))]]}
    assert not real._readable(row)
