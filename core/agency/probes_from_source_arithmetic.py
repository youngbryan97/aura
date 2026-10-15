"""Conditional arithmetic can order authorized probes without supplying control."""
from __future__ import annotations

import math
import re
from typing import Any

from core.perception.motion_inside_source_methods import SourceMethodMotion
from core.verify import invariant


def source_motion_trial_keys(guide: Any, body_keys: tuple[str, ...]) -> tuple[str, ...]:
    """Prefer informative vector-update experiments among the current body's known keys.

    The fields' physical roles and signed alternatives can remain unknown.
    This changes sampling order only; existing observations still have to
    prove the body, its response, and any prerequisites for useful control.
    """
    reading = getattr(getattr(guide, 'code_says', None), 'source_motion', None)
    if (not isinstance(reading, SourceMethodMotion) or reading.unknown
            or re.fullmatch(r'swf:[0-9a-f]{64}', reading.source) is None):
        return ()
    controls = getattr(guide, 'controls', None) or {}
    found = []
    for effect in reading.effects:
        if (effect.kind == 'vector_increment' and effect.key in body_keys and effect.key in controls
                and effect.steps and all(type(v) in (int, float) and math.isfinite(v) for v in effect.steps)
                and any(v != 0 for v in effect.steps) and effect.key not in found):
            found.append(effect.key)
    return tuple(found)


@invariant('agency.conditional_source_arithmetic_only_orders_body_probes', scope='agency',
           owner='core/agency/probes_from_source_arithmetic.py', observational=False)
def _conditional_source_probe_invariant() -> tuple:
    from types import SimpleNamespace

    from core.perception.motion_inside_source_methods import MethodMotion

    effect = MethodMotion('opaque', 'X.z', 'vector_increment', 'X.p', 'X.p.q', ('X.p.r', 'X.p.s'),
                          (-1., 1.), (-18., 18.), (), ('unresolved control flow and units',), (1,))
    reading = SourceMethodMotion(effects=(effect,), source='swf:' + 'a' * 64)
    guide = SimpleNamespace(code_says=SimpleNamespace(source_motion=reading), controls={'opaque': object()})
    assert source_motion_trial_keys(guide, ()) == ()
    assert source_motion_trial_keys(guide, ('opaque',)) == ('opaque',)
    assert not effect.unconditional
    return ()
