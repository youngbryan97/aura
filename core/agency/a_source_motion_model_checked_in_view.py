"""Use a source motion model after current observed response checks it."""
from __future__ import annotations

import math
from typing import Any

from core.agency.what_its_code_says_she_sees import paces_its_code_says
from core.perception.translation_from_visible_support import motion_was_measured
from core.verify import invariant


def source_motion_checked(run: Any, hers: Any, choosing: Any, at: float | None) -> bool:
    """A complete direct-control specification can act without measuring every direction first."""
    mine, moves = getattr(choosing, 'mine', None), getattr(choosing, 'moves', None)
    if (at is None or not math.isfinite(at) or mine is None or moves is None or hers.kind is None
            or hers.number != getattr(mine, 'number', None) or hers.kind != getattr(mine, 'kind', None)
            or getattr(hers, 'carried', False) or getattr(mine, 'seen', None) != at
            or not motion_was_measured(mine, at)):
        return False
    epoch = getattr(getattr(hers, 'identification', None), 'epoch', None)
    for field in ('body_trial', 'facing_trial'):
        trial = getattr(run, field, None)
        if (trial is not None and getattr(run, 'held_trying', False) == trial.independent
                and trial.belongs_to_input(run.held, run.held_since, epoch)
                and (trial.key or at < trial.until)):
            return False
    keys = tuple(getattr(run, 'keys', ()))
    specified = paces_its_code_says(moves, mine, keys)
    if not keys or set(specified) != set(keys):
        return False
    agreed = False
    for key, expected in specified.items():
        if len(expected) != 2 or not all(math.isfinite(value) for value in expected) or math.hypot(*expected) <= 2.:
            return False
        observed = hers.way_of(key)
        if observed is None:
            continue
        if (len(observed) != 2 or not all(math.isfinite(value) for value in observed)
                or math.dist(observed, expected) > max(2., .2 * math.hypot(*expected))):
            return False
        agreed = True
    return agreed


@invariant('agency.source_motion_needs_current_observed_agreement', scope='agency',
           owner='core/agency/a_source_motion_model_checked_in_view.py', observational=False)
def _source_motion_agreement_invariant() -> tuple:
    from types import SimpleNamespace

    from core.cognition.a_guide_to_a_place import THE_GUIDE, Guide
    from core.perception.a_world_as_its_code_makes_it import AWorldAsItsCodeMakesIt, HerBody

    mine = SimpleNamespace(number=1, kind=2, born=1., seen=3., motion_evidence='direct', x=20., y=30.)
    hers = SimpleNamespace(number=1, kind=2, carried=False, way_of=lambda key: (-30., 0.) if key == 'a' else None)
    run = SimpleNamespace(keys=['a', 'b'])
    choosing = SimpleNamespace(mine=mine, moves=SimpleNamespace(things={1: mine}, shape=(80, 120),
                                kinds=[SimpleNamespace(number=2, colour=(80, 200, 20))]))
    guide = Guide(world_its_code_makes=AWorldAsItsCodeMakesIt(stage=(120., 80.),
                  hers=HerBody(name='body', paces={'a': (-30., 0.), 'b': (30., 0.)}, drawn_in='#50c814')))
    token = THE_GUIDE.set(guide)
    try:
        assert source_motion_checked(run, hers, choosing, 3.)
        hers.way_of = lambda _key: None
        assert not source_motion_checked(run, hers, choosing, 3.)
        hers.way_of = lambda _key: (30., 0.)
        assert not source_motion_checked(run, hers, choosing, 3.)
    finally:
        THE_GUIDE.reset(token)
    return ()
