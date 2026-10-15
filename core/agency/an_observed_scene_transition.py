"""Keep observed screen changes when input experiments consume their frames."""
from __future__ import annotations

import math
from typing import Any

from core.verify import invariant


def keep_scene_phase(run: Any, happened: list[dict], at: float) -> bool:
    """Preserve the observed phase's input budget and return from an overlay."""
    changed = any(event.get('what') == 'new screen' for event in happened)
    returned = any(event.get('what') == 'same screen back' for event in happened)
    if changed:
        run.new_screen_at = at
        run.acts_at_new_screen = (sum(run.input_key_downs.values())
                                  + getattr(run, 'pointed', 0) + getattr(run, 'pointer_moves', 0))
    if returned:
        run.new_screen_at = -math.inf
    return changed or returned


def keep_scene_transition(run: Any, happened: list[dict], at: float) -> bool:
    """Retire spatial plans and preserve the current phase before yielding input."""
    transition = keep_scene_phase(run, happened, at)
    if any(event.get('what') == 'new screen' for event in happened):
        for name in ('covered', 'barred'):
            places = getattr(run, name, None)
            if places is not None:
                places.clear()
        way = getattr(run, 'way_through', None)
        if way is not None:
            way.new_screen()
        run.walls_seen_at = -math.inf
    return transition


@invariant('agency.experiment_observations_preserve_scene_transitions', scope='agency',
           owner='core/agency/an_observed_scene_transition.py', observational=False)
def _observed_transition_invariant() -> tuple:
    from types import SimpleNamespace

    run = SimpleNamespace(covered={1}, barred={2}, input_key_downs={'a': 2}, pointed=1, pointer_moves=3)
    assert not keep_scene_transition(run, [], 1.) and run.covered == {1}
    assert keep_scene_transition(run, [{'what': 'new screen'}], 2.)
    assert not run.covered and not run.barred and run.acts_at_new_screen == 6 and run.new_screen_at == 2.
    assert keep_scene_transition(run, [{'what': 'same screen back'}], 3.)
    assert run.new_screen_at == -math.inf
    return ()
