"""Keep a recorded forward use available when its information value has expired."""
from __future__ import annotations

import re
from typing import Any

from core.agency.what_i_can_do_here import what_is_clicked
from core.agency.where_things_lead import WhereThingsLead
from core.language.a_way_on import label_is_a_way_on
from core.skills.screen_pursuit_bearings import restart_controls
from core.verify import invariant


def known_navigation_uses(can_do: Any, available: Any) -> tuple[str, ...]:
    """Current offered labels with a retained forward transition from this screen.

    This is a decision priority. It does not authorize input, bind old pixel
    coordinates, claim new information, or bypass the task's action checks.
    Contradicting outcomes can remove a route's priority.
    """
    held = getattr(can_do, 'leads', None)
    current = getattr(can_do, 'on_screen', ())
    if (not isinstance(held, WhereThingsLead) or not held.knows_this_screen()
            or not isinstance(available, (list, tuple)) or len(available) > 128
            or not isinstance(current, (list, tuple)) or len(current) > 128
            or type(held._here) is not int or not 0 <= held._here < len(held.screens)):
        return ()
    found = []
    for action in available:
        if not isinstance(action, str) or len(action) > 256 or action not in current:
            continue
        label = what_is_clicked(action)
        if not label or not label_is_a_way_on(label, from_ocr=True):
            continue
        # Repeating a reset needs the task's own justification. A historical
        # transition alone cannot discard work to reach another screen.
        if (restart_controls({'layout': [{'text': label}]})
                or re.search(r'\b(?:retry|replay|again)\b', label, re.I)):
            continue
        receipt = f'{held._here}|{action}'
        counts, destination = held.went.get(receipt), held.to.get(receipt)
        if not isinstance(counts, dict) or type(destination) is not int:
            continue
        outcomes = tuple(counts.get(kind, 0) for kind in ('on', 'back', 'nothing', 'moved'))
        if (all(type(value) is int and 0 <= value <= 1_000_000_000 for value in outcomes)
                and outcomes[0] > sum(outcomes[1:])
                and 0 <= destination < len(held.screens) and destination != held._here):
            found.append(action)
    return tuple(dict.fromkeys(found))


@invariant('agency.navigation_use_is_separate_from_discovery', scope='agency',
           owner='core/skills/known_navigation_after_discovery.py', observational=False)
def _known_navigation_use_invariant() -> tuple:
    from types import SimpleNamespace

    action = 'click "Next"'
    graph = WhereThingsLead(screens=[frozenset({'page', 'one'}), frozenset({'page', 'two'})],
                            went={f'0|{action}': {'on': 1}}, to={f'0|{action}': 1}, _here=0)
    current = SimpleNamespace(leads=graph, on_screen=(action,))
    assert known_navigation_uses(current, (action, 'left')) == (action,)
    assert not known_navigation_uses(current, ('left',))
    graph.went[f'0|{action}']['nothing'] = 1
    assert not known_navigation_uses(current, (action,))
    return ()
