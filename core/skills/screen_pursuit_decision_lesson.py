"""Her lesson and her plan followed in their order when she chooses a move on a screen.

Moved out of core/skills/screen_pursuit_decision.py, which re-exports every name here: what goes on before what is
found out, the step to do now before the rest, the act that gets what the next step uses, a pause gone on from by what
paused it, and the other choices left once one is made.
"""
from __future__ import annotations

from typing import Any


def _first_what_goes_on(can_do: Any, telling: dict[str, float]) -> dict[str, float]:
    """A menu is read before it is pressed: a click on a label that goes on (Play, Next) before one on a
    label to read (Score), and keys as they were; and on a screen she has been on before, what led on from it then.

    And a thing done here to find out tells less each time it is done again and leads nowhere: what it
    does on this screen was found out the first times. LIVE 2026-10-06, a game's difficulty screen whose
    picture moved by itself: every press of right "moved" it, so right stayed as worth finding out as at
    first, and she pressed it thirty times without trying anything else.
    """
    from core.agency.what_i_can_do_here import what_is_clicked
    from core.language.a_way_on import how_much_it_leads_on

    held = getattr(can_do, "leads", None)
    leads = getattr(held, "how_it_led", lambda _act: 1.0)
    again = getattr(held, "taken_here_to_no_end", lambda _act: 0)
    material = getattr(can_do, 'material_presses', {})
    # What the screen's own words ask to be clicked goes on as surely as a Play does: it is doing what it says.
    asked = set(getattr(can_do, "clicks_asked_for", ()) or ())
    valued = {name: value * leads(name) * (1.6 if name in asked else how_much_it_leads_on(label) if (label := what_is_clicked(name)) else 1.0)
              / (1 + again(name)) / (1 + material.get(name, 0))
              for name, value in (telling or {}).items()}
    return _as_checked(can_do, _the_lesson_first(_go_on_from_a_pause(can_do, _once_chosen_go_on(can_do, valued)), held,
                                                 can_do))


def _never_tried_here(available: Any, responds: Any, can_do: Any) -> list[str]:
    """The acts on offer she has not taken here: not taken on this screen by her map of it, whether or not what they
    did could be made out (core/agency/where_things_lead.py), nor, for a key, anywhere (a key does what it does on
    every screen). A click is a click on this screen's thing: where her map knows the screen, a "next" clicked on the
    page before is not tried on this one. Offline 2026-10-13 the next page's own "next" was taken for tried, now and
    then, and she clicked the page's back button instead."""
    leads = getattr(can_do, "leads", None)
    taken_here = getattr(leads, "taken_here", lambda _act: False)
    knows_the_screen = bool(getattr(leads, "knows_this_screen", lambda: False)())
    tried = getattr(responds, "tried", set())

    def anywhere(name: str) -> bool:
        return name in tried and not (knows_the_screen and _done_to_a_thing(name))

    material = getattr(can_do, 'material_presses', {})
    return [option.name for option in available
            if not anywhere(option.name) and not taken_here(option.name) and not material.get(option.name)]


def _done_to_a_thing(name: str) -> bool:
    """Whether an act is done to a thing on the screen (a click, a carry), not a key pressed at the whole of it."""
    from core.agency.acts_on_two_places import two_places_of
    from core.agency.what_i_can_do_here import what_is_clicked

    return what_is_clicked(name) is not None or two_places_of(name) is not None


#: How much more a move that does the lesson's next step is worth than the best of the rest, and how much less a move
#: its rules forbid is.
NEXT_STEP, FORBIDDEN = 1.5, 0.1
#: A test of what was made, before anything was: it can show nothing (core/cognition/reading_the_rules.py TRIES_IT).
NOTHING_MADE_YET = 0.3
#: An act that gets what the next step needs, against doing a step itself: below doing the next step, above the rest.
REACHING = 1.35


def _the_lesson_first(valued: dict[str, float], leads: Any = None, can_do: Any = None) -> dict[str, float]:
    """The place's lesson followed in its order: a move that does its earliest step not yet done, with that step's
    control on the screen, before anything else; a move its rules forbid, hardly at all
    (core/cognition/reading_the_rules.py). LIVE 2026-10-10 a game's lesson said to open the library, choose a type and
    drag a device to the end of another's arrow, and she carried its tab names about instead.

    Where nothing on the screen does the next step, because what it uses is not on the screen, the act she has found
    brings that up, or leads toward where it was, comes next (``leads``, core/agency/where_things_lead.py): a step needs
    a thing, and getting the thing is the step before it."""
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    guide = THE_GUIDE.get()
    rules = getattr(guide, "rules", None)
    if rules is None or not valued:
        return valued
    best = max([1.0, *valued.values()])
    # Her plan's steps (core/cognition/a_plan_to_an_end.py) lead as the lesson's do: the lesson says how the place is
    # worked, the plan what she is using it for.
    procedures = [p for p in (rules, getattr(guide, "plan", None)) if p is not None]
    reach = _reaching_what_the_next_step_needs(procedures, valued, leads, getattr(guide, "locks", None))
    # A test of what was made waits until something is. LIVE 2026-10-10 a trap's TEST TRAP was pressed again and again
    # before a part was placed.
    from core.agency.what_i_can_do_here import what_is_clicked
    from core.cognition.reading_the_rules import the_step_that_holds, tries_what_was_made

    to_do = {id(p): p.the_step_to_do(list(valued), reaching=bool(reach)) for p in procedures}
    built = bool(getattr(guide, "built", "")) or bool(can_do.placed_count() if hasattr(can_do, "placed_count") else 0)
    out = {}
    for move, value in valued.items():
        from .screen_step_evidence import artifact_ready_for

        if not artifact_ready_for(move, can_do):
            continue
        blocked = False
        for procedure in procedures:
            step = procedure.step_of(move)
            if step is not None and step is to_do[id(procedure)]:
                refined = procedure.refinement_of(move)
                value = max(value, best * NEXT_STEP * (1.1 if refined is not None else 1.0))
            elif (step is not None and (holding := the_step_that_holds(procedure)) is not None
                  and step.order > holding.order and move not in reach and not procedure.reaches_next(move)):
                # Dependence is a requirement, including when this is the sole
                # candidate or its exploration score overwhelms all alternatives.
                blocked = True
        if blocked or rules.forbids(move):
            continue
        if not built and tries_what_was_made(what_is_clicked(move) or ""):
            value *= NOTHING_MADE_YET
        if move in reach:
            value = max(value, best * REACHING * reach[move])
        out[move] = value * (FORBIDDEN if rules.forbids(move) or rules.passes_over_what_it_teaches(move) else 1.0)
    # The part put down last, pointing away from where the chain must go, is turned before the next is added
    # (core/agency/aiming_what_was_placed.py).
    from core.agency.aiming_what_was_placed import a_turn_wanted

    turn = a_turn_wanted(can_do, guide, list(out)) if can_do is not None else ""
    if turn:
        out[turn] = max(out[turn], best * NEXT_STEP * 1.1)
    # Where a test found the join that broke, that part is selected to be mended before anything new is added
    # (core/agency/mending_what_broke.py); once selected, the turn above is for it.
    from core.agency.mending_what_broke import a_click_to_mend

    mend = a_click_to_mend(guide, list(out))
    if mend and mend != str(getattr(can_do, "last_tried", "") or "") and not turn:
        out[mend] = max(out[mend], best * NEXT_STEP * 1.1)
    return out


def _reaching_what_the_next_step_needs(procedures: list[Any], valued: dict[str, float], leads: Any,
                                       locks: Any = None) -> dict[str, float]:
    """For each procedure whose next step nothing offered does: the acts that get what that step uses, as she has found
    them, scaled so the surest is one. And for each way that would not open whose want has turned up
    (core/cognition/locks_she_met.py): the act on it, where it is here, else the acts that lead toward it."""
    from core.agency.what_i_can_do_here import what_is_clicked

    toward = getattr(leads, "toward", None)
    toward = toward or (lambda _needed: {})
    reach: dict[str, float] = {}
    for procedure in procedures:
        for move in valued:
            if procedure.reaches_next(move):
                reach[move] = 1.0
    for lock in (locks.open_to_try() if locks is not None else ()):
        here = [move for move in valued if what_is_clicked(move) == lock.at]
        for act, worth in ({move: 2.0 for move in here} if here else toward(lock.at)).items():
            if act in valued:
                reach[act] = max(reach.get(act, 0.0), worth)
    for procedure in procedures:
        nxt = procedure.next_step()
        if nxt is None or any(procedure.step_of(move) is nxt for move in valued):
            continue
        for needed in (nxt.thing, nxt.using, nxt.where):
            for act, worth in (toward(needed) if needed else {}).items():
                if act in valued:
                    reach[act] = max(reach.get(act, 0.0), worth)
    top = max(reach.values(), default=0.0)
    return {act: worth / top for act, worth in reach.items()} if top else {}


def _as_checked(can_do: Any, valued: dict[str, float]) -> dict[str, float]:
    """What she valued, checked against the guide to where she is (core/cognition/checking_the_debate.py); and what
    each act she takes does, taught to the check."""
    from core.cognition.a_guide_to_a_place import the_guide
    from core.cognition.checking_the_debate import the_check

    from .screen_pursuit_looking import MOVES_SAID

    paced = MOVES_SAID.get()
    check = the_check()
    if hasattr(can_do, "on_tried"):
        guide = the_guide(paced if isinstance(paced, dict) else None)

        def tried(act: str, answered: bool) -> None:
            check.learned(act, answered)
            # Bound procedure instances are settled before this screen is taken
            # in; a plan produced meanwhile cannot inherit an earlier input.
            if getattr(guide, "locks", None) is not None:
                guide.locks.tried(act, answered)

        can_do.on_tried = tried
    return check.weigh(valued, the_guide(paced if isinstance(paced, dict) else None), can_do)


def _go_on_from_a_pause(can_do: Any, valued: dict[str, float]) -> dict[str, float]:
    """A screen that says it is paused goes on by what paused it, before anything on it is found out; and while it goes
    on, what paused it is not taken again to see what it does.

    LIVE 2026-10-09 trying a game's corner button paused it; on "PAUSED" she clicked the word, the middle of the picture
    and every shape but that button, and the game stood paused until its time ran out.
    """
    by = getattr(can_do, "paused_by", "")
    if not by or by not in valued:
        return valued
    out = dict(valued)
    out[by] = max([1.0, *out.values()]) * 2.0 if getattr(can_do, "paused_here", False) else out[by] * CHOSEN_ALREADY
    return out


def _once_chosen_go_on(can_do: Any, valued: dict[str, float]) -> dict[str, float]:
    """On a screen that asks her to choose, once a choice has been made the other choices are not to find out, and
    what goes on (Next, OK) is wanted, though it did nothing before the choice.

    LIVE 2026-10-09 a golf game's choice of players drew its "next" faded until a player was chosen; she clicked "next"
    first, it did nothing, and after choosing she went on clicking the choices and the shapes for three minutes.
    """
    from core.agency.what_i_can_do_here import what_is_clicked
    from core.language.a_way_on import confirms

    if not getattr(can_do, "chose_here", False):
        return valued
    out = {name: value * (1.0 if confirms(what_is_clicked(name) or "") else CHOSEN_ALREADY)
           for name, value in valued.items()}
    for name in getattr(can_do, "on_screen", ()) or ():
        if confirms(what_is_clicked(name) or ""):
            out[name] = max(out.get(name, 0.0), 1.0)
    return out


#: How much less the other choices on a screen are worth finding out once one has been made.
CHOSEN_ALREADY = 0.1
