"""One reading of the screen, taken for one cycle of a pursuit.

What comes back is what she has to work from: the window she is acting in, the
text and controls in it, and — where a reading could not be taken — the reason,
which the decision needs as much as it needs the reading itself.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

from core.runtime.errors import record_degradation

from .screen_pursuit_bearings import (
    _AS_IT_USUALLY_IS,  # noqa: F401
    A_SCREENFUL_AT_LEAST,  # noqa: F401
    ENOUGH_TO_BE_A_THING,  # noqa: F401
    MOST_OF_A_SCREEN,  # noqa: F401
    SCREENFULS_TO_LOOK,  # noqa: F401
    SETTLE_AFTER_SCROLL_S,  # noqa: F401
    _a_screenful,  # noqa: F401
    _a_step_back,  # noqa: F401
    _by_how_much_room,  # noqa: F401
    _moves_that_leave_her_nothing,  # noqa: F401
    _take_the_run_its_bearings,
    _time_left,  # noqa: F401
    _within_a_move,  # noqa: F401
    )
from .screen_pursuit_looking import (
    _ANSWERING_TOOK,  # noqa: F401
    ASKING_TO_CONFIRM,  # noqa: F401
    LONGER_THAN_USUAL,  # noqa: F401
    OBSERVE_TIMEOUT_S,  # noqa: F401
    _expected_of,  # noqa: F401
    _how_full,  # noqa: F401
    _how_long_a_look_takes,
    _move_her_own_surface_aside,  # noqa: F401
    _narrate,  # noqa: F401
    _where,  # noqa: F401
    _where_clicked,  # noqa: F401
    _where_it_asks,  # noqa: F401
    )
from .screen_pursuit_surface import (
    LABEL_REACH,  # noqa: F401
    _bound_to_a_window,  # noqa: F401
    _ensure_frontmost,
    _ensure_page,
    _matches,  # noqa: F401
    _screen_size,  # noqa: F401
    _value_is_on_screen,  # noqa: F401
    labelled_by,  # noqa: F401
    )


async def observe_the_screen(

    run: SimpleNamespace,
) -> dict[str, Any]:
    # These were imports inside the function this was lifted from, and
    # they stay inside it. A module-level import binds once; a test that
    # patches one of these on its own module would then never be seen,
    # which is exactly what happened to `learn_about`.
    from core.conversation.word_markers import names_any
    from core.runtime.watched_goal import BROWSERS
    # The closure, named. It assigns none of these, so binding them back
    # to locals of the same name is the same program.
    anchor = run.anchor
    at_rest = run.at_rest
    drawn = run.drawn
    expect_page = run.expect_page
    lost_page = run.lost_page
    open_page = run.open_page
    reading_took = run.reading_took
    target_app = run.target_app

    # Reached at call time: the module this came from imports this one,
    # so the other direction cannot be a module-level import, and a
    # call-time one still sees a test's patch of the original.
    from .screen_pursuit import (
        LOOKS_REMEMBERED,
        _frontmost,
        current_page_identity,
        logger,
        read_screen,
    )

    # Put the target back in front before looking at it.
    #
    # Over a long run focus wanders: a notification, a click that lands
    # outside the window, the person switching away. Without this the loop
    # refuses every keystroke for the rest of the run and reads whatever
    # replaced its target — technically correct and completely stuck. A
    # task that is meant to last minutes has to be able to recover the
    # conditions it needs rather than only detect that they are gone.
    # The window this run belongs to, named by the caller or learned on
    # the first cycle. Either way it has to be in front to be acted in.
    # Work out what this run belongs to before anything depends on it.
    #
    # This whole block sat inside "if the run has a window", and the only
    # thing that could give it one was inside the block. With nothing
    # named the run never acquired an anchor, never brought anything
    # forward, and sent every key to whatever happened to be in front —
    # for want of a first cycle it could never have. LIVE 2026-08-26:
    # thirty-five moves into a terminal with the game one window back.
    if not anchor["page"] or not anchor["app"]:
        await _take_the_run_its_bearings(
            anchor,
            expect_page=expect_page,
            open_page=open_page,
            target_app=target_app,
        )
    mine = target_app or anchor["app"]
    # Whether anything had to be moved to get here. A reading taken
    # before a window was brought forward is a reading of what was in
    # front instead.
    undisturbed = True
    if mine:
        try:
            undisturbed = await _ensure_frontmost(mine)
            # Anchor to the page this run STARTED on when the caller did
            # not name one.
            #
            # Otherwise a run is bound to an application and nothing more,
            # and an application is not a context: the browser holds the
            # task's page and a dozen others. A run that only knows "Google
            # Chrome" will send its keys to whatever tab is in front, so
            # arrow keys meant for a game land on a video, a form, or
            # someone's mail — each keystroke legitimately delivered to the
            # wrong world. Measured live: a stray click moved the browser to
            # a different site and the loop kept acting there.
            #
            # Anchoring on the first cycle means a caller never has to
            # remember, and drift is always detectable rather than only
            # detectable when someone thought to declare an expectation.
            if not anchor["page"]:
                page = await current_page_identity()
                # A run that is not about a page has no page.
                #
                # This took whatever a browser happened to be showing.
                # Driving a desktop application, the run anchored itself
                # to somebody's open tab — and then every cycle checked
                # that the tab was still in front and brought the BROWSER
                # forward to restore it, over the window it was driving.
                #
                # LIVE 2026-09-04: "this run belongs to '2048 Game' on
                # 'https://x.com/home'", and readings of the board that
                # came back full of a timeline. The same care is two lines
                # below, deciding which application the run belongs to,
                # and was never applied to the page.
                about_a_page = bool(
                    open_page or expect_page or names_any(target_app, BROWSERS)
                )
                anchor["page"] = str(
                    expect_page
                    or (page.get("url") or page.get("title") or "" if about_a_page else "")
                ).strip()
                # And the window it is in, so a keystroke has somewhere it
                # belongs.
                #
                # Nothing named meant nothing checked: an empty target is
                # read as "no constraint" rather than "I do not know where
                # I am", so every guard passed and every key went to
                # whatever happened to be in front. LIVE 2026-08-26:
                # another part of her closed the browser mid-run, and she
                # played twelve moves of 2048 into a chat window.
                if not anchor["app"]:
                    # The application that holds the page, when this run is
                    # about a page at all. A task about a desktop
                    # application would otherwise anchor itself to a
                    # browser that happens to be open behind it.
                    holder = str(page.get("app") or "") if about_a_page else ""
                    anchor["app"] = (holder or await _frontmost() or "").strip()
                    if anchor["app"]:
                        logger.info("this run belongs to %r", anchor["app"])
            if anchor["page"] and not await _ensure_page(anchor["page"]):
                # The page this run is about is no longer in front and
                # could not be brought back. Reading on would be reading
                # someone else's page.
                lost_page["value"] = True
                return {
                    "ok": False, "text": "", "layout": [],
                    "error": "navigated_away", "at": time.time(),
                }
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "screen_pursuit",
                exc,
                severity="info",
                action="continued a pursuit without refocusing the target window",
            )
    ready = at_rest["reading"]
    at_rest["reading"] = None
    if ready is not None and undisturbed and drawn["where"] is None:
        # She has just watched this surface come to rest. Photographing
        # it again asks the same question of the same still picture.
        return ready
    # As long as reading has taken here, not a number chosen elsewhere.
    #
    # A read and a language pass want the same machine, so a read that
    # takes a second and a half on its own takes many while a resident
    # model is generating. Bounded by a constant, that difference reads as
    # a wedged capture: live 2026-09-07, "no reading inside 8.0s" and a
    # run that ended saying it could not see, on a screen it had been
    # reading perfectly a moment earlier. A busy machine and a broken one
    # are not the same thing and do not have the same answer.
    patience = _how_long_a_look_takes(reading_took)
    began_looking = time.monotonic()
    try:
        seen = await asyncio.wait_for(
            read_screen(target_app, over=drawn["where"]), timeout=patience
        )
    except TimeoutError:
        # A wedged capture is not a reason to keep acting blind.
        logger.info(
            "the screen did not answer inside %.1fs, and looking has been "
            "taking %.1fs here",
            patience,
            (sum(reading_took) / len(reading_took)) if reading_took else 0.0,
        )
        return {
            "ok": False,
            "text": "",
            "layout": [],
            "error": f"observe_timeout: no reading inside {patience:.1f}s",
        }
    reading_took.append(time.monotonic() - began_looking)
    del reading_took[:-LOOKS_REMEMBERED]
    return seen
