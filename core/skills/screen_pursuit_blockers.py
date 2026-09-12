"""Something in the way, and one attempt to move it.

A dialog she did not open, a window in front of the one she is acting in, a
confirmation waiting for an answer. Each attempt is counted, because a pursuit
that keeps clearing the same blocker has misread what the blocker is, and the
count is what turns that into a reason rather than a loop.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the return annotation only; the runtime import
    from core.skills.fluid_executor import Step  # is inside the function

from types import SimpleNamespace
from typing import Any

from .screen_pursuit_surface import (
    LABEL_REACH,  # noqa: F401
    _bound_to_a_window,  # noqa: F401
    _matches,  # noqa: F401
    _screen_size,  # noqa: F401
    _value_is_on_screen,  # noqa: F401
    click_normalized,
    labelled_by,  # noqa: F401
    press,
)


async def clear_what_blocks_the_run(
    observation: dict[str, Any],
    run: SimpleNamespace,
) -> Step | None:
    """A Step that clears whatever is covering the content, or None.

    Tried BEFORE the policy on every cycle, because a dialog owning the
    screen makes every other decision meaningless: the reading is of the
    dialog, and the keys go to the dialog. Measured live — a page opened,
    was read correctly, and six moves in a row changed nothing because a
    modal had focus, with every keystroke reporting success.

    The judgement lives in core/perception/blocking_overlay.py, which
    dismisses and never agrees: a dialog offering only acceptance is
    reported as the person's decision and is left alone. This loop
    inherits that and adds nothing to it.
    """
    # These were imports inside the function this was lifted from, and
    # they stay inside it. A module-level import binds once; a test that
    # patches one of these on its own module would then never be seen,
    # which is exactly what happened to `learn_about`.
    from core.skills.fluid_executor import Step
    # The closure, named. It assigns none of these, so binding them back
    # to locals of the same name is the same program.
    anchor = run.anchor
    intending = run.intending
    needs_person = run.needs_person
    target_app = run.target_app
    unblock_with = run.unblock_with

    # Reached at call time: the module this came from imports this one,
    # so the other direction cannot be a module-level import, and a
    # call-time one still sees a test's patch of the original.
    from .screen_pursuit import (
        TWICE_BEFORE_HANDING_BACK,
        logger,
    )

    try:
        from core.perception.blocking_overlay import assess_overlay
    except ImportError as why:
        logger.info("cannot judge what is covering the content: %s", why)
        return None
    verdict = assess_overlay(observation, intending=intending["value"])
    if verdict.needs_person:
        # A dialog only the person can answer means the task cannot go on.
        #
        # This used to fall through to the policy, which then acted into a
        # dialog that owned the keyboard: measured live, forty moves that
        # each reported success while the board behind the dialog never
        # changed. Continuing past a question addressed to the person is
        # not perseverance, it is acting blind — and it is how a run
        # eventually stumbles into answering that question by accident.
        # Seen twice before the task is handed back.
        #
        # Handing it back ends the run, and a page carries things that go
        # away on their own: an advertising rail rotates, a toast appears
        # and fades, a banner loads late. Measured live, a run stopped
        # after twelve moves over a rail that had already changed by the
        # next reading. A dialog only the person can answer is still
        # there a second later, which is the whole difference.
        same = verdict.needs_person == needs_person["seen"]
        needs_person["seen"] = verdict.needs_person
        needs_person["times"] = needs_person["times"] + 1 if same else 1
        if needs_person["times"] < TWICE_BEFORE_HANDING_BACK:
            return None
        needs_person["reason"] = verdict.needs_person
        return None
    if not verdict.present:
        # Whatever it was is gone, so it was not the thing that stops a run.
        needs_person["seen"] = ""
        needs_person["times"] = 0
        return None

    # A caller may declare its OWN way forward.
    #
    # The detector deliberately refuses to guess at unlabelled controls,
    # because clicking an unknown thing on someone's screen is how a tab
    # gets closed or a row deleted. But a task usually knows its own
    # affordance — "New Game", "Start", "Continue", "Begin" — and that
    # knowledge belongs to whoever set the goal, not to a generic reader.
    #
    # Declared rather than inferred: the loop still never guesses, it just
    # accepts an instruction. Matched case-insensitively against the placed
    # text, and only when something is genuinely in the way.
    if unblock_with:
        wanted = unblock_with.strip().lower()
        candidates: list[tuple[float, float]] = []
        for region in observation.get("layout") or []:
            if wanted not in str(region.get("text") or "").strip().lower():
                continue
            try:
                candidates.append(
                    (
                        float(region.get("center_x", region.get("x"))),
                        float(region.get("center_y", region.get("y"))),
                    )
                )
            except (TypeError, ValueError):
                continue
        if candidates:
            # The one ON the dialog, when the label appears more than once.
            #
            # "New Game", "Start" and "Continue" routinely name both a
            # dialog's button and a permanent control in the app's own
            # toolbar. Measured live: four regions matched, and the first
            # was the toolbar — clicking it started a game BEHIND the
            # dialog and left the dialog up, so the run stayed blocked
            # while every step reported success.
            from core.perception.blocking_overlay import (
                overlay_box,
                overlay_focus,
            )

            box = overlay_box(observation)
            if box is not None:
                left, top, right, bottom = box
                # Inside the dialog's horizontal span, at or below its
                # text. A margin, because a button may sit slightly wider
                # than the sentence above it; and downward only, because a
                # dialog's controls are under its message.
                margin = 0.08
                inside = [
                    point
                    for point in candidates
                    if left - margin <= point[0] <= right + margin
                    and top - 0.02 <= point[1] <= bottom + 0.35
                ]
                if inside:
                    candidates = inside
            focus = overlay_focus(observation)
            if focus is not None:
                candidates.sort(
                    key=lambda point: (point[0] - focus[0]) ** 2
                    + (point[1] - focus[1]) ** 2
                )
            ux, uy = candidates[0]
            label = unblock_with
            frame = list(observation.get("bounds") or [])

            async def click_declared() -> bool:
                return await click_normalized(
                    ux, uy, expect_app=target_app or anchor["app"], bounds=frame
                )

            return Step(
                name=f"clear the way with {label!r}", action=click_declared
            )

    if verdict.suggested_key:
        key = verdict.suggested_key

        async def press_away() -> bool:
            return await press(key, expect_app=target_app or anchor["app"])

        return Step(name=f"dismiss overlay with {key}", action=press_away)

    if verdict.click_x is not None:
        label, x, y = verdict.label, verdict.click_x, verdict.click_y

        frame = list(observation.get("bounds") or [])

        async def click_away() -> bool:
            return await click_normalized(
                x, y, expect_app=target_app or anchor["app"], bounds=frame
            )

        return Step(name=f"dismiss overlay via {label!r}", action=click_away)
    return None
