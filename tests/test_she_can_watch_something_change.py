"""Every skill did ONE act and returned, so nothing could watch.

The pieces all existed and none were connected. FluidExecutor.pursue closes a
perceive-decide-act loop with governance and verification. get_screen_text
reads the screen and reports where each run of text sat. perception_demand
keeps her eyes open at task cadence while she acts, instead of the 0.1Hz a
foreground generation imposes. hotkey presses keys.

What was missing was any way to ASK for the combination. Tasks that are
trivially describable — "wait for that build and tell me if it fails", "keep
pressing next until the form is done", "play this until you win" — had no path
through the system, and the failure always looked the same: she did one step
and stopped.

Nothing here is about any particular screen. A build log, a wizard, an
installer and a game are the same problem: read, decide from what was read,
act, read again.
"""
from __future__ import annotations

import asyncio

import pytest

from core.runtime.perception_demand import (
    perception_is_demanded,
    reset_perception_demand,
)
from core.skills import screen_pursuit as sp


@pytest.fixture(autouse=True)
def _clean():
    reset_perception_demand()
    yield
    reset_perception_demand()


@pytest.fixture
def screen(monkeypatch):
    """A screen that only changes when a key is pressed."""
    state = {"n": 0, "pressed": [], "saw_demand": False}

    async def read(app_name="", over=None):
        state["saw_demand"] = state["saw_demand"] or perception_is_demanded()
        return {
            "ok": True,
            "text": "DONE" if state["n"] >= 4 else f"count {state['n']}",
            "layout": [],
            "error": "",
        }

    async def press(key, *, expect_app=""):
        # Mirrors the real signature. It gained expect_app when the focus guard
        # landed, and a one-argument double raised TypeError inside the Step —
        # so every move "failed" and the loop stalled while the loop itself was
        # correct. A double that does not track its contract reports the
        # opposite of the truth.
        state["n"] += 1
        state["pressed"].append((key, expect_app))
        return True

    async def frontmost():
        return "TheThing"

    async def ensure_frontmost(_app=""):
        return True

    async def to_the_front(_app=""):
        return True

    monkeypatch.setattr(sp, "read_screen", read)
    monkeypatch.setattr(sp, "press", press)
    # The host stays out of this. Left real, these call the machine's window
    # server: three seconds a cycle, and applications brought to the front on
    # whatever desktop the suite happens to be running on.
    monkeypatch.setattr(sp, "_frontmost", frontmost)
    monkeypatch.setattr(sp, "_ensure_frontmost", ensure_frontmost)
    monkeypatch.setattr(sp, "_bring_the_thing_back_to_the_front", to_the_front)
    return state


async def _alternating(observation):
    text = observation["text"]
    n = int(text.split()[-1]) if text.startswith("count") else 0
    return {"key": "right" if n % 2 == 0 else "left", "because": f"count is {n}"}


def _run(**kw):
    defaults = {"goal": "reach DONE", "success_when": r"\bDONE\b", "narrate": False}
    return asyncio.run(sp.pursue_on_screen(**{**defaults, **kw}))


def test_she_keeps_going_until_the_screen_says_it_is_done(screen):
    result = _run(policy=_alternating, max_cycles=30)

    assert result["completed"] is True
    assert result["outcome"] == "goal_reached"
    assert screen["n"] >= 4


def test_each_move_is_decided_from_the_current_reading(screen):
    """Not from a plan: the policy sees the screen before choosing."""
    result = _run(policy=_alternating, max_cycles=30)

    assert [m["key"] for m in result["moves"]] == ["right", "left", "right", "left"]
    assert result["moves"][0]["because"] == "count is 0"


def test_perception_stays_open_for_the_whole_pursuit(screen):
    """The loop acts on what it sees; that is when sight was being throttled."""
    _run(policy=_alternating, max_cycles=30)

    assert screen["saw_demand"] is True
    assert not perception_is_demanded()


def test_an_unreachable_goal_is_bounded(screen):
    result = _run(success_when="IMPOSSIBLE_TOKEN", policy=_alternating, max_cycles=6)

    assert result["completed"] is False
    assert result["outcome"] == "out_of_cycles"
    assert result["cycles"] == 6


def test_no_policy_stops_when_judgement_is_out_of_reach(screen, monkeypatch):
    """It stops rather than pressing something for no reason.

    This asserted `no_move_available` after four cycles, which was true when
    the no-policy path returned None. It does not any more: with no policy she
    reads, learns, and deliberates, and a deliberation that reaches a move IS
    a decision — so a run against a screen that keeps changing correctly
    spends its budget. What must not happen is acting once her judgement is
    unavailable, and that outcome is named.
    """
    import core.agency.deliberate_action as deliberate_action

    class _Unreached:
        reached = False
        reason = "no judgement is available here"
        chosen = None
        rationale = ""

    async def unreachable(*_args, **_kwargs):
        return _Unreached()

    monkeypatch.setattr(deliberate_action, "deliberate", unreachable)

    result = _run(success_when="NOPE", policy=None, max_cycles=50)

    assert result["outcome"] == "cannot_decide"
    assert result["cannot_decide"] == "no judgement is available here"
    assert not result["moves"], "she pressed something without deciding to"


def test_no_policy_spends_its_budget_rather_than_running_forever(screen):
    """Every move it does make carries the reason it was made."""
    result = _run(success_when="NOPE", policy=None, max_cycles=6)

    assert result["outcome"] in {"out_of_cycles", "cannot_decide", "stalled"}
    assert result["cycles"] <= 6
    for move in result["moves"]:
        assert move["because"], move


@pytest.mark.parametrize("forbidden", ["cmd+q", "ctrl+c", "delete", "f13", ""])
def test_a_policy_cannot_press_a_key_outside_the_allowed_set(screen, forbidden):
    """A loop that can press anything can press ⌘Q."""

    async def rogue(_observation):
        return {"key": forbidden, "because": "should never run"}

    result = _run(success_when="NOPE", policy=rogue, max_cycles=10)

    assert result["moves"] == []
    assert screen["pressed"] == []


def test_success_is_judged_on_what_was_read_not_on_what_was_done(screen):
    """An action that ran is not an action that worked."""
    assert sp.goal_reached({"text": "build FAILED"}, "FAILED") is True
    assert sp.goal_reached({"text": "still running"}, "FAILED") is False
    assert sp.goal_reached({"text": ""}, "FAILED") is False


def test_a_malformed_success_pattern_falls_back_to_plain_text():
    """An unbalanced bracket must not take the whole pursuit down."""
    assert sp.goal_reached({"text": "score [4096]"}, "[4096") is True


def test_a_wedged_capture_ends_the_cycle_rather_than_acting_blind(monkeypatch):
    async def hang(app_name="", over=None):
        await asyncio.sleep(30)

    monkeypatch.setattr(sp, "read_screen", hang)
    monkeypatch.setattr(sp, "OBSERVE_TIMEOUT_S", 0.05)

    called: list[str] = []

    async def policy(_o):
        called.append("decided")
        return {"key": "right"}

    result = _run(policy=policy, max_cycles=3)

    assert called == [], "a blind loop must not keep pressing keys"
    assert result["completed"] is False


def test_a_policy_that_raises_does_not_kill_the_run(screen):
    async def broken(_observation):
        raise ValueError("policy exploded")

    result = _run(success_when="NOPE", policy=broken, max_cycles=10)

    assert result["outcome"] == "no_move_available"


# ── Position-scoped goals ─────────────────────────────────────────────────
#
# Measured on the real play2048.co, 2026-08-18. The word being waited for is
# all over the furniture:
#
#     y=0.043  '2048'              (browser tab)
#     y=0.137  '= 2048'            (page heading)
#     y=0.141  'WELCOME TO 2048!'  (welcome modal)
#     y=0.503  '2'                 (a board tile)
#     y=0.611  '2'                 (a board tile)
#
# A whole-reading match for "2048" therefore succeeds before a single move is
# made. Nothing about this is 2048-specific: any page whose chrome repeats the
# word being waited for has it — a build log in a window titled with the
# branch, a progress dialog in an app whose name contains "complete".

REAL_PAGE = {
    "text": "2048 = 2048 WELCOME TO 2048! New Game 2 2",
    "layout": [
        {"text": "2048", "center_y": 0.043},
        {"text": "= 2048", "center_y": 0.137},
        {"text": "WELCOME TO 2048!", "center_y": 0.141},
        {"text": "New Game", "center_y": 0.148},
        {"text": "2", "center_y": 0.503},
        {"text": "2", "center_y": 0.611},
    ],
}

BOARD_BAND = {"region_top": 0.25, "region_bottom": 0.85}


def test_a_whole_screen_match_declares_victory_on_the_page_title():
    """The defect, pinned: this is why a band is needed at all."""
    assert sp.goal_reached(REAL_PAGE, r"\b2048\b") is True


def test_scoping_to_the_board_rejects_the_title():
    assert sp.goal_reached(REAL_PAGE, r"\b2048\b", **BOARD_BAND) is False


def test_a_real_tile_inside_the_band_wins():
    page = {**REAL_PAGE, "layout": REAL_PAGE["layout"] + [{"text": "2048", "center_y": 0.55}]}

    assert sp.goal_reached(page, r"\b2048\b", **BOARD_BAND) is True


def test_a_band_without_geometry_refuses_rather_than_ignoring_it():
    """Falling back to flat text would discard the constraint that mattered."""
    assert sp.goal_reached({"text": "2048", "layout": []}, r"\b2048\b", **BOARD_BAND) is False


def test_an_inverted_band_is_read_as_a_band():
    page = {**REAL_PAGE, "layout": REAL_PAGE["layout"] + [{"text": "2048", "center_y": 0.55}]}

    assert sp.goal_reached(page, r"\b2048\b", region_top=0.85, region_bottom=0.25) is True


def test_the_default_is_still_the_whole_screen():
    """A caller that does not care about position must not have to say so."""
    assert sp.goal_reached({"text": "DONE", "layout": []}, "DONE") is True


def test_the_band_reaches_the_loop_and_is_reported(screen):
    """Wiring: a band that the loop ignores is worse than no band."""
    result = _run(
        success_when=r"\bDONE\b", policy=_alternating, max_cycles=4, **BOARD_BAND
    )

    assert result["success_region"] == [0.25, 0.85]
    # The fake screen returns no layout, so a banded goal can never be met —
    # which is the honest outcome, not a silent fall back to the flat text.
    assert result["completed"] is False


def test_every_keystroke_is_aimed_at_the_target_app(screen):
    """A loop that acts on what it sees must aim at the window it looked at.

    Measured live 2026-08-18: a run opened play2048.co in Chrome, read the
    board correctly, and sent its arrow keys to whatever the person had clicked
    since — reported as success, with the board untouched. hotkey delivers to
    whatever is frontmost, so an unaimed keystroke has no address.
    """
    _run(policy=_alternating, max_cycles=30, target_app="Google Chrome")

    assert screen["pressed"], "no keys were pressed"
    assert all(app == "Google Chrome" for _key, app in screen["pressed"]), screen["pressed"]


def test_the_target_app_is_reported_on_the_receipt(screen):
    result = _run(policy=_alternating, max_cycles=30, target_app="Preview")

    assert result["target_app"] == "Preview"


def test_a_run_with_no_named_app_still_knows_what_it_is_typing_into(screen):
    """She can see the screen, so she is never typing into nothing.

    This asserted the opposite: that a run given no application presses its
    keys with nothing bound to receive them. That is not a gentler kind of
    aiming, it is the unaimed case every guard here exists for — measured
    live, thirty-five moves of a game played into a chat window, every one
    reported as a success. Where she was not told which application to drive,
    the one she is looking at is the answer, and it is the same one for every
    keystroke of the run."""
    _run(policy=_alternating, max_cycles=30)

    aimed = {app for _key, app in screen["pressed"]}
    assert screen["pressed"], "no keys were pressed"
    assert len(aimed) == 1, f"her keystrokes went to more than one place: {aimed}"
    assert aimed != {""}, "she pressed keys with nothing bound to receive them"


# ── Clearing what blocks the content ──────────────────────────────────────

def _blocked_screen(monkeypatch, labels, *, tiles_after_clear=True):
    """A screen that shows a modal until something dismisses it."""
    state = {"cleared": False, "pressed": [], "clicks": []}

    async def read(app_name="", over=None):
        if state["cleared"]:
            return {"ok": True, "text": "DONE", "layout": [{"text": "DONE", "center_y": 0.5}]}
        return {
            "ok": True,
            "text": "We use cookies " + " ".join(labels),
            "layout": [
                {"text": label, "center_x": 0.5, "center_y": 0.8 + i / 100}
                for i, label in enumerate(labels)
            ],
        }

    async def press(key, *, expect_app=""):
        state["pressed"].append((key, expect_app))
        if key == "escape":
            state["cleared"] = True
        return True

    async def click(x, y, *, expect_app="", bounds=None):
        # Mirrors click_normalized, which gained `bounds` when perception was
        # scoped to a window and 0..1 stopped meaning "of the display". A
        # double that lags the signature raises inside the Step, so the action
        # "fails" and the loop looks broken while it is correct — the fourth
        # time that happened today.
        state["clicks"].append((x, y, expect_app, tuple(bounds or ())))
        state["cleared"] = True
        return True

    monkeypatch.setattr(sp, "read_screen", read)
    monkeypatch.setattr(sp, "press", press)
    monkeypatch.setattr(sp, "click_normalized", click)
    return state


async def _never(_observation):
    return None


def test_a_consent_banner_is_cleared_by_declining(monkeypatch):
    """The loop clears blockers before deciding anything else."""
    state = _blocked_screen(monkeypatch, ["Accept All", "Reject All"])

    result = asyncio.run(
        sp.pursue_on_screen(
            goal="get past the banner",
            success_when="DONE",
            policy=_never,
            max_cycles=6,
            narrate=False,
            target_app="Google Chrome",
        )
    )

    assert state["clicks"], "the banner was never dismissed"
    assert result["completed"] is True


def test_dismissal_clicks_are_aimed_like_every_other_input(monkeypatch):
    state = _blocked_screen(monkeypatch, ["Accept All", "Reject All"])

    asyncio.run(
        sp.pursue_on_screen(
            goal="g", success_when="DONE", policy=_never, max_cycles=6,
            narrate=False, target_app="Preview",
        )
    )

    assert all(app == "Preview" for _x, _y, app, _bounds in state["clicks"])


def test_a_modal_with_no_safe_label_is_escaped(monkeypatch):
    """The real play2048 case: neither dismissive nor accepting controls.

    The modal's own prose is POSITIONED, as it is in a real reading — hints are
    counted from placed text so that an app's toolbar cannot masquerade as a
    dialog, and a fixture that puts the prose only in the flat string is
    describing a screen that does not occur.
    """
    state = _blocked_screen(
        monkeypatch,
        ["WELCOME TO THIS", "Would you like a tutorial?", "Play Tutorial", "New Game"],
    )

    asyncio.run(
        sp.pursue_on_screen(
            goal="g", success_when="DONE", policy=_never, max_cycles=6,
            narrate=False, target_app="Google Chrome",
        )
    )

    assert any(key == "escape" for key, _app in state["pressed"])


def test_a_consent_wall_is_left_for_the_person(monkeypatch):
    """Acceptance-only must never be clicked away by the loop."""
    state = _blocked_screen(monkeypatch, ["I Agree"])

    result = asyncio.run(
        sp.pursue_on_screen(
            goal="g", success_when="DONE", policy=_never, max_cycles=5,
            narrate=False, target_app="Google Chrome",
        )
    )

    assert state["clicks"] == []
    assert not any(key == "escape" for key, _app in state["pressed"])
    assert result["completed"] is False


def test_clearing_a_blocker_happens_before_the_policy_is_asked(monkeypatch):
    """A reading of a dialog is not a reading of the task."""
    screen_state = _blocked_screen(monkeypatch, ["Accept All", "Reject All"])
    asked: list[str] = []

    async def policy(observation):
        asked.append(observation["text"])
        return None

    asyncio.run(
        sp.pursue_on_screen(
            goal="g", success_when="DONE", policy=policy, max_cycles=6,
            narrate=False, target_app="Google Chrome",
        )
    )

    assert all("cookies" not in text for text in asked), asked
    assert screen_state["clicks"], "nothing dismissed the dialog"


@pytest.fixture(autouse=True)
def _nothing_above_her_work(monkeypatch):
    """Nothing is over her window, without asking the window server.

    The loop asks what is above her work every cycle, and a test that does
    not answer sends that question to the real window server — several
    seconds a cycle, forty cycles, on a machine that has a screen. This file
    is about what the loop decides, not about what is actually on this
    display.
    """

    async def above(_mine, over=None):
        return ()

    async def on_top(_mine, over=None):
        return ""

    monkeypatch.setattr(sp, "_everything_on_top", above, raising=False)
    monkeypatch.setattr(sp, "_whats_on_top", on_top, raising=False)

    # And the wait for the screen to answer.
    #
    # An act that moves a surface is done when the surface says so, not when
    # the keystroke returns, so the loop waits for the reading to change and
    # then to stop changing. A double whose screen never changes waits the
    # whole patience every cycle: forty cycles at four seconds is ten minutes
    # for a test about what the loop decides. Shortened, not removed — the
    # waiting is the behaviour, and a test that skipped it would pass on a
    # loop that had stopped waiting.
    monkeypatch.setattr(sp, "_how_long_to_wait", lambda: 0.05)


def test_a_blocker_that_will_not_clear_is_reported_not_repeated(monkeypatch):
    """A dismissal that does not work must not be tried forever.

    Measured live: play2048.co's welcome modal ignores Escape. The Step
    succeeded every cycle — the key WAS pressed — so the loop counted verified
    progress and spent forty cycles pressing Escape, making zero moves. A
    dismissal Step's success means the action ran, which is exactly the
    distinction this loop exists to enforce and which the blocker path was
    itself exempt from.
    """
    state = {"pressed": []}

    async def read(app_name="", over=None):
        return {
            "ok": True,
            "text": "welcome tutorial would you like",
            "layout": [
                {"text": "WELCOME", "center_y": 0.30},
                {"text": "Would you like a tutorial?", "center_y": 0.35},
                {"text": "Play Tutorial", "center_y": 0.40},
            ],
        }

    async def press(key, *, expect_app=""):
        state["pressed"].append(key)
        return True  # succeeds, and changes nothing — the live case

    monkeypatch.setattr(sp, "read_screen", read)
    monkeypatch.setattr(sp, "press", press)

    result = asyncio.run(
        sp.pursue_on_screen(
            goal="play", success_when="NEVER", policy=_alternating,
            max_cycles=40, narrate=False, target_app="Google Chrome",
        )
    )

    assert result["outcome"] == "blocked_by_overlay"
    assert result["blocked_by"], "the obstacle must be named"
    assert len(state["pressed"]) <= sp.MAX_BLOCKER_ATTEMPTS + 1, state["pressed"]
    assert result["cycles"] < 40, "it must stop early rather than burn the budget"


def test_a_blocker_that_clears_does_not_count_against_the_budget(monkeypatch):
    """Attempts reset once the way is clear, so a later dialog gets its own tries."""
    screen_state = _blocked_screen(monkeypatch, ["Accept All", "Reject All"])

    result = asyncio.run(
        sp.pursue_on_screen(
            goal="g", success_when="DONE", policy=_never, max_cycles=8,
            narrate=False, target_app="Google Chrome",
        )
    )

    assert result["completed"] is True
    assert result.get("blocked_by", "") == ""
    # The screen state was captured and never read, so a run that reached the
    # goal without ever dismissing anything would have passed this.
    assert len(screen_state["clicks"]) == 1, screen_state["clicks"]
