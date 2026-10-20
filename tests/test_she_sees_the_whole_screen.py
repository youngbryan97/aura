"""A reading of the screen is a reading of the SCREEN, not of one window.

Live 2026-08-04 00:20. Chrome was open on half the display showing a YouTube
channel, Finder, Notes, TextEdit and System Settings were open behind, and
asked "can you tell me what you see on the screen right now?" Aura answered:

    Active app: aura-launcher
    Window: Aura Zenith (Completed 1/1 governed desktop steps.)

Objectively incomplete to the person looking at it, and she could see more.

Two causes.

The capture only took a screenshot — and therefore only ran OCR — when the
step's target string happened to contain "screenshot", "ocr", "visual",
"image" or "see". The planner emits read_screen_text with an EMPTY target, so
the ordinary path never captured, screen_text was always "", and the answer
fell back to the frontmost window's title. A capability that depends on
incidental wording is not a capability.

And capture() already collected the whole desk — every window, front to back,
with what covers what — into window_layout. The result builder dropped it.
That is also why "ignore your own window" and "what's behind you" could not be
answered: nothing downstream ever received anything but the frontmost title.
"""
from __future__ import annotations

import pytest

from core.skills.computer_use import ComputerUseSkill


class _Snapshot:
    """The shape capture() returns, with the desk populated and no OCR."""

    active_app = "aura-launcher"
    window_title = "Aura Zenith"
    frontmost_window_bounds = "820,34,908,1039"
    focused_role = ""
    focused_name = ""
    focused_description = ""
    focused_value = ""
    accessibility_text = ""
    screen_text = ""
    screenshot_path = ""
    text_hash = ""
    has_modal = False
    modal_text = ""
    has_loading = False
    timestamp = 0.0
    window_layout = (
        "8 window(s) open, front to back:\n"
        '  1. Aura "Aura Zenith" — fully visible\n'
        '  2. Google Chrome "Kurzgesagt – In a Nutshell - YouTube" — 21% visible, '
        "partly behind Aura"
    )
    open_apps = ("Aura", "Google Chrome", "Finder", "Notes")


class TestTheAnswerDescribesTheDesk:
    def test_other_windows_are_named(self):
        result = ComputerUseSkill._screen_snapshot_result(_Snapshot())
        assert "Google Chrome" in result["text"]
        assert "Kurzgesagt" in result["text"]

    def test_the_layout_reaches_the_caller(self):
        result = ComputerUseSkill._screen_snapshot_result(_Snapshot())
        assert result["window_layout"]
        assert "Google Chrome" in result["open_apps"]

    def test_knowing_the_desk_is_not_a_limited_read(self):
        """Calling a complete layout "limited" told the caller to distrust it."""
        assert ComputerUseSkill._screen_snapshot_result(_Snapshot())["status"] == "ok"

    def test_ocr_text_is_included_when_present(self):
        snap = _Snapshot()
        snap.screen_text = "SUBSCRIBE  Tired of Doomscrolling?"
        result = ComputerUseSkill._screen_snapshot_result(snap)
        assert "Tired of Doomscrolling?" in result["text"]
        assert "Google Chrome" in result["text"], "OCR must not replace the layout"

    def test_a_truly_blind_read_is_still_limited(self):
        snap = _Snapshot()
        snap.window_layout = ""
        snap.open_apps = ()
        assert ComputerUseSkill._screen_snapshot_result(snap)["status"] == "limited"


class TestAScreenReadActuallyCaptures:
    def test_the_read_path_does_not_sniff_the_target_string(self):
        import inspect

        source = inspect.getsource(ComputerUseSkill)
        assert '"screenshot", "ocr", "visual", "image", "see"' not in source, (
            "capturing only when the target happens to say 'see' is why the "
            "ordinary read never ran OCR"
        )
        assert source.count("capture(save_screenshot=True)") >= 1


class TestLookingPastHerOwnWindow:
    """"Ignore your own window" is the same question as "what's behind it"."""

    ASKED = (
        "ignore your own window, what else is on the screen?",
        "excluding your window, what do you see?",
        "what's on the screen apart from your own window",
        "what else is on my screen?",
        "what is behind you?",
        "what's behind your window?",
    )

    @pytest.mark.parametrize("message", ASKED)
    def test_each_phrasing_is_answered_from_the_layout(self, message):
        from core.conversation.response_reliability import occluded_screen_view_floor

        answer = occluded_screen_view_floor(message)
        assert answer.strip(), f"no answer for {message!r}"

    @pytest.mark.parametrize("message", [
        "what is the weather today", "open Chrome",
        # LIVE 2026-10-20: the answer before the video and every remark during it became a list of windows.
        "Let's watch something together. Go to YouTube and find a random video from Allen Pan, under 15 minutes. "
        "Before it starts, tell me what you think it'll be about.",
        "what else did you notice in that video?",
    ])
    def test_unrelated_requests_are_not_swallowed(self, message):
        from core.conversation.response_reliability import occluded_screen_view_floor

        assert occluded_screen_view_floor(message) == ""


class TestTheArrangementQuestionGoesToTheFloor:
    """A screen CAPTURE reads what is visible. "What else is on the screen?"
    is a question about the arrangement, and sending it down the capture lane
    returned a raw OCR dump of whichever window happened to be readable.
    Measured live 2026-08-04.
    """

    ARRANGEMENT = (
        "ignore your own window — what else is on the screen?",
        "excluding your window, what do you see?",
        "what's behind your window?",
        "what's on the screen apart from your own window",
    )

    CAPTURE = (
        "what is on my screen right now",
        "what's on my screen",
        "take a screenshot",
    )

    @pytest.mark.parametrize("message", ARRANGEMENT)
    def test_the_desktop_lane_declines_an_arrangement_question(self, message):
        from core.runtime.desktop_objective_intent import looks_like_desktop_objective

        assert looks_like_desktop_objective(message) is False

    @pytest.mark.parametrize("message", ARRANGEMENT)
    def test_the_floor_answers_it_instead(self, message):
        from core.conversation.response_reliability import occluded_screen_view_floor

        assert occluded_screen_view_floor(message).strip()

    @pytest.mark.parametrize("message", CAPTURE)
    def test_a_plain_screen_read_still_captures(self, message):
        """Only the arrangement question is redirected."""
        from core.runtime.desktop_objective_intent import looks_like_desktop_objective

        assert looks_like_desktop_objective(message) is True

    def test_real_desktop_work_still_routes(self):
        from core.runtime.desktop_objective_intent import looks_like_desktop_objective

        assert looks_like_desktop_objective("open Chrome and close the window")

    def test_the_two_layers_share_one_definition(self):
        import inspect

        from core.conversation import response_reliability
        from core.runtime import desktop_objective_intent

        for module in (response_reliability, desktop_objective_intent):
            assert "occluded_view_intent" in inspect.getsource(module)
