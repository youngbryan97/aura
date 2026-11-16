"""The pointing half of companion mode had no wire in it.

``screen_highlight`` located an element and asked a ``desktop_overlay``
runtime service to draw a box around it. Nothing registered that service and
no surface ever forwarded the message, so ``get_runtime_service`` returned
None, ``_draw`` returned False, and every highlight in the live runtime came
back "overlay declined". The launcher's ``showHighlight`` — the code that
actually draws — was unreachable from the day it was written.

Nothing caught it because the tests exercised the parts: the locator was
tested, the refusals were tested, the rate limiter was tested by calling the
rate limiter. No test asked whether a rectangle could travel from a decision
in Python to a window on the screen.

These do. They drive the seam end to end — request, queue, poll, forward —
and they pin the two properties that make the seam honest rather than merely
present:

  * she must not claim to have pointed when nothing could draw. "I've
    highlighted it" over a screen with no highlight sends someone hunting for
    a box that was never there, which is worse than describing it in words;
  * a rectangle is collected exactly once and never after it has gone stale.
    A box around where something USED to be points at the wrong thing with
    full confidence.
"""
from __future__ import annotations

import asyncio

import pytest

from core.perception.ambient_presence import AmbientPresence, PresenceMode
from core.perception.desktop_overlay import BubbleOverlay, install_desktop_overlay
from core.runtime.desktop_objective_intent import asks_to_be_shown_where


@pytest.fixture
def presence():
    instance = AmbientPresence()
    instance.set_mode(PresenceMode.BUBBLE)
    return instance


@pytest.fixture
def attached(presence):
    """A presence with a live bubble polling it."""
    presence.note_surface_poll("native-bubble")
    return presence


# ───────────────────────────────── nothing can draw, so she does not claim to


def test_no_surface_means_no_highlight(presence):
    """A runtime with no bubble cannot draw, and says so rather than lying."""
    assert presence.drawing_surface_attached() is False
    assert presence.request_highlight(10, 10, 100, 40, 3.0) is False
    assert presence.take_highlight() is None


def test_a_stale_surface_does_not_count_as_attached(presence, monkeypatch):
    """A launcher that quit ten minutes ago is not somewhere to draw."""
    presence.note_surface_poll("native-bubble")
    monkeypatch.setattr(
        "core.perception.ambient_presence._SURFACE_ALIVE_S", 0.0
    )

    assert presence.drawing_surface_attached() is False
    assert presence.request_highlight(10, 10, 100, 40, 3.0) is False


def test_only_a_host_that_draws_counts_as_a_drawing_surface(presence):
    """The chat window reads the same state and cannot draw an overlay.

    Counting its read as "a surface is attached" would let her claim a
    highlight that no host was listening for.
    """
    presence.note_surface_poll("companion")
    assert presence.drawing_surface_attached() is False

    presence.note_surface_poll("native-bubble")
    assert presence.drawing_surface_attached() is True


def test_her_launcher_draws_while_her_window_is_up_and_the_bubble_is_not():
    """15 November 2026: a highlight asked for from her window was declined, the bubble being ordered out and silent.

    Her launcher reads her state itself then, as native-host, and collects the rectangle as the bubble would.
    """
    from pathlib import Path

    presence = AmbientPresence()
    presence.set_mode(PresenceMode.WINDOW)
    presence.note_surface_poll("native-host")
    assert presence.request_highlight(12, 34, 200, 50, 2.5) is True
    assert presence.state()["highlight"] is None, "the chat window still cannot take it"
    assert presence.state(surface="native-host")["highlight"]["x"] == 12
    launcher = Path("scripts/AuraLauncher.swift").read_text(encoding="utf-8")
    poll = launcher.split("private func collectAHighlightWhileHerWindowIsUp()", 1)[1].split("private func", 1)[0]
    assert "surface=native-host" in poll and "self.showHighlight(" in poll
    assert "desktopWindow?.isVisible == true" in poll and "bubblePanel?.isVisible != true" in poll


def test_hidden_means_she_does_not_draw_either(attached):
    """Hidden is not a display setting. An overlay is maximally present."""
    attached.set_mode(PresenceMode.HIDDEN)

    assert attached.request_highlight(10, 10, 100, 40, 3.0) is False


# ─────────────────────────────────────────── the rectangle actually travels


def test_a_queued_rectangle_reaches_the_polling_bubble(attached):
    assert attached.request_highlight(12, 34, 200, 50, 2.5) is True

    state = attached.state(surface="native-bubble")

    assert state["highlight"] is not None
    assert state["highlight"]["x"] == 12
    assert state["highlight"]["width"] == 200
    assert state["highlight"]["seconds"] == 2.5


def test_a_rectangle_is_drawn_once_not_every_poll(attached):
    attached.request_highlight(12, 34, 200, 50, 2.5)

    assert attached.state(surface="native-bubble")["highlight"] is not None
    assert attached.state(surface="native-bubble")["highlight"] is None, (
        "the same rectangle would be redrawn on every poll, forever"
    )


def test_the_chat_window_cannot_swallow_the_rectangle(attached):
    """The restrained window polls the same endpoint and must not consume it.

    If it did, the bubble would never receive the highlight and she would
    have reported pointing at something nobody drew.
    """
    attached.request_highlight(12, 34, 200, 50, 2.5)

    assert attached.state()["highlight"] is None
    assert attached.state(surface="native-bubble")["highlight"] is not None


def test_a_stale_rectangle_is_dropped_rather_than_drawn(attached, monkeypatch):
    """The screen moved on. A box around where it WAS points at the wrong thing."""
    attached.request_highlight(12, 34, 200, 50, 2.5)
    monkeypatch.setattr("core.perception.ambient_presence._HIGHLIGHT_TTL_S", -1.0)

    assert attached.state(surface="native-bubble")["highlight"] is None


def test_polling_the_state_is_what_marks_the_surface_alive(presence):
    """The liveness signal is the poll the bubble already makes.

    No heartbeat endpoint, no second lane — if this stops being true the
    overlay silently stops being offered, so it is pinned.
    """
    assert presence.drawing_surface_attached() is False

    presence.state(surface="native-bubble")

    assert presence.drawing_surface_attached() is True


# ────────────────────────────────────────────────── the service is reachable


def test_the_overlay_service_registers(monkeypatch):
    """The registration that never happened.

    ``install_desktop_overlay`` returns False when no container sink is
    installed, which is the headless case; what must never regress is the
    call existing at all on the companion boot path.
    """
    registered: dict[str, object] = {}

    def _sink(name, instance, required, metadata):
        registered[name] = instance

    monkeypatch.setattr(
        "core.runtime.service_registry._service_registration_sink", _sink
    )

    assert install_desktop_overlay() is True
    assert "desktop_overlay" in registered
    assert hasattr(registered["desktop_overlay"], "show_rect")


def test_show_rect_reports_honestly_when_nothing_can_draw(monkeypatch):
    fresh = AmbientPresence()
    fresh.set_mode(PresenceMode.BUBBLE)
    monkeypatch.setattr(
        "core.perception.ambient_presence.get_ambient_presence", lambda: fresh
    )

    assert BubbleOverlay().show_rect(
        x=1, y=2, width=30, height=40, seconds=3.0
    ) is False


def test_show_rect_queues_when_a_surface_is_listening(monkeypatch):
    fresh = AmbientPresence()
    fresh.set_mode(PresenceMode.BUBBLE)
    fresh.note_surface_poll("native-bubble")
    monkeypatch.setattr(
        "core.perception.ambient_presence.get_ambient_presence", lambda: fresh
    )

    assert BubbleOverlay().show_rect(
        x=1, y=2, width=30, height=40, seconds=3.0
    ) is True
    assert fresh.take_highlight()["width"] == 30


def test_the_boot_path_installs_the_overlay():
    """aura_main must actually call it, or the service is absent at runtime."""
    from pathlib import Path

    source = Path("aura_main.py").read_text(encoding="utf-8")

    assert "install_desktop_overlay" in source, (
        "the overlay service is registered nowhere, so every highlight refuses"
    )


# ────────────────────────────────── being asked to point is now recognised


@pytest.mark.parametrize(
    ("question", "needle"),
    [
        ("where is the submit button on my screen?", "submit button"),
        ("show me where the error is", "error"),
        ("point at the failing test", "failing test"),
        ("highlight the merge conflict", "merge conflict"),
        ("which one is the stale one", "stale one"),
    ],
)
def test_a_request_to_point_is_recognised_with_its_target(question, needle):
    assert asks_to_be_shown_where(question) == needle


@pytest.mark.parametrize(
    "question",
    [
        "what is on my screen",
        "read me the error message",
        "tell me about the code",
        "",
    ],
)
def test_an_ordinary_screen_question_is_not_a_request_to_point(question):
    assert asks_to_be_shown_where(question) == ""


# ─────────────────────────── the answer lane asks for it, and reports refusals


def test_the_answer_lane_attaches_the_refusal_not_just_the_success():
    """A highlight that did not draw must reach the answer as a fact.

    Otherwise she says "I've highlighted it" over a screen with no highlight
    on it, which is the one outcome worse than describing it in words.
    """
    import core.perception.screen_highlight as highlight_module
    from core.perception.screen_highlight import HighlightResult
    from core.skills.desktop_task import DesktopTaskSkill

    async def _refuse(needle, *, duration_s=3.0, requested=False):
        return HighlightResult(
            shown=False, reason="could not locate it in the accessibility tree"
        )

    real = highlight_module.highlight
    highlight_module.highlight = _refuse
    try:
        payload = asyncio.run(
            DesktopTaskSkill()._attach_pointing(
                "where is the submit button", {"ok": True}
            )
        )
    finally:
        highlight_module.highlight = real

    assert payload["highlight"]["shown"] is False
    assert "could not locate" in payload["pointing_refused_because"]
    assert payload["pointed_at"] == ""


def test_pointing_never_costs_her_the_answer():
    """The overlay is an enhancement. A broken one must not eat the reply."""
    import core.perception.screen_highlight as highlight_module
    from core.skills.desktop_task import DesktopTaskSkill

    real = highlight_module.highlight

    async def _explode(needle, *, duration_s=3.0, requested=False):
        raise RuntimeError("overlay host died")

    highlight_module.highlight = _explode
    try:
        payload = asyncio.run(
            DesktopTaskSkill()._attach_pointing(
                "show me the submit button", {"ok": True, "observation": "text"}
            )
        )
    finally:
        highlight_module.highlight = real

    assert payload["ok"] is True
    assert payload["observation"] == "text"


# ───────────────────────────────────────── the surface forwards it to the host


def test_the_bubble_forwards_the_rectangle_to_its_host():
    """The browser half. Without this the queue fills and nothing draws.

    Asserted against the source because the alternative is a headless browser
    for four lines of message passing, and what must not regress is precise:
    the bubble polls WITH a surface name and posts the rectangle onward.
    """
    from pathlib import Path

    source = Path("interface/static/bubble.js").read_text(encoding="utf-8")

    assert "native-bubble" in source, (
        "without the surface name nothing marks the bubble as able to draw"
    )
    assert 'action: "highlight"' in source, (
        "the rectangle is collected and then dropped on the floor"
    )
    assert "forwardHighlight" in source


def test_a_highlight_that_raises_is_said_as_not_drawn(monkeypatch):
    """A highlight that raised left nothing on the answer, so she could say she pointed at a box nobody drew."""
    import core.perception.screen_highlight as screen_highlight
    from core.skills.desktop_task import DesktopTaskSkill

    async def broken(*_args, **_kwargs):
        raise RuntimeError("no accessibility tree")

    monkeypatch.setattr(screen_highlight, "highlight", broken)
    payload = asyncio.run(DesktopTaskSkill._attach_pointing(None, "show me where the submit button is", {}))
    assert "no accessibility tree" in payload["pointing_refused_because"]
    assert "highlight" not in payload
