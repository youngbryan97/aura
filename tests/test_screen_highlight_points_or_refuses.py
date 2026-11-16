"""Pointing at the wrong thing is worse than not pointing.

"Which one is the failing test?" sometimes has a better answer than a
paragraph: a rectangle around it. But a rectangle carries the authority of a
measurement, so it has to BE one.

The rules these tests drive:

  * coordinates come from the accessibility tree or nowhere. A guess from
    character offsets in an OCR dump is a confident wrong answer wearing the
    appearance of a measurement, and "could not locate it" is the honest
    outcome rather than a degraded one;
  * she does not point unasked. An unrequested rectangle appearing on
    someone's screen is a jump scare, not help;
  * a rectangle covering most of the screen points at nothing;
  * private windows are not drawn on, because drawing on one means having
    located something inside it.
"""
from __future__ import annotations

import asyncio

import pytest

from core.perception.screen_highlight import (
    MAX_SCREEN_FRACTION,
    HighlightResult,
    Rect,
    _suppressed,
    highlight,
    locate_on_screen,
    rate_limited,
)

#: Captured before the autouse fixture replaces the module attribute.
_REAL_SUPPRESSION_CHECK = _suppressed


@pytest.fixture(autouse=True)
def _allow(monkeypatch):
    monkeypatch.setattr("core.perception.screen_highlight._suppressed", lambda: False)
    monkeypatch.setattr(
        "core.senses.screen_context.frontmost_window_hint", lambda: ("Terminal", "zsh")
    )
    # highlight() now enforces its own rate limit, so the module global is a
    # real piece of cross-test state: without this reset the SECOND case in
    # the file refuses for being too soon after the first and every assertion
    # about a different refusal reason becomes accidentally true.
    monkeypatch.setattr("core.perception.screen_highlight._LAST_HIGHLIGHT_AT", 0.0)


def _run(**kwargs):
    return asyncio.run(highlight(**kwargs))


# ────────────────────────────────────────────── she does not point unasked


def test_an_unrequested_highlight_is_refused():
    result = _run(needle="anything", requested=False)

    assert result.shown is False
    assert "asked" in result.reason


def test_suppressed_proactivity_stops_her_drawing(monkeypatch):
    """The same suppression that stops her speaking."""
    monkeypatch.setattr("core.perception.screen_highlight._suppressed", lambda: True)

    result = _run(needle="anything", requested=True)

    assert result.shown is False
    assert "suppressed" in result.reason


def test_the_gate_this_overlay_asks_actually_exists():
    """Reachable, not just fail-closed when unreachable.

    The earlier version of this test deleted the symbol from a module that
    never defined it and asserted the refusal — which passed for the wrong
    reason. Every highlight refused in the live runtime while this was green.
    """
    from core.brain.initiative_engine import proactivity_suppressed_now

    assert callable(proactivity_suppressed_now)


def test_the_gate_is_not_stuck_shut(monkeypatch):
    """With permission granted the gate must open, or nothing can ever draw."""
    from core.container import ServiceContainer

    class _PermittingOrchestrator:
        _suppress_unsolicited_proactivity_until = 0.0

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: (
                _PermittingOrchestrator() if name == "orchestrator" else default
            )
        ),
    )

    assert _REAL_SUPPRESSION_CHECK() is False


def test_an_unavailable_permission_check_refuses(monkeypatch):
    """Fail closed. An unreachable authority is not permission."""
    from core.container import ServiceContainer

    def _explode(name, default=None):
        raise RuntimeError("container unavailable")

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(_explode))

    assert _REAL_SUPPRESSION_CHECK() is True


# ──────────────────────────────── it measures, or it says it could not


def test_an_unlocatable_element_is_refused_not_guessed(monkeypatch):
    monkeypatch.setattr(
        "core.perception.screen_highlight.locate_on_screen", lambda needle: None
    )

    result = _run(needle="some text", requested=True)

    assert result.shown is False
    assert "could not locate" in result.reason
    assert result.rect is None, (
        "a rectangle was produced for something that could not be located"
    )


def test_a_located_element_is_drawn(monkeypatch):
    drawn = {}

    async def _draw(rect, seconds):
        drawn["rect"] = rect
        drawn["seconds"] = seconds
        return True

    monkeypatch.setattr(
        "core.perception.screen_highlight.locate_on_screen",
        lambda needle: Rect(x=100, y=200, width=300, height=40),
    )
    monkeypatch.setattr("core.perception.screen_highlight._screen_area", lambda: 2_000_000)
    monkeypatch.setattr("core.perception.screen_highlight._draw", _draw)

    result = _run(needle="test_retry_budget", requested=True)

    assert result.shown is True
    assert drawn["rect"].width == 300


def test_a_rectangle_covering_the_screen_points_at_nothing(monkeypatch):
    monkeypatch.setattr(
        "core.perception.screen_highlight.locate_on_screen",
        lambda needle: Rect(x=0, y=0, width=1900, height=1000),
    )
    monkeypatch.setattr("core.perception.screen_highlight._screen_area", lambda: 1920 * 1080)

    result = _run(needle="everything", requested=True)

    assert result.shown is False
    assert "covers most of the screen" in result.reason


@pytest.mark.parametrize(
    "rect",
    [
        Rect(x=0, y=0, width=0, height=0),
        Rect(x=0, y=0, width=2, height=40),
        Rect(x=0, y=0, width=-10, height=10),
    ],
)
def test_a_degenerate_rectangle_is_not_a_location(rect):
    assert rect.is_usable is False


# ───────────────────────────────────────────────────────── privacy holds


def test_a_private_foreground_is_never_drawn_on(monkeypatch):
    monkeypatch.setattr(
        "core.senses.screen_context.frontmost_window_hint",
        lambda: ("Google Chrome", "Bank — Incognito"),
    )
    located = {"called": False}

    def _locate(needle):
        located["called"] = True
        return Rect(x=1, y=1, width=50, height=20)

    monkeypatch.setattr("core.perception.screen_highlight.locate_on_screen", _locate)

    result = _run(needle="balance", requested=True)

    assert result.shown is False
    assert "private window" in result.reason
    assert located["called"] is False, (
        "the element was located inside a private window before the refusal"
    )


def test_the_refusal_does_not_name_the_private_window(monkeypatch):
    monkeypatch.setattr(
        "core.senses.screen_context.frontmost_window_hint",
        lambda: ("Google Chrome", "Chase Bank — Incognito"),
    )

    assert "Chase" not in str(_run(needle="x", requested=True).to_dict())


# ─────────────────────────────────────────────── bounds and degradation


def test_the_duration_is_bounded(monkeypatch):
    seen = {}

    async def _draw(rect, seconds):
        seen["seconds"] = seconds
        return True

    monkeypatch.setattr(
        "core.perception.screen_highlight.locate_on_screen",
        lambda needle: Rect(x=10, y=10, width=60, height=20),
    )
    monkeypatch.setattr("core.perception.screen_highlight._screen_area", lambda: 2_000_000)
    monkeypatch.setattr("core.perception.screen_highlight._draw", _draw)

    _run(needle="x", requested=True, duration_s=9999.0)

    assert seen["seconds"] <= 10.0, "a highlight could outlive the answer it explains"


def test_no_overlay_host_refuses_rather_than_claiming_it_drew(monkeypatch):
    """Running headless there is nothing to draw on, and saying so is honest."""
    monkeypatch.setattr(
        "core.perception.screen_highlight.locate_on_screen",
        lambda needle: Rect(x=10, y=10, width=60, height=20),
    )
    monkeypatch.setattr("core.perception.screen_highlight._screen_area", lambda: 2_000_000)
    monkeypatch.setattr(
        "core.runtime.service_registry.get_runtime_service",
        lambda name, default=None: default,
    )

    result = _run(needle="x", requested=True)

    assert result.shown is False


def test_a_broken_overlay_degrades_to_words(monkeypatch):
    async def _boom(rect, seconds):
        raise RuntimeError("overlay surface gone")

    monkeypatch.setattr(
        "core.perception.screen_highlight.locate_on_screen",
        lambda needle: Rect(x=10, y=10, width=60, height=20),
    )
    monkeypatch.setattr("core.perception.screen_highlight._screen_area", lambda: 2_000_000)
    monkeypatch.setattr("core.perception.screen_highlight._draw", _boom)

    result = _run(needle="x", requested=True)

    assert result.shown is False
    assert "overlay failed" in result.reason


def test_repeated_highlights_are_rate_limited():
    """A rectangle flashing repeatedly is flickering, not pointing."""
    rate_limited(min_gap_s=0.0)
    assert rate_limited(min_gap_s=10.0) is True


def test_a_short_needle_is_not_searched_for():
    """One character matches everywhere, so it locates nothing in particular."""
    assert locate_on_screen("a") is None
    assert locate_on_screen("") is None


def test_the_result_serialises_for_a_receipt():
    payload = HighlightResult(
        shown=True, reason="drawn", rect=Rect(1.234, 2.345, 30.0, 12.0), matched_text="x"
    ).to_dict()

    assert payload["rect"]["x"] == 1.2
    assert payload["shown"] is True


def test_the_whole_front_window_is_searched_and_an_exact_name_wins(monkeypatch):
    """LIVE 15 November 2026, Finder in front: "Downloads", "Applications" and "Search" were each not found, because
    only the window's own children were asked, and in a real window those are groups and toolbars."""
    from types import SimpleNamespace

    import AppKit
    import ApplicationServices

    def node(children=(), **names):
        return SimpleNamespace(children=list(children), names=names)

    exact = node(AXTitle="Downloads", AXPosition=(714, 530), AXSize=(56, 18))
    longer = node(AXTitle="Downloads folder settings", AXPosition=(10, 10), AXSize=(200, 20))
    window = node([node([node([longer]), node([exact])])])
    app = node(AXFocusedWindow=window)

    def copy(element, name, _out):
        if name == "AXChildren":
            return (0, element.children) if element.children else (1, None)
        value = element.names.get(name)
        return (0, value) if value is not None else (1, None)

    def value_of(value, kind, _out):
        x, y = value
        return True, SimpleNamespace(x=x, y=y, width=x, height=y)

    monkeypatch.setattr(ApplicationServices, "AXUIElementCreateApplication", lambda pid: app)
    monkeypatch.setattr(ApplicationServices, "AXUIElementCopyAttributeValue", copy)
    monkeypatch.setattr(ApplicationServices, "AXValueGetValue", value_of)
    monkeypatch.setattr(AppKit, "NSWorkspace", SimpleNamespace(sharedWorkspace=lambda: SimpleNamespace(
        frontmostApplication=lambda: SimpleNamespace(processIdentifier=lambda: 1))))

    found = locate_on_screen("Downloads")
    assert found is not None and (found.x, found.y) == (714.0, 530.0), "three levels down, named exactly"
    assert locate_on_screen("nothing named this") is None


def test_a_name_the_window_does_not_use_finds_nothing_and_a_kind_of_control_is_not_a_name(monkeypatch):
    """"The save button" with no Save on screen drew around another button while help text counted as a name."""
    from types import SimpleNamespace

    import AppKit
    import ApplicationServices

    def node(children=(), **names):
        return SimpleNamespace(children=list(children), names=names)

    share = node(AXDescription="Share", AXRoleDescription="button", AXHelp="this button shares", AXPosition=(1, 1),
                 AXSize=(36, 36))
    back = node(AXDescription="back", AXRoleDescription="button", AXPosition=(795, 287), AXSize=(36, 36))
    back_row = node(AXTitle="back issues", AXRoleDescription="row", AXPosition=(10, 400), AXSize=(200, 18))
    app = node(AXFocusedWindow=node([node([share, back_row]), node([back])]))

    def copy(element, name, _out):
        if name == "AXChildren":
            return (0, element.children) if element.children else (1, None)
        value = element.names.get(name)
        return (0, value) if value is not None else (1, None)

    monkeypatch.setattr(ApplicationServices, "AXUIElementCreateApplication", lambda pid: app)
    monkeypatch.setattr(ApplicationServices, "AXUIElementCopyAttributeValue", copy)
    monkeypatch.setattr(ApplicationServices, "AXValueGetValue",
                        lambda value, kind, _out: (True, SimpleNamespace(x=value[0], y=value[1], width=value[0],
                                                                          height=value[1])))
    monkeypatch.setattr(AppKit, "NSWorkspace", SimpleNamespace(sharedWorkspace=lambda: SimpleNamespace(
        frontmostApplication=lambda: SimpleNamespace(processIdentifier=lambda: 1))))

    assert locate_on_screen("save button") is None, "no Save on screen: no rectangle around another button"
    found = locate_on_screen("back button")
    assert found is not None and (found.x, found.y) == (795.0, 287.0), "the button named back, not the row"
