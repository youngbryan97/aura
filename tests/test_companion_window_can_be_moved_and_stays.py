"""The companion has to move where she is put, and stay where she is left.

Reported live 2026-08-10, one sitting AFTER the fix that added dragging:
"i still cant drag across the screen" and "when i click on another window,
the companion goes away".

Both surfaces asked for the drag with `-webkit-app-region: drag`, an Electron
property WKWebView does not implement. The declaration was inert on every
surface it appeared on, and the comment beside it described a mechanism that
was never present — which is why the first fix went looking in the wrong
layer and the second report was identical to the first.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUBBLE_JS = (ROOT / "interface/static/bubble.js").read_text(encoding="utf-8")
BUBBLE_HTML = (ROOT / "interface/static/bubble.html").read_text(encoding="utf-8")
COMPANION_HTML = (ROOT / "interface/static/companion_chat.html").read_text(encoding="utf-8")
LAUNCHER = (ROOT / "scripts/AuraLauncher.swift").read_text(encoding="utf-8")


def test_no_surface_relies_on_the_electron_drag_property():
    """It does nothing in WKWebView, and reads as though it does everything."""
    for name, source in (
        ("bubble.html", BUBBLE_HTML),
        ("companion_chat.html", COMPANION_HTML),
    ):
        # Prose explaining why it is gone is fine; a live declaration is not.
        declarations = re.findall(r"^\s*-webkit-app-region\s*:", source, re.MULTILINE)
        assert not declarations, f"{name} still declares -webkit-app-region"


def test_the_bubble_drags_from_anywhere_on_its_surface():
    """The guarantee is the panel's own tracking loop, and the test follows it.

    Three mechanisms have owned this. A JS drag from mousemove deltas could not
    work from a 56x56 web view, because WebKit only synthesises mousemove for
    points inside the view. The pan recognizer that replaced it could not
    either: the bubble is a .nonactivatingPanel and the report was "pretty sure
    this icon in companion mode stopped being draggable". What owns it now is
    AppKit's own event-tracking loop, started from the page's mousedown.

    This test asserted the second mechanism, which the file has not used for
    some time — so it failed while dragging worked, which is the same amount of
    information as not existing, and worse, because it trains the reader to
    ignore it.
    """

    # The page says a gesture began; only the page knows whether the pointer
    # went down on the x or the reply control.
    assert 'postToHost({ action: "dragStart" })' in BUBBLE_JS
    assert 'event.target.closest("#close, #say")' in BUBBLE_JS

    # The host answers with the panel's own tracking loop, in screen
    # coordinates, so the gesture is not bounded by the web view.
    assert 'case "dragStart":' in LAUNCHER
    assert "beginNativeBubbleDrag()" in LAUNCHER
    assert "panel.trackEvents(" in LAUNCHER
    assert "matching: [.leftMouseDragged, .leftMouseUp]" in LAUNCHER

    # The companion window is a different kind of window and keeps the
    # recognizer, with a strip so dragging across the transcript still selects.
    assert "installWindowDrag(on: webView, topStrip:" in LAUNCHER
    assert "final class TopStripPanGestureRecognizer: NSPanGestureRecognizer" in LAUNCHER


def test_a_drag_does_not_also_open_the_chat():
    """A tap opens the chat; a drag moves her. One gesture decides which.

    The tracking loop counts the pointer's travel: under a few pixels of slop
    it dispatches aura-bubble-click and the page opens the chat, and past it
    the panel moves and no click is sent. Nothing in the page competes for the
    gesture — its mousedown starts the host's loop and does no moving itself.
    """

    assert "delaysPrimaryMouseButtonEvents = false" in LAUNCHER
    assert "aura-bubble-click" in LAUNCHER
    assert 'window.addEventListener("aura-bubble-click", openChat)' in BUBBLE_JS

    # Code only. The prose quotes the removed approach verbatim, including
    # `{action:"move"}`, and a check that reads comments cannot tell an
    # explanation of a mistake from the mistake.
    code = re.sub(r"/\*.*?\*/", "", BUBBLE_JS, flags=re.DOTALL)
    code = re.sub(r"^\s*//.*$", "", code, flags=re.MULTILINE)
    assert 'addEventListener("mousemove"' not in code, (
        "bubble.js is moving the window again; the host's tracking loop owns "
        "that, and two movers for one gesture is how a drag ends somewhere "
        "nobody asked for"
    )
    assert "relative: true" not in code, (
        "the page is posting drag deltas again"
    )
    # `{action:"move"}` itself stays: forwardMove carries an ABSOLUTE position
    # the runtime asked for, with a sequence the host acknowledges. That is her
    # repositioning herself, not a pointer drag, and banning the message would
    # ban the wrong thing.
    assert "forwardMove" in code


def test_the_bubble_has_exactly_one_move_mechanism():
    """A native gesture plus the page's own drag moves the panel twice per motion."""
    bubble_block = LAUNCHER.split("bubblePanel = panel", 1)[0]
    # The bubble's panel construction must not install a native drag: only the
    # page knows whether the pointer went down on × or the reply control.
    assert "installWindowDrag(on: webView)" not in bubble_block


def test_the_companion_window_is_draggable_by_its_title_strip():
    assert "installWindowDrag(on: webView, topStrip:" in LAUNCHER
    assert "TopStripPanGestureRecognizer" in LAUNCHER
    # A pan that delays the primary mouse button would swallow the clicks the
    # composer and the FULL button need.
    assert "delaysPrimaryMouseButtonEvents = false" in LAUNCHER


def test_the_companion_does_not_hide_itself_when_another_app_is_clicked():
    """NSPanel defaults hidesOnDeactivate to TRUE, unlike NSWindow.

    Unset, every click into another app ordered the companion out, which reads
    as the window closing itself.
    """
    companion_block = LAUNCHER.split("let panel = KeyablePanel(", 1)[1].split(
        "companionPanel = panel", 1
    )[0]
    assert "hidesOnDeactivate = false" in companion_block
