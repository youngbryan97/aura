"""A delivered reply must reach the screen whole, whatever the animation does.

Observed live on the desktop surface 2026-07-26: the server returned
"Yes, I am okay and steady enough to stay with you. My distress is bounded and
my continuity is holding. My attention is on body_pressure." with HTTP 200 and
`delivery_state: completed`, and the chat displayed **"Yes,"** — permanently.

The chat renders Aura's replies with a word-at-a-time typewriter driven by
`requestAnimationFrame`. `requestAnimationFrame` does not run while the document
is hidden, so a reply that arrives while the user is looking at another window
animates one word and then freezes on that fragment for good. Nothing else ever
writes the remaining text, so the message is not merely delayed — it is lost.

Every backend contract passed on that turn. The defect lives entirely between a
correct response and the user's eyes, which is exactly the class of failure the
server-side suites cannot see. These contracts pin the invariant: the animation
is decoration, and the full text has a path to the DOM that does not depend on
it.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AURA_JS = PROJECT_ROOT / "interface" / "static" / "aura.js"


def _aura_js() -> str:
    return AURA_JS.read_text(encoding="utf-8")


def _append_msg_body(source: str) -> str:
    """The body of appendMsg, where a reply becomes DOM."""
    start = source.index("async function appendMsg(")
    # The next top-level `function ` declaration ends it; appendMsg is followed
    # by the streaming helpers.
    end = source.index("\nlet activeStreamDiv", start)
    return source[start:end]


def test_typewriter_is_skipped_while_the_document_is_hidden() -> None:
    """A hidden document cannot animate, so it must not try."""
    body = _append_msg_body(_aura_js())
    guard = body[body.index("const canTypewriterRender") : body.index("if (canTypewriterRender)")]
    assert "!document.hidden" in guard, (
        "the typewriter must not be chosen while the document is hidden — "
        "requestAnimationFrame will not run and the reply freezes mid-word"
    )


def test_full_text_render_exists_and_is_the_non_animated_path() -> None:
    """One helper renders the complete text, and the else branch uses it."""
    body = _append_msg_body(_aura_js())
    assert "const renderFinal" in body, "a whole-message render path must exist"
    tail = body[body.rindex("} else {") :]
    assert "renderFinal()" in tail, (
        "the non-typewriter branch must render the complete text through renderFinal"
    )


def test_animation_has_a_stall_guard_that_completes_the_message() -> None:
    """If frames stop arriving, a timer still finishes the message.

    setTimeout keeps firing when requestAnimationFrame does not, so it is the
    backstop that turns a frozen animation into a complete message.
    """
    body = _append_msg_body(_aura_js())
    assert "setTimeout(finish" in body, (
        "a stall guard must finish the message when frames stop arriving"
    )
    assert "visibilitychange" in body, (
        "going hidden mid-animation must finalize the message"
    )
    finish = body[body.index("function finish()") : body.index("function typeChunk")]
    assert "renderFinal()" in finish, "finishing must render the complete text"


def test_typewriter_cannot_leave_a_message_marked_typing_forever() -> None:
    """Every exit from the animation clears the in-progress marker."""
    body = _append_msg_body(_aura_js())
    # The only place `typing` is removed is renderFinal, which every terminal
    # path routes through — so there is no exit that leaves a half-message.
    assert body.count("classList.remove('typing')") == 1, (
        "exactly one place should clear the typing marker: the whole-text render"
    )
    loop_tail = body[body.index("if (i < words.length)") :]
    assert re.search(r"else\s*\{\s*finish\(\);", loop_tail), (
        "the animation's completion branch must go through finish()"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_numbered_answers_keep_their_ordinals_across_paragraphs() -> None:
    result = subprocess.run(
        [
            shutil.which("node"),
            str(PROJECT_ROOT / "tests/js/chat_numbered_answers.mjs"),
            str(AURA_JS),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
