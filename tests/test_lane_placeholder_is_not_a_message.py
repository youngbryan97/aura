"""Lane status is a pinned overlay, never an entry in the transcript.

Reported live on 2026-08-03: "Conversation lane initializing. Waiting for
verified Aura reply path..." sometimes appeared in the MIDDLE of the chat
rather than at the top. It could, because it shipped as a .sys-box child of
#messages — a transcript entry. Anything that reorders or re-hydrates the
transcript could leave it sitting between two real turns, reading as though
Aura had said it mid-conversation.

It lives outside #messages now, so there is no slot for it to occupy.

The second defect these pin is why the transcript could be rewritten under a
live turn: hydration asked "is this pane empty?" by reading textContent, but
appendMsg appends an EMPTY div and types the text in afterwards. For the
length of that animation a populated pane read as empty, and a poll landing
in that window wiped it and hydrated over the top.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = PROJECT_ROOT / "interface" / "static" / "index.html"
AURA_JS = PROJECT_ROOT / "interface" / "static" / "aura.js"
AURA_CSS = PROJECT_ROOT / "interface" / "static" / "aura.css"

PLACEHOLDER_TEXT = "Conversation lane initializing. Waiting for verified Aura reply path..."


@pytest.fixture(scope="module")
def html() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def js() -> str:
    return AURA_JS.read_text(encoding="utf-8")


class TestThePlaceholderIsNotInTheTranscript:
    def test_messages_ships_empty(self, html):
        """#messages must contain no markup that could read as a turn."""

        match = re.search(r'<div id="messages"[^>]*>(.*?)</div>', html, re.DOTALL)
        assert match, "#messages element not found"
        assert not match.group(1).strip(), (
            "#messages ships with content; anything in here is a transcript entry "
            "and can be reordered into the middle of a conversation"
        )

    def test_the_placeholder_lives_in_its_own_element(self, html):
        assert 'id="lane-placeholder"' in html
        placeholder_at = html.index('id="lane-placeholder"')
        messages_at = html.index('<div id="messages"')
        assert placeholder_at < messages_at, (
            "the lane status must be pinned above the transcript, not after it"
        )
        assert PLACEHOLDER_TEXT in html

    def test_it_is_announced_as_status_not_speech(self, html):
        segment = html[html.index('id="lane-placeholder"'):][:400]
        assert 'role="status"' in segment, (
            "a screen reader must hear this as system status, not as Aura talking"
        )


class TestVisibilityHasOneOwner:
    def test_a_single_function_decides(self, js):
        assert "function updateLanePlaceholder()" in js
        # Every append into the transcript funnels through pruneVisibleMessages.
        prune = js[js.index("function pruneVisibleMessages"):][:600]
        assert "updateLanePlaceholder" in prune, (
            "a new message must hide the lane status immediately"
        )

    def test_it_hides_once_there_is_anything_to_read(self, js):
        body = js[js.index("function updateLanePlaceholder()"):][:600]
        assert "transcriptIsEmpty" in body
        assert "state.conversationReady" in body

    def test_it_starts_visible(self, html):
        """If the runtime never answers, this is the only explanation on screen."""

        segment = html[html.index('id="lane-placeholder"'):][:400]
        opening_tag = segment[: segment.index(">") + 1]
        assert "hidden" not in opening_tag, (
            "starting hidden leaves a blank panel when the backend is down"
        )


class TestEmptinessIsCountedNotRead:
    def test_transcript_emptiness_does_not_read_text(self, js):
        body = js[js.index("function transcriptIsEmpty"):][:400]
        assert "children.length" in body
        assert "textContent" not in body, (
            "appendMsg appends an empty div and types into it; a textContent "
            "test makes a populated pane look empty mid-animation"
        )

    @pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
    def test_hydration_reconciles_known_exchanges_without_replaying_the_pane(self):
        result = subprocess.run(
            [shutil.which("node"), str(PROJECT_ROOT / "tests/js/restored_pending_exchange.mjs"), str(AURA_JS)],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    @pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
    def test_workload_changes_cannot_starve_transcript_polling(self):
        result = subprocess.run(
            [shutil.which("node"), str(PROJECT_ROOT / "tests/js/bootstrap_poll_fairness.mjs"), str(AURA_JS)],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, result.stdout + result.stderr


class TestTheOverlayIsStyled:
    def test_the_placeholder_has_its_own_rule(self):
        css = AURA_CSS.read_text(encoding="utf-8")
        assert ".lane-placeholder {" in css
        assert ".lane-placeholder[hidden]" in css, (
            "without an explicit rule a flex/grid display can defeat [hidden]"
        )
