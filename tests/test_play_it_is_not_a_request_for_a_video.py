"""A video nobody asked for appeared in the chat, and she denied sending it.

LIVE 2026-08-19. The message was "2048 is open in Chrome. Play it — keep
going until you get a 128 tile. Tell me what you are doing as you go." A
video from Downloads started playing in the chat window.

Three things had to go wrong together, and each is general:

  * the query ran to the end of the message, so a title spanned two
    sentences and seventy-eight characters;
  * "it" was accepted as the name of something;
  * one query word appearing INSIDE a filename counted as a match — "it" is
    a substring of "Cognitive".

She denied it truthfully. The card is published by the route before the turn
is routed, so nothing about the video reached the lane that answered.
"""
from __future__ import annotations

import pytest

from core.media.library import _hits, _searchable_terms
from core.media.playback import parse_play_request


class _Item:
    def __init__(self, title, parent="Downloads", kind="video"):
        self.title = title
        self.kind = kind
        from pathlib import Path

        self.path = Path(f"/tmp/{parent}/{title}.mp4")


def test_the_live_message_is_not_a_request_to_play_media():
    assert parse_play_request(
        "2048 is open in Chrome. Play it — keep going until you get a 128 tile. "
        "Tell me what you are doing as you go."
    ) == ("", "")


@pytest.mark.parametrize("pronoun", ["play it", "play that", "play this", "play them", "play the one"])
def test_a_pronoun_is_not_a_title(pronoun):
    assert parse_play_request(pronoun) == ("", "")


def test_a_title_does_not_run_past_the_end_of_a_sentence():
    assert parse_play_request("Play Kind of Blue. Then tell me about it.")[0] == "Kind of Blue"
    assert parse_play_request("play Blue in Green — and dim the lights")[0] == "Blue in Green"


def test_real_requests_still_work():
    assert parse_play_request("play Kind of Blue")[0] == "Kind of Blue"
    assert parse_play_request("put on the Radiohead album") == ("Radiohead", "audio")
    assert parse_play_request("watch Aura Demo - 01") == ("Aura Demo - 01", "video")
    assert parse_play_request("can you play some jazz")[0] == "jazz"


def test_a_short_or_ordinary_word_cannot_identify_a_file():
    assert _searchable_terms("it keep going until you get a 128 tile") == ["128", "tile"]
    assert _searchable_terms("kind of blue") == ["kind", "blue"]
    assert _searchable_terms("it that the") == []


def test_a_title_made_only_of_ordinary_words_is_still_findable(monkeypatch):
    """"So What" and "Let It Be" are real names, not noise."""
    from core.media import library as lib

    class _Scan:
        items = [_Item("So What", kind="audio"), _Item("Aura Demo - 01 (Cognitive Engine)")]
        truncated = False

        def narrative(self):
            return "2 items"

    shelf = lib.MediaLibrary.__new__(lib.MediaLibrary)
    monkeypatch.setattr(shelf, "index", lambda: _Scan(), raising=False)
    assert [item.title for item in shelf.search("So What")][0] == "So What"


def test_a_query_word_must_be_a_word_and_not_a_substring():
    words = {"aura", "demo", "01", "cognitive", "engine"}
    assert not _hits("it", words), "'it' is inside 'cognitive', which is not a match"
    assert _hits("cognitive", words)
    assert _hits("cogn", words), "a prefix still finds 'remastered' from 'remaster'"


def test_most_of_the_query_has_to_be_present(monkeypatch, tmp_path):
    """One word in common is not evidence that this is the file."""
    from core.media import library as lib

    class _Scan:
        items = [_Item("Aura Demo - 01 (Cognitive Engine)")]
        truncated = False

        def narrative(self):
            return "1 item"

    shelf = lib.MediaLibrary.__new__(lib.MediaLibrary)
    monkeypatch.setattr(shelf, "index", lambda: _Scan(), raising=False)
    assert shelf.search("128 tile keep going you get") == []
    assert [item.title for item in shelf.search("aura demo cognitive")] == [
        "Aura Demo - 01 (Cognitive Engine)"
    ]
