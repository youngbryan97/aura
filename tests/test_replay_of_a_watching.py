"""A watching replays as one page: the video in its own site's player, what she said before and after on their own,
and what she said as it played timed to the player's clock."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_a_watching_replays_with_her_words_where_she_said_them():
    from tools.replay_of_a_watching import replay

    url = "https://www.youtube.com/watch?v=NtQkz0aRDe8"
    rows = [{"kind": "before", "title": "A moon base", "url": url, "wall": 1},
            {"kind": "said", "when": "before", "at": 0.0, "words": "I expect regolith.", "url": url},
            {"kind": "said", "when": "during", "at": 21.0, "words": "Apollo footage, of course.", "url": url},
            {"kind": "said", "when": "after", "at": 538.0, "words": "More hopeful than I expected <really>.", "url": url}]
    page = replay(rows)
    assert "embed/NtQkz0aRDe8" in page
    assert '<li data-at="21.0"><span class="at">0:21</span><p>Apollo footage, of course.</p></li>' in page
    assert "Before it played" in page and "I expect regolith." in page
    assert "After it ended" in page and "&lt;really&gt;" in page  # her words escaped, never markup
