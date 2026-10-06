"""A requested number of attempts ends without inventing a win."""
from types import SimpleNamespace

import pytest

from core.language.how_a_game_ended import requested_attempts


@pytest.mark.parametrize("asked,count", [
    ("Fix the file, then play it for three attempts to show it works.", 3),
    ("Try it two times.", 2), ("Make 4 attempts.", 4),
    ("Play until you win.", None), ("First to three points wins.", None),
    ("Play three different games.", None),
])
def test_attempt_counts_are_read_without_confusing_rules_or_multiple_games(asked, count):
    assert requested_attempts(asked) == count


@pytest.mark.asyncio
@pytest.mark.parametrize("goal,endings,expected_complete", [
    ("Play it for three attempts.", ["lost", "lost", "lost"], True),
    ("Play it for three attempts.", ["won", "lost", "lost"], True),
    ("Try to win; make three attempts.", ["lost", "lost", "lost"], False),
    ("Try to win; make three attempts.", ["won"], True),
])
async def test_requested_attempts_and_requested_wins_keep_their_own_completion_rules(monkeypatch, goal, endings, expected_complete):
    from core.skills import sovereign_browser_drawing as drawing

    seen = []

    async def one(page, band, goal, url, deadline, keep):
        seen.append(len(seen))
        return {"moves": [{"key": "observed action"}], "completed": False}, SimpleNamespace(
            keep=keep, stretches=[{"pictures": 20}], over_because="observed terminal",
            words=["first to seven points wins"])

    def ended(reflexes, result):
        return {"ended": endings[len(seen)-1], "words": "observed end", "said": ""}

    monkeypatch.setattr(drawing, "_one_run", one)
    monkeypatch.setattr(drawing, "_how_the_run_went", ended)
    result = await drawing._played(None, (0,0,1,1), goal, "file:///a-general-world.html", {})
    assert len(seen) == len(endings)
    assert result["runs"] == endings
    assert result["completed"] is expected_complete and result["ok"] is expected_complete
    assert result["won"] is ("won" in endings)
