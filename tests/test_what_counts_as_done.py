"""What counts as done comes from the person's words, for whatever shape the thing turns out to have.

Bryan, 8 October 2026: three games picked by the rule. One that can be won is
played until it is won; one that only keeps a score is played three times and
its best said; one with neither, like a character maker, is done once and what
was made, and why, said. Said in general words, it is about any task with an
end to reach, a measure kept, or a thing to make.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from core.language.how_a_game_ended import offers_a_win, requested_attempts, the_score_in
from core.language.what_counts_as_done import (
    END,
    MADE,
    MEASURE,
    conditions_as_said,
    shape_in,
    terms_asked,
    without_conditions,
)
from core.skills import sovereign_browser_drawing as drawing

pytestmark = pytest.mark.unit

CONDITIONS = ("If a game can be won, keep playing it until you win; if it only keeps a score, play it three times and "
              "tell me your best; if it has neither, like a character maker, do it once and tell me what you made and why.")
ASKED = ("Go to https://www.webdesignmuseum.org/flash-game-exhibitions/cartoon-network-flash-games and play three of "
         "the games, one after another. To pick them: number the games on the list from 0. " + CONDITIONS)


def test_the_conditions_are_read_into_terms_for_each_shape():
    terms = terms_asked(ASKED)
    assert terms.until_the_end
    assert terms.measure_runs == 3 and terms.tell_the_best
    assert terms.made_once and terms.tell_what_was_made
    assert len(terms.said) == 3


def test_the_same_reading_holds_outside_games():
    terms = terms_asked("For each support ticket: if it can be solved, keep at it until it is done; "
                        "if it only has a response time, try two drafts and keep the best; "
                        "if it has neither, write it up once and tell me what you made and why.")
    assert terms.until_the_end and terms.measure_runs == 2 and terms.made_once and terms.tell_what_was_made


def test_a_count_said_of_one_shape_is_not_every_things():
    assert requested_attempts(ASKED) is None
    assert "three times" not in without_conditions(ASKED)
    assert requested_attempts("Fix it, then play it for three attempts.") == 3
    assert terms_asked("Play the game and win it.") == terms_asked("")


def test_the_conditions_travel_in_the_persons_words():
    assert conditions_as_said(ASKED) == (
        "If a game can be won, keep playing it until you win. If it only keeps a score, play it three times and tell "
        "me your best. If it has neither, like a character maker, do it once and tell me what you made and why."
    )


def test_a_things_shape_is_read_from_its_own_words():
    assert shape_in("Collect as many points as possible before the time runs out.") == MEASURE
    assert shape_in("Level 1. Rescue the princess.") == END
    assert shape_in("Create your own KND by selecting your favorite body parts.") == MADE
    assert shape_in("Use the arrow keys to move.") == ""
    assert shape_in("Use the arrow keys to move.", a_measure_seen=True) == MEASURE
    # An objective is an end to reach; levels that each run ends with a score are an arcade game's, and it is played for the score.
    assert shape_in("Help Dexter find his missing robot! Look for special pairs of glasses.") == END
    assert shape_in("LEVEL 1 GAME OVER YOUR SCORE: 300", a_measure_seen=True) == MEASURE
    assert shape_in("LEVEL 2") == END
    assert shape_in("YOU WIN!", a_measure_seen=True) == END
    assert the_score_in("GAME OVER YOUR SCORE: 1,250 PLAY AGAIN") == 1250
    assert not offers_a_win("Catch the falling fruit. GAME OVER YOUR SCORE 12")


class _Run:
    """One run as `_one_run` hands it back."""

    def __init__(self, words: list[str], ending: str, *, gains: int = 0, over: str = "a way to start again appeared") -> None:
        self.keep: dict[str, Any] = {}
        self.words = words
        self.ending_words = ending
        self.ending_parts: list[str] = [ending]
        self.over_because = over
        self.new_screens = 0
        self.stretches = [{"pictures": 40, "gains": gains, "counters": {}}]

    def what_it_came_to(self) -> str:
        return "played it as it happened"


def _played_through(monkeypatch: pytest.MonkeyPatch, runs: list[_Run], goal: str,
                    moves: list[dict[str, Any]] | None = None) -> tuple[dict[str, Any], list[str]]:
    said: list[str] = []
    queue = list(runs)

    async def _one_run(*_a: Any, **_k: Any) -> tuple[dict[str, Any], _Run]:
        return {"moves": list(moves or [{"key": "left"}])}, queue.pop(0)

    async def _from_the_top(page: Any, band: Any) -> Any:
        said.append("(started again)")
        return band

    monkeypatch.setattr(drawing, "_one_run", _one_run)
    monkeypatch.setattr(drawing, "_from_the_top", _from_the_top)
    monkeypatch.setattr(drawing, "_tell", said.append)
    monkeypatch.setattr("core.skills.screen_pursuit_as_it_happens.begin_run", lambda keep: None)
    played = asyncio.run(drawing._played(SimpleNamespace(), (0, 0, 1, 1), goal, "https://a.example/thing", {}))
    return played, said


def test_a_measure_is_tried_the_asked_number_of_times_and_its_best_said(monkeypatch):
    words = ["Collect as many points as possible before the time runs out."]
    runs = [_Run(words, f"GAME OVER YOUR SCORE: {score} PLAY AGAIN") for score in (120, 340, 200)]
    played, said = _played_through(monkeypatch, runs, "Play this game. " + CONDITIONS)
    assert played["best_score"] == 340 and len(played["runs"]) == 3 and played["completed"]
    assert any("Run 1 of 3: scored 120" in line for line in said)
    assert said[-1].startswith("That's 3 runs. My best was run 2, with 340")


def test_a_thing_for_making_is_made_once_and_what_was_made_said(monkeypatch):
    from core.skills.screen_pursuit_as_it_happens import MADE_NOT_WON

    words = ["NUMBUH GENERATOR Create your own KND by selecting your favorite body parts"]
    runs = [_Run(words, "NUMBUH 86 AGENT FANNY", over=MADE_NOT_WON)]
    moves = [{"key": 'click "HEAD"'}, {"key": 'click "HEAD"'}, {"key": 'click "ARMOR"'}, {"key": 'click "GENERATE"'}]
    played, said = _played_through(monkeypatch, runs, "Play this game. " + CONDITIONS, moves)
    assert played["completed"] and played["stopped_because"] == drawing.MADE_AS_ASKED
    assert "it shows 'NUMBUH 86 AGENT FANNY'" in said[-1] and "I chose HEAD, ARMOR, GENERATE" in said[-1]


def test_a_thing_with_an_end_is_kept_at_past_not_getting_better(monkeypatch):
    words = ["Level 1. Use the arrow keys to move."]
    runs = [_Run(words, "GAME OVER") for _ in range(13)] + [_Run(words, "YOU WIN! Level 1 complete")]
    played, said = _played_through(monkeypatch, runs, "Play this game. " + CONDITIONS)
    assert played["won"] and played["runs"][-1] == "won" and len(played["runs"]) == 14
    assert not any("no better than my best" in line for line in said)


def test_without_the_conditions_nothing_changes(monkeypatch):
    words = ["Collect as many points as possible before the time runs out."]
    played, said = _played_through(monkeypatch, [_Run(words, "GAME OVER YOUR SCORE: 120")], "Play this game and win it.")
    assert played["runs"] == ["finished"] and "no winner, only a score" in said[-1]


def test_each_thing_a_rule_picks_carries_the_conditions():
    from core.skills.sovereign_browser_picking import the_task_for_each

    assert the_task_for_each(ASKED) == "Play this game. " + conditions_as_said(ASKED)


def test_what_a_request_asks_to_be_done_is_read_without_its_conditions():
    from core.intent.declared_capability import rank_declaration_matches

    catalogue = {"repair": (frozenset({"repair", "fix"}), frozenset({"program", "game", "win"})),
                 "browse": (frozenset({"go", "open"}), frozenset({"page", "list"}))}
    selective = {"repair": frozenset({"program", "game", "win"}), "browse": frozenset({"page", "list"})}
    ranked = dict(rank_declaration_matches("Open the page and its list. " + CONDITIONS, catalogue, selective))
    assert "repair" not in ranked or ranked["repair"] < ranked.get("browse", 0.0)
    assert terms_asked("If you make a mistake, undo it.") == terms_asked("")
    assert terms_asked("If there's time, tidy the tests.") == terms_asked("")
