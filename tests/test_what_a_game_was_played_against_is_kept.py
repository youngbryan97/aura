"""What a game was played against is kept with it, and a game used to judge a change is kept apart from one used to
make it.

A claim about how she plays is only as good as what it was measured against: which engine, which build of it, how
many threads and how much table it had, whether its strength was held back, and what clock the game was on. Her record
of each game keeps what its window said as it began; tools/board_game_receipts.py adds the engine's own account of
itself and its settings, and the purpose declared before the game (suggestions of 23 November 2026, item 23).
"""
from __future__ import annotations

import asyncio
import json
import stat

import core.skills.screen_pursuit_a_way_on as way_on
import core.skills.screen_pursuit_board_of_pieces as board_of_pieces
from tools import board_game_receipts as receipts

AS_IT_BEGAN = ["Aura vs. Stockfish", "WHITE", "Aura", "BLACK", "Stockfish 19", "3635", "3:00", "+2s"]


def test_the_words_a_window_shows_are_read_run_by_run(monkeypatch) -> None:
    async def read(_app):
        return {"ok": True, "layout": [{"text": f" {text} "} for text in AS_IT_BEGAN] + [{"text": " "}]}

    monkeypatch.setattr(way_on, "_its_own_pixels_read", read)
    assert asyncio.run(board_of_pieces.the_words_in("a board")) == " / ".join(AS_IT_BEGAN)


def _an_engine(tmp_path):
    engine = tmp_path / "engine"
    engine.write_text("#!/bin/sh\necho 'id name Engine 7'\n"
                      "echo 'option name Threads type spin default 1 min 1 max 8'\n"
                      "echo 'option name UCI_LimitStrength type check default false'\necho uciok\n")
    engine.chmod(engine.stat().st_mode | stat.S_IEXEC)
    return engine


def _a_log(tmp_path, *games):
    log = tmp_path / "aura_json.log"
    log.write_text("".join(json.dumps({"event": "a board of pieces played: " + repr(record), "timestamp": at}) + "\n"
                           for at, record in games))
    return log


def test_a_receipt_names_the_engine_its_settings_the_window_and_the_purpose_declared_before(tmp_path,
                                                                                          monkeypatch) -> None:
    monkeypatch.setattr(receipts, "KEPT", tmp_path / "kept")
    record = {"ended": "lost", "moves": 2, "the_moves": ["e2-e4", "e7-e5", "g1-f3", "b8-c6"], "first_was_hers": True,
              "the_window_when_it_began": " / ".join(AS_IT_BEGAN)}
    receipts.declare("evaluation", "before the game")
    log = _a_log(tmp_path, ("2999-01-01T10:00:00+00:00", {"ended": "not taking moves", "moves": 0}),
                 ("2999-01-01T10:05:00+00:00", record))
    settings = tmp_path / "engines.json"
    settings.write_text(json.dumps([{"name": "Engine", "settings": [{"name": "Threads", "value": 4}]}]))
    row = receipts.receipt(_an_engine(tmp_path), settings, [log])
    assert row["purpose"] == "evaluation" and row["ended_at"] == "2999-01-01T10:05:00+00:00"
    assert row["engine"]["name"] == "Engine 7" and row["engine"]["defaults"]["UCI_LimitStrength"] == "false"
    assert row["settings"][0]["settings"][0]["value"] == 4
    assert row["the_window_when_it_began"].startswith("Aura vs. Stockfish") and row["moves"] == record["the_moves"]
    # One receipt a game.
    assert receipts.receipt(_an_engine(tmp_path), settings, [log]) is None


def test_a_game_with_no_purpose_declared_before_it_ended_counts_as_tuning(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(receipts, "KEPT", tmp_path / "kept")
    log = _a_log(tmp_path, ("2000-01-01T10:05:00+00:00", {"ended": "lost", "moves": 3, "the_moves": []}))
    receipts.declare("evaluation", "declared after the game")
    row = receipts.receipt(_an_engine(tmp_path), None, [log])
    assert row["purpose"] == "tuning" and row["declared"] is None
