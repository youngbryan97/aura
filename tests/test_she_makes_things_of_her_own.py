"""Bryan, 27 November 2026: "aura should be able to autonomously use that workspace to build apps on her own if she wants
without prompting", and "not that she has to... if she wanted to or saw use for one in some way, she's free to. even if just
experimental and she wants to delete it later or practice something or whatever. dealer's choice"."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from core.agency import what_she_could_do
from core.agency.her_own_time import HerOwnTime
from core.agency.what_she_could_do import WhatSheWouldMake
from core.agency.what_she_does_next import WhatSheDoesNext


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    from core.runtime import her_writes
    from core.understanding import requests_of_her
    from core.understanding.requests_of_her import RequestsOfHer

    held = RequestsOfHer(tmp_path / "r.json")
    monkeypatch.setattr(requests_of_her, "get_requests_of_her", lambda: held)

    def governed(how, path, *payload, **_how):
        if how == "ensure_directory":
            Path(path).mkdir(parents=True, exist_ok=True)
        elif how == "move_path":
            Path(path).rename(payload[0])
        else:
            raise AssertionError(f"not a write this does: {how}")

    monkeypatch.setattr(her_writes, "governed", governed)
    return held


@pytest.fixture
def wants_nothing(monkeypatch):
    from core.conation import what_she_wants

    class Record:
        def wants(self, limit=None):
            return []

    monkeypatch.setattr(what_she_wants, "get_what_she_wants", lambda: Record())


def _hers(tmp_path) -> HerOwnTime:
    return HerOwnTime(WhatSheDoesNext(tmp_path / "c.json", seed=3), tmp_path / "t.json")


def _built(tmp_path, request, holds: int, of: int) -> Path:
    folder = tmp_path / "live_apps" / request.id
    folder.mkdir(parents=True)
    (folder / "index.html").write_text("<!doctype html>", encoding="utf-8")
    (folder / "program.json").write_text(json.dumps({"tried": [{"ok": i < holds} for i in range(of)]}), encoding="utf-8")
    request.made["page"] = str(folder / "index.html")
    return folder


def test_whether_she_makes_anything_is_hers_and_nothing_is_made_where_she_wants_nothing(tmp_path, ledger, wants_nothing):
    hers, made, read = _hers(tmp_path), [], []

    async def make(request):
        made.append(request)

    async def imagines(facts):
        read.append(json.loads(facts))
        return WhatSheWouldMake(wants_to=False)

    (option,) = what_she_could_do.things_to_make(hers, make, imagines)
    assert option.kind == "make" and option.id == "make:new"
    came = asyncio.run(option.does())
    assert came.got == 0.0 and not came.put_down and came.declined and "nothing she wanted to make" in came.what
    assert -0.1 < came.reward(option.cost_s) <= 0.0, "wanting nothing made is worth what resting is, less what asking took"
    assert made == [] and ledger.all() == [] and read[0]["what_you_have_made_for_yourself"] == []


def test_what_she_wants_made_is_asked_of_herself_and_made_as_anybody_s_is(tmp_path, ledger, wants_nothing):
    from core.understanding.requests_of_her import MET, Part

    hers = _hers(tmp_path)

    async def make(request):
        assert request.conversation == "herself" and request.effect == "program"
        assert "practise drawing with the frame's dialogs" in request.whole
        _built(tmp_path, request, 9, 10)
        request.parts, request.stands = [Part("it shows the tides", "met"), Part("it rings at high water", "met")], MET

    async def imagines(_facts):
        return WhatSheWouldMake(wants_to=True, what="A tide table with a bell for high water.", why="To practise drawing with the frame's dialogs.")

    (option,) = what_she_could_do.things_to_make(hers, make, imagines)
    came = asyncio.run(option.does())
    assert came.got == 0.9 and "2 of 2 things asked of it are so" in came.what and "90%" in came.what
    (request,) = ledger.all()
    assert request.said == "A tide table with a bell for high water." and what_she_could_do.her_own_makes() == [request]
    facts = json.loads(what_she_could_do.what_she_has_to_go_on(hers))
    assert facts["what_you_have_made_for_yourself"][0]["id"] == request.id and facts["what_you_have_made_for_yourself"][0]["of_what_it_is_tried_by_holds"] == 0.9


def test_what_she_set_out_to_make_and_did_not_finish_is_hers_to_go_on_with_and_somebody_s_is_not_among_it(tmp_path, ledger, wants_nothing):
    from core.understanding.requests_of_her import PARTLY, Part

    hers = _hers(tmp_path)
    own = ledger.asked("A pond of ripples to practise motion.", effect="program", conversation="herself")
    own.parts, own.stands, own.changed_at = [Part("it ripples", "met"), Part("it stills", "unmet")], PARTLY, time.time()
    _built(tmp_path, own, 1, 2)
    theirs = ledger.asked("A dinner planner.", effect="program", conversation="c")
    theirs.stands, theirs.changed_at = PARTLY, time.time()

    async def make(request):
        request.parts[1].stands = "met"
        (Path(request.made["page"]).parent / "program.json").write_text(json.dumps({"tried": [{"ok": True}, {"ok": True}]}), encoding="utf-8")

    async def imagines(_facts):
        raise AssertionError("going on with what she began asks nothing")

    go_on, new = what_she_could_do.things_to_make(hers, make, imagines)
    assert go_on.id == f"make:{own.id}" and "pond of ripples" in go_on.what and new.id == "make:new" and new.known["held"] == 0.5
    assert asyncio.run(go_on.does()).got == 1.0
    assert [o.id for o in what_she_could_do.left_unmet(make)] == [f"carry_on:{theirs.id}"], "what she makes for herself is not somebody's"


def test_what_she_no_longer_wants_kept_is_put_away_and_not_destroyed_and_only_hers_is(tmp_path, ledger, wants_nothing):
    from core.understanding.requests_of_her import MET

    hers = _hers(tmp_path)
    own = ledger.asked("A sketch of falling sand, to try a cellular rule.", effect="program", conversation="herself")
    own.stands = MET
    folder = _built(tmp_path, own, 3, 3)
    theirs = ledger.asked("A flat-sharing app.", effect="program", conversation="c")
    theirs.stands = MET
    their_folder = _built(tmp_path, theirs, 3, 3)

    async def make(_request):
        raise AssertionError("nothing was wanted made")

    async def imagines(_facts):
        return WhatSheWouldMake(wants_to=False, let_go=[own.id, theirs.id])

    (option,) = what_she_could_do.things_to_make(hers, make, imagines)
    came = asyncio.run(option.does())
    assert "she put away 1 thing(s) she had made: A sketch of falling sand" in came.what
    assert not folder.exists() and (tmp_path / "live_apps" / "put_away" / own.id / "index.html").is_file()
    assert their_folder.is_dir() and what_she_could_do.her_own_makes() == [], "what she put away is not among what she has made"


def test_while_a_thing_of_hers_is_being_made_she_does_not_begin_another(tmp_path, ledger, wants_nothing, monkeypatch):
    from core.understanding import carrying_out

    own = ledger.asked("A clock face.", effect="program", conversation="herself")
    monkeypatch.setattr(carrying_out, "is_under_way", lambda request: request.id == own.id)

    async def nothing(*_a):
        return None

    assert what_she_could_do.things_to_make(_hers(tmp_path), nothing, nothing) == []


def test_she_makes_nothing_of_her_own_where_nothing_would_make_it(tmp_path):
    async def nothing(*_a):
        return None

    hers = what_she_could_do.begin_her_own_time(_hers(tmp_path), carry_on=nothing, looks=nothing, tell=nothing, here=lambda: 0.0)
    assert "makes" not in hers._offers
    hers = what_she_could_do.begin_her_own_time(_hers(tmp_path), carry_on=nothing, looks=nothing, tell=nothing, here=lambda: 0.0, make=nothing, imagines=nothing)
    assert "makes" in hers._offers and what_she_could_do.way_of("make") != ""
