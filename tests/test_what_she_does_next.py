"""What she does on her own time: chosen among acts that each have something that does them, and learned from what came of it."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from core.agency import her_own_time, what_she_could_do
from core.agency.her_own_time import HerOwnTime
from core.agency.what_she_does_next import REST, CameOf, Option, WhatSheDoesNext


@pytest.fixture(autouse=True)
def written(monkeypatch):
    from core.rebuilding import her_build_writes

    def governed(what, path, *args, **_how):
        Path(path).mkdir(parents=True, exist_ok=True) if what == "ensure_directory" else Path(path).write_text(args[0], encoding="utf-8")

    monkeypatch.setattr(her_build_writes, "governed", governed)


async def _nothing() -> CameOf:
    return CameOf()


def _option(kind: str, which: int = 0, **over) -> Option:
    return Option(f"{kind}:{which}", kind, f"{kind} {which}", does=over.pop("does", _nothing), **over)


def _chooser(tmp_path, seed: int = 3) -> WhatSheDoesNext:
    return WhatSheDoesNext(tmp_path / "c.json", seed=seed)


def _lives(chooser: WhatSheDoesNext, worth, days: int, offered) -> list[str]:
    """So many choices, each coming to what ``worth`` says of it, give or take."""
    rng, done = np.random.default_rng(5), []
    for day in range(days):
        picked = chooser.choose(offered(day))
        if picked is None:
            done.append(REST)
            continue
        option, chosen = picked
        got = min(1.0, max(0.0, worth(option, day) + rng.normal(0, 0.15)))
        chooser.learn(chosen, CameOf(got=got, seconds=option.cost_s), cost_s=option.cost_s)
        done.append(option.kind)
    return done


def test_she_comes_to_do_what_is_worth_doing_and_still_tries_the_rest(tmp_path):
    chooser = _chooser(tmp_path)
    worth = {"tell": 0.35, "find_out": 0.9, "goal": 0.55}
    done = _lives(chooser, lambda o, _d: worth[o.kind], 300, lambda _d: [_option(kind) for kind in worth])
    assert done[-100:].count("find_out") > 80 and {"tell", "find_out", "goal"} <= set(done[:30])
    learned = chooser.how_she_chooses()
    assert learned["lately"] > learned["before"] and learned["kinds"]["find_out"]["expected"] > learned["kinds"]["tell"]["expected"]


def test_what_stops_being_worth_doing_is_let_go_of(tmp_path):
    """Wants change, and what is worth doing with them."""
    chooser = _chooser(tmp_path)
    worth = lambda o, d: {"find_out": 0.9 if d < 250 else 0.2, "goal": 0.6}[o.kind]  # noqa: E731
    done = _lives(chooser, worth, 700, lambda _d: [_option("find_out"), _option("goal")])
    assert done[150:250].count("find_out") > 85 and done[-100:].count("goal") > 70


def test_what_is_known_of_an_act_before_it_is_done_tells_which_of_a_sort_to_do(tmp_path):
    chooser = _chooser(tmp_path)
    rng = np.random.default_rng(2)

    def offered(_day):
        return [_option("find_out", n, known={"new": float(rng.random()), "wanted": float(rng.random())}) for n in range(4)]

    _lives(chooser, lambda o, _d: 0.1 + 0.8 * o.known["new"], 250, offered)
    new, old = _option("find_out", 1, known={"new": 0.95, "wanted": 0.2}), _option("find_out", 2, known={"new": 0.05, "wanted": 0.9})
    assert chooser.expected(new)[0] > chooser.expected(old)[0] + 0.5


def test_where_nothing_is_worth_doing_she_rests(tmp_path):
    chooser = _chooser(tmp_path)
    done = _lives(chooser, lambda _o, _d: 0.02, 120, lambda _d: [_option("tell"), _option("goal")])
    assert done[-40:].count(REST) > 25, "an act expected to come to less than nothing is not done"
    assert chooser.choose([]) is None and chooser.choose([Option("x", "tell", "x")]) is None, "nothing that has nothing to do it is an act"


def test_a_drive_is_no_option_and_weighs_what_would_feed_it(tmp_path):
    """LIVE 24 November 2026: 205 of 220 choices in seven hours were "Acting on curiosity: explore_knowledge"."""
    chooser = _chooser(tmp_path)
    rng = np.random.default_rng(4)

    def offered(_day):
        asking = float(rng.random())
        return [_option("find_out", known={}, drives={"curiosity": asking}), _option("goal")]

    _lives(chooser, lambda o, _d: 0.2 + 0.7 * o.drives.get("curiosity", 0.0) if o.kind == "find_out" else 0.5, 400, offered)
    hungry, fed = _option("find_out", drives={"curiosity": 0.95}), _option("find_out", drives={"curiosity": 0.05})
    assert chooser.expected(hungry)[0] > 0.4 + chooser.expected(fed)[0]
    import inspect

    from core.phases.motivation_update import MotivationUpdatePhase

    body = inspect.getsource(MotivationUpdatePhase._integrate_drives)
    assert "{decision.action}" not in body and "_what_would_feed" in body


def test_what_is_owed_comes_before_what_is_chosen(tmp_path):
    chooser = _chooser(tmp_path)
    for _ in range(20):
        option, _chosen = chooser.choose([_option("find_out"), _option("carry_on", owed=True)])
        assert option.kind == "carry_on"


def test_what_stopped_an_act_that_was_not_its_own_says_nothing_of_it(tmp_path):
    chooser = _chooser(tmp_path)
    _option_, chosen = chooser.choose([_option("goal")])
    before = chooser.expected(_option("goal"))
    assert chooser.learn(chosen, CameOf(put_down=True)) is None and chooser.expected(_option("goal")) == before and chosen.reward is None
    assert chooser.learn(chosen, CameOf(got=1.0)) > 0 and chooser.expected(_option("goal"))[0] > before[0] and chooser.expected(_option("goal"))[1] < before[1]
    assert CameOf(got=1.0, worth=-1.0).reward(60) < CameOf(got=1.0).reward(60) and CameOf(got=1.0, seconds=9000).reward(60) < CameOf(got=1.0, seconds=30).reward(60)


def test_she_chooses_as_she_had_learned_to_after_she_has_been_closed(tmp_path):
    chooser = _chooser(tmp_path)
    _lives(chooser, lambda o, _d: {"tell": 0.2, "goal": 0.9}[o.kind], 80, lambda _d: [_option("tell"), _option("goal")])
    chooser.keep()
    again = WhatSheDoesNext(tmp_path / "c.json", seed=9)
    again._loaded = False
    for kind in ("tell", "goal"):
        assert again.expected(_option(kind)) == pytest.approx(chooser.expected(_option(kind)), abs=1e-6)
    assert len(again.choices(40)) == 40 and again.choices(1)[0].says().startswith(("goal", "tell"))
    (tmp_path / "c.json").write_text('{"names": ["a"], "precision": [[1, 2]], "pulled": []}')
    begun = WhatSheDoesNext(tmp_path / "c.json")
    begun._loaded = False
    assert begun.expected(_option("goal"))[0] == 0.0, "what cannot be read is begun again"


def test_every_choice_is_kept_with_its_grounds(tmp_path):
    chooser = _chooser(tmp_path)
    _lives(chooser, lambda o, _d: {"tell": 0.2, "goal": 0.9, "find_out": 0.4}[o.kind], 60, lambda _d: [_option("tell"), _option("goal"), _option("find_out")])
    chosen = chooser.choices(1)[0]
    assert {row["kind"] for row in chosen.over} == {"tell", "goal", "find_out"} - {chosen.kind} and chosen.counted and chosen.reward is not None
    assert "expected" in chosen.says() and " over " in chosen.says() and math.isfinite(chosen.drawn)


# ── her own time ─────────────────────────────────────────────────────

def _hers(tmp_path) -> HerOwnTime:
    return HerOwnTime(_chooser(tmp_path), tmp_path / "t.json")


def test_what_she_chooses_is_done_and_what_came_of_it_is_learned_and_kept(tmp_path):
    hers = _hers(tmp_path)

    async def does():
        assert hers.doing["what"] == "goal 0" and "now goal 0" in hers.where_she_is()
        return CameOf(got=0.75, what="three steps further")

    hers.offers("goals", lambda: [_option("goal", does=does)])
    hers.offers("broken", lambda: (_ for _ in ()).throw(RuntimeError("away")))
    done = asyncio.run(hers.once())
    assert done["what"] == "goal 0" and done["came"] == "three steps further" and done["reward"] > 0 and hers.doing == {}
    again = HerOwnTime(_chooser(tmp_path), tmp_path / "t.json")
    again._load()
    assert "last goal 0" in again.where_she_is() and "three steps further" in again.where_she_is()


def test_what_she_was_in_the_middle_of_when_she_was_closed_is_known_when_she_is_opened(tmp_path):
    hers = _hers(tmp_path)

    async def closed():
        await hers.keep()
        raise KeyboardInterrupt

    hers.offers("goals", lambda: [_option("goal", does=closed)])
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(hers.once())
    again = HerOwnTime(_chooser(tmp_path), tmp_path / "t.json")
    again._load()
    assert "left off when she was closed" in again.where_she_is()


def test_what_breaks_off_came_to_nothing_and_what_her_will_refuses_is_not_done(tmp_path):
    hers = _hers(tmp_path)

    async def breaks():
        raise ValueError("no")

    hers.offers("goals", lambda: [_option("goal", does=breaks)])
    done = asyncio.run(hers.once())
    assert done["got"] == 0.0 and "broke off" in done["came"] and done["reward"] < 0
    ran: list[bool] = []

    async def would():
        ran.append(True)
        return CameOf(got=1.0)

    refused = HerOwnTime(_chooser(tmp_path / "b"), tmp_path / "u.json")
    refused.offers("goals", lambda: [_option("goal", does=would)])

    async def never(_option):
        return False

    refused.may = never
    assert asyncio.run(refused.once()) is None and not ran and refused.chooser.choices(1)[0].reward is None


def test_a_drive_asks_for_the_act_that_would_feed_it(tmp_path):
    hers = _hers(tmp_path)
    hers.offers("wants", lambda: [_option("find_out", drives={"curiosity": 0.8}), _option("goal", drives={"competence": 0.3})])
    assert hers.that_feeds("curiosity") == "", "where nothing has been looked for there is nothing it asks for"
    asyncio.run(hers.options())
    assert hers.that_feeds("curiosity") == "find_out 0" and hers.that_feeds("social") == ""


def test_whether_somebody_answers_is_what_telling_them_came_to(tmp_path, monkeypatch):
    hers = _hers(tmp_path)
    hers.finds("limpets", "Limpets find their own scar again", "https://tide.example/limpets")
    told: list[str] = []

    async def tell(find):
        told.append(find["about"])
        return True

    hers.offers("finds", lambda: what_she_could_do.things_to_tell(hers, tell, lambda: 1.0))
    asyncio.run(hers.once())
    assert told == ["limpets"] and hers.found == [] and hers.told is not None
    before = hers.chooser.choices(1)[0].reward
    hers.somebody_spoke()
    assert hers.told is None and hers.chooser.choices(1)[0].reward > before
    hers.finds("tides", "Neap tides come after the quarter moons")
    asyncio.run(hers.once())
    hers.told["at"] -= 2 * her_own_time.ANSWERED_WITHIN_S
    hers.somebody_spoke()
    assert hers.chooser.choices(1)[0].reward < before


def test_a_thing_she_wants_is_looked_into_and_only_what_she_had_not_read_counts(tmp_path, monkeypatch):
    from core.conation import what_she_wants

    went, failed = [], []

    class Want:
        thing, label, way, utility, contacts, unfinished, why = "subject:limpets", "limpets", "find out", 0.4, 3, "", "its turns were worth more"

    class Record:
        def wants(self, limit=None):
            return [Want]

        def started(self, *a, **k):
            return None

        stopped = started

        def went(self, thing, way, label=""):
            went.append(thing)

        def failed(self, thing):
            failed.append(thing)

    monkeypatch.setattr(what_she_wants, "get_what_she_wants", lambda: Record())
    hers = _hers(tmp_path)
    given = [{"ok": True, "summary": "They home.", "citations": [{"title": "Limpets", "url": "https://a.example"}, {"title": "More", "url": "https://b.example"}]}] * 2 + [{"ok": False, "error": "offline"}]
    answers = iter(given)

    async def search(words):
        assert words == "limpets"
        return next(answers)

    (option,) = what_she_could_do.things_to_find_out(hers, search)
    first, second, third = (asyncio.run(option.does()) for _ in range(3))
    assert (first.got, second.got, third.got) == (1.0, 0.0, 0.0) and went == ["subject:limpets"] * 3 and failed == ["subject:limpets"]
    assert len(hers.found) == 1 and hers.found[0]["where"] == "https://a.example" and option.known["new"] == 0.25


def test_what_somebody_asked_and_did_not_get_is_gone_on_with(tmp_path, monkeypatch):
    from core.understanding import requests_of_her
    from core.understanding.requests_of_her import MET, PARTLY, Part, RequestsOfHer

    ledger = RequestsOfHer(tmp_path / "r.json")
    monkeypatch.setattr(requests_of_her, "get_requests_of_her", lambda: ledger)
    request, done, hers = (ledger.asked(said, effect="program", conversation=who) for said, who in (("Make a tide clock.", "c"), ("Make a map.", "c"), ("Make a bell.", "herself")))
    request.parts, request.stands, request.changed_at = [Part("it runs", "met"), Part("it rings", "unmet")], PARTLY, time.time()
    done.stands, hers.stands = MET, PARTLY
    old = ledger.asked("Make a sundial.", effect="program", conversation="c")
    old.stands, old.changed_at = PARTLY, time.time() - 5 * 86400

    async def carry_on(which):
        which.parts[1].stands = "met"

    (option,) = what_she_could_do.left_unmet(carry_on)
    assert "tide clock" in option.what and asyncio.run(option.does()).got == 1.0


def test_a_conversation_is_the_same_one_after_she_has_been_closed(tmp_path):
    """LIVE 24 November 2026: what was asked before a restart belonged to a conversation nobody was in any more."""
    from core.understanding.requests_of_her import RequestsOfHer, line_of

    ledger = RequestsOfHer(tmp_path / "r.json")
    request = ledger.asked("Make a tide clock.", effect="program", conversation="127.0.0.1:0a1b2c3d:99aabbcc")
    ledger.gave("127.0.0.1:0a1b2c3d:99aabbcc", "here", ["f1"])
    assert ledger.of("127.0.0.1:ffffffff:00000000") == [request] and ledger.given("127.0.0.1:ffffffff:12345678") == ["f1"]
    assert ledger.of("wb-test-1") == [] and line_of("wb-test-1") == "wb-test-1" and line_of("device:abc") == "device:abc"


def test_what_she_was_doing_for_somebody_is_gone_on_with_without_being_told_to(tmp_path, monkeypatch):
    from core.understanding import carrying_out
    from core.understanding.requests_of_her import (
        CUT_SHORT,
        MET,
        SET_ASIDE,
        WORKING,
        Part,
        RequestsOfHer,
    )

    async def nobody():
        return None

    monkeypatch.setattr(carrying_out, "nobody_waiting", nobody)
    monkeypatch.setattr(carrying_out, "_keeps_the_bench", nobody)
    monkeypatch.setattr(carrying_out, "_bench_kept", nobody)
    ledger = RequestsOfHer(tmp_path / "r.json")
    first, second, third = (ledger.asked(f"Make {what}.", effect="program", conversation="c") for what in ("a tide clock", "a map", "a bell"))
    first.stands = second.stands = third.stands = WORKING
    (tmp_path / "r.json").write_text(json.dumps({"requests": [dataclasses.asdict(r) for r in (first, second, third)], "gave": {}}))
    again = RequestsOfHer(tmp_path / "r.json")
    assert [r.said for r in again.left_off()] == ["Make a tide clock.", "Make a map.", "Make a bell."] and all(r.stands == CUT_SHORT for r in again.all())
    again.all()[2].stands = SET_ASIDE
    worked: list[str] = []

    async def carrier(request, tell):
        worked.append(request.said)
        return carrying_out.CameTo(how="built", parts=[Part("it runs", "met")])

    monkeypatch.setattr(carrying_out, "carriers", lambda: {"program": carrier})
    monkeypatch.setattr(carrying_out, "get_requests_of_her", lambda: again)

    async def told(request):
        return None

    async def run():
        taken = await carrying_out.take_up_what_was_left(tell=lambda line: None, told=told, ledger=again, waits_s=0.01)
        while any(not t.done() for t in carrying_out._UNDER_WAY.values()):
            await asyncio.sleep(0.01)
        return taken

    again.keep = nobody
    taken = asyncio.run(run())
    assert [r.said for r in taken] == worked == ["Make a tide clock.", "Make a map."] and all(r.stands == MET for r in taken)


def test_the_loop_of_her_acts_does_what_she_chooses_first(tmp_path, monkeypatch):
    from core.container import ServiceContainer
    from core.runtime.overt_action_loop import OvertActionLoop

    hers = _hers(tmp_path)

    async def does():
        return CameOf(got=1.0, what="done")

    hers.offers("goals", lambda: [_option("goal", does=does)])
    held = {"her_own_time": hers}
    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda name, default=None: held.get(name, default)))
    loop = OvertActionLoop()
    result = asyncio.run(loop._on_her_own_time())
    assert result["verified"] and result["selection_provenance"] == "her_own_choice" and result["objective"] == "goal 0"
    held.clear()
    assert asyncio.run(loop._on_her_own_time()) is None
