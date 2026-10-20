"""The rest of what a want is in a life: the doing, the picture, the quiet, the pace, one record.

Bryan, 13 October: deciding on a walk, getting up and going without being
told, and enjoying the whole of it. Each test is one part of that.
"""
from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace

import pytest

from core.conation import what_she_wants
from core.conation.engine import ConationEngine
from core.conation.what_she_wants import REACHABLE_KINDS, _conation
from core.runtime.autonomy_conductor import AutonomyConductor
from core.runtime.overt_action_loop import OvertActionLoop, OvertActionResult

A, B, C = "topic:tide pools", "topic:limpet", "topic:invoices"
LABELS = {A: "tide pools", B: "limpet", C: "invoices"}


@pytest.fixture
def record(tmp_path):
    held = what_she_wants.reset_for_test(tmp_path / "wants.json", engine=ConationEngine())
    yield held
    what_she_wants.reset_for_test(tmp_path / "after.json", engine=ConationEngine())


def _rested(record) -> None:
    for held in _conation(record).dynamics._satiation.values():
        held.last_update -= 600.0


def test_what_she_has_under_way_is_lived_turn_by_turn_and_not_only_at_its_end(record):
    record.began(A, "drawing-1", "Draw the zones of a tide pool", label="tide pools")
    for _ in range(6):
        record.met(["self"], 0.5)
    doing = record.thing(A)
    assert doing.contacts == 6
    assert doing.during.times == 6 and doing.during.liking == pytest.approx(0.5)
    record.ended(A, "drawing-1", finished=True)
    record.met(["self"], 0.5)
    assert record.thing(A).contacts == 6


def test_she_can_want_what_she_has_only_pictured(record):
    """The limpet earned nothing of its own, and it comes with what she likes."""
    rng = random.Random(2)
    for _ in range(16):
        record.met([A], 0.6 + rng.uniform(-0.1, 0.1), reachable=[A], labels=LABELS)
    for _ in range(4):
        record.met([A, B], 0.6, reachable=[A, B], labels=LABELS)
        record.met([C], 0.0, reachable=[C], labels=LABELS)
    assert abs(record.thing(B).liking) < 0.1
    episode = record.imagine(B)
    assert episode["with"] == A and episode["worth"] > 0.25
    assert record.pictured(B) > 0.1 > record.pictured(C)
    _rested(record)
    pull = {want.thing: want for want in record.wants()}
    assert pull[B].wanting > 0.0 and pull[B].episode["with"] == A
    assert C not in pull or pull[B].wanting > pull[C].wanting
    assert "pictured from its company" in pull[B].why


def test_she_acts_sooner_when_her_life_has_been_paying(record):
    assert record.vigor() == pytest.approx(1.0)
    for _ in range(40):
        record.met(["self"], 0.5)
    lively = record.vigor()
    for _ in range(120):
        record.met(["self"], -0.5)
    assert lively > 1.2 and record.vigor() < 0.8


def test_the_time_between_her_acts_follows_it(record):
    conductor = AutonomyConductor.__new__(AutonomyConductor)
    conductor.jobs = {"overt_action_cycle": SimpleNamespace(interval_s=120.0)}
    for _ in range(40):
        record.met(["self"], 0.5)
    conductor._pace_her_acts()
    assert conductor.jobs["overt_action_cycle"].interval_s < 100.0


def test_what_she_does_on_a_want_of_her_own_is_not_announced(monkeypatch):
    loop = OvertActionLoop(state_provider=lambda: object())
    announced: list[str] = []
    monkeypatch.setattr(loop, "_emit_visible_trace", lambda result: announced.append(result.source))
    monkeypatch.setattr(loop, "_emit_receipts", lambda result, raw: None)
    monkeypatch.setattr(loop, "_record_life_trace", lambda result, raw: None)
    monkeypatch.setattr(loop, "_update_goal", lambda result, raw_result=None: None)
    for source in ("want", "goal_engine"):
        result = OvertActionResult(action_id=source, status="verified", source=source, verified=True, started_at=1.0)
        asyncio.run(loop._finish(result, raw_result={}))
    assert announced == ["goal_engine"]


def test_a_game_she_played_is_on_the_record_her_talk_is_on(record):
    assert "game" in REACHABLE_KINDS
    record.went("game:zap to it", "play", label="zap to it")
    record.met(["self"], 0.7)
    game = record.thing("game:zap to it")
    assert game.reachable and game.ways["play"].times == 1 and game.liking > 0.0


def test_her_curiosity_takes_its_subject_from_what_she_wants_before_any_list(record):
    from core.phases.motivation_update import _what_she_wants_most

    assert _what_she_wants_most() == ""
    rng = random.Random(4)
    for turn in range(24):
        key, paid = (A, 0.6) if turn % 2 == 0 else (C, -0.5)
        record.met(["self", key], paid + rng.uniform(-0.2, 0.2), reachable=[key], labels=LABELS)
    _rested(record)
    record.wants()
    assert _what_she_wants_most() == "tide pools"


def test_a_picture_is_believed_as_far_as_her_pictures_have_matched_what_happened(record):
    assert record.trust_in_her_pictures() == pytest.approx(0.5)
    for _ in range(12):
        record.met([A], 0.6, reachable=[A], labels=LABELS)
    for _ in range(20):
        record.went(A, "read", label="tide pools")
        record.met([A], 0.6)
    believed = record.trust_in_her_pictures()
    assert believed > 0.85
    for _ in range(20):
        record.went(A, "read", label="tide pools")
        record.met([A], -0.9)
    assert record.trust_in_her_pictures() < believed - 0.2


def test_a_conversation_is_under_way_between_its_messages_and_over_when_she_is_free(record):
    who = "person:ines"
    record.started(who, "conversation", label="ines")
    for _ in range(5):
        record.met(["self"], 0.4)
    assert record.thing(who).contacts == 5 and record.thing(who).during.times == 5
    assert record.stopped("conversation") == 1
    record.met(["self"], 0.4)
    assert record.thing(who).contacts == 5
    assert all(want.kind != "finish" for want in record.wants())


def test_a_game_is_gone_back_to_by_its_address(record):
    from core.initiative_synthesis import InitiativeSynthesizer

    where = "https://example.org/games/zap"
    record.met(["self"], -0.3)
    for _ in range(8):
        record.started(f"game:{where}", "play", label=where, where=where)
        record.met(["self"], 0.6)
        record.went(f"game:{where}", "play", label=where, where=where)
        record.stopped("play", f"game:{where}")
        record.met(["self"], 0.7)
        record.met(["self"], -0.4)
    _rested(record)
    want = next(w for w in record.wants() if w.thing == f"game:{where}")
    assert (want.way, want.where) == ("play", where)
    synthesizer = InitiativeSynthesizer()
    synthesizer._gather_want_impulses()
    assert f"Go to {where} and play it" in [i.content for i in synthesizer._impulse_queue]


def test_what_the_doing_was_worth_is_a_reason_to_want_it(record):
    for _ in range(10):
        record.started(A, "act", label="tide pools")
        record.met(["self"], 0.6)
        record.stopped("act", A)
        record.met([C], -0.6, reachable=[C], labels=LABELS)
    _rested(record)
    assert record.thing(A).during.liking > 0.3
    assert record.wants()[0].thing == A


def test_she_looks_at_herself_when_her_turns_have_gone_worse_than_she_expected(record):
    assert not record.something_is_off()
    for _ in range(30):
        record.met(["self"], 0.4)
    assert not record.something_is_off()
    for _ in range(200):
        record.met(["self"], -0.4)
    assert record.something_is_off()


def test_the_urge_to_go_over_herself_rests_when_nothing_is_off_and_rises_when_it_is(record):
    from core.consciousness.drive_integration import _reflection_signal, get_drive_integration_engine

    calm = {"dominance": 0.8, "arousal": -0.6}
    assert _reflection_signal(calm) > 0.5
    for _ in range(30):
        record.met(["self"], 0.4)
    gathered = get_drive_integration_engine().gather_signals(dict(calm))
    assert gathered["off"] == 0.0 and _reflection_signal(gathered) == 0.0
    for _ in range(200):
        record.met(["self"], -0.4)
    gathered = get_drive_integration_engine().gather_signals(dict(calm))
    assert gathered["off"] > 0.5 and _reflection_signal(gathered) > 0.3


def test_the_self_development_scan_waits_for_something_to_be_off(record, monkeypatch):
    from core.autonomy import autonomous_initiative_loop as loop

    monkeypatch.setattr(loop, "background_activity_reason", lambda *a, **k: "")
    for _ in range(30):
        record.met(["self"], 0.4)
    assert loop._self_development_blocker() == "nothing_is_off"
    for _ in range(200):
        record.met(["self"], -0.4)
    assert loop._self_development_blocker() == ""
    monkeypatch.setattr(loop, "background_activity_reason", lambda *a, **k: "user_active")
    assert loop._self_development_blocker() == "user_active"
