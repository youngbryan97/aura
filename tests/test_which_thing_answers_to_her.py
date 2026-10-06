"""Her thing is the one whose movement depends on which key she holds, measured only while she tries keys."""
from __future__ import annotations

import pytest

from core.agency.which_one_answers_to_her import WhichIsHers

pytestmark = pytest.mark.unit


class _Thing:
    def __init__(self, number, kind):
        self.number, self.kind = number, kind
        self.x = self.y = 50.0
        self.w, self.h = (3.0, 20.0) if kind == 0 else (3.0, 3.0)
        self.size = self.w * self.h
        self.vx = self.vy = 0.0
        self.seen = self.born = 0.0
        self.moved = True


class _Moves:
    def __init__(self, things):
        self.things = {t.number: t for t in things}


def _run(trying: bool, ball_follows_keys: bool):
    mine, ball = _Thing(1, 0), _Thing(2, 1)
    moves = _Moves([mine, ball])
    hers = WhichIsHers()
    at = 0.0
    for step in range(120):
        key = ("up", "down", "")[(step // 10) % 3]
        hers.holding(key, at, trying=trying)
        at += 0.03
        mine.vy = {"up": -120.0, "down": 120.0, "": 0.0}[key]
        ball.vx = 150.0 if (step // 25) % 2 else -150.0
        ball.vy = mine.vy * 1.5 if ball_follows_keys else (80.0 if (step // 7) % 2 else -80.0)
        for thing in (mine, ball):
            thing.seen = at
        hers.saw(moves, [], at)
    return hers


def test_the_thing_that_answers_to_her_keys_is_hers():
    hers = _run(trying=True, ball_follows_keys=False)
    assert hers.number == 1
    up, down = hers.way_of("up"), hers.way_of("down")
    assert up[1] < -100 and down[1] > 100


def test_what_her_own_choices_make_move_is_not_taken_for_hers():
    """While she plays she presses up because the ball goes up; that is no evidence about the ball."""
    hers = _run(trying=False, ball_follows_keys=True)
    assert hers.number is None


def test_immediate_motion_after_a_key_change_is_not_evidence_for_the_old_key():
    mine = _Thing(1, 0)
    hers = WhichIsHers()
    hers.number, hers.kind = 1, 0
    hers.holding("left", 1.0, trying=True)
    hers.holding("up", 2.0, trying=True)
    mine.vy, mine.seen = -120.0, 2.01
    hers.saw(_Moves([mine]), [], mine.seen)
    assert not hers._hers.free.get("left")
    assert not hers._hers.free.get("up")
    assert "left" not in hers._by_thing[1].by_press
    mine.seen = 2.3
    hers.saw(_Moves([mine]), [], mine.seen)
    assert hers._hers.free["up"] == [(0.0, -120.0)]


def test_a_ball_that_bounced_between_presses_is_not_taken_for_hers():
    """Picture by picture, a ball going one way through one press and the other way through the next answers to the keys."""
    ball = _Thing(2, 1)
    moves = _Moves([ball])
    hers = WhichIsHers()
    at = 0.0
    for step in range(200):
        key = ("up", "", "down", "")[(step // 10) % 4]
        hers.holding(key, at, trying=True)
        at += 0.03
        ball.vy = 100.0 if (step // 13) % 2 else -100.0
        ball.seen = at
        hers.saw(moves, [], at)
    assert hers.number is None


def _pointer_run(follower_x):
    """The pointer is swept there and back; ``follower_x(pointer_x)`` says where the thing is."""
    thing = _Thing(1, 0)
    moves = _Moves([thing])
    hers = WhichIsHers()
    at = 0.0
    xs = ([20 + 4 * i for i in range(15)] + [80 - 4 * i for i in range(15)]) * 4
    for x in xs:
        hers.pointed(float(x), 50.0, at)
        at += 0.2
        thing.x, thing.seen = follower_x(x, at), at
        hers.saw(moves, [], at)
    return hers


def test_a_thing_that_turns_when_the_pointer_turns_follows_it():
    assert _pointer_run(lambda x, _at: float(x)).follows_pointer


def test_a_thing_going_the_pointer_s_way_by_itself_does_not_follow_it():
    """A thing that goes on the same way while the pointer comes back is not following it."""
    assert not _pointer_run(lambda _x, at: 20.0 + 10.0 * at).follows_pointer


def test_a_thing_that_does_not_go_where_her_key_sends_it_is_not_hers_after_all():
    mine = _Thing(1, 0)
    moves = _Moves([mine])
    hers = WhichIsHers()
    hers.number, hers.kind = 1, 0
    for _ in range(10):
        hers._hers.add("up", 0.0, -120.0)
    at = 0.0
    hers.holding("up", at)
    for _step in range(60):
        at += 0.03
        mine.seen, mine.vy = at, 0.0
        hers.saw(moves, [], at)
    assert hers.number is None and 1 in hers.not_mine


def test_a_disproven_identity_cannot_promote_an_old_rivals_correlation():
    mine, rival = _Thing(1, 0), _Thing(3, 0)
    rival.x = 400.0
    moves = _Moves([mine, rival])
    hers = WhichIsHers()
    hers.number, hers.kind = 1, 0
    for _ in range(10):
        hers._hers.add("up", 0.0, -120.0)
    for press in range(8):
        key, speed = ("up", -120.0) if press % 2 else ("down", 120.0)
        for _ in range(10):
            hers._by_thing[3].add(key, 0.0, speed, press=float(press))
    assert hers._by_thing[3].ratio()[0] > 5
    hers.holding("up", 0)
    for step in range(60):
        at = (step + 1) * 0.03
        mine.seen, mine.vy = at, 0.0
        rival.seen = at
        hers.saw(moves, [], at)
        assert hers.number != rival.number
    assert hers.number is None and hers.kind is None
    assert hers.way_of("up") is None and not hers._by_thing


def test_a_missing_identity_cannot_transfer_control_to_a_distant_lookalike():
    mine, rival = _Thing(1, 0), _Thing(3, 0)
    rival.x = 400.0
    hers = WhichIsHers()
    hers.number, hers.kind = 1, 0
    mine.seen = 1.0
    hers.saw(_Moves([mine]), [], 1.0)
    for press in range(8):
        key = "up" if press % 2 else "down"
        for _ in range(10):
            hers._by_thing[3].add(key, 0.0, -120.0 if key == "up" else 120.0, press=float(press))
    hers.saw(_Moves([rival]), [], 1.03)
    assert hers.number is None and hers.lost()
    assert not hers._by_thing and hers.lost_at == 1.03
    # Looking like her, and seen for a while, is still not answering to her keys.
    for step in range(60):
        rival.seen = at = 1.06 + step * 0.03
        hers.saw(_Moves([rival]), [], at)
    assert hers.number is None


def test_lost_from_sight_she_is_found_again_by_trying_her_keys():
    """Offline 2026-10-06 a game's end screen hid her paddle; what her keys do was forgotten and never tried again."""
    hers = _run(trying=True, ball_follows_keys=False)
    assert hers.number == 1
    up = hers.way_of("up")
    # The end screen: every thing gone, and a new game draws them afresh, far from where she was.
    hers.saw(_Moves([]), [], 10.0)
    assert hers.lost() and hers.way_of("up") == up
    mine, ball, rival = _Thing(7, 0), _Thing(8, 1), _Thing(9, 0)
    mine.y, rival.x = 90.0, 400.0
    moves = _Moves([mine, ball, rival])
    at = 12.0
    for step in range(90):
        key = ("up", "down", "")[(step // 10) % 3]
        hers.holding(key, at, trying=True)
        at += 0.03
        mine.vy = {"up": -120.0, "down": 120.0, "": 0.0}[key]
        rival.vy = 60.0 if (step // 13) % 2 else -60.0
        ball.vx, ball.vy = 150.0, (80.0 if (step // 7) % 2 else -80.0)
        for thing in (mine, ball, rival):
            thing.seen = at
        hers.saw(moves, [], at)
    assert hers.number == 7 and not hers.lost()


def test_a_nearby_continuing_track_keeps_its_measured_controls():
    mine, continuing, rival = _Thing(1, 0), _Thing(4, 0), _Thing(3, 0)
    continuing.x, continuing.seen = 53.0, 1.03
    rival.x = 400.0
    hers = WhichIsHers()
    hers.number, hers.kind = 1, 0
    mine.seen = 1.0
    for _ in range(10):
        hers._hers.add("up", 0.0, -120.0)
    hers.saw(_Moves([mine]), [], 1.0)
    hers.saw(_Moves([continuing, rival]), [], 1.03)
    assert hers.number == 4 and hers.way_of("up") == (0.0, -120.0)


def test_one_contradicted_key_cannot_disown_another_measured_control():
    mine = _Thing(1, 0)
    hers = WhichIsHers()
    hers.number, hers.kind = 1, 0
    for _ in range(10):
        hers._hers.add("up", 0.0, -120.0)
        hers._hers.add("down", 0.0, 120.0)
    hers.holding("up", 0.0)
    for step in range(70):
        mine.seen = at = (step + 1) * 0.03
        hers.saw(_Moves([mine]), [], at)
    assert hers.number == 1


def test_a_thing_that_keeps_pace_with_the_pointer_far_from_it_does_not_follow_it():
    """A target sliding to and fro can keep pace with a sweep; one the mouse moves is under the mouse."""
    thing = _Thing(1, 0)
    moves = _Moves([thing])
    moves.shape = (100, 200)
    hers = WhichIsHers()
    at = 0.0
    xs = [20 + 4 * i for i in range(30)] + [140 - 4 * i for i in range(30)]
    for x in xs:
        hers.pointed(float(x), 50.0, at)
        at += 0.2
        thing.x, thing.seen = 200.0 - x * 0.9 + (x - 80) * 1.9, at
        hers.saw(moves, [], at)
    assert not hers.follows_pointer


def test_one_key_wrongly_believed_does_not_disown_her_thing():
    """LIVE 2026-10-05: 'left' was believed to move her paddle; while she held
    it the paddle did not move, she disowned her own paddle and played the
    computer's for a game."""
    mine, other = _Thing(1, 0), _Thing(3, 0)
    other.x = 400.0
    moves = _Moves([mine, other])
    hers = WhichIsHers()
    at = 0.0
    for _ in range(60):
        hers._hers.add("left", 0.0, -120.0)  # the wrong belief
    for step in range(400):
        key = ("up", "down", "left", "left", "left")[(step // 10) % 5]
        hers.holding(key, at, trying=True)
        at += 0.03
        mine.vy = {"up": -120.0, "down": 120.0}.get(key, 0.0)
        mine.y = min(400.0, max(10.0, mine.y + mine.vy * 0.03))
        other.vy = 90.0 if (step // 4) % 2 else -90.0
        other.y = min(400.0, max(10.0, other.y + other.vy * 0.03))
        for thing in (mine, other):
            thing.seen = at
        hers.saw(moves, [], at)
    assert hers.number == 1 and 1 not in hers.not_mine


def test_a_transient_seen_during_one_key_press_cannot_establish_control():
    from core.agency.which_one_answers_to_her import _Speeds

    speeds = _Speeds()
    for press, key, speed in [(0.0, "a", 120.0), (1.0, "b", -120.0), (2.0, "b", -120.0)]:
        for _ in range(10):
            speeds.add(key, speed, 0.0, press=press)
    assert speeds.ratio() == (0.0, 0.0)


def test_competing_old_trial_evidence_cannot_revoke_a_control_that_still_answers():
    mine, other = _Thing(1, 0), _Thing(2, 1)
    hers = WhichIsHers()
    hers.number, hers.kind = mine.number, mine.kind
    for number, speed in [(mine.number, 120.0), (other.number, 400.0)]:
        for press in range(6):
            key = "a" if press % 2 else "b"
            for _ in range(10):
                hers._by_thing[number].add(key, speed if key == "a" else -speed, 0.0, press=press)
    hers._decide(_Moves([mine, other]))
    assert hers.number == mine.number


def test_blocked_or_unsettled_pictures_do_not_establish_a_control():
    from core.agency.which_one_answers_to_her import _Speeds

    speeds = _Speeds()
    for _ in range(50):
        speeds.add("a", 0.0, 0.0, pinned=True)
        speeds.add("b", 0.0, -80.0, settled=False)
    assert speeds.typical("a") is None
    assert speeds.typical("b") is None
    for _ in range(4):
        speeds.add("a", 80.0, 0.0)
    assert speeds.typical("a") == (80.0, 0.0)


def test_control_discovery_keeps_only_settled_free_trial_measurements():
    mine = _Thing(1, 0)
    hers = WhichIsHers()
    trials = hers._by_thing[mine.number]
    for press in range(8):
        key = "q" if press % 2 else "a"
        for _ in range(4):
            trials.add(key, 0.0, 80.0 if key == "q" else 0.0,
                       press=press, pinned=key == "a")
    hers._decide(_Moves([mine]))
    assert hers.number == mine.number
    assert hers.way_of("q") == (0.0, 80.0)
    assert hers.way_of("a") is None
    assert not hers.keys_known(["q", "a"])
    for _ in range(4):
        hers._hers.add("a", 0.0, -80.0)
    assert hers.keys_known(["q", "a"])
    assert hers.way_of("a") == (0.0, -80.0)


def test_stationary_observations_at_a_visited_extreme_remain_ambiguous():
    from core.agency.which_one_answers_to_her import _Speeds

    speeds = _Speeds()
    speeds.add("q", 80.0, 0.0, position=(20.0, 50.0))
    speeds.add("q", 80.0, 0.0, position=(80.0, 50.0))
    for _ in range(20):
        speeds.add("a", 0.0, 0.0, position=(80.0, 50.0))
    assert speeds.typical("a") is None
    # A key observed doing nothing away from either extreme is measured.
    for _ in range(4):
        speeds.add("b", 0.0, 0.0, position=(50.0, 50.0))
    assert speeds.typical("b") == (0.0, 0.0)


def test_control_evidence_claim_runs_its_registered_measurement():
    from core.agency.which_one_answers_to_her import _control_measurement_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _control_measurement_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == "ambiguous_controls_remain_unknown")
    assert check.predict(None) is True
    assert any(c.test == check.name for c in suite.claims())


def test_lost_she_knows_herself_by_how_her_keys_move_her_in_two_presses():
    """What her keys do predicts how her thing moves: two trial presses find her, not a new experiment of eight."""
    hers = _run(trying=True, ball_follows_keys=False)
    hers.saw(_Moves([]), [], 10.0)
    assert hers.lost()
    mine, still, rival = _Thing(7, 0), _Thing(8, 0), _Thing(9, 0)
    still.x, rival.x = 300.0, 400.0
    moves = _Moves([mine, still, rival])
    at = 12.0
    for step in range(20):
        key = "up" if step < 10 else ""
        hers.holding(key, at, trying=True)
        at += 0.03
        mine.vy = -120.0 if key == "up" else 0.0
        rival.vy = 90.0 if step % 6 < 3 else -90.0  # a lookalike going its own way, as the other side's does
        for thing in (mine, still, rival):
            thing.seen = at
        hers.saw(moves, [], at)
    assert hers.number == 7
    assert hers.identification.receipts[-1]["reason"] == "her measured controls predicted its trial presses"


def test_lost_she_does_not_take_a_lookalike_her_keys_do_not_move():
    hers = _run(trying=True, ball_follows_keys=False)
    hers.saw(_Moves([]), [], 10.0)
    rival = _Thing(9, 0)
    moves = _Moves([rival])
    at = 12.0
    for step in range(40):
        key = ("up", "", "down", "")[step // 10]
        hers.holding(key, at, trying=True)
        at += 0.03
        rival.vy = 90.0  # moving, but the same way whatever she holds
        rival.seen = at
        hers.saw(moves, [], at)
    assert hers.number is None and hers.lost()
