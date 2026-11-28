"""She cares about what she gives her own turns to, and its fate reaches her.

Bryan, 13 October 2026: "We need to give Aura the ability to care. Not just
'Care for' but intrinsically care about. Genuine interest that matters to her
and whatever branches/connects to that."

Each test is a small life. The module under test is
core/conation/what_she_cares_about.py, and the record it reads a thing's
company from is core/conation/what_she_wants.py.
"""
from __future__ import annotations

import pytest

from core.conation.engine import ConationEngine
from core.conation.what_she_cares_about import WhatSheCaresAbout, _at_least
from core.conation.what_she_wants import WhatSheWants, _conation, read_turn
from core.runtime.atomic_writer import flush_writes_behind

HERON, LOGS, REEDS, WATER = "topic:heron", "topic:log files", "topic:reeds", "topic:water"
LABELS = {HERON: "heron", LOGS: "log files", REEDS: "reeds", WATER: "water"}


@pytest.fixture
def record(tmp_path):
    return WhatSheWants(tmp_path / "wants.json", engine=ConationEngine())


def _care(company=None) -> WhatSheCaresAbout:
    return WhatSheCaresAbout(company=company)


# ── giving ──────────────────────────────────────────────────────────


def test_she_cares_about_what_she_keeps_going_back_to_of_her_own_accord():
    care = _care()
    for _ in range(12):
        care.gave(HERON)
    for key in (LOGS, REEDS, WATER):
        care.gave(key)
        care.was_brought(key)
    assert care.cares(HERON) is True
    assert care.cared_for() == [HERON]


def test_a_thing_she_was_only_ever_brought_is_not_cared_about_however_often():
    care = _care()
    for _ in range(40):
        care.was_brought(LOGS)
    for _ in range(12):
        care.gave(HERON)
    care.gave(REEDS)
    assert care.share(LOGS) == 0.0
    assert care.cares(LOGS) is False
    assert care.stake(LOGS) == 0.0


def test_one_turn_given_is_not_caring():
    care = _care()
    care.gave(HERON)
    care.was_brought(LOGS)
    assert care.cares(HERON) is None
    # The whole of what she has given, and still only what one turn can show.
    assert care.share(HERON) == pytest.approx(0.05, abs=1e-6)


def test_with_nothing_to_set_it_against_she_cannot_say():
    care = _care()
    for _ in range(30):
        care.gave(HERON)
    assert care.cares(HERON) is None


def test_her_turns_spread_evenly_are_caring_about_none_of_them():
    care = _care()
    for _ in range(6):
        for key in (HERON, LOGS, REEDS, WATER):
            care.gave(key)
    assert care.cared_for() == []
    assert all(care.cares(key) is None for key in (HERON, LOGS, REEDS, WATER))


def test_the_share_is_the_least_the_counts_allow():
    assert _at_least(0, 10) == 0.0
    assert _at_least(1, 1) == pytest.approx(0.05, abs=1e-6)
    assert _at_least(10, 10) == pytest.approx(0.05 ** 0.1, abs=1e-6)
    assert _at_least(5, 20) < 5 / 20
    assert _at_least(50, 200) > _at_least(5, 20)


# ── who put a subject there ─────────────────────────────────────────


def _summed_upper_tail(k: int, n: int, chance: float) -> float:
    """The binomial upper tail as it was computed before 16 November 2026, term by term."""
    import math

    return min(1.0, sum(
        math.exp(math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
                 + j * math.log(chance) + (n - j) * math.log(1.0 - chance))
        for j in range(k, n + 1)))


@pytest.mark.parametrize(("k", "n"), [(1, 2), (1, 50), (3, 40), (12, 40), (40, 40), (3, 4000), (40, 4000)])
def test_the_least_share_is_the_bound_the_summed_tail_gave(k, n):
    """The same Clopper-Pearson bound, read as a beta quantile in one call.

    LIVE 16 November 2026, 01:08: at four thousand turns one bound took 118 ms by the summed tail, her appraisal read
    hundreds of them on the event loop, and the loop stood still 7.0 s.
    """
    from core.conation.what_she_cares_about import _WRONG, _upper_tail

    low = _at_least(k, n)

    assert _summed_upper_tail(k, n, low) == pytest.approx(_WRONG, rel=1e-6)
    for chance in (0.001, 0.05, 0.5, 0.9):
        assert _upper_tail(k, n, chance) == pytest.approx(_summed_upper_tail(k, n, chance), rel=1e-9, abs=1e-15)


def test_a_bound_over_a_long_life_is_read_at_once():
    """A million turns given: the summed tail would add a million terms 48 times over for each thing."""
    low = _at_least(3, 1_000_000)

    assert 0.0 < low < 3 / 1_000_000


def test_a_subject_the_other_named_was_brought_and_one_only_she_named_she_gave():
    care = _care()
    read = care.exchange([HERON, REEDS], theirs=[HERON])
    assert read == {HERON: "brought", REEDS: "given"}
    assert (care.given(HERON), care.given(REEDS)) == (0, 1)


def test_staying_with_what_is_being_talked_about_is_not_picking_it():
    care = _care()
    care.exchange([HERON], theirs=[HERON])
    read = care.exchange([HERON], theirs=[])
    assert read == {HERON: "carried on"}
    assert care.given(HERON) == 0


def test_coming_back_to_it_after_the_talk_moved_on_is_hers():
    care = _care()
    care.exchange([HERON], theirs=[HERON])
    care.exchange([LOGS], theirs=[LOGS])
    read = care.exchange([LOGS, HERON], theirs=[LOGS])
    assert read[HERON] == "given"


# ── what reaches a thing from its company ───────────────────────────


def _with_company(table):
    return lambda key: table.get(key, (0, {}))


def test_a_thing_that_only_arrives_with_what_she_cares_about_has_her_stake_whole():
    company = {REEDS: (8, {HERON: 8}), WATER: (80, {HERON: 8})}
    care = _care(_with_company(company))
    for _ in range(12):
        care.gave(HERON)
    care.gave(LOGS)
    own = care.share(HERON)
    assert care.given(REEDS) == 0
    assert care.stake(REEDS) == pytest.approx(own)
    # A thing that turns up everywhere takes a tenth of it.
    assert care.stake(WATER) == pytest.approx(own / 10)
    assert care.stake(LOGS) < care.stake(REEDS)


def test_its_branches_are_what_it_reaches_the_most_reached_first():
    company = {
        HERON: (12, {REEDS: 8, WATER: 8}),
        REEDS: (8, {HERON: 8}),
        WATER: (80, {HERON: 8}),
    }
    care = _care(_with_company(company))
    for _ in range(12):
        care.gave(HERON)
    branches = care.branches(HERON)
    assert [key for key, _reach in branches] == [REEDS, WATER]
    assert branches[0][1] == pytest.approx(10 * branches[1][1])


# ── how a thing fared ───────────────────────────────────────────────


def test_what_she_feels_of_a_fate_is_her_stake_times_how_it_went():
    care = _care()
    for _ in range(12):
        care.gave(HERON)
    care.gave(LOGS)
    stake = care.stake(HERON)
    assert care.fared(HERON, -1.0) == {"stake": stake, "felt": -stake}
    assert care.fared(HERON, 0.5)["felt"] == pytest.approx(stake / 2)
    assert care.felt() == {HERON: pytest.approx(-stake / 2)}
    assert care.felt() == {}


def test_with_no_stake_the_news_is_only_news():
    care = _care()
    for _ in range(12):
        care.gave(HERON)
    assert care.fared(LOGS, -1.0) == {"stake": 0.0, "felt": 0.0}
    assert care.felt() == {}
    assert care.fate(LOGS) == -1.0


def test_two_lives_that_differ_only_in_what_she_gave_her_turns_to_feel_the_same_news_differently():
    felt = []
    for given_to in (HERON, LOGS):
        care = _care()
        for _ in range(12):
            care.gave(given_to)
        care.gave(REEDS)
        felt.append(care.fared(HERON, -1.0)["felt"])
    assert felt[0] < -0.5
    assert felt[1] == 0.0


def test_bad_news_of_what_she_has_a_stake_in_stays_with_her_until_better_news():
    care = _care()
    for _ in range(12):
        care.gave(HERON)
    care.gave(LOGS)
    care.fared(HERON, -1.0)
    care.fared(LOGS, -1.0)
    care.exchange([REEDS], theirs=[REEDS])
    concerns = care.concerns()
    assert [c.thing for c in concerns] == [HERON, LOGS]
    assert concerns[0].turns_since == 1 and concerns[0].weight > concerns[1].weight
    care.fared(HERON, 0.8)
    assert [c.thing for c in care.concerns()] == [LOGS]


def test_a_fate_that_is_not_a_number_is_no_news():
    care = _care()
    care.gave(HERON)
    for how in ("badly", None, float("nan")):
        assert care.fared(HERON, how) == {"stake": 0.0, "felt": 0.0}
    assert care.fate(HERON) is None


# ── in the record of what she has met ───────────────────────────────


class _Worth:
    measured, payoff = True, {}

    def __init__(self, worth: float, size: float = 0.7) -> None:
        self.worth, self.size = worth, size


ORDINARY = (
    "Can you check the time for me and tell me what you think of the plan?",
    "I think the plan is fine, and the time is what you would expect for a second look.",
)
HERON_THEIRS = "Tell me about the plan again, and whether the second look changes it."
HERON_HERS = (
    "The plan holds. And the heron: a heron waits because a heron's strike is the whole "
    "hunt, so the heron stands until the fish forgets the heron."
)


def _talked_before(record, exchanges: int = 45) -> None:
    for _ in range(exchanges):
        record.talk.about(*ORDINARY)


def test_a_subject_only_her_answer_names_is_a_turn_she_gave(record):
    from tests.fixtures.things_her_library_names import NamesThese

    _talked_before(record)
    record.about = NamesThese("heron")
    read_turn(["self"], _Worth(1.0), record, said=(HERON_THEIRS, HERON_HERS))
    assert record.care.given(HERON) == 1


def test_a_subject_the_other_person_named_is_not(record):
    from tests.fixtures.things_her_library_names import NamesThese

    _talked_before(record)
    record.about = NamesThese("heron")
    read_turn(["self"], _Worth(1.0), record, said=(HERON_HERS, "Yes. A heron stands and a heron waits, the heron."))
    assert record.thing(HERON) is not None
    assert record.care.given(HERON) == 0


def test_deciding_on_a_thing_when_she_is_free_is_a_turn_she_gave(record):
    for turn in range(24):
        record.met(["self", HERON], 0.6 + 0.01 * (turn % 3), reachable=[HERON], labels=LABELS)
        record.met(["self", LOGS], -0.5, reachable=[LOGS], labels=LABELS)
    for held in _conation(record).dynamics._satiation.values():
        held.last_update -= 600.0
    decided = record.decide()
    assert decided["thing"] == HERON
    assert record.care.given(HERON) == 1


def test_bad_news_of_a_thing_does_not_teach_her_to_like_it_less(record):
    pond = "game:heron pond"
    for _ in range(20):
        record.met(["self", pond], 0.6, reachable=[pond], labels={pond: "heron pond"})
    for _ in range(12):
        record.care.gave(pond)
    record.care.gave(LOGS)
    liked, contacts = record.thing(pond).liking, record.thing(pond).contacts
    record.care.fared(pond, -1.0)
    read_turn(["self", pond], _Worth(-1.0, 0.9), record)
    assert record.thing(pond).liking == liked
    assert record.thing(pond).contacts == contacts
    # The same turn with no news in it is a turn with the thing, and teaches.
    read_turn(["self", pond], _Worth(-1.0, 0.9), record)
    assert record.thing(pond).liking < liked


def test_what_she_started_and_finished_went_well_for_the_thing_and_what_failed_went_badly(record):
    record.met(["self", HERON], 0.6, reachable=[HERON], labels=LABELS)
    record.began(HERON, "act-1", "Draw the heron's strike frame by frame")
    record.ended(HERON, "act-1", finished=True)
    assert record.care.fate(HERON) == 1.0
    record.began(HERON, "act-2", "Find where the herons roost")
    record.ended(HERON, "act-2", finished=False)
    assert record.care.fate(HERON) == pytest.approx(0.0)
    assert [c.thing for c in record.care.concerns()] == []  # no stake yet


def test_a_thing_mentioned_in_passing_with_what_she_cares_about_is_wanted_for_its_sake(tmp_path):
    """Two lives that differ only in whether she has given her turns to the heron."""
    wanted = []
    for cares in (False, True):
        record = WhatSheWants(tmp_path / f"{cares}.json", engine=ConationEngine())
        for _ in range(20):
            record.met(["self", HERON], 0.0, reachable=[HERON], labels=LABELS)
        for _ in range(4):
            record.met(
                ["self", HERON, REEDS], 0.0, reachable=[HERON, REEDS], labels=LABELS,
                stood_out={REEDS: 0.05},
            )
        if cares:
            for _ in range(12):
                record.care.gave(HERON)
            record.care.gave(LOGS)
        for held in _conation(record).dynamics._satiation.values():
            held.last_update -= 600.0
        reeds = next((want for want in record.wants() if want.thing == REEDS), None)
        wanted.append(reeds.utility if reeds else 0.0)
        if cares:
            assert f"her stake in how it fares {record.care.stake(REEDS):.3f}" in reeds.why
    assert wanted[1] > 2 * wanted[0]


def test_what_she_cares_about_is_still_hers_after_a_restart(record, tmp_path):
    for _ in range(12):
        record.care.gave(HERON)
    record.care.gave(LOGS)
    record.care.fared(HERON, -1.0)
    record.met(["self"], 0.1)
    flush_writes_behind()
    again = WhatSheWants(tmp_path / "wants.json", engine=ConationEngine())
    assert again.care.cares(HERON) is True
    assert again.care.share(HERON) == pytest.approx(record.care.share(HERON))
    assert [c.thing for c in again.care.concerns()] == [HERON]
    assert again.status()["care"]["cares_about"][0]["thing"] == HERON


# ── felt ────────────────────────────────────────────────────────────


def _felt(record, *, subject=None, object_=None):
    from core.interiority.appraisal import AppraisalEngine
    from core.interiority.core_affect import core_affect
    from core.interiority.event import EventKind, InteriorEvent
    from core.interiority.ledger import RelationalLedger
    from core.interiority.stakes import _harvest_regard

    ledger = RelationalLedger()
    _harvest_regard(record, ledger)
    frame = AppraisalEngine(ledger).appraise(
        InteriorEvent(kind=EventKind.SOCIAL, subject=subject, object=object_, source="test")
    )
    return ledger, frame, core_affect(frame)


def _gives_herself_to(record, key) -> None:
    for _ in range(12):
        record.care.gave(key)
    record.care.gave(LOGS)


def test_what_she_cares_about_is_at_stake_in_an_event_about_it(record):
    _gives_herself_to(record, HERON)
    ledger, frame, _felt_now = _felt(record, object_="heron")
    stakes = ledger.stakes_for(subject=None, object_="heron")
    assert [(s.kind, s.weight) for s in stakes] == [("care", pytest.approx(record.care.stake(HERON)))]
    assert frame["relevance"].value == pytest.approx(record.care.stake(HERON))
    # And nothing is at stake in a thing she was only brought.
    _ledger, frame, _felt_now = _felt(record, object_="water")
    assert frame["relevance"].value == 0.0


def test_news_that_it_went_badly_is_felt_as_bad_and_good_news_as_good(record):
    """She has no like or dislike of the heron on record: what is felt is its fate."""
    _gives_herself_to(record, HERON)
    record.care.fared(HERON, -1.0)
    _ledger, frame, bad = _felt(record, object_="heron")
    assert frame["congruence"].value == -1.0
    assert bad.valence < -0.5
    record.care.fared(HERON, 1.0)
    record.care.fared(HERON, 1.0)
    record.care.fared(HERON, 1.0)
    _ledger, _frame, good = _felt(record, object_="heron")
    assert good.valence > 0.0


def test_the_same_news_of_a_thing_she_has_given_nothing_to_is_not_felt(record):
    _gives_herself_to(record, HERON)
    record.care.fared(WATER, -1.0)
    _ledger, _frame, felt = _felt(record, object_="water")
    assert felt.valence == 0.0


def test_a_person_she_cares_about_doing_worse_is_her_loss_and_doing_better_her_gain(record, monkeypatch):
    from core.container import ServiceContainer
    from core.interiority.evidence import absent, inferred
    from core.phases import learning_phase

    ines = "person:ines"
    _gives_herself_to(record, ines)
    monkeypatch.setattr("core.conation.what_she_wants._RECORD", record)
    monkeypatch.setattr(learning_phase, "_THEIR_DISTRESS", {})

    class Other:
        distress = absent(source="test")

    class Interior:
        last_other = Other()

    monkeypatch.setattr(
        ServiceContainer, "get",
        classmethod(lambda cls, name, default=None: Interior() if name == "interiority" else default),
    )
    stake = record.care.stake(ines)
    learning_phase._how_they_are_reaches_her(ines)
    assert record.care.fate(ines) is None  # no read of them is no news
    for distress, felt in ((0.2, None), (0.8, -0.6 * stake), (0.3, 0.5 * stake)):
        Other.distress = inferred(distress, confidence=1.0, source="test")
        record.care.felt()
        learning_phase._how_they_are_reaches_her(ines)
        got = record.care.felt().get(ines)
        assert got == (pytest.approx(felt) if felt is not None else None)
    assert record.care.concerns() == []


def test_what_a_process_was_in_the_middle_of_ended_with_it(record, tmp_path):
    """Live, 13 October: two games stayed under way from runs long gone, so she was always busy."""
    pond = "game:heron pond"
    record.started(pond, "play", label="heron pond")
    record.met(["self"], 0.1)
    assert record.thing(pond).under_way
    flush_writes_behind()
    again = WhatSheWants(tmp_path / "wants.json", engine=ConationEngine())
    assert again.thing(pond) is not None and again.thing(pond).under_way == {}
