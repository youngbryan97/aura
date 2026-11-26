"""What is saved of the positions her longer look searched never holds less of what was kept than the cap allows
(core/agency/what_a_kind_is_worth.py ``LookedAtLonger.with_what_was_kept``, core/skills/screen_pursuit_board_of_pieces.py
``_keep_what_she_learned``).

LIVE 25 November 2026 a run that began its rehearsal afresh, its weights lost, saved its 5,531 positions over the 83,781
kept, and how the pieces stand together was learned again from the few.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.agency import what_a_kind_is_worth
from core.agency.what_a_kind_is_worth import LookedAtLonger, Rehearsal, Weights
from core.runtime import what_she_learned
from core.skills import screen_pursuit_board_of_pieces as board_of_pieces

pytestmark = pytest.mark.unit

SQUARES = 64


def _store(kinds: list[str], placed: list[list[tuple[str, int, int]]], game0: int) -> LookedAtLonger:
    """A store of positions, each a list of (kind, whose: 0 the mover's, square), each worth its index."""
    parts = []
    for position in placed:
        parts.append(np.array([(kinds.index(kind) * 2 + whose) * SQUARES + square for kind, whose, square in position],
                              dtype=np.int16))
    return LookedAtLonger(kinds=kinds, squares=SQUARES, columns=8, parts=parts, glance=[0.0] * len(parts),
                          slow=[float(n) for n in range(len(parts))], game=[game0 + n for n in range(len(parts))],
                          scales=[1.0])


def _named(store: LookedAtLonger) -> list[tuple[tuple[str, int, int], ...]]:
    out = []
    for part in store.parts:
        slot, rest = np.divmod(part.astype(np.int64), 2 * SQUARES)
        whose, square = np.divmod(rest, SQUARES)
        out.append(tuple(sorted((store.kinds[s], int(w), int(q)) for s, w, q in zip(slot, whose, square, strict=True))))
    return out


KEPT = [[("K", 0, 4), ("K", 1, 60), ("Q", 0, 3)], [("K", 0, 6), ("K", 1, 62), ("R", 1, 0)]]
NEW = [[("K", 0, 5), ("K", 1, 61), ("N", 0, 21)]]


def test_a_store_begun_afresh_keeps_every_position_kept_before_it() -> None:
    kept = _store(["K", "Q", "R"], KEPT, game0=0)
    fresh = _store(["K", "N"], NEW, game0=900)
    merged = fresh.with_what_was_kept(kept)
    assert len(merged.parts) == 3
    assert _named(merged)[:2] == [tuple(sorted(p)) for p in KEPT], "what was kept comes first, as it was"
    assert _named(merged)[2] == tuple(sorted(NEW[0]))
    assert merged.game == [0, 1, 900]


def test_a_store_read_from_what_was_kept_holds_nothing_twice() -> None:
    kept = _store(["K", "Q", "R"], KEPT, game0=0)
    grown = _store(["K", "Q", "R"], KEPT, game0=0)
    more = _store(["K", "Q", "R", "N"], NEW, game0=900)
    grown.kinds = more.kinds
    grown.parts.append(more.parts[0])
    grown.glance.append(0.0)
    grown.slow.append(0.0)
    grown.game.append(900)
    assert grown.with_what_was_kept(kept) is grown


def test_kinds_counted_in_another_order_are_read_by_name() -> None:
    kept = _store(["R", "Q", "K"], KEPT, game0=0)
    fresh = _store(["K", "N"], NEW, game0=900)
    merged = fresh.with_what_was_kept(kept)
    assert set(_named(merged)) == {tuple(sorted(p)) for p in KEPT + NEW}


def test_past_the_cap_what_was_kept_longest_goes_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(what_a_kind_is_worth, "LOOKED_AT_LONGER_AT_MOST", 2)
    merged = _store(["K", "N"], NEW, game0=900).with_what_was_kept(_store(["K", "Q", "R"], KEPT, game0=0))
    assert merged.game == [1, 900]


def test_a_save_from_a_rehearsal_begun_afresh_keeps_what_was_kept(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(what_she_learned, "_kept_in", lambda: tmp_path)
    record = board_of_pieces._her_record("A Board")
    what_she_learned.remember_beside(record, board_of_pieces._LOOKED_AT_LONGER,
                                     _store(["K", "Q", "R"], KEPT, game0=0).as_bytes())
    fresh = Rehearsal(weights=Weights(worth={"K": 0.0, "N": 300.0}), earned=["worth"], royal=["K"])
    fresh.looked_at_longer = _store(["K", "N"], NEW, game0=900)
    board_of_pieces._keep_what_she_learned("A Board", SimpleNamespace(as_memory=lambda: {}), None, fresh)
    saved = LookedAtLonger.from_bytes(what_she_learned.recall_beside(record, board_of_pieces._LOOKED_AT_LONGER))
    assert len(saved.parts) == 3
    assert len(fresh.looked_at_longer.parts) == 3, "she learns from what was kept from now on"


def test_a_store_merged_with_what_it_carries_on_from_does_not_double_its_scales() -> None:
    """LIVE 26 November 2026 the scales kept beside her board numbered 1,314,914,303 beside 100,000 positions: every
    merge set the kept history before the history that carried on from it. Read back, they were 1.3 billion Python
    floats, about 40 GB in her own process, and her memory watchdog ended her three times in twenty minutes."""
    kept = _store(["K", "Q", "R"], KEPT, game0=0)
    kept.game = [0, 10_000]
    kept.scales = [1.0, 2.0]
    store = LookedAtLonger.from_bytes(kept.as_bytes())
    store.scales.append(3.0)
    store.game.append(20_000)
    store.parts.append(store.parts[0])
    store.glance.append(0.0)
    store.slow.append(9.0)
    for _ in range(40):
        store = store.with_what_was_kept(LookedAtLonger.from_bytes(store.as_bytes()))
    assert store.scales == [1.0, 2.0, 3.0], "each round's scale once, for the three rounds whose positions are kept"


def test_scales_are_kept_for_the_rounds_whose_positions_are_kept_and_no_more() -> None:
    kept = _store(["K", "Q", "R"], KEPT, game0=0)
    kept.scales = [float(n) for n in range(1_000)]
    assert LookedAtLonger.from_bytes(kept.as_bytes()).scales == [999.0], "both positions came from one round"
    kept.game = [0, 10_000]
    assert LookedAtLonger.from_bytes(kept.as_bytes()).scales == [998.0, 999.0]
