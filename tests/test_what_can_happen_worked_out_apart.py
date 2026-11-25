"""What each side can do, worked out in a process of its own (core/agency/what_can_happen_worked_out_apart.py): asked,
it answers; stopped, it ends at its next step and keeps what it had finished; and the one process serves request after
request."""
from __future__ import annotations

import time

from core.agency.what_can_happen_worked_out_apart import WorkedOutApart
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

WORTH = {"P": 100.0, "N": 300.0, "B": 300.0, "R": 500.0, "Q": 900.0, "K": 0.0}


def test_worked_out_apart_it_answers_ends_when_stopped_and_keeps_what_it_finished() -> None:
    apart = WorkedOutApart()
    try:
        before = from_fen("4k3/8/8/8/1p6/2N5/1P6/4K3 b - -")
        counters, found = apart.submit(CHESS, before, WORTH, plies_at_most=4, seconds=20.0).result(timeout=120)
        theirs_take = [answer for (start, end, _becomes), (answer, _comes_to, _deep) in counters.items()
                       if (start, end) == ((3, 1), (2, 2))]
        assert theirs_take, "their pawn taking the knight is among what they can do, with her answer worked out"
        assert any(possible.deep >= 2 for possible in found)

        busy = from_fen("r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R b KQ -")
        work = apart.submit(CHESS, busy, WORTH, plies_at_most=10, seconds=60.0)
        time.sleep(2.0)
        stopped_at = time.monotonic()
        apart.stop()
        counters, found = work.result(timeout=30)
        assert time.monotonic() - stopped_at < 5.0, "stopped, it ends at its next step, not when its minute is up"
        assert counters and not all(possible.complete for possible in found)
    finally:
        apart.shutdown()
