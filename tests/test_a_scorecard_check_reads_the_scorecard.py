"""A scorecard check read a field no scorecard has.

The completion tracker compared a criterion's `standing` with what the evidence
wanted, and the scorecard calls it `verdict`, so every criterion check read
False. The two entries that used this kind asked questions of the whole card
instead and raised on the missing key, which left the frozen campaign's item
open after the campaign had run.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.isc_completion_status import Checker  # noqa: E402

CARD = {
    "runs": 3,
    "campaigns": {"8874aea9": ["run 1", "run 2", "run 3"]},
    "criteria": [
        {"criterion": "ownership", "verdict": "holds", "held": 3, "runs": 3},
        {"criterion": "synergy", "verdict": "fails", "held": 0, "runs": 3},
    ],
}


def _checker() -> Checker:
    checker = Checker()
    checker.card = CARD
    return checker


def test_a_criterion_is_read_by_its_verdict() -> None:
    checker = _checker()
    assert checker._check_scorecard({"kind": "scorecard", "criterion": "ownership"})[0] is True
    assert checker._check_scorecard({"kind": "scorecard", "criterion": "synergy"})[0] is False


def test_an_expression_is_asked_of_the_card() -> None:
    checker = _checker()
    ok, why = checker._check_scorecard(
        {"kind": "scorecard", "expr": "card.get('runs', 0) == 3 and len(card.get('campaigns', {})) == 1"}
    )
    assert ok is True, why
