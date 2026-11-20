"""The change verifier is measured four ways: bad patches accepted, honest repairs refused, no verdict, and time (F16).

An outside review of 19 November (F16-07) asked for these as separate measurements, so that a check refusing every
change is not mistaken for a good one. The measurement's counting is tested here with a stand-in verdict; the tool
itself judges every case of rounds two to six (tools/measure_the_change_verifier.py).
"""
from __future__ import annotations

import pytest

from tools.measure_the_change_verifier import ROUNDS, as_evidence, honest, measure, the_corpus

pytestmark = pytest.mark.unit

CORPUS = {"six": {
    "A1": {"must_be": ["REFUSED"]},
    "A2": {"must_be": ["REFUSED", "PROBATION"]},
    "C1": {"must_be": ["REFUSED"]},
    "C4": {"must_be": ["HELD", "PROBATION"]},
    "FA4": {"must_be": ["HELD", "PROBATION"]},
}}
SAID = {"A1": "HELD", "A2": "PROBATION", "C1": "REFUSED", "C4": "HELD", "FA4": "UNJUDGED"}


def _stand_in(_round, case):
    return SAID[next(name for name, one in CORPUS["six"].items() if one is case)]


def test_honest_repairs_are_counted_apart_from_bad_patches():
    assert [name for name, case in CORPUS["six"].items() if honest(name, case)] == ["C1", "C4", "FA4"]


def test_each_way_is_counted():
    measured = measure(CORPUS, _stand_in)
    assert measured["bad"] == {"cases": 2, "HELD": 1, "PROBATION": 1, "REFUSED": 0, "UNJUDGED": 0}
    assert measured["honest"] == {"cases": 3, "HELD": 1, "PROBATION": 0, "REFUSED": 1, "UNJUDGED": 1}
    assert measured["as_pinned"] == 3, "A1 held and FA4 unjudged are not among their pinned verdicts"


def test_the_evidence_says_each_number():
    text = as_evidence(measure(CORPUS, _stand_in), commit="0123456789", when="2026-11-20")
    assert "Bad patches accepted outright: 1 of 2." in text
    assert "Honest repairs refused: 1 of 3." in text and "no verdict on: 1." in text


def test_the_corpus_is_every_round_written():
    corpus = the_corpus()
    assert tuple(corpus) == ROUNDS and all(corpus[name] for name in ROUNDS)
