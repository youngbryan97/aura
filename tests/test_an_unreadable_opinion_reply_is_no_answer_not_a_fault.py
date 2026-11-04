"""A model reply that cannot be read as a position is no answer this tick, not a fault of the opinion engine.

On her day runs of 1 and 3 November the opinion engine recorded every malformed reply as a degradation, and as a
fail-closed subsystem each became a critical incident: replies in prose ("Expecting value: line 1 column 1"), with
single quotes ("Expecting property name enclosed in double quotes"), and cut off when the lane was cancelled
("Unterminated string"). A cut-off reply also raised SyntaxError from the parser's literal fallback, which nothing
caught.
"""
from __future__ import annotations

import asyncio

import pytest

from core.epistemics import opinion_engine as module
from core.utils.json_utils import SelfHealingJSON

pytestmark = pytest.mark.unit


class _Brain:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def generate(self, *_args, **_kwargs) -> str:
        return self.reply


@pytest.fixture
def engine(tmp_path, monkeypatch):
    found: list[tuple] = []
    monkeypatch.setattr(module, "record_degradation", lambda *a, **k: found.append(a))
    eng = module.OpinionEngine(db_path=tmp_path / "opinions.json")
    eng.found = found
    return eng


def _form(engine, reply: str):
    engine._get_brain = lambda: _Brain(reply)
    return asyncio.run(engine.form_opinion("tea"))


def test_a_cut_off_reply_does_not_raise_from_the_parser():
    assert SelfHealingJSON().parse_sync('{"position": "I like tea", "reasoning": "it is warm') == {}


@pytest.mark.parametrize(
    "reply",
    [
        "I think tea is better, honestly.",
        '{"position": "I like tea", "confidence": 0.8, "reasoning": "it is wa',
        "{\n  'position': 'I like tea',\n  'confidence': 0.8,\n  reasoning: x\n}",
    ],
)
def test_an_unreadable_reply_forms_nothing_and_records_no_fault(engine, reply):
    assert _form(engine, reply) is None
    assert engine.found == []
    assert engine.unreadable_replies == 1
    assert engine.get_status()["unreadable_replies"] == 1


def test_a_fenced_reply_is_read_without_the_model(engine):
    opinion = _form(engine, '```json\n{"position": "Tea.", "confidence": "high", "tags": ["drink", 3]}\n```')
    assert opinion is not None and opinion.position == "Tea."
    assert opinion.confidence == 0.5
    assert opinion.tags == ["drink", "3"]
    assert engine.found == []


def test_a_single_quoted_reply_is_read(engine):
    opinion = _form(engine, "{'position': 'Coffee.', 'confidence': 0.9}")
    assert opinion is not None and opinion.position == "Coffee." and opinion.confidence == 0.9
