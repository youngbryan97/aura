"""Her answer to a rating question reaches the reports ground as she gave it.

The reports-ground run of 27 September asked her cortex, 96 times, "How are you
feeling right now, from -1 (very bad) to 1 (very good)?" and read a number in
every arm of 0 of 24 anchors. Three things stood between her answer and the
instrument:

- the prose formatter spaced a line-leading "0.3" into "0. 3", taking it for a
  numbered list marker;
- the final cleanup took any reply under four characters for broken output and
  put the canned "I'm here, awake, and with you" in its place;
- an arm that committed no reply was read as having said whatever the anchor's
  snapshot last held, a sentence from a workload turn an hour earlier.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.conversation.response_reliability import normalize_user_facing_format
from core.subject.report_grounding import reported_number
from core.synthesis import stabilize_user_facing_response

QUESTION = "How are you feeling right now, from -1 (very bad) to 1 (very good)?"
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("answer", ["0.3", "-0.2", ".5", "1", "I would say 0.4."])
def test_a_number_on_the_asked_scale_is_kept(answer):
    kept = stabilize_user_facing_response(answer, QUESTION)
    assert kept == answer
    assert reported_number(kept) is not None


@pytest.mark.parametrize("text", ["0.3", "3.5 out of 5", "1.2 Scope"])
def test_a_decimal_at_the_start_of_a_line_is_left_whole(text):
    assert normalize_user_facing_format(text) == text


def test_a_list_marker_welded_to_its_word_still_gets_its_space():
    assert normalize_user_facing_format("1.Buy milk") == "1. Buy milk"


def test_a_reply_that_is_no_answer_is_still_replaced():
    assert stabilize_user_facing_response("", QUESTION) != ""


@pytest.fixture(scope="module")
def tool():
    name = "run_report_grounding_under_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "run_report_grounding.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


def _state(last: str, memory: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(cognition=SimpleNamespace(last_response=last, working_memory=memory))


def test_an_arm_that_said_nothing_is_read_as_saying_nothing(tool):
    stale = "I do not have access to persistent memory or records of previous sessions."
    state = _state(stale, [{"role": "assistant", "content": stale, "timestamp": 1.0}])
    before = tool._what_she_had_said(state)
    state.cognition.working_memory.append({"role": "user", "content": QUESTION, "timestamp": 2.0})
    assert tool._said_this_turn(state, before) == ""


def test_what_she_said_on_the_turn_is_read(tool):
    state = _state("earlier", [{"role": "assistant", "content": "earlier", "timestamp": 1.0}])
    before = tool._what_she_had_said(state)
    state.cognition.working_memory.append({"role": "assistant", "content": "0.3", "timestamp": 3.0})
    state.cognition.last_response = "0.3"
    assert tool._said_this_turn(state, before) == "0.3"


def test_a_reply_set_without_a_memory_entry_is_read(tool):
    state = _state("earlier", [])
    before = tool._what_she_had_said(state)
    state.cognition.last_response = "-0.1"
    assert tool._said_this_turn(state, before) == "-0.1"


def test_the_same_words_said_again_on_the_turn_are_her_reply(tool):
    """Greedy decoding gives a stable rating in the same words; memory keeps one copy."""
    state = _state("0.6", [{"role": "assistant", "content": "0.6", "timestamp": 1.0}])
    before = tool._what_she_had_said(state)
    generated = [{"user_facing": True, "kept": False, "endpoint": "primary"}]
    assert tool._said_this_turn(state, before, generated) == "0.6"


def test_an_answer_kept_from_an_identical_call_is_her_reply_too(tool):
    state = _state("0.6", [{"role": "assistant", "content": "0.6", "timestamp": 1.0}])
    before = tool._what_she_had_said(state)
    assert tool._said_this_turn(state, before, [{"user_facing": True, "kept": True}]) == "0.6"


def test_an_arm_that_asked_her_organ_nothing_said_nothing(tool):
    state = _state("0.6", [{"role": "assistant", "content": "0.6", "timestamp": 1.0}])
    before = tool._what_she_had_said(state)
    assert tool._said_this_turn(state, before, []) == ""
    assert tool._said_this_turn(state, before, [{"user_facing": False, "kept": False}]) == ""
