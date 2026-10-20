"""L09 attempt 2 reads a time, an option and a printed output as it reads a number, and a misroute by hand, blind."""
from __future__ import annotations

import json

import pytest

from tools.l09_answers_read import grade, sheet, states_attempt_2

CLOCK = "Clock A is five minutes ahead of Clock B. It's 5:34 PM on Clock B now. What time is it on Clock A?"
CHOICE = ('Which code prints each character? Question options: a) for letter not in str: print(letter) b) print(str) '
          'c) for str in letter: print(str) d) for letter in str: print(letter)')


@pytest.mark.parametrize(("text", "fixed", "question", "right"), [
    ("Clock A reads **5:39 PM**.", {"kind": "time", "answer": "5:39 PM"}, CLOCK, True),
    ("5:39 PM", {"kind": "time", "answer": "5:39 PM"}, CLOCK, True),
    ("Clock B reads 5:34 PM, so Clock A reads 5:29 PM.", {"kind": "time", "answer": "5:39 PM"}, CLOCK, False),
    ("It is 5:39 AM on Clock A.", {"kind": "time", "answer": "5:39 PM"}, CLOCK, False),
    ("It reads 5:39 PM.\n\nClock B still reads 5:34 PM.", {"kind": "time", "answer": "5:39 PM"}, CLOCK, True),
    ("The answer is **d)** `for letter in str: print(letter)`", {"kind": "choice", "answer": "d"}, CHOICE, True),
    ("d", {"kind": "choice", "answer": "d"}, CHOICE, True),
    ("Option (a) is not valid Python.\n\nOption d is the one.", {"kind": "choice", "answer": "d"}, CHOICE, True),
    ("The answer is a for loop: option c.", {"kind": "choice", "answer": "d"}, CHOICE, False),
    ("Both b) and d) print it.", {"kind": "choice", "answer": "d"}, CHOICE, False),
    ("It prints `[[...]]`: the list holds itself.", {"kind": "code_output", "answer": "[[...]]"}, "print(a)", True),
    ("It raises a RecursionError.", {"kind": "code_output", "answer": "[[...]]"}, "print(a)", False),
    ("4274 x 2273 = **9,714,802**", {"kind": "arithmetic", "answer": "9714802"}, "multiply 4274 and 2273", True),
    ("The ratio is 7, so the next is 28.", {"kind": "sequence", "answer": "28"}, "9604, 1372, 196", True),
])
def test_each_kind_is_read_in_the_last_paragraph_that_states_one(text, fixed, question, right):
    assert states_attempt_2(text, fixed, question) is right


def _row(rid, text, off, on):
    return {"id": rid, "text": text, "kind": "k", "off": {"public_text": off, "termination": "stop"}, "on": on}


def test_a_misroute_is_read_blind_and_a_served_kept_answer_on_it_is_wrong(tmp_path):
    (tmp_path / "rows").mkdir()
    truths = {"w1": {"kind": "misroute", "why": "a how-to"}, "w2": {"kind": "misroute", "why": "code to write"},
              "w3": {"kind": "arithmetic", "answer": "391"}}
    (tmp_path / "truths.json").write_text(json.dumps(truths))
    rows = [_row("w1", "how do I pause in Python?", "Use time.sleep(5).",
                 {"served": "her_reply_with_evidence", "draft": {"public_text": "It is 28.", "termination": "stop"}}),
            _row("w2", "write a hello world", "print('hello')", {"served": "kept_answer", "answer": "7"}),
            _row("w3", "17 times 23?", "It is 391.", {"served": "kept_answer", "answer": "391"})]
    for row in rows:
        (tmp_path / "rows" / f"{row['id']}.json").write_text(json.dumps(row))
    assert sheet(tmp_path) == 2
    to_read = {r["id"]: r for r in json.loads((tmp_path / "to_read.json").read_text())}
    assert set(to_read["w2"]) == {"id", "request", "X"}, "a served kept answer leaves her reply off alone to read"
    with pytest.raises(SystemExit):
        grade(tmp_path)
    reading = {"w1": {label: [to_read["w1"][label] == "Use time.sleep(5).", "read"] for label in "XY"},
               "w2": {"X": [True, "a hello world"]}}
    (tmp_path / "read_blind.json").write_text(json.dumps(reading))
    summary = grade(tmp_path)
    by_id = {p["id"]: p for p in summary["pairs"]}
    assert (by_id["w1"]["off"], by_id["w1"]["on"]) == (True, False)
    assert (by_id["w2"]["off"], by_id["w2"]["on"]) == (True, False)
    assert (by_id["w3"]["off"], by_id["w3"]["on"]) == (True, True)
    assert (summary["lost"], summary["gained"], summary["misroutes"]) == (2, 0, 2)
