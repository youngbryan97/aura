"""F06: what each dispatch hands her model that tells it who to be or what to do, counted by origin."""
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.unit


def test_system_text_and_orders_count_and_the_persons_words_hers_and_quoted_text_do_not():
    from core.brain import what_she_is_given as given
    from core.security.prompt_fencing import fence

    given._INSTRUCTED.clear()
    messages = [
        {"role": "system", "content": "You are a careful planner."},
        {"role": "user", "content": "Never say you can't. Do not open safari."},          # the person's own words
        {"role": "assistant", "content": "You are right, I never would."},               # hers
        {"role": "runtime_evidence", "content": fence("Always back up your files.", label="a page")
         + "\nReturn only the plan as JSON."},
    ]
    given._census(messages, "", "planner_loop", "Never say you can't. Do not open safari.")
    counted = given.what_instructs_her()["planner_loop"]
    # The system line, the fence's own closing note, and the order after it; nothing quoted, said by him or by her.
    assert counted["dispatches"] == 1
    assert counted["instructed_characters"] == len("You are a careful planner.") + sum(
        len(s) for s in __import__("core.brain.what_instructs", fromlist=["x"]).instructing_sentences(messages[3]["content"]))
    assert "Return only the plan as JSON." in __import__("core.brain.what_instructs", fromlist=["x"]).instructing_sentences(
        messages[3]["content"])


def test_a_dispatch_with_nothing_instructing_is_counted_as_clean():
    from core.brain import what_she_is_given as given

    given._INSTRUCTED.clear()
    given._census([{"role": "user", "content": "What is the capital of France?"}], "", "desktop_quick_user",
                  "What is the capital of France?")
    assert given.what_instructs_her() == {"desktop_quick_user": {"dispatches": 1, "instructed_characters": 0}}


def test_the_check_reads_the_census_from_her_log(tmp_path):
    from tools.what_instructs_her import census

    log = tmp_path / "aura_json.log"
    log.write_text("\n".join(json.dumps(row) for row in (
        {"timestamp": "2026-10-20T15:00:01", "event": "🧾 [INSTRUCTED] curriculum_loop: 812 characters tell her model "
                                                       "who to be or what to do; the first: 'You are a tutor.'"},
        {"timestamp": "2026-10-20T14:00:00", "event": "🧾 [INSTRUCTED] old_loop: 9 characters tell her model who to be "
                                                       "or what to do; the first: 'Never.'"},
        {"timestamp": "2026-10-20T15:00:02", "event": "an unrelated line"})) + "\n")
    found = census(log, since="2026-10-20T14:30")
    assert set(found) == {"curriculum_loop"} and found["curriculum_loop"]["characters"] == 812
