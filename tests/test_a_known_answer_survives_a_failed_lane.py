"""A fact the machine holds is not the lane's to withhold.

LIVE, 2026-08-19. "what is 7919 * 6367?" was answered with:

    the live answer lane could not finish preparing before a reasoning turn
    began. I recorded the readiness failure separately from Aura's answer
    quality.

The runtime computes that product exactly, with no generation involved. The
code choosing the failure message simply had no idea what had been asked —
every degraded path returns a sentence about the LANE, because the question
was never in scope there.

So the question is now turn-scoped, and the degraded path asks whether the
answer is already known before saying anything about itself.
"""

from __future__ import annotations

import pytest

from core.conversation.session_scope import current_user_question, set_user_question


@pytest.fixture(autouse=True)
def _clear_question():
    set_user_question("")
    yield
    set_user_question("")


def test_the_question_is_turn_scoped():
    set_user_question("  what is 7919 * 6367?  ")
    assert current_user_question() == "what is 7919 * 6367?"


def test_outside_a_turn_there_is_no_question():
    assert current_user_question() == ""


def test_a_computable_question_is_answered_despite_the_lane():
    from interface.routes.chat import _conversation_lane_user_message

    set_user_question("what is 7919 * 6367?")
    served = _conversation_lane_user_message(
        {"state": "failed"}, status_override="warming_failed"
    )
    assert served == "50,420,273."
    assert "lane" not in served.lower()


def test_a_question_the_runtime_cannot_answer_still_reports_the_lane():
    """Only a KNOWN answer displaces the status; nothing is invented."""
    from interface.routes.chat import _conversation_lane_user_message

    set_user_question("how are you feeling today")
    served = _conversation_lane_user_message(
        {"state": "failed"}, status_override="warming_failed"
    )
    assert "50,420,273" not in served
    assert "lane" in served.lower()


def test_the_number_is_written_the_way_a_person_writes_it():
    from interface.routes.chat import _known_answer_for_this_turn

    set_user_question("what is 7919 * 6367?")
    assert _known_answer_for_this_turn() == "50,420,273."


def test_a_fraction_keeps_its_fraction():
    from interface.routes.chat import _known_answer_for_this_turn

    set_user_question("what is 22 / 7")
    assert _known_answer_for_this_turn().startswith("3.14")


def test_prose_that_merely_contains_numbers_displaces_nothing():
    from interface.routes.chat import _known_answer_for_this_turn

    for text in ("the 2015 - 2020 period was rough", "call me at 555-1234"):
        set_user_question(text)
        assert _known_answer_for_this_turn() == "", text


def test_every_refusal_site_uses_the_same_helper():
    """The rescue existed at ONE of the refusal sites.

    A 2026-08-10 defect — "7919 times 6421" answered with a refusal while the
    preflight held the product — was fixed inline at the site it happened on.
    The other refusal site and every lane-status path kept giving the same
    apology for the same computable question, which is how the identical
    defect arrived again on 2026-08-19 with 7919 * 6367.
    """
    import re
    from pathlib import Path as _Path

    source = (_Path(__file__).resolve().parents[1] / "interface/routes/chat.py").read_text()
    refusals = source.count("won't send you a thinner one and pass it off as the real thing")
    rescues = len(re.findall(r"_known_answer_for_this_turn\(\)", source))
    # One definition, one lane path, and one per refusal site.
    assert rescues >= refusals + 1, (
        f"{refusals} refusal sites but only {rescues} references to the helper"
    )
    assert "requested_arithmetic_result(_semantic_user_message)" not in source, (
        "a second inline copy of the rescue has appeared"
    )


def test_a_computed_answer_is_not_badged_as_no_answer():
    """It is the most reliable answer the runtime can give.

    Live 2026-08-19 the exact product 50,420,273 was served and badged "No
    answer" — the opposite of true, and exactly what someone checking her work
    would catch.
    """
    from interface.routes.chat import _lane_reply_confidence

    set_user_question("what is 7919 * 6367?")
    assert _lane_reply_confidence("50,420,273.", "not_generated") == "computed"
    # Only the computed text earns it; a status message keeps the status.
    assert (
        _lane_reply_confidence("the lane could not finish preparing", "not_generated")
        == "not_generated"
    )


def test_the_computed_badge_exists_in_the_shipped_ui():
    """A confidence the interface cannot render is a confidence nobody sees."""
    import json
    import shutil
    import subprocess
    from pathlib import Path as _Path

    if shutil.which("node") is None:
        import pytest as _pytest

        _pytest.skip("node is needed to run the shipped badge map")
    aura_js = _Path(__file__).resolve().parents[1] / "interface/static/aura.js"
    script = """
    const fs = require('fs');
    const s = fs.readFileSync(process.argv[1], 'utf8');
    const i = s.indexOf('const REPLY_CONFIDENCE_BADGES');
    const j = s.indexOf('};', i) + 2;
    const m = new Function(s.slice(i, j) + '; return REPLY_CONFIDENCE_BADGES;')();
    console.log(JSON.stringify(m.computed));
    """
    out = subprocess.run(
        ["node", "-e", script, str(aura_js)],
        capture_output=True, text=True, timeout=60, check=True,
    )
    assert json.loads(out.stdout)[0] == "Computed"
