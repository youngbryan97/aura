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


async def test_a_computable_question_is_answered_despite_the_lane(monkeypatch):
    """The exact answer reaches the person, in the runtime's notice and not as her words.

    Since 18 November the lane's exits return the runtime's notice
    (core/conversation/runtime_notice.py), and what the runtime held goes
    under its own heading in it.
    """
    import json

    from interface.routes import chat

    async def receipt(text, **kwargs):
        return None

    monkeypatch.setattr(chat, "_emit_chat_output_receipt", receipt)
    set_user_question("what is 7919 * 6367?")
    response = await chat._the_lane_formed_no_reply(
        {"state": "failed"}, status="conversation_warming", status_override="warming_failed"
    )
    out = json.loads(bytes(response.body).decode())
    assert out["response"] == ""
    assert out["notice"].endswith("What the runtime held for it:\n50,420,273.")


async def test_a_question_the_runtime_cannot_answer_still_reports_the_lane(monkeypatch):
    """Only a KNOWN answer is added; nothing is invented."""
    import json

    from interface.routes import chat

    async def receipt(text, **kwargs):
        return None

    monkeypatch.setattr(chat, "_emit_chat_output_receipt", receipt)
    set_user_question("how are you feeling today")
    response = await chat._the_lane_formed_no_reply(
        {"state": "failed"}, status="conversation_warming", status_override="warming_failed"
    )
    out = json.loads(bytes(response.body).decode())
    assert "50,420,273" not in out["notice"]
    assert "could not finish preparing" in out["notice"]


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
    defect arrived again on 2026-08-19 with 7919 * 6367. Both refusal sites
    and the lane's exits now go through one reading each.
    """
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1] / "interface/routes"
    chat_source = (root / "chat.py").read_text()
    no_reply = (root / "chat_no_reply.py").read_text()
    assert chat_source.count("_chat_no_reply.what_a_turn_with_no_reply_returns(") == 2
    assert "_known_answer_for_this_turn()" in no_reply.split("def what_the_runtime_held", 1)[1]
    lane_exit = chat_source.split("async def _the_lane_formed_no_reply", 1)[1].split("\ndef ", 1)[0]
    assert "held=_known_answer_for_this_turn()" in lane_exit
    assert "requested_arithmetic_result(_semantic_user_message)" not in chat_source, (
        "a second inline copy of the rescue has appeared"
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
