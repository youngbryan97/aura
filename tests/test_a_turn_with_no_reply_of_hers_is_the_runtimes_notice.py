"""A turn that ends with no reply of hers carries the runtime's notice, not a sentence in her voice.

On 17 November at 07:33 Bryan asked "Anything you notice about how you are
running right now?" and got "I couldn't get to an answer I'd stand behind on
that one", the runtime's apology, sent as her reply and kept in her
conversation as something she had said; it went out seven times in the week
before. The lane's lines were the same kind, with her mood put in front:
"Ugh, the local answer path failed before producing a coherent reply. I'm
restarting it instead of pretending that was a real answer." Bryan's rule of
13 November is that the runtime's text never stands in her voice.

Now each of those exits returns an empty reply and a notice, the window draws
the notice as the runtime's, her voice says nothing, and the exchange keeps
the person's words with no reply of hers.
"""
from __future__ import annotations

import ast
import json
import re

import pytest

from core.conversation.runtime_notice import (
    NO_REPLY_FORMED,
    is_no_reply,
    no_reply_fields,
    notice_with_what_was_held,
)

#: A sentence in her first person, or a mood put in front of one.
_IN_HER_VOICE = re.compile(r"\b(?:I|I'm|I'd|I'll|me|my|myself)\b|^(?:Ugh|Mmm|Hmm)\b")


def test_the_fields_say_nothing_in_her_voice():
    fields = no_reply_fields("Her model is still loading.")
    assert fields["response"] == ""
    assert fields["notice"] == "Her model is still loading."
    assert fields["said_by"] == "runtime"
    assert fields["response_confidence"] == "failed"
    assert is_no_reply(fields)
    assert not is_no_reply({"response": "Hello.", "notice": "x"})


def test_what_the_runtime_held_goes_under_its_own_heading():
    said = notice_with_what_was_held(NO_REPLY_FORMED, "50,847,899")
    assert said.startswith(NO_REPLY_FORMED)
    assert said.endswith("What the runtime held for it:\n50,847,899")
    assert notice_with_what_was_held("", "") == NO_REPLY_FORMED


@pytest.mark.parametrize(
    "lane, kwargs",
    [
        ({"state": "failed", "last_failure_reason": "mlx_runtime_unavailable:boom"}, {}),
        ({"state": "failed", "last_failure_reason": "model_load_headroom"}, {}),
        ({"state": "warming"}, {"status_override": "warming_timeout"}),
        ({"state": "warming"}, {"status_override": "warming_failed"}),
        ({"state": "ready"}, {"timed_out": True}),
        ({"state": "recovering"}, {}),
        ({"state": "failed"}, {}),
        ({"state": "warming", "conversation_ready": False}, {}),
    ],
)
def test_the_lane_notices_are_the_runtimes(lane, kwargs):
    from interface.routes import chat

    said = chat._lane_status_message_body(lane, **kwargs)
    assert said and said[0].isupper(), said
    assert not _IN_HER_VOICE.search(said), said


@pytest.mark.parametrize(
    "lane",
    [
        {"last_failure_reason": "foreground_timeout"},
        {"warmup_in_flight": True},
        {"readiness_blockers": ["worker_not_alive"]},
        {"readiness_blockers": ["something_without_words"]},
        {"last_failure_reason": "no_reply"},
        {"last_failure_reason": "odd_condition"},
    ],
)
def test_why_there_is_no_answer_is_the_runtimes(lane):
    from interface.routes.chat_refusals import _why_there_is_no_answer

    said = _why_there_is_no_answer(lane)
    assert said
    assert not _IN_HER_VOICE.search(said), said


def test_the_error_notice_names_the_error_and_nothing_of_hers():
    from interface.routes.chat_reply_shaping import _grounded_chat_failure_reply

    said = _grounded_chat_failure_reply("TypeError")
    assert "TypeError" in said
    assert not _IN_HER_VOICE.search(said)


def test_an_empty_answer_at_the_boundary_becomes_a_notice():
    from fastapi.responses import JSONResponse

    from interface.routes.chat_delivery import _report_an_empty_answer_as_one

    response = JSONResponse({"response": "", "status": "ok"})
    payload = json.loads(bytes(response.body).decode())
    reported = _report_an_empty_answer_as_one(response, payload)
    out = json.loads(bytes(reported.body).decode())
    assert out["response"] == ""
    assert out["notice"] == NO_REPLY_FORMED
    assert out["response_confidence"] == "failed"
    assert out["status"] == "empty_answer_withheld:ok"


def test_a_notice_is_delivered_as_a_failed_turn():
    """The window draws a failed turn as the runtime's; a completed one as hers."""
    from interface.routes.chat_delivery import _chat_delivery_state_for_response

    payload = {"status": "timeout", **no_reply_fields("The time for this reply ran out before it finished.")}
    assert _chat_delivery_state_for_response(payload, 200).value == "failed"
    payload = {"status": "desktop_full_mind_contract_not_proven", **no_reply_fields("x")}
    assert _chat_delivery_state_for_response(payload, 200).value == "failed"


async def test_the_exchange_keeps_the_persons_words_and_no_reply_of_hers(monkeypatch):
    from core.conversation.delivered_history import _attested_pairs
    from core.utils.injected_blocks import GROUNDING_STAMP, _PAYLOAD_STAMP_KEY
    from interface.routes import chat_no_reply, chat_preflight

    async def persisted(**_kwargs):
        return "committed"

    monkeypatch.setattr(chat_preflight, "_persist_pending_conversation_user", persisted)
    log = chat_preflight._conversation_log
    before = list(log)
    try:
        log.append({"id": "x-unanswered", "user": "How are you running?", "aura": "", "status": "pending",
                    "user_persisted": True})
        state = await chat_no_reply.close_unanswered("x-unanswered", "How are you running?", notice="Her model is still loading.")
        entry = next(item for item in log if item.get("id") == "x-unanswered")
        assert state == "committed"
        assert entry["aura"] == ""
        assert entry["unanswered"] is True
        assert entry["status"] == "complete"
        assert entry["metadata"]["runtime_notice"] == "Her model is still loading."
        # Nothing of the notice reaches her as a turn of her own.
        assert _attested_pairs([dict(entry, **{_PAYLOAD_STAMP_KEY: GROUNDING_STAMP})]) == []
    finally:
        log[:] = before


async def test_a_turn_her_engine_could_not_serve_holds_no_words_of_hers(monkeypatch):
    from interface.routes import chat, chat_no_reply

    receipts: list[tuple[str, dict]] = []

    async def receipt(text, **kwargs):
        receipts.append((text, kwargs))

    async def silent_ladder(message, *, reason, budget_s=None):
        return ""

    monkeypatch.setattr(chat, "_known_answer_for_this_turn", lambda: "")
    monkeypatch.setattr(chat, "_what_the_tools_found", lambda: "")
    monkeypatch.setattr(chat._chat_conversation_repair, "_self_health_answer_or_empty", lambda _m: "")
    monkeypatch.setattr(chat, "_answer_from_fallback_ladder", silent_ladder)
    monkeypatch.setattr(chat, "_emit_chat_output_receipt", receipt)

    response = await chat_no_reply.what_a_turn_with_no_reply_returns(
        semantic_message="Anything you notice about how you are running right now?",
        original_message="Anything you notice about how you are running right now?",
        session_id="s", pending_exchange_id=None, status="desktop_cognitive_engine_unavailable",
        reason="desktop_cognitive_engine_required_no_reply", lane={"state": "failed"},
        reply_source="desktop_cognitive_engine_required_no_reply", is_benchmark=False, budget_s=1.0,
        live_turn_contract=lambda **kwargs: dict(kwargs),
    )
    out = json.loads(bytes(response.body).decode())
    assert out["response"] == ""
    assert out["notice"] == NO_REPLY_FORMED
    assert out["said_by"] == "runtime"
    assert receipts and receipts[0][1]["target"] == "notice"


async def test_what_the_runtime_held_is_shown_as_its_own(monkeypatch):
    from interface.routes import chat, chat_no_reply

    async def receipt(text, **kwargs):
        return None

    async def ladder(message, *, reason, budget_s=None):  # pragma: no cover - must not be asked
        raise AssertionError("the ladder is asked only when the runtime holds nothing")

    monkeypatch.setattr(chat, "_known_answer_for_this_turn", lambda: "7919 × 6421 = 50,847,899")
    monkeypatch.setattr(chat, "_answer_from_fallback_ladder", ladder)
    monkeypatch.setattr(chat, "_emit_chat_output_receipt", receipt)

    response = await chat_no_reply.what_a_turn_with_no_reply_returns(
        semantic_message="7919 times 6421?", original_message="7919 times 6421?", session_id="s",
        pending_exchange_id=None, status="desktop_cognitive_engine_unavailable", reason="r",
        lane={}, reply_source="r", is_benchmark=False, budget_s=1.0, live_turn_contract=lambda **kwargs: dict(kwargs),
    )
    out = json.loads(bytes(response.body).decode())
    assert out["response"] == ""
    assert out["notice"].endswith("What the runtime held for it:\n7919 × 6421 = 50,847,899")


async def test_what_the_smaller_model_wrote_is_its_answer(monkeypatch):
    from interface.routes import chat, chat_no_reply, chat_preflight

    async def receipt(text, **kwargs):
        return None

    async def ladder(message, *, reason, budget_s=None):
        return "Kobe's 81."

    async def logged(*args, **kwargs):
        return "committed"

    monkeypatch.setattr(chat, "_known_answer_for_this_turn", lambda: "")
    monkeypatch.setattr(chat, "_what_the_tools_found", lambda: "")
    monkeypatch.setattr(chat._chat_conversation_repair, "_self_health_answer_or_empty", lambda _m: "")
    monkeypatch.setattr(chat, "_answer_from_fallback_ladder", ladder)
    monkeypatch.setattr(chat, "_emit_chat_output_receipt", receipt)
    monkeypatch.setattr(chat_preflight, "_log_exchange", logged)

    response = await chat_no_reply.what_a_turn_with_no_reply_returns(
        semantic_message="Bam or Kobe?", original_message="Bam or Kobe?", session_id="s",
        pending_exchange_id=None, status="desktop_cognitive_engine_unavailable", reason="r",
        lane={}, reply_source="r", is_benchmark=False, budget_s=1.0, live_turn_contract=lambda **kwargs: dict(kwargs),
    )
    out = json.loads(bytes(response.body).decode())
    assert out["response"] == "Kobe's 81."
    assert out["response_confidence"] == "fallback"
    assert "notice" not in out


def test_no_exit_of_the_chat_route_writes_the_old_sentences():
    """The sentences are gone from every module that serves a turn."""
    from pathlib import Path

    retired = (
        "I couldn't get to an answer I'd stand behind",
        "I hit an error before a coherent answer formed",
        "I couldn't get my full attention onto that one",
        "I couldn't confirm that my answer matched",
        "That one took me longer than I'm willing",
        "I couldn't put together a better version",
        "Nothing came back from my own reasoning",
    )
    for path in sorted(Path("interface/routes").glob("chat*.py")):
        tree = ast.parse(path.read_text("utf-8"))
        documented = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body and isinstance(node.body[0], ast.Expr)
        }
        # What a module could send: its string literals, not its docstrings
        # and comments, which record the incidents these sentences caused.
        said = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in documented
        ]
        for sentence in retired:
            assert not any(sentence in text for text in said), f"{path}: {sentence}"
        assert "_mood_prefix" not in {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}, path
