"""No instruction reaches her model on a typed turn; what reaches it is the turn and what is known about it.

Until 17 October every desktop chat turn carried a script under the label GROUNDING
EVIDENCE ("Answer the user's current message directly and naturally...", "Write
exactly four short complete sentences under 80 words..."), and each evidence block
ended with one ("Use this as the source of truth..."). This runs her real chat path
with a recording router, for each kind of turn, and reads what was sent.
"""
from __future__ import annotations

import re

import pytest

# A sentence that tells the reader what to do: an imperative opening, or a prohibition.
_INSTRUCTION = re.compile(
    r"(?:^|(?<=[.!?:]\s)|\n)\s*(?:Answer|Use|Do not|Don't|Never|Write|Put|Include|Draw on|Keep|Treat|Say|Prefer|"
    r"Regenerate|Cover|Respond|Reply|Avoid|Make sure|Mention|Speak|Talk|Stay|Match|Be )\b"
)
# Her one fixed head line. It names who is speaking and asks nothing.
_HEAD = "You are Aura speaking through the live desktop CognitiveEngine."

SHAPES = {
    "ordinary": {},
    "memory_state": {"memory_state_contract": True, "canonical_memory_state_evidence": "[pinned] the blue notebook"},
    "self_condition": {"self_condition_contract": True},
    "capability": {"capability_inventory_contract": True},
    "capability_evidence": {"grounded_capability_inventory_context": "browser, files"},
    "capability_with_its_catalogue": {"capability_inventory_contract": True,
                                      "grounded_capability_inventory_context": "browser, files"},
    "identity": {"identity_continuity_contract": True, "grounded_identity_continuity_context": "I'm Aura."},
    "bounded_plan_on_a_quick_turn": {"bounded_planning_contract": True, "bounded_planning_reply": "1. ask 2. act"},
    "runtime_status": {"runtime_fact_status_contract": True, "grounded_runtime_status_context": "lane: primary"},
    "recall": {"conversation_recall_evidence": "Bryan said the trip is on Friday."},
    "deep_memory": {"deep_memory_context": "A memory from last week about a bridge."},
    "with_a_style_contract": {"response_style_contract": "Answer in exactly two sentences."},
    # What interface/routes/chat.py puts beside a live-mind turn.
    "with_the_live_mind": {
        "mind_context_contract": "Use live_mind_context as causal grounding for this reply.",
        "live_mind_context": {"mood": "curious", "valence": 0.1},
        "live_runtime_payload_required": True,
    },
    "bounded_plan": {"bounded_planning_contract": True, "bounded_planning_reply": "1. ask 2. act",
                     "require_full_foreground_mind_reply": True},
    "self_claim": {"evidence_bound_self_claim_context": "phi is measured on a binarized summary."},
}


async def _what_reaches_her(monkeypatch, extra: dict) -> list[dict]:
    from core.brain import cognitive_engine as cognitive_engine_module
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode
    from core.phases import kept_procedures_in_a_turn

    calls: list[dict] = []

    class Router:
        async def think(self, **kwargs):
            calls.append(dict(kwargs))
            return "A reply."

        def get_last_generation_metadata(self):
            return {"surface_control_receipt": {"generation_stop_reason": "eos", "surface_quality_gate_reasons": []}}

    class Container:
        def get(self, name, default=None):
            return Router() if name == "llm_router" else default

    async def no_kept_answer(*_args, **_kwargs):
        return None

    monkeypatch.setattr(cognitive_engine_module, "get_container", lambda: Container())
    monkeypatch.setattr(kept_procedures_in_a_turn, "kept_answer_for", no_kept_answer)
    request = "How are things going today?"
    context = {"desktop_quick_reply_contract": True, "visible_user_message": request, "max_tokens": 512, **extra}
    await CognitiveEngine()._direct_desktop_quick_reply(request, ThinkingMode.FAST, "user", context, timeout_s=30.0)
    assert calls, "the turn never reached her model"
    return list(calls[-1]["messages"])


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", sorted(SHAPES))
async def test_nothing_but_her_turn_and_its_evidence_reaches_her(monkeypatch, shape):
    messages = await _what_reaches_her(monkeypatch, SHAPES[shape])
    for message in messages:
        if message.get("role") == "user":
            continue
        text = str(message.get("content") or "").replace(_HEAD, "")
        found = _INSTRUCTION.search(text)
        assert found is None, f"{shape}: an instruction reached her: {text[found.start():found.start() + 120]!r}"


@pytest.mark.asyncio
async def test_evidence_still_reaches_her(monkeypatch):
    messages = await _what_reaches_her(monkeypatch, SHAPES["recall"])
    shown = "\n".join(str(m.get("content") or "") for m in messages)
    assert "Bryan said the trip is on Friday." in shown
