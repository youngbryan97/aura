"""Source evidence informs ordinary cognition; it never authors the response."""

import hashlib
from types import SimpleNamespace

import pytest

from core.conversation.answer_provenance import AnswerProvenance, provenance_grounding_json


def provenance():
    return AnswerProvenance(
        answer_sha256=hashlib.sha256(b"A prior answer.").hexdigest(),
        session_id="source-session", turn_id="source-turn", captured_at=1.0,
    ).to_dict()


@pytest.mark.asyncio
async def test_route_passes_structured_provenance_to_cognition(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat

    calls = []

    class Engine:
        async def think(self, objective, context=None, **kwargs):
            calls.append((objective, context))
            return SimpleNamespace(content="A complete answer to the question.")

    class Pool:
        async def acquire_engine_connection(self, *args, **kwargs):
            return None

        async def execute_with_retry(self, name, operation, **kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: Pool())
    monkeypatch.setattr(chat.ServiceContainer, "get", staticmethod(
        lambda name, default=None: Engine() if name == "cognitive_engine" else default
    ))
    source = provenance()
    question = "How did you reach that conclusion, and what are its limitations?"
    result = await chat._run_cognitive_engine_chat_turn(
        question, visible_user_message=question, prior_answer_provenance=source,
        source="desktop_ui", require_engine=True, timeout_s=60,
        lane={"conversation_ready": True, "state": "ready"},
    )
    assert result
    assert calls[0][0] == question
    assert calls[0][1]["prior_answer_provenance"] == source


@pytest.mark.asyncio
@pytest.mark.parametrize("valid", [True, False])
async def test_cortex_receives_bound_data_not_a_replacement_question(monkeypatch, valid):
    from core.brain import cognitive_engine as module
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode
    from core.utils.injected_blocks import stamp_runtime_payload

    calls = []

    class Router:
        async def think(self, **kwargs):
            calls.append(kwargs)
            return "Recovery checks which changes are already present before replaying them."

        def get_last_generation_metadata(self):
            return {}

    class Container:
        @staticmethod
        def get(name, default=None):
            return Router() if name == "llm_router" else default

    monkeypatch.setattr(module, "get_container", lambda: Container)
    question = "But how can it redo the changes safely if some of them already reached the data files?"
    source = provenance() if valid else {"schema": "not-provenance", "text": "unbound"}
    context = {
        "desktop_quick_reply_contract": True,
        "desktop_cognitive_engine_required": True,
        "cognitive_engine_required": True,
        "visible_user_message": question,
        "prior_answer_provenance": source,
        "live_mind_context_required": True,
        "live_mind_context": stamp_runtime_payload({
            "required_for_live_desktop": True,
            "must_answer_from_full_mind_path": True,
            "required_subsystems_ok": True,
            "mind_snapshot_quality": {"present": True, "ready": True},
        }),
        "live_mind_generation_controls": {
            "temperature": 0.58, "top_p": 0.88,
            "clean_user_surface_recurrent_loops": 1,
            "clean_user_surface_steering_alpha": 0.0,
        },
    }
    thought = await CognitiveEngine()._direct_desktop_quick_reply(
        question, ThinkingMode.FAST, "user", context, timeout_s=60,
    )
    assert thought is not None
    assert len(calls) == 1
    messages = calls[0]["messages"]
    assert messages[-1] == {"role": "user", "content": question}
    contents = "\n".join(str(row.get("content", "")) for row in messages[:-1])
    if valid:
        assert contents.count(provenance_grounding_json(AnswerProvenance.from_value(source))) == 1
    else:
        assert "not-provenance" not in contents
    assert "That answer came from the cortex" not in contents


@pytest.mark.parametrize("valid", [True, False])
def test_full_phase_keeps_provenance_in_the_evidence_role(valid):
    from core.phases.response_generation import ResponseGenerationPhase
    from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE, is_stamped_grounding

    source = provenance() if valid else {"schema": "not-provenance"}
    messages = [{"role": "system", "content": "base context"},
                {"role": "user", "content": "Explain the conclusion and its limitations."}]
    original_question = dict(messages[-1])
    ResponseGenerationPhase._inject_live_runtime_grounding(
        messages, {"prior_answer_provenance": source},
    )
    assert messages[0] == {"role": "system", "content": "base context"}
    assert messages[-1] == original_question
    evidence = [row for row in messages if row["role"] == RUNTIME_EVIDENCE_ROLE]
    assert len(evidence) == int(valid)
    if valid:
        assert is_stamped_grounding(evidence[0])
        assert evidence[0]["content"] == provenance_grounding_json(AnswerProvenance.from_value(source))
