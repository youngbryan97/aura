import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel

from core.brain.llm import structured_llm as structured_module
from core.brain.llm.structured_llm import StructuredLLM
from tools.audit_degradation import analyze_file


class _TaskModel(BaseModel):
    action: str
    priority: int


class _MetadataRouter:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def generate_with_metadata(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def test_structured_llm_degradation_audit_is_clean():
    assert analyze_file(Path("core/brain/llm/structured_llm.py")) == []


@pytest.mark.asyncio
async def test_validation_telemetry_failure_does_not_block_schema_retry(monkeypatch):
    router = _MetadataRouter(
        {"text": '{"action": "test", "priority": "high"}'},
        {"text": '{"action": "test", "priority": 10}'},
    )
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *_args, **_kwargs: "",
    )
    monkeypatch.setattr(
        structured_module,
        "record_degraded_event",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("telemetry down")),
    )

    result = await StructuredLLM(_TaskModel, max_retries=2, llm_router=router).generate("Return a task.")

    assert result == _TaskModel(action="test", priority=10)
    assert len(router.calls) == 2
    assert router.calls[1]["prefer_tier"] == "primary"


@pytest.mark.asyncio
async def test_router_technical_failure_escalates_and_recovers(monkeypatch):
    router = _MetadataRouter(
        RuntimeError("local lane down"),
        {"text": '{"action": "recover", "priority": 2}'},
    )
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *_args, **_kwargs: "",
    )

    result = await StructuredLLM(_TaskModel, max_retries=2, llm_router=router).generate("Return a task.")

    assert result == _TaskModel(action="recover", priority=2)
    assert len(router.calls) == 2
    assert router.calls[1]["prefer_tier"] == "primary"


@pytest.mark.asyncio
async def test_background_policy_failure_defers_instead_of_running_router(monkeypatch):
    import core.runtime.background_policy as background_policy

    router = _MetadataRouter({"text": '{"action": "should-not-run", "priority": 1}'})

    policy_failures = []

    def fail_policy(*_args, **_kwargs):
        policy_failures.append((_args, _kwargs))
        raise RuntimeError("policy unavailable")

    monkeypatch.setattr(background_policy, "background_activity_reason", fail_policy)

    structured = StructuredLLM(_TaskModel, max_retries=2, llm_router=router)
    result = await structured.generate("Return a task.")

    assert result is None
    assert len(policy_failures) == 1
    assert structured.last_defer_reason == "background_policy_unavailable"
    assert router.calls == []


# ─────────────────── CP126: failing is not a request for priority ──────────
#
# Three criticals, all reachable from one loop:
#
# * after any failure `escalated_tier` became non-empty, and `is_background`
#   was computed as `not escalated_tier` — so a background task promoted
#   itself to a foreground lane by failing once, with no scheduler lease and
#   no user-facing admission decision. The same flag ALSO disabled the
#   background deferral check, so the failure both took the lane and removed
#   the gate that would have stopped it;
# * retries shared no absolute deadline, so `max_retries` bounded the count
#   and nothing bounded the request;
# * `json.loads` returns lists, strings, numbers and null; `model_class(**data)`
#   on any of those raises TypeError, which the validation block did not
#   catch, so a valid non-object escaped the documented repair path.


class _CountingRouter:
    """Records the lane every attempt asked for."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls: list[dict] = []

    async def generate_with_meta(self, prompt, **kwargs):
        self.calls.append(dict(kwargs))
        reply = self._replies.pop(0) if self._replies else "{}"
        return {"text": reply, "error_code": ""}

    async def generate(self, prompt, **kwargs):
        self.calls.append(dict(kwargs))
        return self._replies.pop(0) if self._replies else "{}"


def test_a_failed_attempt_does_not_promote_background_work_to_the_foreground():
    router = _CountingRouter(["not json at all", '{"action": "t", "priority": 1}'])
    structured = StructuredLLM(_TaskModel, max_retries=3, llm_router=router)

    asyncio.run(structured.generate("Return a task.", is_background=True))

    lanes = [call.get("is_background") for call in router.calls if "is_background" in call]
    assert lanes, "the router was never told which lane this was"
    assert all(lane is True for lane in lanes), (
        f"a background task promoted itself to the foreground by failing: {lanes}"
    )


def test_a_foreground_request_stays_foreground():
    router = _CountingRouter(['{"action": "t", "priority": 1}'])
    structured = StructuredLLM(_TaskModel, max_retries=2, llm_router=router)

    asyncio.run(structured.generate("Return a task.", is_background=False))

    lanes = [call.get("is_background") for call in router.calls if "is_background" in call]
    assert lanes and all(lane is False for lane in lanes)


def test_escalation_no_longer_disables_the_background_gate():
    """The gate is about the caller, not about how the last attempt went."""
    structured = StructuredLLM(_TaskModel, max_retries=2, llm_router=_CountingRouter([]))

    assert structured._background_defer_reason(is_background=False) == ""
    # A background caller still consults the policy — whatever it returns,
    # the point is that the question is asked at all.
    assert isinstance(structured._background_defer_reason(is_background=True), str)


def test_a_json_list_reaches_the_repair_path_instead_of_escaping():
    """`[1, 2, 3]` parses fine and then blew up as an uncaught TypeError."""
    router = _CountingRouter(["[1, 2, 3]", '{"action": "t", "priority": 1}'])
    structured = StructuredLLM(_TaskModel, max_retries=3, llm_router=router)

    result = asyncio.run(structured.generate("Return a task."))

    assert result is not None, (
        "a valid JSON array escaped the retry loop; the documented autonomous "
        "repair never ran"
    )
    assert len(router.calls) >= 2, "no repair attempt was made"


@pytest.mark.parametrize("root", ['"a string"', "42", "null", "true", "[]"])
def test_every_non_object_json_root_is_repairable(root):
    router = _CountingRouter([root, '{"action": "t", "priority": 1}'])
    structured = StructuredLLM(_TaskModel, max_retries=3, llm_router=router)

    assert asyncio.run(structured.generate("Return a task.")) is not None


def test_the_campaign_stops_at_its_deadline_not_at_its_retry_count():
    """max_retries bounded the count; nothing bounded the request."""
    import time

    class _SlowRouter:
        def __init__(self):
            self.calls = 0

        async def generate_with_meta(self, prompt, **kwargs):
            self.calls += 1
            await asyncio.sleep(0.05)
            return {"text": "not json", "error_code": ""}

        async def generate(self, prompt, **kwargs):
            self.calls += 1
            await asyncio.sleep(0.05)
            return "not json"

    router = _SlowRouter()
    structured = StructuredLLM(_TaskModel, max_retries=50, llm_router=router)

    started = time.monotonic()
    result = asyncio.run(structured.generate("Return a task.", deadline_s=0.15))
    elapsed = time.monotonic() - started

    assert result is None
    assert router.calls < 50, (
        f"all {router.calls} retries ran; the deadline bounded nothing"
    )
    assert elapsed < 2.0
    assert structured.last_defer_reason == "structured_generation_deadline_exhausted"


@pytest.mark.parametrize("bad", [None, "soon", float("nan"), float("inf")])
def test_an_unusable_deadline_falls_back_to_the_declared_default(bad):
    structured = StructuredLLM(_TaskModel, max_retries=1, llm_router=_CountingRouter([]))
    assert (
        structured._campaign_budget_seconds(bad)
        == StructuredLLM.DEFAULT_CAMPAIGN_BUDGET_S
    )


def test_an_explicit_zero_deadline_means_unbounded_not_the_default():
    """A caller that means unbounded must be able to say so."""
    structured = StructuredLLM(_TaskModel, max_retries=1, llm_router=_CountingRouter([]))
    assert structured._campaign_budget_seconds(0) == 0.0


@pytest.mark.anyio
async def test_the_decoder_is_asked_to_hold_a_json_object():
    """A caller that will parse JSON says so on the job, and the sampler
    cannot then produce prose. LIVE 2026-09-20: nine swarm shards failed three
    retries each on a brainstem that narrated before answering; the schema
    alone only pinned the temperature."""
    from pydantic import BaseModel

    from core.brain.llm.structured_llm import StructuredLLM

    class _Shape(BaseModel):
        value: int

    class _Router:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def generate_with_metadata(self, prompt, **kwargs):
            self.calls.append(kwargs)
            return {"text": '{"value": 3}', "error": ""}

    router = _Router()
    llm = StructuredLLM(_Shape, max_retries=1, llm_router=router)
    result = await llm.generate("give me a value", is_background=False)
    assert result is not None and result.value == 3
    assert router.calls and router.calls[0]["output_shape"] == "json_object"
    assert router.calls[0]["schema"]["title"] == "_Shape"
