"""CP126 contract tests for core/brain/llm/autonomous_brain_integration.py.

This module decides which model sees a user's prompt. CP126 found the decision
made from constants and caller assertions: cloud endpoints registered because a
key existed, an agentic endpoint chosen by dictionary order, tool maps sent with
no authority, confidence read off the branch, and a recovery path that
recomputed cloud permission from the RAW caller flag after the user opted out.

1a184861 25ec2c1d b4b991d8 0d25ad7d a8f1f66d 4771bf45 953459cc 8f06436a
c0fbd774 4bfedd29 ed70915c a170e6db af6a3b7d 69d1f269 37b3103e aa3235d2
480a31a0 78a30f5c 389d937a 53dd1ff6 f51792ca 3c7abca6 64ab3dab a5b9aa03
21cb172b.
"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from core.brain.llm import autonomous_brain_integration as module
from core.brain.llm.autonomous_brain_integration import (
    MAX_OBJECTIVE_CHARS,
    MAX_TURNS,
    MIN_TURNS,
    AutonomousCognitiveEngine,
    LLMEndpoint,
    LLMTier,
    ThinkRequest,
    bounded_deadline,
    bounded_priority,
    bounded_turns,
    normalize_objective,
    objective_ref,
)
from core.brain.llm.model_registry import (
    BRAINSTEM_ENDPOINT,
    DEEP_ENDPOINT,
    PRIMARY_ENDPOINT,
)


class _Router:
    def __init__(self, endpoints=None, healthy=True):
        self.endpoints = dict(endpoints or {})
        self.health_monitor = SimpleNamespace(
            is_healthy=lambda _n: healthy, peek_healthy=lambda _n: healthy
        )
        self.calls: list[dict] = []

    async def think(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return "routed answer"

    def register_endpoint(self, endpoint):
        self.endpoints[endpoint.name] = endpoint


class _Client:
    """Conversational only — deliberately has no ``think_and_act`` attribute,
    because that is exactly what the agentic allowlist looks for."""

    def __init__(self, result=None, delay=0.0):
        self.result = result
        self.delay = delay
        self.seen: dict | None = None

    async def think(self, prompt, **kwargs):
        return "client answer"


class _AgenticClient(_Client):
    async def think_and_act(self, objective, system_prompt, **kwargs):
        self.seen = dict(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.result if self.result is not None else {"content": "acted"}


def _endpoint(name, *, tier=LLMTier.PRIMARY, agentic=False, **kwargs):
    client = _AgenticClient(**kwargs) if agentic else _Client(**kwargs)
    return LLMEndpoint(name=name, tier=tier, model_name=f"{name}-model", client=client)


def _engine(router, **attrs):
    engine = AutonomousCognitiveEngine.__new__(AutonomousCognitiveEngine)
    engine.llm_router = router
    engine._last_think_error_time = 0.0

    async def _no_state():
        return None

    engine._get_live_state = _no_state
    for key, value in attrs.items():
        setattr(engine, key, value)
    return engine


@pytest.fixture(autouse=True)
def _no_guardian(monkeypatch):
    """Default: no stability guardian (safe mode, but not measured-unhealthy)."""
    from core.container import ServiceContainer

    monkeypatch.setattr(
        ServiceContainer, "get", classmethod(lambda cls, name, default=None: default)
    )


@pytest.fixture()
def cloud_off(monkeypatch):
    monkeypatch.setattr(module, "get_runtime_setting", lambda key, default=None: default)


@pytest.fixture()
def healthy_guardian(monkeypatch):
    """A guardian reporting health — leaves safe mode."""
    from core.container import ServiceContainer

    guardian = SimpleNamespace(get_health_summary=lambda: {"healthy": True})
    monkeypatch.setattr(
        ServiceContainer, "get",
        classmethod(
            lambda cls, name, default=None: guardian
            if name == "stability_guardian" else default
        ),
    )


def _run(coro):
    return asyncio.run(coro)


# --- c0fbd774 / 4bfedd29: prompts never reach logs or receipts ----------


def test_the_objective_reference_carries_no_content():
    ref = objective_ref("my password is hunter2")

    assert "hunter2" not in str(ref)
    assert ref["chars"] == len("my password is hunter2")
    assert len(ref["ref"]) == 16


def test_the_same_objective_gets_the_same_reference():
    assert objective_ref("a")["ref"] == objective_ref("a")["ref"]
    assert objective_ref("a")["ref"] != objective_ref("b")["ref"]


def test_the_objective_is_no_longer_logged_or_sliced_into_extras():
    source = inspect.getsource(module.AutonomousCognitiveEngine.think)
    assert "objective[:160]" not in source
    assert "objective[:100]" not in source
    assert 'logger.info("Mind pondering objective: %s", objective)' not in source


@pytest.mark.parametrize("bad", [None, 12345, {"a": 1}, ["x"], object()])
def test_a_non_string_objective_does_not_raise(bad):
    assert isinstance(normalize_objective(bad), str)


def test_an_oversized_objective_is_bounded_once():
    assert len(normalize_objective("x" * (MAX_OBJECTIVE_CHARS + 500))) == MAX_OBJECTIVE_CHARS


def test_a_non_string_objective_reaches_the_router_as_a_string(cloud_off):
    router = _Router({PRIMARY_ENDPOINT: _endpoint(PRIMARY_ENDPOINT)})

    result = _run(_engine(router).think(12345))

    assert result["status"] == "ok"
    assert isinstance(router.calls[0]["prompt"], str)


# --- 21cb172b: work bounds are policy-owned -----------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(-5, MIN_TURNS), (0, MIN_TURNS), (10_000, MAX_TURNS), ("lots", 5), (None, 5), (7, 7)],
)
def test_turns_are_bounded(value, expected):
    assert bounded_turns(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [(float("nan"), 1.0), (float("inf"), 1.0), (-2, 0.0), (5, 1.0), ("hi", 1.0), (0.4, 0.4)],
)
def test_priority_is_bounded(value, expected):
    assert bounded_priority(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, module.DEFAULT_AGENTIC_DEADLINE_S),
        (0, module.DEFAULT_AGENTIC_DEADLINE_S),
        (-1, module.DEFAULT_AGENTIC_DEADLINE_S),
        (float("inf"), module.DEFAULT_AGENTIC_DEADLINE_S),
        (10**9, module.MAX_AGENTIC_DEADLINE_S),
        (30.0, 30.0),
    ],
)
def test_deadlines_are_bounded(value, expected):
    assert bounded_deadline(value) == expected


# --- local-only endpoint authority -------------------------------------


def test_a_remote_compatibility_flag_cannot_enable_a_provider(cloud_off):
    router = _Router({PRIMARY_ENDPOINT: _endpoint(PRIMARY_ENDPOINT)})
    result = _run(_engine(router).think("hi", allow_cloud_fallback=True))

    assert result["remote_model_provider_available"] is False
    assert router.calls[0]["allow_cloud_fallback"] is False
    assert router.calls[0]["prefer_endpoint"] == PRIMARY_ENDPOINT


def test_endpoint_selection_contains_only_declared_local_lanes(cloud_off):
    router = _Router({
        PRIMARY_ENDPOINT: _endpoint(PRIMARY_ENDPOINT),
        BRAINSTEM_ENDPOINT: _endpoint(BRAINSTEM_ENDPOINT),
    })
    engine = _engine(router)
    request = _run(engine._build_request("hi", None, "sys", 5, 1.0, {}))

    assert engine._select_endpoints(request)["fast"].name == PRIMARY_ENDPOINT


def test_recovery_keeps_remote_provider_compatibility_disabled(cloud_off):
    class _Failing(_Router):
        async def think(self, prompt, **kwargs):
            self.calls.append({"prompt": prompt, **kwargs})
            if len(self.calls) == 1:
                raise RuntimeError("primary failed")
            return "recovered"

    router = _Failing({PRIMARY_ENDPOINT: _endpoint(PRIMARY_ENDPOINT)})

    _run(_engine(router).think("hi", allow_cloud_fallback=True))

    assert [call["allow_cloud_fallback"] for call in router.calls] == [False, False]


def test_tier_initialization_contains_no_remote_registration():
    source = inspect.getsource(module.AutonomousCognitiveEngine._init_tiers)
    assert "remote_model_provider_removed" not in source
    assert "api_key" not in source
    assert "is_local=False" not in source


# --- 389d937a / f51792ca: recovery keeps the request envelope -----------


def test_recovery_does_not_duplicate_keyword_arguments(cloud_off):
    """CP126 389d937a: explicit keywords PLUS the original kwargs expanded on
    top of them raised TypeError on any caller who set deep_handoff."""
    class _Failing(_Router):
        async def think(self, prompt, **kwargs):
            self.calls.append({"prompt": prompt, **kwargs})
            if len(self.calls) == 1:
                raise RuntimeError("primary failed")
            return "recovered"

    router = _Failing({PRIMARY_ENDPOINT: _endpoint(PRIMARY_ENDPOINT)})

    result = _run(
        _engine(router).think(
            "hi", deep_handoff=True, allow_cloud_fallback=True, is_background=False,
        )
    )

    assert result["content"] == "recovered"
    assert result["route"] == "recovery"


def test_recovery_retains_the_system_prompt_and_context(cloud_off):
    """CP126 f51792ca: recovery dropped both, producing output under different
    identity and task constraints than the request asked for."""
    class _Failing(_Router):
        async def think(self, prompt, **kwargs):
            self.calls.append({"prompt": prompt, **kwargs})
            if len(self.calls) == 1:
                raise RuntimeError("primary failed")
            return "recovered"

    router = _Failing({PRIMARY_ENDPOINT: _endpoint(PRIMARY_ENDPOINT)})

    _run(
        _engine(router).think(
            "hi", context={"constraint": "keep"}, system_prompt="BE AURA",
        )
    )

    recovery = router.calls[-1]
    assert recovery["system_prompt"] == "BE AURA"
    assert recovery["context"] == {"constraint": "keep"}


def test_recovery_no_longer_bypasses_the_race_guard():
    """Behaviour, not text: the explanatory comment necessarily names the flag
    it removed."""
    source = inspect.getsource(module.AutonomousCognitiveEngine._recover)
    code = "\n".join(
        line for line in source.splitlines()
        if not line.strip().startswith("#")
    ).split('"""')
    body = "".join(code[2:]) if len(code) > 2 else source
    assert "bypass_race" not in body


def test_bypass_race_from_a_caller_cannot_leak_into_a_router_call(cloud_off):
    router = _Router({PRIMARY_ENDPOINT: _endpoint(PRIMARY_ENDPOINT)})

    _run(_engine(router).think("hi", bypass_race=True))

    assert "bypass_race" not in router.calls[0]


# --- 69d1f269: context reaches every route ------------------------------


@pytest.mark.parametrize(
    "kwargs,endpoints",
    [
        ({}, {PRIMARY_ENDPOINT: _endpoint(PRIMARY_ENDPOINT)}),
        ({"is_background": True}, {BRAINSTEM_ENDPOINT: _endpoint(BRAINSTEM_ENDPOINT)}),
        ({}, {}),
    ],
)
def test_context_reaches_the_router_on_every_route(cloud_off, kwargs, endpoints):
    router = _Router(endpoints)

    _run(_engine(router).think("hi", context={"evidence": "required"}, **kwargs))

    assert router.calls[0]["context"] == {"evidence": "required"}


def test_context_reaches_the_agentic_route(cloud_off):
    endpoint = _endpoint(DEEP_ENDPOINT, tier=LLMTier.SECONDARY, agentic=True)
    router = _Router({DEEP_ENDPOINT: endpoint})

    _run(_engine(router).think("hi", context={"evidence": "x"}, tools={"t": 1}))

    assert endpoint.client.seen["context"] == {"evidence": "x"}


# --- 37b3103e: tools are not passed twice -------------------------------


def test_caller_supplied_tools_do_not_raise_a_duplicate_keyword(cloud_off):
    endpoint = _endpoint(DEEP_ENDPOINT, tier=LLMTier.SECONDARY, agentic=True)
    router = _Router({DEEP_ENDPOINT: endpoint})

    result = _run(_engine(router).think("hi", tools={"search": 1}))

    assert result["content"] == "acted"
    assert endpoint.client.seen["tools"] == {"search": 1}


def test_reserved_keys_never_survive_into_the_passthrough_set(cloud_off):
    engine = _engine(_Router())
    request = _run(
        engine._build_request(
            "hi", None, "sys", 5, 1.0,
            {
                "tools": {"a": 1}, "context": {"b": 2}, "system_prompt": "x",
                "priority": 0.5, "max_turns": 3, "deep_handoff": True,
                "prefer_endpoint": "E", "is_background": True, "keep_me": "yes",
            },
        )
    )

    assert request.passthrough == {"keep_me": "yes"}
    assert request.caller_tools == {"a": 1}


# --- aa3235d2: tool requirement is detected before conversational routing


def test_a_healthy_fast_endpoint_no_longer_hides_the_agentic_path(cloud_off):
    """CP126 aa3235d2: the fast route returned first, so in normal operation
    supplied tools and the skill router were simply ignored."""
    agentic = _endpoint(DEEP_ENDPOINT, tier=LLMTier.SECONDARY, agentic=True)
    router = _Router({
        PRIMARY_ENDPOINT: _endpoint(PRIMARY_ENDPOINT),
        DEEP_ENDPOINT: agentic,
    })

    result = _run(_engine(router).think("do a thing", tools={"search": 1}))

    assert result["route"] == "agentic"
    assert router.calls == []  # the conversational router was NOT used


def test_without_tools_the_fast_route_still_wins(cloud_off):
    router = _Router({
        PRIMARY_ENDPOINT: _endpoint(PRIMARY_ENDPOINT),
        DEEP_ENDPOINT: _endpoint(DEEP_ENDPOINT, tier=LLMTier.SECONDARY, agentic=True),
    })

    assert _run(_engine(router).think("chat"))["route"] == "fast"


def test_a_context_flag_can_require_tools(cloud_off):
    agentic = _endpoint(DEEP_ENDPOINT, tier=LLMTier.SECONDARY, agentic=True)
    router = _Router({PRIMARY_ENDPOINT: _endpoint(PRIMARY_ENDPOINT), DEEP_ENDPOINT: agentic})

    result = _run(_engine(router).think("x", context={"require_tools": True}))

    assert result["route"] == "agentic"


# --- af6a3b7d: agentic selection is an allowlist -------------------------


def test_an_unlisted_endpoint_is_never_chosen_for_tool_work(cloud_off):
    """CP126 af6a3b7d: dictionary order chose the agentic endpoint."""
    router = _Router({"Random-Thing": _endpoint("Random-Thing", agentic=True)})
    engine = _engine(router)
    request = _run(engine._build_request("hi", None, "s", 5, 1.0, {}))

    assert engine._select_endpoints(request)["agentic"] is None


def test_the_allowlist_prefers_the_primary_local_tier(cloud_off):
    router = _Router({
        DEEP_ENDPOINT: _endpoint(DEEP_ENDPOINT, tier=LLMTier.SECONDARY, agentic=True),
        PRIMARY_ENDPOINT: _endpoint(PRIMARY_ENDPOINT, agentic=True),
    })
    engine = _engine(router)
    request = _run(engine._build_request("hi", None, "s", 5, 1.0, {}))

    assert engine._select_endpoints(request)["agentic"].name == PRIMARY_ENDPOINT


# --- a170e6db: the background lane has its own allowlist -----------------


def test_a_caller_endpoint_cannot_ride_the_background_lane(cloud_off):
    """CP126 a170e6db: with no healthy background endpoint the branch used the
    CALLER's endpoint while still labelling the call tertiary/cloud-disabled."""
    router = _Router({PRIMARY_ENDPOINT: _endpoint(PRIMARY_ENDPOINT)})

    _run(_engine(router).think("bg", is_background=True, prefer_endpoint=PRIMARY_ENDPOINT))

    call = router.calls[0]
    assert call["prefer_endpoint"] is None
    assert call["prefer_tier"] == "tertiary"
    assert call["allow_cloud_fallback"] is False


def test_the_background_lane_uses_its_own_endpoints(cloud_off):
    router = _Router({BRAINSTEM_ENDPOINT: _endpoint(BRAINSTEM_ENDPOINT, tier=LLMTier.TERTIARY)})

    _run(_engine(router).think("bg", is_background=True))

    assert router.calls[0]["prefer_endpoint"] == BRAINSTEM_ENDPOINT


# --- 78a30f5c: no invented confidence ------------------------------------


@pytest.mark.parametrize(
    "kwargs,endpoints",
    [
        ({}, {PRIMARY_ENDPOINT: _endpoint(PRIMARY_ENDPOINT)}),
        ({"is_background": True}, {BRAINSTEM_ENDPOINT: _endpoint(BRAINSTEM_ENDPOINT)}),
        ({}, {}),
    ],
)
def test_no_route_reports_a_manufactured_confidence(cloud_off, kwargs, endpoints):
    result = _run(_engine(_Router(endpoints)).think("hi", **kwargs))

    assert "confidence" not in result
    assert result["verified"] is False
    assert "unmeasured" in result["confidence_basis"]


def test_the_route_and_endpoint_are_reported_instead(cloud_off):
    router = _Router({PRIMARY_ENDPOINT: _endpoint(PRIMARY_ENDPOINT)})

    result = _run(_engine(router).think("hi"))

    assert result["route"] == "fast"
    assert result["endpoint"] == PRIMARY_ENDPOINT
    assert result["status"] == "ok"


def test_an_empty_answer_is_reported_as_empty(cloud_off):
    class _Empty(_Router):
        async def think(self, prompt, **kwargs):
            return "   "

    assert _run(_engine(_Empty()).think("hi"))["status"] == "empty"


def test_the_branch_constants_are_gone():
    source = inspect.getsource(module.AutonomousCognitiveEngine)
    for constant in ('"confidence": 0.75', '"confidence": 1.0', '"confidence": 0.9'):
        assert constant not in source


# --- 480a31a0: tool authority is reported --------------------------------


def test_the_tool_map_provenance_travels_with_the_result(cloud_off):
    endpoint = _endpoint(DEEP_ENDPOINT, tier=LLMTier.SECONDARY, agentic=True)
    router = _Router({DEEP_ENDPOINT: endpoint})

    result = _run(_engine(router).think("x", tools={"a": 1}))

    authority = result["tool_authority"]
    assert authority["source"] == "caller_supplied"
    assert authority["scoped_authority"] is False
    assert "actuator boundary" in authority["note"]


def test_a_built_map_is_labelled_as_built(cloud_off, monkeypatch):
    monkeypatch.setattr(module, "build_agentic_tool_map", lambda **k: {"t": 1})
    endpoint = _endpoint(DEEP_ENDPOINT, tier=LLMTier.SECONDARY, agentic=True)
    router = _Router({DEEP_ENDPOINT: endpoint})

    result = _run(_engine(router).think("x", context={"require_tools": True}))

    assert result["tool_authority"]["source"] == "runtime_wiring"
    assert result["tool_authority"]["count"] == 1


# --- 3c7abca6: the agentic lane has a deadline ---------------------------


def test_a_hung_agentic_turn_does_not_hold_its_slot_forever(cloud_off):
    endpoint = _endpoint(DEEP_ENDPOINT, tier=LLMTier.SECONDARY, agentic=True, delay=2.0)
    router = _Router({DEEP_ENDPOINT: endpoint})

    result = _run(_engine(router).think("x", tools={"a": 1}, deadline_s=0.2))

    assert result["status"] == "deadline_exceeded"
    assert result["error_code"] == "brain.agentic_deadline"
    assert result["content"] == ""


def test_a_prompt_deadline_is_bounded_before_use(cloud_off):
    engine = _engine(_Router())
    request = _run(engine._build_request("hi", None, "s", 5, 1.0, {"deadline_s": -3}))

    assert request.deadline_s == module.DEFAULT_AGENTIC_DEADLINE_S


# --- 64ab3dab / a5b9aa03: defects are classified, details are not leaked --


def test_a_contract_defect_is_reported_as_an_internal_error(cloud_off, monkeypatch):
    class _Broken(_Router):
        async def think(self, prompt, **kwargs):
            raise AttributeError("someone renamed a field")

    result = _run(_engine(_Broken({PRIMARY_ENDPOINT: _endpoint(PRIMARY_ENDPOINT)})).think("hi"))

    assert result["status"] == "internal_error"
    assert result["error_code"] == "brain.internal_defect"
    assert result["error_class"] == "AttributeError"
    assert "someone renamed a field" not in str(result)


def test_an_endpoint_failure_still_recovers(cloud_off):
    class _Failing(_Router):
        async def think(self, prompt, **kwargs):
            self.calls.append({"prompt": prompt, **kwargs})
            if len(self.calls) == 1:
                raise OSError("endpoint gone")
            return "recovered"

    result = _run(_engine(_Failing({PRIMARY_ENDPOINT: _endpoint(PRIMARY_ENDPOINT)})).think("hi"))

    assert result["content"] == "recovered"


def test_the_final_reflex_response_leaks_no_exception_text(cloud_off):
    class _AlwaysFailing(_Router):
        async def think(self, prompt, **kwargs):
            raise OSError("/Users/secret/path: token sk-abc failed at api.example.com")

    result = _run(_engine(_AlwaysFailing()).think("hi"))

    assert result["error_code"] == "brain.all_routes_unavailable"
    assert result["error_class"] == "OSError"
    assert "/Users/secret" not in str(result)
    assert "sk-abc" not in str(result)


def test_defects_and_endpoint_failures_are_separate_tuples():
    assert AttributeError in module.PROGRAMMING_DEFECT_ERRORS
    assert AttributeError not in module.ENDPOINT_FAILURE_ERRORS
    assert OSError in module.ENDPOINT_FAILURE_ERRORS


# --- 1a184861: degradations are classified -------------------------------


def test_the_degradation_kinds_differ_in_classification():
    from core.runtime.errors import FallbackClassification

    kinds = module._DEGRADATION_KINDS
    assert kinds["local_omission"] == (FallbackClassification.SAFE_FALLBACK, False)
    # NOT GOVERNANCE_BYPASS — that classification raises CapabilityDenied
    # fail-closed, and a policy that correctly withholds a capability has not
    # been bypassed.
    assert kinds["policy"][0] is not FallbackClassification.GOVERNANCE_BYPASS
    assert kinds["policy"][1] is True
    assert kinds["routing"][1] is True
    assert kinds["internal_defect"][1] is True


def test_a_policy_degradation_requires_a_receipt(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        module, "record_degradation", lambda *a, **k: captured.update(k) or object()
    )

    module._record_brain_degradation(
        RuntimeError("x"), action="a", kind="policy",
    )

    assert captured["receipt_required"] is True
    assert captured["extra"]["degradation_kind"] == "policy"


def test_an_unknown_kind_falls_back_to_the_safe_default(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        module, "record_degradation", lambda *a, **k: captured.update(k) or object()
    )

    module._record_brain_degradation(RuntimeError("x"), action="a", kind="nonsense")

    assert captured["receipt_required"] is False


# --- 0d25ad7d: safe mode is a control, not dead policy -------------------


def test_safe_mode_is_fail_closed_when_health_cannot_be_established():
    """The contract from tests/test_runtime_health_truthfulness.py."""
    engine = AutonomousCognitiveEngine.__new__(AutonomousCognitiveEngine)

    assert engine._is_safe_mode() is True
    assert engine._stability_state()[1] == "guardian_unavailable"


def test_a_healthy_guardian_leaves_safe_mode(monkeypatch):
    from core.container import ServiceContainer

    guardian = SimpleNamespace(get_health_summary=lambda: {"healthy": True})
    monkeypatch.setattr(
        ServiceContainer, "get",
        classmethod(lambda cls, name, default=None: guardian if name == "stability_guardian" else default),
    )
    engine = AutonomousCognitiveEngine.__new__(AutonomousCognitiveEngine)

    assert engine._is_safe_mode() is False
    assert engine._stability_state()[1] == "guardian_healthy"


def test_safe_mode_keeps_remote_provider_compatibility_disabled():
    engine = _engine(_Router())
    request = _run(
        engine._build_request("hi", None, "s", 5, 1.0, {"allow_cloud_fallback": True})
    )

    assert request.safe_mode is True
    assert request.allow_cloud is False


def test_measured_ill_health_withdraws_tools_and_turns(monkeypatch, cloud_off):
    from core.container import ServiceContainer

    guardian = SimpleNamespace(get_health_summary=lambda: {"healthy": False})
    monkeypatch.setattr(
        ServiceContainer, "get",
        classmethod(lambda cls, name, default=None: guardian if name == "stability_guardian" else default),
    )
    agentic = _endpoint(DEEP_ENDPOINT, tier=LLMTier.SECONDARY, agentic=True)
    router = _Router({PRIMARY_ENDPOINT: _endpoint(PRIMARY_ENDPOINT), DEEP_ENDPOINT: agentic})

    result = _run(_engine(router).think("x", tools={"a": 1}, max_turns=20))

    assert result["route"] == "fast"  # tools withheld
    assert result["stability_basis"] == "guardian_unhealthy"


def test_an_absent_guardian_does_NOT_disable_tools(cloud_off):
    """Withdrawing agency for missing evidence would be a silent capability
    regression on every machine without a guardian registered."""
    agentic = _endpoint(DEEP_ENDPOINT, tier=LLMTier.SECONDARY, agentic=True)
    router = _Router({DEEP_ENDPOINT: agentic})

    assert _run(_engine(router).think("x", tools={"a": 1}))["route"] == "agentic"


def test_safe_mode_is_actually_consulted_by_think():
    source = inspect.getsource(module.AutonomousCognitiveEngine._build_request)
    assert "_stability_state()" in source


# --- 8f06436a / a8f1f66d: membership is not capacity ---------------------


def test_an_endpoint_with_no_client_is_not_usable():
    assert module._endpoint_is_usable(SimpleNamespace(client=None)) is False
    assert module._endpoint_is_usable(SimpleNamespace(client=object())) is False


def test_an_endpoint_with_a_callable_client_is_usable():
    assert module._endpoint_is_usable(_endpoint(PRIMARY_ENDPOINT)) is True


def test_the_sanity_check_requires_a_usable_endpoint():
    source = inspect.getsource(module.AutonomousCognitiveEngine._init_tiers)
    assert "if not self.llm_router.endpoints:" not in source
    assert "_endpoint_is_usable(ep)" in source


def test_the_tier_layout_does_not_claim_readiness():
    source = inspect.getsource(module.AutonomousCognitiveEngine._init_tiers)
    assert "NOT probed for load/generation" in source
    assert "(unusable)" in source


# --- 953459cc: the emergency tier checks what it loads -------------------


def test_the_emergency_tier_gates_on_the_path_it_loads():
    source = inspect.getsource(module.AutonomousCognitiveEngine._init_tiers)
    assert "fallback_model = brainstem_model_path or cortex_model_path" not in source
    assert "and fallback_path and FALLBACK_ENDPOINT not in" in source


# --- b4b991d8: tier initialization is guarded ----------------------------


def test_tier_initialization_holds_a_lock():
    source = inspect.getsource(module.AutonomousCognitiveEngine.__init__)
    assert "with self._tier_init_lock:" in source


def test_the_router_interface_is_validated_first():
    engine = AutonomousCognitiveEngine.__new__(AutonomousCognitiveEngine)
    engine.llm_router = SimpleNamespace()

    assert engine._router_interface_ok() is False

    engine.llm_router = _Router()
    assert engine._router_interface_ok() is True


# --- 25ec2c1d: one arbiter for every lane --------------------------------


def test_every_lane_has_its_own_limit():
    engine = AutonomousCognitiveEngine.__new__(AutonomousCognitiveEngine)

    for lane in ("foreground", "background", "deep", "agentic"):
        assert isinstance(engine._lane(lane), asyncio.Semaphore)


def test_the_same_lane_returns_the_same_semaphore():
    engine = AutonomousCognitiveEngine.__new__(AutonomousCognitiveEngine)

    assert engine._lane("foreground") is engine._lane("foreground")


def test_conversational_routes_are_arbitrated_too():
    source = inspect.getsource(module.AutonomousCognitiveEngine._router_call)
    assert "async with self._lane(lane):" in source


def test_the_request_envelope_is_immutable():
    request = ThinkRequest(
        objective="o", ref={}, context={}, system_prompt="", max_turns=1, priority=1.0,
        deadline_s=1.0, is_background=False, deep_handoff=False, allow_cloud=False,
        safe_mode=False, measured_unhealthy=False, stability_basis="x",
        requested_endpoint=None, caller_tools=None,
    )

    with pytest.raises((AttributeError, TypeError)):
        request.allow_cloud = True
