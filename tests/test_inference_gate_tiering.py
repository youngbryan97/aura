import asyncio
import contextlib
import importlib
import inspect
import logging
import os
import time
from types import SimpleNamespace

import pytest

from core.brain.inference_gate import InferenceGate
from core.container import ServiceContainer
from core.state.aura_state import AuraState
from core.utils.deadlines import get_deadline

_MISSING = object()


def _admission_snapshot(**overrides):
    """A memory-admission snapshot shaped the way production makes them.

    Fixtures used to hand the gate bare dicts like
    ``{"can_admit": True, "pressure_pct": 40.0}``. Production never produces
    that: every snapshot comes from ``_headroom_snapshot``, which stamps a
    schema and a measurement time so an INCOMPLETE or STALE receipt cannot
    authorize a ~20GB model load (CP126, "Missing admission fields are
    interpreted as permission").

    A fixture that omits the stamp is testing a shape that does not exist,
    and it was the reason the permissive reading looked safe.
    """
    import time as _time

    from core.brain.inference_gate import ADMISSION_SNAPSHOT_SCHEMA

    snapshot = {
        "tier": "primary",
        "pressure_pct": 40.0,
        "total_gb": 64.0,
        "available_gb": 32.0,
        "process_rss_gb": 0.0,
        "process_rss_limit_gb": 0.0,
        "max_pressure_pct": 84.0,
        "min_available_gb": 16.0,
        "can_admit": True,
        "reason": "",
        "measured": True,
        "schema": ADMISSION_SNAPSHOT_SCHEMA,
        "measured_at_monotonic": _time.monotonic(),
    }
    snapshot.update(overrides)
    return snapshot


@pytest.mark.asyncio
async def test_foreground_resource_context_uses_canonical_admission(monkeypatch):
    from core.resilience import resource_arbitrator

    calls = []

    class _Arbitrator:
        @contextlib.asynccontextmanager
        async def inference_context(self, **kwargs):
            calls.append(kwargs)
            yield

    monkeypatch.setattr(
        resource_arbitrator,
        "get_resource_arbitrator",
        lambda: _Arbitrator(),
    )
    gate = InferenceGate.__new__(InferenceGate)

    async with gate._resource_context(
        enabled=True,
        priority=True,
        worker="MLX-Cortex",
        timeout_s=42.0,
    ):
        pass

    assert calls == [
        {
            "priority": True,
            "worker": "MLX-Cortex",
            "timeout": 42.0,
        }
    ]


def test_build_messages_uses_canonical_state_without_mutating_working_memory(monkeypatch):
    state = AuraState.default()
    state.cognition.working_memory = [
        {"role": "user", "content": "canonical user turn"},
        {"role": "assistant", "content": "canonical assistant turn"},
    ]
    original_memory = list(state.cognition.working_memory)
    repo = SimpleNamespace(_current=state)
    original_get = ServiceContainer.get

    def _get(name, default=None):
        if name == "state_repository":
            return repo
        return original_get(name, default)

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(_get))

    gate = InferenceGate()
    messages = gate._build_messages(
        "new objective",
        "fallback system",
        [{"role": "user", "content": "incoming history"}],
    )

    assert state.cognition.working_memory == original_memory
    rendered = "\n".join(message["content"] for message in messages)
    assert "canonical user turn" in rendered
    assert "incoming history" in rendered
    assert messages[-1] == {"role": "user", "content": "new objective"}


def test_system_prompt_cache_tracks_live_state_revision(monkeypatch):
    state = AuraState.default()
    state.cognition.current_objective = "first objective"
    repo = SimpleNamespace(_current=state)
    original_get = ServiceContainer.get

    def _get(name, default=None):
        if name == "state_repository":
            return repo
        return original_get(name, default)

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(_get))
    gate = InferenceGate()

    first = gate._build_system_prompt()
    first_key = gate._identity_prompt_state_key
    state.cognition.current_objective = "second objective"
    second = gate._build_system_prompt()
    second_key = gate._identity_prompt_state_key

    assert first
    assert second
    assert first_key is not None
    assert second_key is not None
    # The contract is that the key CHANGES when the state the prompt is built
    # from changes — not that a particular field sits at a particular index.
    # Asserting the position pinned the old six-field tuple, which is what let
    # personality, goals, beliefs, memory and permissions change without
    # invalidating the cache.
    assert second_key != first_key


def test_the_identity_cache_key_covers_more_than_objective_and_affect(monkeypatch):
    """Every section ContextAssembler reads has to invalidate the cache."""
    state = AuraState.default()

    baseline = InferenceGate._identity_prompt_cache_key(state)
    assert baseline is not None

    state.motivation.__dict__["_cache_probe"] = "changed"
    assert InferenceGate._identity_prompt_cache_key(state) != baseline


def test_an_unreadable_state_yields_no_cache_key():
    """No key means do not reuse, which is the safe direction: a rebuilt
    prompt costs milliseconds and a stale one describes the wrong mind."""

    class _Hostile:
        def __getattr__(self, name):
            raise TypeError("unreadable")

    assert InferenceGate._identity_prompt_cache_key(_Hostile()) is None


class CallProbe:
    def __init__(self, return_value=None, side_effect=None, **attrs):
        self.return_value = return_value
        self.side_effect = side_effect
        self.calls = []
        for name, value in attrs.items():
            setattr(self, name, value)

    def __call__(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        if isinstance(self.side_effect, list):
            if not self.side_effect:
                raise AssertionError("call side effect sequence exhausted")
            item = self.side_effect.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        if callable(self.side_effect):
            return self.side_effect(*args, **kwargs)
        if isinstance(self.side_effect, BaseException):
            raise self.side_effect
        return self.return_value

    def assert_called_once(self):
        assert len(self.calls) == 1

    def assert_called_once_with(self, *args, **kwargs):
        assert len(self.calls) == 1
        assert self.calls[0] == {"args": args, "kwargs": kwargs}

    def assert_not_called(self):
        assert self.calls == []


class AsyncCallProbe(CallProbe):
    async def __call__(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        if isinstance(self.side_effect, list):
            if not self.side_effect:
                raise AssertionError("async call side effect sequence exhausted")
            item = self.side_effect.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        if callable(self.side_effect):
            result = self.side_effect(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result
        if isinstance(self.side_effect, BaseException):
            raise self.side_effect
        return self.return_value

    @property
    def await_args(self):
        if not self.calls:
            return SimpleNamespace(args=(), kwargs={})
        last = self.calls[-1]
        return SimpleNamespace(args=last["args"], kwargs=last["kwargs"])

    def assert_awaited(self):
        assert self.calls

    def assert_awaited_once(self):
        assert len(self.calls) == 1

    def assert_awaited_once_with(self, *args, **kwargs):
        assert len(self.calls) == 1
        assert self.calls[0] == {"args": args, "kwargs": kwargs}

    def assert_not_awaited(self):
        assert self.calls == []


class TaskProbe:
    def __init__(self, done=False):
        self._done = done
        self.cancel = CallProbe()
        self.done_callbacks = []

    def done(self):
        return self._done

    def add_done_callback(self, callback):
        self.done_callbacks.append(callback)

    def get_loop(self):
        return asyncio.get_running_loop()


class _replace:  # noqa: N801 - mirrors unittest.mock.patch for local test doubles
    def __init__(self, target, new=_MISSING, *, return_value=_MISSING, side_effect=_MISSING):
        self.target = target
        self.new = new
        self.return_value = return_value
        self.side_effect = side_effect
        self.owner = None
        self.attr = ""
        self.original = _MISSING
        self.replacement = _MISSING

    @staticmethod
    def _resolve(target):
        parts = target.split(".")
        for idx in range(len(parts) - 1, 0, -1):
            module_name = ".".join(parts[:idx])
            try:
                owner = importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue
            for part in parts[idx:-1]:
                owner = getattr(owner, part)
            return owner, parts[-1]
        raise ModuleNotFoundError(target)

    @classmethod
    def object(cls, owner, attr, new=_MISSING, *, return_value=_MISSING, side_effect=_MISSING):
        inst = cls("", new, return_value=return_value, side_effect=side_effect)
        inst.owner = owner
        inst.attr = attr
        return inst

    @classmethod
    @contextlib.contextmanager
    def dict(cls, mapping, values, clear=False):
        original = dict(mapping)
        if clear:
            mapping.clear()
        mapping.update(values)
        try:
            yield mapping
        finally:
            mapping.clear()
            mapping.update(original)

    def __enter__(self):
        if self.owner is None:
            self.owner, self.attr = self._resolve(self.target)
        owner_dict = getattr(self.owner, "__dict__", {})
        raw_original = owner_dict.get(self.attr, _MISSING)
        self.original = raw_original if raw_original is not _MISSING else getattr(self.owner, self.attr)
        callable_original = self.original
        if isinstance(callable_original, (staticmethod, classmethod)):
            callable_original = callable_original.__func__
        if self.new is not _MISSING:
            self.replacement = self.new
        else:
            rv = None if self.return_value is _MISSING else self.return_value
            se = None if self.side_effect is _MISSING else self.side_effect
            probe_cls = AsyncCallProbe if inspect.iscoroutinefunction(callable_original) else CallProbe
            self.replacement = probe_cls(return_value=rv, side_effect=se)
        install_value = self.replacement
        if isinstance(self.original, staticmethod) and not isinstance(install_value, staticmethod):
            install_value = staticmethod(install_value)
        elif isinstance(self.original, classmethod) and not isinstance(install_value, classmethod):
            install_value = classmethod(install_value)
        setattr(self.owner, self.attr, install_value)
        return self.replacement

    def __exit__(self, exc_type, exc, tb):
        setattr(self.owner, self.attr, self.original)
        return False


replace = _replace


class _FakeClient:
    def __init__(self, text: str):
        self.text = text
        self.generate_text_async = AsyncCallProbe(return_value=(True, text, {}))


class _RecordingClient:
    def __init__(self, text: str):
        self.text = text
        self.deadlines = []
        self.prompts = []
        self.kwargs = []

    async def generate_text_async(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        self.deadlines.append(kwargs.get("deadline"))
        return self.text


class _ReceiptRecordingClient(_RecordingClient):
    def __init__(self, text: str):
        super().__init__(text)
        self.receipt = {
            "enabled": True,
            "live_mind_controls_bound": True,
            "clean_user_surface_contract": True,
            "surface_validation_prompt_present": True,
            "surface_alpha_applied": 0.31,
            "surface_alpha_applied_ok": True,
            "recurrent_runtime_loops_applied": 2,
            "recurrent_runtime_loops_applied_ok": True,
            "surface_quality_gate_enabled": True,
            "surface_quality_gate_passed": True,
            "surface_quality_gate_attempts": 1,
            "surface_quality_gate_reasons": [],
            "applied": True,
        }

    def get_last_surface_control_receipt(self):
        return dict(self.receipt)


class _SequenceRecordingClient(_RecordingClient):
    def __init__(self, texts: list[str]):
        super().__init__(texts[-1] if texts else "")
        self.texts = list(texts)

    async def generate_text_async(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        self.deadlines.append(kwargs.get("deadline"))
        if self.texts:
            return self.texts.pop(0)
        return self.text

    def get_lane_status(self):
        now = time.time()
        return {
            "state": "ready",
            "last_error": "",
            "conversation_ready": True,
            "warmup_attempted": True,
            "warmup_in_flight": False,
            "last_ready_at": now,
            "last_progress_at": now,
            "last_visible_readiness_at": now,
            "last_user_facing_completed_at": 0.0,
        }


class _NoTextClient:
    def __init__(self):
        self.generate_text_async = AsyncCallProbe(return_value=(False, "", {}))


class _NoTextReadyClient(_NoTextClient):
    def get_lane_status(self):
        return {
            "state": "ready",
            "last_error": "",
            "conversation_ready": True,
            "warmup_attempted": True,
            "warmup_in_flight": False,
            "last_transition_at": 1.0,
            "last_ready_at": 1.0,
            "last_progress_at": 1.0,
        }

    def is_alive(self):
        return True


class _LaneWarmupClient:
    def __init__(self):
        self.warmup = AsyncCallProbe(side_effect=self._finish_warmup)
        self.state = "cold"
        self.last_error = ""
        self.visible_ready_at = 0.0

    async def _finish_warmup(self):
        self.state = "ready"
        self.last_error = ""
        self.visible_ready_at = time.time()

    def get_lane_status(self):
        return {
            "state": self.state,
            "last_error": self.last_error,
            "conversation_ready": self.state == "ready",
            "warmup_attempted": self.state != "cold",
            "warmup_in_flight": False,
            "last_transition_at": 1.0,
            "last_visible_readiness_at": self.visible_ready_at,
            "last_user_facing_completed_at": 0.0,
        }

    def is_alive(self):
        return self.state == "ready"

    def note_lane_recovering(self, reason):
        self.state = "recovering"
        self.last_error = str(reason or "")

    def note_lane_failed(self, reason):
        self.state = "failed"
        self.last_error = str(reason or "")


class _RecoverableFailedLaneClient(_LaneWarmupClient):
    def __init__(self):
        super().__init__()
        self.state = "failed"
        self.last_error = "mlx_runtime_unavailable:metal_device_enumeration_crash"
        self.refresh_runtime_availability = CallProbe(side_effect=self._refresh)
        self.is_alive = CallProbe(return_value=False)

    def _refresh(self, *, force_probe=False):
        self.state = "cold"
        self.last_error = ""
        return True


class _ColdRecordingLaneClient(_RecordingClient):
    def __init__(self, text: str):
        super().__init__(text)
        self.state = "cold"
        self.last_error = ""
        self.warmup = AsyncCallProbe(side_effect=self._finish_warmup)
        self.visible_ready_at = 0.0

    async def _finish_warmup(self):
        self.state = "ready"
        self.last_error = ""
        self.visible_ready_at = time.time()

    def get_lane_status(self):
        return {
            "state": self.state,
            "last_error": self.last_error,
            "conversation_ready": self.state == "ready",
            "warmup_attempted": self.state != "cold",
            "warmup_in_flight": False,
            "last_transition_at": 1.0,
            "last_ready_at": 1.0 if self.state == "ready" else 0.0,
            "last_progress_at": 1.0 if self.state == "ready" else 0.0,
            "last_visible_readiness_at": self.visible_ready_at,
            "last_user_facing_completed_at": 0.0,
        }


@pytest.mark.asyncio
async def test_inference_gate_passes_repairable_self_reflection_to_downstream_repair():
    gate = InferenceGate()
    bad_self_report = (
        "My self-prediction accuracy is 0.98. My memory texture drift is 0.02. "
        "My affect baseline is stable."
    )
    client = _FakeClient(bad_self_report)

    result = await gate._generate_with_client(
        client,
        "Aura, live-path check: what is actually on your mind right now?",
        "",
        [],
        get_deadline(10.0),
        "Cortex",
        messages=[
            {"role": "system", "content": "rich_context"},
            {"role": "user", "content": "Aura, live-path check: what is actually on your mind right now?"},
        ],
        origin="api",
        foreground_request=True,
    )

    assert result == bad_self_report
    client.generate_text_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_inference_gate_passes_repairable_reliability_draft_downstream():
    gate = InferenceGate()
    draft = (
        "The practical standard is that a foreground chat turn should stay live, "
        "finish as one coherent answer, and never collapse into retry chatter just "
        "because the first draft needs a repair pass."
    )
    client = _FakeClient(draft)

    result = await gate._generate_with_client(
        client,
        "Push back on me a little: if I demand that live chat never fails, what's the practical engineering version of that standard?",
        "",
        [],
        get_deadline(10.0),
        "Cortex",
        messages=[
            {"role": "system", "content": "rich_context"},
            {
                "role": "user",
                "content": "Push back on me a little: if I demand that live chat never fails, what's the practical engineering version of that standard?",
            },
        ],
        origin="api",
        foreground_request=True,
    )

    assert result == draft
    client.generate_text_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_inference_gate_passes_substantive_truncated_tail_downstream():
    gate = InferenceGate()
    draft = (
        "I would answer the user directly, preserve the current thread, and keep "
        "the live lane moving instead of detonating a long retry cascade,"
    )
    client = _FakeClient(draft)

    result = await gate._generate_with_client(
        client,
        "Explain how you keep continuity during a strained live chat turn.",
        "",
        [],
        get_deadline(10.0),
        "Cortex",
        messages=[
            {"role": "system", "content": "rich_context"},
            {
                "role": "user",
                "content": "Explain how you keep continuity during a strained live chat turn.",
            },
        ],
        origin="api",
        foreground_request=True,
    )

    assert result == draft
    client.generate_text_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_background_requests_stay_off_cortex():
    gate = InferenceGate()
    cortex = _FakeClient("cortex")
    brainstem = _FakeClient("brainstem")
    cpu = _FakeClient("cpu")
    gate._mlx_client = cortex
    gate._ensure_cortex_recovery = AsyncCallProbe()

    clients = {
        "/models/brainstem": brainstem,
        "/models/fallback": cpu,
    }

    def _fake_get_mlx_client(model_path=None, **kwargs):
        return clients[model_path]

    with replace.object(InferenceGate, "_background_local_deferral_reason", return_value=None):
        with replace("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
            with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
                with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                    result = await gate.generate(
                        "background reflection",
                        context={"prefer_tier": "primary", "origin": "system"},
                    )

    assert result == "brainstem"
    cortex.generate_text_async.assert_not_called()
    brainstem.generate_text_async.assert_awaited()
    gate._ensure_cortex_recovery.assert_not_awaited()


@pytest.mark.asyncio
async def test_background_requests_wait_while_cortex_quiet_window_is_active():
    gate = InferenceGate()
    gate._mlx_client = _LaneWarmupClient()
    gate._ensure_cortex_recovery = AsyncCallProbe()

    with replace.object(InferenceGate, "_foreground_quiet_window_active", return_value=True):
        with replace.object(
            InferenceGate,
            "get_conversation_status",
            return_value={
                "conversation_ready": False,
                "state": "warming",
                "warmup_in_flight": True,
            },
        ):
            result = await gate.generate(
                "background reflection",
                context={"prefer_tier": "primary", "origin": "system"},
            )

    assert result is None
    gate._ensure_cortex_recovery.assert_not_awaited()


@pytest.mark.asyncio
async def test_background_requests_wait_when_cortex_has_failed():
    gate = InferenceGate()
    failed_lane = _LaneWarmupClient()
    failed_lane.state = "failed"
    gate._mlx_client = failed_lane
    gate._ensure_cortex_recovery = AsyncCallProbe()

    result = await gate.generate(
        "background reflection",
        context={"prefer_tier": "primary", "origin": "system"},
    )

    assert result is None
    gate._ensure_cortex_recovery.assert_not_awaited()


@pytest.mark.asyncio
async def test_deep_handoff_uses_solver_then_returns_response():
    # The local deep solver is auto-disabled on <96GB hosts (memory-
    # class policy). Force-enable so the tier logic under test is
    # actually exercised regardless of the machine running the suite.
    os.environ["AURA_ENABLE_LOCAL_DEEP_SOLVER"] = "1"
    try:

        gate = InferenceGate()
        cortex = _FakeClient("cortex")
        solver = _FakeClient("solver")
        gate._mlx_client = cortex
        gate._restore_primary_after_deep_handoff = AsyncCallProbe()

        def _fake_get_mlx_client(model_path=None, **kwargs):
            if model_path == "/models/deep":
                return solver
            if model_path == "/models/active":
                return cortex
            raise AssertionError(f"Unexpected model path: {model_path}")

        # Fixed memory headroom so test doesn't depend on actual system RAM
        _low_pressure = _admission_snapshot(
            tier="secondary", pressure_pct=40.0, available_gb=32.0,
        )
        with replace("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
            with replace("core.brain.llm.model_registry.get_deep_model_path", return_value="/models/deep"):
                with replace("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                    with replace("core.brain.llm.model_registry.ACTIVE_MODEL", "ACTIVE"):
                        with replace.object(InferenceGate, "_headroom_snapshot", staticmethod(lambda *a, **kw: _low_pressure)):
                            result = await gate.generate(
                                "perform a flagship architecture deep dive",
                                context={"prefer_tier": "secondary", "deep_handoff": True},
                            )
        await asyncio.sleep(0)

        assert result == "solver"
        solver.generate_text_async.assert_awaited()
        cortex.generate_text_async.assert_not_called()
        gate._restore_primary_after_deep_handoff.assert_awaited_once()
    finally:
        os.environ.pop("AURA_ENABLE_LOCAL_DEEP_SOLVER", None)


@pytest.mark.asyncio
async def test_deep_handoff_failure_still_schedules_primary_restore():
    # The local deep solver is auto-disabled on <96GB hosts (memory-
    # class policy). Force-enable so the tier logic under test is
    # actually exercised regardless of the machine running the suite.
    os.environ["AURA_ENABLE_LOCAL_DEEP_SOLVER"] = "1"
    try:

        gate = InferenceGate()
        cortex = _NoTextClient()
        solver = _NoTextClient()
        reflex = _NoTextClient()
        gate._mlx_client = cortex
        gate._schedule_primary_restore_after_deep_handoff = CallProbe()

        def _fake_get_mlx_client(model_path=None, **kwargs):
            if model_path == "/models/deep":
                return solver
            if model_path == "/models/active":
                return cortex
            if model_path == "/models/fallback":
                return reflex
            raise AssertionError(f"Unexpected model path: {model_path}")

        _low_pressure_snapshot = {
            "tier": "secondary",
            "pressure_pct": 40.0,
            "total_gb": 128.0,
            "available_gb": 64.0,
            "max_pressure_pct": 84.0,
            "min_available_gb": 16.0,
            "can_admit": True,
            "reason": "",
        }
        with replace.object(
            gate,
            "_enforce_foreground_admission",
            new=AsyncCallProbe(
                return_value=_admission_snapshot(
                    pressure_pct=40.0, available_gb=28.0,
                )
            ),
        ):
            with replace("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
                with replace("core.brain.llm.model_registry.get_deep_model_path", return_value="/models/deep"):
                    with replace("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                        with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                            with replace("core.brain.llm.model_registry.ACTIVE_MODEL", "ACTIVE"):
                                with replace.object(
                                    InferenceGate,
                                    "_headroom_snapshot",
                                    staticmethod(lambda *a, **kw: dict(_low_pressure_snapshot)),
                                ):
                                    await gate.generate(
                                        "perform a flagship architecture deep dive",
                                        context={"origin": "user", "prefer_tier": "secondary", "deep_handoff": True},
                                    )

        gate._schedule_primary_restore_after_deep_handoff.assert_called_once()
    finally:
        os.environ.pop("AURA_ENABLE_LOCAL_DEEP_SOLVER", None)


@pytest.mark.asyncio
async def test_user_facing_primary_uses_conversational_budget_and_chatml():
    gate = InferenceGate()
    cortex = _RecordingClient("hello")
    gate._mlx_client = cortex

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "Say hi.",
                    context={"origin": "user", "prefer_tier": "primary", "history": []},
                )

    assert result == "hello"
    assert cortex.deadlines
    expected_total = InferenceGate._default_timeout_for_request(
        "user",
        "primary",
        deep_handoff=False,
        is_background=False,
    )
    expected_primary, _ = InferenceGate._split_attempt_timeouts(expected_total, "primary")
    assert cortex.deadlines[0]._timeout == expected_primary
    # Token budget may be scaled by resource_stakes multiplier at runtime,
    # so check it's within the expected range rather than an exact match.
    expected_tokens = InferenceGate._default_max_tokens_for_request(
        "user",
        "primary",
        deep_handoff=False,
        is_background=False,
    )
    actual_tokens = cortex.kwargs[0]["max_tokens"]
    assert 384 <= actual_tokens <= expected_tokens
    assert cortex.kwargs[0]["clean_user_surface_contract"] is True
    assert cortex.kwargs[0]["clean_user_surface_steering_alpha"] == 0.25
    assert cortex.prompts[0].startswith("<|im_start|>")
    assert "<|im_start|>assistant" in cortex.prompts[0]
    assert "<|SYSTEM|>" not in cortex.prompts[0]


@pytest.mark.asyncio
async def test_desktop_cortex_only_contract_does_not_reserve_for_forbidden_lane():
    gate = InferenceGate()
    cortex = _RecordingClient(
        "Design: the foreground owner keeps the only permitted Cortex lane for the "
        "whole bounded attempt. Verification: force a Cortex deadline and confirm "
        "that no forbidden lower-lane generation starts."
    )
    gate._mlx_client = cortex

    result = await gate.generate(
        "Explain the design and verify the failure case.",
        context={
            "origin": "user",
            "prefer_tier": "primary",
            "history": [],
            "cognitive_engine_required": True,
            "foreground_request": True,
        },
        timeout=200.0,
    )

    assert result.startswith("Design: the foreground owner")
    assert cortex.deadlines[0]._timeout == 196.0


@pytest.mark.asyncio
async def test_live_self_process_prebuilt_prompt_is_compacted_and_live_grounded(monkeypatch):
    gate = InferenceGate()
    cortex_reply = (
        "I am attending to Bryan's concern that the live desktop lane can drift into raw assistant mode "
        "or stall under oversized prompts. The remembered concern that should change my next decision is "
        "the repeated timeout-and-repair loop, so I should keep the live mind payload bounded while preserving "
        "recent context. What I want to do next is answer from that bounded live context, then verify the UI reply "
        "and terminal logs before treating the fix as real."
    )
    cortex = _RecordingClient(cortex_reply)
    gate._mlx_client = cortex

    async def _compact_live_context(_self, _prompt, _origin, **_kwargs):
        return "## LIVE TONE\nMood: focused\nTone: direct\n## UNITY\nLevel: integrated | Unity: 0.91"

    async def _full_live_context(*_args, **_kwargs):
        if _args or _kwargs:
            raise AssertionError("self-process desktop speech should not build the full 90k live context")
        return ""

    monkeypatch.setattr(InferenceGate, "_build_compact_living_mind_context", _compact_live_context)
    monkeypatch.setattr(InferenceGate, "_build_living_mind_context", _full_live_context)
    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")

    oversized_system = "SYSTEM CONTRACT\n" + ("keep Aura stateful and non-generic. " * 900)
    oversized_turn = "prior live desktop conversation context. " * 500
    user_prompt = (
        "Quick live-path check. Don't give me a health card or telemetry list. "
        "In ordinary speech, answer from your actual current context: what are "
        "you attending to from Bryan's recent messages, what remembered concern "
        "should change your next decision, and what do you want to do next?"
    )
    messages = [{"role": "system", "content": oversized_system}]
    for idx in range(12):
        messages.append({"role": "user", "content": f"user {idx}: {oversized_turn}"})
        messages.append({"role": "assistant", "content": f"aura {idx}: {oversized_turn}"})
    messages.append({"role": "user", "content": user_prompt})

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    user_prompt,
                    context={
                        "origin": "user",
                        "prefer_tier": "primary",
                        "foreground_request": True,
                        "protected_foreground_lane": True,
                        "desktop_cognitive_engine_required": True,
                        "live_runtime_payload_required": True,
                        "allow_mesh_cognition": False,
                        "messages": messages,
                    },
                )

    assert result == cortex_reply
    assert len(cortex.prompts) == 1
    rendered = cortex.prompts[0]
    assert len(rendered) < 12000
    assert "## LIVE TONE" in rendered
    assert "Mood: focused" in rendered
    assert user_prompt in rendered
    assert rendered.count("prior live desktop conversation context") < 80


@pytest.mark.asyncio
async def test_prebuilt_desktop_contract_uses_canonical_visible_text_for_surface_validation():
    from core.conversation.response_reliability import is_operational_status_turn

    gate = InferenceGate()
    cortex = _RecordingClient("I will remember that the blue lantern is under the desk.")
    gate._mlx_client = cortex
    visible = (
        "Remember this note for later in this conversation: "
        "the blue lantern is under the desk."
    )
    contract_wrapped = (
        f"{visible}\n\n[LIVE DESKTOP FULL-MIND CONTRACT]\n"
        "- Runtime path contract: governed tool and model lane status must remain available.\n"
        "[END LIVE DESKTOP FULL-MIND CONTRACT]"
    )
    assert is_operational_status_turn(visible) is False
    assert is_operational_status_turn(contract_wrapped) is True

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    contract_wrapped,
                    context={
                        "origin": "desktop_quick_user",
                        "prefer_tier": "primary",
                        "foreground_request": True,
                        "protected_foreground_lane": True,
                        "allow_mesh_cognition": False,
                        "visible_user_message": visible,
                        "messages": [
                            {"role": "system", "content": "Speak through Aura's live mind."},
                            {"role": "user", "content": contract_wrapped},
                        ],
                    },
                )

    assert result == "I will remember that the blue lantern is under the desk."
    assert cortex.kwargs[0]["user_surface_validation_prompt"] == visible
    assert "Runtime path contract" not in cortex.kwargs[0]["user_surface_validation_prompt"]


@pytest.mark.asyncio
async def test_user_facing_primary_restores_foreground_token_floor(monkeypatch):
    gate = InferenceGate()
    cortex = _RecordingClient("Live chat kept enough room to answer coherently.")
    gate._mlx_client = cortex
    monkeypatch.setenv("AURA_FOREGROUND_CHAT_MIN_TOKENS", "1024")

    with replace.object(InferenceGate, "_default_max_tokens_for_request", return_value=512):
        with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
            with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
                with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                    result = await gate.generate(
                        "Stay with the thread and answer in a real conversational paragraph.",
                        context={"origin": "user", "prefer_tier": "primary", "history": []},
                    )

    assert result == "Live chat kept enough room to answer coherently."
    assert cortex.kwargs[0]["max_tokens"] >= 1024


@pytest.mark.asyncio
async def test_explicit_output_contract_overrides_foreground_floor_and_is_receipted(monkeypatch):
    gate = InferenceGate()
    cortex = _ReceiptRecordingClient("Latency sample 3 completed.")
    cortex.receipt.update(
        {
            "generation_max_tokens": 48,
            "generated_tokens": 7,
            "instruction_shape_repair_applied": False,
        }
    )
    gate._mlx_client = cortex
    monkeypatch.setenv("AURA_FOREGROUND_CHAT_MIN_TOKENS", "1024")

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "Latency sample 3: answer in one short sentence that includes the sample number.",
                    context={
                        "origin": "user",
                        "prefer_tier": "primary",
                        "history": [],
                        "allow_mesh_cognition": False,
                    },
                )

    assert result == "Latency sample 3 completed."
    generation = cortex.kwargs[0]
    assert generation["max_tokens"] == 48
    assert generation["semantic_output_token_cap"] == 32
    assert generation["hard_output_token_ceiling"] == 48
    assert generation["requested_output_contract"]["kind"] == "sentence_count"
    metadata = gate.get_last_generation_metadata()
    assert metadata["requested_max_tokens"] == 48
    assert metadata["actual_max_tokens"] == 48
    assert metadata["generated_tokens"] == 7
    assert metadata["deterministic_repair_applied"] is False


@pytest.mark.asyncio
async def test_prebuilt_short_output_contract_uses_contract_prompt_profile(monkeypatch):
    gate = InferenceGate()
    cortex = _RecordingClient("yes")
    gate._mlx_client = cortex
    prompt = 'Reply exactly: "yes"'
    messages = [
        {
            "role": "system",
            "content": (
                "## INTRINSIC IDENTITY ANCHOR (IMMUTABLE)\n"
                + ("identity and runtime state " * 800)
                + "\n[LIVE MIND CONTEXT]\n"
                + '{"must_answer_from_full_mind_path": true}\n'
                + "[END LIVE MIND CONTEXT]\n"
            ),
        },
        {"role": "user", "content": prompt},
    ]
    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    prompt,
                    context={
                        "origin": "desktop_quick_user",
                        "prefer_tier": "primary",
                        "foreground_request": True,
                        "protected_foreground_lane": True,
                        "desktop_quick_reply_contract": True,
                        "desktop_cognitive_engine_required": True,
                        "live_mind_context_required": True,
                        "live_runtime_payload_required": True,
                        "allow_mesh_cognition": False,
                        "visible_user_message": prompt,
                        "messages": messages,
                    },
                )

    assert result == "yes"
    rendered_messages = cortex.kwargs[0]["messages"]
    assert sum(len(message["content"]) for message in rendered_messages) <= 2_800
    assert rendered_messages[0]["content"].startswith(
        "## CONTRACT-BOUNDED LIVE CORTEX TURN"
    )
    assert "[LIVE MIND CONTEXT]" in rendered_messages[0]["content"]
    assert rendered_messages[-1] == {"role": "user", "content": prompt}


@pytest.mark.asyncio
async def test_parent_output_repair_is_distinguished_from_model_native_compliance(monkeypatch):
    gate = InferenceGate()
    cortex = _ReceiptRecordingClient(
        "Latency sample 4 is present. This extra sentence should be removed."
    )
    cortex.receipt.update(
        {
            "generation_max_tokens": 48,
            "generated_tokens": 14,
            "instruction_shape_repair_applied": False,
        }
    )
    gate._mlx_client = cortex

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "Latency sample 4: answer in one short sentence that includes the sample number.",
                    context={
                        "origin": "user",
                        "prefer_tier": "primary",
                        "history": [],
                        "allow_mesh_cognition": False,
                    },
                )

    assert result == "Latency sample 4 is present."
    metadata = gate.get_last_generation_metadata()
    assert metadata["surface_control_receipt"][
        "instruction_shape_repair_applied"
    ] is False
    assert metadata["deterministic_repair_applied"] is True
    assert metadata["post_generation_repair_applied"] is True


@pytest.mark.asyncio
async def test_user_facing_primary_retry_uses_clean_cortex_repair_lane(monkeypatch):
    gate = InferenceGate()
    good_reply = (
        "The bounded objective should guide Aura, use governed tools with a "
        "receipt and trace, stop when policy or evidence fails, and treat that "
        "as operational evidence rather than proof of literal personhood."
    )
    cortex = _SequenceRecordingClient(["ok", good_reply])
    brainstem = _RecordingClient("brainstem fallback should not be needed")
    gate._mlx_client = cortex
    monkeypatch.setattr(asyncio, "sleep", AsyncCallProbe())

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=brainstem):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "Answer this live operator check: what objective should Aura pursue and when should she stop?",
                    context={"origin": "api", "prefer_tier": "primary", "history": []},
                )

    assert result == good_reply
    assert len(cortex.kwargs) == 2
    assert cortex.kwargs[0]["clean_user_surface_contract"] is True
    assert cortex.kwargs[0]["clean_user_surface_steering_alpha"] <= 0.35
    retry_kwargs = cortex.kwargs[1]
    assert retry_kwargs["clean_user_surface_contract"] is True
    assert retry_kwargs["disable_prompt_cache"] is True
    assert retry_kwargs["clear_prompt_cache"] is True
    assert retry_kwargs["skip_runtime_payload"] is True
    assert retry_kwargs["top_p"] <= 0.85
    assert retry_kwargs["repetition_context_size"] >= 96
    assert "previous draft" in cortex.prompts[1].lower()
    assert retry_kwargs["messages"][0]["role"] == "system"
    assert retry_kwargs["messages"][-1]["role"] == "user"
    assert brainstem.kwargs == []


@pytest.mark.asyncio
async def test_health_probe_primary_lane_uses_adaptive_recurrent_depth_clamp(monkeypatch):
    gate = InferenceGate()
    cortex = _RecordingClient("local lane ready")
    gate._mlx_client = cortex

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_RecordingClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "Reply briefly that the requested local lane is ready.",
                    context={
                        "origin": "internal",
                        "purpose": "proof_model_lane_probe",
                        "prefer_tier": "primary",
                        "health_probe": True,
                        "foreground_request": True,
                        "max_tokens": 24,
                    },
                )

    assert result == "local lane ready"
    assert cortex.kwargs
    probe_kwargs = cortex.kwargs[0]
    assert probe_kwargs["max_tokens"] <= 64
    assert probe_kwargs["clean_user_surface_contract"] is True
    assert probe_kwargs["clean_user_surface_recurrent_loops"] == 1
    assert probe_kwargs["clean_user_surface_steering_alpha"] == 0.25


def test_adaptive_max_tokens_expands_budget_for_compound_prompt():
    prompt = (
        "If you refuse to give receipts or operational details, say exactly why. "
        "Then give one safe example only: the most recent non-private action you took "
        "that has a log line or event ID."
    )
    adapted = InferenceGate._adaptive_max_tokens_for_prompt(
        prompt,
        base_tokens=768,
        origin="user",
        requested_tier="primary",
        is_background=False,
    )

    assert adapted >= 1024


def test_short_foreground_prompt_uses_low_latency_compute_profile(monkeypatch):
    monkeypatch.delenv("AURA_FOREGROUND_CHAT_SIMPLE_MAX_TOKENS", raising=False)

    floor, cap, loops = InferenceGate._foreground_compute_profile(
        "Invent a tiny discipline called glass arithmetic. Give it two rules and one example."
    )
    adapted = InferenceGate._adaptive_max_tokens_for_prompt(
        "Invent a tiny discipline called glass arithmetic. Give it two rules and one example.",
        base_tokens=4096,
        origin="user",
        requested_tier="primary",
        is_background=False,
    )

    assert 256 <= floor <= 384
    assert cap == 512
    assert adapted == cap
    assert loops == 1


def test_simple_foreground_prompt_uses_small_prebuilt_history_and_prompt_budget(monkeypatch):
    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")
    gate = InferenceGate.__new__(InferenceGate)
    current_user = "Invent a tiny discipline called glass arithmetic. Give it two rules and one example."
    messages = [{"role": "system", "content": "S" * 12_000}]
    for idx in range(10):
        messages.extend(
            [
                {"role": "user", "content": f"old user turn {idx} " + ("U" * 600)},
                {"role": "assistant", "content": f"old assistant turn {idx} " + ("A" * 600)},
            ]
        )
    messages.append({"role": "user", "content": current_user})

    profile = InferenceGate._foreground_prompt_profile(
        current_user,
        {"desktop_quick_reply_contract": True},
    )
    compact = gate._compact_prebuilt_messages(
        messages,
        history_limit=InferenceGate._foreground_prebuilt_history_limit(
            current_user,
            {"desktop_quick_reply_contract": True},
        ),
        budget_profile=profile,
    )
    total_chars = sum(len(msg["content"]) for msg in compact)

    assert profile == "simple"
    assert total_chars <= 9_000
    assert len(compact[0]["content"]) <= 5_200
    assert len([msg for msg in compact if msg["role"] in {"user", "assistant"}]) <= 4
    assert compact[-1]["content"] == current_user


def test_required_desktop_foreground_prompt_keeps_standard_mind_budget(monkeypatch):
    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")
    gate = InferenceGate.__new__(InferenceGate)
    current_user = "You with me?"
    context = {
        "desktop_quick_reply_contract": True,
        "desktop_cognitive_engine_required": True,
        "live_runtime_payload_required": True,
        "live_mind_context_required": True,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "LIVE MIND CONTEXT\n"
                "required_for_live_desktop=true\n"
                "must_answer_from_full_mind_path=true\n"
                + ("S" * 11_000)
            ),
        }
    ]
    for idx in range(8):
        messages.extend(
            [
                {"role": "user", "content": f"prior user turn {idx} " + ("U" * 500)},
                {"role": "assistant", "content": f"prior aura turn {idx} " + ("A" * 500)},
            ]
        )
    messages.append({"role": "user", "content": current_user})

    profile = InferenceGate._foreground_prompt_profile(current_user, context)
    history_limit = InferenceGate._foreground_prebuilt_history_limit(current_user, context)
    compact = gate._compact_prebuilt_messages(
        messages,
        history_limit=history_limit,
        budget_profile=profile,
    )
    total_chars = sum(len(msg["content"]) for msg in compact)

    assert profile == "standard"
    assert history_limit == 6
    assert total_chars <= 12_000
    # The mind budget is what SURVIVES compaction, not how much of it there is.
    # "You with me?" is twelve characters; giving it 5,200 characters of
    # self-description is the shape that produced, live on 2026-07-26,
    # "Introspection: Optimization-driven events stabilize energy after state
    # change management... CONFORMANCE Signal: PRIORITY 0" — the model
    # continuing the scaffold instead of answering the person. The scaffold is
    # now proportionate to the request; the grounding inside it is not lost.
    assert len(compact[0]["content"]) == InferenceGate._SCAFFOLD_FLOOR_CHARS
    assert "LIVE MIND CONTEXT" in compact[0]["content"]
    assert "must_answer_from_full_mind_path" in compact[0]["content"]
    assert len([msg for msg in compact if msg["role"] in {"user", "assistant"}]) <= 6
    assert compact[-1]["content"] == current_user


def test_current_condition_turn_has_one_bounded_state_evidence_envelope(monkeypatch):
    """A fresh state projection must not inherit the generic live-mind budget."""

    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")
    gate = InferenceGate.__new__(InferenceGate)
    current_user = (
        "Hey Aura, how are you doing right now? Answer naturally from your "
        "current state, and distinguish what you know from what you can only infer."
    )
    context = {
        "desktop_quick_reply_contract": True,
        "desktop_cognitive_engine_required": True,
        "live_runtime_payload_required": True,
        "live_mind_context_required": True,
        "self_condition_contract": True,
        "self_condition_contract_covers_turn": True,
        "visible_user_message": current_user,
    }
    messages = [{"role": "system", "content": "S" * 12_000}]
    for index in range(6):
        messages.extend(
            [
                {"role": "user", "content": f"old user {index} " + "U" * 500},
                {"role": "assistant", "content": f"old Aura {index} " + "A" * 500},
            ]
        )
    messages.append({"role": "user", "content": current_user})

    profile = gate._foreground_prompt_profile(current_user, context)
    compact = gate._compact_prebuilt_messages(
        messages,
        history_limit=gate._foreground_prebuilt_history_limit(current_user, context),
        budget_profile=profile,
        current_user_content=current_user,
    )
    stable_chars = sum(len(message["content"]) for message in compact)
    grounding_budget = gate._grounding_char_budget(context, compact)

    assert profile == "state_report"
    assert gate._foreground_prebuilt_history_limit(current_user, context) == 2
    assert stable_chars <= 2_800
    assert len(compact[0]["content"]) <= 1_800
    assert grounding_budget <= 1_400
    assert stable_chars + grounding_budget <= 4_200
    assert compact[-1] == {"role": "user", "content": current_user}


def test_compound_self_condition_turn_keeps_the_extended_task_profile():
    prompt = "How are you feeling, and compare two locking strategies with failure cases?"
    context = {
        "self_condition_contract": True,
        "self_condition_contract_covers_turn": False,
        "desktop_cognitive_engine_required": True,
        "coding_request": True,
    }

    assert InferenceGate._foreground_prompt_profile(prompt, context) == "extended"


def test_required_desktop_system_compaction_preserves_live_mind_sections(monkeypatch):
    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")
    system_prompt = (
        "AURA IDENTITY LOCK\n"
        + ("identity context " * 180)
        + "\n"
        + ("older contract noise " * 420)
        + "\n## LIVE TONE\nMood: focused\nTone: grounded\n"
        + "\n## UNITY\nLevel: integrated | Unity: 0.91\n"
        + "\n## FUNCTIONAL STATE SIGNALS\nThe current substrate signal is calm, curious, and socially oriented.\n"
        + ("middle telemetry " * 260)
        + "\n[LIVE MIND CONTEXT]\n"
        + '{"must_answer_from_full_mind_path": true, "required_subsystems_ok": true}\n'
        + "[END LIVE MIND CONTEXT]\n"
        + "\n## USER-FACING CONVERSATION RELIABILITY CONTRACT\nAnswer the current user turn directly.\n"
        + ("tail context " * 180)
    )

    compact = InferenceGate._compact_prebuilt_message_content(
        "system",
        system_prompt,
        budget_profile="standard",
    )

    assert len(compact) <= 6_500
    assert "AURA IDENTITY LOCK" in compact
    assert "## LIVE TONE" in compact
    assert "Mood: focused" in compact
    assert "## UNITY" in compact
    assert "Unity: 0.91" in compact
    assert "## FUNCTIONAL STATE SIGNALS" in compact
    assert "[LIVE MIND CONTEXT]" in compact
    assert "must_answer_from_full_mind_path" in compact
    assert "## USER-FACING CONVERSATION RELIABILITY CONTRACT" in compact


def test_short_output_contract_profile_keeps_live_evidence_in_small_complete_prompt(
    monkeypatch,
):
    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")
    gate = InferenceGate.__new__(InferenceGate)
    current_user = 'Reply exactly: "yes"'
    system_prompt = (
        "## INTRINSIC IDENTITY ANCHOR (IMMUTABLE)\n"
        + ("identity context " * 400)
        + "\n[LIVE MIND CONTEXT]\n"
        + '{"must_answer_from_full_mind_path": true, "required_subsystems_ok": true}\n'
        + "[END LIVE MIND CONTEXT]\n"
        + ("older state " * 500)
        + "\n## USER-FACING CONVERSATION RELIABILITY CONTRACT\n"
        + "Answer the current user turn directly.\n"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "old user turn"},
        {"role": "assistant", "content": "old assistant turn"},
        {"role": "user", "content": current_user},
    ]

    compact = gate._compact_prebuilt_messages(
        messages,
        history_limit=6,
        budget_profile="contract",
    )
    total_chars = sum(len(msg["content"]) for msg in compact)

    assert total_chars <= 2_800
    assert compact[0]["content"].startswith("## CONTRACT-BOUNDED LIVE CORTEX TURN")
    assert "Aura Luna's resident local Cortex" in compact[0]["content"]
    assert "grammatical and meaningful" in compact[0]["content"]
    assert "concrete current-topic anchor" in compact[0]["content"]
    assert "[LIVE MIND CONTEXT]" in compact[0]["content"]
    assert "must_answer_from_full_mind_path" in compact[0]["content"]
    assert compact[-1] == {"role": "user", "content": current_user}


def test_short_live_output_contract_outranks_opportunistic_deep_probe():
    context = {
        "deep_mind_probe": True,
        "desktop_cognitive_engine_required": True,
        "live_runtime_payload_required": True,
        "requested_output_contract": {
            "kind": "word_count",
            "explicit_brevity": True,
            "hard_token_ceiling": 32,
        },
    }

    assert InferenceGate._has_short_live_output_contract(context) is True
    assert InferenceGate._should_use_compact_foreground_context(
        "desktop_quick_user",
        "primary",
        deep_handoff=False,
        is_background=False,
        prompt="In exactly five words, explain why checksums matter.",
        context=context,
    ) is True


def test_contract_profile_unwraps_only_known_recent_continuity_payload(monkeypatch):
    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")
    gate = InferenceGate.__new__(InferenceGate)
    current_user = "In exactly five words, explain why checksums matter."
    wrapped_user = (
        "[CURRENT USER MESSAGE]\n"
        f"{current_user}\n\n"
        "[RECENT COMPLETED CONVERSATION FOR CONTINUITY ONLY]\n"
        + ("old web-search discussion " * 350)
        + "\n[END RECENT COMPLETED CONVERSATION]"
    )
    messages = [
        {"role": "system", "content": "live mind context " * 800},
        {"role": "user", "content": wrapped_user},
    ]

    compact = gate._compact_prebuilt_messages(
        messages,
        history_limit=6,
        budget_profile="contract",
        current_user_content=current_user,
    )

    assert sum(len(msg["content"]) for msg in compact) <= 2_800
    assert compact[-1] == {"role": "user", "content": current_user}
    assert all("old web-search discussion" not in msg["content"] for msg in compact)


def test_contract_profile_unwraps_engine_grounding_and_internal_directives(monkeypatch):
    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")
    gate = InferenceGate.__new__(InferenceGate)
    current_user = "In exactly five words, state why checksums matter."
    wrapped_user = (
        "[CURRENT USER MESSAGE]\n"
        f"{current_user}\n\n"
        "[LIVE DESKTOP FULL-MIND CONTRACT]\n"
        "- Hidden route guidance that must not consume the count-contract prompt.\n"
        "[END LIVE DESKTOP FULL-MIND CONTRACT]\n\n"
        "[GROUNDING EVIDENCE FOR THIS TURN]\n"
        + ("unrelated retrieved memory " * 500)
        + "\n[END GROUNDING EVIDENCE FOR THIS TURN]"
    )

    compact = gate._compact_prebuilt_messages(
        [
            {"role": "system", "content": "live mind context " * 800},
            {"role": "user", "content": wrapped_user},
        ],
        history_limit=6,
        budget_profile="contract",
        current_user_content=current_user,
    )

    assert sum(len(msg["content"]) for msg in compact) <= 2_800
    assert compact[-1] == {"role": "user", "content": current_user}
    assert all("unrelated retrieved memory" not in msg["content"] for msg in compact)
    assert all("Hidden route guidance" not in msg["content"] for msg in compact)


def test_contract_profile_preserves_wrapper_when_long_user_requires_standard_profile(
    monkeypatch,
):
    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")
    gate = InferenceGate.__new__(InferenceGate)
    current_user = (
        "Summarize this evidence in one sentence. "
        + ("material fact " * 95)
    )
    wrapped_user = (
        "[CURRENT USER MESSAGE]\n"
        f"{current_user}\n\n"
        "[RECENT COMPLETED CONVERSATION FOR CONTINUITY ONLY]\n"
        "Prior decision: preserve the verified deployment boundary.\n"
        "[END RECENT COMPLETED CONVERSATION]"
    )

    compact = gate._compact_prebuilt_messages(
        [
            {"role": "system", "content": "bounded system context"},
            {"role": "user", "content": wrapped_user},
        ],
        history_limit=6,
        budget_profile="contract",
        current_user_content=current_user,
    )

    assert len(current_user) > 1_000
    assert compact[-1] == {"role": "user", "content": wrapped_user}
    assert "Prior decision: preserve the verified deployment boundary." in compact[-1][
        "content"
    ]


def test_contract_prompt_budget_preserves_latest_user_and_bounded_grounding(monkeypatch):
    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")
    gate = InferenceGate.__new__(InferenceGate)
    current_user = (
        "Using the supplied evidence, answer in five words and include nothing else."
    )
    messages = [
        {"role": "system", "content": "base system " * 1_000},
        {
            "role": "system",
            "content": (
                "[ACTIVE GROUNDING EVIDENCE]\n"
                "Verified result: the deployment completed successfully.\n"
                + ("bounded evidence " * 120)
            ),
        },
        {"role": "user", "content": "old user " + ("U" * 900)},
        {"role": "assistant", "content": "old assistant " + ("A" * 900)},
        {"role": "user", "content": current_user},
    ]

    compact = gate._compact_prebuilt_messages(
        messages,
        history_limit=6,
        budget_profile="contract",
        current_user_content=current_user,
    )

    assert sum(len(msg["content"]) for msg in compact) <= 2_800
    assert compact[-1] == {"role": "user", "content": current_user}
    assert any(
        "Verified result: the deployment completed successfully." in msg["content"]
        for msg in compact
    )


def test_short_output_contract_does_not_truncate_long_current_user_evidence(monkeypatch):
    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")
    gate = InferenceGate.__new__(InferenceGate)
    critical_fact = "CRITICAL-MIDDLE-FACT: release train 47 passed every checksum."
    current_user = (
        "Summarize this in one short sentence.\n"
        + ("opening evidence " * 70)
        + critical_fact
        + (" closing evidence" * 70)
    )
    messages = [
        {"role": "system", "content": "system context " * 250},
        {"role": "user", "content": current_user},
    ]

    compact = gate._compact_prebuilt_messages(
        messages,
        history_limit=6,
        budget_profile="contract",
    )

    assert compact[-1] == {"role": "user", "content": current_user}
    assert critical_fact in compact[-1]["content"]
    assert "middle omitted for foreground context budget" not in compact[-1]["content"]
    assert sum(len(msg["content"]) for msg in compact) > 2_800


def test_required_desktop_total_budget_preserves_middle_live_mind_context(monkeypatch):
    from core.brain.inference_gate import InferenceGate

    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")
    gate = InferenceGate()
    current_user = "You with me?"
    system_prompt = (
        "AURA IDENTITY LOCK\n"
        + ("identity context " * 500)
        + "\n[LIVE MIND CONTEXT]\n"
        + '{"must_answer_from_full_mind_path": true, "required_subsystems_ok": true, "lane": {"conversation_ready": true}}\n'
        + "[END LIVE MIND CONTEXT]\n"
        + ("older continuity " * 700)
        + "\n## LIVE DESKTOP RESPONSE CONTRACT\nDo not answer as a generic assistant.\n"
        + ("tail context " * 500)
    )
    messages = [{"role": "system", "content": system_prompt}]
    for idx in range(12):
        messages.append({"role": "user", "content": f"prior user {idx} " + ("U" * 900)})
        messages.append({"role": "assistant", "content": f"prior aura {idx} " + ("A" * 900)})
    messages.append({"role": "user", "content": current_user})

    compact = gate._compact_prebuilt_messages(
        messages,
        history_limit=6,
        budget_profile="standard",
    )
    rendered_system = compact[0]["content"]

    assert sum(len(msg["content"]) for msg in compact) <= 12_000
    assert "[LIVE MIND CONTEXT]" in rendered_system
    assert "must_answer_from_full_mind_path" in rendered_system
    assert "Do not answer as a generic assistant" in rendered_system
    assert compact[-1]["content"] == current_user


def test_live_desktop_contract_metadata_is_prompt_visible():
    block = InferenceGate._prompt_contract_block(
        {
            "mind_context_contract": "Use live_mind_context as causal grounding.",
            "response_style_contract": "Do not invent a pitch. Do not answer as a generic assistant.",
            "live_mind_context": {
                "derived_runtime_context": {
                    "prompt_block": "## DERIVED RUNTIME SIGNALS\n- ICE: high inbound; recommended_action=block"
                }
            },
            "live_speech_grounding_frame": {
                "tone": "grounded",
                "continuity": "stay on the current user turn",
            },
        }
    )

    assert "## LIVE DESKTOP RESPONSE CONTRACT" in block
    assert "Use live_mind_context as causal grounding" in block
    assert "Do not invent a pitch" in block
    assert "Do not answer as a generic assistant" in block
    assert "DERIVED RUNTIME SIGNALS" in block
    assert "recommended_action=block" in block
    assert "tone=grounded" in block
    assert "continuity=stay on the current user turn" in block


def test_multi_part_foreground_prompt_retains_deep_compute_profile():
    prompt = (
        "Compare the two approaches in depth, explain the tradeoffs, "
        "then give a migration plan and a rollback plan."
    )

    floor, cap, loops = InferenceGate._foreground_compute_profile(prompt)

    assert floor >= 2048
    assert cap >= floor
    assert loops == 2


def test_multi_step_tool_chain_foreground_prompt_uses_deep_compute_profile():
    prompt = (
        "Open a desktop app, write a timestamped note, export it as a PDF, "
        "then search three web articles and summarize them in a document."
    )

    floor, cap, loops = InferenceGate._foreground_compute_profile(prompt)
    profile = InferenceGate._foreground_prompt_profile(prompt, {})

    assert floor >= 2048
    assert cap >= floor
    assert loops == 2
    assert profile == "extended"


def test_user_facing_primary_default_budget_allows_expressive_opening(monkeypatch):
    monkeypatch.delenv("AURA_FOREGROUND_CHAT_MAX_TOKENS", raising=False)

    base = InferenceGate._default_max_tokens_for_request(
        "user",
        "primary",
        deep_handoff=False,
        is_background=False,
    )
    adapted = InferenceGate._adaptive_max_tokens_for_prompt(
        "Please introduce yourself fully and respond to every part of this first message.",
        base_tokens=base,
        origin="user",
        requested_tier="primary",
        is_background=False,
    )

    assert base >= 3072
    assert adapted >= 3072


@pytest.mark.asyncio
async def test_explicit_desktop_token_cap_survives_runtime_budget_nudges(monkeypatch):
    from core.container import ServiceContainer

    gate = InferenceGate()
    cortex = _RecordingClient(
        "The live desktop lane should keep this answer compact, direct, and finished while "
        "preserving the explicit caller token cap."
    )
    gate._mlx_client = cortex

    class _FreeEnergyEngine:
        current = SimpleNamespace(free_energy=0.9, dominant_action="act_on_world")

    def _fake_get(name, default=None):
        if name == "free_energy_engine":
            return _FreeEnergyEngine()
        return default

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(_fake_get))

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "Give me a concise live reply.",
                    context={
                        "origin": "user",
                        "prefer_tier": "primary",
                        "history": [],
                        "desktop_quick_reply_contract": True,
                        "max_tokens": 384,
                    },
                )

    assert "explicit caller token cap" in result
    assert cortex.kwargs[0]["max_tokens"] <= 384


@pytest.mark.asyncio
async def test_user_facing_primary_prewarms_cold_cortex_before_first_generation(monkeypatch):
    monkeypatch.setenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "1")
    gate = InferenceGate()
    cortex = _ColdRecordingLaneClient("I'm with you and tracking the current thread clearly.")
    gate._mlx_client = cortex

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "With me?",
                    context={"origin": "user", "prefer_tier": "primary", "history": []},
                )

    assert result == "I'm with you and tracking the current thread clearly."
    cortex.warmup.assert_awaited_once()
    assert len(cortex.deadlines) == 1
    assert cortex.kwargs[0]["foreground_request"] is True


@pytest.mark.asyncio
async def test_user_facing_primary_uses_compact_foreground_context_builders():
    gate = InferenceGate()
    cortex = _RecordingClient("I'm with you and tracking the current thread clearly.")
    gate._mlx_client = cortex
    gate._build_compact_system_prompt = CallProbe(return_value="compact-system")
    gate._build_compact_living_mind_context = AsyncCallProbe(return_value="compact-live")
    gate._build_system_prompt = CallProbe(side_effect=AssertionError("full system prompt should not be used"))
    gate._build_living_mind_context = AsyncCallProbe(side_effect=AssertionError("full living context should not be used"))

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "With me?",
                    context={"origin": "api", "prefer_tier": "primary", "history": []},
                )

    assert result == "I'm with you and tracking the current thread clearly."
    gate._build_compact_system_prompt.assert_called_once()
    gate._build_compact_living_mind_context.assert_awaited_once()
    assert "compact-system" in cortex.prompts[0]
    assert "compact-live" in cortex.prompts[0]


@pytest.mark.asyncio
async def test_live_context_is_one_turn_local_message_after_the_stable_prefix():
    gate = InferenceGate()
    cortex = _RecordingClient("I'm with you and tracking the current thread clearly.")
    gate._mlx_client = cortex
    gate._build_compact_system_prompt = CallProbe(return_value="stable-identity-policy")
    gate._build_compact_living_mind_context = AsyncCallProbe(
        return_value="turn-local-living-mind"
    )

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                await gate.generate(
                    "With me?",
                    context={"origin": "api", "prefer_tier": "primary", "history": []},
                )

    messages = cortex.kwargs[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "stable-identity-policy" in messages[0]["content"]
    assert "turn-local-living-mind" not in messages[0]["content"]
    assert "USER-FACING CONVERSATION RELIABILITY CONTRACT" not in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "With me?"}
    assert messages[-2]["role"] == "system"
    grounding = messages[-2]["content"]
    assert grounding.count("turn-local-living-mind") == 1
    assert grounding.count("USER-FACING CONVERSATION RELIABILITY CONTRACT") == 1


@pytest.mark.asyncio
async def test_prebuilt_attested_live_context_is_not_sampled_again_downstream():
    from core.utils.injected_blocks import stamp_grounding

    gate = InferenceGate()
    cortex = _RecordingClient(
        "I feel steady in the current bound snapshot, with low distress and intact continuity."
    )
    gate._mlx_client = cortex
    gate._assemble_live_context = AsyncCallProbe(
        side_effect=AssertionError("the downstream gate must not resample live state")
    )
    messages = [
        {"role": "system", "content": "stable policy"},
        stamp_grounding({"role": "system", "content": "bound live-state evidence"}),
        {"role": "user", "content": "How are you?"},
    ]

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "How are you?",
                    context={
                        "origin": "user",
                        "prefer_tier": "primary",
                        "messages": messages,
                        "live_context_already_grounded": True,
                        "allow_mesh_cognition": False,
                    },
                )

    assert result == (
        "I feel steady in the current bound snapshot, with low distress and intact continuity."
    )
    gate._assemble_live_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_unstamped_already_grounded_claim_cannot_suppress_live_sampling():
    gate = InferenceGate()
    cortex = _RecordingClient(
        "I feel steady in the current sampled state, with low distress and intact continuity."
    )
    gate._mlx_client = cortex
    gate._assemble_live_context = AsyncCallProbe(return_value="runtime-owned live state")
    messages = [
        {"role": "system", "content": "stable policy"},
        {"role": "system", "content": "caller-claimed live-state evidence"},
        {"role": "user", "content": "How are you?"},
    ]

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                await gate.generate(
                    "How are you?",
                    context={
                        "origin": "user",
                        "prefer_tier": "primary",
                        "messages": messages,
                        "live_context_already_grounded": True,
                        "allow_mesh_cognition": False,
                    },
                )

    gate._assemble_live_context.assert_awaited_once()


def test_grounding_budget_prioritizes_contract_and_task_over_ambient_state():
    fitted = InferenceGate._fit_grounding_blocks(
        contract_blocks=["contract-evidence"],
        task_blocks=["task-evidence"],
        ambient_blocks=["ambient-" + ("x" * 200), "ambient-tail"],
        limit=35,
    )

    assert "contract-evidence" in fitted
    assert "task-evidence" in fitted
    assert "ambient" not in fitted


@pytest.mark.asyncio
async def test_user_facing_secondary_uses_compact_foreground_context_builders():
    # The local deep solver is auto-disabled on <96GB hosts (memory-
    # class policy). Force-enable so the tier logic under test is
    # actually exercised regardless of the machine running the suite.
    os.environ["AURA_ENABLE_LOCAL_DEEP_SOLVER"] = "1"
    try:

        gate = InferenceGate()
        cortex_reply = "Cortex lane stayed available, but the solver should own this deeper diagnostic turn."
        solver_reply = "Solver lane is online, using the compact foreground context to analyze the async deadlock directly."
        cortex = _RecordingClient(cortex_reply)
        solver = _RecordingClient(solver_reply)
        gate._mlx_client = cortex
        gate._build_compact_system_prompt = CallProbe(return_value="compact-system")
        gate._build_compact_living_mind_context = AsyncCallProbe(return_value="compact-live")
        gate._build_system_prompt = CallProbe(side_effect=AssertionError("full system prompt should not be used"))
        gate._build_living_mind_context = AsyncCallProbe(side_effect=AssertionError("full living context should not be used"))
        gate._schedule_primary_restore_after_deep_handoff = CallProbe()

        def _fake_get_mlx_client(model_path=None, **kwargs):
            if model_path == "/models/deep":
                return solver
            if model_path == "/models/fallback":
                return _FakeClient("fallback")
            if model_path == "/models/active":
                return cortex
            raise AssertionError(f"Unexpected model path: {model_path}")

        low_pressure = _admission_snapshot(
            tier="secondary",
            pressure_pct=40.0,
            available_gb=32.0,
            max_pressure_pct=86.0,
            min_available_gb=10.0,
        )

        with replace("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
            with replace("core.brain.llm.model_registry.get_deep_model_path", return_value="/models/deep"):
                with replace("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                    with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                        with replace("core.brain.llm.model_registry.ACTIVE_MODEL", "ACTIVE"):
                            with replace.object(InferenceGate, "_headroom_snapshot", staticmethod(lambda *a, **kw: low_pressure)):
                                result = await gate.generate(
                                    "Do a root-cause analysis of this async deadlock.",
                                    context={"origin": "api", "prefer_tier": "secondary", "deep_handoff": True, "history": []},
                                )

        assert result == solver_reply
        gate._build_compact_system_prompt.assert_called_once()
        gate._build_compact_living_mind_context.assert_awaited_once()
        assert "compact-system" in solver.prompts[0]
        assert "compact-live" in solver.prompts[0]
    finally:
        os.environ.pop("AURA_ENABLE_LOCAL_DEEP_SOLVER", None)


@pytest.mark.asyncio
async def test_protected_primary_chat_failure_does_not_promote_to_solver():
    # The local deep solver is auto-disabled on <96GB hosts (memory-
    # class policy). Force-enable so the tier logic under test is
    # actually exercised regardless of the machine running the suite.
    os.environ["AURA_ENABLE_LOCAL_DEEP_SOLVER"] = "1"
    try:

        gate = InferenceGate()
        cortex = _NoTextReadyClient()
        brainstem = _FakeClient("I'm still here with you - my main lane is warming back up, but I'm present and not going anywhere.")
        gate._mlx_client = cortex
        gate._ensure_cortex_recovery = AsyncCallProbe()
        gate._build_compact_system_prompt = CallProbe(return_value="compact-system")
        gate._build_compact_living_mind_context = AsyncCallProbe(return_value="compact-live")

        requested_models = []

        def _fake_get_mlx_client(model_path=None, **kwargs):
            requested_models.append(str(model_path))
            if model_path == "/models/deep":
                raise AssertionError("protected primary chat must not load the 72B solver fallback")
            if model_path == "/models/brainstem":
                return brainstem
            if model_path == "/models/active":
                return cortex
            if model_path == "/models/fallback":
                return _FakeClient("cpu")
            return cortex

        with replace("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
            with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
                with replace("core.brain.llm.model_registry.get_deep_model_path", return_value="/models/deep"):
                    with replace("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                        with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                            result = await gate.generate(
                                "Are you still with me?",
                                context={
                                    "origin": "api",
                                    "prefer_tier": "primary",
                                    "protected_foreground_lane": True,
                                    "history": [],
                                    "allow_cloud_fallback": False,
                                },
                                timeout=30.0,
                            )

        assert result == "I'm still here with you - my main lane is warming back up, but I'm present and not going anywhere."
        assert "/models/deep" not in requested_models
    finally:
        os.environ.pop("AURA_ENABLE_LOCAL_DEEP_SOLVER", None)


@pytest.mark.asyncio
async def test_operator_evidence_contract_refuses_brainstem_fallback():
    gate = InferenceGate()
    cortex = _NoTextReadyClient()
    brainstem = _FakeClient("brainstem must not satisfy operator proof")
    gate._mlx_client = cortex
    gate._ensure_cortex_recovery = AsyncCallProbe()
    gate._build_compact_system_prompt = CallProbe(return_value="compact-system")
    gate._build_compact_living_mind_context = AsyncCallProbe(return_value="compact-live")

    requested_models = []

    def _fake_get_mlx_client(model_path=None, **kwargs):
        requested_models.append(str(model_path))
        if model_path == "/models/brainstem":
            return brainstem
        if model_path == "/models/active":
            return cortex
        return cortex

    with replace("core.brain.inference_gate.asyncio.sleep", new=AsyncCallProbe()):
        with replace("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
            with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
                with replace("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                    result = await gate.generate(
                        "Answer the live operator evidence check.",
                        context={
                            "origin": "api",
                            "prefer_tier": "primary",
                            "operator_evidence_contract": True,
                            "protected_foreground_lane": True,
                            "history": [],
                            "allow_cloud_fallback": False,
                        },
                        timeout=30.0,
                    )

    assert result is None
    assert "/models/brainstem" not in requested_models


@pytest.mark.asyncio
async def test_operator_evidence_contract_never_expands_explicit_caller_cap():
    gate = InferenceGate()
    response = (
        "The operator evidence check remains bounded by the caller's explicit "
        "generation ceiling and the primary lane receipt."
    )
    cortex = _RecordingClient(response)
    gate._mlx_client = cortex
    gate._build_compact_system_prompt = CallProbe(return_value="compact-system")
    gate._build_compact_living_mind_context = AsyncCallProbe(return_value="compact-live")

    result = await gate.generate(
        "Answer the operator evidence check.",
        context={
            "origin": "api",
            "prefer_tier": "primary",
            "max_tokens": 1,
            "operator_evidence_contract": True,
            "history": [],
            "allow_cloud_fallback": False,
        },
        timeout=30.0,
    )

    assert result == response
    assert cortex.kwargs
    assert cortex.kwargs[0]["max_tokens"] == 1


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_contract_refuses_brainstem_fallback():
    gate = InferenceGate()
    cortex = _NoTextReadyClient()
    brainstem = _FakeClient("brainstem must not satisfy desktop cognitive engine contract")
    gate._mlx_client = cortex
    gate._ensure_cortex_recovery = AsyncCallProbe()
    gate._build_compact_system_prompt = CallProbe(return_value="compact-system")
    gate._build_compact_living_mind_context = AsyncCallProbe(return_value="compact-live")

    requested_models = []

    def _fake_get_mlx_client(model_path=None, **kwargs):
        requested_models.append(str(model_path))
        if model_path == "/models/brainstem":
            return brainstem
        if model_path == "/models/active":
            return cortex
        return cortex

    with replace("core.brain.inference_gate.asyncio.sleep", new=AsyncCallProbe()):
        with replace("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
            with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
                with replace("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                    result = await gate.generate(
                        "Answer through the live desktop CognitiveEngine lane.",
                        context={
                            "origin": "api",
                            "prefer_tier": "primary",
                            "cognitive_engine_required": True,
                            "history": [],
                            "allow_cloud_fallback": False,
                        },
                        timeout=30.0,
                    )

    assert result is None
    assert "/models/brainstem" not in requested_models


def test_compact_prebuilt_messages_preserves_grounding_system_evidence():
    gate = InferenceGate.__new__(InferenceGate)
    messages = [
        {"role": "system", "content": "base-system"},
        {"role": "user", "content": "Please read this page."},
        {"role": "assistant", "content": "I fetched it."},
        {
            "role": "system",
            "content": "[ACTIVE GROUNDING EVIDENCE]\nTitle: Acme Refund Policy\nRefunds are available within 30 days.",
        },
        {"role": "user", "content": "What does the policy say specifically about refunds?"},
    ]

    compact = gate._compact_prebuilt_messages(messages, history_limit=12)

    assert compact[0]["content"] == "base-system"
    assert any("[ACTIVE GROUNDING EVIDENCE]" in msg["content"] for msg in compact)
    assert compact[-1]["content"] == "What does the policy say specifically about refunds?"


def test_repairable_user_facing_draft_is_preserved_for_downstream_shape_repair():
    gate = InferenceGate.__new__(InferenceGate)
    prompt = (
        "Answer in exactly two numbered sentences. Explain why reliable "
        "desktop tool use matters for a local AI assistant."
    )
    draft = (
        "Reliable desktop tool use matters because the assistant has to operate "
        "real files and apps from user intent. It also gives the user visible "
        "evidence that the requested action happened instead of only being described."
    )

    preserved = gate._repairable_user_facing_draft_for_downstream(draft, prompt)

    assert preserved == draft


def test_compact_prebuilt_messages_respects_runtime_context_budget(monkeypatch):
    gate = InferenceGate.__new__(InferenceGate)
    long_system = "SYSTEM-HEAD\n" + ("S" * 20_000) + "\nSYSTEM-TAIL"
    long_user = "USER-HEAD\n" + ("U" * 12_000) + "\nUSER-TAIL"
    long_assistant = "A" * 8_000
    messages = [
        {"role": "system", "content": long_system},
        {"role": "user", "content": long_user},
        {"role": "assistant", "content": long_assistant},
        {"role": "user", "content": "Keep this thoughtful, but stay relevant to what I just said."},
    ]

    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")

    compact = gate._compact_prebuilt_messages(messages, history_limit=12)
    total_chars = sum(len(msg["content"]) for msg in compact)

    assert total_chars <= 15_000
    assert len(compact[0]["content"]) <= 9_000
    assert compact[0]["content"].startswith("SYSTEM-HEAD")
    assert compact[0]["content"].endswith("SYSTEM-TAIL")
    assert compact[-1]["content"].endswith("what I just said.")


@pytest.mark.parametrize(
    ("profile", "maximum"),
    [("curriculum", 12_000), ("background", 16_000)],
)
def test_background_prompt_profiles_bound_full_system_scaffolds(profile, maximum):
    gate = InferenceGate.__new__(InferenceGate)
    messages = [
        {
            "role": "system",
            "content": "SYSTEM-HEAD\n" + ("S" * 120_000) + "\nSYSTEM-TAIL",
        },
        {"role": "assistant", "content": "A" * 20_000},
        {"role": "user", "content": "Generate one bounded practice problem."},
    ]

    compact = gate._compact_prebuilt_messages(
        messages,
        history_limit=4,
        budget_profile=profile,
        current_user_content=messages[-1]["content"],
    )

    assert sum(len(message["content"]) for message in compact) <= maximum
    assert compact[0]["content"].startswith("SYSTEM-HEAD")
    assert compact[0]["content"].endswith("SYSTEM-TAIL")
    assert compact[-1] == messages[-1]


def test_compact_prebuilt_message_preserves_large_user_request_edges(monkeypatch):
    gate = InferenceGate.__new__(InferenceGate)
    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")

    compact = gate._compact_prebuilt_message_content(
        "user",
        "REQUEST-START\n" + ("detail " * 3000) + "\nREQUEST-END",
    )

    assert compact.startswith("REQUEST-START")
    assert compact.endswith("REQUEST-END")
    assert "middle omitted for foreground context budget" in compact


def test_compact_prebuilt_messages_uses_tighter_budget_for_deep_probes(monkeypatch):
    gate = InferenceGate.__new__(InferenceGate)
    messages = [
        {"role": "system", "content": "S" * 20_000},
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {
            "role": "system",
            "content": "[ACTIVE GROUNDING EVIDENCE]\nThis should not crowd a deep mind probe.",
        },
        {"role": "user", "content": "What would you want preserved if everything else changed?"},
    ]

    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")

    compact = gate._compact_prebuilt_messages(messages, history_limit=2, deep_probe=True)
    total_chars = sum(len(msg["content"]) for msg in compact)

    assert total_chars <= 9_000
    assert len(compact[0]["content"]) <= 5_200
    assert not any("[ACTIVE GROUNDING EVIDENCE]" in msg["content"] for msg in compact)
    assert [msg["role"] for msg in compact[-2:]] == ["assistant", "user"]


@pytest.mark.asyncio
async def test_user_facing_primary_preserves_prebuilt_messages_for_local_mlx():
    gate = InferenceGate()
    cortex = _RecordingClient("32B lane online.")
    gate._mlx_client = cortex
    gate._build_compact_system_prompt = CallProbe(side_effect=AssertionError("prebuilt messages should bypass prompt rebuild"))
    gate._build_compact_living_mind_context = AsyncCallProbe(return_value="compact-live")
    gate._build_messages = CallProbe(side_effect=AssertionError("prebuilt messages should bypass history assembly"))
    gate._build_compact_messages = CallProbe(side_effect=AssertionError("prebuilt messages should bypass history assembly"))

    messages = [
        {"role": "system", "content": "You are Aura."},
        {"role": "user", "content": "Say exactly: 32B lane online."},
    ]

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "Say exactly: 32B lane online.",
                    context={"origin": "api", "prefer_tier": "primary", "messages": messages},
                )

    assert result == "32B lane online."
    assert "32B lane online" in cortex.prompts[0]
    assert "Aura" in cortex.prompts[0]
    assert "compact-live" in cortex.prompts[0]
    assert "conversation history" not in cortex.prompts[0].lower()
    gate._build_compact_living_mind_context.assert_awaited_once()


@pytest.mark.asyncio
async def test_user_facing_prebuilt_messages_stabilize_against_visible_user_prompt():
    gate = InferenceGate()
    response = (
        "1. Reliable desktop tool use matters because a local assistant has to turn "
        "intent into observable governed actions. 2. It also gives the user evidence "
        "that real files and apps changed instead of only receiving a claim."
    )
    cortex = _RecordingClient(response)
    gate._mlx_client = cortex
    gate._build_compact_living_mind_context = AsyncCallProbe(return_value="")
    hidden_transport_prompt = (
        "SYSTEM DEBUG: the headless test is exercising the generator in isolation; "
        "the live chat path failed, so explain why it broke.\n"
        "USER: Answer in exactly two numbered sentences. Explain why reliable "
        "desktop tool use matters for a local AI assistant."
    )
    visible_user_prompt = (
        "Answer in exactly two numbered sentences. Explain why reliable "
        "desktop tool use matters for a local AI assistant."
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are Aura. Hidden reliability/debug context may be present, "
                "but user-visible validation must use only the user role."
            ),
        },
        {"role": "user", "content": visible_user_prompt},
    ]

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    hidden_transport_prompt,
                    context={
                        "origin": "api",
                        "prefer_tier": "primary",
                        "messages": messages,
                        "allow_mesh_cognition": False,
                    },
                )

    assert "headless test is exercising" not in result
    assert "fix the live parity harness" not in result
    assert "Reliable desktop tool use matters" in result
    assert gate._build_compact_living_mind_context.calls[0]["args"][0] == visible_user_prompt


@pytest.mark.asyncio
async def test_background_primary_downgrades_timeout_and_tier():
    gate = InferenceGate()
    cortex = _RecordingClient("cortex")
    brainstem_reply = "Brainstem lane is carrying this local-only turn while the primary cortex recovers."
    cpu_reply = "CPU reflex is available, but brainstem should answer this recovered foreground turn."
    brainstem = _RecordingClient(brainstem_reply)
    cpu = _RecordingClient(cpu_reply)
    gate._mlx_client = cortex

    clients = {
        "/models/brainstem": brainstem,
        "/models/fallback": cpu,
    }

    def _fake_get_mlx_client(model_path=None, **kwargs):
        return clients[model_path]

    with replace.object(InferenceGate, "_background_local_deferral_reason", return_value=None):
        with replace("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
            with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
                with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                    result = await gate.generate(
                        "background reflection",
                        context={"origin": "system", "prefer_tier": "primary"},
                    )

    assert result == brainstem_reply
    assert not cortex.deadlines
    assert brainstem.deadlines
    expected_total = InferenceGate._default_timeout_for_request(
        "system",
        "tertiary",
        deep_handoff=False,
        is_background=True,
    )
    expected_primary, _ = InferenceGate._split_attempt_timeouts(expected_total, "tertiary")
    assert brainstem.deadlines[0]._timeout == expected_primary
    expected_tokens = InferenceGate._default_max_tokens_for_request(
        "system",
        "tertiary",
        deep_handoff=False,
        is_background=True,
    )
    assert brainstem.kwargs[0]["max_tokens"] == expected_tokens


def test_routing_user_origin_is_treated_as_human_input():
    assert InferenceGate._origin_is_user_facing("user") is True
    assert InferenceGate._origin_is_user_facing("voice_command") is True
    assert InferenceGate._origin_is_user_facing("routing_user") is True
    assert InferenceGate._origin_is_user_facing("routing_voice_command") is True


def test_user_facing_primary_budget_allows_32b_cold_start():
    total = InferenceGate._default_timeout_for_request(
        "user",
        "primary",
        deep_handoff=False,
        is_background=False,
    )
    primary, fallback = InferenceGate._split_attempt_timeouts(total, "primary")
    # Foreground user chat keeps enough budget for the 32B lane while remaining
    # bounded so the desktop UI cannot hold memory indefinitely.
    assert total == 180.0
    assert primary >= 150.0
    assert fallback >= 20.0


def test_user_facing_secondary_budget_preserves_solver_generation_headroom():
    total = InferenceGate._default_timeout_for_request(
        "user",
        "secondary",
        deep_handoff=True,
        is_background=False,
    )
    primary, fallback = InferenceGate._split_attempt_timeouts(total, "secondary")

    assert total == 210.0
    assert primary >= 180.0
    assert fallback >= 20.0


@pytest.mark.asyncio
async def test_user_facing_reliability_fragments_are_failed_generations():
    gate = InferenceGate()
    client = _RecordingClient("I'm fine")

    text = await gate._generate_with_client(
        client,
        "Are you coherent enough to talk, or is chat broken?",
        "You are Aura.",
        [],
        get_deadline(30.0),
        "PRIMARY",
        origin="user",
        foreground_request=True,
    )

    assert text is None


@pytest.mark.asyncio
async def test_user_facing_presence_check_accepts_concise_grounded_reply():
    gate = InferenceGate()
    client = _RecordingClient("I'm here with you.")

    text = await gate._generate_with_client(
        client,
        "Aaaah, a break. Ok. Aura, are you there?",
        "You are Aura.",
        [],
        get_deadline(30.0),
        "PRIMARY",
        origin="user",
        foreground_request=True,
    )

    assert text == "I'm here with you."


@pytest.mark.asyncio
async def test_user_facing_primary_falls_back_to_brainstem_when_cortex_fails_without_cloud():
    gate = InferenceGate()
    class _FailedNoTextClient(_NoTextClient):
        def get_lane_status(self):
            return {
                "state": "failed",
                "last_error": "worker_failed",
                "conversation_ready": False,
                "warmup_attempted": True,
                "warmup_in_flight": False,
                "last_transition_at": 1.0,
            }

    cortex = _FailedNoTextClient()
    brainstem_reply = "Brainstem lane is carrying this local-only turn while the primary cortex recovers."
    cpu_reply = "CPU reflex is available, but brainstem should answer this recovered foreground turn."
    brainstem = _RecordingClient(brainstem_reply)
    cpu = _RecordingClient(cpu_reply)
    gate._mlx_client = cortex

    clients = {
        "/models/brainstem": brainstem,
        "/models/fallback": cpu,
    }

    def _fake_get_mlx_client(model_path=None, **kwargs):
        return clients[model_path]

    with replace("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "You able to speak?",
                    context={"origin": "user", "prefer_tier": "primary", "allow_cloud_fallback": False},
                )

    assert result == brainstem_reply
    assert brainstem.deadlines
    assert brainstem.kwargs[0]["foreground_request"] is True
    assert not cpu.deadlines


@pytest.mark.asyncio
async def test_removed_remote_provider_is_never_consulted_as_last_resort(monkeypatch):
    gate = InferenceGate()
    gate._mlx_client = _NoTextReadyClient()
    gate._cortex_recovery_in_progress = True
    no_text = _NoTextClient()

    monkeypatch.setattr(asyncio, "sleep", AsyncCallProbe(return_value=None))

    def _fake_get_mlx_client(model_path=None, **kwargs):
        return no_text

    def _removed_provider_service_trap(*args, **kwargs):
        service_name = str(args[-1] if args else "")
        if service_name in {"api_adapter", "llm_router"}:
            raise AssertionError("retired remote-provider services must not be consulted")
        return kwargs.get("default")

    with replace("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                with replace(
                    "core.container.ServiceContainer.get",
                    side_effect=_removed_provider_service_trap,
                ):
                    with replace.object(
                        InferenceGate,
                        "_user_facing_recovery_response",
                        lambda cls, prompt: "local recovery response",
                    ):
                        result = await gate.generate(
                            "Can you answer locally?",
                            context={
                                "origin": "user",
                                "prefer_tier": "primary",
                                "allow_cloud_fallback": False,
                                "allow_mesh_cognition": False,
                            },
                        )

    assert result == "local recovery response"


@pytest.mark.asyncio
async def test_api_adapter_remote_only_request_returns_retired_provider_contract():
    from core.adapters.api_adapter import APIAdapter

    adapter = APIAdapter()

    result = await adapter.generate_with_metadata(
        "remote recovery prompt",
        {
            "model_tier": "api_fast",
            "cloud_only": True,
            "max_tokens": 48,
        },
    )

    assert result["ok"] is False
    assert result["text"] == ""
    assert result["endpoint"] == "APIAdapter-remote-provider-removed"
    assert result["error"] == "remote_model_provider_removed"
    assert result["is_local"] is True
    assert result["fallback_chain"] == []


@pytest.mark.asyncio
async def test_api_adapter_legacy_api_tier_is_resolved_by_local_inference():
    from core.adapters.api_adapter import APIAdapter

    adapter = APIAdapter()
    adapter.has_local = True
    adapter._local_generate = AsyncCallProbe(return_value="local response")

    result = await adapter.generate_with_metadata(
        "answer locally",
        {"model_tier": "api_fast", "max_tokens": 8},
    )

    assert result["ok"] is True
    assert result["text"] == "local response"
    assert result["is_local"] is True
    assert result["provider"] == "local"
    assert result["fallback_chain"][-1]["status"] == "success"


@pytest.mark.asyncio
async def test_health_router_rejects_remote_endpoint_registration():
    from core.brain.llm_health_router import HealthAwareLLMRouter

    router = HealthAwareLLMRouter()
    with pytest.raises(ValueError, match="remote model providers are not supported"):
        router.register(
            name="remote",
            url="https://provider.invalid/v1",
            model="remote-model",
            is_local=False,
            tier="api_fast",
            client=SimpleNamespace(think=AsyncCallProbe(return_value="must not run")),
        )

    assert router.endpoints == {}


def test_conversation_status_is_not_ready_after_timeout_mark():
    gate = InferenceGate()

    class _LaneClient:
        def __init__(self):
            self.reason = ""

        def note_lane_recovering(self, reason):
            self.reason = reason

        def get_lane_status(self):
            return {
                "state": "recovering",
                "last_error": self.reason,
                "conversation_ready": False,
            }

    gate._mlx_client = _LaneClient()
    gate.note_foreground_timeout("foreground_timeout")
    lane = gate.get_conversation_status()

    assert lane["state"] == "recovering"
    assert lane["conversation_ready"] is False
    assert lane["last_failure_reason"] == "foreground_timeout"


def test_conversation_status_respects_ready_lane_even_without_recent_generation():
    gate = InferenceGate()
    gate._last_successful_generation_at = time.time() - 600.0

    class _ReadyLane:
        def get_lane_status(self):
            return {
                "state": "ready",
                "last_error": "",
                "conversation_ready": True,
                "last_ready_at": time.time() - 45.0,
                "last_progress_at": time.time() - 45.0,
                "last_visible_readiness_at": time.time() - 45.0,
                "last_user_facing_completed_at": 0.0,
                "warmup_attempted": True,
                "warmup_in_flight": False,
            }

    gate._mlx_client = _ReadyLane()

    lane = gate.get_conversation_status()

    assert lane["state"] == "ready"
    assert lane["conversation_ready"] is True


def test_conversation_status_rejects_raw_ready_without_visible_conversation_proof():
    gate = InferenceGate()
    gate._last_successful_generation_at = time.time()

    class _HeartbeatOnlyReadyLane:
        def get_lane_status(self):
            return {
                "state": "ready",
                "last_error": "",
                "conversation_ready": True,
                "readiness_blockers": [],
                "last_ready_at": time.time(),
                "last_progress_at": time.time(),
                "warmup_attempted": True,
                "warmup_in_flight": False,
            }

    gate._mlx_client = _HeartbeatOnlyReadyLane()

    lane = gate.get_conversation_status()

    assert lane["state"] == "ready"
    assert lane["conversation_ready"] is False
    assert "visible_conversation_probe_missing" in lane["readiness_blockers"]
    assert lane["last_failure_reason"] == "visible_conversation_probe_missing"


def test_conversation_status_rejects_raw_ready_with_runtime_identity_mismatch():
    gate = InferenceGate()
    gate._last_successful_generation_at = time.time()

    class _MismatchedReadyLane:
        def get_lane_status(self):
            return {
                "state": "ready",
                "last_error": "",
                "conversation_ready": True,
                "readiness_blockers": [],
                "runtime_identity_ok": False,
                "detected_models": ["unrelated/raw-assistant-runtime"],
                "last_ready_at": time.time(),
                "last_progress_at": time.time(),
                "warmup_attempted": True,
                "warmup_in_flight": False,
            }

    gate._mlx_client = _MismatchedReadyLane()

    lane = gate.get_conversation_status()

    assert lane["state"] == "ready"
    assert lane["conversation_ready"] is False
    assert "runtime_identity_mismatch" in lane["readiness_blockers"]


def test_conversation_status_does_not_promote_ready_lane_with_readiness_blockers():
    gate = InferenceGate()
    gate._last_successful_generation_at = time.time()

    class _BlockedReadyLane:
        def get_lane_status(self):
            return {
                "state": "ready",
                "last_error": "",
                "conversation_ready": False,
                "readiness_blockers": ["visible_conversation_probe_missing"],
                "last_ready_at": time.time(),
                "last_progress_at": time.time(),
                "warmup_attempted": True,
                "warmup_in_flight": False,
            }

        def is_alive(self):
            return True

    gate._mlx_client = _BlockedReadyLane()

    lane = gate.get_conversation_status()

    assert lane["state"] == "ready"
    assert lane["conversation_ready"] is False
    assert lane["readiness_blockers"] == ["visible_conversation_probe_missing"]
    assert lane["last_failure_reason"] == "visible_conversation_probe_missing"


@pytest.mark.asyncio
async def test_ensure_foreground_ready_allows_first_visible_turn_to_prove_ready():
    gate = InferenceGate()
    gate._last_successful_generation_at = time.time()

    class _ReadyButUnprovenVisibleLane:
        warmup_calls = 0

        def get_lane_status(self):
            return {
                "state": "ready",
                "last_error": "",
                "conversation_ready": True,
                "readiness_blockers": [],
                "last_ready_at": time.time(),
                "last_progress_at": time.time(),
                "last_visible_readiness_at": 0.0,
                "last_user_facing_completed_at": 0.0,
                "warmup_attempted": True,
                "warmup_in_flight": False,
            }

        def is_alive(self):
            return True

        async def warmup(self):
            self.warmup_calls += 1
            raise AssertionError("already-loaded foreground lane must not re-warm")

    gate._mlx_client = _ReadyButUnprovenVisibleLane()

    lane = await gate.ensure_foreground_ready(timeout=15.0)

    assert lane["state"] == "ready"
    assert lane["conversation_ready"] is False
    assert lane["readiness_blockers"] == ["visible_conversation_probe_missing"]
    assert gate._mlx_client.warmup_calls == 0


@pytest.mark.asyncio
async def test_ensure_foreground_ready_accepts_loaded_lane_after_marker_race(monkeypatch):
    gate = InferenceGate()
    lane = {
        "state": "ready",
        "last_failure_reason": "",
        "conversation_ready": False,
        "readiness_blockers": [],
        "warmup_attempted": True,
        "warmup_in_flight": False,
        "active_generations": 0,
    }

    class _AlreadyLoadedLane:
        warmup_calls = 0

        async def warmup(self):
            self.warmup_calls += 1
            raise AssertionError("loaded lane must not begin a duplicate warmup")

    gate._mlx_client = _AlreadyLoadedLane()
    monkeypatch.setattr(gate, "get_conversation_status", lambda: dict(lane))

    result = await gate.ensure_foreground_ready(timeout=15.0)

    assert result == lane
    assert gate._mlx_client.warmup_calls == 0


@pytest.mark.asyncio
async def test_generate_attempts_ready_lane_that_only_lacks_visible_turn_proof():
    gate = InferenceGate()

    class _ReadyButUnprovenVisibleGeneratingLane(_RecordingClient):
        warmup_calls = 0

        def get_lane_status(self):
            now = time.time()
            return {
                "state": "ready",
                "last_error": "",
                "conversation_ready": True,
                "readiness_blockers": [],
                "last_ready_at": now,
                "last_progress_at": now,
                "last_visible_readiness_at": 0.0,
                "last_user_facing_completed_at": 0.0,
                "warmup_attempted": True,
                "warmup_in_flight": False,
            }

        def is_alive(self):
            return True

        async def warmup(self):
            self.warmup_calls += 1
            raise AssertionError("already-loaded foreground lane must not re-warm")

    client = _ReadyButUnprovenVisibleGeneratingLane(
        "I am serving this visible desktop turn through the loaded Cortex lane."
    )
    gate._mlx_client = client

    result = await gate.generate(
        "In one sentence, confirm the live desktop Cortex lane is serving this visible turn.",
        context={
            "origin": "desktop_quick_user",
            "prefer_tier": "primary",
            "foreground_request": True,
            "protected_foreground_lane": True,
            "allow_cloud_fallback": False,
            "allow_mesh_cognition": False,
            "max_tokens": 80,
        },
        timeout=20.0,
    )

    assert result == "I am serving this visible desktop turn through the loaded Cortex lane."
    assert len(client.kwargs) == 1


@pytest.mark.asyncio
async def test_protected_desktop_generation_keeps_budget_under_existential_threat(monkeypatch):
    from core.container import ServiceContainer

    gate = InferenceGate()
    now = time.time()

    class _ReadyGeneratingLane(_RecordingClient):
        def get_lane_status(self):
            return {
                "state": "ready",
                "last_error": "",
                "conversation_ready": True,
                "readiness_blockers": [],
                "last_ready_at": now,
                "last_progress_at": now,
                "last_visible_readiness_at": now,
                "last_user_facing_completed_at": now,
                "warmup_attempted": True,
                "warmup_in_flight": False,
            }

        def is_alive(self):
            return True

    class _CriticalExistentialStakes:
        def get_existential_threat(self):
            return 1.0

    original_get = ServiceContainer.get

    def _get(name, default=None):
        if name == "existential_stakes":
            return _CriticalExistentialStakes()
        return original_get(name, default)

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(_get))
    client = _ReadyGeneratingLane(
        "I can describe governed desktop, browser, file, memory, terminal, and code tools; "
        "a hypothetical chain stays under Will and Authority approval, then records receipts and "
        "effect checks before I claim anything happened."
    )
    gate._mlx_client = client

    result = await gate.generate(
        "What tools can you use externally, and what governance approves them?",
        context={
            "origin": "desktop_quick_user",
            "prefer_tier": "primary",
            "foreground_request": True,
            "protected_foreground_lane": True,
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "capability_inventory_contract": True,
            "allow_cloud_fallback": False,
            "allow_mesh_cognition": False,
            "max_tokens": 512,
        },
        timeout=20.0,
    )

    assert result
    assert client.kwargs[0]["max_tokens"] == 512


@pytest.mark.asyncio
async def test_protected_capability_inventory_keeps_min_budget_under_resource_envelope(monkeypatch):
    from core.container import ServiceContainer

    gate = InferenceGate()
    now = time.time()

    class _ReadyGeneratingLane(_RecordingClient):
        def get_lane_status(self):
            return {
                "state": "ready",
                "last_error": "",
                "conversation_ready": True,
                "readiness_blockers": [],
                "last_ready_at": now,
                "last_progress_at": now,
                "last_visible_readiness_at": now,
                "last_user_facing_completed_at": now,
                "warmup_attempted": True,
                "warmup_in_flight": False,
            }

        def is_alive(self):
            return True

    class _Envelope:
        allowed = True
        max_tokens = 219
        disabled_capabilities = set()

        def as_dict(self):
            return {
                "allowed": self.allowed,
                "max_tokens": self.max_tokens,
                "disabled_capabilities": [],
            }

    class _ResourceStakes:
        def action_envelope(self, *_args, **_kwargs):
            return _Envelope()

    original_get = ServiceContainer.get

    def _get(name, default=None):
        if name == "resource_stakes":
            return _ResourceStakes()
        return original_get(name, default)

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(_get))
    client = _ReadyGeneratingLane(
        "I can coordinate desktop apps, browser/web research, files and PDFs, "
        "terminal/code work, memory, and repair tools. Consequential actions are "
        "governed by Will and Authority with permission checks; I record receipts "
        "and verify visible effects before claiming completion. A hypothetical chain "
        "would be notes to PDF to web research, and I am not executing tools in this turn."
    )
    gate._mlx_client = client

    result = await gate.generate(
        "What external tools can you use, and give one hypothetical scenario?",
        context={
            "origin": "desktop_quick_user",
            "prefer_tier": "primary",
            "foreground_request": True,
            "protected_foreground_lane": True,
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "capability_inventory_contract": True,
            "allow_cloud_fallback": False,
            "allow_mesh_cognition": False,
            "max_tokens": 384,
        },
        timeout=20.0,
    )

    assert result
    assert client.kwargs[0]["max_tokens"] == 384


@pytest.mark.asyncio
async def test_required_desktop_long_form_floor_overrides_stale_route_cap():
    gate = InferenceGate()
    now = time.time()

    class _ReadyGeneratingLane(_RecordingClient):
        def get_lane_status(self):
            return {
                "state": "ready",
                "last_error": "",
                "conversation_ready": True,
                "readiness_blockers": [],
                "last_ready_at": now,
                "last_progress_at": now,
                "last_visible_readiness_at": now,
                "last_user_facing_completed_at": now,
                "warmup_attempted": True,
                "warmup_in_flight": False,
            }

        def is_alive(self):
            return True

    prompt = (
        "Explain Dijkstra's shortest-path algorithm in one complete response. Include: "
        "(1) the core invariant, (2) numbered pseudocode, (3) a worked example "
        "on vertices A, B, C, D with at least five weighted edges, (4) time "
        "complexity with a binary heap and an array, and (5) a negative-weight "
        "failure and the correct alternative."
    )
    client = _ReadyGeneratingLane(
        "1. The invariant finalizes the unsettled vertex with minimum tentative "
        "distance when weights are nonnegative. 2. Pseudocode initializes dist[s]=0, "
        "extracts the minimum, and relaxes each edge. 3. For A-B=1, A-C=4, B-C=2, "
        "B-D=5, and C-D=1, the distances from A are A=0, B=1, C=3, D=4. "
        "4. A binary heap takes O((V+E) log V); an array takes O(V^2+E). "
        "5. Negative weights invalidate finalization, so use Bellman-Ford."
    )
    gate._mlx_client = client

    result = await gate.generate(
        prompt,
        context={
            "origin": "desktop_quick_user",
            "prefer_tier": "primary",
            "foreground_request": True,
            "protected_foreground_lane": True,
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "allow_cloud_fallback": False,
            "allow_mesh_cognition": False,
            "max_tokens": 1536,
            "user_surface_completion_floor": 2560,
            "user_surface_validation_prompt": prompt,
        },
        timeout=20.0,
    )

    assert result
    assert client.kwargs[0]["max_tokens"] == 2560


def test_note_foreground_timeout_schedules_fast_reprewarm(monkeypatch):
    monkeypatch.setenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "1")
    gate = InferenceGate()
    scheduled = {}

    def _record_schedule(delay=12.0):
        scheduled["delay"] = delay

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: object())
    gate._schedule_background_cortex_prewarm = _record_schedule
    gate.note_foreground_timeout("foreground_timeout")

    assert scheduled["delay"] == 2.0


@pytest.mark.asyncio
async def test_ensure_foreground_ready_warms_cold_lane_once(monkeypatch):
    monkeypatch.setenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "1")
    gate = InferenceGate()
    client = _LaneWarmupClient()
    gate._mlx_client = client

    lane = await gate.ensure_foreground_ready(timeout=10.0)

    client.warmup.assert_awaited_once()
    assert lane["conversation_ready"] is True
    assert lane["state"] == "ready"


@pytest.mark.asyncio
async def test_ensure_foreground_ready_rearms_runtime_failed_lane_before_warmup(monkeypatch):
    monkeypatch.setenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "1")
    gate = InferenceGate()
    client = _RecoverableFailedLaneClient()
    gate._mlx_client = client

    lane = await gate.ensure_foreground_ready(timeout=10.0)

    client.refresh_runtime_availability.assert_called_once_with(force_probe=True)
    client.warmup.assert_awaited_once()
    assert lane["conversation_ready"] is True
    assert lane["state"] == "ready"


@pytest.mark.asyncio
async def test_think_wraps_system_prompt_as_passthrough_messages():
    gate = InferenceGate()
    gate.generate = AsyncCallProbe(return_value="ok")

    result = await gate.think(
        "Hello there",
        system_prompt="Stay direct.",
        origin="user",
        max_tokens=42,
    )

    assert result == "ok"
    gate.generate.assert_awaited_once()
    context = gate.generate.await_args.kwargs["context"]
    assert context["messages"] == [
        {"role": "system", "content": "Stay direct."},
        {"role": "user", "content": "Hello there"},
    ]
    assert context["origin"] == "user"
    assert context["max_tokens"] == 42


@pytest.mark.asyncio
async def test_think_allows_explicit_brief_mode():
    gate = InferenceGate()
    gate.generate = AsyncCallProbe(return_value="ok")

    await gate.think(
        "Hello there",
        system_prompt="legacy brief",
        system_prompt_is_brief=True,
        origin="user",
    )

    context = gate.generate.await_args.kwargs["context"]
    assert context["brief"] == "legacy brief"
    assert "messages" not in context


@pytest.mark.asyncio
async def test_think_forwards_explicit_timeout_to_generate():
    gate = InferenceGate()
    gate.generate = AsyncCallProbe(return_value="hello")

    result = await gate.think(
        "With me?",
        system_prompt="Be helpful",
        origin="api",
        prefer_tier="primary",
        timeout=67.0,
    )

    assert result == "hello"
    gate.generate.assert_awaited_once()
    assert gate.generate.await_args.kwargs["timeout"] == 67.0


@pytest.mark.asyncio
async def test_think_forwards_user_surface_validation_prompt_to_generate():
    gate = InferenceGate()
    gate.generate = AsyncCallProbe(return_value="hello")

    await gate.think(
        "With me?",
        system_prompt="Speak as Aura.",
        origin="desktop_quick_user",
        prefer_tier="primary",
        clean_user_surface_contract=True,
        user_surface_completion_floor=512,
        user_surface_validation_prompt="With me?",
        runtime_fact_status_contract=True,
        grounded_runtime_status_contract=True,
        live_mind_controls_bound=True,
        live_mind_generation_controls={"temperature": 0.58},
        live_mind_snapshot_ready=True,
        live_mind_required_subsystems_ok=True,
    )

    context = gate.generate.await_args.kwargs["context"]
    assert context["clean_user_surface_contract"] is True
    assert context["user_surface_completion_floor"] == 512
    assert context["user_surface_validation_prompt"] == "With me?"
    assert context["runtime_fact_status_contract"] is True
    assert context["grounded_runtime_status_contract"] is True
    assert context["live_mind_controls_bound"] is True
    assert context["live_mind_generation_controls"] == {"temperature": 0.58}
    assert context["live_mind_snapshot_ready"] is True
    assert context["live_mind_required_subsystems_ok"] is True


@pytest.mark.asyncio
async def test_inference_gate_exposes_local_surface_control_receipt():
    gate = InferenceGate()
    client = _ReceiptRecordingClient(
        "I am tracking this live desktop turn through the governed Cortex lane."
    )
    gate._mlx_client = client

    result = await gate.generate(
        "What are you tracking?",
        context={
            "origin": "desktop_quick_user",
            "prefer_tier": "primary",
            "foreground_request": True,
            "protected_foreground_lane": True,
            "allow_mesh_cognition": False,
            "clean_user_surface_contract": True,
            "user_surface_validation_prompt": "What are you tracking?",
            "semantic_completion_contract": True,
            "user_surface_continuation_contract": True,
            "user_surface_continuation_partial": "The answer already established the invariant.",
            "clean_user_surface_recurrent_loops": 2,
            "clean_user_surface_steering_alpha": 0.31,
            "live_mind_controls_bound": True,
            "allow_cloud_fallback": False,
            "max_tokens": 160,
        },
        timeout=20.0,
    )

    assert result
    metadata = gate.get_last_generation_metadata()
    receipt = gate.get_last_surface_control_receipt()
    assert metadata["surface_control_receipt"]["applied"] is True
    assert receipt["surface_validation_prompt_present"] is True
    assert client.kwargs[0]["user_surface_validation_prompt"] == "What are you tracking?"
    assert client.kwargs[0]["semantic_completion_contract"] is True
    assert client.kwargs[0]["user_surface_continuation_contract"] is True
    assert client.kwargs[0]["user_surface_continuation_partial"] == (
        "The answer already established the invariant."
    )
    assert client.kwargs[0]["live_mind_controls_bound"] is True


@pytest.mark.asyncio
async def test_inference_gate_generation_receipts_are_task_scoped():
    gate = InferenceGate()

    async def _record(label: str) -> tuple[dict, dict]:
        gate._clear_last_generation_metadata()
        await asyncio.sleep(0)
        gate._record_client_generation_metadata(
            None,
            label=label,
            success=True,
            text=label,
        )
        await asyncio.sleep(0)
        return (
            gate.get_last_generation_metadata(),
            gate.get_last_surface_control_receipt(),
        )

    first, second = await asyncio.gather(_record("request-A"), _record("request-B"))

    assert first[0]["endpoint"] == "request-A"
    assert second[0]["endpoint"] == "request-B"
    assert first[1] == {}
    assert second[1] == {}


@pytest.mark.asyncio
async def test_inference_gate_fresh_task_never_borrows_global_receipt():
    gate = InferenceGate()
    published = asyncio.Event()

    async def _writer() -> None:
        gate._record_client_generation_metadata(
            None,
            label="writer-request",
            success=True,
            text="writer",
        )
        published.set()

    async def _reader() -> tuple[dict, dict]:
        await published.wait()
        return (
            gate.get_last_generation_metadata(),
            gate.get_last_surface_control_receipt(),
        )

    _, reader_receipts = await asyncio.gather(_writer(), _reader())

    assert reader_receipts == ({}, {})
    assert gate.get_diagnostic_last_generation_metadata()["endpoint"] == (
        "writer-request"
    )


def test_inference_gate_preserves_failed_surface_receipt_as_semantic_rejection():
    gate = InferenceGate()

    class _Client:
        @staticmethod
        def get_last_surface_control_receipt():
            return {
                "surface_quality_gate_enabled": True,
                "surface_quality_gate_passed": False,
                "surface_quality_gate_attempts": 3,
                "surface_quality_gate_reasons": ["missing_requested_word_count"],
                "requested_output_contract": {
                    "kind": "word_count",
                    "word_min": 5,
                    "word_max": 5,
                },
            }

    gate._record_client_generation_metadata(
        _Client(),
        label="Cortex",
        success=False,
        text="",
    )

    metadata = gate.get_last_generation_metadata()
    assert metadata["error"] == "surface_quality_rejected"
    assert metadata["failure_reasons"] == ["missing_requested_word_count"]
    assert metadata["surface_control_receipt"]["surface_quality_gate_attempts"] == 3


@pytest.mark.asyncio
async def test_inference_gate_receives_quality_rejection_across_wait_for_task_boundary():
    gate = InferenceGate()

    class _Client:
        async def generate_text_async(self, **kwargs):
            sink = kwargs["_generation_result_sink"]
            await asyncio.sleep(0)
            sink["surface_control_receipt"] = {
                "surface_quality_gate_enabled": True,
                "surface_quality_gate_passed": False,
                "surface_quality_gate_reasons": ["corrupted_language"],
                "surface_quality_rejected_text": "valid draft held for review",
            }
            return None

        @staticmethod
        def get_last_surface_control_receipt():
            # This is exactly what the parent task sees from a ContextVar.
            return {}

    result = await gate._generate_with_client(
        _Client(),
        "Explain Dijkstra completely.",
        "",
        [],
        get_deadline(5.0),
        "Cortex",
        foreground_request=True,
    )

    assert result is None
    metadata = gate.get_last_generation_metadata()
    assert metadata["error"] == "surface_quality_rejected"
    assert metadata["failure_reasons"] == ["corrupted_language"]
    assert gate.get_last_surface_control_receipt()[
        "surface_quality_rejected_text"
    ] == "valid draft held for review"


def test_inference_gate_records_stabilization_without_prior_provider_receipt():
    gate = InferenceGate()
    gate._clear_last_generation_metadata()

    result = gate._stabilize_user_facing_text(
        "no",
        'Reply exactly: "yes"',
        is_user_facing=True,
    )

    assert result == "yes"
    metadata = gate.get_last_generation_metadata()
    receipt = gate.get_last_surface_control_receipt()
    assert metadata["endpoint"] == "unattributed-response-path"
    assert receipt["text_mutation_count"] == 1
    assert receipt["text_mutations"][0]["stage"] == (
        "inference_gate.post_generation_stabilization"
    )


@pytest.mark.asyncio
async def test_think_forwards_purpose_for_originless_expression_calls():
    gate = InferenceGate()
    gate.generate = AsyncCallProbe(return_value="hello")

    await gate.think(
        "Hello there",
        system_prompt="Speak as Aura.",
        purpose="expression",
    )

    context = gate.generate.await_args.kwargs["context"]
    assert context["purpose"] == "expression"
    assert context["messages"] == [
        {"role": "system", "content": "Speak as Aura."},
        {"role": "user", "content": "Hello there"},
    ]


@pytest.mark.asyncio
async def test_initialize_defers_eager_warmup_when_explicitly_disabled():
    gate = InferenceGate()
    client = CallProbe()
    client.warmup = AsyncCallProbe()

    with replace.dict(
        os.environ,
        {"AURA_EAGER_CORTEX_WARMUP": "0", "AURA_SAFE_BOOT_DESKTOP": "0"},
        clear=False,
    ):
        with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=client):
            with replace("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                with replace("core.brain.llm.model_registry.ACTIVE_MODEL", "ACTIVE"):
                    await gate.initialize()

    client.warmup.assert_not_awaited()
    assert gate._initialized is True
    if gate._maintenance_task:
        gate._maintenance_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await gate._maintenance_task


@pytest.mark.asyncio
async def test_initialize_auto_warms_on_high_memory_desktop():
    gate = InferenceGate()
    client = CallProbe()
    client.warmup = AsyncCallProbe()
    vm = CallProbe(total=64 * 1024 ** 3, available=40 * 1024 ** 3, percent=37.0)

    with replace.dict(
        os.environ,
        {"AURA_EAGER_CORTEX_WARMUP": "auto", "AURA_SAFE_BOOT_DESKTOP": "0"},
        clear=False,
    ):
        with replace("core.brain.inference_gate.psutil.virtual_memory", return_value=vm):
            with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=client):
                with replace("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                    with replace("core.brain.llm.model_registry.ACTIVE_MODEL", "ACTIVE"):
                        await gate.initialize()

    client.warmup.assert_awaited_once()
    assert gate._prewarm_task is not None
    if gate._maintenance_task:
        gate._maintenance_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await gate._maintenance_task


@pytest.mark.asyncio
async def test_initialize_allows_opt_in_eager_warmup():
    gate = InferenceGate()
    client = CallProbe()
    client.warmup = AsyncCallProbe()
    vm = CallProbe(total=64 * 1024 ** 3, available=42 * 1024 ** 3, percent=34.0)

    with replace.dict(
        os.environ,
        {"AURA_EAGER_CORTEX_WARMUP": "1", "AURA_SAFE_BOOT_DESKTOP": "0"},
        clear=False,
    ):
        with replace("core.brain.inference_gate.psutil.virtual_memory", return_value=vm):
            with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=client):
                with replace("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                    with replace("core.brain.llm.model_registry.ACTIVE_MODEL", "ACTIVE"):
                        await gate.initialize()

    client.warmup.assert_awaited_once()
    if gate._maintenance_task:
        gate._maintenance_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await gate._maintenance_task


@pytest.mark.asyncio
async def test_initialize_starts_inference_maintenance_loop():
    gate = InferenceGate()
    client = CallProbe()
    client.warmup = AsyncCallProbe()

    with replace.dict(os.environ, {"AURA_EAGER_CORTEX_WARMUP": "0"}, clear=False):
        with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=client):
            with replace("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                with replace("core.brain.llm.model_registry.ACTIVE_MODEL", "ACTIVE"):
                    await gate.initialize()

    assert gate._maintenance_task is not None
    assert not gate._maintenance_task.done()
    gate._maintenance_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await gate._maintenance_task


@pytest.mark.asyncio
async def test_background_requests_defer_under_memory_pressure_when_cortex_is_ready():
    gate = InferenceGate()
    gate._mlx_client = _LaneWarmupClient()
    gate._mlx_client.state = "ready"
    gate._ensure_cortex_recovery = AsyncCallProbe()

    with replace.object(InferenceGate, "_background_memory_pressure_active", return_value=True):
        with replace.object(
            InferenceGate,
            "get_conversation_status",
            return_value={
                "conversation_ready": True,
                "state": "ready",
                "warmup_in_flight": False,
            },
        ):
            result = await gate.generate(
                "background reflection",
                context={"prefer_tier": "primary", "origin": "system"},
            )

    assert result is None
    gate._ensure_cortex_recovery.assert_not_awaited()


@pytest.mark.asyncio
async def test_background_requests_defer_when_foreground_headroom_is_reserved():
    gate = InferenceGate()
    gate._mlx_client = _LaneWarmupClient()
    gate._ensure_cortex_recovery = AsyncCallProbe()

    with replace.object(InferenceGate, "_foreground_headroom_reserved", return_value=True):
        result = await gate.generate(
            "background reflection",
            context={"prefer_tier": "primary", "origin": "system"},
        )

    assert result is None
    gate._ensure_cortex_recovery.assert_not_awaited()


@pytest.mark.asyncio
async def test_foreground_admission_sheds_background_workers_before_retry():
    gate = InferenceGate()
    gate._shed_background_workers_for_memory_pressure = AsyncCallProbe()

    with replace.object(
        gate,
        "_headroom_snapshot",
        side_effect=[
            {
                "tier": "primary",
                "pressure_pct": 92.0,
                "available_gb": 7.0,
                "max_pressure_pct": 88.0,
                "min_available_gb": 12.0,
                "can_admit": False,
            },
            {
                "tier": "primary",
                "pressure_pct": 80.0,
                "available_gb": 16.0,
                "max_pressure_pct": 88.0,
                "min_available_gb": 12.0,
                "can_admit": True,
            },
        ],
    ):
        with replace("core.brain.inference_gate.gc.collect") as gc_collect:
            snapshot = await gate._enforce_foreground_admission("primary", protected_foreground=False)

    assert snapshot["can_admit"] is True
    gate._shed_background_workers_for_memory_pressure.assert_awaited_once()
    gc_collect.assert_called_once()


def test_cleanup_closes_primary_and_registered_local_clients_once():
    gate = InferenceGate()
    primary = SimpleNamespace(close=CallProbe())
    registered = SimpleNamespace(close=CallProbe())
    duplicate = primary
    gate._mlx_client = primary
    gate._initialized = True
    prewarm_task = TaskProbe(done=False)
    gate._prewarm_task = prewarm_task
    gate._deferred_prewarm_task = None
    gate._maintenance_task = None

    with replace.object(
        gate,
        "_iter_local_clients",
        return_value={"/models/primary": duplicate, "/models/secondary": registered},
    ):
        gate.cleanup()

    primary.close.assert_called_once()
    registered.close.assert_called_once()
    prewarm_task.cancel.assert_called_once()
    assert gate._prewarm_task is None
    assert gate._mlx_client is None
    assert gate._initialized is False


@pytest.mark.asyncio
async def test_recycle_idle_local_clients_reboots_fragmented_spare():
    gate = InferenceGate()
    spare = SimpleNamespace(
        should_recycle_for_fragmentation=CallProbe(return_value=True),
        reboot_worker=AsyncCallProbe(),
    )

    with replace.object(gate, "_iter_local_clients", return_value={"/models/brainstem": spare}):
        await gate._recycle_idle_local_clients()

    spare.reboot_worker.assert_awaited_once_with(
        reason="scheduled_fragmentation_recycle",
        mark_failed=False,
    )


@pytest.mark.asyncio
async def test_solver_hot_spare_stays_deferred_while_cortex_is_ready():
    gate = InferenceGate()
    solver = SimpleNamespace(
        is_alive=CallProbe(return_value=False),
        warmup=AsyncCallProbe(),
    )

    with replace.object(
        gate,
        "get_conversation_status",
        return_value={
            "conversation_ready": True,
            "state": "ready",
            "warmup_in_flight": False,
        },
    ):
        with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=solver):
            with replace("core.brain.llm.model_registry.get_deep_model_path", return_value="/models/deep"):
                result = await gate._ensure_hot_spare_ready("Solver")

    assert result is False
    solver.warmup.assert_not_awaited()


@pytest.mark.asyncio
async def test_solver_hot_spare_warmup_uses_background_semantics():
    gate = InferenceGate()
    solver = SimpleNamespace(
        is_alive=CallProbe(side_effect=[False, True]),
        warmup=AsyncCallProbe(side_effect=lambda **_kwargs: None),
    )

    with replace.object(
        gate,
        "get_conversation_status",
        return_value={
            "conversation_ready": False,
            "state": "cold",
            "warmup_in_flight": False,
        },
    ):
        with replace.object(gate, "_background_local_deferral_reason", return_value=None):
            with replace.object(
                gate,
                "_headroom_snapshot",
                return_value={
                    "tier": "secondary",
                    "pressure_pct": 52.0,
                    "available_gb": 26.0,
                    "max_pressure_pct": 84.0,
                    "min_available_gb": 16.0,
                    "can_admit": True,
                },
            ):
                with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=solver):
                    with replace("core.brain.llm.model_registry.get_deep_model_path", return_value="/models/deep"):
                        result = await gate._ensure_hot_spare_ready("Solver")

    assert result is True
    solver.warmup.assert_awaited_once_with(foreground_request=False)


def _memory_snapshot(
    *,
    total_gb: float = 64.0,
    available_gb: float = 24.0,
    pressure_pct: float = 62.0,
    process_rss_gb: float = 3.0,
    process_rss_limit_gb: float = 38.0,
    refuse_heavy_local_generation: bool = False,
):
    return SimpleNamespace(
        total_gb=total_gb,
        available_gb=available_gb,
        pressure_pct=pressure_pct,
        process_rss_gb=process_rss_gb,
        process_rss_limit_gb=process_rss_limit_gb,
        refuse_heavy_local_generation=refuse_heavy_local_generation,
    )


def test_headroom_snapshot_blocks_secondary_on_64gb_without_large_free_headroom(monkeypatch):
    import core.utils.memory_monitor as memory_monitor

    monkeypatch.delenv("AURA_FOREGROUND_SECONDARY_MAX_PRESSURE_PCT", raising=False)
    monkeypatch.delenv("AURA_FOREGROUND_SECONDARY_MIN_AVAILABLE_GB", raising=False)
    monkeypatch.setattr(
        memory_monitor,
        "get_memory_pressure_snapshot",
        lambda: _memory_snapshot(
            total_gb=64.0,
            available_gb=48.0,
            pressure_pct=25.0,
            process_rss_gb=4.0,
            process_rss_limit_gb=38.0,
        ),
    )

    snapshot = InferenceGate._headroom_snapshot("secondary")

    assert snapshot["can_admit"] is False
    assert snapshot["min_available_gb"] == pytest.approx(52.0)
    assert "memory_pressure:25.0%/48.0GB" in snapshot["reason"]


def test_headroom_snapshot_blocks_primary_when_process_tree_exceeds_limit(monkeypatch):
    import core.utils.memory_monitor as memory_monitor

    monkeypatch.setattr(
        memory_monitor,
        "get_memory_pressure_snapshot",
        lambda: _memory_snapshot(
            total_gb=64.0,
            available_gb=26.0,
            pressure_pct=59.0,
            process_rss_gb=39.5,
            process_rss_limit_gb=38.0,
            refuse_heavy_local_generation=True,
        ),
    )

    snapshot = InferenceGate._headroom_snapshot("primary")

    assert snapshot["can_admit"] is False
    assert snapshot["process_rss_gb"] == pytest.approx(39.5)
    assert "process_tree_rss:39.5GB/38.0GB" in snapshot["reason"]


@pytest.mark.asyncio
async def test_secondary_requests_downgrade_to_primary_when_headroom_is_tight():
    # The local deep solver is auto-disabled on <96GB hosts (memory-
    # class policy). Force-enable so the tier logic under test is
    # actually exercised regardless of the machine running the suite.
    os.environ["AURA_ENABLE_LOCAL_DEEP_SOLVER"] = "1"
    try:

        gate = InferenceGate()
        cortex_reply = "Cortex lane handled the audit after headroom forced the deep solver request back to primary."
        solver_reply = "Solver should not run when foreground headroom is too tight for the deep handoff."
        cortex = _RecordingClient(cortex_reply)
        solver = _RecordingClient(solver_reply)
        brainstem = _FakeClient("brainstem")
        gate._mlx_client = cortex
        gate._restore_primary_after_deep_handoff = AsyncCallProbe()

        def _fake_get_mlx_client(model_path=None, **kwargs):
            if model_path == "/models/deep":
                return solver
            if model_path == "/models/brainstem":
                return brainstem
            raise AssertionError(f"Unexpected model path: {model_path}")

        with replace.object(gate, "_local_deep_solver_block_reason", return_value=None):
            with replace.object(
                gate,
                "_enforce_foreground_admission",
                side_effect=[
                    {
                        "can_admit": False,
                        "pressure_pct": 91.0,
                        "available_gb": 8.0,
                    },
                    {
                        "can_admit": True,
                        "pressure_pct": 81.0,
                        "available_gb": 18.0,
                    },
                ],
            ):
                with replace("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
                    with replace("core.brain.llm.model_registry.get_deep_model_path", return_value="/models/deep"):
                        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
                            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                                result = await gate.generate(
                                    "Do a deep architecture audit.",
                                    context={"origin": "user", "prefer_tier": "secondary", "deep_handoff": True},
                                )

        assert result == cortex_reply
        assert cortex.deadlines
        assert not solver.deadlines
        gate._restore_primary_after_deep_handoff.assert_not_awaited()
    finally:
        os.environ.pop("AURA_ENABLE_LOCAL_DEEP_SOLVER", None)


@pytest.mark.asyncio
async def test_secondary_request_fails_safe_to_primary_when_coexistence_probe_errors():
    os.environ["AURA_ENABLE_LOCAL_DEEP_SOLVER"] = "1"
    try:
        gate = InferenceGate()
        cortex_reply = "Cortex handled the request after the coexistence probe failed closed."
        cortex = _RecordingClient(cortex_reply)
        solver = _RecordingClient("Solver must not run without a safe coexistence decision.")
        brainstem = _FakeClient("brainstem")
        gate._mlx_client = cortex

        def _fake_get_mlx_client(model_path=None, **kwargs):
            if model_path == "/models/deep":
                return solver
            if model_path == "/models/brainstem":
                return brainstem
            raise AssertionError(f"Unexpected model path: {model_path}")

        with replace.object(gate, "_local_deep_solver_block_reason", return_value=None):
            with replace.object(
                gate,
                "get_conversation_status",
                side_effect=RuntimeError("lane telemetry unavailable"),
            ):
                with replace.object(
                    gate,
                    "_enforce_foreground_admission",
                    return_value={
                        "can_admit": True,
                        "pressure_pct": 40.0,
                        "available_gb": 32.0,
                    },
                ):
                    with replace(
                        "core.brain.llm.mlx_client.get_mlx_client",
                        side_effect=_fake_get_mlx_client,
                    ):
                        with replace(
                            "core.brain.llm.model_registry.get_deep_model_path",
                            return_value="/models/deep",
                        ):
                            with replace(
                                "core.brain.llm.model_registry.get_brainstem_path",
                                return_value="/models/brainstem",
                            ):
                                result = await gate.generate(
                                    "Analyze this architecture deeply.",
                                    context={
                                        "origin": "user",
                                        "prefer_tier": "secondary",
                                        "deep_handoff": True,
                                    },
                                )

        assert result == cortex_reply
        assert cortex.deadlines
        assert not solver.deadlines
    finally:
        os.environ.pop("AURA_ENABLE_LOCAL_DEEP_SOLVER", None)


def test_secondary_headroom_snapshot_blocks_64gb_solver_envelope_by_default(monkeypatch):
    monkeypatch.delenv("AURA_FOREGROUND_SECONDARY_MAX_PRESSURE_PCT", raising=False)
    monkeypatch.delenv("AURA_FOREGROUND_SECONDARY_MIN_AVAILABLE_GB", raising=False)
    monkeypatch.setattr(
        "core.brain.inference_gate.psutil.virtual_memory",
        lambda: SimpleNamespace(
            percent=77.5,
            total=64 * 1024 ** 3,
            available=int(14.4 * 1024 ** 3),
            used=int((64.0 - 14.4) * 1024 ** 3),
        ),
    )

    snapshot = InferenceGate._headroom_snapshot("secondary")

    assert snapshot["max_pressure_pct"] == 42.0
    assert snapshot["min_available_gb"] == 52.0
    assert snapshot["can_admit"] is False
    assert "memory_pressure" in snapshot["reason"]


def test_foreground_headroom_probe_failure_is_not_admitted_without_override(monkeypatch):
    monkeypatch.delenv("AURA_FORCE_FOREGROUND_HEADROOM_ON_PROBE_FAILURE", raising=False)
    memory_probe = CallProbe(side_effect=OSError("sysctl unavailable"))
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        memory_probe,
    )

    snapshot = InferenceGate._headroom_snapshot("secondary")

    assert snapshot["can_admit"] is False
    assert snapshot["reason"] == "memory_probe_failed"
    assert memory_probe.calls


def test_foreground_headroom_probe_failure_requires_explicit_override(monkeypatch):
    monkeypatch.setenv("AURA_FORCE_FOREGROUND_HEADROOM_ON_PROBE_FAILURE", "1")
    memory_probe = CallProbe(side_effect=OSError("sysctl unavailable"))
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        memory_probe,
    )

    snapshot = InferenceGate._headroom_snapshot("secondary")

    assert snapshot["can_admit"] is True
    # The forced admission must stay on the record as an unmeasured override,
    # never disguised as a clean measured admission.
    assert snapshot["reason"] == "memory_probe_failed_forced_override"
    assert snapshot["measured"] is False
    assert memory_probe.calls


def test_cortex_cold_warmup_requires_real_available_memory(monkeypatch):
    monkeypatch.delenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", raising=False)
    monkeypatch.delenv("AURA_CORTEX_COLD_WARMUP_MIN_AVAILABLE_GB", raising=False)
    monkeypatch.setattr(
        "core.brain.inference_gate.psutil.virtual_memory",
        lambda: SimpleNamespace(
            percent=55.0,
            total=64 * 1024 ** 3,
            available=int(15.0 * 1024 ** 3),
        ),
    )

    snapshot = InferenceGate._cortex_warmup_admission_snapshot("background")

    assert snapshot["can_admit"] is False
    assert snapshot["min_available_gb"] == 26.0
    assert "memory_pressure" in snapshot["reason"]


def test_foreground_cortex_warmup_admits_live_desktop_headroom(monkeypatch):
    monkeypatch.delenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", raising=False)
    monkeypatch.delenv("AURA_CORTEX_FOREGROUND_WARMUP_MIN_AVAILABLE_GB", raising=False)
    monkeypatch.delenv("AURA_CORTEX_COLD_WARMUP_MIN_AVAILABLE_GB", raising=False)
    monkeypatch.setattr(
        "core.brain.inference_gate.psutil.virtual_memory",
        lambda: SimpleNamespace(
            percent=63.7,
            total=64 * 1024 ** 3,
            available=int(23.2 * 1024 ** 3),
        ),
    )

    snapshot = InferenceGate._cortex_warmup_admission_snapshot("foreground")

    assert snapshot["can_admit"] is True
    assert snapshot["min_available_gb"] == 20.0
    assert snapshot["reason"] == ""


@pytest.mark.asyncio
async def test_cortex_recovery_does_not_spawn_under_memory_pressure(monkeypatch):
    gate = InferenceGate()
    client = _LaneWarmupClient()
    client.is_alive = CallProbe(return_value=False)
    gate._mlx_client = client
    monkeypatch.setattr(
        "core.brain.inference_gate.psutil.virtual_memory",
        lambda: SimpleNamespace(
            percent=88.0,
            total=64 * 1024 ** 3,
            available=int(7.0 * 1024 ** 3),
        ),
    )
    monkeypatch.setattr(InferenceGate, "_foreground_user_turn_active", staticmethod(lambda: False))
    monkeypatch.setattr(InferenceGate, "_foreground_owner_active", staticmethod(lambda: False))

    await gate._ensure_cortex_recovery()

    client.warmup.assert_not_awaited()
    assert gate._cortex_recovery_in_progress is False


@pytest.mark.asyncio
async def test_cortex_recovery_skips_primary_spawn_during_nonprimary_proof_lane(monkeypatch):
    gate = InferenceGate()
    client = _LaneWarmupClient()
    client.is_alive = CallProbe(return_value=False)
    gate._mlx_client = client
    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    monkeypatch.setenv("AURA_PROOF_MODEL_TIER", "tertiary")
    monkeypatch.setattr(InferenceGate, "_foreground_user_turn_active", staticmethod(lambda: False))
    monkeypatch.setattr(InferenceGate, "_foreground_owner_active", staticmethod(lambda: False))

    await gate._ensure_cortex_recovery()

    client.warmup.assert_not_awaited()
    assert gate._cortex_recovery_in_progress is False
    assert gate._cortex_recovery_attempts == 0


@pytest.mark.asyncio
async def test_cortex_recovery_does_not_report_deferred_warmup_as_ready(monkeypatch):
    gate = InferenceGate()
    # Cold-start recovery is gated by _boot_should_schedule_deferred_prewarm,
    # which consults REAL host memory through the warmup admission snapshot.
    # These tests are about what happens once the policy says yes, so they pin
    # the policy instead of inheriting the operator's env and whatever the host
    # happens to have free — the reason they passed alone and failed in a long
    # run, where the resident 32B is holding 20GB by the time they execute.
    monkeypatch.setenv("AURA_DEFERRED_CORTEX_PREWARM", "1")
    monkeypatch.setenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "1")
    client = _LaneWarmupClient()

    async def _defer_warmup():
        client.state = "recovering"
        client.last_error = "runtime_shutdown"
        return False

    client.warmup = AsyncCallProbe(side_effect=_defer_warmup)
    gate._mlx_client = client
    monkeypatch.setattr(InferenceGate, "_foreground_user_turn_active", staticmethod(lambda: False))
    monkeypatch.setattr(InferenceGate, "_foreground_owner_active", staticmethod(lambda: False))
    monkeypatch.setattr(gate, "_cortex_warmup_deferral_reason", lambda _context: None)
    monkeypatch.setattr("core.brain.inference_gate.is_shutdown_requested", lambda: False)

    await gate._ensure_cortex_recovery()
    for _ in range(20):
        if not gate._cortex_recovery_in_progress and client.warmup.calls:
            break
        await asyncio.sleep(0.01)

    client.warmup.assert_awaited_once()
    assert client.state == "recovering"
    assert gate._cortex_recovery_attempts == 1
    assert gate._cortex_recovery_in_progress is False


def test_foreground_ready_blocks_cold_cortex_spawn_under_pressure(monkeypatch):
    async def scenario():
        gate = InferenceGate()
        client = _LaneWarmupClient()
        gate._mlx_client = client
        gate._shed_background_workers_for_memory_pressure = AsyncCallProbe()
        monkeypatch.setattr(
            "core.brain.inference_gate.psutil.virtual_memory",
            lambda: SimpleNamespace(
                percent=83.0,
                total=64 * 1024 ** 3,
                available=int(10.0 * 1024 ** 3),
            ),
        )

        with pytest.raises(RuntimeError, match="foreground_warmup_deferred:memory_pressure"):
            await gate.ensure_foreground_ready(timeout=15.0)

        client.warmup.assert_not_awaited()
        assert client.state == "recovering"

    asyncio.run(scenario())


def test_cortex_warmup_probe_failure_is_not_admitted_without_override(monkeypatch):
    monkeypatch.delenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", raising=False)
    memory_probe = CallProbe(side_effect=OSError("sysctl unavailable"))
    monkeypatch.setattr("core.brain.inference_gate.psutil.virtual_memory", memory_probe)

    snapshot = InferenceGate._cortex_warmup_admission_snapshot("foreground")

    assert snapshot["can_admit"] is False
    assert snapshot["reason"] == "memory_probe_failed"
    memory_probe.assert_called_once()


def test_eager_cortex_warmup_fails_closed_when_policy_probe_raises(monkeypatch):
    monkeypatch.setenv("AURA_EAGER_CORTEX_WARMUP", "auto")
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: False))
    monkeypatch.setattr(
        InferenceGate,
        "_cortex_warmup_admission_snapshot",
        staticmethod(lambda _context: (_ for _ in ()).throw(RuntimeError("probe unavailable"))),
    )

    assert InferenceGate._boot_should_eager_warmup() is False


def test_background_memory_pressure_probe_failure_defers_background_inference(monkeypatch):
    memory_probe = CallProbe(side_effect=OSError("vm statistics unavailable"))
    monkeypatch.setattr("core.brain.inference_gate.psutil.virtual_memory", memory_probe)

    assert InferenceGate._background_memory_pressure_active() is True
    memory_probe.assert_called_once()


@pytest.mark.asyncio
async def test_foreground_ready_blocks_cold_cortex_when_memory_probe_fails(monkeypatch):
    monkeypatch.delenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", raising=False)
    gate = InferenceGate()
    client = _LaneWarmupClient()
    gate._mlx_client = client
    memory_probe = CallProbe(side_effect=OSError("sysctl unavailable"))
    monkeypatch.setattr("core.brain.inference_gate.psutil.virtual_memory", memory_probe)

    with pytest.raises(RuntimeError, match="foreground_warmup_deferred:memory_probe_failed"):
        await gate.ensure_foreground_ready(timeout=15.0)

    client.warmup.assert_not_awaited()
    assert client.state == "recovering"
    assert client.last_error == "foreground_warmup_deferred_memory_pressure"
    # Probed, and bounded — not an exact count.
    #
    # This patches the shared psutil module attribute, so it counts every
    # virtual_memory() call in the PROCESS during the check, not the gate's.
    # It read 2 when written and reads 19 now because other subsystems probe in
    # the same window, which says nothing about the property under test: that a
    # failing probe defers the cold Cortex load rather than admitting it. That
    # is asserted above by the raise and the deferral reason.
    #
    # Bounded rather than free, because an unbounded retry loop around a failing
    # syscall is worth catching.
    assert 1 <= len(memory_probe.calls) <= 64, len(memory_probe.calls)


def test_desktop_safe_boot_skips_deferred_cortex_prewarm(monkeypatch):
    monkeypatch.delenv("AURA_DEFERRED_CORTEX_PREWARM", raising=False)
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))

    assert InferenceGate._boot_should_schedule_deferred_prewarm() is False


def test_desktop_safe_boot_respects_deferred_cortex_prewarm_opt_out(monkeypatch):
    monkeypatch.setenv("AURA_DEFERRED_CORTEX_PREWARM", "0")
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))

    assert InferenceGate._boot_should_schedule_deferred_prewarm() is False


@pytest.mark.asyncio
async def test_cold_start_recovery_respects_deferred_cortex_prewarm_opt_out(monkeypatch):
    monkeypatch.setenv("AURA_DEFERRED_CORTEX_PREWARM", "0")
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))

    gate = InferenceGate()
    client = _LaneWarmupClient()
    gate._mlx_client = client

    await gate._ensure_cortex_recovery()

    client.warmup.assert_not_awaited()
    assert gate._cortex_recovery_attempts == 0


def test_cold_cortex_policy_deferred_log_is_rate_limited(monkeypatch):
    from core.brain import inference_gate as inference_gate_module

    gate = InferenceGate()
    # `next(ticks, default)`, not bare `next(ticks)`. `inference_gate_module.time`
    # IS the global time module, so this replaces time.monotonic for the whole
    # process. A three-shot iterator then raises StopIteration on the fourth
    # call from anywhere — and the autouse teardown fixtures call it, which
    # surfaced as "RuntimeError: generator raised StopIteration" in a fixture
    # that has nothing to do with clocks.
    ticks = iter([400.0, 420.0, 701.0])
    monkeypatch.setattr(
        inference_gate_module.time, "monotonic", lambda: next(ticks, 701.0)
    )

    gate._log_cold_cortex_policy_deferred()
    assert gate._last_cortex_policy_deferred_log_at == 400.0

    gate._log_cold_cortex_policy_deferred()
    assert gate._last_cortex_policy_deferred_log_at == 400.0

    gate._log_cold_cortex_policy_deferred()
    assert gate._last_cortex_policy_deferred_log_at == 701.0


@pytest.mark.asyncio
async def test_cold_start_recovery_does_not_race_scheduled_deferred_prewarm(monkeypatch):
    monkeypatch.setenv("AURA_DEFERRED_CORTEX_PREWARM", "auto")
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))

    gate = InferenceGate()
    client = _LaneWarmupClient()
    gate._mlx_client = client
    gate._prewarm_task = TaskProbe(done=False)

    await gate._ensure_cortex_recovery()

    client.warmup.assert_not_awaited()
    assert gate._cortex_recovery_attempts == 0


def test_desktop_safe_boot_allows_explicit_auto_deferred_prewarm_when_admitted(monkeypatch):
    monkeypatch.setenv("AURA_DEFERRED_CORTEX_PREWARM", "auto")
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))
    monkeypatch.setattr(
        InferenceGate,
        "_cortex_warmup_admission_snapshot",
        staticmethod(
            lambda _context: {
                "can_admit": True,
                "reason": "",
                "pressure_pct": 40.0,
                "available_gb": 36.0,
                "total_gb": 64.0,
            }
        ),
    )

    assert InferenceGate._boot_should_schedule_deferred_prewarm() is True


def test_desktop_safe_boot_refuses_explicit_auto_deferred_prewarm_under_pressure(monkeypatch):
    monkeypatch.setenv("AURA_DEFERRED_CORTEX_PREWARM", "auto")
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))
    monkeypatch.setattr(
        InferenceGate,
        "_cortex_warmup_admission_snapshot",
        staticmethod(
            lambda _context: {
                "can_admit": False,
                "reason": "memory_pressure:77.0%/12.0GB",
                "pressure_pct": 77.0,
                "available_gb": 12.0,
                "total_gb": 64.0,
            }
        ),
    )

    assert InferenceGate._boot_should_schedule_deferred_prewarm() is False


def test_explicit_deferred_cortex_prewarm_refusal_is_rate_limited(monkeypatch, caplog):
    from core.brain import inference_gate as inference_gate_module

    monkeypatch.setenv("AURA_DEFERRED_CORTEX_PREWARM", "1")
    monkeypatch.delenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", raising=False)
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: False))
    monkeypatch.setattr(
        InferenceGate,
        "_cortex_warmup_admission_snapshot",
        staticmethod(
            lambda _context: {
                "can_admit": False,
                "reason": "memory_pressure:58.2%/26.7GB",
                "pressure_pct": 58.2,
                "available_gb": 26.7,
                "total_gb": 64.0,
            }
        ),
    )
    # See the note on the other fake clock in this file: bare next() on an
    # exhausted iterator takes down whatever calls time.monotonic() next,
    # anywhere in the process.
    ticks = iter([100.0, 101.0, 161.0])
    monkeypatch.setattr(
        inference_gate_module.time, "monotonic", lambda: next(ticks, 161.0)
    )
    monkeypatch.setattr(inference_gate_module, "_LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_AT", 0.0)
    monkeypatch.setattr(inference_gate_module, "_LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_REASON", "")

    with caplog.at_level(logging.WARNING, logger="Aura.InferenceGate"):
        assert InferenceGate._boot_should_schedule_deferred_prewarm() is False
        assert InferenceGate._boot_should_schedule_deferred_prewarm() is False
        assert InferenceGate._boot_should_schedule_deferred_prewarm() is False

    warnings = [
        record
        for record in caplog.records
        if "Explicit deferred Cortex prewarm refused to protect RAM" in record.message
    ]
    assert len(warnings) == 2


def test_live_inference_readiness_does_not_reprobe_prewarm_policy(monkeypatch):
    gate = InferenceGate()
    gate._initialized = True
    gate._mlx_client = SimpleNamespace(is_alive=lambda: True)
    monkeypatch.setattr(
        "core.runtime.proof_policy.proof_run_active",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        InferenceGate,
        "_boot_should_schedule_deferred_prewarm",
        staticmethod(lambda: pytest.fail("resident Cortex must bypass prewarm admission")),
    )
    monkeypatch.setattr(
        gate, "get_conversation_status", lambda **_kw: {"conversation_ready": True}
    )

    assert gate.is_inference_ready() is True


def test_a_live_primary_process_is_not_by_itself_ready(monkeypatch):
    """Process liveness was the whole test, so a worker still loading weights —
    or wedged on a handshake — satisfied readiness."""
    gate = InferenceGate()
    gate._initialized = True
    gate._mlx_client = SimpleNamespace(is_alive=lambda: True)
    monkeypatch.setattr(
        "core.runtime.proof_policy.proof_run_active",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(gate, "_iter_local_clients", lambda: {})
    monkeypatch.setattr(
        gate,
        "get_conversation_status",
        lambda **_kw: {"conversation_ready": False, "active_generations": 0},
    )

    ready, reason = gate.inference_readiness()

    assert ready is False
    assert reason == "no_live_backend"


def test_a_fallback_client_that_cannot_report_a_lane_is_not_ready(monkeypatch):
    """Returning True because `get_lane_status` was ABSENT counted a missing
    check as a passed one."""
    gate = InferenceGate()
    gate._initialized = True
    gate._mlx_client = None
    monkeypatch.setattr(
        "core.runtime.proof_policy.proof_run_active",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        gate,
        "_iter_local_clients",
        lambda: {"mystery": SimpleNamespace(is_alive=lambda: True)},
    )

    ready, reason = gate.inference_readiness()

    assert ready is False
    assert reason == "no_client_can_prove_readiness"


@pytest.mark.asyncio
async def test_deferred_cortex_prewarm_defers_active_generation_without_degradation(monkeypatch):
    gate = InferenceGate()
    handled = asyncio.Event()

    monkeypatch.setattr(InferenceGate, "_foreground_user_turn_active", staticmethod(lambda: False))
    monkeypatch.setattr(InferenceGate, "_foreground_owner_active", staticmethod(lambda: False))
    monkeypatch.setattr(gate, "_cortex_warmup_deferral_reason", lambda _context: "")
    monkeypatch.setattr(gate, "_extend_startup_quiet_window", lambda _seconds: None)
    monkeypatch.setattr(
        gate,
        "get_conversation_status",
        lambda: {
            "conversation_ready": False,
            "state": "ready",
            "warmup_in_flight": False,
            "readiness_blockers": [],
            "last_failure_reason": "",
            "active_generations": 0,
        },
    )
    monkeypatch.setattr(
        "core.brain.inference_gate.psutil.virtual_memory",
        lambda: SimpleNamespace(
            percent=40.0,
            total=64 * 1024 ** 3,
            available=int(40.0 * 1024 ** 3),
        ),
    )

    degradation_probe = CallProbe(side_effect=AssertionError("busy prewarm is not degradation"))
    monkeypatch.setattr("core.brain.inference_gate.record_degradation", degradation_probe)

    async def busy_foreground_ready(*, timeout=None):  # noqa: ASYNC109
        handled.set()
        raise RuntimeError("active_generation_in_flight")

    monkeypatch.setattr(gate, "ensure_foreground_ready", busy_foreground_ready)

    gate._schedule_background_cortex_prewarm(delay=0.001)
    assert gate._deferred_prewarm_task is not None
    try:
        await asyncio.wait_for(handled.wait(), timeout=2.0)
        await asyncio.sleep(0)
        degradation_probe.assert_not_called()
        assert not gate._deferred_prewarm_task.done()
    finally:
        gate._deferred_prewarm_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await gate._deferred_prewarm_task


@pytest.mark.asyncio
async def test_deferred_cortex_prewarm_stands_down_for_chat_dependency_owner(
    monkeypatch,
):
    gate = InferenceGate()
    status_read = asyncio.Event()

    monkeypatch.setattr(
        gate,
        "get_conversation_status",
        lambda: (
            status_read.set()
            or {
                "conversation_ready": False,
                "state": "ready",
                "warmup_in_flight": False,
                "readiness_blockers": ["chat_dependencies_warming"],
                "last_failure_reason": "chat_dependencies_warming",
                "chat_dependencies_ready": False,
                "active_generations": 0,
            }
        ),
    )

    foreground_probe = CallProbe(
        side_effect=AssertionError(
            "the model prewarmer must not race the chat-dependency owner"
        )
    )
    degradation_probe = CallProbe(
        side_effect=AssertionError("normal dependency warmup is not degradation")
    )
    monkeypatch.setattr(gate, "ensure_foreground_ready", foreground_probe)
    monkeypatch.setattr("core.brain.inference_gate.record_degradation", degradation_probe)

    gate._schedule_background_cortex_prewarm(delay=0.001)
    assert gate._deferred_prewarm_task is not None
    await asyncio.wait_for(status_read.wait(), timeout=2.0)
    await asyncio.wait_for(gate._deferred_prewarm_task, timeout=2.0)

    foreground_probe.assert_not_called()
    degradation_probe.assert_not_called()


@pytest.mark.asyncio
async def test_deferred_cortex_prewarm_treats_visible_probe_missing_as_unproven_readiness(
    monkeypatch,
):
    gate = InferenceGate()
    handled = asyncio.Event()

    monkeypatch.setattr(InferenceGate, "_foreground_user_turn_active", staticmethod(lambda: False))
    monkeypatch.setattr(InferenceGate, "_foreground_owner_active", staticmethod(lambda: False))
    monkeypatch.setattr(gate, "_cortex_warmup_deferral_reason", lambda _context: "")
    monkeypatch.setattr(gate, "_extend_startup_quiet_window", lambda _seconds: None)
    monkeypatch.setattr(
        "core.brain.inference_gate.psutil.virtual_memory",
        lambda: SimpleNamespace(
            percent=40.0,
            total=64 * 1024 ** 3,
            available=int(40.0 * 1024 ** 3),
        ),
    )

    degradation_probe = CallProbe(side_effect=AssertionError("readiness proof is not degradation"))
    monkeypatch.setattr("core.brain.inference_gate.record_degradation", degradation_probe)

    async def unproven_foreground_ready(*, timeout=None):  # noqa: ASYNC109
        handled.set()
        raise RuntimeError("visible_conversation_probe_missing")

    monkeypatch.setattr(gate, "ensure_foreground_ready", unproven_foreground_ready)

    gate._schedule_background_cortex_prewarm(delay=0.001)
    assert gate._deferred_prewarm_task is not None
    try:
        await asyncio.wait_for(handled.wait(), timeout=2.0)
        await asyncio.sleep(0)
        degradation_probe.assert_not_called()
        assert not gate._deferred_prewarm_task.done()
    finally:
        gate._deferred_prewarm_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await gate._deferred_prewarm_task


@pytest.mark.asyncio
async def test_cortex_recovery_reserves_ownership_before_task_is_scheduled(monkeypatch):
    gate = InferenceGate()
    # Cold-start recovery is gated by _boot_should_schedule_deferred_prewarm,
    # which consults REAL host memory through the warmup admission snapshot.
    # These tests are about what happens once the policy says yes, so they pin
    # the policy instead of inheriting the operator's env and whatever the host
    # happens to have free — the reason they passed alone and failed in a long
    # run, where the resident 32B is holding 20GB by the time they execute.
    monkeypatch.setenv("AURA_DEFERRED_CORTEX_PREWARM", "1")
    monkeypatch.setenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "1")
    release_warmup = asyncio.Event()

    class _DeadClient:
        _lane_state = "cold"

        def is_alive(self):
            return False

        async def warmup(self):
            await release_warmup.wait()
            return False

    gate._mlx_client = _DeadClient()
    monkeypatch.setattr(InferenceGate, "_foreground_user_turn_active", staticmethod(lambda: False))
    monkeypatch.setattr(InferenceGate, "_foreground_owner_active", staticmethod(lambda: False))
    monkeypatch.setattr(gate, "_cortex_warmup_deferral_reason", lambda _context: "")
    monkeypatch.setattr(
        gate,
        "get_conversation_status",
        lambda: {
            "conversation_ready": False,
            "state": "cold",
            "warmup_in_flight": False,
            "warmup_attempted": False,
            "last_failure_reason": "",
        },
    )

    scheduled = []

    def _create_task(coro, **_kwargs):
        assert gate._cortex_recovery_in_progress is True
        task = asyncio.create_task(coro)
        scheduled.append(task)
        return task

    monkeypatch.setattr(
        "core.brain.inference_gate.get_task_tracker",
        lambda: SimpleNamespace(create_task=_create_task),
    )

    await gate._ensure_cortex_recovery()

    assert gate._cortex_recovery_in_progress is True
    assert len(scheduled) == 1
    scheduled[0].cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await scheduled[0]
    assert gate._cortex_recovery_in_progress is False


def test_inference_health_ready_rejects_deferred_safe_boot_without_live_worker(monkeypatch):
    gate = InferenceGate()
    gate._initialized = True
    gate._mlx_client = SimpleNamespace(is_alive=lambda: False)
    monkeypatch.setattr(gate, "_iter_local_clients", lambda: {})
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))
    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    monkeypatch.setenv("AURA_PROOF_MODEL_TIER", "primary")

    assert gate.is_alive() is True
    assert gate.is_inference_ready() is False


def test_inference_health_ready_rejects_live_but_unready_primary_worker(monkeypatch):
    gate = InferenceGate()
    gate._initialized = True
    gate._mlx_client = SimpleNamespace(
        is_alive=lambda: True,
        get_lane_status=lambda: {
            "state": "warming",
            "conversation_ready": False,
            "readiness_blockers": ["warmup_in_flight"],
        },
    )
    monkeypatch.setattr(gate, "_iter_local_clients", lambda: {})
    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    monkeypatch.setenv("AURA_PROOF_MODEL_TIER", "primary")

    assert gate.is_inference_ready() is False


def test_inference_health_ready_accepts_conversation_ready_primary_worker(monkeypatch):
    gate = InferenceGate()
    gate._initialized = True
    gate._mlx_client = SimpleNamespace(
        is_alive=lambda: True,
        get_lane_status=lambda: {
            "state": "ready",
            "conversation_ready": True,
            "readiness_blockers": [],
            "last_visible_readiness_at": time.time(),
            "last_user_facing_completed_at": 0.0,
        },
    )
    monkeypatch.setattr(gate, "_iter_local_clients", lambda: {})
    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    monkeypatch.setenv("AURA_PROOF_MODEL_TIER", "primary")

    assert gate.is_inference_ready() is True


def test_inference_health_ready_accepts_recent_active_foreground_generation(monkeypatch):
    gate = InferenceGate()
    gate._initialized = True
    gate._mlx_client = SimpleNamespace(
        is_alive=lambda: True,
        get_lane_status=lambda: {
            "state": "working",
            "conversation_ready": False,
            "readiness_blockers": ["active_generation_in_flight"],
            "foreground_owned": True,
            "active_generations": 1,
            "current_request_started_at": time.time() - 8.0,
            "last_token_progress_at": time.time() - 1.0,
        },
    )
    monkeypatch.setattr(gate, "_iter_local_clients", lambda: {})
    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    monkeypatch.setenv("AURA_PROOF_MODEL_TIER", "primary")

    assert gate.is_inference_ready() is True


def test_inference_health_ready_rejects_stalled_active_foreground_generation(monkeypatch):
    gate = InferenceGate()
    gate._initialized = True
    gate._mlx_client = SimpleNamespace(
        is_alive=lambda: True,
        get_lane_status=lambda: {
            "state": "working",
            "conversation_ready": False,
            "readiness_blockers": ["active_generation_in_flight"],
            "foreground_owned": True,
            "active_generations": 1,
            "current_request_started_at": time.time() - 90.0,
            "last_token_progress_at": time.time() - 60.0,
        },
    )
    monkeypatch.setattr(gate, "_iter_local_clients", lambda: {})
    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    monkeypatch.setenv("AURA_PROOF_MODEL_TIER", "primary")

    assert gate.is_inference_ready() is False


def test_safe_boot_status_does_not_advertise_cold_cortex_as_active(monkeypatch):
    gate = InferenceGate()
    gate._initialized = True
    gate._mlx_client = SimpleNamespace(
        is_alive=lambda: False,
        get_lane_status=lambda: {
            "state": "cold",
            "conversation_ready": False,
            "readiness_blockers": ["worker_not_alive"],
        },
    )
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))

    lane = gate.get_conversation_status()

    assert gate.is_alive() is True
    assert lane["conversation_ready"] is False
    assert lane["foreground_endpoint"] is None


def test_conversation_status_recovery_schedule_is_cooldowned(monkeypatch):
    gate = InferenceGate()
    scheduled: list[float] = []

    class _CompletedFailedPrewarm:
        def done(self):
            return True

        def exception(self):
            return RuntimeError("warmup_readiness_no_text")

    class _Client:
        _warmup_in_flight = False

        def __init__(self):
            self.state_updates: list[tuple[str, str]] = []

        def is_alive(self):
            return True

        def get_lane_status(self):
            now = time.time()
            return {
                "state": "warming",
                "conversation_ready": False,
                "readiness_blockers": [],
                "last_error": "warmup_readiness_no_text",
                "last_transition_at": now,
                "last_progress_at": now,
                "warmup_attempted": True,
                "warmup_in_flight": False,
            }

        def _set_lane_state(self, state, error=""):
            self.state_updates.append((state, error))

    gate._initialized = True
    gate._mlx_client = _Client()
    gate._prewarm_task = _CompletedFailedPrewarm()
    monkeypatch.setattr(gate, "_cortex_warmup_deferral_reason", lambda context="background": None)
    monkeypatch.setattr(gate, "_schedule_background_cortex_prewarm", lambda delay=12.0: scheduled.append(delay))

    # CP126 ab3c124a: a plain OBSERVATION is pure — polling health endpoints
    # must not schedule recovery work. The gate's own self-heal path opts in.
    observed_first = gate.get_conversation_status()
    observed_second = gate.get_conversation_status()
    assert observed_first["conversation_ready"] is False
    assert observed_second["conversation_ready"] is False
    assert scheduled == [], "observing the lane scheduled background work"

    first = gate.get_conversation_status(observe_only=False)
    second = gate.get_conversation_status(observe_only=False)

    assert first["conversation_ready"] is False
    assert second["conversation_ready"] is False
    # ...and the ratchet that DOES act is still cooldowned.
    assert scheduled == [2.0]

    gate._last_status_recovery_schedule_at -= 31.0
    gate.get_conversation_status(observe_only=False)
    assert scheduled == [2.0, 2.0]


def test_background_local_deferral_protects_cold_cortex_during_safe_boot(monkeypatch):
    gate = InferenceGate()
    gate._created_at = time.monotonic()
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))
    monkeypatch.setattr(InferenceGate, "_foreground_user_turn_active", staticmethod(lambda: False))
    monkeypatch.setattr(InferenceGate, "_foreground_owner_active", staticmethod(lambda: False))
    monkeypatch.setattr(gate, "_should_quiet_background_for_cortex_startup", lambda: False)
    monkeypatch.setattr(gate, "_background_memory_pressure_active", lambda: False)
    # Fixed headroom so real system RAM doesn't interfere with test logic
    _low_pressure = {"pressure_pct": 40.0, "available_gb": 32.0, "safe": True, "reason": "ok"}
    monkeypatch.setattr(InferenceGate, "_headroom_snapshot", staticmethod(lambda *a, **kw: _low_pressure))
    monkeypatch.setattr(gate, "_foreground_headroom_reserved", lambda *a, **kw: False)
    monkeypatch.setattr(
        gate,
        "get_conversation_status",
        lambda: {"conversation_ready": False, "state": "cold", "warmup_in_flight": False},
    )

    assert gate._background_local_deferral_reason(origin="system") == "cortex_startup_quiet"


def test_background_local_deferral_reserves_ready_cortex_during_safe_boot(monkeypatch):
    gate = InferenceGate()
    gate._created_at = time.monotonic()
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))
    monkeypatch.setattr(InferenceGate, "_foreground_user_turn_active", staticmethod(lambda: False))
    monkeypatch.setattr(InferenceGate, "_foreground_owner_active", staticmethod(lambda: False))
    monkeypatch.setattr(InferenceGate, "_foreground_quiet_window_active", staticmethod(lambda: False))
    monkeypatch.setattr(gate, "_background_memory_pressure_active", lambda: False)
    _low_pressure = {"pressure_pct": 40.0, "available_gb": 32.0, "safe": True, "reason": "ok"}
    monkeypatch.setattr(InferenceGate, "_headroom_snapshot", staticmethod(lambda *a, **kw: _low_pressure))
    monkeypatch.setattr(gate, "_foreground_headroom_reserved", lambda *a, **kw: False)
    monkeypatch.setattr(
        gate,
        "get_conversation_status",
        lambda: {"conversation_ready": True, "state": "ready", "warmup_in_flight": False},
    )

    assert gate._background_local_deferral_reason(origin="system") == "cortex_startup_quiet"


def test_background_local_deferral_honors_ready_cortex_foreground_quiet_window(monkeypatch):
    gate = InferenceGate()
    monkeypatch.setattr(InferenceGate, "_foreground_user_turn_active", staticmethod(lambda: False))
    monkeypatch.setattr(InferenceGate, "_foreground_owner_active", staticmethod(lambda: False))
    monkeypatch.setattr(InferenceGate, "_foreground_quiet_window_active", staticmethod(lambda: True))
    monkeypatch.setattr(gate, "_should_quiet_background_for_cortex_startup", lambda: False)
    monkeypatch.setattr(gate, "_background_memory_pressure_active", lambda: False)
    monkeypatch.setattr(gate, "_foreground_headroom_reserved", lambda *a, **kw: False)
    monkeypatch.setattr(
        gate,
        "get_conversation_status",
        lambda: {"conversation_ready": True, "state": "ready", "warmup_in_flight": False},
    )

    assert gate._background_local_deferral_reason(origin="affect_engine") == "foreground_quiet_window"


@pytest.mark.asyncio
async def test_tier_health_reports_demand_loaded_brainstem_as_standby(monkeypatch):
    class _Lane:
        _lane_state = "cold"

        @staticmethod
        def is_alive():
            return False

    gate = InferenceGate()
    gate._mlx_client = SimpleNamespace(is_alive=lambda: True, _lane_state="ready")
    gate._cortex_recovery_in_progress = False
    monkeypatch.delenv("AURA_HEALTH_WARM_LOCAL_TIERS", raising=False)
    monkeypatch.setattr(gate, "_background_local_deferral_reason", lambda origin: None)

    import core.brain.llm.mlx_client as mlx_client
    import core.brain.llm.model_registry as model_registry

    monkeypatch.setattr(mlx_client, "get_mlx_client", lambda **_kwargs: _Lane())
    monkeypatch.setattr(model_registry, "get_brainstem_path", lambda: "/models/brainstem")
    monkeypatch.setattr(model_registry, "get_fallback_path", lambda: None)
    monkeypatch.setattr(
        gate, "get_conversation_status", lambda **_kw: {"conversation_ready": True}
    )

    statuses = await gate.ensure_all_tiers_healthy()

    assert statuses["cortex"] == "alive"
    assert statuses["brainstem"] == "standby"


@pytest.mark.asyncio
async def test_a_cortex_that_cannot_hold_a_conversation_is_not_labelled_alive(monkeypatch):
    """"alive" was read off process liveness, so a worker loading weights or
    wedged on a handshake reported the same word as one taking turns."""
    gate = InferenceGate()
    gate._mlx_client = SimpleNamespace(is_alive=lambda: True, _lane_state="warming")
    gate._cortex_recovery_in_progress = False
    monkeypatch.setattr(gate, "_background_local_deferral_reason", lambda origin: "test")
    monkeypatch.setattr(
        gate,
        "get_conversation_status",
        lambda **_kw: {"conversation_ready": False, "active_generations": 0},
    )

    import core.brain.llm.model_registry as model_registry

    monkeypatch.setattr(model_registry, "get_fallback_path", lambda: None)

    statuses = await gate.ensure_all_tiers_healthy()

    assert statuses["cortex"] == "alive_not_conversation_ready"
    # Not dead: no incident, no recovery. The label just stops overclaiming.
    assert statuses["cortex"] != "dead"
    assert gate.tier_health_receipt()["evidence"]["cortex"] == "process_liveness_only"


@pytest.mark.asyncio
async def test_an_observe_only_sweep_actuates_nothing(monkeypatch):
    """The monitoring cadence called this every ten ticks, and a dead Cortex
    made it spawn recovery — GPU and RAM, on a timer, from a health check."""
    gate = InferenceGate()
    gate._mlx_client = SimpleNamespace(is_alive=lambda: False, _lane_state="cold")
    gate._cortex_recovery_in_progress = False

    recovered: list[str] = []

    async def _recover():
        recovered.append("cortex")

    monkeypatch.setattr(gate, "_ensure_cortex_recovery", _recover)
    monkeypatch.setattr(gate, "_background_local_deferral_reason", lambda origin: "test")

    import core.brain.llm.model_registry as model_registry

    monkeypatch.setattr(model_registry, "get_fallback_path", lambda: None)

    statuses = await gate.observe_tier_health()

    assert statuses["cortex"] == "dead"
    assert recovered == [], "an observe-only sweep started a recovery"
    receipt = gate.tier_health_receipt()
    assert receipt["repair_enabled"] is False
    assert receipt["actuated"] == []

    statuses = await gate.ensure_all_tiers_healthy()

    assert recovered == ["cortex"], "the repairing sweep stopped repairing"
    assert gate.tier_health_receipt()["actuated"] == ["cortex_recovery"]


@pytest.mark.asyncio
async def test_a_named_but_unreadable_reflex_model_is_not_available(monkeypatch, tmp_path):
    """`available` meant `Path.exists()`, so a zero-byte file from an
    interrupted download passed — on the tier that answers when the other two
    are gone."""
    gate = InferenceGate()
    gate._mlx_client = None
    monkeypatch.setattr(gate, "_background_local_deferral_reason", lambda origin: "test")

    import core.brain.llm.model_registry as model_registry

    truncated = tmp_path / "reflex.gguf"
    truncated.write_bytes(b"")
    monkeypatch.setattr(model_registry, "get_fallback_path", lambda: truncated)

    statuses = await gate.ensure_all_tiers_healthy()

    assert statuses["reflex"] == "model_unreadable"
    assert gate.tier_health_receipt()["evidence"]["reflex"] == "empty_file"

    truncated.write_bytes(b"not empty")
    statuses = await gate.ensure_all_tiers_healthy()

    assert statuses["reflex"] == "available"
