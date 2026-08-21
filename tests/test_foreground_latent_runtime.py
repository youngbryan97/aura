import hashlib

import pytest

from core.brain.foreground_latent_runtime import (
    latent_owner_exhausted,
    materialized_latent_incumbent,
    run_foreground_latent_episode,
)


class _Ingress:
    stakes = 0.81
    uncertainty = 0.72
    epistemic_genesis = "genesis"
    epistemic_state = "epistemic-state"
    memory_result = "memory-result"

    def to_receipt(self):
        return {"schema": "test.ingress.v1", "stakes": self.stakes}


class _LatentService:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def deep_reason_with_acquisition(self, **kwargs):
        self.calls.append(kwargs)
        return self.result

    def get_status(self):
        return {
            "last_failure_receipt": {},
            "last_receipt": {},
            "last_progress": {},
        }


class _UnexpectedService(_LatentService):
    async def deep_reason_with_acquisition(self, **kwargs):
        self.calls.append(kwargs)
        raise _WorkerProtocolError("custom worker protocol failure")


class _WorkerProtocolError(Exception):
    pass


class _CircuitOpenService(_LatentService):
    def foreground_admission(self):
        return {
            "admitted": False,
            "reason": "unchanged_terminal_bridge_contract_failure",
            "failure_streak": 1,
        }


class _QualifiedService(_LatentService):
    def __init__(self, result):
        super().__init__({})
        self.qualified_result = result
        self.qualified_calls = []

    async def qualified_recurrent_reason(self, objective, **kwargs):
        self.qualified_calls.append((objective, kwargs))
        return self.qualified_result


def _selected(**_kwargs):
    return {
        "latent_cortex_selected": True,
        "latent_cortex_selection_reason": "multipart_or_extended_prompt",
        "latent_cortex_depth_worthy": True,
        "stakes": 0.70,
        "uncertainty": 0.80,
    }


@pytest.mark.asyncio
async def test_foreground_latent_runner_executes_full_stack_with_typed_ingress(
    monkeypatch,
):
    service = _LatentService(
        {
            "ok": True,
            "text": "The complete latent answer.",
            "receipt": {
                "episode_id": "ep-1",
                "last_stage": "complete",
                "runtime_identity": {"identity_bound": True},
            },
            "progress": {"stage": "complete"},
        }
    )
    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime.select_foreground_episode",
        _selected,
    )
    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime._resolve_service",
        lambda: service,
    )
    monkeypatch.setattr(
        "core.brain.cognitive_ingress.assemble_cognitive_ingress_async",
        _async_value(_Ingress()),
    )
    monkeypatch.setattr(
        "core.brain.cognitive_ingress.cognitive_context_items",
        lambda _ingress: [{"source": "memory", "text": "bounded evidence"}],
    )

    outcome = await run_foreground_latent_episode(
        orchestrator="orchestrator",
        messages=[{"role": "user", "content": "Compare both designs."}],
        visible_objective="Compare both designs.",
        foreground=True,
        desktop_required=True,
        cognitive_mode="deliberate",
        request_timeout_s=180.0,
        decode_max_tokens=900,
        recurrent_loops=2,
        capability_modifiers={
            "last_skill_run": "run_code",
            "last_skill_ok": True,
            "last_skill_objective_hash": hashlib.sha256(
                b"Compare both designs."
            ).hexdigest(),
            "last_skill_result_payload": {
                "ok": True,
                "stdout": "measured delta=0.42",
                "exit_code": 0,
            },
        },
    )

    assert outcome.succeeded is True
    assert outcome.text == "The complete latent answer."
    assert outcome.trace["latent_cortex_identity_bound"] is True
    assert outcome.trace["latent_cortex_ingress"]["schema"] == "test.ingress.v1"
    assert len(service.calls) == 1
    call = service.calls[0]
    assert call["require_full_stack"] is True
    assert call["cognitive_context"][0] == {
        "source": "memory",
        "text": "bounded evidence",
    }
    assert call["cognitive_context"][1]["source"] == "capability.run_code"
    assert call["cognitive_context"][1]["text"] == "measured delta=0.42"
    assert call["cognitive_context"][1]["instruction_authority"] is False
    assert outcome.trace["latent_cortex_capability_evidence"]["admitted"] is True
    assert outcome.trace["latent_cortex_context_merge"]["admitted_items"] == 2
    assert call["epistemic_genesis"] == "genesis"
    assert call["config_overrides"]["decode_max_tokens"] == 900
    assert call["runtime_controls"]["clean_user_surface_recurrent_loops"] == 2


@pytest.mark.asyncio
async def test_foreground_latent_runner_allows_fallback_after_terminal_receipt_failure(
    monkeypatch,
):
    service = _LatentService(
        {
            "ok": False,
            "reason": "receipt_contract_failed:answer_replacement_unproven",
            "receipt": {"episode_id": "ep-2", "last_stage": "complete"},
        }
    )
    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime.select_foreground_episode",
        _selected,
    )
    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime._resolve_service",
        lambda: service,
    )
    monkeypatch.setattr(
        "core.brain.cognitive_ingress.assemble_cognitive_ingress_async",
        _async_value(_Ingress()),
    )
    monkeypatch.setattr(
        "core.brain.cognitive_ingress.cognitive_context_items",
        lambda _ingress: [],
    )

    outcome = await run_foreground_latent_episode(
        orchestrator=None,
        messages=[{"role": "user", "content": "Compare both designs."}],
        visible_objective="Compare both designs.",
        foreground=True,
        desktop_required=True,
        cognitive_mode="deliberate",
        request_timeout_s=180.0,
    )

    assert outcome.attempted is True
    assert outcome.succeeded is False
    assert outcome.fallback_allowed is True
    assert outcome.trace["latent_cortex_fallback_used"] is True


def test_host_reconstructed_incumbent_is_served_without_worker_fallback_flag(
    monkeypatch,
):
    result = {
        "ok": False,
        "text": "A complete ordinary answer.",
        "tokens": [1, 2, 3],
        "receipt": {
            "honest_flags": ["vanilla_incumbent_captured_before_adaptation"],
            "resident_owner_released": True,
            "resident_state_reusable": True,
            "answer_replacement": {"receipt_sha256": "source"},
            "host_incumbent_disposition": {"receipt_sha256": "host"},
        },
    }
    observed = {}

    def _validate(value, **kwargs):
        observed["value"] = value
        observed.update(kwargs)
        return dict(value)

    monkeypatch.setattr(
        "core.brain.llm.latent_cortex.answer_replacement.validate_host_incumbent_disposition",
        _validate,
    )

    incumbent = materialized_latent_incumbent(result)

    assert incumbent == (result["text"], result["receipt"])
    assert observed["expected_text"] == result["text"]
    assert observed["expected_tokens"] == result["tokens"]
    assert observed["answer_replacement_receipt"] == result["receipt"]["answer_replacement"]


def test_host_reconstructed_incumbent_rejects_invalid_disposition(monkeypatch):
    result = {
        "ok": False,
        "text": "Unbound text.",
        "tokens": [1],
        "receipt": {
            "honest_flags": ["vanilla_incumbent_captured_before_adaptation"],
            "resident_owner_released": True,
            "resident_state_reusable": True,
            "answer_replacement": {},
            "host_incumbent_disposition": {},
        },
    }

    def _reject(*_args, **_kwargs):
        raise ValueError("tampered")

    monkeypatch.setattr(
        "core.brain.llm.latent_cortex.answer_replacement.validate_host_incumbent_disposition",
        _reject,
    )

    assert materialized_latent_incumbent(result) is None


@pytest.mark.asyncio
async def test_foreground_latent_runner_uses_complete_owner_window(monkeypatch):
    service = _LatentService({"ok": False, "reason": "worker_not_ready"})
    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime.select_foreground_episode",
        _selected,
    )
    monkeypatch.setattr(
        "core.brain.cognitive_ingress.assemble_cognitive_ingress_async",
        _async_value(_Ingress()),
    )
    monkeypatch.setattr(
        "core.brain.cognitive_ingress.cognitive_context_items",
        lambda _ingress: [],
    )

    await run_foreground_latent_episode(
        orchestrator=None,
        messages=[{"role": "user", "content": "Compare all five designs."}],
        visible_objective="Compare all five designs.",
        foreground=True,
        desktop_required=True,
        cognitive_mode="deliberate",
        request_timeout_s=480.0,
        service=service,
    )

    assert service.calls[0]["timeout_s"] == 472.0


@pytest.mark.asyncio
async def test_failed_episode_serves_only_receipted_materialized_incumbent(monkeypatch):
    incumbent = "The already completed ordinary answer remains authoritative."
    service = _LatentService(
        {
            "ok": False,
            "reason": "receipt_contract_failed:latent_optimization_budget_exhausted",
            "text": incumbent,
            "receipt": {
                "episode_id": "ep-incumbent",
                "last_stage": "latent_optimization",
                "honest_flags": [
                    "vanilla_incumbent_captured_before_adaptation",
                    "fallback_reused_materialized_incumbent",
                ],
                "resident_owner_released": True,
                "resident_state_reusable": True,
            },
        }
    )
    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime.select_foreground_episode",
        _selected,
    )
    monkeypatch.setattr(
        "core.brain.cognitive_ingress.assemble_cognitive_ingress_async",
        _async_value(_Ingress()),
    )
    monkeypatch.setattr(
        "core.brain.cognitive_ingress.cognitive_context_items",
        lambda _ingress: [],
    )

    outcome = await run_foreground_latent_episode(
        orchestrator=None,
        messages=[{"role": "user", "content": "Compare both designs."}],
        visible_objective="Compare both designs.",
        foreground=True,
        desktop_required=True,
        cognitive_mode="deliberate",
        request_timeout_s=180.0,
        service=service,
    )

    assert outcome.succeeded is False
    assert outcome.answer_available is True
    assert outcome.text == incumbent
    assert outcome.fallback_allowed is False
    assert outcome.trace["latent_cortex_incumbent_fallback_served"] is True


@pytest.mark.asyncio
async def test_unchanged_terminal_bridge_failure_skips_the_expensive_general_episode(
    monkeypatch,
):
    service = _CircuitOpenService({})
    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime.select_foreground_episode",
        _selected,
    )
    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime._resolve_service",
        lambda: service,
    )

    outcome = await run_foreground_latent_episode(
        orchestrator=None,
        messages=[{"role": "user", "content": "Compare both designs."}],
        visible_objective="Compare both designs.",
        foreground=True,
        desktop_required=True,
        cognitive_mode="deliberate",
        request_timeout_s=180.0,
    )

    assert outcome.attempted is False
    assert outcome.fallback_allowed is True
    assert service.calls == []
    assert outcome.trace["latent_cortex_failure_reason"] == (
        "unchanged_terminal_bridge_contract_failure"
    )


@pytest.mark.asyncio
async def test_unclassified_post_acquisition_exception_suppresses_colliding_decode(
    monkeypatch,
):
    service = _UnexpectedService({})
    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime.select_foreground_episode",
        _selected,
    )
    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime._resolve_service",
        lambda: service,
    )
    monkeypatch.setattr(
        "core.brain.cognitive_ingress.assemble_cognitive_ingress_async",
        _async_value(_Ingress()),
    )
    monkeypatch.setattr(
        "core.brain.cognitive_ingress.cognitive_context_items",
        lambda _ingress: [],
    )

    outcome = await run_foreground_latent_episode(
        orchestrator=None,
        messages=[{"role": "user", "content": "Compare both designs."}],
        visible_objective="Compare both designs.",
        foreground=True,
        desktop_required=True,
        cognitive_mode="deliberate",
        request_timeout_s=180.0,
    )

    assert outcome.attempted is True
    assert outcome.fallback_allowed is False
    assert outcome.trace["latent_cortex_failure_reason"].startswith(
        "latent_integrity:runtime_error:_WorkerProtocolError"
    )


@pytest.mark.asyncio
async def test_foreground_latent_runner_does_not_acquire_service_for_incompatible_turn(
    monkeypatch,
):
    service_resolution_attempted = False

    def _resolve():
        nonlocal service_resolution_attempted
        service_resolution_attempted = True
        return _LatentService({})

    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime._resolve_service",
        _resolve,
    )

    outcome = await run_foreground_latent_episode(
        orchestrator=None,
        messages=[{"role": "user", "content": "Return exact JSON."}],
        visible_objective="Return exact JSON.",
        foreground=True,
        desktop_required=True,
        cognitive_mode="deliberate",
        request_timeout_s=180.0,
        incompatible_contract=True,
    )

    assert outcome.selected is False
    assert outcome.trace["latent_cortex_selection_reason"] == "incompatible_contract"
    assert service_resolution_attempted is False


@pytest.mark.asyncio
async def test_qualified_exact_domain_precedes_general_incompatible_exclusion(monkeypatch):
    from core.learning.recurrence_curriculum import modular_chain

    task = modular_chain(4, 2026081221)
    service = _QualifiedService(
        {
            "eligible": True,
            "attempted": True,
            "ok": True,
            "reason": "qualified_recurrent_completed",
            "text": task.answer,
            "receipt": {"schema": "qualified.test", "receipt_sha256": "a" * 64},
        }
    )
    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime._resolve_service", lambda: service
    )
    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime.select_foreground_episode",
        lambda **_kwargs: pytest.fail("qualified task must precede general selection"),
    )

    outcome = await run_foreground_latent_episode(
        orchestrator=None,
        messages=[{"role": "user", "content": task.prompt}],
        visible_objective=task.prompt,
        foreground=True,
        desktop_required=True,
        cognitive_mode="reactive",
        request_timeout_s=180.0,
        strict_output_contract=True,
        incompatible_contract=True,
    )

    assert outcome.succeeded is True
    assert outcome.text == task.answer
    assert outcome.fallback_allowed is False
    assert outcome.trace["qualified_recurrent_succeeded"] is True
    assert outcome.trace["latent_cortex_selection_reason"] == (
        "qualified_recurrent_exact_domain"
    )
    assert outcome.evidence == ("qualified_recurrent_typed_execution",)


@pytest.mark.asyncio
async def test_qualified_semantic_neural_domain_is_observable_and_canonical(monkeypatch):
    from core.learning.frontier_process_supervision import (
        frontier_process_task_battery,
    )

    task = frontier_process_task_battery(
        ("coding",),
        (1,),
        1,
        seed=2026081560,
    )[0]
    service = _QualifiedService(
        {
            "eligible": True,
            "attempted": True,
            "ok": True,
            "reason": "qualified_semantic_neural_completed",
            "text": task.answer,
            "receipt": {
                "schema": "qualified.test",
                "receipt_sha256": "a" * 64,
                "activation_receipt": {"promotion_mode": "active"},
            },
        }
    )
    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime._resolve_service", lambda: service
    )
    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime.select_foreground_episode",
        lambda **_kwargs: pytest.fail("semantic task must precede general selection"),
    )

    outcome = await run_foreground_latent_episode(
        orchestrator=None,
        messages=[{"role": "user", "content": task.prompt}],
        visible_objective=task.prompt,
        foreground=True,
        desktop_required=False,
        cognitive_mode="reactive",
        request_timeout_s=30.0,
        strict_output_contract=True,
        incompatible_contract=True,
    )

    assert outcome.succeeded is True
    assert outcome.text == task.answer
    assert outcome.trace["latent_cortex_selection_reason"] == (
        "qualified_semantic_neural_exact_domain"
    )
    assert outcome.evidence == ("qualified_semantic_neural_execution",)
    assert service.qualified_calls[0][1]["timeout_s"] == pytest.approx(22.5)


@pytest.mark.asyncio
async def test_qualified_semantic_shadow_preserves_ordinary_authority(monkeypatch):
    from core.learning.frontier_process_supervision import (
        frontier_process_task_battery,
    )

    task = frontier_process_task_battery(
        ("calibration",),
        (1,),
        1,
        seed=2026081574,
    )[0]
    service = _QualifiedService(
        {
            "eligible": True,
            "attempted": True,
            "ok": True,
            "reason": "qualified_semantic_neural_completed",
            "text": task.answer,
            "receipt": {
                "schema": "qualified.test",
                "receipt_sha256": "b" * 64,
                "activation_receipt": {
                    "package_id": "cp568-resident-semantic-neural-shadow",
                    "promotion_mode": "shadow",
                    "activation_sha256": "c" * 64,
                },
            },
        }
    )
    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime._resolve_service", lambda: service
    )

    outcome = await run_foreground_latent_episode(
        orchestrator=None,
        messages=[{"role": "user", "content": task.prompt}],
        visible_objective=task.prompt,
        foreground=True,
        desktop_required=True,
        cognitive_mode="reactive",
        request_timeout_s=30.0,
        strict_output_contract=True,
        incompatible_contract=True,
    )

    assert outcome.succeeded is False
    assert outcome.fallback_allowed is True
    assert outcome.text == ""
    assert outcome.shadow_text == task.answer
    assert outcome.trace["qualified_recurrent_succeeded"] is True
    assert outcome.trace["qualified_recurrent_shadowed"] is True
    assert outcome.trace["latent_cortex_selection_reason"] == (
        "qualified_semantic_neural_shadow"
    )
    assert outcome.evidence == ("qualified_semantic_neural_shadow",)


@pytest.mark.asyncio
async def test_qualified_recurrent_service_absence_is_a_typed_disposition(monkeypatch):
    from core.learning.frontier_process_supervision import (
        frontier_process_task_battery,
    )

    task = frontier_process_task_battery(
        ("calibration",),
        (1,),
        1,
        seed=2026081582,
    )[0]
    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime._resolve_service",
        lambda: None,
    )

    outcome = await run_foreground_latent_episode(
        orchestrator=None,
        messages=[],
        visible_objective=task.prompt,
        foreground=True,
        desktop_required=False,
        cognitive_mode="fast",
        request_timeout_s=8.0,
    )

    assert outcome.text == ""
    assert outcome.fallback_allowed is True
    assert outcome.trace["qualified_recurrent_eligible"] is True
    assert outcome.trace["qualified_recurrent_attempted"] is False
    assert outcome.trace["qualified_recurrent_reason"] == (
        "qualified_recurrent_service_not_registered"
    )


@pytest.mark.asyncio
async def test_activated_qualified_failure_suppresses_uncertified_fallback(monkeypatch):
    from core.learning.recurrence_curriculum import khop_reachability

    task = khop_reachability(2, 2026081222)
    service = _QualifiedService(
        {
            "eligible": True,
            "attempted": True,
            "ok": False,
            "reason": "qualified_decode_receipt_invalid",
        }
    )
    monkeypatch.setattr(
        "core.brain.foreground_latent_runtime._resolve_service", lambda: service
    )

    outcome = await run_foreground_latent_episode(
        orchestrator=None,
        messages=[{"role": "user", "content": task.prompt}],
        visible_objective=task.prompt,
        foreground=True,
        desktop_required=True,
        cognitive_mode="deliberate",
        request_timeout_s=180.0,
    )

    assert outcome.attempted is True
    assert outcome.succeeded is False
    assert outcome.fallback_allowed is False
    assert outcome.trace["latent_cortex_failure_reason"] == (
        "qualified_decode_receipt_invalid"
    )


@pytest.mark.parametrize(
    ("reason", "receipt", "expected"),
    [
        ("soft_cancel:watchdog", {"episode_id": "e", "last_stage": "decode"}, False),
        (
            "receipt_contract_failed:x",
            {"episode_id": "e", "last_stage": "complete"},
            False,
        ),
        ("latent_timeout:TimeoutError", {}, True),
        ("worker_failure", {"episode_id": "e", "last_stage": "branch_select"}, True),
        (
            "latent_and_fallback_failed:NonFiniteLogitsError",
            {
                "episode_id": "e",
                "last_stage": "incumbent_restore",
                "resident_owner_released": True,
                "resident_state_reusable": True,
            },
            False,
        ),
        (
            "latent_and_fallback_failed:NonFiniteLogitsError",
            {
                "episode_id": "e",
                "last_stage": "incumbent_restore",
                "resident_owner_released": True,
                "resident_state_reusable": False,
            },
            True,
        ),
    ],
)
def test_latent_owner_disposition_is_explicit(reason, receipt, expected):
    assert latent_owner_exhausted(reason, receipt) is expected


def _async_value(value):
    async def _return(*_args, **_kwargs):
        return value

    return _return
