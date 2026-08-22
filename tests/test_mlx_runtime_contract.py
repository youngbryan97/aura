"""Runtime contract tests for MLX client/worker hardening."""

import asyncio
import contextlib
import logging
import os
import sys
import types

import pytest

from core.runtime.shutdown_coordinator import clear_shutdown_request, request_shutdown


def test_setup_worker_env_not_called_at_import():
    """_setup_worker_env must NOT run when the module is imported.

    If it does, the parent process inherits Metal-specific env vars that
    conflict with multi-process orchestration.
    """
    # Record current env state for key worker-only vars
    sentinel_keys = (
        "MLX_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "MLX_FORCE_SERIAL_COMPILE",
        "METAL_COMPILER_TIMEOUT_MS",
    )
    # Clear them so we can detect if import sets them
    saved = {}
    for key in sentinel_keys:
        saved[key] = os.environ.pop(key, None)

    try:
        # Force-reimport the module
        mod_name = "core.brain.llm.mlx_worker"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        # We can't fully reimport due to heavy deps, but we can verify the
        # module source doesn't call _setup_worker_env() at the top level
        import inspect

        from core.brain.llm import mlx_worker

        source = inspect.getsource(mlx_worker)
        # Find all top-level calls (not inside function bodies)
        # The fix moved _setup_worker_env() inside _mlx_worker_loop
        lines = source.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            # Top-level call (indent == 0) to _setup_worker_env
            if indent == 0 and stripped.startswith("_setup_worker_env("):
                raise AssertionError(
                    f"_setup_worker_env() called at module level (line {i}). "
                    "It must only be called inside _mlx_worker_loop()."
                )
    finally:
        for key, val in saved.items():
            if val is not None:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)


def test_bounded_max_tokens_narrowed_exceptions():
    """_bounded_max_tokens must not swallow arbitrary exceptions."""
    from core.brain.llm.mlx_client import _bounded_max_tokens

    # Normal operation
    assert _bounded_max_tokens(100, 200, 512) == 100
    assert _bounded_max_tokens(None, None, 256) == 256
    assert _bounded_max_tokens("", "", 128) == 128

    # Edge: non-numeric values should fallback, not crash
    assert _bounded_max_tokens("abc", 50, 64) == 50
    assert _bounded_max_tokens(50, object(), 64) == 50


def test_mlx_output_contract_ceiling_cannot_be_expanded_by_bridge():
    from core.brain.llm.mlx_client import _bounded_generation_max_tokens

    assert _bounded_generation_max_tokens(384, 768, 48, 4096) == 48
    assert _bounded_generation_max_tokens(24, 768, 48, 4096) == 24
    assert _bounded_generation_max_tokens(384, 128, None, 4096) == 128
    assert _bounded_generation_max_tokens(384, 128, "invalid", 4096) == 128


def test_adaptive_budget_cannot_make_exact_output_contract_impossible():
    from core.brain.llm.mlx_client import _bounded_generation_max_tokens

    contract = {
        "exact_reply": True,
        "exact_reply_utf8_bytes": 3,
        "semantic_token_cap": 8,
        "hard_token_ceiling": 19,
    }

    assert _bounded_generation_max_tokens(19, 3, 19, 4096, contract) == 4
    assert _bounded_generation_max_tokens(2, 1, 19, 4096, contract) == 2
    assert _bounded_generation_max_tokens(19, 3, 3, 4096, contract) == 3


def test_adaptive_budget_preserves_word_and_sentence_contract_floor():
    from core.brain.llm.mlx_client import _bounded_generation_max_tokens

    contract = {"kind": "sentence_count", "semantic_token_cap": 32}

    assert _bounded_generation_max_tokens(48, 3, 48, 4096, contract) == 32
    assert _bounded_generation_max_tokens(20, 3, 48, 4096, contract) == 20


def test_adaptive_bridge_cannot_erase_clean_user_surface_completion_reserve():
    from core.brain.llm.mlx_client import _bounded_generation_max_tokens

    assert (
        _bounded_generation_max_tokens(
            896,
            128,
            None,
            4096,
            user_surface_completion_floor=512,
            preserve_user_surface_completion_floor=True,
        )
        == 512
    )
    # The post-pressure caller cap remains authoritative under a critical
    # resource condition; the reserve cannot expand it back into danger.
    assert (
        _bounded_generation_max_tokens(
            64,
            32,
            None,
            4096,
            user_surface_completion_floor=512,
            preserve_user_surface_completion_floor=True,
        )
        == 64
    )
    assert (
        _bounded_generation_max_tokens(
            896,
            128,
            256,
            4096,
            user_surface_completion_floor=512,
            preserve_user_surface_completion_floor=True,
        )
        == 256
    )


def test_mlx_main_generation_builds_contract_cap_after_bridge_lookup():
    import inspect

    from core.brain.llm.mlx_client import MLXLocalClient

    source = inspect.getsource(MLXLocalClient._generate_inner)
    bridge_index = source.index("def _bridge_get")
    contract_index = source.index('requested_output_contract = kwargs.get("requested_output_contract")')
    request_index = source.index('"max_tokens": generation_max_tokens')

    assert bridge_index < contract_index < request_index
    assert "requested_output_contract" not in inspect.getsource(
        MLXLocalClient.generate_batch_async
    )


def test_mlx_surface_receipt_reports_contract_tokens_and_repair():
    from core.brain.llm.mlx_client import MLXLocalClient
    from core.brain.llm.mlx_worker import _surface_generation_control_receipt

    contract = {
        "kind": "sentence_count",
        "sentence_count": 1,
        "semantic_token_cap": 32,
        "hard_token_ceiling": 48,
    }
    worker_receipt = _surface_generation_control_receipt(
        {
            "max_tokens": 48,
            "clean_user_surface_contract": True,
            "requested_output_contract": contract,
            "semantic_output_token_cap": 32,
            "hard_output_token_ceiling": 48,
        },
        {
            "enabled": True,
            "generation_stop_reason": "configured_stop",
            "generation_configured_stop_sequence": "<|im_start|>",
            "instruction_shape_repair_applied": True,
            "surface_quality_gate_enabled": True,
            "surface_quality_gate_passed": True,
        },
    )
    client = MLXLocalClient.__new__(MLXLocalClient)
    client._last_surface_control_receipt = {}
    client._record_surface_control_receipt_from_response(
        {
            "surface_control_receipt": worker_receipt,
            "tokens_used": 11,
        }
    )

    receipt = client.get_last_surface_control_receipt()
    assert receipt["generation_max_tokens"] == 48
    assert receipt["generated_tokens"] == 11
    assert receipt["generation_stop_reason"] == "configured_stop"
    assert receipt["generation_configured_stop_sequence"] == "<|im_start|>"
    assert receipt["instruction_shape_repair_applied"] is True
    assert receipt["requested_output_contract"]["kind"] == "sentence_count"


def test_worker_retry_budget_never_expands_past_output_contract():
    from core.brain.llm.mlx_worker import (
        _expand_user_surface_retry_budget,
        _surface_generation_control_receipt,
    )

    constrained = {"max_tokens": 48}
    assert (
        _expand_user_surface_retry_budget(
            constrained,
            ["truncated_tail"],
            hard_ceiling=48,
        )
        is False
    )
    assert constrained["max_tokens"] == 48

    unconstrained = {"max_tokens": 48}
    assert _expand_user_surface_retry_budget(unconstrained, ["truncated_tail"])
    assert unconstrained["max_tokens"] == 432

    receipt = _surface_generation_control_receipt(
        {"max_tokens": 48, "hard_output_token_ceiling": 48},
        {"generation_max_tokens_applied": constrained["max_tokens"]},
    )
    assert receipt["generation_max_tokens"] == 48
    assert receipt["hard_output_token_ceiling"] == 48


def test_worker_count_contract_retry_demands_semantic_task_ownership():
    from core.brain.llm.mlx_worker import _messages_with_user_surface_retry

    retry_messages = _messages_with_user_surface_retry(
        [
            {"role": "system", "content": "Answer accurately."},
            {
                "role": "user",
                "content": "In exactly five words, state why checksums matter.",
            },
        ],
        ["missing_requested_word_count"],
        {
            "user_surface_validation_prompt": (
                "In exactly five words, state why checksums matter."
            ),
            "requested_output_contract": {
                "kind": "word_count",
                "word_min": 5,
                "word_max": 5,
            }
        },
    )

    assert retry_messages is not None
    system = retry_messages[0]["content"]
    assert "Solve the current semantic task first" in system
    assert "retain a concrete topic noun" in system
    assert "never describe the word or sentence constraint" in system
    assert "exactly 5 words" in system
    assert "current-topic terms" in system
    assert "checksum" in system
    assert "Count the final visible answer" in system


def test_worker_never_expands_admitted_cap_for_mode_specific_contracts():
    import inspect

    from core.brain.llm import mlx_worker

    source = inspect.getsource(mlx_worker._mlx_worker_loop)
    operator_cap = source.index(
        "if operator_evidence_contract:\n                    max_tokens = min(max_tokens, 192)"
    )
    hard_ceiling = source.index(
        'hard_output_token_ceiling = _safe_int(\n                    job.get("hard_output_token_ceiling")'
    )
    kwargs_build = source.index('kwargs = {"max_tokens": max_tokens')

    assert operator_cap < hard_ceiling < kwargs_build
    assert "max_tokens = max(max_tokens, min(exact_token_requirement" not in source


@pytest.mark.asyncio
async def test_mlx_surface_receipts_are_task_scoped_and_empty_resets_stale_state():
    from core.brain.llm.mlx_client import MLXLocalClient

    client = MLXLocalClient.__new__(MLXLocalClient)
    client._last_surface_control_receipt = {"request": "legacy"}
    client._set_task_surface_control_receipt({})
    assert client.get_last_surface_control_receipt() == {}

    async def task_receipt(request: str) -> dict:
        client._set_task_surface_control_receipt({"request": request})
        await asyncio.sleep(0)
        return client.get_last_surface_control_receipt()

    first, second = await asyncio.gather(task_receipt("first"), task_receipt("second"))

    assert first == {"request": "first"}
    assert second == {"request": "second"}
    assert client.get_last_surface_control_receipt() == {}


@pytest.mark.asyncio
async def test_mlx_fresh_task_never_borrows_global_surface_receipt():
    from core.brain.llm.mlx_client import MLXLocalClient

    client = MLXLocalClient.__new__(MLXLocalClient)
    client._last_surface_control_receipt = {}
    published = asyncio.Event()

    async def _writer() -> None:
        client._set_task_surface_control_receipt({"request": "writer"})
        client._last_surface_control_receipt = {"request": "writer"}
        published.set()

    async def _reader() -> dict:
        await published.wait()
        return client.get_last_surface_control_receipt()

    _, reader_receipt = await asyncio.gather(_writer(), _reader())

    assert reader_receipt == {}
    assert client.get_diagnostic_last_surface_control_receipt() == {
        "request": "writer"
    }


def test_surface_receipt_preserves_a_bounded_rejected_draft():
    from core.brain.llm.mlx_client import _sanitize_surface_control_receipt

    receipt = _sanitize_surface_control_receipt(
        {
            "surface_quality_gate_enabled": True,
            "surface_quality_gate_passed": False,
            "surface_quality_gate_reasons": ["corrupted_language"],
            "surface_quality_rejected_text": "x" * 9_000,
        }
    )

    assert receipt["surface_quality_rejected_text"] == "x" * 8_000


def test_worker_rejects_and_repairs_exact_reply_mismatch_before_ipc_success():
    from core.brain.llm.mlx_worker import (
        _repair_live_user_surface_instruction_shape,
        _surface_quality_failure_reasons,
    )

    job = {
        "clean_user_surface_contract": True,
        "user_surface_validation_prompt": "Please answer exactly: yes",
    }
    mismatch = "No, I disagree with that requested token."

    assert "missing_requested_exact_reply" in _surface_quality_failure_reasons(
        job,
        mismatch,
    )
    repaired = _repair_live_user_surface_instruction_shape(job, mismatch)
    assert repaired == "yes"
    assert _surface_quality_failure_reasons(job, repaired) == []


def test_exact_reply_ceiling_covers_selected_tokenizer_requirement():
    from core.brain.llm.mlx_worker import _exact_reply_token_requirement
    from core.conversation.response_reliability import requested_output_contract

    class _ByteTokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            return list(str(text).encode("utf-8"))

    target = "Aa0!" * 20
    prompt = f'Reply exactly: "{target}"'
    contract = requested_output_contract(prompt)
    job = {
        "user_surface_validation_prompt": prompt,
        "requested_output_contract": contract.as_dict(),
        "hard_output_token_ceiling": contract.hard_token_ceiling,
    }

    token_count, requirement = _exact_reply_token_requirement(job, _ByteTokenizer())

    assert token_count == len(target.encode("utf-8"))
    assert requirement == token_count + 1
    assert contract.hard_token_ceiling >= requirement


def test_exact_reply_token_evidence_preserves_immutable_admitted_caps():
    from core.brain.llm.mlx_worker import _record_exact_reply_token_evidence

    class _OneTokenTokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            del text, add_special_tokens
            return [7]

    job = {
        "max_tokens": 1,
        "hard_output_token_ceiling": 1,
        "user_surface_validation_prompt": 'Reply exactly: "X"',
        "requested_output_contract": {
            "exact_reply": True,
            "hard_token_ceiling": 1,
        },
    }

    _record_exact_reply_token_evidence(
        job,
        _OneTokenTokenizer(),
        generation_max_tokens=job["max_tokens"],
        hard_output_token_ceiling=job["hard_output_token_ceiling"],
    )

    assert job["max_tokens"] == 1
    assert job["hard_output_token_ceiling"] == 1
    assert job["requested_output_contract"]["hard_token_ceiling"] == 1
    assert job["exact_reply_token_count"] == 1
    assert job["exact_reply_required_termination_headroom"] == 1
    assert job["exact_reply_available_termination_headroom"] == 0
    assert job["exact_reply_content_capacity_sufficient"] is True
    assert job["exact_reply_termination_headroom_sufficient"] is False
    assert job["exact_reply_native_capacity_sufficient"] is False
    assert job["exact_reply_token_ceiling_valid"] is False


def test_exact_reply_token_evidence_reports_native_capacity_with_one_stop_slot():
    from core.brain.llm.mlx_worker import _record_exact_reply_token_evidence

    class _OneTokenTokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            del text, add_special_tokens
            return [7]

    job = {
        "max_tokens": 3,
        "hard_output_token_ceiling": 16,
        "user_surface_validation_prompt": 'Reply exactly: "yes"',
        "requested_output_contract": {"exact_reply": True},
    }

    _record_exact_reply_token_evidence(
        job,
        _OneTokenTokenizer(),
        generation_max_tokens=job["max_tokens"],
        hard_output_token_ceiling=job["hard_output_token_ceiling"],
    )

    assert job["exact_reply_token_count"] == 1
    assert job["exact_reply_required_termination_headroom"] == 1
    assert job["exact_reply_available_termination_headroom"] == 2
    assert job["exact_reply_content_capacity_sufficient"] is True
    assert job["exact_reply_termination_headroom_sufficient"] is True
    assert job["exact_reply_native_capacity_sufficient"] is True
    assert job["exact_reply_token_ceiling_valid"] is True


def test_worker_receipt_preserves_ordered_text_mutation_provenance():
    from core.brain.live_mind_contract import append_text_mutation
    from core.brain.llm.mlx_worker import _surface_generation_control_receipt

    state = {"enabled": False, "text_mutations": []}
    append_text_mutation(
        state,
        stage="mlx_worker.truncated_tail",
        method="retain_complete_sentences",
        reasons=["truncated_tail"],
        before="Complete sentence. clipped",
        after="Complete sentence.",
        deterministic=True,
    )

    receipt = _surface_generation_control_receipt(
        {"max_tokens": 48, "clean_user_surface_contract": True},
        state,
    )

    assert receipt["text_mutation_count"] == 1
    assert receipt["text_mutations"][0]["stage"] == "mlx_worker.truncated_tail"
    assert receipt["deterministic_repair_applied"] is True


def test_text_mutation_sequence_preserves_distinct_same_shaped_events():
    from core.brain.live_mind_contract import append_text_mutation, merge_text_mutations

    receipt = {"text_mutations": []}
    append_text_mutation(
        receipt,
        stage="same-stage",
        method="same-method",
        reasons=["same-reason"],
        before="aa",
        after="bb",
        deterministic=True,
    )
    append_text_mutation(
        receipt,
        stage="same-stage",
        method="same-method",
        reasons=["same-reason"],
        before="cc",
        after="dd",
        deterministic=True,
    )

    assert receipt["text_mutation_count"] == 2
    assert [item["sequence"] for item in receipt["text_mutations"]] == [1, 2]

    independent = {"text_mutations": []}
    append_text_mutation(
        independent,
        stage="same-stage",
        method="same-method",
        reasons=["same-reason"],
        before="ee",
        after="ff",
        deterministic=True,
    )
    merged = merge_text_mutations(
        receipt["text_mutations"][:1],
        independent["text_mutations"],
    )
    assert len(merged) == 2
    assert [item["sequence"] for item in merged] == [1, 2]
    assert len({item["event_id"] for item in merged}) == 2
    assert len(merge_text_mutations(merged, merged)) == 2


def test_text_mutation_authorship_is_typed_and_legacy_entries_fail_closed():
    from core.brain.live_mind_contract import (
        append_text_mutation,
        normalize_text_mutations,
    )

    legacy = normalize_text_mutations(
        [
            {
                "stage": "legacy",
                "method": "unknown_rewrite",
                "before_sha256": "a" * 64,
                "after_sha256": "b" * 64,
            }
        ]
    )
    assert legacy[0]["authorship_effect"] == "replaced_by_runtime"

    receipt = {"text_mutations": []}
    append_text_mutation(
        receipt,
        stage="response_generation.voice",
        method="surface_cleanup",
        reasons=["voice_profile"],
        before="raw",
        after="polished",
        deterministic=True,
        authorship_effect="preserved",
    )
    assert receipt["text_mutations"][0]["authorship_effect"] == "preserved"
    assert receipt["authorship_replacement_applied"] is False
    assert receipt["authorship_augmentation_applied"] is False
    assert receipt["model_replacement_applied"] is False


def test_text_mutation_chain_cryptographically_binds_exact_public_bytes():
    import hashlib

    from core.brain.live_mind_contract import (
        append_text_mutation,
        verify_text_mutation_chain,
    )

    raw = "Complete model answer."
    intermediate = "Complete model answer.\n\nVerified."
    public = "Complete model answer.\n\nVerified and delivered."
    receipt = {"text_mutations": []}
    append_text_mutation(
        receipt,
        stage="response_generation.voice",
        method="join_complete_chunks",
        reasons=["voice_profile"],
        before=raw,
        after=intermediate,
        deterministic=False,
    )
    append_text_mutation(
        receipt,
        stage="chat.final_contract",
        method="deterministic_instruction_shape",
        reasons=["verification_detail"],
        before=intermediate,
        after=public,
        deterministic=True,
    )

    proof = verify_text_mutation_chain(
        receipt["text_mutations"],
        before_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        after_sha256=hashlib.sha256(public.encode()).hexdigest(),
    )

    assert proof["passed"] is True
    assert proof["chain_length"] == 2
    assert receipt["text_mutations"][0]["before_sha256"] == hashlib.sha256(
        raw.encode()
    ).hexdigest()


def test_text_mutation_chain_rejects_tampered_or_disconnected_ledger():
    import hashlib

    from core.brain.live_mind_contract import (
        append_text_mutation,
        verify_text_mutation_chain,
    )

    receipt = {"text_mutations": []}
    append_text_mutation(
        receipt,
        stage="response_generation.clean",
        method="cleanup",
        reasons=["surface_cleanup"],
        before="raw",
        after="clean",
        deterministic=True,
    )
    tampered = [dict(receipt["text_mutations"][0])]
    tampered[0]["after_sha256"] = "f" * 64

    proof = verify_text_mutation_chain(
        tampered,
        before_sha256=hashlib.sha256(b"raw").hexdigest(),
        after_sha256=hashlib.sha256(b"clean").hexdigest(),
    )

    assert proof["passed"] is False
    assert proof["reasons"] == ["mutation_hash_chain_mismatch"]


def test_mlx_degradation_records_have_explicit_runtime_actions():
    """MLX failures must connect to recovery behavior, not only log telemetry."""
    from pathlib import Path

    from tools.audit_degradation import analyze_file

    assert analyze_file(Path("core/brain/llm/mlx_client.py")) == []


def test_mlx_runtime_probe_subprocess_is_bounded_and_reviewed():
    """The MLX probe subprocess must stay governed and bounded."""
    import inspect

    from core.brain.llm import mlx_client
    from tools.aura_enterprise_gate import subprocess_must_use_gateway

    source = inspect.getsource(mlx_client._probe_mlx_runtime)
    assert subprocess_must_use_gateway("core/brain/llm/mlx_client.py") is True
    assert "get_subprocess_gateway().run(" in source
    # The probe timeout became an operator-configurable bound rather than a
    # hardcoded 25.0: on a host whose page cache is thrashing, importing MLX
    # alone can exceed a fixed budget and the probe reported
    # "mlx_runtime_unavailable:exit_124" on a perfectly healthy machine. What
    # this contract requires is that the call stays BOUNDED, and that the
    # bound has a floor so it cannot be configured away.
    assert "timeout=_MLX_RUNTIME_PROBE_TIMEOUT_S" in source
    probe_timeout = float(mlx_client._MLX_RUNTIME_PROBE_TIMEOUT_S)
    assert 5.0 <= probe_timeout <= 600.0
    assert "source=\"runtime_probe:mlx_runtime_probe\"" in source
    assert "read_only=True" in source
    assert "AURA_TEST_MODE" not in source
    assert "shell=True" not in source


def test_worker_health_probe_bypasses_prompt_cache():
    from core.brain.llm import mlx_worker

    assert mlx_worker._job_requires_prompt_cache_bypass({"health_probe": True}) is True


def test_worker_only_accepts_empty_output_for_one_token_warmup_precompile():
    from core.brain.llm import mlx_worker

    assert (
        mlx_worker._expected_empty_warmup_precompile(
            {"warmup_precompile": True, "max_tokens": 1}
        )
        is True
    )
    assert (
        mlx_worker._expected_empty_warmup_precompile(
            {"warmup_precompile": True, "max_tokens": 16}
        )
        is False
    )
    assert (
        mlx_worker._expected_empty_warmup_precompile(
            {"health_probe": True, "max_tokens": 1}
        )
        is False
    )


def test_record_mlx_degradation_preserves_action_and_severity():
    from core.brain.llm.mlx_client import _record_mlx_degradation
    from core.runtime.errors import get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()

    _record_mlx_degradation(
        RuntimeError("spawn wedged"),
        action="marked lane failed and applied spawn backoff",
        severity="error",
    )

    recent = tracker.recent(subsystem="mlx_client", limit=1)
    assert recent
    assert recent[0].severity == "error"
    assert recent[0].action == "marked lane failed and applied spawn backoff"


@pytest.mark.asyncio
async def test_mlx_warmup_refuses_before_worker_recovery_after_shutdown(monkeypatch):
    from core.brain.llm.mlx_client import MLXLocalClient

    clear_shutdown_request()
    client = MLXLocalClient("/models/Aura-32B-test")
    calls: list[str] = []

    async def _should_not_recover(*_args, **_kwargs):
        calls.append("recover")
        return True

    monkeypatch.setattr(client, "_ensure_worker_alive", _should_not_recover)

    try:
        request_shutdown("unit-test")
        assert await client.warmup(foreground_request=False) is False
    finally:
        clear_shutdown_request()

    assert calls == []
    assert client._warmup_attempted is False
    assert client._warmup_in_flight is False


def test_mlx_worker_spawn_refuses_before_orphan_scan_after_shutdown(monkeypatch):
    from core.brain.llm import mlx_client
    from core.brain.llm.mlx_client import MLXLocalClient

    clear_shutdown_request()
    client = MLXLocalClient("/models/Aura-32B-test")

    def _should_not_scan(*_args, **_kwargs):
        raise AssertionError("shutdown latch must stop spawn before process scans")

    monkeypatch.setattr(mlx_client.psutil, "process_iter", _should_not_scan)

    try:
        request_shutdown("unit-test")
        with pytest.raises(RuntimeError, match="runtime_shutdown"):
            client._spawn_worker_blocking()
    finally:
        clear_shutdown_request()


def test_mlx_worker_spawn_rechecks_shutdown_after_file_lock(monkeypatch, tmp_path):
    from core.brain.llm import mlx_client
    from core.brain.llm.mlx_client import MLXLocalClient

    client = MLXLocalClient.__new__(MLXLocalClient)
    client.model_path = "/models/Aura-32B-test"
    client._req_q = object()
    client._res_q = object()
    client.device = "gpu"
    client._substrate_mem = None
    client._steering_active = None
    client._cancel_seq = None
    client._mp_context = types.SimpleNamespace(
        Process=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("worker Process constructed after shutdown crossed file lock")
        )
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(mlx_client.psutil, "process_iter", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(mlx_client, "_memory_pressure_blocks_worker_spawn", lambda *_args: None)
    monkeypatch.setattr(mlx_client, "_probe_mlx_runtime", lambda: (True, "ok"))

    def _flock(_file, operation):
        if operation & mlx_client.fcntl.LOCK_NB:
            request_shutdown("crossed-file-lock")

    monkeypatch.setattr(mlx_client.fcntl, "flock", _flock)

    with pytest.raises(RuntimeError, match="runtime_shutdown"):
        client._spawn_worker_blocking()


@pytest.mark.asyncio
async def test_warmup_precompile_shutdown_cancellation_is_not_degradation(monkeypatch):
    from core.brain.llm import mlx_client
    from core.brain.llm.mlx_client import MLXLocalClient

    client = MLXLocalClient.__new__(MLXLocalClient)
    client.model_path = "/models/Aura-32B"

    async def cancelled_generation(*_args, **_kwargs):
        await asyncio.sleep(0)
        raise asyncio.CancelledError()

    monkeypatch.setattr(client, "_generate_inner", cancelled_generation)
    monkeypatch.setattr(mlx_client, "is_shutdown_requested", lambda: True)
    monkeypatch.setattr(
        mlx_client,
        "_record_mlx_degradation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("shutdown cancellation must not record MLX degradation")
        ),
    )

    with pytest.raises(asyncio.CancelledError):
        await client._run_warmup_precompile(
            request_is_background=False,
            foreground_request=True,
            owner_name="warmup:Aura-32B",
            warmup_timeout=1.0,
        )


def _install_worker_fakes(monkeypatch, mlx_worker, *, load_impl, steering_engine=None):
    class FakeQueue:
        def __init__(self, items=()):
            self.items = list(items)
            self.writes = []

        def get(self, *_, timeout=None, **__):
            # Mirrors the real loop contract: the worker polls with a bounded
            # timeout. Over-reading an exhausted queue is still a test defect.
            if not self.items:
                raise AssertionError("worker read from an empty fake request queue")
            return self.items.pop(0)

        def put(self, item, *_, **__):
            self.writes.append(item)

    class FakeIPCWriter:
        def __init__(self, response_queue):
            self.response_queue = response_queue
            # The real writer exposes a broken-pipe event the loop checks
            # every iteration; without it every pass raised AttributeError
            # into the catch-and-continue handler — an infinite spin.
            self.broken = types.SimpleNamespace(is_set=lambda: False)

        def start(self):
            return None

        def put(self, item):
            self.response_queue.put(item)

    class FakeWorkerThread:
        def __init__(self, *_, **__):
            return None

        def start(self, *_, **__):
            return None

        def start_job(self, *_, **__):
            return None

        def activity(self, *_, **__):
            return None

        def stop_job(self, *_, **__):
            return None

    mlx_core = types.ModuleType("mlx.core")
    mlx_core.cpu = object()
    mlx_core.gpu = object()
    mlx_core._default_device = mlx_core.cpu
    mlx_core.default_device = lambda: mlx_core._default_device
    mlx_core.set_default_device = lambda device: setattr(
        mlx_core, "_default_device", device
    )
    mlx_core.set_cache_limit = lambda *_args, **_kwargs: None
    mlx_core.set_memory_limit = lambda *_args, **_kwargs: None
    mlx_core.clear_cache = lambda *_args, **_kwargs: None
    mlx_core.metal = types.SimpleNamespace(
        set_cache_limit=lambda *_args, **_kwargs: None,
        clear_cache=lambda *_args, **_kwargs: None,
    )
    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = mlx_core
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.load = load_impl
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **_kwargs: object()

    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    monkeypatch.setattr(mlx_worker, "_setup_worker_env", lambda: None)
    monkeypatch.setattr(mlx_worker, "resolve_personality_adapter", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mlx_worker, "IPCWriterThread", FakeIPCWriter)
    monkeypatch.setattr(mlx_worker, "HeartbeatThread", FakeWorkerThread)
    monkeypatch.setattr(mlx_worker, "WorkerMemorySentinel", FakeWorkerThread)
    monkeypatch.setattr(mlx_worker, "JobWatchdog", FakeWorkerThread)

    if steering_engine is not None:
        steering_mod = types.ModuleType("core.consciousness.affective_steering")
        steering_mod.get_steering_engine = lambda: steering_engine
        monkeypatch.setitem(sys.modules, "core.consciousness.affective_steering", steering_mod)

    return FakeQueue


def test_worker_init_failure_exits_before_accepting_jobs(monkeypatch):
    from core.brain.llm import mlx_worker
    from core.runtime.errors import get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()

    load_failures = []

    def load_failure(*_args, **_kwargs):
        load_failures.append((_args, _kwargs))
        raise RuntimeError("model load failed")

    queue_factory = _install_worker_fakes(monkeypatch, mlx_worker, load_impl=load_failure)
    requests = queue_factory([{"action": "generate", "prompt": "must not be read"}])
    responses = queue_factory()

    mlx_worker._mlx_worker_loop("fake-model", requests, responses)

    assert len(load_failures) == 1
    assert requests.items == [{"action": "generate", "prompt": "must not be read"}]
    assert responses.writes[-1]["status"] == "error"
    assert responses.writes[-1]["action"] == "init"
    recent = tracker.recent(subsystem="mlx_worker", limit=1)
    assert recent
    assert recent[0].severity == "critical"
    assert recent[0].action == "reported initialization error and exited worker loop before accepting jobs"


def test_worker_shutdown_releases_every_owned_resource(monkeypatch):
    from core.brain.llm import mlx_worker

    calls = []

    class Resource:
        def __init__(self, name):
            self.name = name

        def stop(self):
            calls.append((self.name, "stop"))

        def stop_job(self):
            calls.append((self.name, "stop_job"))

        def clear(self):
            calls.append((self.name, "clear"))

        def join(self, timeout=None):
            calls.append((self.name, "join", timeout))

    writer = Resource("writer")
    writer.local_queue = types.SimpleNamespace(empty=lambda: True)
    mx = types.SimpleNamespace(
        synchronize=lambda: calls.append(("mlx", "synchronize")),
        clear_cache=lambda: calls.append(("mlx", "clear_cache")),
        metal=types.SimpleNamespace(
            clear_cache=lambda: calls.append(("mlx.metal", "clear_cache"))
        ),
    )
    monkeypatch.setattr(mlx_worker.gc, "collect", lambda: calls.append(("gc", "collect")))

    mlx_worker._shutdown_worker_runtime(
        ipc_writer=writer,
        watchdog=Resource("watchdog"),
        heartbeat=Resource("heartbeat"),
        memory_sentinel=Resource("memory"),
        latent_bridge=Resource("bridge"),
        steering_engine=Resource("steering"),
        prompt_cache_lru=Resource("cache"),
        mx_module=mx,
    )

    assert ("watchdog", "stop_job") in calls
    assert ("bridge", "stop") in calls
    assert ("steering", "stop") in calls
    assert ("cache", "clear") in calls
    assert ("mlx", "synchronize") in calls
    assert ("writer", "stop") in calls
    assert ("heartbeat", "stop") in calls
    assert ("memory", "stop") in calls
    assert ("watchdog", "stop") in calls
    assert {call[0] for call in calls if len(call) >= 2 and call[1] == "join"} == {
        "heartbeat",
        "memory",
        "watchdog",
        "writer",
    }


def test_optional_steering_cannot_gate_model_generation():
    import inspect

    from core.brain.llm.mlx_worker import _mlx_worker_loop

    source = inspect.getsource(_mlx_worker_loop)
    generate_start = source.index('if action == "generate":')
    prompt_start = source.index('prompt = job.get("prompt")', generate_start)
    admission = source[generate_start:prompt_start]

    assert "engine.is_active" not in admission
    assert "generation blocked" not in admission
    assert "steering liveness" not in admission


def test_response_listener_shutdown_awareness():
    """The response listener loop condition should check shutdown state."""
    import inspect

    from core.brain.llm.mlx_client import MLXLocalClient

    source = inspect.getsource(MLXLocalClient._response_listener_loop)
    forbidden_loop = "while " + "True"
    assert "_runtime_shutdown_requested()" in source, (
        "_response_listener_loop must check _runtime_shutdown_requested() "
        "instead of using an unbounded loop"
    )
    assert forbidden_loop not in source, (
        "_response_listener_loop should not use an unbounded loop; it must be shutdown-aware"
    )


def test_worker_docstring_placement():
    """_mlx_worker_loop must have its docstring as the first thing in the body."""
    from core.brain.llm.mlx_worker import _mlx_worker_loop

    assert _mlx_worker_loop.__doc__ is not None, "_mlx_worker_loop is missing its docstring"
    assert "subprocess" in _mlx_worker_loop.__doc__.lower(), (
        "_mlx_worker_loop docstring should mention subprocess isolation"
    )


def test_no_duplicate_consecutive_empty_reset():
    """_consecutive_empty should only be reset once on successful generation."""
    import inspect

    from core.brain.llm.mlx_client import MLXLocalClient

    source = inspect.getsource(MLXLocalClient._generate_inner)

    # The success path is the block between "res.get('status') == 'ok'" and
    # the next except clause. Count resets in the full method — there should
    # be exactly one on the success path, not two adjacent lines.
    # A duplicate was the original bug; verify it stays gone.
    lines_with_reset = [
        line.strip() for line in source.splitlines() if "self._consecutive_empty = 0" in line
    ]
    assert len(lines_with_reset) == 1, (
        f"Expected exactly 1 reset of _consecutive_empty in _generate_inner, "
        f"found {len(lines_with_reset)}. Duplicate resets obscure intent."
    )


def test_stream_path_no_mid_generation_cache_clear():
    """The stream path must NOT clear MLX cache every N tokens.

    Mid-generation cache clearing forces Metal to reallocate GPU memory,
    creating micro-stalls that degrade token throughput. Post-generation
    cleanup is sufficient.
    """
    import inspect

    from core.brain.llm.mlx_worker import _mlx_worker_loop

    source = inspect.getsource(_mlx_worker_loop)
    # Find the stream action handler
    stream_start = source.find('elif action == "stream":')
    assert stream_start > 0, "Could not find stream action handler"
    stream_section = source[stream_start:]

    # The old pattern was: if token_count % 10 == 0: _clear_mlx_cache(mx)
    assert "token_count % 10" not in stream_section, (
        "Stream path still contains mid-generation cache clearing every 10 tokens. "
        "This was identified as harmful to throughput and must be removed."
    )


def test_stream_path_uses_surface_generation_controls():
    """Streaming user-visible text must use the same controls as normal generation.

    The desktop UI commonly consumes streamed tokens. If this path bypasses
    the surface clamp, live chat can diverge from proof/backend generation.
    """
    import inspect

    from core.brain.llm.mlx_worker import _mlx_worker_loop

    source = inspect.getsource(_mlx_worker_loop)
    stream_start = source.find('elif action == "stream":')
    assert stream_start > 0, "Could not find stream action handler"
    stream_section = source[stream_start:]
    apply_idx = stream_section.find("_apply_surface_generation_controls")
    generate_idx = stream_section.find("stream_generate(")
    restore_idx = stream_section.find("_restore_surface_generation_controls")

    assert apply_idx > 0, "Stream path does not apply surface generation controls."
    assert generate_idx > 0, "Stream path does not call stream_generate."
    assert restore_idx > 0, "Stream path does not restore surface generation controls."
    assert apply_idx < generate_idx < restore_idx, (
        "Stream path must apply surface controls before token generation and "
        "restore them after generation exits."
    )


def test_latent_reason_path_applies_and_restores_surface_controls():
    import inspect

    from core.brain.llm.mlx_worker import _mlx_worker_loop

    source = inspect.getsource(_mlx_worker_loop)
    latent_start = source.find('elif action == "latent_reason":')
    assert latent_start > 0, "Could not find latent_reason action handler"
    latent_section = source[latent_start:]
    apply_idx = latent_section.find("_apply_surface_generation_controls")
    reason_idx = latent_section.find("handle_latent_reason(")
    restore_idx = latent_section.find("_restore_surface_generation_controls")

    assert apply_idx > 0
    assert reason_idx > 0
    assert restore_idx > 0
    assert apply_idx < reason_idx < restore_idx


@pytest.mark.asyncio
async def test_primary_lane_warmup_exempt_from_foreground_owner_deferral(monkeypatch):
    """Priority-inversion pin (lived 2026-07-10): a turn owning the
    foreground lane deferred the PRIMARY lane's own warmup precompile —
    the exact work the turn was waiting on — deadlocking the cortex into
    warming/recovering for 75 minutes. The primary's warmup must proceed
    under an owned foreground; only non-primary background lanes defer."""
    from core.brain.llm import mlx_client
    from core.brain.llm.mlx_client import MLXLocalClient

    client = MLXLocalClient("/models/Aura-32B-cortex-test")
    assert client._is_primary_lane() is True

    ran: list[str] = []

    async def _alive(*_a, **_k):
        return True

    async def _precompile(*_a, **_k):
        ran.append("precompile")

    monkeypatch.setattr(client, "_ensure_worker_alive", _alive)
    monkeypatch.setattr(client, "_run_warmup_precompile", _precompile)
    monkeypatch.setattr(client, "_prove_conversation_path", _precompile, raising=False)
    monkeypatch.setattr(mlx_client, "_foreground_owner_active", lambda: True)

    await client.warmup(foreground_request=False)
    assert "precompile" in ran, (
        "primary-lane warmup must run its precompile even while the "
        "foreground lane is owned"
    )


@pytest.mark.asyncio
async def test_primary_lane_recovery_exempt_from_foreground_owner_guards(monkeypatch):
    """Second half of the 2026-07-10 inversion family: with the cortex dead,
    the Reflex FALLBACK serving turns owned the foreground, and the spawn
    guards then blocked the primary's own background recovery — the cortex
    could never return while its fallback answered for it. Primary recovery
    must pass every foreground-owner / gate-quiet guard."""
    from core.brain.llm import mlx_client
    from core.brain.llm.mlx_client import MLXLocalClient

    client = MLXLocalClient("/models/Aura-32B-cortex-test")
    assert client._is_primary_lane() is True

    reached: list[str] = []

    async def _inner(**_kw):
        reached.append("inner")
        return True

    monkeypatch.setattr(client, "_ensure_worker_alive_inner", _inner)
    monkeypatch.setattr(mlx_client, "_foreground_owner_active", lambda: True)
    monkeypatch.setattr(
        mlx_client, "_background_deferral_active", lambda *_a, **_k: "foreground_reserved"
    )

    @contextlib.asynccontextmanager
    async def _admitted(_client, *, foreground_request):
        del foreground_request
        yield None

    monkeypatch.setattr(mlx_client, "_model_load_admission_context", _admitted)

    result = await client._ensure_worker_alive(request_is_background=True)
    assert reached == ["inner"], (
        "primary-lane background recovery must reach the spawn path despite "
        "foreground ownership and gate quiet policy"
    )
    assert result is True


@pytest.mark.asyncio
async def test_non_primary_background_recovery_still_yields(monkeypatch):
    """The anti-thrash side of the same law: a brainstem background recovery
    still defers while the foreground lane is owned."""
    from core.brain.llm import mlx_client
    from core.brain.llm.mlx_client import MLXLocalClient

    client = MLXLocalClient("/models/Qwen2.5-7B-Instruct-4bit")
    assert client._is_primary_lane() is False

    reached: list[str] = []

    async def _inner(**_kw):
        reached.append("inner")
        return True

    monkeypatch.setattr(client, "_ensure_worker_alive_inner", _inner)
    monkeypatch.setattr(mlx_client, "_foreground_owner_active", lambda: True)

    result = await client._ensure_worker_alive(request_is_background=True)
    assert reached == []
    assert result is False


@pytest.mark.asyncio
async def test_model_load_admission_denial_backoff_suppresses_background_retry_storm(
    monkeypatch,
    caplog,
):
    from core.brain.llm import mlx_client
    from core.brain.llm.mlx_client import MLXLocalClient

    client = MLXLocalClient("/models/Qwen2.5-1.5B-Instruct-4bit")
    attempts = []

    @contextlib.asynccontextmanager
    async def _denied(_client, *, foreground_request):
        attempts.append(foreground_request)
        raise mlx_client._ModelLoadAdmissionDeniedError(
            "event_loop_lag_3.638s",
            receipt_id=f"receipt-{len(attempts)}",
        )
        yield  # pragma: no cover - required async-contextmanager shape

    monkeypatch.setattr(mlx_client, "_model_load_admission_context", _denied)
    monkeypatch.setattr(mlx_client, "_foreground_owner_active", lambda: False)
    monkeypatch.setattr(mlx_client, "_background_deferral_active", lambda *_a, **_k: None)
    caplog.set_level(logging.INFO, logger="LLM.MLX")

    first = await client._ensure_worker_alive(request_is_background=True)
    second = await client._ensure_worker_alive(request_is_background=True)

    assert first is False
    assert second is False
    assert attempts == [False]
    status = client.get_lane_status()["model_load_admission"]
    assert status["backing_off"] is True
    assert status["reason"] == "event_loop_lag_3.638s"
    assert status["receipt_id"] == "receipt-1"
    assert status["denial_count"] == 1
    assert status["suppressed_calls"] == 1
    background_messages = [
        record
        for record in caplog.records
        if "Model-load admission deferred" in record.getMessage()
    ]
    assert background_messages
    assert all(record.levelno == logging.INFO for record in background_messages)

    # User-facing work bypasses a background retry delay and asks the current
    # control plane directly; admission policy, not stale client state, decides.
    foreground = await client._ensure_worker_alive(foreground_request=True)

    assert foreground is False
    assert attempts == [False, True]
