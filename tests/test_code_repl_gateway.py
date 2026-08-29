from __future__ import annotations

import json
from pathlib import Path

import pytest

import core.skills.code_repl as code_repl_mod
from core.brain.llm import mlx_client
from core.skills.code_repl import (
    CODE_REPL_MODEL_RESULT_FIELDS,
    CODE_REPL_MODEL_RESULT_SCHEMA,
    CodeREPLSkill,
    normalize_code_repl_model_result,
)


def test_code_repl_model_result_normalization_is_stable_and_minimal() -> None:
    normalized = normalize_code_repl_model_result(
        {
            "ok": True,
            "status": "ok",
            "stdout": "42\n",
            "stderr": "",
            "returncode": 0,
            "engine": "sandbox_runner",
            "summary": "done",
            "session_id": "private-session",
            "working_directory": "/private/tmp",
            "duration_ms": 12.5,
            "retries": 1,
            "skill": "code_repl",
        }
    )

    assert set(normalized) == CODE_REPL_MODEL_RESULT_FIELDS
    assert normalized == {
        "schema": CODE_REPL_MODEL_RESULT_SCHEMA,
        "ok": True,
        "status": "ok",
        "stdout": "42\n",
        "stderr": "",
        "returncode": 0,
        "engine": "sandbox_runner",
        "summary": "Code execution completed successfully.",
        "error": "",
        "stdout_truncated": False,
        "stdout_original_chars": 3,
        "stdout_sha256": code_repl_mod.hashlib.sha256(b"42\n").hexdigest(),
        "stdout_preview_sha256": code_repl_mod.hashlib.sha256(b"42\n").hexdigest(),
        "stderr_truncated": False,
        "stderr_original_chars": 0,
        "stderr_sha256": code_repl_mod.hashlib.sha256(b"").hexdigest(),
        "stderr_preview_sha256": code_repl_mod.hashlib.sha256(b"").hexdigest(),
        "error_truncated": False,
        "error_original_chars": 0,
        "error_sha256": code_repl_mod.hashlib.sha256(b"").hexdigest(),
        "error_preview_sha256": code_repl_mod.hashlib.sha256(b"").hexdigest(),
    }


def test_code_repl_model_result_normalization_preserves_failure_evidence() -> None:
    normalized = normalize_code_repl_model_result(
        {
            "ok": False,
            "status": "error",
            "stdout": "",
            "stderr": "NameError: missing",
            "returncode": 0,
            "engine": "sandbox_runner",
        }
    )

    assert normalized["ok"] is False
    assert normalized["status"] == "error"
    assert normalized["error"] == "NameError: missing"


@pytest.mark.parametrize(
    "raw",
    (
        {"ok": True, "status": "ok", "returncode": 1},
        {"ok": True, "status": "error", "returncode": 0},
        {"ok": True, "status": "ok", "returncode": 0, "error": "boom"},
        {"ok": False, "status": "ok", "returncode": 0},
    ),
)
def test_code_repl_model_result_reconciles_contradictory_failure_evidence(
    raw,
) -> None:
    normalized = normalize_code_repl_model_result(raw)

    assert normalized["ok"] is False
    assert normalized["status"] == "error"
    assert normalized["summary"] == "Code execution failed."
    assert normalized["error"]


def test_code_repl_model_result_preserves_runner_exception_evidence() -> None:
    normalized = normalize_code_repl_model_result(
        {
            "ok": False,
            "status": "error",
            "returncode": 1,
            "repr": "NameError('missing')",
            "traceback": "Traceback...\nNameError: missing",
            "engine": "sandbox_runner",
        }
    )

    assert "NameError('missing')" in normalized["error"]
    assert "Traceback" in normalized["error"]


@pytest.mark.parametrize(
    "status",
    ("blocked", "deferred", "cancelled", "aborted", "partial", "unknown"),
)
def test_code_repl_model_result_fails_closed_on_non_success_status(status) -> None:
    normalized = normalize_code_repl_model_result(
        {"ok": True, "status": status, "returncode": 0}
    )

    assert normalized["ok"] is False
    assert normalized["status"] == "error"
    assert normalized["error"]


def test_code_repl_model_result_hashes_malformed_unicode_losslessly() -> None:
    first = normalize_code_repl_model_result(
        {"ok": True, "status": "ok", "returncode": 0, "stdout": "\ud800"}
    )
    second = normalize_code_repl_model_result(
        {"ok": True, "status": "ok", "returncode": 0, "stdout": "\udfff"}
    )

    assert first["stdout_sha256"] != second["stdout_sha256"]


def test_code_repl_model_result_bounds_output_inside_canonical_schema() -> None:
    raw = {
        "ok": False,
        "status": "error",
        "stdout": "o" * 10_000,
        "stderr": "e" * 10_000,
        "error": "x" * 10_000,
        "returncode": 1,
        "engine": "sandbox_runner",
    }
    normalized = normalize_code_repl_model_result(raw)
    serialized = mlx_client._serialize_tool_result_for_model("code_repl", raw)

    assert set(normalized) == CODE_REPL_MODEL_RESULT_FIELDS
    assert normalized["stdout_truncated"] is True
    assert normalized["stderr_truncated"] is True
    assert normalized["error_truncated"] is True
    assert normalized["stdout_original_chars"] == 10_000
    assert normalized["stdout_sha256"] == code_repl_mod.hashlib.sha256(
        b"o" * 10_000
    ).hexdigest()
    assert len(serialized) <= 4000
    assert set(json.loads(serialized)) == CODE_REPL_MODEL_RESULT_FIELDS


def test_code_repl_model_result_bounds_after_json_escaping() -> None:
    raw = {
        "ok": False,
        "status": "error",
        "stdout": "\0" * 10_000,
        "stderr": "\ud800" * 10_000,
        "error": "\0" * 10_000,
        "returncode": 1,
        "engine": "sandbox_runner",
    }

    serialized = mlx_client._serialize_tool_result_for_model("code_repl", raw)
    parsed = json.loads(serialized)

    assert len(serialized) <= 4000
    assert set(parsed) == CODE_REPL_MODEL_RESULT_FIELDS
    assert parsed["stdout_truncated"] is True
    assert parsed["stderr_truncated"] is True
    assert parsed["error_truncated"] is True
    assert parsed["error"]
    assert normalize_code_repl_model_result(parsed) == parsed


@pytest.mark.parametrize(
    ("field", "value"),
    (("returncode", 1), ("summary", "Code execution failed.")),
)
def test_code_repl_canonical_result_rejects_contradictions(field, value) -> None:
    canonical = normalize_code_repl_model_result(
        {"ok": True, "status": "ok", "returncode": 0, "stdout": "42"}
    )
    canonical[field] = value

    with pytest.raises(ValueError, match="canonical result"):
        normalize_code_repl_model_result(canonical)


def test_code_repl_canonical_result_binds_truncated_preview() -> None:
    canonical = normalize_code_repl_model_result(
        {
            "ok": True,
            "status": "ok",
            "returncode": 0,
            "stdout": "x" * 10_000,
        }
    )
    canonical["stdout"] = "forged preview"

    with pytest.raises(ValueError, match="stdout evidence"):
        normalize_code_repl_model_result(canonical)


def test_code_repl_serializer_rejects_valid_irreducible_oversized_schema() -> None:
    canonical = normalize_code_repl_model_result(
        {"ok": False, "status": "error", "error": "x", "returncode": 1}
    )

    with pytest.raises(ValueError, match="cannot contain canonical schema"):
        mlx_client._serialize_tool_result_for_model(
            "code_repl",
            canonical,
            limit=1,
        )


def test_code_repl_serializer_makes_strict_progress_to_exact_budget() -> None:
    canonical = normalize_code_repl_model_result(
        {
            "ok": True,
            "status": "ok",
            "returncode": 0,
            "stdout": "x" * 1000,
        }
    )
    full = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    limit = len(full) - 1

    serialized = mlx_client._serialize_tool_result_for_model(
        "code_repl",
        canonical,
        limit=limit,
    )

    assert len(serialized) <= limit
    assert json.loads(serialized)["stdout_truncated"] is True


@pytest.mark.asyncio
async def test_think_and_act_missing_engine_uses_canonical_code_repl_result(
    monkeypatch,
) -> None:
    from core.container import ServiceContainer

    client = mlx_client.MLXLocalClient.__new__(mlx_client.MLXLocalClient)
    client.max_tokens = 64

    async def generate_text_async(*_args, **_kwargs):
        return '{"tool":"code_repl","args":{"code":"print(1)"}}'

    client.generate_text_async = generate_text_async
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    result = await client.think_and_act(
        "calculate",
        "system",
        tools={
            "code_repl": {
                "description": "execute code",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            }
        },
        max_turns=1,
    )

    model_result = json.loads(result["tool_calls"][0]["result"])
    assert set(model_result) == CODE_REPL_MODEL_RESULT_FIELDS
    assert model_result["ok"] is False
    assert model_result["engine"] == "capability_engine"


@pytest.mark.asyncio
async def test_think_and_act_keeps_executed_arguments_typed_in_its_transcript(
    monkeypatch,
) -> None:
    from core.container import ServiceContainer

    client = mlx_client.MLXLocalClient.__new__(mlx_client.MLXLocalClient)
    client.max_tokens = 64
    responses = [
        '{"tool":"code_repl","args":{"code":"print(1)"}}',
        "The result was 1.",
    ]
    transcripts = []

    async def generate_text_async(*_args, **kwargs):
        transcripts.append(kwargs["messages"])
        return responses.pop(0)

    class _CapabilityEngine:
        async def execute(self, *_args, **_kwargs):
            return {"ok": True, "status": "ok", "returncode": 0, "stdout": "1\n"}

    client.generate_text_async = generate_text_async
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _CapabilityEngine()
            if name == "capability_engine"
            else default
        ),
    )

    result = await client.think_and_act(
        "calculate",
        "system",
        tools={
            "code_repl": {
                "description": "execute code",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            }
        },
        max_turns=2,
        context={"authorised_effect_scope": "read_write_artifacts"},
    )

    function = transcripts[1][-2]["tool_calls"][0]["function"]
    assert function["arguments"] == {"code": "print(1)"}
    assert result["content"] == "The result was 1."


@pytest.mark.asyncio
async def test_live_safe_execute_result_matches_model_visible_contract(
    monkeypatch,
    tmp_path,
) -> None:
    class Breaker:
        @staticmethod
        def allow_request():
            return True

        @staticmethod
        def record_success():
            return None

    monkeypatch.setattr(
        "infrastructure.resilience._get_or_create_breaker",
        lambda _name: Breaker(),
    )
    skill = CodeREPLSkill()

    async def session_dir(_session_id):
        return tmp_path

    async def sandbox_result(_code, _timeout, _cwd, library_root=""):
        return {
            "ok": True,
            "status": "ok",
            "stdout": "z" * 10_000,
            "stderr": "",
            "returncode": 0,
            "engine": "sandbox_runner",
        }

    monkeypatch.setattr(skill, "_get_session_dir", session_dir)
    monkeypatch.setattr(
        skill,
        "_execute_via_sandbox_runner",
        sandbox_result,
    )
    live_result = await skill.safe_execute(
        {
            "code": "print(1)",
            "capture_files": False,
            "timeout": 2,
        },
        {"source": "test"},
    )
    model_result = json.loads(
        mlx_client._serialize_tool_result_for_model(
            "code_repl",
            live_result,
        )
    )

    assert live_result["skill"] == "code_repl"
    assert set(model_result) == CODE_REPL_MODEL_RESULT_FIELDS
    assert model_result == normalize_code_repl_model_result(live_result)
    assert model_result["stdout_truncated"] is True
    assert len(json.dumps(model_result, separators=(",", ":"))) <= 4000


@pytest.mark.asyncio
async def test_code_repl_subprocess_fallback_uses_file_gateway_and_action_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_calls: list[tuple[str, str]] = []
    action_calls: list[dict[str, object]] = []

    class FakeFileWriteGateway:
        def write_text(self, path, text, *, encoding="utf-8", source="unknown") -> None:
            target = Path(path)
            file_calls.append((target.name, source))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding=encoding)

        # Async lane delegators: production code now calls *_async; fakes
        # must mirror the gateway surface or every governed write breaks.
        async def write_text_async(self, *args, **kwargs):
            return self.write_text(*args, **kwargs)

    class FakeActionExecutor:
        @classmethod
        async def execute(cls, **kwargs):
            action_calls.append(kwargs)
            return {"ok": True, "stdout": "ok\n", "stderr": "", "exit_code": 0}

    monkeypatch.setattr(
        code_repl_mod,
        "get_file_write_gateway",
        lambda: FakeFileWriteGateway(),
    )
    monkeypatch.setattr(code_repl_mod, "ActionExecutor", FakeActionExecutor)

    result = await CodeREPLSkill()._execute_via_subprocess(
        "print('ok')",
        timeout_s=2,
        cwd=tmp_path,
    )

    assert result == {
        "ok": True,
        "stdout": "ok\n",
        "stderr": "",
        "returncode": 0,
        "engine": "subprocess",
        "summary": "Code executed via ActionExecutor.",
    }
    assert file_calls
    temp_name, source = file_calls[0]
    assert source == "core.skills.code_repl.temp_script"
    assert not (tmp_path / temp_name).exists()
    assert action_calls
    params = action_calls[0]["params"]
    assert isinstance(params, dict)
    assert params["cwd"] == str(tmp_path)
    assert params["timeout"] == 2.0
