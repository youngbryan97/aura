import ast
import json
import pytest
from unittest.mock import MagicMock, patch

from core.actuators.actuator_validator import ActuatorCodeValidator, ValidationResult
from core.actuators.actuator_synthesis import ActuatorSynthesizer
from core.actuators.actuator_registry import BaseActuator, ActuatorResult


# --- Mock classes for AST base check ---
class MockBaseActuator:
    pass


# Clean compliant code string for tests
SAFE_CODE = """
class SafeActuator(BaseActuator):
    @property
    def name(self) -> str:
        return "safe_actuator"

    @property
    def description(self) -> str:
        return "A safe test actuator"

    def validate_params(self, params: dict) -> bool:
        return True

    def execute(self, params: dict) -> ActuatorResult:
        return ActuatorResult(True, "Success", {})
"""

# Malicious code strings
BAD_IMPORT_CODE = """
import os
class MaliciousActuator(BaseActuator):
    @property
    def name(self) -> str:
        return "bad"
    @property
    def description(self) -> str:
        return "bad"
    def validate_params(self, params: dict) -> bool:
        return True
    def execute(self, params: dict) -> ActuatorResult:
        return ActuatorResult(True, "Success", {})
"""

BAD_BUILTIN_CODE = """
class MaliciousActuator(BaseActuator):
    @property
    def name(self) -> str:
        eval("print('evil')")
        return "bad"
    @property
    def description(self) -> str:
        return "bad"
    def validate_params(self, params: dict) -> bool:
        return True
    def execute(self, params: dict) -> ActuatorResult:
        return ActuatorResult(True, "Success", {})
"""

MISSING_METHOD_CODE = """
class BadActuator(BaseActuator):
    @property
    def name(self) -> str:
        return "bad"
    # Missing description, validate_params, execute
"""


def test_validate_ast_safe():
    res = ActuatorCodeValidator.validate_ast(SAFE_CODE)
    assert res.success
    assert res.details["class_name"] == "SafeActuator"


def test_validate_ast_forbidden_import():
    res = ActuatorCodeValidator.validate_ast(BAD_IMPORT_CODE)
    assert not res.success
    assert "Forbidden import" in res.error


def test_validate_ast_forbidden_builtin():
    res = ActuatorCodeValidator.validate_ast(BAD_BUILTIN_CODE)
    assert not res.success
    assert "Forbidden function call" in res.error


def test_validate_ast_missing_methods():
    res = ActuatorCodeValidator.validate_ast(MISSING_METHOD_CODE)
    assert not res.success
    assert "missing required methods" in res.error


def test_validate_ast_syntax_error():
    res = ActuatorCodeValidator.validate_ast("class SyntaxErrorActuator: def syntax_err:")
    assert not res.success
    assert "Syntax error" in res.error


def test_validate_sandbox_success():
    mock_run_untrusted = MagicMock(
        return_value={
            "status": "completed",
            "stdout": json.dumps(
                {
                    "success": True,
                    "name": "safe_actuator",
                    "description": "desc",
                    "validate_empty": True,
                    "has_test_params": True,
                }
            ),
        }
    )

    with patch("core.sandbox.runner.run_untrusted", mock_run_untrusted):
        res = ActuatorCodeValidator.validate_sandbox(SAFE_CODE)
        assert res.success
        assert res.details["name"] == "safe_actuator"
        mock_run_untrusted.assert_called_once()


def test_validate_sandbox_timeout_or_error():
    mock_run_untrusted = MagicMock(
        return_value={
            "status": "timeout",
            "stderr": "Execution timed out",
            "stdout": "",
        }
    )

    with patch("core.sandbox.runner.run_untrusted", mock_run_untrusted):
        res = ActuatorCodeValidator.validate_sandbox(SAFE_CODE)
        assert not res.success
        assert "Sandbox failed with status: timeout" in res.error


def test_validate_sandbox_json_decode_error():
    mock_run_untrusted = MagicMock(
        return_value={
            "status": "completed",
            "stdout": "Not JSON Output",
        }
    )

    with patch("core.sandbox.runner.run_untrusted", mock_run_untrusted):
        res = ActuatorCodeValidator.validate_sandbox(SAFE_CODE)
        assert not res.success
        assert "invalid JSON" in res.error


def test_validate_causal_successful_compilation():
    mock_entity = MagicMock()
    mock_entity.entity_id = "test_ent"
    mock_entity.kind = "sensor"
    mock_entity.capacity = 100
    mock_entity.load = 10
    mock_entity.flow_rate = 1.0
    mock_entity.max_flow_rate = 5.0
    mock_entity.latency = 0.1
    mock_entity.coordinates = (0, 0)
    mock_entity.attributes = {"temp": 20.0}

    mock_world = MagicMock()
    mock_world.seed = 42
    mock_world.sim_time = 1.0
    mock_world.entities = {"test_ent": mock_entity}

    with patch(
        "core.world.world_model.get_physics_world_model", return_value=mock_world
    ):
        res = ActuatorCodeValidator.validate_causal(SAFE_CODE)
        assert res.success


@pytest.mark.asyncio
async def test_actuator_synthesizer_synthesis():
    from unittest.mock import AsyncMock
    mock_client = MagicMock()
    mock_client.generate = AsyncMock(
        return_value={"response": "```python\n" + SAFE_CODE + "\n```"}
    )
    mock_client.close = AsyncMock(return_value=None)

    synthesizer = ActuatorSynthesizer()

    from core.actuators.actuator_synthesis import SynthesisRequest

    with patch("core.actuators.actuator_synthesis.LocalBrain", return_value=mock_client):
        # Patch all validators to pass
        with patch.object(
            ActuatorCodeValidator, "validate_ast", return_value=ValidationResult(True)
        ), patch.object(
            ActuatorCodeValidator,
            "validate_sandbox",
            return_value=ValidationResult(True, details={"name": "safe_actuator"}),
        ), patch.object(
            ActuatorCodeValidator,
            "validate_causal",
            return_value=ValidationResult(True),
        ), patch.object(
            synthesizer,
            "_governance_approve",
            return_value=True,
        ):

            req = SynthesisRequest(problem_description="Read temp from sensor")
            result = await synthesizer.request_synthesis(req)

            assert result is not None
            assert result.name == "safe_actuator"
