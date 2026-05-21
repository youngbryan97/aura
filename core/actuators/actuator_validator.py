"""core/actuators/actuator_validator.py
======================================
Multi-stage validation pipeline for runtime-synthesized actuator code.
"""

from __future__ import annotations

import ast
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Actuators.Validator")

_VALIDATOR_RUNTIME_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
)


@dataclass
class ValidationResult:
    """The result of running a validation stage."""

    success: bool
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class ActuatorCodeValidator:
    """Multi-stage validation pipeline for synthesized actuator code."""

    FORBIDDEN_IMPORTS: set[str] = {
        "os",
        "sys",
        "subprocess",
        "socket",
        "urllib",
        "requests",
        "ctypes",
        "shutil",
        "pty",
        "platform",
        "builtins",
        "importlib",
        "gc",
        "multiprocessing",
        "threading",
        "asyncio",
        "http",
        "tempfile",
        "pdb",
    }

    FORBIDDEN_BUILTINS: set[str] = {"eval", "exec", "compile", "__import__", "open", "input"}

    REQUIRED_METHODS: set[str] = {"name", "description", "validate_params", "execute"}

    @staticmethod
    def _json_safe(value: Any) -> Any:
        try:
            return json.loads(json.dumps(value))
        except (TypeError, ValueError):
            return {}

    @staticmethod
    def _parse_sandbox_payload(stdout: str) -> dict[str, Any]:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Sandbox returned no structured payload")
        payload = lines[-1]
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = ast.literal_eval(payload)
        if not isinstance(data, dict):
            raise ValueError(f"Sandbox payload was not a dict: {type(data).__name__}")
        return data

    @classmethod
    def validate_ast(cls, source: str) -> ValidationResult:
        """Stage 1: Static AST validation to check for illegal imports, builtins, and structure."""
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return ValidationResult(False, f"Syntax error in source code: {exc}")

        # Check for forbidden imports and builtins
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    base_module = name.name.split(".")[0]
                    if base_module in cls.FORBIDDEN_IMPORTS:
                        return ValidationResult(False, f"Forbidden import: {name.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    base_module = node.module.split(".")[0]
                    if base_module in cls.FORBIDDEN_IMPORTS:
                        return ValidationResult(False, f"Forbidden import: {node.module}")

            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in cls.FORBIDDEN_BUILTINS:
                        return ValidationResult(False, f"Forbidden function call: {node.func.id}")

        # Verify class definition
        class_nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        if not class_nodes:
            return ValidationResult(False, "No class defined in the source code")
        if len(class_nodes) > 1:
            return ValidationResult(
                False,
                "Multiple classes defined; only one class inheriting from BaseActuator is allowed",
            )

        cls_node = class_nodes[0]
        # Check base classes (one of them should be BaseActuator)
        has_base_actuator = False
        for base in cls_node.bases:
            if isinstance(base, ast.Name) and base.id == "BaseActuator":
                has_base_actuator = True
                break
            elif isinstance(base, ast.Attribute) and base.attr == "BaseActuator":
                has_base_actuator = True
                break

        if not has_base_actuator:
            return ValidationResult(
                False, f"Class '{cls_node.name}' must inherit from BaseActuator"
            )

        # Check methods defined
        defined_methods = {node.name for node in cls_node.body if isinstance(node, ast.FunctionDef)}
        missing = cls.REQUIRED_METHODS - defined_methods
        if missing:
            return ValidationResult(
                False, f"Class '{cls_node.name}' is missing required methods/properties: {missing}"
            )

        return ValidationResult(True, details={"class_name": cls_node.name})

    @classmethod
    def validate_sandbox(cls, source: str) -> ValidationResult:
        """Stage 2: Runs the source in a secure isolated process via run_untrusted to test instantiation."""
        from core.sandbox.runner import run_untrusted

        ast_res = cls.validate_ast(source)
        if not ast_res.success:
            return ast_res
        class_name = ast_res.details["class_name"]

        # Build a script that wraps the untrusted code and verifies it instantiates and validates params
        test_script = f"""
class BaseActuator:
    pass

class ActuatorResult:
    def __init__(self, success, message, updates=None):
        self.success = success
        self.message = message
        self.updates = updates or {{}}

{source}

try:
    inst = {class_name}()
    # Call validate_params with empty dict (should not raise exception, should return a bool)
    val = inst.validate_params({{}})
    # Check test_params attribute
    test_params = getattr(inst, "test_params", {{}})
    print({{
        "success": True,
        "name": str(inst.name),
        "description": str(inst.description),
        "validate_empty": bool(val),
        "has_test_params": isinstance(test_params, dict),
        "test_params": test_params if isinstance(test_params, dict) else {{}}
    }})
except Exception as e:
    print({{"success": False, "error": "Instantiation error: " + str(e)}})
"""
        try:
            res = run_untrusted(test_script, timeout=5, mem_bytes=100 * 1024 * 1024)
            status = res.get("status")
            if status != "completed" and status != "ok":
                stderr = res.get("stderr", "")
                stdout = res.get("stdout", "")
                return ValidationResult(
                    False,
                    f"Sandbox failed with status: {status}. stderr: {stderr}. stdout: {stdout}",
                )

            stdout = res.get("stdout", "")
            try:
                payload = cls._parse_sandbox_payload(stdout)
            except (SyntaxError, ValueError, json.JSONDecodeError):
                return ValidationResult(False, f"Sandbox returned invalid JSON: {stdout}")

            if not payload.get("success"):
                return ValidationResult(False, f"Sandbox validation failed: {payload.get('error')}")

            return ValidationResult(True, details=payload)
        except _VALIDATOR_RUNTIME_ERRORS as exc:
            return ValidationResult(False, f"Sandbox execution failed with exception: {exc}")

    @classmethod
    def execute_sandboxed(cls, source: str, params: dict[str, Any]) -> ValidationResult:
        """Execute synthesized actuator code in the untrusted runner and return its result payload."""
        from core.sandbox.runner import run_untrusted

        ast_res = cls.validate_ast(source)
        if not ast_res.success:
            return ast_res
        class_name = ast_res.details["class_name"]
        params_literal = repr(cls._json_safe(params))
        test_script = f"""
class BaseActuator:
    pass

class ActuatorResult:
    def __init__(self, success, message, updates=None):
        self.success = success
        self.message = message
        self.updates = updates or {{}}

PARAMS = {params_literal}

{source}

try:
    inst = {class_name}()
    if not inst.validate_params(PARAMS):
        print({{"success": False, "error": "Parameter validation failed"}})
    else:
        result = inst.execute(PARAMS)
        updates = getattr(result, "updates", {{}})
        if not isinstance(updates, dict):
            updates = {{}}
        print({{
            "success": bool(getattr(result, "success", False)),
            "message": str(getattr(result, "message", "")),
            "updates": updates,
        }})
except Exception as e:
    print({{"success": False, "error": "Execution error: " + str(e)}})
"""
        try:
            res = run_untrusted(test_script, timeout=5, mem_bytes=100 * 1024 * 1024)
            status = res.get("status")
            if status != "completed" and status != "ok":
                stderr = res.get("stderr", "")
                stdout = res.get("stdout", "")
                return ValidationResult(
                    False,
                    f"Sandbox failed with status: {status}. stderr: {stderr}. stdout: {stdout}",
                )
            payload = cls._parse_sandbox_payload(res.get("stdout", ""))
            if not payload.get("success"):
                return ValidationResult(
                    False, str(payload.get("error") or "Sandbox execution failed")
                )
            return ValidationResult(True, details=payload)
        except (SyntaxError, ValueError, json.JSONDecodeError, *_VALIDATOR_RUNTIME_ERRORS) as exc:
            return ValidationResult(False, f"Sandboxed actuator execution failed: {exc}")

    @classmethod
    def validate_causal(cls, source: str, world_snapshot: dict | None = None) -> ValidationResult:
        """Stage 3: execute test params in the sandbox and verify returned updates are bounded."""
        try:
            sandbox_res = cls.validate_sandbox(source)
            if not sandbox_res.success:
                return sandbox_res

            test_params = sandbox_res.details.get("test_params", {})
            exec_res = cls.execute_sandboxed(source, test_params)
            if not exec_res.success:
                return exec_res

            updates = exec_res.details.get("updates", {})
            if not isinstance(updates, dict):
                return ValidationResult(False, "Actuator returned non-dict updates")
            for entity_id, fields in updates.items():
                if not isinstance(entity_id, str) or not isinstance(fields, dict):
                    return ValidationResult(False, "Actuator returned malformed update payload")
                for field, value in fields.items():
                    if field == "attributes":
                        if not isinstance(value, dict):
                            return ValidationResult(
                                False, "Actuator returned malformed attributes update"
                            )
                        continue
                    if field == "coordinates":
                        if not isinstance(value, (list, tuple)) or len(value) != 2:
                            return ValidationResult(
                                False, "Actuator returned malformed coordinates"
                            )
                        if not all(math.isfinite(float(coord)) for coord in value):
                            return ValidationResult(
                                False, "Actuator returned non-finite coordinates"
                            )
                        continue
                    if not math.isfinite(float(value)) or float(value) < 0.0:
                        return ValidationResult(
                            False, f"Actuator returned invalid update {field}={value!r}"
                        )

            return ValidationResult(True, details=exec_res.details)

        except _VALIDATOR_RUNTIME_ERRORS as exc:
            return ValidationResult(False, f"Causal simulation validation failed: {exc}")
