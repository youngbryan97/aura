"""core/actuators/code_execution_actuator.py
=========================================
Runs arbitrary Python code inside the sandbox.
Ensures safety by validating AST for banned imports/functions.
"""
import ast
import hashlib
from typing import Any

from core.actuators.actuator_registry import ActuatorResult, BaseActuator
from core.actuators.authority import verify_actuator_authority
from core.runtime.numeric_guards import positive_float

_BANNED_MODULES = {
    "ctypes", "importlib", "os", "pathlib", "pty", "shutil", "subprocess", "sys",
}
_BANNED_NETWORK_MODULES = {"socket", "urllib", "requests", "httpx", "http"}
_BANNED_CALLS = {
    "__import__", "compile", "eval", "exec", "globals", "input", "locals", "open", "vars",
}
_BANNED_ATTR_CALLS = {"system", "popen", "spawn", "remove", "unlink", "rmdir"}


def why_code_is_not_ast_safe(code: Any, *, network_access: bool = False) -> str:
    """What is unsafe about this code, or "" when nothing is.

    The reason, not just the verdict. A refusal reading "banned import or
    call" tells whoever wrote the code neither which import nor which call,
    and the caller here is usually a model that will be handed the error and
    asked to try again — so a refusal that does not say what it found costs a
    whole turn to learn one word.

    LIVE, 2026-08-28: asked to use a library the person had named, the model
    wrote the three lines anyone would write — import sys, put the directory on
    the path, import the library — and got back "banned import or call". The
    library was already importable inside the sandbox; only the first line was
    the problem, and nothing said so.
    """

    if not isinstance(code, str):
        return "code must be a string"
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError, MemoryError) as exc:
        return f"code does not parse: {exc}"
    banned = set(_BANNED_MODULES)
    if not network_access:
        banned |= _BANNED_NETWORK_MODULES
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                head = alias.name.split(".")[0]
                if head in banned:
                    return f"import of {head!r} is not allowed in the sandbox"
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in banned:
                return f"import from {node.module.split('.')[0]!r} is not allowed in the sandbox"
            for alias in node.names:
                head = alias.name.split(".")[0]
                if head in banned:
                    return f"import of {head!r} is not allowed in the sandbox"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BANNED_CALLS:
                return f"calling {node.func.id!r} is not allowed in the sandbox"
            if isinstance(node.func, ast.Attribute) and node.func.attr in _BANNED_ATTR_CALLS:
                return f"calling {node.func.attr!r} is not allowed in the sandbox"
    return ""


def code_is_ast_safe(code: Any, *, network_access: bool = False) -> bool:
    """Shared AST safety gate for synthesized code.

    Rejects banned imports (filesystem/process/interpreter, plus network unless
    explicitly allowed) and dangerous call names. Used by both the code-execution
    actuator and the sandbox operator so no execution path skips the check.
    """

    return not why_code_is_not_ast_safe(code, network_access=network_access)


class CodeExecutionActuator(BaseActuator):
    """Actuator that runs Python code in a sandbox with security controls."""

    requires_authority = True

    @property
    def name(self) -> str:
        return "code_execution"

    @property
    def description(self) -> str:
        return "Executes arbitrary Python code in a sandboxed environment with import and parameter validation."

    def validate_params(self, params: dict[str, Any]) -> bool:
        if not isinstance(params, dict) or "code" not in params:
            return False
        return code_is_ast_safe(
            params["code"], network_access=bool(params.get("network_access", False))
        )

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        _authorized, _auth_reason = verify_actuator_authority(params, actuator=self.name)
        if not _authorized:
            return ActuatorResult(False, _auth_reason, {})
        if not self.validate_params(params):
            return ActuatorResult(False, "Safety validation failed: code contains banned imports or functions.", {})

        from core.actuators.sandbox_operator import SandboxOperator
        operator = SandboxOperator()
        
        code = params["code"]
        # CP126 (high): "Execution timeout accepts arbitrary floats."
        # float() alone admits NaN, inf, zero and negatives. A NaN timeout
        # compares False against every deadline check, so a synthesized tool
        # ran unbounded; a zero or negative one means "already expired" and
        # kills it before it starts. This is caller data reaching a
        # subprocess boundary.
        timeout_s = positive_float(
            params.get("timeout_s", 15.0), default=15.0, maximum=600.0,
        )
        
        res = operator.execute_synthesized_tool(code, timeout_s=timeout_s)
        
        # Calculate digest for receipts
        output_combined = f"{res.get('stdout', '')}\n{res.get('stderr', '')}"
        output_hash = hashlib.sha256(output_combined.encode("utf-8")).hexdigest()
        
        updates = {
            "exit_code": res.get("exit_code"),
            "stdout": res.get("stdout"),
            "stderr": res.get("stderr"),
            "output_hash": output_hash,
            "success": res.get("success")
        }
        
        msg = f"Code executed successfully (exit code {res.get('exit_code')})." if res.get("success") else f"Code execution failed (exit code {res.get('exit_code')}): {res.get('stderr')}"
        
        return ActuatorResult(
            success=res.get("success", False),
            message=msg,
            updates=updates
        )
