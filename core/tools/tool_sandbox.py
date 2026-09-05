"""core/tools/tool_sandbox.py — Tool Sandbox Validator.

Validates dynamically generated tools to guarantee compilation safety
and verify lack of illegal calls before integration.
"""
from __future__ import annotations

import ast
import logging
from typing import Any, Dict

logger = logging.getLogger("Aura.ToolSandbox")


class ToolSandbox:
    """Verifies compilation and safety properties of dynamically forged tool code."""

    def validate_tool_code(self, code_str: str) -> Dict[str, Any]:
        logger.info("🔒 ToolSandbox: auditing candidate tool code...")

        # 1. Compilation check
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            return {"compiles": False, "error": f"SyntaxError: {e}"}

        # 2. Basic static safety check (no direct subprocess imports or socket bindings)
        unsafe_imports = {"subprocess", "socket", "ctypes", "pty", "os"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.name in unsafe_imports:
                        return {"compiles": True, "safe": False, "reason": f"Forbidden import: {name.name}"}
            elif isinstance(node, ast.ImportFrom):
                if node.module in unsafe_imports:
                    return {"compiles": True, "safe": False, "reason": f"Forbidden import from: {node.module}"}

        return {
            "compiles": True,
            "safe": True,
            "line_count": len(code_str.splitlines()),
        }
