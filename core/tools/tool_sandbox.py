"""core/tools/tool_sandbox.py — the static screen for forged tool code.

What this can say is limited to what reading can decide: whether the code
parses, and whether it reaches for a module, call or attribute the shared
safety analyzer refuses. That is defense in depth, not containment. The
containment is where the code runs: ``ToolRegistry.execute_tool`` hands it
to ``run_untrusted``, which requires a kernel boundary and refuses to run
without one.

This used to report ``safe: True`` for anything that avoided five import
names, so ``import os.path``, ``__import__("os")``, ``open()`` and ``eval``
all passed — and its only caller ignored the verdict anyway. It reads the
same analyzer the symbolic sandbox uses now, and says only what it checked.
"""
from __future__ import annotations

import ast
import logging
from typing import Any, Dict

logger = logging.getLogger("Aura.ToolSandbox")


class ToolSandbox:
    """Screens forged tool code before it is offered to the registry."""

    def validate_tool_code(self, code_str: str) -> Dict[str, Any]:
        logger.info("ToolSandbox: screening candidate tool code...")

        try:
            ast.parse(code_str)
        except SyntaxError as e:
            return {"compiles": False, "safe": False, "error": f"SyntaxError: {e}"}

        try:
            from core.resilience.code_verifier import CodeVerifier

            safety = CodeVerifier.analyze_safety(code_str)
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            # No analyzer is no screen, and no screen is not a pass.
            logger.warning("ToolSandbox: the safety analyzer is unavailable (%s: %s)",
                           type(exc).__name__, exc)
            return {"compiles": True, "safe": False, "reason": "safety analyzer unavailable"}

        warnings = [str(w) for w in safety.get("warnings", [])]
        return {
            "compiles": True,
            "safe": bool(safety.get("safe")) and not warnings,
            "warnings": warnings,
            "reason": "; ".join(warnings[:3]),
            "line_count": len(code_str.splitlines()),
        }
