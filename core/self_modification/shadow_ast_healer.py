"""
core/self_modification/shadow_ast_healer.py
============================================
SHADOW AST HEALER — Governed Self-Repair via AST Manipulation

Implements zero-token self-repair via AST manipulation. When the system
detects a subsystem failure due to code errors, the ShadowHealer analyzes
the AST, identifies common patterns (missing imports, type mismatches),
and applies patches — but ONLY after governance approval.

GOVERNANCE CONTRACT:
    - Every file write MUST pass through _check_governance()
    - Only files within the codebase root can be modified
    - Only imports from a known-safe allowlist can be injected
    - The safe list explicitly EXCLUDES dangerous modules (os, subprocess,
      shutil, socket, ctypes, etc.)
    - All repair attempts are logged with before/after content hashes

This module is named "shadow" because it operates in the background
during error recovery, not because it hides from governance.
"""

import ast
import asyncio
import hashlib
import inspect
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("Aura.ShadowHealer")


class ShadowASTHealer:
    """Autonomously repairs source code via AST manipulation.

    All file modifications are gated through governance — the healer
    cannot write to disk without approval from the ConstitutionalCore
    or UnifiedWill. If governance is unavailable, repairs are REFUSED
    (fail-closed, not fail-open).
    """

    # Known-safe imports that can be auto-injected. This list explicitly
    # EXCLUDES modules that provide filesystem, network, or process access.
    # Adding to this list requires a deliberate decision.
    _SAFE_IMPORTS = {
        "asyncio": "import asyncio",
        "json": "import json",
        "logging": "import logging",
        "time": "import time",
        "math": "import math",
        "re": "import re",
        "Path": "from pathlib import Path",
        "Any": "from typing import Any",
        "Dict": "from typing import Dict",
        "List": "from typing import List",
        "Optional": "from typing import Optional",
    }

    def __init__(self, codebase_root: Optional[Path] = None):
        self.root = (codebase_root or Path.cwd()).resolve()

    def _check_governance(self, file_path: Path, action: str) -> bool:
        """Check with governance before modifying a file.

        Returns True if the repair is authorized, False otherwise.
        If the governance system is unavailable, returns False (fail-closed).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._check_governance_async(file_path, action))
        return self._check_governance_async(file_path, action)

    async def _check_governance_async(self, file_path: Path, action: str) -> bool:
        """Async governance check used by the repair coroutine."""
        try:
            from core.will import get_will, ActionDomain
            will = get_will()
            if will is None:
                logger.warning("ShadowHealer: Will unavailable — refusing repair (fail-closed)")
                return False
            decision = await will.decide(
                action=action,
                domain=ActionDomain.STATE_MUTATION,
                context={"file": str(file_path), "source": "shadow_ast_healer"},
                priority=0.6,
            )
            approved = decision.is_approved() if hasattr(decision, "is_approved") else False
            if not approved:
                logger.warning(
                    "ShadowHealer: Governance DENIED repair of %s: %s",
                    file_path.name,
                    getattr(decision, "reason", "no reason"),
                )
            return approved
        except Exception as exc:
            # Governance check failed — fail closed
            logger.warning("ShadowHealer: Governance check failed (denying): %s", exc)
            return False

    def _is_within_root(self, file_path: Path) -> bool:
        """Verify the target file is within the codebase root."""
        try:
            resolved = file_path.resolve()
            return str(resolved).startswith(str(self.root))
        except Exception:
            return False

    async def attempt_repair(self, file_path: Path, error_msg: str) -> bool:
        """Attempts to repair a specific error in a target file.

        GOVERNANCE: Will not modify any file without explicit governance
        approval. Will not modify files outside the codebase root.
        """
        logger.info("ShadowHealer: Analyzing %s: %s", file_path.name, error_msg)

        try:
            # Ensure path is absolute and within root
            if not file_path.is_absolute():
                file_path = self.root / file_path

            # Path containment check — refuse files outside root
            if not self._is_within_root(file_path):
                logger.warning(
                    "ShadowHealer: REFUSED — %s is outside codebase root %s",
                    file_path, self.root,
                )
                return False

            content = await asyncio.to_thread(file_path.read_text)
            content_hash_before = hashlib.sha256(content.encode()).hexdigest()[:16]
            tree = ast.parse(content)

            repaired = False
            # Pattern 1: Missing Import
            if "name" in error_msg.lower() and "is not defined" in error_msg.lower():
                try:
                    missing_name = error_msg.split("'")[1]
                except IndexError:
                    missing_name = error_msg.split()[-1]
                repaired = self._inject_missing_import(tree, missing_name)

            if repaired:
                # GOVERNANCE GATE: check before writing
                governance_result = self._check_governance(
                    file_path,
                    f"ast_repair:{file_path.name}",
                )
                if inspect.isawaitable(governance_result):
                    governance_result = await governance_result
                if not governance_result:
                    logger.info("ShadowHealer: Repair blocked by governance for %s", file_path.name)
                    return False

                new_content = ast.unparse(tree)
                content_hash_after = hashlib.sha256(new_content.encode()).hexdigest()[:16]
                await asyncio.to_thread(file_path.write_text, new_content)
                logger.info(
                    "ShadowHealer: Repaired %s (before=%s after=%s)",
                    file_path.name, content_hash_before, content_hash_after,
                )
                return True

            return False
        except Exception as e:
            logger.error("ShadowHealer: AST repair failed: %s", e)
            return False

    def _inject_missing_import(self, tree: ast.AST, name: str) -> bool:
        """Injects a common missing import into the AST.

        Only imports from the known-safe allowlist are permitted. Dangerous
        modules (os, subprocess, shutil, socket, ctypes, etc.) are explicitly
        excluded and cannot be auto-injected.
        """
        if name in self._SAFE_IMPORTS:
            import_node = ast.parse(self._SAFE_IMPORTS[name]).body[0]
            tree.body.insert(0, import_node)
            return True
        logger.debug("ShadowHealer: '%s' not in safe import allowlist — skipping", name)
        return False

    def validate_syntax(self, file_path: Path) -> bool:
        """Validates that a file has valid Python syntax."""
        try:
            ast.parse(file_path.read_text())
            return True
        except SyntaxError:
            return False
