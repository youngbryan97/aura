"""core/factory/regression_guard.py — Regression Guard.

Compares changes against linting, safety, and security policies.
Ensures new edits don't introduce performance or security regressions.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("Aura.RegressionGuard")


class RegressionGuard:
    """Checks patches for potential quality, performance, or security regressions."""

    async def run_checks(
        self, repo_path: str, patches: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Scans patches for common regressions (e.g. infinite loops, unclosed file descriptors, print statements)."""
        logger.info("🛡️ RegressionGuard checking %d patches...", len(patches))

        regressions_found = 0
        issues: list[dict[str, Any]] = []

        # High-risk patterns in code changes
        unsafe_patterns = [
            (re.compile(r"exec\(|eval\("), "Use of eval/exec is extremely dangerous", "high"),
            (re.compile(r"subprocess\.Popen\(.*shell=True|subprocess\.run\(.*shell=True"), "Shell execution with shell=True bypasses safety", "critical"),
            (re.compile(r"while True:"), "Infinite while loop without explicit break/timeout check", "medium"),
            (re.compile(r"print\("), "Production code should use logging instead of print", "low"),
            (re.compile(r"open\(.*\)"), "Raw open calls bypass the Archive/File Write Gateways", "high"),
        ]

        for patch in patches:
            module = patch.get("module", "unknown")
            content = patch.get("patch", "")

            # Check for syntax errors first
            if not patch.get("syntax_valid", True):
                regressions_found += 1
                issues.append({
                    "module": module,
                    "type": "syntax_error",
                    "severity": "critical",
                    "details": "Patch content contains invalid Python syntax",
                })

            # Scan lines for patterns
            for line_no, line in enumerate(content.splitlines(), start=1):
                for regex, msg, severity in unsafe_patterns:
                    if regex.search(line):
                        regressions_found += 1
                        issues.append({
                            "module": module,
                            "type": "unsafe_pattern",
                            "severity": severity,
                            "line": line_no,
                            "snippet": line.strip(),
                            "details": msg,
                        })

        return {
            "checks_run": len(patches),
            "regressions_found": regressions_found,
            "issues": issues,
            "passed": regressions_found == 0
        }
