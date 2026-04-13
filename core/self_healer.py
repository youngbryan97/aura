"""core/self_healer.py
Pattern-based diagnostic and auto-fix system.
"""
import logging
import re
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("Core.SelfHealer")

IssueFixer = Callable[[re.Match[str], Exception], bool]


class SelfHealer:
    def __init__(self, container: Any | None = None) -> None:
        self.container = container
        self.issue_patterns: dict[str, IssueFixer] = {
            r"No module named '(\w+)'": self._fix_missing_package,
            r"Connection refused": self._report_connection_issue,
            r"429.*quota": self._trigger_provider_switch,
            r"Permission denied": self._suggest_permissions_fix
        }

    def diagnose_and_fix(self, exception: Exception) -> bool:
        """Takes an exception, matches it against patterns, and attempts a fix.
        Returns True if a fix was attempted.
        """
        error_msg = str(exception)
        for pattern, fixer in self.issue_patterns.items():
            match = re.search(pattern, error_msg)
            if match:
                logger.info("SelfHealer: Detected pattern %s. Attempting fix...", pattern)
                try:
                    return fixer(match, exception)
                except Exception as e:
                    logger.error("SelfHealer: Fixer failed: %s", e)
                    return False
        return False

    def _fix_missing_package(self, match: re.Match[str], exc: Exception) -> bool:
        package = match.group(1)
        # SECURITY: Validate package name (alphanumeric + hyphens/underscores only)
        if not re.match(r'^[a-zA-Z0-9_-]+$', package):
            logger.error("SelfHealer: Package name '%s' looks suspicious. Refusing.", package)
            return False
        # Don't auto-install. Log the suggestion for operator review.
        logger.warning("SelfHealer: Missing package '%s'. NOT auto-installing (security policy). "
                       "Operator should run: pip install %s", package, package)
        return False

    def _report_connection_issue(self, match: re.Match[str], exc: Exception) -> bool:
        logger.error("SelfHealer: Connection refused. Subsystems may be offline (Check Ollama/Docker).")
        # Could attempt to restart a service here if paths are known
        return False

    def _trigger_provider_switch(self, match: re.Match[str], exc: Exception) -> bool:
        logger.warning("SelfHealer: Quota exceeded. ResilienceEngine should handle fallback.")
        # We notify the resilience engine to trip the breaker immediately
        if self.container:
            engine = self.container.get("resilience_engine")
            if engine:
                # We identify which service tripped (this is heuristic)
                if "ollama" in str(exc).lower() or "local" in str(exc).lower():
                    engine.get_breaker("local_llm").record_failure()
                    # Trip it immediately 
                    engine.get_breaker("local_llm").state = "OPEN"
                    engine.get_breaker("local_llm").last_failure_time = 0
        return True

    def _suggest_permissions_fix(self, match: re.Match[str], exc: Exception) -> bool:
        logger.error("SelfHealer: Permission denied. Manual intervention required (chmod/sudo).")
        return False
