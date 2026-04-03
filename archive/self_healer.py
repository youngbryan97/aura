"""core/self_healer.py
Pattern-based diagnostic and auto-fix system.
"""
import logging
import os
import re
import subprocess
import sys
from typing import Dict, Optional

logger = logging.getLogger("Core.SelfHealer")

class SelfHealer:
    def __init__(self, container=None):
        self.container = container
        self.issue_patterns = {
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
            match = re.search(pattern, error_msg, re.IGNORECASE)
            if match:
                logger.info("SelfHealer: Detected pattern %s. Attempting fix...", pattern)
                try:
                    return fixer(match, exception)
                except Exception:
                    logger.error("SelfHealer: Fixer failed for pattern %s", pattern, exc_info=True)
                    return False
        return False

    def _fix_missing_package(self, match: re.Match, exc: Exception) -> bool:
        package = match.group(1)
        # SECURITY: Validate package name (alphanumeric + hyphens/underscores only)
        if not re.match(r'^[a-zA-Z0-9_-]+$', package):
            logger.error("SelfHealer: Package name '%s' looks suspicious. Refusing.", package)
            return False
        # Don't auto-install. Log the suggestion for operator review.
        logger.warning("SelfHealer: Missing package '%s'. NOT auto-installing (security policy). "
                       "Operator should run: pip install %s", package, package)
        return False

    def _report_connection_issue(self, match: re.Match, exc: Exception) -> bool:
        logger.error("SelfHealer: Connection refused. Subsystems may be offline (Check Local Models/Docker).")
        # Could attempt to restart a service here if paths are known
        return False

    def _trigger_provider_switch(self, match: re.Match, exc: Exception) -> bool:
        logger.warning("SelfHealer: Quota exceeded. Attempting provider switch via ResilienceEngine.")
        if self.container:
            engine = self.container.get("resilience_engine")
            if engine:
                # We identify which service tripped
                breaker_name = "local_llm" if any(kw in str(exc).lower() for kw in ["local", "mlx"]) else "remote_llm"
                breaker = engine.get_breaker(breaker_name)
                if breaker:
                    if breaker.is_open():
                        logger.warning("SelfHealer: Breaker '%s' already OPEN.", breaker_name)
                        return False
                    breaker.record_failure()
                    # Force trip if quota specifically
                    breaker.state = "OPEN" 
                    return True
        return False

    def _suggest_permissions_fix(self, match: re.Match, exc: Exception) -> bool:
        logger.error("SelfHealer: Permission denied. Manual intervention required (chmod/sudo).")
        return False