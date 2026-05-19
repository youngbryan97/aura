from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Any
import logging

logger = logging.getLogger("Aura.BootValidator")

@dataclass
class ValidationResult:
    passed: bool
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class BootValidator:
    @staticmethod
    def validate_boot(container: Any = None) -> ValidationResult:
        if container is None:
            from core.container import get_container
            container = get_container()
            
        failures: List[str] = []

        # 1. Infrastructure Ready
        if not container.has("event_bus"):
            failures.append("Event Bus Ready")
            
        if not container.has("orchestrator"):
            failures.append("Core Orchestrator Ready")

        # 2. Functional Protocols
        if not container.has("llm_interface"):
            failures.append("LLM Interface Bound")

        # 3. Persistence & Memory
        if not container.has("state_repo"):
            failures.append("State Repository (Persistence) Ready")

        # NOTE: memory_facade and voice_engine are lazily initialized during
        # _async_init_subsystems() and are NOT available at this pre-boot stage.
        # Checking them here caused a silent boot abort. They are now verified
        # in the post-boot StartupValidator.validate_all() instead.

        return ValidationResult(passed=not failures, failures=failures)
