# core/app_context.py
from dataclasses import dataclass
from typing import Any

# Forward references to avoid circular imports during type checking at runtime
# In a real scenario, use actual types or Protocol definitions

@dataclass
class AppContext:
    """Singleton-like container for long-lived application services.
    Injected into components (Orchestrator, API) to provide access to shared infrastructure.
    """

    input_bus: Any # InputBus
    process_manager: Any # ProcessManager
    memory: Any # MemoryStoreV2
    
    # Optional shared resources
    config: dict[str, Any] | None = None
    logger: Any | None = None
