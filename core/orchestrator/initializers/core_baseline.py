import logging
from typing import Any

from core.runtime.service_registry import register_runtime_service

logger = logging.getLogger(__name__)

async def init_enterprise_layer(orchestrator: Any) -> None:
    """Initialize the enterprise layer subsystems."""
    # 1. Structured Logging & Metrics
    from core.observability.logging_config import setup_logging
    from core.observability.metrics import get_metrics
    setup_logging()
    metrics = get_metrics()
    register_runtime_service("metrics", metrics)
    orchestrator.metrics = metrics

    # 2. Database Migrations
    #
    # A drift stops the boot on purpose. Everything below this line queries a
    # schema it believes the code wrote; when the database says otherwise, the
    # useful outcome is a named failure here rather than wrong rows later.
    from core.db.migrations import MigrationDriftError, get_migrator
    from core.runtime.errors import record_degradation

    migrator = get_migrator()
    try:
        migrator.run_all()
    except MigrationDriftError as exc:
        record_degradation(
            "migrations",
            exc,
            severity="critical",
            action="stopped the boot rather than query a schema the code does not match",
            extra={"database": migrator.db_path, "problems": migrator.verify()},
        )
        raise

    # 3. Conversation Persistence
    from core.conversation.persistence import get_persistence
    orchestrator.persistence = get_persistence()
    orchestrator.persistence.start_session()
    register_runtime_service("persistence", orchestrator.persistence)
    if hasattr(orchestrator.persistence, "on_start_async"):
        await orchestrator.persistence.on_start_async()

    # 4. Automated Backups & Vacuum
    from core.ops.backup import BackupManager
    orchestrator.backup_manager = BackupManager()
    register_runtime_service("backup_manager", orchestrator.backup_manager)
    if hasattr(orchestrator.backup_manager, "on_start_async"):
        await orchestrator.backup_manager.on_start_async()

    # 5. Dead Letter Queue
    from core.tasks.dead_letter_queue import get_dlq
    orchestrator.dlq = get_dlq()
    register_runtime_service("dlq", orchestrator.dlq)

    # 6. Immutable Audit Trail
    from core.audit import get_audit
    orchestrator.audit = get_audit()
    register_runtime_service("audit", orchestrator.audit)
    orchestrator.audit.record("system_boot", "RobustOrchestrator Enterprise Layer initialized")

    # 7. LLM Guards & Context Window Manager
    #
    # Registered as "context_window_manager", not "context_manager". Two
    # different objects were sharing the latter name: this one, and the
    # CognitiveContextManager that boot_cognitive publishes later under the same
    # key. Which one a caller got depended on how it asked — the attribute set
    # here resolved to the window manager, while `ServiceContainer.get(
    # "context_manager")` returned the cognitive one. They share no methods, so
    # hasattr checks downstream decided behaviour by accident.
    from core.config import config
    from core.context.context_manager import ContextWindowManager
    orchestrator.context_window_manager = ContextWindowManager(model_name=config.llm.chat_model)
    register_runtime_service("context_window_manager", orchestrator.context_window_manager)

    # 8. Core Messaging
    from core.event_bus import get_event_bus
    orchestrator.event_bus = get_event_bus()
    register_runtime_service("event_bus", orchestrator.event_bus)
    
    logger.info("✓ [BOOT] Enterprise Layer Baseline initialized.")
