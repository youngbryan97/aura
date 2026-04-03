import logging
from typing import Any
from core.container import ServiceContainer

logger = logging.getLogger(__name__)

async def init_enterprise_layer(orchestrator: Any):
    """Initialize the enterprise layer subsystems."""
    # 1. Structured Logging & Metrics
    from core.logging_config import setup_logging
    from core.observability.metrics import get_metrics
    setup_logging()
    metrics = get_metrics()
    ServiceContainer.register_instance("metrics", metrics)
    orchestrator.metrics = metrics

    # 2. Secrets Management
    from core.secrets import get_secret
    orchestrator._gemini_key = get_secret("GEMINI_API_KEY")

    # 3. Database Migrations
    from core.db.migrations import get_migrator
    migrator = get_migrator()
    migrator.run_all()

    # 4. Conversation Persistence
    from core.conversation.persistence import get_persistence
    orchestrator.persistence = get_persistence()
    orchestrator.persistence.start_session()
    ServiceContainer.register_instance("persistence", orchestrator.persistence)
    if hasattr(orchestrator.persistence, "on_start_async"):
        await orchestrator.persistence.on_start_async()

    # 5. Automated Backups & Vacuum
    from core.backup import BackupManager
    orchestrator.backup_manager = BackupManager()
    ServiceContainer.register_instance("backup_manager", orchestrator.backup_manager)
    if hasattr(orchestrator.backup_manager, "on_start_async"):
        await orchestrator.backup_manager.on_start_async()

    # 6. Dead Letter Queue
    from core.dead_letter_queue import get_dlq
    orchestrator.dlq = get_dlq()
    ServiceContainer.register_instance("dlq", orchestrator.dlq)

    # 7. Immutable Audit Trail
    from core.audit import get_audit
    orchestrator.audit = get_audit()
    ServiceContainer.register_instance("audit", orchestrator.audit)
    orchestrator.audit.record("system_boot", "RobustOrchestrator Enterprise Layer initialized")

    # 8. LLM Guards & Context Manager
    from core.context_manager import ContextWindowManager
    from core.config import config
    orchestrator.context_manager = ContextWindowManager(model_name=config.llm.chat_model)
    ServiceContainer.register_instance("context_manager", orchestrator.context_manager)

    # 9. Core Messaging
    from core.event_bus import get_event_bus
    orchestrator.event_bus = get_event_bus()
    ServiceContainer.register_instance("event_bus", orchestrator.event_bus)
    
    logger.info("✓ [BOOT] Enterprise Layer Baseline initialized.")
