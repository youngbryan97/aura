"""core/providers/memory_provider.py — Memory & Storage Registration
"""

import logging
import os

from core.governance_context import local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.service_registry import SERVICE_LIFETIME_SINGLETON

logger = logging.getLogger("Aura.Providers.Memory")

def register_memory_services(container):
    # 7. Memory (Base Store)
    def create_memory():
        from core.config import config
        from core.memory.sqlite_storage import SQLiteMemory
        db_path = config.paths.data_dir / "memory" / "atomic_knowledge.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return SQLiteMemory(storage_file=str(db_path))
    container.register('memory', create_memory, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)

    def create_persistent_state():
        try:
            from core.db.orm import PersistentState

            return PersistentState()
        except ImportError as exc:
            if "sqlalchemy" in str(exc).lower():
                from core.db.sqlite_persistent_state import SQLitePersistentState

                logger.info("SQLAlchemy unavailable; using stdlib SQLitePersistentState audit log.")
                return SQLitePersistentState()
            record_degradation("memory_provider", exc)
            logger.warning("PersistentState audit log unavailable: %s", exc)
            return None
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("memory_provider", exc)
            logger.warning("PersistentState audit log unavailable: %s", exc)
            return None

    container.register(
        'persistent_state',
        create_persistent_state,
        lifetime=SERVICE_LIFETIME_SINGLETON,
        required=False,
    )

    # 8. Memory Manager
    def create_memory_manager():
        from core.managers.memory_manager import MemoryManager
        memory = container.get("memory")
        vector = container.get("memory_vector", None)
        return MemoryManager(sqlite_memory=memory, vector_memory=vector)
    container.register('memory_manager', create_memory_manager, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)

    # 24. Black Hole Vault (The Unified Semantic Memory)
    def create_vector_memory():
        try:
            from core.config import config
            from core.memory.black_hole_vault import BlackHoleVault
            # Store in ~/.aura/vault as specified in Phase 4 plan
            vault_path = config.paths.data_dir / "vault"
            vault_path.mkdir(parents=True, exist_ok=True)
            return BlackHoleVault(data_dir=str(vault_path))
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('memory_provider', e)
            logger.warning("BlackHoleVault registration failed: %s", e)
            return None
    container.register('memory_vector', create_vector_memory, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)
    container.register('vector_memory', lambda: container.get("memory_vector"), lifetime=SERVICE_LIFETIME_SINGLETON, required=False)
    container.register('semantic_memory', lambda: container.get("memory_vector"), lifetime=SERVICE_LIFETIME_SINGLETON, required=False)
    container.register('vector_memory_engine', lambda: container.get("memory_vector"), lifetime=SERVICE_LIFETIME_SINGLETON, required=False)
    # These are named interfaces over the same durable vault, not additional
    # databases. Registering the canonical name lets MemoryFacade and the
    # inventory describe the live object instead of reporting it absent.
    container.register(
        'blackhole_vault',
        lambda: container.get("memory_vector"),
        lifetime=SERVICE_LIFETIME_SINGLETON,
        required=False,
    )

    def create_knowledge_ledger():
        from core.memory.knowledge_ledger import get_knowledge_ledger

        return get_knowledge_ledger()

    container.register(
        'knowledge_ledger',
        create_knowledge_ledger,
        lifetime=SERVICE_LIFETIME_SINGLETON,
        required=False,
    )

    def create_cold_store():
        from core.config import config
        from core.memory.cold_store import ColdMemoryStore

        return ColdMemoryStore(config.paths.data_dir / "memory" / "cold_store.db")

    container.register(
        'cold_store',
        create_cold_store,
        lifetime=SERVICE_LIFETIME_SINGLETON,
        required=False,
    )

    # 23. Knowledge Graph
    def create_knowledge_graph():
        try:
            from core.config import config
            from core.memory.knowledge_graph import PersistentKnowledgeGraph
            kg_dir = config.paths.data_dir / "knowledge_graph"
            if kg_dir.exists() and not kg_dir.is_dir():
                staging_dir = kg_dir.with_name(f".{kg_dir.name}.migration-{os.getpid()}")
                staging_db = staging_dir / "knowledge.db"
                gateway = get_file_write_gateway()
                with local_internal_governed_scope(
                    "memory_provider.knowledge_graph_migration",
                    domain="file_write",
                    constraints={"artifact": "knowledge_graph"},
                ):
                    gateway.ensure_directory(
                        staging_dir,
                        source="memory_provider.knowledge_graph_migration.stage",
                    )
                    gateway.replace_file(
                        kg_dir,
                        staging_db,
                        source="memory_provider.knowledge_graph_migration.database",
                    )
                    try:
                        gateway.move_path(
                            staging_dir,
                            kg_dir,
                            source="memory_provider.knowledge_graph_migration.publish",
                        )
                    except (OSError, RuntimeError, TypeError, ValueError):
                        gateway.replace_file(
                            staging_db,
                            kg_dir,
                            source="memory_provider.knowledge_graph_migration.rollback",
                        )
                        raise
                logger.info(
                    "Migrated legacy knowledge graph database into canonical directory: %s",
                    kg_dir,
                )
            else:
                with local_internal_governed_scope(
                    "memory_provider.knowledge_graph_directory",
                    domain="file_write",
                    constraints={"artifact": "knowledge_graph"},
                ):
                    get_file_write_gateway().ensure_directory(
                        kg_dir,
                        source="memory_provider.knowledge_graph_directory",
                    )
            db_path = kg_dir / "knowledge.db"
            return PersistentKnowledgeGraph(str(db_path))
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("memory_provider", exc)
            logger.warning("Knowledge graph unavailable: %s", exc)
            return None
    container.register('knowledge_graph', create_knowledge_graph, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    # 23.5 Dreamer V2 (idle consolidation). Keep this registered so the
    # SleepTrigger does real consolidation instead of quietly skipping.
    def create_dreamer_v2():
        try:
            from core.sleep.dreamer_v2 import DreamerV2
            brain = container.get("cognitive_engine", default=None)
            kg = container.get("knowledge_graph", default=None)
            if brain is None or kg is None:
                return None
            return DreamerV2(
                brain=brain,
                knowledge_graph=kg,
                vector_memory=container.get("vector_memory", default=None),
                belief_graph=container.get("belief_graph", default=None),
            )
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("memory_provider", exc)
            logger.warning("DreamerV2 unavailable: %s", exc)
            return None
    container.register('dreamer_v2', create_dreamer_v2, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    # 25. Memory Subsystem (Lifecycle Manager)
    def create_memory_subsystem():
        from core.memory.memory_subsystem import MemorySubsystem
        # Note: orchestrator will be auto-wired or resolved later if available
        return MemorySubsystem()
    container.register('memory_subsystem', create_memory_subsystem, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    # 26. Episodic Memory
    def create_episodic_memory():
        from core.memory.episodic_memory import get_episodic_memory
        vector = container.get("memory_vector", None)
        return get_episodic_memory(vector_memory=vector)
    container.register('episodic_memory', create_episodic_memory, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)

    # 27. Memory Facade
    def create_memory_facade():
        from core.memory.memory_facade import MemoryFacade
        return MemoryFacade()
    container.register('memory_facade', create_memory_facade, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)

    # 28. Interpersonal Memory — her typed, evidence-bound notes on a person.
    # Registered rather than imported so context assembly, which lives in the
    # runtime foundation, can read it without the foundation depending on it.
    def create_interpersonal_memory():
        from core.memory.interpersonal_store import get_interpersonal_store
        return get_interpersonal_store()
    container.register(
        'interpersonal_memory',
        create_interpersonal_memory,
        lifetime=SERVICE_LIFETIME_SINGLETON,
        required=False,
    )
