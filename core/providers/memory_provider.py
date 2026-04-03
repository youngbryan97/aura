"""core/providers/memory_provider.py — Memory & Storage Registration
"""

import logging
from core.container import ServiceLifetime

logger = logging.getLogger("Aura.Providers.Memory")

def register_memory_services(container):
    # 7. Memory (Base Store)
    def create_memory():
        from core.config import config
        from core.memory.sqlite_storage import SQLiteMemory
        db_path = config.paths.data_dir / "memory" / "atomic_knowledge.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return SQLiteMemory(storage_file=str(db_path))
    container.register('memory', create_memory, lifetime=ServiceLifetime.SINGLETON, required=True)

    # 8. Memory Manager
    def create_memory_manager():
        from core.managers.memory_manager import MemoryManager
        memory = container.get("memory")
        vector = container.get("memory_vector", None)
        return MemoryManager(sqlite_memory=memory, vector_memory=vector)
    container.register('memory_manager', create_memory_manager, lifetime=ServiceLifetime.SINGLETON, required=True)

    # 24. Black Hole Vault (The Unified Semantic Memory)
    def create_vector_memory():
        try:
            from core.memory.black_hole_vault import BlackHoleVault
            from core.config import config
            # Store in ~/.aura/vault as specified in Phase 4 plan
            vault_path = config.paths.data_dir / "vault"
            vault_path.mkdir(parents=True, exist_ok=True)
            return BlackHoleVault(data_dir=str(vault_path))
        except Exception as e:
            logger.warning("BlackHoleVault registration failed: %s", e)
            return None
    container.register('memory_vector', create_vector_memory, lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('vector_memory', lambda: container.get("memory_vector"), lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('semantic_memory', lambda: container.get("memory_vector"), lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('vector_memory_engine', lambda: container.get("memory_vector"), lifetime=ServiceLifetime.SINGLETON, required=False)

    # 23. Knowledge Graph
    def create_knowledge_graph():
        try:
            from core.config import config
            from core.memory.knowledge_graph import PersistentKnowledgeGraph
            kg_path = config.paths.data_dir / "knowledge_graph"
            kg_path.mkdir(parents=True, exist_ok=True)
            return PersistentKnowledgeGraph(str(kg_path))
        except Exception:
            logger.debug("Knowledge graph unavailable")
            return None
    container.register('knowledge_graph', create_knowledge_graph, lifetime=ServiceLifetime.SINGLETON, required=False)

    # 25. Memory Subsystem (Lifecycle Manager)
    def create_memory_subsystem():
        from core.memory.memory_subsystem import MemorySubsystem
        # Note: orchestrator will be auto-wired or resolved later if available
        return MemorySubsystem()
    container.register('memory_subsystem', create_memory_subsystem, lifetime=ServiceLifetime.SINGLETON, required=False)

    # 26. Episodic Memory
    def create_episodic_memory():
        from core.memory.episodic_memory import get_episodic_memory
        vector = container.get("memory_vector", None)
        return get_episodic_memory(vector_memory=vector)
    container.register('episodic_memory', create_episodic_memory, lifetime=ServiceLifetime.SINGLETON, required=True)

    # 27. Memory Facade
    def create_memory_facade():
        from core.memory.memory_facade import MemoryFacade
        return MemoryFacade()
    container.register('memory_facade', create_memory_facade, lifetime=ServiceLifetime.SINGLETON, required=True)
