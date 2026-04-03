"""Shared pytest fixtures for Aura smoke tests."""
import asyncio
import inspect
import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so `core.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def service_container():
    """Provide a fresh ServiceContainer with cleared registry."""
    from core.container import ServiceContainer

    def _resolve_hook(instance, hook_name):
        try:
            inspect.getattr_static(instance, hook_name)
        except (NameError, AttributeError):
            return None
        try:
            hook = getattr(instance, hook_name)
        except Exception:
            return None
        return hook if callable(hook) else None

    def _finish_cleanup(result):
        if inspect.isawaitable(result):
            asyncio.run(result)

    def _close_service_instances():
        seen = set()
        for desc in list(getattr(ServiceContainer, "_services", {}).values()):
            instance = getattr(desc, "instance", None)
            if instance is None or id(instance) in seen:
                continue
            seen.add(id(instance))

            for method_name in ("shutdown", "stop", "close"):
                method = _resolve_hook(instance, method_name)
                if method is None:
                    continue
                try:
                    _finish_cleanup(method())
                except Exception:
                    pass

            db = getattr(instance, "_db", None)
            db_close = _resolve_hook(db, "close") if db is not None else None
            if db_close is not None:
                try:
                    _finish_cleanup(db_close())
                except Exception:
                    pass
    
    ServiceContainer.clear()
    
    # Snapshot existing registry to restore after test
    original = dict(ServiceContainer._registry) if hasattr(ServiceContainer, "_registry") else {}
    
    yield ServiceContainer

    try:
        from core.utils.task_tracker import get_task_tracker, task_tracker
        asyncio.run(get_task_tracker().shutdown(timeout=1.0))
        asyncio.run(task_tracker.shutdown(timeout=1.0))
    except Exception:
        pass

    _close_service_instances()
    ServiceContainer.clear()

    # Restore original registry
    if hasattr(ServiceContainer, "_registry"):
        ServiceContainer._registry.clear()
        ServiceContainer._registry.update(original)


@pytest.fixture(autouse=True)
def _disable_redis_event_bus_for_tests():
    """Keep the test suite local-only so Redis client coroutines don't leak warnings."""
    from core.config import config
    from core import event_bus as event_bus_module

    prev_use_for_events = bool(getattr(config.redis, "use_for_events", False))
    prev_bus_use_redis = bool(getattr(event_bus_module.get_event_bus(), "_use_redis", False))
    prev_bus_redis = getattr(event_bus_module.get_event_bus(), "_redis", None)

    config.redis.use_for_events = False
    event_bus_module.get_event_bus()._use_redis = False
    event_bus_module.get_event_bus()._redis = None

    yield

    config.redis.use_for_events = prev_use_for_events
    event_bus_module.get_event_bus()._use_redis = prev_bus_use_redis
    event_bus_module.get_event_bus()._redis = prev_bus_redis

@pytest.fixture
def mock_container(service_container):
    """Full architectural mock registry for Aura tests."""
    from unittest.mock import MagicMock, AsyncMock, patch
    from core.container import ServiceContainer
    
    # Mock AgencyBus to allow impulses to pass
    mock_bus = MagicMock()
    mock_bus.submit.return_value = True
    
    with patch("core.agency_bus.AgencyBus.get", return_value=mock_bus):
        # Brain / Cognitive Engine
        mock_cognition = MagicMock()
        mock_cognition.record_interaction = AsyncMock()
        mock_cognition.process_turn = AsyncMock(return_value="Mocked response")
        mock_cognition.think = AsyncMock(return_value=MagicMock(content="Mocked thought"))
        mock_cognition.think_stream = AsyncMock()
        
        async def mock_stream(*args, **kwargs):
            yield "Mocked "
            yield "stream"
        mock_cognition.think_stream.side_effect = mock_stream
        
        # Memory
        mock_memory = MagicMock()
        mock_memory.retrieve_unified_context = AsyncMock(return_value="Memories")
        mock_memory.commit_interaction = AsyncMock()
        mock_memory.run_maintenance = AsyncMock()
        mock_memory.get_hot_memory = AsyncMock(return_value={})
        mock_memory.store = AsyncMock()
        
        # Meta-Learning
        mock_meta = MagicMock()
        mock_meta.recall_strategy = AsyncMock(return_value={})
        mock_meta.index_experience = AsyncMock()
        mock_meta.run_maintenance = AsyncMock()

        mock_personality = MagicMock()
        mock_personality.update = MagicMock()
        mock_personality.filter_response = MagicMock(side_effect=lambda text: text)
        mock_personality.get_emotional_context_for_response = MagicMock(
            return_value={"mood": "neutral", "tone": "balanced", "emotional_state": {}}
        )
        mock_personality.get_time_context = MagicMock(return_value={"formatted": "12:00 PM"})
        mock_personality.get_sovereign_context = MagicMock(return_value="")
        mock_personality.current_mood = "balanced"

        mock_strategic_planner = MagicMock()
        mock_strategic_planner.get_next_task.return_value = None

        mock_project_store = MagicMock()
        mock_project_store.get_active_projects.return_value = []
        mock_project_store.get_tasks_for_project.return_value = []

        mock_knowledge_graph = MagicMock()
        mock_knowledge_graph.add_knowledge = MagicMock()
        mock_knowledge_graph.remember_person = MagicMock()
        mock_knowledge_graph.ask_question = MagicMock()
        
        # Senses & State
        mock_ls = MagicMock()
        mock_ls.update = AsyncMock()
        mock_ls.get_status = MagicMock(return_value={"health": 1.0, "status": { "initialized": True, "running": True }})
        mock_ls.current = MagicMock(curiosity=0.5, frustration=0.1, energy=0.8)
        
        mock_affect = MagicMock()
        mock_affect.state = MagicMock(dominant_emotion="Joy")
        mock_affect.get_current_state.return_value = {"valence": 0.5}
        
        # Core Registry
        ServiceContainer.register_instance("cognitive_engine", mock_cognition)
        ServiceContainer.register_instance("cognition", mock_cognition)
        ServiceContainer.register_instance("memory", mock_memory)
        ServiceContainer.register_instance("memory_facade", mock_memory)
        ServiceContainer.register_instance("metacognition", mock_meta)
        ServiceContainer.register_instance("meta_learning", mock_meta)
        ServiceContainer.register_instance("personality_engine", mock_personality)
        ServiceContainer.register_instance("strategic_planner", mock_strategic_planner)
        ServiceContainer.register_instance("project_store", mock_project_store)
        ServiceContainer.register_instance("knowledge_graph", mock_knowledge_graph)
        ServiceContainer.register_instance("affect_engine", mock_affect)
        ServiceContainer.register_instance("liquid_state", mock_ls)
        ServiceContainer.register_instance("conscious_substrate", mock_ls)
        
        # Infrastructure
        mock_watchdog = MagicMock()
        ServiceContainer.register_instance("watchdog", mock_watchdog)
        ServiceContainer.register_instance("output_gate", MagicMock(emit=AsyncMock()))
        ServiceContainer.register_instance("capability_engine", MagicMock(execute=AsyncMock(return_value={"ok": True})))
        
        # Fallbacks for missing services identified in audit
        mock_drives = MagicMock()
        mock_drives.satisfy = AsyncMock()
        mock_alignment = MagicMock()
        mock_alignment.filter_response = AsyncMock(side_effect=lambda x, *args, **kwargs: x)  # Returns the response unchanged
        for svc in ["homeostasis", "subsystem_audit", "lnn", "mortality", "identity", "curiosity",
                    "intent_router", "cognitive_router", "world_model",
                    "belief_graph", "output_gate"]:
            ServiceContainer.register_instance(svc, AsyncMock())
        # These need to be MagicMock because they have sync methods/properties used in pipeline
        ServiceContainer.register_instance("mycelium", MagicMock())
        ServiceContainer.register_instance("state_machine", MagicMock())
        ServiceContainer.register_instance("drives", mock_drives)
        ServiceContainer.register_instance("alignment_engine", mock_alignment)
            
        yield ServiceContainer

@pytest.fixture
def orchestrator(mock_container):
    """Hardened RobustOrchestrator fixture with full dependency injection."""
    from unittest.mock import MagicMock, AsyncMock
    from core.orchestrator import RobustOrchestrator
    from core.orchestrator.orchestrator_types import SystemStatus
    import asyncio
    import time

    # Initialize instance WITHOUT class patching
    orch = RobustOrchestrator()
    
    # Setup core status
    status_obj = SystemStatus()
    status_obj.initialized = True
    status_obj.running = True
    status_obj.cycle_count = 0
    status_obj.start_time = time.time()
    orch.status = status_obj
    
    # Ensure queues and locks exist
    orch.message_queue = asyncio.Queue()
    orch.reply_queue = asyncio.Queue()
    orch._lock = asyncio.Lock()
    orch._history_lock = asyncio.Lock()
    
    # Setup core dependencies from container
    for component in ["cognitive_engine", "memory", "capability_engine", 
                     "strategic_planner", "project_store", "intent_router",
                     "personality_engine", "world_model", "curiosity",
                     "knowledge_graph", "drives", "state_machine", 
                     "output_gate", "liquid_state", "mycelium"]:
        svc = mock_container.get(component)
        if component == "mycelium":
            # Mycelium has sync methods like match_hardwired and rooted_flow call
            from unittest.mock import MagicMock
            from core.orchestrator.main import AsyncNullContext
            svc = MagicMock()
            svc.rooted_flow.return_value = AsyncNullContext()
            svc.match_hardwired.return_value = None
        elif component == "state_machine":
             from unittest.mock import MagicMock, AsyncMock
             svc = MagicMock()
             svc.execute = AsyncMock()
        elif component == "intent_router":
             from unittest.mock import MagicMock, AsyncMock
             svc = MagicMock()
             svc.classify = AsyncMock(return_value="chitchat")
        elif component == "output_gate":
             from unittest.mock import AsyncMock
             svc = AsyncMock()
        setattr(orch, component, svc)
        setattr(orch, f"_{component}", svc)
    
    # Mock specific async methods that tests expect to be mocked
    orch.hooks = MagicMock()
    orch.hooks.trigger = AsyncMock()
    
    # Ensure _finalize_response and _handle_incoming_message remain UNMOCKED
    # unless a specific test mocks them.
    
    try:
        yield orch
    finally:
        try:
            status = getattr(orch, "status", None)
            if status is not None:
                if hasattr(status, "running"):
                    status.running = False
                if hasattr(status, "is_processing"):
                    status.is_processing = False

            stop_event = getattr(orch, "_stop_event", None)
            if stop_event is not None and hasattr(stop_event, "set"):
                stop_event.set()

            async def _cleanup_tasks():
                for attr in ("_current_thought_task", "_autonomous_task"):
                    task = getattr(orch, attr, None)
                    if isinstance(task, asyncio.Task) and not task.done():
                        task.cancel()
                        try:
                            await task
                        except Exception:
                            pass

            asyncio.run(_cleanup_tasks())
        except Exception:
            pass
