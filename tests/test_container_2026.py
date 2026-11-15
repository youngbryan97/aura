################################################################################


import pytest

from core.container import ServiceContainer, ServiceLifetime
from core.exceptions import (
    CircularDependencyError,
    LifecycleError,
    ServiceNotFoundError,
)


@pytest.fixture(autouse=True)
def clean_container():
    ServiceContainer.clear()
    yield
    ServiceContainer.clear()

def test_basic_registration_and_resolution():
    ServiceContainer.register("simple", lambda: "hello")
    assert ServiceContainer.get("simple") == "hello"

def test_singleton_behavior():
    class MyService:
        pass
    ServiceContainer.register("singleton", MyService)
    s1 = ServiceContainer.get("singleton")
    s2 = ServiceContainer.get("singleton")
    assert s1 is s2


def test_runtime_service_bridge_observes_without_cold_starting_factory():
    from core.runtime.service_registry import get_runtime_service

    created = []

    def factory():
        instance = object()
        created.append(instance)
        return instance

    ServiceContainer.register("runtime_probe", factory)

    assert get_runtime_service("runtime_probe", default=None) is None
    assert created == []

    instance = ServiceContainer.get("runtime_probe")
    assert get_runtime_service("runtime_probe", default=None) is instance
    assert created == [instance]

def test_transient_behavior():
    class MyService:
        pass
    ServiceContainer.register("transient", MyService, lifetime=ServiceLifetime.TRANSIENT)
    t1 = ServiceContainer.get("transient")
    t2 = ServiceContainer.get("transient")
    assert t1 is not t2  # Wait, transient should be new?
    # Ah, let me check the implementation of TRANSIENT in my edit...
    # I kept Phase 3: "if descriptor.lifetime == ServiceLifetime.SINGLETON: descriptor.instance = instance"
    # So transient is NOT stored. Good.

def test_auto_wiring():
    ServiceContainer.register("database", lambda: {"db": "postgres"})
    
    def factory(database):
        return f"Connected to {database['db']}"
    
    ServiceContainer.register("app", factory)
    
    # "app" depends on "database" via parameter name
    assert ServiceContainer.get("app") == "Connected to postgres"

def test_circular_dependency():
    ServiceContainer.register("A", lambda B: "A", dependencies=["B"])  # noqa: N803
    ServiceContainer.register("B", lambda A: "B", dependencies=["A"])  # noqa: N803
    
    with pytest.raises(CircularDependencyError):
        ServiceContainer.get("A")

def test_service_not_found():
    with pytest.raises(ServiceNotFoundError):
        ServiceContainer.get("missing")

def test_lifecycle_hooks():
    class HookService:
        def __init__(self):
            self.started = False
        def on_start(self):
            self.started = True

    ServiceContainer.register("hooked", HookService)
    s = ServiceContainer.get("hooked")
    assert s.started is True

@pytest.mark.asyncio
async def test_wake_async():
    class AsyncService:
        def __init__(self):
            self.awake = False
        async def on_start_async(self):
            self.awake = True

    ServiceContainer.register("async_srv", AsyncService, required=True)
    await ServiceContainer.wake()
    s = ServiceContainer.get("async_srv")
    assert s.awake is True

def test_registration_during_shutdown_suppresses_softly():
    """Lived 2026-07-12: the shutdown-latch registration guard RAISED
    ContainerError out of the aura_now telemetry republish INSIDE
    kernel.shutdown(), aborting teardown before the vault closed — leaked
    non-daemon aiosqlite worker threads held interpreter exit until the
    chunk runner's silent 2400s timeout. Suppression must be a soft no-op:
    recorded, logged, never raised into the path that touched it."""
    from core.runtime.shutdown_coordinator import clear_shutdown_request, request_shutdown

    clear_shutdown_request()
    try:
        request_shutdown("unit-test-suppression")
        # Must NOT raise — new registrations are silently refused.
        ServiceContainer.register_instance("shutdown_probe_service", object())
        assert ServiceContainer.get("shutdown_probe_service", default=None) is None
        ServiceContainer.register("shutdown_probe_factory", lambda: object())
        assert ServiceContainer.get("shutdown_probe_factory", default=None) is None
    finally:
        clear_shutdown_request()


def test_zero_param_factory_with_declared_dependencies_boots():
    """Boot-regression pin (lived 2026-07-10): actor_supervision declared
    dependencies=['runtime_control_plane'] with a ZERO-parameter factory.
    The legacy positional fallback appended the resolved dependency anyway
    -> TypeError -> LifecycleError -> the whole runtime failed its first
    health evaluation and never opened :8000. Declared dependencies on a
    zero-parameter factory are ordering constraints: resolve them, then
    call the factory with no arguments."""
    order: list[str] = []

    ServiceContainer.register("upstream", lambda: order.append("upstream") or "up")

    def zero_param_factory():
        order.append("dependent")
        return "dependent-service"

    ServiceContainer.register(
        "dependent", zero_param_factory, dependencies=["upstream"]
    )

    assert ServiceContainer.get("dependent") == "dependent-service"
    assert order == ["upstream", "dependent"]  # dependency resolved FIRST


def test_factory_failure():
    factory_calls = []

    def broken_factory():
        factory_calls.append("attempted")
        raise ValueError("Boom")
    
    ServiceContainer.register("broken", broken_factory)
    with pytest.raises(LifecycleError) as excinfo:
        ServiceContainer.get("broken")
    assert factory_calls == ["attempted"]
    assert "Boom" in str(excinfo.value)


##


def test_register_instance_upsert_swaps_without_descriptor_rebuild():
    """Hot-path contract (live 8.3s loop-lag root, Jul 9): re-publishing a
    value under an existing non-protected name swaps the instance in place —
    no new ServiceDescriptor, no frame walking, provenance from the first
    registration stands."""
    from core.container import ServiceContainer

    name = "test_upsert_hot_value"
    ServiceContainer.register_instance(name, {"seq": 1}, required=False)
    desc_before = ServiceContainer._services.get(name)
    ServiceContainer.register_instance(name, {"seq": 2}, required=False)
    desc_after = ServiceContainer._services.get(name)

    assert desc_after is desc_before, "upsert must reuse the descriptor object"
    assert ServiceContainer.get(name) == {"seq": 2}
    # cleanup
    ServiceContainer._services.pop(name, None)


def test_determine_caller_is_filesystem_free(monkeypatch):
    """The caller display must never touch the filesystem (Path.resolve/stat
    on the event loop was the lag mechanism)."""
    import core.container as container_mod

    def _boom(*a, **k):
        raise AssertionError("filesystem touched during caller determination")

    # Only across the call it guards: patched for the whole test, Path.stat
    # stayed broken through the suite's own teardown, which stats a file
    # (core/brain/llm/thinking_reserve.reset_for_test), and the test erred there.
    with monkeypatch.context() as patched:
        patched.setattr(container_mod.Path, "resolve", _boom)
        patched.setattr(container_mod.Path, "stat", _boom, raising=False)
        caller = container_mod._determine_caller()
    assert isinstance(caller, str) and caller
