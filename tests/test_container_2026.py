################################################################################

import pytest
import asyncio
from core.container import ServiceContainer, ServiceLifetime
from core.exceptions import (
    ServiceNotFoundError,
    CircularDependencyError,
    LifecycleError,
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
    ServiceContainer.register("A", lambda B: "A", dependencies=["B"])
    ServiceContainer.register("B", lambda A: "B", dependencies=["A"])
    
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

def test_factory_failure():
    def broken_factory():
        raise ValueError("Boom")
    
    ServiceContainer.register("broken", broken_factory)
    with pytest.raises(LifecycleError) as excinfo:
        ServiceContainer.get("broken")
    assert "Boom" in str(excinfo.value)


##
