import asyncio
import logging
import multiprocessing
import os
from multiprocessing import shared_memory

# Add project root to sys.path
import sys
sys.path.append(os.getcwd())

from core.bus.actor_bus import ActorBus
from core.kernel.kernel_interface import KernelInterface
from core.orchestrator.main import RobustOrchestrator
from core.event_bus import get_event_bus, reset_event_bus
from core.container import ServiceContainer
from core.resilience.startup_validator import StartupValidator
from core.supervisor.tree import reset_tree

logging.basicConfig(level=logging.INFO)

async def _reset_runtime_boundary():
    """Return shared process globals to a clean slate before/after validation."""
    await KernelInterface.reset_instance()
    ServiceContainer.clear()
    await ActorBus.reset_singleton()
    reset_tree()
    await reset_event_bus()
    try:
        shm = shared_memory.SharedMemory(name="aura_state_shm")
    except FileNotFoundError:
        return
    except Exception as shm_err:
        print(f"⚠️ Shared memory probe failed: {shm_err}")
        return

    try:
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass
    except Exception as shm_err:
        print(f"⚠️ Shared memory cleanup failed: {shm_err}")

async def _run_validation():
    await _reset_runtime_boundary()
    print("\n🔍 LOADING ORCHESTRATOR...")
    orch = RobustOrchestrator()
    bus = get_event_bus()

    try:
        print("🚀 INIT SUBSYSTEMS...")
        try:
            # We manually run the init steps that register the services
            await orch._async_init_subsystems()
        except Exception as e:
            print(f"❌ Subsystem init failed: {e}")
            # Continue to see what registered

        print("\n🧐 RUNNING STARTUP VALIDATOR...")
        validator = StartupValidator(orch)
        success = await validator.validate_all()

        if success:
            print("\n✅ ALL CRITICAL CHECKS PASSED.")
        else:
            print("\n❌ VALIDATION FAILED.")

        assert success, "Startup validator reported critical failures."
    finally:
        bus._use_redis = False
        try:
            if getattr(orch, "status", None):
                orch.status.running = True
            await orch.stop()
        except Exception as stop_err:
            print(f"⚠️ Orchestrator cleanup failed: {stop_err}")

        try:
            await orch.supervisor.stop()
        except Exception as supervisor_err:
            print(f"⚠️ Supervisor cleanup failed: {supervisor_err}")

        for proc in multiprocessing.active_children():
            if proc.name.startswith("AuraActor:"):
                proc.terminate()
                proc.join(timeout=1.0)
                if proc.is_alive():
                    proc.kill()

        try:
            await bus.shutdown()
        except Exception as bus_err:
            print(f"⚠️ Event bus cleanup failed: {bus_err}")

        await _reset_runtime_boundary()

def test_validation():
    asyncio.run(_run_validation())

if __name__ == "__main__":
    asyncio.run(_run_validation())
