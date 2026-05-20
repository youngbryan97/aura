import asyncio
from types import SimpleNamespace

from core.container import ServiceContainer
from core.kernel import kernel_interface as kernel_module
from core.kernel.kernel_interface import MAX_CONSECUTIVE_TICK_FAILURES, KernelInterface
from core.runtime.errors import get_degradation_tracker, get_subsystem_registry


class WorkingKernel:
    def __init__(self):
        self.state = None
        self.organs = {"llm": SimpleNamespace(instance=self)}
        self.closed = False
        self.ticks = 0

    async def boot(self):
        self.state = SimpleNamespace(
            cognition=SimpleNamespace(
                working_memory=[],
                current_origin=None,
                last_response="",
            )
        )

    async def tick(self, message, priority=False):
        self.ticks += 1
        self.state.cognition.last_response = f"handled:{message}:{priority}"
        return SimpleNamespace(response_preview="preview")

    def loop_state(self):
        return {
            "phi": "0.42",
            "valence": "0.7",
            "mood": "focused",
            "arousal": "0.9",
            "status": "engaged",
        }

    async def shutdown(self):
        self.closed = True


class MissingCognitionKernel(WorkingKernel):
    async def boot(self):
        self.state = SimpleNamespace()

    async def tick(self, message, priority=False):
        self.ticks += 1
        self.state.last_message = message
        return SimpleNamespace(response_preview="should-not-run")


class FailingTickKernel(WorkingKernel):
    async def tick(self, message, priority=False):
        self.ticks += 1
        self.last_priority = priority
        raise RuntimeError("tick lane broken")


def _reset_process_state():
    ServiceContainer.clear()
    get_degradation_tracker().reset()
    kernel_module._ki = None
    KernelInterface._instance = None


def test_kernel_interface_processes_turn_and_updates_health():
    async def scenario():
        _reset_process_state()
        orch = SimpleNamespace(_last_user_interaction_time=0.0)
        ServiceContainer.register_instance("orchestrator", orch)
        ki = KernelInterface()
        kernel = WorkingKernel()

        await ki.boot(kernel=kernel)
        response = await ki.process("hello", origin="user", priority=True)

        assert response == "handled:hello:True"
        assert kernel.state.cognition.working_memory[-1]["content"] == "hello"
        assert kernel.state.cognition.current_origin == "user"
        assert orch._last_user_interaction_time > 0.0
        assert ki.health_snapshot()["ready"] is True
        assert get_subsystem_registry().get("kernel_interface").status == "healthy"

    asyncio.run(scenario())


def test_kernel_interface_diverts_state_handoff_fault_before_tick():
    async def scenario():
        _reset_process_state()
        ki = KernelInterface()
        kernel = MissingCognitionKernel()

        await ki.boot(kernel=kernel)
        response = await ki.process("hello", origin="system")

        assert response == ""
        assert kernel.ticks == 0
        snapshot = ki.health_snapshot()
        assert snapshot["ready"] is True
        assert "cognition" in snapshot["last_fault"]
        assert get_degradation_tracker().count("kernel_interface") == 1
        assert get_subsystem_registry().get("kernel_interface").status == "degraded"

    asyncio.run(scenario())


def test_kernel_interface_opens_tick_circuit_after_repeated_failures():
    async def scenario():
        _reset_process_state()
        ki = KernelInterface()
        kernel = FailingTickKernel()

        await ki.boot(kernel=kernel)
        for _ in range(MAX_CONSECUTIVE_TICK_FAILURES):
            assert await ki.process("hello", origin="system", priority=True) == ""

        snapshot = ki.health_snapshot()
        assert snapshot["ready"] is False
        assert snapshot["consecutive_tick_failures"] == MAX_CONSECUTIVE_TICK_FAILURES
        assert kernel.ticks == MAX_CONSECUTIVE_TICK_FAILURES
        assert get_subsystem_registry().get("kernel_interface").status == "failed_closed"

    asyncio.run(scenario())


def test_kernel_interface_attach_without_state_repository_is_explicitly_unready():
    async def scenario():
        _reset_process_state()
        orch = SimpleNamespace()

        ki = await KernelInterface.attach_to_orchestrator(orch)

        assert orch.kernel_interface is ki
        assert ServiceContainer.get("kernel_interface") is ki
        assert ki.is_ready() is False
        assert "StateRepository" in ki.health_snapshot()["last_fault"]
        assert get_subsystem_registry().get("kernel_interface").status == "unavailable"

    asyncio.run(scenario())
