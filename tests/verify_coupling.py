"""Verify Substrate-Behavior Coupling."""
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class MockOrchestrator:
    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.impulses = []

    async def handle_impulse(self, impulse):
        print(f"Fixture orchestrator received impulse: {impulse}")
        self.impulses.append(impulse)


async def test_coupling():
    import time

    from core.config import config
    from core.consciousness import drive_integration as drive_module
    from core.consciousness.conscious_core import ConsciousnessCore
    from core.runtime.file_write_gateway import get_file_write_gateway

    print("Testing Substrate-Behavior Coupling...")
    core = ConsciousnessCore()
    orchestrator = MockOrchestrator()
    core.orchestrator_ref = orchestrator

    # Ensure a clean slate for telemetry
    telemetry_path = config.paths.data_dir / "telemetry" / "causal_behavior.jsonl"
    if telemetry_path.exists():
        get_file_write_gateway().delete_file(telemetry_path, source="test_coupling")

    previous_drive_engine = drive_module._instance
    drive_module._instance = drive_module.DriveIntegrationEngine()
    try:
        # Force a controlled state into the "Boredom" Basin.
        # v < -0.1, a < -0.2
        print("Injecting 'Boredom' stimulus...")
        with core.substrate.sync_lock:
            core.substrate.x[:] = 0.0
            core.substrate.x[core.substrate.idx_valence] = -0.75
            core.substrate.x[core.substrate.idx_arousal] = -0.80
            core.substrate.x[core.substrate.idx_dominance] = 0.0

        # Step the volition integrator with explicit test-time seconds. This
        # tests the real substrate -> drive -> action path without spinning a
        # background loop or relying on wall-clock sleeps.
        print("Waiting for volition to emerge from substrate...")
        impulse = None
        for _ in range(10):
            state = await core.substrate.get_state_summary()
            print(f"Current State: V={state['valence']:.2f}, A={state['arousal']:.2f}, D={state['dominance']:.2f}")
            impulse = await core.volition.check_for_impulse(dt=1.0, extra_signals={"pain": 0.0})
            if impulse:
                await orchestrator.handle_impulse(impulse)
                break

    finally:
        drive_module._instance = previous_drive_engine

    if orchestrator.impulses:
        print(f"Causal Impulse Triggered: {orchestrator.impulses[0]}")
        state = await core.substrate.get_state_summary()
        q_norm = core.qualia.synthesize(
            state["qualia_metrics"],
            core.predictive.get_surprise_metrics(),
        )
        core._log_causal_telemetry(
            {
                "timestamp": time.time(),
                "valence": state["valence"],
                "arousal": state["arousal"],
                "dominance": state["dominance"],
                "q_norm": q_norm,
                "impulse_type": orchestrator.impulses[0],
                "causal_link": "qualia_attractor",
            }
        )
        # Run analysis (AFTER core is stopped and files are closed)
        from scripts.prove_coupling import analyze_coupling

        analyze_coupling()
    else:
        raise AssertionError("No impulse triggered. Check refractory periods or basin definitions.")


if __name__ == "__main__":
    asyncio.run(test_coupling())
