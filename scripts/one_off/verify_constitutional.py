import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from core.kernel.aura_kernel import AuraKernel, KernelConfig
from core.state.aura_state import AuraState

async def verify():
    print("🚀 Verifying Constitutional Closure...")
    config = KernelConfig()
    from core.state.state_repository import StateRepository
    vault = StateRepository()
    
    kernel = AuraKernel(config=config, vault=vault)
    await kernel.boot()
    
    print("\n--- Running Sovereign Tick ---")
    # By running the tick natively, we ensure the pipeline executes and states update.
    response = await kernel.tick("Hello, Aura. This is a constitutional check.", priority=True)
    print(f"Kernel response: {response.response_preview if response else 'None'}")
    
    # Introspect constitutional trace fields in CognitiveContext
    ctx = kernel.state.cognition
    print("\n[Constitutional State Trace]")
    print(f"last_kernel_cycle_id: {ctx.last_kernel_cycle_id}")
    print(f"kernel_decision_count: {ctx.kernel_decision_count}")
    print(f"kernel_veto_count: {ctx.kernel_veto_count}")
    print(f"last_action_source: {ctx.last_action_source}")
    print(f"last_veto_reasons: {ctx.last_veto_reasons}")
    
    assert ctx.last_kernel_cycle_id is not None, "Cycle ID was not stamped!"
    assert ctx.kernel_decision_count > 0, "Decision count did not increment!"
    print("\n✅ Constitutional closure metadata is populating correctly!")
    
    try:
        await kernel.shutdown()
    except Exception:
        pass

if __name__ == "__main__":
    asyncio.run(verify())
