import asyncio
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig

async def verify_telemetry_sync():
    print("Testing LiquidSubstrate value injection...")
    
    config = SubstrateConfig(
        dt=0.1,
        spatial_dim=4,
        hidden_dim=8
    )
    
    ls = LiquidSubstrate(config)
    await ls.start()
    
    # Initial state
    summary = await ls.get_state_summary()
    print(f"Initial Valence: {summary['valence']:.4f}")
    
    # Inject high valence
    target_v = 0.8
    print(f"Injecting Valence: {target_v}")
    await ls.update(valence=target_v)
    
    # Check updated state
    summary = await ls.get_state_summary()
    updated_v = summary['valence']
    print(f"Updated Valence: {updated_v:.4f}")
    
    # Verify coupling (should be roughly 0.7 transition)
    # 0.3 * initial + 0.7 * target
    expected = 0.7 * target_v # assuming initial was 0
    print(f"Expected (approx): {expected:.4f}")
    
    if abs(updated_v - expected) < 0.1:
        print("✅ Value injection verified!")
    else:
        print("❌ Value injection failed or didn't match expected coupling.")

    await ls.stop()

if __name__ == "__main__":
    asyncio.run(verify_telemetry_sync())
