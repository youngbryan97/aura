import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
root = Path.home() / ".aura"
sys.path.append(str(root))

async def main():
    print("🚀 Zenith-HF2: Triggering LoRA Training cycle...")
    
    # We need to mock a few things if we are running standalone
    os.environ["AURA_ROOT"] = str(root)
    
    from core.adaptation.self_optimizer import get_self_optimizer
    
    optimizer = get_self_optimizer()
    
    # Run with 20 iterations for fast verification of the cognitive upgrade.
    result = await optimizer.optimize(iters=20, batch_size=4)
    
    if result["ok"]:
        print(f"✅ Success! Duration: {result['duration']:.2f}s")
        print(f"📂 Adapter: {result['adapter']}")
        print(f"📊 Samples: {result['samples']}")
    else:
        print(f"❌ Failed: {result['error']}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
