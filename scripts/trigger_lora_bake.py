import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

async def main():
    print("🚀 Zenith-HF2: Triggering LoRA Training cycle...")
    
    # We need to mock a few things if we are running standalone
    os.environ["AURA_ROOT"] = str(root)
    
    from core.adaptation.self_optimizer import get_self_optimizer
    
    optimizer = get_self_optimizer()
    
    # Run with 1000 iterations for a massive, high-fidelity personality bake.
    result = await optimizer.optimize(iters=1000, batch_size=4)
    
    if result["ok"]:
        print(f"✅ Success! Duration: {result['duration']:.2f}s")
        print(f"📂 Adapter: {result['adapter']}")
        print(f"📊 Samples: {result['samples']}")
    else:
        print(f"❌ Failed: {result['error']}")
        raise SystemExit(1)

if __name__ == "__main__":
    asyncio.run(main())
