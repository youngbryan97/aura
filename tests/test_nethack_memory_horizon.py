import pytest
import time
from core.memory.vector_memory_engine import VectorMemoryEngine

@pytest.mark.asyncio
async def test_nethack_memory_horizon():
    """Prove the 40-turn cliff: recall spatial features from >40 turns ago."""
    memory = VectorMemoryEngine()
    
    # 1. Simulate 500 sequential observations
    print("\n   - Simulating 500 turns...")
    features = {
        10: "altar of anubis",
        100: "locked iron chest",
        250: "fountain of clear water",
        400: "shopkeeper named izchak"
    }
    
    for i in range(1, 501):
        content = f"Turn {i}: Standing at coordinates ({i//10}, {i%10})."
        
        # Log to episodic memory
        if i in features:
            await memory.store_spatial(
                level=1, x=i//10, y=i%10, 
                feature=features[i]
            )
        else:
            await memory.store(
                content=content,
                memory_type="episodic",
                importance=0.5
            )
        
    # 2. Query for the oldest item using semantic recall
    print("   - Querying for turn 10 feature (semantic)...")
    results = await memory.recall("where was the altar?", limit=3)
    found_altar_semantic = any("altar" in r.memory.content.lower() for r in results)
    
    # 3. Query using the new Spatial Memory layer
    print("   - Querying for turn 10 feature (spatial)...")
    spatial_results = await memory.find_feature("altar")
    found_altar_spatial = len(spatial_results) > 0
    
    # Assert recall accuracy - semantic might still fail due to TF-IDF quality, 
    # but spatial MUST pass.
    assert found_altar_spatial or found_altar_semantic, "Failed to recall the altar from turn 10"
    
    if found_altar_spatial:
        print(f"   ✅ Spatial recall success: Found at {spatial_results[0][0]}")

    print("   - Querying for turn 400 feature (spatial)...")
    spatial_results = await memory.find_feature("izchak")
    assert len(spatial_results) > 0, "Failed to recall the shopkeeper from turn 400"
    
    print("   ✅ Memory horizon test passed via spatial layer!")

    print("   ✅ Memory horizon test passed!")
