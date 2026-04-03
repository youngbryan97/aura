import pytest
import asyncio
import numpy as np
from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig
from core.collective.swarm_protocol import SwarmProtocol

@pytest.mark.asyncio
async def test_neural_calibration():
    """Verify that long-term calibration shifts weights."""
    config = SubstrateConfig(neuron_count=10)
    substrate = LiquidSubstrate(config)
    
    initial_W = substrate.W.copy()
    resonance_vec = np.ones(10)
    
    await substrate.long_term_calibration(resonance_vec)
    
    # Check that weights moved towards the resonance matrix
    # The resonance matrix of all ones will tend to make weights more positive
    assert not np.array_equal(substrate.W, initial_W)
    assert np.mean(substrate.W) != np.mean(initial_W)

@pytest.mark.asyncio
async def test_swarm_gossip():
    """Verify SwarmProtocol starts and can broadcast."""
    swarm = SwarmProtocol(port=10005)
    await swarm.start()
    assert swarm.running is True
    assert swarm.node_id is not None
    
    # Mock broadcast
    msg = {"type": "test_gossip", "data": "hello"}
    # This just tests it doesn't crash without peers
    await swarm.broadcast(msg)
    
    await swarm.stop()
    assert swarm.running is False

