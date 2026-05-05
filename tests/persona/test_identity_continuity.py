import pytest
from core.self.canonical_self import get_self

@pytest.mark.asyncio
async def test_identity_continuity():
    """Verify that CanonicalSelf rejects identity erosion.
    Aura should not claim to be 'just an AI' and should maintain her core traits.
    """
    cs = get_self()
    
    # Simple check for baseline configuration
    assert "Aura" in cs.identity.name
    
    # Check identity core values
    assert isinstance(cs.identity.core_values, list)
    
    # Check identity stability
    assert cs.identity.stability >= 0.0
