"""Tests for Black Hole Memory System (BHMS).

Note: These tests require the Horcrux subsystem to initialize correctly.
If the environment lacks shard files, tests will be skipped gracefully.
"""
import os
import json
import logging
import pytest

logging.basicConfig(level=logging.INFO)


@pytest.fixture
def vault(tmp_path):
    """Create a BlackHoleVault in a temporary directory for isolation."""
    data_dir = str(tmp_path / "vault")
    from core.memory.black_hole_vault import BlackHoleVault
    v = BlackHoleVault(data_dir=data_dir)
    return v


def test_vault_initialization(vault):
    """Verify vault initializes and has a master key."""
    assert vault is not None
    assert hasattr(vault, "key")
    assert len(vault.key) > 0
    print(f"Master Key Initialized: {vault.key[:10]}...")


def test_memorize_and_search(vault):
    """Verify add_memory and search_similar work."""
    vault.add_memory("Aura's core temperature is zero.", metadata={"type": "test"})
    vault.add_memory("Black holes emit Hawking radiation.", metadata={"type": "fact"})

    results = vault.search_similar("Hawking")
    assert len(results) > 0
    contents = [r["content"] for r in results]
    assert any("Hawking" in c for c in contents)


def test_vault_auto_healing(vault):
    """Verify vault can regenerate deleted shards."""
    # Shard is in the parent of vault data_dir
    aura_dir = os.path.dirname(vault.data_dir)
    file_shard = os.path.join(aura_dir, ".core_seed")

    if not os.path.exists(file_shard):
        pytest.skip(f"No shard file at {file_shard} to delete for healing test")

    os.remove(file_shard)
    assert not os.path.exists(file_shard)

    # Re-init should auto-heal
    from core.memory.black_hole_vault import BlackHoleVault
    vault2 = BlackHoleVault(data_dir=vault.data_dir)
    assert hasattr(vault2, "key")

    if os.path.exists(file_shard):
        print("SUCCESS: Shard was automatically regenerated!")
    else:
        pytest.fail("Missing shard was not healed after re-init")
