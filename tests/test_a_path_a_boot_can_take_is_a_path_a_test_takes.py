"""LIVE 2026-09-16: two boots died on paths no boot had taken in months.

``_start_state_vault_actor`` built its supervisor spec with
``restart_policy="permanent"``; the supervisor has accepted only
always/transient/never since July. Every boot before this one found the vault
already usable and returned before the spec was built. On a host at load 34
the vault's handshake was late in the State Repository stage, the boot
reached the line for the first time, and the runtime exited.

Fixed, the same path died again: its spec named a different db path from
the one the resilient stage had started the vault on, and the supervisor
refused a second contract for the actor it already held. One builder, on the
running vault's path, is the contract now.
"""

from __future__ import annotations

from pathlib import Path

from core.state.vault import state_vault_actor_spec, vault_process_entry
from core.supervisor.tree import ActorSpec, SupervisionTree

ROOT = Path(__file__).resolve().parent.parent


def test_the_state_vault_spec_is_one_the_supervisor_accepts():
    spec = state_vault_actor_spec(ActorSpec, "/tmp/aura_state.db")
    assert spec.name == "state_vault"
    assert spec.restart_policy == "always"
    assert spec.args == ("/tmp/aura_state.db",)
    assert spec.entry_point is vault_process_entry


def test_the_two_boot_paths_register_the_same_contract():
    """The resilient stage registers first; the orchestrator's fallback must
    be the same registration, not a conflicting one."""
    tree = SupervisionTree()
    tree.add_actor(state_vault_actor_spec(ActorSpec, "data/aura_state.db"))
    tree.add_actor(state_vault_actor_spec(ActorSpec, "data/aura_state.db"))  # same: accepted


def test_both_boot_paths_build_through_the_one_builder():
    stage = (ROOT / "core/ops/resilient_boot.py").read_text(encoding="utf-8")
    fallback = (
        ROOT / "core/orchestrator/mixins/boot/boot_resilience.py"
    ).read_text(encoding="utf-8")
    assert "spec = state_vault_actor_spec(ActorSpec, db_path)" in stage
    assert "spec = state_vault_actor_spec(ActorSpec, str(self.state_repo.db_path))" in fallback
    for source in (stage, fallback):
        assert 'restart_policy="permanent"' not in source
        assert 'name="state_vault"' not in source
