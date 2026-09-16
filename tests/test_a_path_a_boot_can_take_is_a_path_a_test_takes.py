"""LIVE 2026-09-16: the desktop process died on a ValueError two months old.

``_start_state_vault_actor`` built its supervisor spec with
``restart_policy="permanent"``; the supervisor has accepted only
always/transient/never since July. Every boot before this one found the vault
already usable and returned before the spec was built. On a host at load 34
the vault's handshake timed out in the State Repository stage, the boot
reached the line for the first time, and the runtime exited.
"""

from __future__ import annotations

from core.orchestrator.mixins.boot.boot_resilience import _state_vault_actor_spec
from core.supervisor.tree import ActorSpec


def _entry(*_args: object) -> None:
    return None


def test_the_state_vault_spec_the_boot_builds_is_one_the_supervisor_accepts():
    spec = _state_vault_actor_spec(ActorSpec, _entry, "/tmp/aura_state.db")
    assert spec.name == "state_vault"
    assert spec.restart_policy == "always"
    assert spec.args == ("/tmp/aura_state.db",)
    assert spec.entry_point is _entry


def test_the_boot_builds_its_spec_through_the_tested_builder():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent
        / "core/orchestrator/mixins/boot/boot_resilience.py"
    ).read_text(encoding="utf-8")
    assert source.count("permanent") == 1  # the docstring that records the incident
    assert "spec = _state_vault_actor_spec(" in source
