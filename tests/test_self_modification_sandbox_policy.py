"""Tests for self-modification sandbox policy.

Verifies that:
  - Generated skill writes are allowed
  - Core governance spine writes are refused
  - Config writes are refused
  - Will/Gateway writes are refused
  - Patch proposals are allowed
  - Tests directory is writable
  - Plugin directory is writable
"""
from __future__ import annotations

import pytest

from core.self_modification.safe_modification import SafeSelfModification


@pytest.fixture
def safe_mod():
    return SafeSelfModification(code_base_path=".")


class TestAllowedPaths:
    """Paths that Aura should be able to modify for recursive improvement."""

    def test_skill_write_allowed(self, safe_mod):
        assert safe_mod.is_allowed_path("skills/generated/new_skill.py")

    def test_plugin_write_allowed(self, safe_mod):
        assert safe_mod.is_allowed_path("plugins/generated/my_plugin.py")

    def test_core_cognitive_allowed(self, safe_mod):
        assert safe_mod.is_allowed_path("core/consciousness/endogenous_fitness.py")

    def test_core_brain_allowed(self, safe_mod):
        assert safe_mod.is_allowed_path("core/brain/inference_gate.py")

    def test_interface_route_allowed(self, safe_mod):
        assert safe_mod.is_allowed_path("interface/routes/chat.py")

    def test_test_write_allowed(self, safe_mod):
        assert safe_mod.is_allowed_path("tests/test_new_module.py")

    def test_patch_proposal_allowed(self, safe_mod):
        assert safe_mod.is_allowed_path("patches/proposals/fix_123.py")

    def test_scratch_allowed(self, safe_mod):
        assert safe_mod.is_allowed_path("scratch/temp_experiment.py")


class TestProtectedPaths:
    """Paths that Aura must NEVER modify at runtime (governance spine)."""

    def test_will_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/will.py")

    def test_authority_gateway_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/executive/authority_gateway.py")

    def test_executive_core_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/executive/executive_core.py")

    def test_runtime_gateways_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/runtime/gateways.py")

    def test_runtime_conformance_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/runtime/conformance.py")

    def test_runtime_executors_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/runtime/executors.py")

    def test_runtime_errors_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/runtime/errors.py")

    def test_runtime_receipts_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/runtime/receipts.py")

    def test_runtime_atomic_writer_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/runtime/atomic_writer.py")

    def test_memory_write_gateway_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/memory/memory_write_gateway.py")

    def test_state_gateway_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/state/state_gateway.py")

    def test_state_repository_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/state/state_repository.py")

    def test_security_dir_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/security/threat_model.py")

    def test_guardians_dir_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/guardians/safety_gate.py")

    def test_constitution_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/constitution.py")

    def test_config_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/config.py")

    def test_self_mod_engine_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/self_modification/safe_modification.py")

    def test_llm_router_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/brain/llm/llm_router.py")

    def test_model_registry_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/brain/llm/model_registry.py")

    def test_aura_main_protected(self, safe_mod):
        assert safe_mod.is_protected_path("aura_main.py")

    def test_phi_core_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/consciousness/phi_core.py")

    def test_hierarchical_phi_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/consciousness/hierarchical_phi.py")

    def test_actor_bus_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/bus/actor_bus.py")

    def test_shared_memory_bus_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/bus/shared_mem_bus.py")

    def test_scar_formation_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/memory/scar_formation.py")

    def test_self_modification_tier_policy_protected(self, safe_mod):
        assert safe_mod.is_protected_path("core/self_modification/mutation_tiers.py")


class TestAllowedButNotProtected:
    """Cognitive modules that ARE in allowed_paths but NOT in protected_paths.
    These are the modules Aura can safely improve herself."""

    def test_context_assembler_modifiable(self, safe_mod):
        path = "core/brain/llm/context_assembler.py"
        assert safe_mod.is_allowed_path(path)
        assert not safe_mod.is_protected_path(path)

    def test_endogenous_fitness_modifiable(self, safe_mod):
        path = "core/consciousness/endogenous_fitness.py"
        assert safe_mod.is_allowed_path(path)
        assert not safe_mod.is_protected_path(path)

    def test_plasticity_monitor_modifiable(self, safe_mod):
        path = "core/consciousness/plasticity_monitor.py"
        assert safe_mod.is_allowed_path(path)
        assert not safe_mod.is_protected_path(path)

    def test_executive_closure_modifiable(self, safe_mod):
        path = "core/consciousness/executive_closure.py"
        assert safe_mod.is_allowed_path(path)
        assert not safe_mod.is_protected_path(path)

    def test_context_gate_modifiable(self, safe_mod):
        path = "core/brain/llm/context_gate.py"
        assert safe_mod.is_allowed_path(path)
        assert not safe_mod.is_protected_path(path)

    def test_phi_core_sealed_not_modifiable(self, safe_mod):
        path = "core/consciousness/phi_core.py"
        assert safe_mod.is_allowed_path(path)
        assert safe_mod.is_protected_path(path)
