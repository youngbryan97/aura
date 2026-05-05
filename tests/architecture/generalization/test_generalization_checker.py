import pytest

from core.environment.generalization_checker import GeneralizationClaim


def test_feature_has_generalization_claim_and_cross_environment_test():
    claim = GeneralizationClaim(
        feature_name="inventory_parser",
        general_primitive="object_container_state_parsing",
        primary_environment="terminal_grid:nethack",
        secondary_environments=["browser:shopping_cart", "email:attachments", "codebase:file_tree"],
        environment_specific_code_paths=["core/environments/terminal_grid/nethack_parser.py"],
        shared_code_paths=["core/environment/ontology.py", "core/environment/belief_graph.py"],
        required_tests=["tests/architecture/perception/test_observation_state_ontology.py"],
    )
    claim.validate()


def test_environment_specific_logic_not_in_core_kernel_claim():
    with pytest.raises(ValueError):
        GeneralizationClaim(
            feature_name="nethack_only",
            general_primitive="",
            primary_environment="terminal_grid:nethack",
            secondary_environments=[],
            environment_specific_code_paths=[],
            shared_code_paths=[],
            required_tests=[],
        ).validate()
