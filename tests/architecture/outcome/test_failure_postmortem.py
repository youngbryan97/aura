from core.environment.failure_taxonomy import FAILURE_CLASSES


def test_failure_taxonomy_contains_required_classes():
    for name in [
        "perception_error",
        "belief_error",
        "modal_error",
        "action_compilation_error",
        "gateway_error",
        "authorization_error",
        "execution_error",
        "prediction_error",
        "resource_management_error",
        "planning_error",
        "loop_stagnation",
        "unsafe_irreversible_action",
        "knowledge_gap",
        "learning_error",
        "environment_unavailable",
        "trace_integrity_error",
    ]:
        assert name in FAILURE_CLASSES
