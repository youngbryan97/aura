from core.environment.ablation import ABLATION_MODES, AblationConfig
from core.environment.benchmark import ProofPack
from core.environment.curriculum import CurriculumEngine, CurriculumTask


def test_curriculum_task_schema_and_stage_gate():
    task = CurriculumTask(
        task_id="observe",
        environment_family="terminal_grid",
        difficulty=0,
        objective="parse fixtures",
        allowed_capabilities={"observe"},
        success_metrics={"parse_accuracy": 0.95},
        failure_metrics={"crash_rate": 0.0},
    )
    assert task.success_metrics
    assert CurriculumEngine().next_stage_allowed(0, 1)
    assert not CurriculumEngine().next_stage_allowed(0, 3)


def test_proof_pack_schema_multiple_environment_families():
    pack = ProofPack.load("benchmarks/proof_packs/embodied_general_v1.yaml")
    families = {env["id"].split(":")[0] for env in pack.environments}
    assert len(families) >= 3
    assert pack.shared_requirements["trace_replay"]
    assert pack.shared_requirements["receipts"]
    assert pack.shared_requirements["ablations"]
    assert pack.shared_requirements["holdout_tasks"]


def test_ablation_config_schema_and_dry_run_gate():
    assert "prompt_only" in ABLATION_MODES
    config = AblationConfig("no_action_gateway_dry_run_only")
    assert config.dry_run_only
