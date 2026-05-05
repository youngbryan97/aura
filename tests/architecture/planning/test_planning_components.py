from core.environment.attention_scheduler import AttentionClaim, AttentionScheduler
from core.environment.episode_manager import EpisodeManager
from core.environment.options import OptionCompiler, OptionLibrary, OptionRun


def test_option_policy_schema_and_failure_suppression():
    library = OptionLibrary()
    option = library.get("RESOLVE_MODAL")
    assert option.initiation_conditions
    assert option.termination_conditions
    intent = OptionCompiler().compile(option)
    assert intent.name == "resolve_modal"
    library.record_run(OptionRun("RESOLVE_MODAL", 1, 2, "failed"), seq=2)
    library.record_run(OptionRun("RESOLVE_MODAL", 3, 4, "failed"), seq=4)
    assert library.suppressed_after_repeated_failure("RESOLVE_MODAL")


def test_attention_scheduler_prioritizes_modal_crisis_over_progress():
    decision = AttentionScheduler().select(
        [
            AttentionClaim("progress", 0.2, 0.2, 0.2, 0.1, 0.8, "go deeper"),
            AttentionClaim("modal", 1.0, 1.0, 0.4, 0.9, 0.9, "resolve prompt"),
        ]
    )
    assert decision.selected_claims[0].source == "modal"


def test_episode_phase_transitions():
    manager = EpisodeManager("run", "env")
    assert manager.state.phase == "boot"
    manager.transition(valid_state=True)
    assert manager.state.phase == "orientation"
    manager.transition(stable=True)
    assert manager.state.phase == "early_exploration"
    manager.transition(critical_risk=True)
    assert manager.state.phase == "crisis"
    manager.transition(stable=True)
    assert manager.state.phase == "recovery"
    manager.transition(terminal=True)
    assert manager.state.phase == "terminal"
    manager.transition()
    assert manager.state.phase == "postmortem"
