import pytest

from core.environment.command import ActionIntent, CommandSpec, CommandStep
from core.environments.terminal_grid import NetHackCommandCompiler


def test_no_raw_action_string_executes_through_compiler():
    compiler = NetHackCommandCompiler()
    with pytest.raises(AttributeError):
        compiler.compile("pray")  # type: ignore[arg-type]


def test_multistep_extended_command_and_modal_contract():
    compiler = NetHackCommandCompiler()
    command = compiler.compile(ActionIntent(name="pray", expected_effect="prayer_attempted"))
    assert [step.value for step in command.steps] == ["#", "pray\n"]
    assert command.preconditions
    assert command.expected_effects


def test_irreversible_command_requires_authority_flag():
    spec = CommandSpec(
        command_id="c",
        environment_id="env",
        intent=ActionIntent(name="submit", risk="irreversible", requires_authority=False),
        preconditions=[],
        steps=[CommandStep("api", "submit")],
        expected_effects=["submitted"],
    )
    with pytest.raises(ValueError):
        spec.validate()


def test_unknown_intent_rejected():
    with pytest.raises(ValueError):
        NetHackCommandCompiler().compile(ActionIntent(name="raw prose"))
