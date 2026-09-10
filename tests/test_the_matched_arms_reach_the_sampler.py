"""An ablation arm is worth running only if its lesion moves something.

The matched protocol generates by calling the model directly, and the first
version of it wired one channel to that call: the affect circumplex, which
sets the temperature. Three more were declared, lesionable, and unreachable —
not because the lesion did not work but because nothing in the harness had
produced the frames they gate, so cutting them changed a value nobody had
computed.

The three advisory passes are plain functions over an ``AuraState`` and a
prompt. Running them here gives those channels something to carry, and the
arm that removes endogenous state then removes four things instead of one.

What is held: the lesion does exactly one thing, and the harness folds the
biases with the same classmethod the live phase folds them with rather than
with arithmetic written beside it.
"""

from __future__ import annotations

import importlib.util
import sys
from contextlib import ExitStack
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "matched_substrate_under_test", ROOT / "tools" / "matched" / "run_matched_substrate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._wake_the_faculties()
    return module


@pytest.fixture(scope="module")
def frames(runner):
    return runner._what_the_faculties_say("What is 12 times 13?")


def test_the_advisory_passes_produce_the_frames_their_channels_gate(frames):
    """Without these there is nothing for the three biases to carry."""
    carried = [
        frames.get(key, {}).get("sampling_bias")
        for key in ("spiking_active_inference", "imagination_workspace", "bicameral_advisory")
    ]
    assert sum(1 for one in carried if isinstance(one, dict) and one) >= 2, (
        f"the advisories produced no sampling bias: {carried}"
    )


def test_the_endogenous_arm_removes_every_channel_that_reaches_the_sampler(runner):
    removed = set(runner.WHAT_EACH_ARM_REMOVES[runner.NO_ENDOGENOUS])
    assert "affect.circumplex_sampling" in removed
    assert "spiking.sampling_bias" in removed
    assert "imagination.sampling_bias" in removed
    assert "bicameral.sampling_bias" in removed


def test_every_channel_the_arm_removes_is_registered(runner):
    """The defect this file exists after: an arm that lesions nothing."""
    from core.verify.lesion_registry import get_lesion_registry

    known = set(get_lesion_registry().channels())
    for arm, channels in runner.WHAT_EACH_ARM_REMOVES.items():
        for channel in channels:
            assert channel in known, f"{arm} removes {channel}, which nothing registers"


def test_the_lesion_puts_the_sampler_at_the_neutral_and_nothing_else_does(runner, frames):
    from core.verify.lesion_registry import get_lesion_registry

    intact = runner._temperature_under(runner.INTACT, frames)
    registry = get_lesion_registry()
    with ExitStack() as removed:
        for channel in runner.WHAT_EACH_ARM_REMOVES[runner.NO_ENDOGENOUS]:
            removed.enter_context(registry.lesion(channel))
        lesioned = runner._temperature_under(runner.NO_ENDOGENOUS, frames)

    assert lesioned == pytest.approx(runner.NEUTRAL_TEMPERATURE), (
        "removing every channel that reaches the sampler must leave the neutral"
    )
    assert intact != pytest.approx(runner.NEUTRAL_TEMPERATURE), (
        "the intact arm sampled at the neutral, so the arm measures nothing"
    )


def test_the_biases_are_counted_so_a_zero_delta_says_which_kind_it_is(runner, frames):
    """A temperature can land on the neutral by arithmetic. A count cannot."""
    from core.verify.lesion_registry import get_lesion_registry

    runner._BIASES_READ.clear()
    runner._temperature_under(runner.INTACT, frames)
    registry = get_lesion_registry()
    with ExitStack() as removed:
        for channel in ("spiking.sampling_bias", "imagination.sampling_bias", "bicameral.sampling_bias"):
            removed.enter_context(registry.lesion(channel))
        runner._temperature_under(runner.NO_ENDOGENOUS, frames)

    assert runner._BIASES_READ[runner.INTACT][-1] >= 2
    assert runner._BIASES_READ[runner.NO_ENDOGENOUS][-1] == 0


def test_the_fold_is_the_one_the_live_phase_uses(runner):
    """Reimplementing the clamp would agree until one of them changed."""
    import inspect

    source = inspect.getsource(runner._temperature_under)
    assert "_apply_generation_sampling_bias" in source
    from core.phases.response_generation import ResponseGenerationPhase

    assert callable(ResponseGenerationPhase._apply_generation_sampling_bias)


def test_the_token_budget_is_not_moved_by_a_bias(runner, frames):
    """Tokens are a budget dimension; parity across arms is the point."""
    import inspect

    source = inspect.getsource(runner._temperature_under)
    assert "_tokens" in source, "the fold's token budget must be discarded explicitly"
    assert "MAX_TOKENS" in source
