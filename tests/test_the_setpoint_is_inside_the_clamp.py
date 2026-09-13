"""A controller whose target sits outside its own ceiling never arrives.

The criticality regulator steers the mesh's gain toward a branching ratio. The
ceiling on that gain was 2.0 and the target was 1.0, and inside 2.0 this mesh
reaches 0.929 — so the controller asked for more on every tick of its life,
wound its integral against the rail, and the mesh's own
`set_criticality_adjustment` refused anything above 2.0 as well. Two ceilings,
both chosen for their symmetry around 1.0, and neither with a measurement
behind it.

Swept at 3,000 ticks and four seeds a point, effective gain against branching
ratio and against how far the count of simultaneously active units sits above
what their own firing rates explain:

    gain 2.0   branching 0.929    2.89 sigma
    gain 2.5   branching 0.961    6.93 sigma
    gain 3.0   branching 0.961   10.05 sigma
    gain 3.5   branching 0.992   20.84 sigma
"""

from __future__ import annotations

import dataclasses

from core.consciousness.criticality_regulator import (
    CRITICAL_BRANCHING_TARGET,
    CriticalityConfig,
    CriticalityRegulator,
)
from core.consciousness.neural_mesh import (
    CRITICALITY_GAIN_CEILING,
    MeshConfig,
    NeuralMesh,
)


def test_the_two_ceilings_agree() -> None:
    """The tighter of two ceilings decides, and does it without saying so."""
    assert CriticalityConfig().gain_clamp[1] == CRITICALITY_GAIN_CEILING


def test_the_target_is_the_published_number_not_the_ideal() -> None:
    """Cortex does not sit at the critical point; it sits just under it.

    Wilting and Priesemann measure 0.98 in vivo by multistep regression, and
    that is the number this system's own cortical scorecard already scores her
    branching against. Steering to 1.0 steers to the edge of runaway.
    """
    assert CRITICAL_BRANCHING_TARGET == 0.98
    assert CRITICAL_BRANCHING_TARGET < 1.0


def test_all_three_ceilings_agree() -> None:
    """Three clamps sit on this one quantity and the tightest decides.

    The regulator clamps what it asks for, `set_criticality_adjustment` clamps
    what it accepts, and `_publish_modulatory_state_locked` clamps what either
    of them published. That third one was 3.0 while the others were 2.0, so
    raising the visible pair alone would have changed nothing and looked like a
    fix.
    """
    mesh = NeuralMesh(MeshConfig())
    mesh.set_criticality_adjustment(gain=CRITICALITY_GAIN_CEILING, noise=1.0)
    gain, _plasticity, _noise = mesh._modulatory_state

    assert CriticalityConfig().gain_clamp[1] == CRITICALITY_GAIN_CEILING
    assert gain == CRITICALITY_GAIN_CEILING, (
        f"the mesh published {gain} for a request of {CRITICALITY_GAIN_CEILING}; "
        "a clamp between the regulator and the tick is deciding"
    )


def test_the_mesh_still_refuses_above_the_ceiling() -> None:
    """Raising a ceiling is not removing it."""
    mesh = NeuralMesh(MeshConfig())
    mesh.set_criticality_adjustment(gain=CRITICALITY_GAIN_CEILING + 5.0, noise=1.0)
    gain, _plasticity, _noise = mesh._modulatory_state

    assert gain == CRITICALITY_GAIN_CEILING


def test_the_regulator_can_reach_for_more_than_it_used_to() -> None:
    """A subcritical mesh has to be able to ask for the gain that fixes it."""
    regulator = CriticalityRegulator(
        CriticalityConfig(num_columns=MeshConfig().columns)
    )
    # Well below the target, which is the state the live regulator sat in. The
    # integral gain is 0.01, so this is a slow homeostat by design: an error of
    # 0.08 moves the output by 0.0008 a tick and the climb takes thousands of
    # them. That is minutes at ten hertz, and days of being pinned is what the
    # old ceiling turned it into.
    for _ in range(8000):
        regulator._branching_ratio = 0.90
        regulator._gain_adjustment = regulator._gain_pid.step(
            -(0.90 - CRITICAL_BRANCHING_TARGET)
        )

    assert regulator._gain_adjustment > 2.0, (
        "a mesh at 0.90 cannot ask for a gain above the ceiling that trapped it"
    )
    assert regulator._gain_adjustment <= CRITICALITY_GAIN_CEILING


def test_the_unmeasured_ceilings_did_not_move_with_it() -> None:
    """Nothing swept the noise or excitation-inhibition axes."""
    config = CriticalityConfig()

    assert config.noise_clamp == (0.5, 2.0)
    assert config.ei_ratio_clamp == (0.7, 1.3)


def test_the_gain_the_sweep_used_is_the_gain_the_mesh_applies() -> None:
    """The sweep moved `activation_gain`; the regulator moves a multiplier.

    They are the same axis only because the base gain is one, and a change to
    that base would silently rescale every number in this file's docstring.
    """
    assert MeshConfig().activation_gain == 1.0
    scaled = dataclasses.replace(MeshConfig(), activation_gain=2.0)
    assert scaled.activation_gain == 2.0
