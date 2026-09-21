"""ISC-v4's irreducibility reading, and what it must not do.

v3 divides the cut's cost by everything the target does, including the part
nothing can predict, so the score falls as she is recorded more widely. v4
divides by what the intact model can explain at all. The correction has existed
since 16 September behind a parameter that defaults to off and that nothing
outside its own tests has ever passed. See docs/ISC_V4_PREREGISTRATION.md.
"""

from __future__ import annotations

import numpy as np

from core.subject.irreducibility import phi_do
from core.subject.nulls import architecture, toy_recording
from core.subject.recording import Recording


def _reference(seed: int = 3, steps: int = 1200) -> Recording:
    return toy_recording(architecture("recurrent", seed=seed), steps=steps, seed=seed)


def _shuffled(recording: Recording, seed: int = 11) -> Recording:
    order = np.random.default_rng(seed).permutation(recording.x.shape[0])
    return Recording(
        x=recording.x[order],
        conditions=recording.conditions,
        tags=recording.tags,
        times=recording.times,
        env=recording.env,
        env_names=recording.env_names,
        columns=recording.columns,
        slices=recording.slices,
        notes=dict(recording.notes),
    )


def test_the_two_versions_read_the_same_cuts_on_the_same_recording():
    """Only the denominator moves. A v3 number and a v4 number from one run
    differ in that and in nothing else."""
    recording = _reference()
    three = phi_do(recording)
    four = phi_do(recording, explained_share=True)
    assert three.best_cut == four.best_cut
    assert sorted(three.scores) == sorted(four.scores)


def test_v4_is_not_uniformly_easier_and_the_reference_pays_for_it():
    """v4 exceeds v3 only when what nothing can predict is large beside what
    the intact model explains. The reference is well predicted, so it reads
    lower under v4 — a change that helped whatever it was pointed at would
    not do that."""
    recording = _reference()
    three = phi_do(recording)
    four = phi_do(recording, explained_share=True)
    assert three.loss_full == four.loss_full
    assert three.loss_cut == four.loss_cut
    assert four.phi < three.phi, (three.phi, four.phi)
    # And it still clears the line it exists to demonstrate.
    assert four.phi > 0.05, four.phi


def test_destroying_the_structure_is_not_repaired_by_the_denominator():
    """The dilution v4 removes is not signal, so removing it cannot make any."""
    shuffled = _shuffled(_reference())
    assert phi_do(shuffled, explained_share=True).phi <= 0.01


def test_the_default_is_off_so_a_run_that_does_not_declare_v4_is_unaffected():
    recording = _reference()
    assert phi_do(recording).phi == phi_do(recording, explained_share=False).phi


def test_a_run_reports_the_v4_reading_beside_the_v3_one():
    """A version nothing computes is a version nothing can be judged under."""
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "tools" / "run_subject_core.py"
    text = source.read_text()
    assert 'evidence["phi_v4"]' in text
    assert "explained_share=True" in text
