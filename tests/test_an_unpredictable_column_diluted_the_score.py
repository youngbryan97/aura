"""A column nothing can predict shrank the irreducibility of everything else.

`phi_do` is a ratio: the loss a cut costs, over the loss the cut model already
carries. A column no model can explain adds its variance to both, so adding
four such columns to each domain of the recurrent reference took its score from
0.062 to 0.005. The system was unchanged and thirteen fourteenths of its
irreducibility had gone.

Two earlier corrections were tried and both failed — one did nothing, and one
read +0.021 on a recording whose rows had been shuffled, where the answer is
zero. Both chose the target directions by what predicts them, and a basis
chosen for predictability finds the drift in a random walk.

A third was tried here and fell the same way. Predicting each domain's columns
directly instead of its four components does recover the numerator — the gap
between the cut model's error and the intact model's went from 0.002 back to
0.009 under added noise — and it read +0.007 on shuffled rows. It is gone.

What survives is the denominator. The cut's cost is read against what the
intact model can explain at all rather than against everything the target does,
because the unexplainable part cancels in that ratio and not in the published
one. It reads −0.005 on shuffled rows and exactly zero on a random walk, where
the intact model explains nothing and there is no transition law to damage.

It is off by default. Changing the published estimator is a preregistration
amendment, not a flag flip.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from core.subject.irreducibility import phi_do
from core.subject.recording import Recording

DOMAINS_UNDER_TEST = ("P", "I", "A")


def _coupled(frames: int = 1200, width: int = 3, seed: int = 7) -> Recording:
    """Three domains that drive each other, with their own noise."""
    rng = np.random.default_rng(seed)
    state = {key: rng.normal(size=width) for key in DOMAINS_UNDER_TEST}
    rows = []
    order = list(DOMAINS_UNDER_TEST)
    for _ in range(frames):
        nxt = {}
        for index, key in enumerate(order):
            other = state[order[(index + 1) % len(order)]]
            nxt[key] = 0.55 * state[key] + 0.35 * other + rng.normal(scale=0.1, size=width)
        state = nxt
        rows.append(np.concatenate([state[key] for key in order]))
    return _shape(np.vstack(rows), width)


def _shape(x: np.ndarray, width: int) -> Recording:
    slices, columns, start = {}, [], 0
    for key in DOMAINS_UNDER_TEST:
        slices[key] = slice(start, start + width)
        columns.extend(f"{key}.{i}" for i in range(width))
        start += width
    frames = x.shape[0]
    return Recording(
        x=x,
        conditions=tuple("toy" for _ in range(frames)),
        tags=tuple("" for _ in range(frames)),
        times=np.arange(frames, dtype=np.float64),
        env=np.zeros((frames, 1)),
        env_names=("clock",),
        columns=tuple(columns),
        slices=slices,
        notes={},
    )


def _with_noise(rec: Recording, per_domain: int, seed: int) -> Recording:
    rng = np.random.default_rng(seed)
    blocks, slices, columns, start = [], {}, [], 0
    for key in DOMAINS_UNDER_TEST:
        block = rec.x[:, rec.slices[key]]
        wide = np.hstack([block, rng.normal(size=(rec.x.shape[0], per_domain))])
        blocks.append(wide)
        slices[key] = slice(start, start + wide.shape[1])
        start += wide.shape[1]
        columns.extend(f"{key}.{i}" for i in range(wide.shape[1]))
    return dataclasses.replace(
        rec, x=np.hstack(blocks), slices=slices, columns=tuple(columns)
    )


def _phi(rec: Recording, **kw: object) -> float:
    return float(phi_do(rec, domains=DOMAINS_UNDER_TEST, **kw).phi)


def test_the_published_estimator_is_diluted() -> None:
    """The finding itself, held so it stays a fact rather than a memory."""
    clean = _phi(_coupled())
    diluted = _phi(_with_noise(_coupled(), 4, 11))
    assert clean > 0.02, clean
    assert diluted < clean / 2.0, (clean, diluted)


def test_reading_the_cost_against_what_is_explainable_lifts_the_score() -> None:
    clean = _coupled()
    noisy = _with_noise(clean, 4, 11)
    published = _phi(noisy)
    corrected = _phi(noisy, explained_share=True)
    assert corrected > published, (published, corrected)


def test_the_correction_reads_nothing_on_shuffled_rows() -> None:
    rec = _with_noise(_coupled(), 4, 11)
    rng = np.random.default_rng(3)
    shuffled = dataclasses.replace(rec, x=rec.x[rng.permutation(rec.x.shape[0])])
    assert _phi(shuffled, explained_share=True) <= 0.01


def test_the_correction_reads_nothing_on_a_random_walk() -> None:
    """The trap the second earlier correction fell into."""
    rec = _coupled()
    rng = np.random.default_rng(5)
    walk = dataclasses.replace(
        rec, x=np.cumsum(rng.normal(scale=0.1, size=rec.x.shape), axis=0)
    )
    assert _phi(walk, explained_share=True) <= 0.01


def test_a_model_that_explains_nothing_is_not_a_denominator() -> None:
    """Two vanishing numbers over each other is not a ratio."""
    rng = np.random.default_rng(9)
    noise = _shape(rng.normal(size=(1200, 9)), 3)
    assert _phi(noise, explained_share=True) <= 0.01


def test_the_published_reading_is_unchanged_by_default() -> None:
    rec = _coupled()
    assert _phi(rec) == _phi(rec, explained_share=False)
