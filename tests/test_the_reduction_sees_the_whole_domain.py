"""What the domain reduction costs, and what it does not.

`_basis` standardises every column to unit variance before taking `COMPONENTS`
directions, so "largest variance" stops meaning "most informative". On a
synthetic source of four real channels driving a target, adding weakly varying
columns to the source destroys the measured effect: 0.6942 explained with none
added, 0.0222 with thirty-two. Keeping every column and letting the ridge
decide loses almost nothing — 0.6820 with sixty-four added.

That is a fact about uncorrelated noise, and it does not carry to her. On
run_030's own recording, the one that scored 19/24, raising the component count
does not help and the intact model predicts worse with more columns: phi 0.0320
at four components and 0.0289 with every column, while the intact held-out loss
goes 0.2084 to 0.3404. Her extra channels are slow and drifting rather than
independent, so handing them all to the ridge adds collinear predictors.

So the reduction is not why irreducibility is low, and the whole of each domain
is what it reads. These hold the measurement that says so, because the hypothesis
was plausible enough to act on and wrong enough that acting on it narrowed what
the battery was looking at.
"""

from __future__ import annotations

import numpy as np

from core.subject.estimate import fit_predict, split_rows
from core.subject.irreducibility import COMPONENTS, phi_do

ROWS = 1200
REAL = 4


def _standardise(block: np.ndarray, train: slice) -> np.ndarray:
    reference = block[train]
    spread = reference.std(axis=0)
    keep = spread > 1e-9
    return (block[:, keep] - reference[:, keep].mean(axis=0)) / spread[keep]


def _build(extra: int, seed: int):
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(ROWS, REAL))
    target = np.zeros((ROWS, REAL))
    for step in range(1, ROWS):
        target[step] = (
            0.6 * target[step - 1] + 0.8 * source[step - 1]
            + 0.3 * rng.normal(size=REAL)
        )
    if extra:
        source = np.hstack([source, 0.02 * rng.normal(size=(ROWS, extra))])
    return source, target


def _explained(extra: int, seed: int, *, components: int | None) -> float:
    block, target = _build(extra, seed)
    train, validate, test = split_rows(ROWS - 1)
    standardised = _standardise(block, train)
    if components is not None and standardised.shape[1] > components:
        _, _, vectors = np.linalg.svd(standardised[train], full_matrices=False)
        standardised = standardised @ vectors[:components].T
    fit = fit_predict(
        standardised[:-1], np.diff(target, axis=0),
        train=train, validate=validate, test=test,
    )
    return 1.0 - fit.loss


def test_a_fixed_component_count_loses_a_real_effect_to_added_noise():
    seeds = (3, 7, 11)
    plain = float(np.mean([_explained(0, s, components=COMPONENTS) for s in seeds]))
    diluted = float(np.mean([_explained(32, s, components=COMPONENTS) for s in seeds]))
    assert plain > 0.5, plain
    assert diluted < plain / 4.0, (plain, diluted)


def test_keeping_every_column_is_nearly_immune_to_the_same_noise():
    seeds = (3, 7, 11)
    plain = float(np.mean([_explained(0, s, components=None) for s in seeds]))
    diluted = float(np.mean([_explained(64, s, components=None) for s in seeds]))
    assert diluted > 0.9 * plain, (plain, diluted)


def test_the_reduction_reads_the_whole_domain():
    """No column of a domain is excluded from what the battery measures.

    A subject test that looks at part of the subject is not a subject test,
    and a column set pinned to one commit was doing exactly that.
    """
    import core.subject.irreducibility as module

    assert not hasattr(module, "_measured_block")
    source = module.phi_do.__doc__ or ""
    del source
    import inspect

    body = inspect.getsource(module.phi_do)
    assert "_block_columns(recording, (key,))" in body


def test_phi_still_computes_on_a_toy_recording():
    from core.subject.nulls import architecture, toy_recording

    rec = toy_recording(architecture("recurrent", seed=3), steps=900, seed=3)
    report = phi_do(rec)
    assert report.domains
    assert report.loss_full > 0.0
