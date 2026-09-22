"""The schema is part of the instrument, and it had been drifting.

Every domain is reduced to four principal components, and `_basis` standardises
each column to unit variance before the decomposition — so a channel that
barely moves is scaled up to the same variance as one that carries the domain.
Between 45a74c913 and e22e89192 the schema grew from 210 columns to 315 as
organ readings were added to it.

That bears on everything read through the reduction. It does not bear on the
interventional edges: `_paired_divergence` takes a maximum over the domain's
columns and reads effect and floor off the same one, so another column can only
raise it. The last test here holds that, because a wrong mechanism recorded as
a right one is worse than no mechanism at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.estimate import fit_predict, split_rows
from core.subject.irreducibility import COMPONENTS
from core.subject.measured_columns import MEASURED_COLUMNS, PINNED_AT
from core.subject.state import DOMAINS, feature_names

ROWS = 1200


def test_the_instrument_is_the_commit_the_peak_was_read_on():
    assert PINNED_AT.startswith("45a74c913")
    assert len(MEASURED_COLUMNS) == 210


def test_every_pinned_column_still_exists_in_the_schema():
    """A pinned column the schema no longer has would make the instrument
    unbuildable, which must fail loudly rather than quietly shrink."""
    assert set(MEASURED_COLUMNS) <= set(feature_names())


def test_the_pinned_set_is_in_schema_order():
    order = [name for name in feature_names() if name in set(MEASURED_COLUMNS)]
    assert order == list(MEASURED_COLUMNS)


def test_every_domain_keeps_columns():
    """A domain reduced to nothing is a domain the battery cannot measure."""
    for domain in DOMAINS:
        assert any(name.startswith(f"{domain}.") for name in MEASURED_COLUMNS), domain


def _basis(block: np.ndarray, train: slice, k: int):
    reference = block[train]
    spread = reference.std(axis=0)
    keep = spread > 1e-9
    centre = reference[:, keep].mean(axis=0)
    scale = spread[keep]
    if int(keep.sum()) <= k:
        return lambda values: (values[:, keep] - centre) / scale
    scaled = (reference[:, keep] - centre) / scale
    _, _, vectors = np.linalg.svd(scaled, full_matrices=False)
    directions = vectors[:k].T
    return lambda values: ((values[:, keep] - centre) / scale) @ directions


def _explained(extra: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(ROWS, 4))
    target = np.zeros((ROWS, 4))
    for step in range(1, ROWS):
        target[step] = (
            0.6 * target[step - 1] + 0.8 * source[step - 1]
            + 0.3 * rng.normal(size=4)
        )
    block = source
    if extra:
        block = np.hstack([source, 0.02 * rng.normal(size=(ROWS, extra))])
    train, validate, test = split_rows(ROWS - 1)
    reduced = _basis(block, train, COMPONENTS)(block)
    fit = fit_predict(
        reduced[:-1], np.diff(target, axis=0),
        train=train, validate=validate, test=test,
    )
    return 1.0 - fit.loss


def test_weak_columns_added_to_a_domain_destroy_a_real_effect():
    """The reason the set is pinned, as a measurement rather than a story.

    Same estimator, same four components, same causal relationship. Only the
    number of standardised channels those four directions have to cover
    changes.
    """
    plain = float(np.mean([_explained(0, seed) for seed in (3, 7, 11)]))
    diluted = float(np.mean([_explained(32, seed) for seed in (3, 7, 11)]))
    assert plain > 0.5, plain
    assert diluted < plain / 4.0, (plain, diluted)


def test_an_edge_is_a_maximum_over_columns_and_cannot_be_diluted():
    """The claim the first version of this got wrong.

    An edge effect is not a fit. Each column's peak displacement is taken over
    the lags, the column with the largest margin over its own sham floor is
    chosen, and the effect is read off that column — so adding a column can
    only raise the maximum, never lower it.
    """
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "core" / "subject" / "causal.py"
    ).read_text()
    tree = ast.parse(source)
    paired = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_paired_divergence"
    )
    body = ast.unparse(paired)
    assert "argmax(column_effect - column_floor)" in body
    assert "column_effect = moved.max(axis=0)" in body


def test_more_added_columns_never_read_as_more_signal():
    """Monotone, so this is a property of the reduction and not one draw."""
    scores = [
        float(np.mean([_explained(extra, seed) for seed in (3, 7, 11)]))
        for extra in (0, 8, 24)
    ]
    assert scores == sorted(scores, reverse=True), scores


def test_the_reduction_reads_the_instrument_and_the_frames_keep_their_width():
    """Where the pin belongs, and where it must not be.

    The interventional path reads `CoreState.domain()` directly, so the raw
    frames and the recording have to agree in width — filtering
    `build_recording` made `_paired_divergence` index a 17-wide domain with a
    12-wide mask and every run died there. The reduction is where the width
    matters and where the pin now is.
    """
    import inspect

    from core.subject.irreducibility import _measured_block
    from core.subject.recording import build_recording

    assert "full" not in (build_recording.__kwdefaults__ or {})
    assert "instrument" not in inspect.signature(build_recording).parameters
    assert len(MEASURED_COLUMNS) < len(feature_names())
    assert callable(_measured_block)


def test_a_domain_the_instrument_does_not_name_is_read_whole():
    """An empty block is not a measurement."""
    from core.subject.irreducibility import _measured_block

    class _Recording:
        columns = ("Z.one", "Z.two")

        def __init__(self) -> None:
            self.slices = {"Z": slice(0, 2)}

    import core.subject.irreducibility as module

    keep = module._block_columns
    module._block_columns = lambda recording, keys: [0, 1]
    try:
        assert _measured_block(_Recording(), "Z") == [0, 1]
    finally:
        module._block_columns = keep
