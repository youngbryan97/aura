"""The one predictor everything is scored against.

Half the battery is a comparison of two fits: with a channel and without it,
with the internal state and without it, whole against cut. A comparison like
that is only worth anything when both sides get the same model class, the same
regularisation and the same held-out rows, because otherwise the winner is
whichever side was allowed more freedom.

So there is one estimator here and everything uses it. Ridge regression from a
standardised feature block to a standardised target block, with the ridge
strength chosen on a validation split rather than set by hand, scored as
normalised squared error on rows the fit never saw. A model that has learned
nothing scores 1.0, which is what predicting the training mean gets you, and
that ceiling is what makes a loss ratio readable.

The ridge strength is chosen per target column, not once for the block. One
alpha for forty targets is one compromise for forty different problems, and it
falls hardest on the model with the most inputs: a wide model that needs heavy
shrinkage on three of its targets and none on the rest gets neither. That
asymmetry is enough to make a cut model beat the intact one out of sample,
which reads as "cutting helps" and means the comparison was not fair. Ten
alphas, chosen column by column on the validation rows.

And when one model is a restriction of another — the cut model is the intact
one with the cross-block weights held at zero — the wider model gets a second
penalty of its own for the columns the narrower one does not have. Without it
a target that its own block already predicts almost perfectly is made worse by
every extra input, however well regularised, because one shared penalty cannot
shrink the additions without also shrinking what was working: deliberation's
change is predicted at a loss of 0.06 from deliberation alone and 0.17 once
thirty-six more columns are offered. With a separate penalty the wide model can
reproduce the narrow one exactly by sending that penalty to infinity, so a
negative result is no longer available to it and a positive one is information.

Two guards matter more than the arithmetic. A column that never moves is
dropped rather than standardised, because dividing by its spread invents
structure out of float noise. And the split is contiguous in time, never
shuffled: rows next to each other in a trajectory are nearly the same row, and
a shuffled split lets the model see the answer sitting beside the question.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Fit", "fit_predict", "held_out_loss", "split_rows"]

#: Ridge strengths tried. The top of this grid matters as much as the bottom.
#: A cut model is a restriction of the intact model — the same fit with the
#: cross-block weights held at zero — so the intact model should never lose to
#: it out of sample. It can anyway, if the grid does not reach far enough for
#: the wide model to shrink its extra inputs away, and then the irreducibility
#: score comes out negative, which reads as "cutting helps" and means "the wide
#: model was not allowed to regularise". Ten decades, so it is.
ALPHAS: tuple[float, ...] = (
    1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6,
)

_FLAT = 1e-9


@dataclass(frozen=True, slots=True)
class Fit:
    """What a fit cost on rows it never saw."""

    loss: float
    #: Median of the per-column ridge strengths chosen, for the record. The
    #: fit itself used one per target column.
    alpha: float
    n_train: int
    n_test: int
    width_in: int
    width_out: int
    degenerate: bool = False
    #: Raw test error and the error of predicting the training mean, kept so
    #: several fits over different target blocks can be added up exactly
    #: rather than averaged as ratios, which would weight a one-column block
    #: the same as a forty-column one.
    sse: float = 0.0
    base: float = 0.0

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "loss": round(self.loss, 6),
            "alpha": self.alpha,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "width_in": self.width_in,
            "width_out": self.width_out,
            "degenerate": self.degenerate,
        }


def split_rows(n: int, *, train: float = 0.6, validate: float = 0.2) -> tuple[slice, slice, slice]:
    """Contiguous train / validate / test, in time order."""
    a = int(n * train)
    b = int(n * (train + validate))
    return slice(0, a), slice(a, b), slice(b, n)


def _standardise(block: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centre and scale by the reference rows only, so test rows stay unseen."""
    centre = reference.mean(axis=0)
    spread = reference.std(axis=0)
    keep = spread > _FLAT
    if not keep.any():
        return np.zeros((block.shape[0], 0)), keep
    return (block[:, keep] - centre[keep]) / spread[keep], keep


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float | np.ndarray) -> np.ndarray:
    n_features = x.shape[1]
    penalty = np.full(n_features, float(alpha)) if np.isscalar(alpha) else np.asarray(alpha)
    gram = x.T @ x + np.diag(penalty)
    try:
        return np.linalg.solve(gram, x.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ (x.T @ y)


def _penalties(width: int, own_width: int | None) -> list[tuple[float, float, np.ndarray]]:
    """Every penalty vector to try, with the two strengths that made it."""
    if own_width is None or own_width >= width:
        return [(alpha, alpha, np.full(width, alpha)) for alpha in ALPHAS]
    out: list[tuple[float, float, np.ndarray]] = []
    for own in ALPHAS:
        for extra in ALPHAS:
            vector = np.empty(width)
            vector[:own_width] = own
            vector[own_width:] = extra
            out.append((own, extra, vector))
    return out


def fit_predict(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    train: slice,
    validate: slice,
    test: slice,
    own_width: int | None = None,
) -> Fit:
    """Ridge from features to targets, alpha picked on validate, scored on test.

    The loss is squared error on the test rows divided by the error of
    predicting the training mean. 1.0 means the features carried nothing;
    below 1.0 is how much they carried.

    ``own_width`` marks how many leading columns belong to the narrower model
    this one is being compared against. Those get one penalty and the rest get
    another, both searched, so the wide model can shrink its additions away and
    reproduce the narrow model rather than being beaten by it.
    """
    reference = features[train]
    if reference.shape[0] < 4 or targets[train].shape[0] < 4:
        return Fit(1.0, 0.0, reference.shape[0], targets[test].shape[0], 0, targets.shape[1], True)

    x_all, keep_x = _standardise(features, reference)
    y_reference = targets[train]
    y_centre = y_reference.mean(axis=0)
    y_spread = y_reference.std(axis=0)
    keep_y = y_spread > _FLAT
    if not keep_y.any() or x_all.shape[1] == 0:
        return Fit(
            1.0,
            0.0,
            reference.shape[0],
            targets[test].shape[0],
            int(keep_x.sum()),
            int(keep_y.sum()),
            True,
        )
    y_all = (targets[:, keep_y] - y_centre[keep_y]) / y_spread[keep_y]

    x_train, y_train = x_all[train], y_all[train]
    x_validate, y_validate = x_all[validate], y_all[validate]
    x_test, y_test = x_all[test], y_all[test]

    width_out = y_all.shape[1]
    # `own_width` counts columns of the caller's block; standardising may have
    # dropped some, so it is remapped through the same mask.
    kept_own = int(keep_x[:own_width].sum()) if own_width is not None else None
    grid = _penalties(x_all.shape[1], kept_own)
    scores = np.full((len(grid), width_out), np.inf)
    for index, (_own, _extra, penalty) in enumerate(grid):
        weights = _ridge(x_train, y_train, penalty)
        if x_validate.size:
            scores[index] = ((x_validate @ weights - y_validate) ** 2).mean(axis=0)
    chosen = scores.argmin(axis=0) if x_validate.size else np.zeros(width_out, dtype=int)

    # Refit on train + validate once the alphas are chosen: the validation rows
    # are data, and holding them out of the final fit throws away a fifth of
    # the trajectory for no benefit once they have done their job.
    x_full = np.vstack([x_train, x_validate])
    y_full = np.vstack([y_train, y_validate])
    predicted = np.zeros_like(y_test)
    for index, (_own, _extra, penalty) in enumerate(grid):
        columns = np.flatnonzero(chosen == index)
        if columns.size == 0:
            continue
        weights = _ridge(x_full, y_full[:, columns], penalty)
        if x_test.size:
            predicted[:, columns] = x_test @ weights

    residual = float(np.sum((predicted - y_test) ** 2)) if x_test.size else 0.0
    baseline = float(np.sum(y_test**2)) if y_test.size else 0.0
    loss = residual / baseline if baseline > _FLAT else 1.0
    best_alpha = float(np.median([grid[i][0] for i in chosen])) if width_out else 0.0
    return Fit(
        loss=float(loss),
        alpha=best_alpha,
        n_train=int(x_full.shape[0]),
        n_test=int(x_test.shape[0]),
        width_in=int(x_all.shape[1]),
        width_out=int(y_all.shape[1]),
        sse=residual,
        base=baseline,
    )


def held_out_loss(features: np.ndarray, targets: np.ndarray) -> Fit:
    """The common case: one trajectory, split in time, scored once."""
    train, validate, test = split_rows(features.shape[0])
    return fit_predict(features, targets, train=train, validate=validate, test=test)
