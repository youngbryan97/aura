"""Information that exists only in the combination.

Two sources can carry the same thing about a target, or different things, or
something neither carries alone. The last is synergy, and it is the one that
distinguishes a state whose parts are read together from a state whose parts
are read separately and added up.

The decomposition used is the minimum-information one: redundancy is the
smaller of the two single-source informations, which for jointly Gaussian
variables is the standard choice and has the advantage of being computable in
closed form from a covariance matrix rather than estimated from a histogram
nobody has enough samples to fill.

    Syn = I(X1,X2;Y) - I(X1;Y) - I(X2;Y) + min(I(X1;Y), I(X2;Y))

Each domain is reduced to a few principal components first. Forty columns
against a few thousand rows makes a covariance matrix that is nearly singular,
and a determinant near zero can report any amount of information at all.

Each component is then carried to a standard normal by its rank, the Gaussian
copula, so the estimate reads how the columns depend on each other and not the
shape of any one of them. The mutual information is the closed-form Gaussian
one with its sample-size bias taken off analytically, which holds no rows out
and so never scores one stretch of a drifting life against a model of another.

Because a positive number here would be easy to produce by accident, the same
computation runs against a null built by sliding the two sources together in
time against the target. The shift keeps every marginal, every autocorrelation and every
within-source relationship, and destroys only the alignment that synergy is
supposed to be about.

The linear estimator cannot see an interaction, so a second measurement asks
the question a different way: does adding the products of the two sources'
components improve held-out prediction over the two sources side by side? That
one is model-based and coarse, and it is here because it fails differently
from the Gaussian estimate, so agreement between them means more than either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import ndtri, psi
from scipy.stats import rankdata

from core.subject.estimate import fit_predict, split_rows
from core.subject.irreducibility import LOWER_BOUND_Z, MIN_TRANSITIONS, _folds
from core.subject.recording import Recording

__all__ = ["SynergyReport", "synergy", "synergy_suite"]

#: Components kept per domain. Three is enough to carry the shape of a domain
#: and small enough that the joint covariance stays invertible.
COMPONENTS: int = 3

#: Shifts used for the null. Each one is a whole-trajectory circular slide.
#:
#: A quantile is read off this, and the ninety-ninth percentile of two hundred
#: draws is the second-largest of them — an estimate with the shape of a
#: maximum, which moves by more between two runs than the quantity it is the
#: bar for. Each draw is two Gaussian mutual informations over a covariance
#: matrix of nine columns, so a thousand costs milliseconds.
NULL_DRAWS: int = 1000

#: What the mutual information is estimated with. Named so the campaign
#: fingerprint carries it: a different estimator is a different measurement.
ESTIMATOR: str = "gaussian_copula_bias_corrected"


def _components(block: np.ndarray, k: int = COMPONENTS) -> np.ndarray:
    spread = block.std(axis=0)
    keep = spread > 1e-9
    if not keep.any():
        return np.zeros((block.shape[0], 1))
    centred = (block[:, keep] - block[:, keep].mean(axis=0)) / spread[keep]
    if centred.shape[1] <= k:
        return centred
    _, _, vectors = np.linalg.svd(centred, full_matrices=False)
    return centred @ vectors[:k].T


def _plugin_mi(x: np.ndarray, y: np.ndarray, *, ridge: float = 1e-6) -> float:
    """I(X;Y) for jointly Gaussian blocks, from log determinants, in sample."""
    if x.size == 0 or y.size == 0:
        return 0.0
    joint = np.hstack([x, y])
    cxx = np.cov(x, rowvar=False).reshape(x.shape[1], x.shape[1]) + ridge * np.eye(x.shape[1])
    cyy = np.cov(y, rowvar=False).reshape(y.shape[1], y.shape[1]) + ridge * np.eye(y.shape[1])
    cjj = np.cov(joint, rowvar=False) + ridge * np.eye(joint.shape[1])
    sign_x, log_x = np.linalg.slogdet(cxx)
    sign_y, log_y = np.linalg.slogdet(cyy)
    sign_j, log_j = np.linalg.slogdet(cjj)
    if min(sign_x, sign_y, sign_j) <= 0:
        return 0.0
    return max(0.0, 0.5 * float(log_x + log_y - log_j))


def _copula_normal(block: np.ndarray) -> np.ndarray:
    """Each column carried to a standard normal by its rank.

    Mutual information does not change when one variable is stretched by a
    monotone function, and a Gaussian estimate does: a column that saturates
    or jumps reports a different amount of information from the same
    dependence. Ranks remove that. Ties share their average rank, so a flat
    stretch stays flat instead of taking an order from the row index.
    """
    rows = block.shape[0]
    out = np.empty(block.shape, dtype=np.float64)
    for column in range(block.shape[1]):
        ranks = rankdata(block[:, column], method="average")
        out[:, column] = ndtri(ranks / (rows + 1.0))
    return out


def _gaussian_mi(x: np.ndarray, y: np.ndarray, *, ridge: float = 1e-9) -> float:
    """I(X;Y) in nats for Gaussian blocks, with the sample-size bias removed.

    The plug-in estimate is biased upward and the bias grows with the number
    of columns, so it does not cancel in a synergy: the joint over nine
    columns carries about twice the bias of either marginal. On run_019 the
    shifted null scored a synergy fraction of 0.78 against the system's 0.056.

    Cross-fitting took the bias away and put a worse fault in its place. It
    scored each contiguous quarter of the run against a covariance fitted on
    the other three, and the life drifts: in run_023 the first quarter of the
    affect domain scored a held-out log-likelihood ratio of -51 nats, the
    average over folds clamped to zero, the self-model's information survived
    at 1.88, and A+S->G read a synergy of -1.88, which no decomposition can
    produce.

    The bias of a Gaussian log determinant is known exactly. For a sample
    covariance on rows - 1 degrees of freedom, E[log det S] is log det Sigma
    plus width * log(2 / (rows - 1)) plus the sum of digamma((rows - i) / 2).
    Taking half of that offset off each entropy leaves each one unbiased, and
    the information with them, with every row used and none scored against a
    model of a different stretch.
    """
    if x.size == 0 or y.size == 0:
        return 0.0
    rows, width_x, width_y = x.shape[0], x.shape[1], y.shape[1]
    width = width_x + width_y
    if rows <= width + 1:
        # The correction needs more rows than columns, and below that the
        # sample covariance is singular anyway. Nothing can be said.
        return 0.0
    cov = np.cov(np.hstack([x, y]), rowvar=False).reshape(width, width)
    cov = cov + ridge * np.eye(width)
    try:
        joint = float(np.sum(np.log(np.diag(np.linalg.cholesky(cov)))))
        own_x = float(np.sum(np.log(np.diag(np.linalg.cholesky(cov[:width_x, :width_x])))))
        own_y = float(np.sum(np.log(np.diag(np.linalg.cholesky(cov[width_x:, width_x:])))))
    except np.linalg.LinAlgError:
        return 0.0
    digamma = psi((rows - np.arange(1, width + 1)) / 2.0) / 2.0
    offset = (np.log(2.0) - np.log(rows - 1.0)) / 2.0
    own_x -= width_x * offset + digamma[:width_x].sum()
    own_y -= width_y * offset + digamma[:width_y].sum()
    joint -= width * offset + digamma.sum()
    return float(own_x + own_y - joint)


@dataclass
class SynergyReport:
    sources: tuple[str, str]
    target: str
    joint: float
    unique_a: float
    unique_b: float
    redundancy: float
    synergy: float
    normalised: float
    null_q99: float
    interaction_gain: float
    rows: int
    #: How many shifts the bar was read from, and the bar's own spread.
    null_draws: int = 0
    null_median: float = 0.0
    null_spread: float = 0.0
    #: The same bar on the unnormalised synergy. A fraction compares two
    #: numbers whose denominators can differ between the arms; this one does
    #: not have that problem and is the stricter of the two to clear.
    raw_null_q99: float = 0.0
    #: The interaction gain on each forward-chaining fold, and the standard
    #: error of their mean. The single-split gain passed noise around zero: an
    #: additive target with no interaction at all passed 10 of 20 seeds at a
    #: gain of a few millionths either side of zero.
    interaction_gain_folds: tuple[float, ...] = ()
    interaction_gain_se: float = 0.0
    #: ISC-v3's null (docs/ISC_V3_PREREGISTRATION.md): the raw synergy of sources
    #: simulated from their own fitted autoregression, without the target. Its
    #: 99th percentile, its spread, how many simulations it was read from, and
    #: the fit's spectral radius. Computed on the change only; a level reading
    #: carries no draws.
    bootstrap_q99: float = 0.0
    bootstrap_spread: float = 0.0
    bootstrap_draws: int = 0
    bootstrap_radius: float = 0.0

    @property
    def interaction_lower_bound(self) -> float:
        """The folds' mean interaction gain less its bound, or 0.0 with no folds."""
        if not self.interaction_gain_folds:
            return 0.0
        mean = float(np.mean(self.interaction_gain_folds))
        return mean - LOWER_BOUND_Z * self.interaction_gain_se

    @property
    def passes(self) -> bool:
        """Four things, and the last two are about the bar rather than the value.

        The information-theoretic quantity has to clear its absolute bar and
        its own shifted null, on the fraction and on the raw value. The
        held-out interaction gain has to be positive, because a linear
        estimator cannot see an interaction and the two fail differently —
        which is the whole reason both are computed.

        And the margin has to beat the bar's own spread. `null_spread` was
        computed for exactly this — its comment beside the draw loop says "a
        synergy a hundredth above a null estimated to within two hundredths
        has not cleared it" — and then it was reported and read by nothing,
        which is the same defect the interaction gain had before it was
        wired in here.

        A spread needs draws to be estimated from. With one draw or none it
        is 0.0 by absence rather than by measurement, and a comparison
        against it would pass everything: absence of a check reported as a
        passed check. So the draws are required too.
        """
        return (
            self.normalised >= 0.10
            and self.normalised > self.null_q99
            and self.synergy > self.raw_null_q99
            and self.interaction_gain > 0.0
            # And established, not just positive: the same lower bound the
            # battery reads irreducibility by, on the same folds.
            and self.interaction_lower_bound > 0.0
            and self.null_draws > 1
            and self.normalised - self.null_q99 >= self.null_spread
        )

    @property
    def passes_v3(self) -> bool:
        """ISC-v3's line: `passes`, with the fraction's two null bars read on the raw value.

        The fraction's null bars ask a ratio of two small quantities to clear a
        null of the same ratio. Against sources near a random walk that null
        stays wide whatever it is built from, and v2 could not register a
        coupling of two spreads at a lag-one persistence of 0.99. v3 keeps the
        absolute floor on the fraction, the raw bar against the shifted null and
        both interaction bars. In place of the fraction's two null bars it asks
        the raw synergy to clear the bootstrap null's 99th percentile by at
        least that null's spread, which is the form the fraction bar already had.
        """
        return (
            self.normalised >= 0.10
            and self.synergy > self.raw_null_q99
            and self.synergy > self.bootstrap_q99
            and self.synergy - self.bootstrap_q99 >= self.bootstrap_spread
            and self.interaction_gain > 0.0
            and self.interaction_lower_bound > 0.0
            and self.null_draws > 1
            and self.bootstrap_draws > 1
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "sources": list(self.sources),
            "target": self.target,
            "joint_information": round(self.joint, 4),
            "redundancy": round(self.redundancy, 4),
            "unique": [round(self.unique_a, 4), round(self.unique_b, 4)],
            "synergy": round(self.synergy, 4),
            "synergy_fraction": round(self.normalised, 4),
            "null_q99_fraction": round(self.null_q99, 4),
            "interaction_gain": round(self.interaction_gain, 4),
            "interaction_gain_folds": [round(value, 5) for value in self.interaction_gain_folds],
            "interaction_gain_lower_bound": round(self.interaction_lower_bound, 5),
            "rows": self.rows,
            "null_draws": self.null_draws,
            "null_median_fraction": round(self.null_median, 4),
            "null_spread": round(self.null_spread, 5),
            "margin_over_null": round(self.normalised - self.null_q99, 5),
            "raw_null_q99": round(self.raw_null_q99, 5),
            "raw_margin_over_null": round(self.synergy - self.raw_null_q99, 5),
            "passes": self.passes,
            "bootstrap_q99": round(self.bootstrap_q99, 5),
            "bootstrap_spread": round(self.bootstrap_spread, 5),
            "bootstrap_draws": self.bootstrap_draws,
            "bootstrap_radius": round(self.bootstrap_radius, 5),
            "margin_over_bootstrap": round(self.synergy - self.bootstrap_q99, 5),
            "passes_v3": self.passes_v3,
        }


def _interaction_gain(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
    """Held-out improvement from letting the two sources multiply."""
    rows = a.shape[0]
    if rows < 80:
        return 0.0
    train, validate, test = split_rows(rows)
    side_by_side = np.hstack([a, b])
    products = np.einsum("ti,tj->tij", a, b).reshape(rows, -1)
    plain = fit_predict(side_by_side, y, train=train, validate=validate, test=test)
    crossed = fit_predict(
        np.hstack([side_by_side, products]),
        y,
        train=train,
        validate=validate,
        test=test,
        own_width=side_by_side.shape[1],
    )
    if plain.loss <= 1e-12:
        return 0.0
    return float((plain.loss - crossed.loss) / plain.loss)


def _interaction_gain_folds(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> tuple[tuple[float, ...], float]:
    """The interaction gain on each forward-chaining fold, and the error of their mean.

    The folds irreducibility is scored on: each fits on the past and tests on
    the block after it, so a gain that holds is one that predicts forward.
    """
    rows = a.shape[0]
    if rows < 80:
        return (), 0.0
    side_by_side = np.hstack([a, b])
    products = np.einsum("ti,tj->tij", a, b).reshape(rows, -1)
    widened = np.hstack([side_by_side, products])
    gains: list[float] = []
    for train, validate, test in _folds(rows):
        plain = fit_predict(side_by_side, y, train=train, validate=validate, test=test)
        crossed = fit_predict(
            widened, y, train=train, validate=validate, test=test, own_width=side_by_side.shape[1]
        )
        if plain.loss > 1e-12:
            gains.append(float((plain.loss - crossed.loss) / plain.loss))
    if len(gains) < 2:
        return tuple(gains), 0.0
    return tuple(gains), float(np.std(gains, ddof=1) / np.sqrt(len(gains)))


#: How many simulations ISC-v3's bootstrap null is read from.
BOOTSTRAP_DRAWS = 200


def _bootstrap_raw_nulls(
    raw_a: np.ndarray, raw_b: np.ndarray, y: np.ndarray, *, seed: int
) -> tuple[np.ndarray, float]:
    """Raw synergy about `y` from the two sources simulated without it.

    Both sources' components are fitted together as one first-order vector
    autoregression with an intercept. Each draw starts from the recording's
    first row and runs the fit forward with Gaussian shocks of the residual
    covariance, so it keeps each source's persistence and the relation between
    the sources and carries nothing about the target. A slid copy keeps the
    series itself, and against sources near a random walk a slid copy still
    shares slow structure with the target's history.

    The draws use a generator of their own, so the shifted null draws exactly
    what v1 and v2 read. A fit at or above a spectral radius of one is kept and
    its radius reported: its simulations wander further and widen the null,
    which can only make the line harder to pass. A draw that overflows is
    dropped, and the line needs at least two.
    """
    both = np.hstack([raw_a, raw_b])
    rows, width = both.shape
    split = raw_a.shape[1]
    design = np.hstack([both[:-1], np.ones((rows - 1, 1))])
    coefficients, *_ = np.linalg.lstsq(design, both[1:], rcond=None)
    residual = both[1:] - design @ coefficients
    covariance = np.atleast_2d(np.cov(residual, rowvar=False)) + 1e-9 * np.eye(width)
    shocks_shape = np.linalg.cholesky(covariance)
    transition, intercept = coefficients[:width], coefficients[width]
    radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    rng = np.random.default_rng([int(seed), 27])
    paths = np.empty((BOOTSTRAP_DRAWS, rows, width))
    paths[:, 0] = both[0]
    shocks = rng.normal(size=paths.shape) @ shocks_shape.T
    with np.errstate(over="ignore", invalid="ignore"):
        for row in range(1, rows):
            paths[:, row] = paths[:, row - 1] @ transition + intercept + shocks[:, row]
    values = []
    for path in paths:
        if not np.all(np.isfinite(path)):
            continue
        a, b = _copula_normal(path[:, :split]), _copula_normal(path[:, split:])
        mi_a, mi_b = _gaussian_mi(a, y), _gaussian_mi(b, y)
        values.append(_gaussian_mi(np.hstack([a, b]), y) - mi_a - mi_b + min(mi_a, mi_b))
    return np.asarray(values, dtype=np.float64), radius


#: What a triple's information is about. ISC-v1 scores the target's next
#: level; ISC-v2 scores its change, because a slow level shares information with
#: a slid copy of any slow series and its shifted null rises with the drift
#: (docs/ISC_V2_PREREGISTRATION.md). v1 stays the default and stays reported.
TARGET_READINGS: tuple[str, ...] = ("level", "change")


def synergy(
    recording: Recording,
    source_a: str,
    source_b: str,
    target: str,
    *,
    seed: int = 0,
    of: str = "level",
) -> SynergyReport:
    """Syn(A_t, B_t ; Y_{t+1}) with a shifted null and an interaction check.

    ``of="change"`` scores Y_{t+1} - Y_t instead of Y_{t+1}. The null and the
    interaction check read the same target, so the comparison stays one
    comparison.
    """
    if of not in TARGET_READINGS:
        raise ValueError(f"synergy is about a target's {' or '.join(TARGET_READINGS)}, not {of!r}")
    raw_a = _components(recording.domain(source_a)[:-1])
    raw_b = _components(recording.domain(source_b)[:-1])
    following = recording.domain(target)
    raw_y = _components(following[1:] - following[:-1] if of == "change" else following[1:])
    rows = raw_a.shape[0]
    # The floor a partition is scored on, so an arm too short for one measure
    # is too short for both.
    if rows < MIN_TRANSITIONS or min(raw_a.shape[1], raw_b.shape[1], raw_y.shape[1]) < 1:
        return SynergyReport((source_a, source_b), target, 0, 0, 0, 0, 0, 0, 1, 0, rows)
    # Ranks once, before the null. A circular shift keeps every marginal, so a
    # slid copy of these is already the copula of the slid series.
    a, b, y = (_copula_normal(block) for block in (raw_a, raw_b, raw_y))

    joint = _gaussian_mi(np.hstack([a, b]), y)
    mi_a = _gaussian_mi(a, y)
    mi_b = _gaussian_mi(b, y)
    redundancy = min(mi_a, mi_b)
    unique_a = mi_a - redundancy
    unique_b = mi_b - redundancy
    value = joint - redundancy - unique_a - unique_b
    fraction = value / joint if joint > 1e-9 else 0.0

    # Both sources slide together, by the same shift. Sliding only B destroys
    # the A-to-B relationship as well as the B-to-target one, and the two are
    # not the same thing: redundancy between the sources is structure an
    # integrated system is entitled to, and taking it away shrinks the null's
    # joint information. The fraction is that value over the joint, so a
    # smaller denominator inflates it — three of the four triples in run_019
    # scored under a null that was reading a smaller system, A+S->G at 0.079
    # against 0.255. Sliding the pair keeps every within-source and
    # between-source relationship and destroys only the alignment to the
    # target, which is what synergy is about.
    rng = np.random.default_rng(seed)
    nulls = np.empty(NULL_DRAWS, dtype=np.float64)
    raw_nulls = np.empty(NULL_DRAWS, dtype=np.float64)
    for draw in range(NULL_DRAWS):
        shift = int(rng.integers(rows // 8, rows - rows // 8)) if rows > 16 else 1
        slid_a = np.roll(a, shift, axis=0)
        slid_b = np.roll(b, shift, axis=0)
        null_joint = _gaussian_mi(np.hstack([slid_a, slid_b]), y)
        null_a = _gaussian_mi(slid_a, y)
        null_b = _gaussian_mi(slid_b, y)
        null_red = min(null_a, null_b)
        null_value = null_joint - null_red - (null_a - null_red) - (null_b - null_red)
        raw_nulls[draw] = null_value
        # A fraction needs a denominator. Once the estimator is unbiased the
        # null's joint information collapses towards zero — which is the point —
        # and dividing by what is left is dividing by noise: one triple in
        # run_019 read a null fraction of 0.58 from a joint of about a
        # thousandth. So the ratio is only taken while the null still has a
        # tenth of the real system's joint to divide by, and below that the raw
        # comparison is the one that applies.
        floor = 0.1 * joint
        nulls[draw] = null_value / null_joint if null_joint > max(1e-6, floor) else 0.0

    # The bar's own uncertainty. A synergy a hundredth above a null estimated
    # to within two hundredths has not cleared it, and a report carrying only
    # the quantile cannot say so.
    null_spread = float(np.std(nulls, ddof=1)) if nulls.size > 1 else 0.0
    null_median = float(np.median(nulls))
    # And the same comparison on the unnormalised quantity, because a ratio
    # whose denominator differs between the two arms is not one comparison.
    raw_q99 = float(np.quantile(raw_nulls, 0.99))
    folds, fold_error = _interaction_gain_folds(raw_a, raw_b, raw_y)
    # ISC-v3 reads the change against sources simulated without the target.
    if of == "change":
        bootstrap, radius = _bootstrap_raw_nulls(raw_a, raw_b, y, seed=seed)
    else:
        bootstrap, radius = np.empty(0), 0.0

    return SynergyReport(
        sources=(source_a, source_b),
        target=target,
        joint=joint,
        unique_a=unique_a,
        unique_b=unique_b,
        redundancy=redundancy,
        synergy=value,
        normalised=float(fraction),
        null_q99=float(np.quantile(nulls, 0.99)),
        # On the components before the rank transform. This one is here to
        # fail differently from the information estimate, and giving it the
        # same input would take away half of that.
        interaction_gain=_interaction_gain(raw_a, raw_b, raw_y),
        interaction_gain_folds=folds,
        interaction_gain_se=fold_error,
        rows=rows,
        null_draws=int(nulls.size),
        null_median=null_median,
        null_spread=null_spread,
        raw_null_q99=raw_q99,
        bootstrap_q99=float(np.quantile(bootstrap, 0.99)) if bootstrap.size else 0.0,
        bootstrap_spread=float(np.std(bootstrap, ddof=1)) if bootstrap.size > 1 else 0.0,
        bootstrap_draws=int(bootstrap.size),
        bootstrap_radius=radius,
    )


#: The three triples named in the specification, plus one that closes the
#: self/action loop. Fixed before the run so a passing triple cannot be found
#: by searching all four hundred of them.
TRIPLES: tuple[tuple[str, str, str], ...] = (
    ("A", "S", "G"),
    ("P", "M", "W"),
    ("W", "A", "D"),
    ("S", "D", "C"),
)


def synergy_suite(recording: Recording, *, seed: int = 0, of: str = "level") -> list[SynergyReport]:
    return [synergy(recording, a, b, y, seed=seed, of=of) for a, b, y in TRIPLES]
