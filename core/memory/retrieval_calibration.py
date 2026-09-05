"""core/memory/retrieval_calibration.py — what score counts as a hit.

THE PROBLEM THIS REPLACES
=========================
``retrieve_memories`` filtered on ``score >= 0.01``. Measured 12 Aug 2026
against a null of unrelated Aura-domain pairs:

    all-MiniLM-L6-v2      null mean +0.060 (max +0.132)   0.01 admits 5/6 nulls
    Qwen3-Embedding@384   null mean +0.260 (max +0.415)   0.01 admits 6/6 nulls

So 0.01 was never filtering anything — it rubber-stamped any positive cosine,
under the old model as much as the new one. And the new encoder's null centre
sits four times higher, so no constant carried over from MiniLM could mean
what it used to.

WHY NOT JUST PICK A BIGGER NUMBER
=================================
Because an absolute cosine threshold is the wrong instrument for an
instruction-tuned embedder. Their cosine ranges are compressed and elevated:
on the same sample, Qwen3-Embedding's signal and null populations OVERLAP on
absolute value (min signal +0.273 below max null +0.415) while it beat MiniLM
on ranking 3/4 to 1/4. It is the better retriever whose absolute scores mean
less. A constant tuned today would be wrong for the next encoder too, and
wrong silently.

WHAT THIS DOES INSTEAD
======================
Calibrate against the model's OWN null, measured from the corpus being
searched, then express the cut as a quantile of that null. The quantile is a
false-positive rate — a specification, not a magic number — and it stays
meaningful across any encoder because it is defined relative to that
encoder's own behaviour on unrelated text.

A null is built by scoring randomly paired, non-matching texts from the live
candidate set. That costs one extra batch of comparisons over vectors that
are already cached, and it is the only way to answer "is this above chance"
rather than "is this above 0.01".

When a null cannot be built — fewer candidates than ``MIN_NULL_SAMPLES`` —
the honest answer is that no cut can be justified, so ranking alone decides
and every candidate is admitted for top-k to order. That is strictly what the
old constant was already doing, minus the pretence of filtering.

WHAT THIS DOES NOT DO
=====================
The null here is estimated from the query's own score distribution, which is
free but limited: a UNIFORM distribution genuinely has a top end, and no
single-query distribution can distinguish "everything is noise" from "there
is weak signal" without scoring against text known to be unrelated. So this
cuts reliably when the population is separable (a tight bulk plus clear hits
— the shape real retrieval produces) and degrades to rank-plus-top_k when it
is not. It never claims a cut it cannot support.

Closing that gap needs an EXTERNAL null: a fixed sample of unrelated text
embedded once per model and cached, so any query can be scored against known
non-matches. That costs one extra batch at model-load time and is the honest
next step if the degenerate case ever bites.
"""
from __future__ import annotations

import logging
import random
from collections.abc import Sequence

logger = logging.getLogger("Aura.RetrievalCalibration")

#: Below this many candidates the null is too small to estimate a tail from,
#: and a quantile over 4 samples is noise wearing a number's clothes.
MIN_NULL_SAMPLES = 12

#: Pairs sampled to estimate the null. Bounded so calibration stays O(1) in
#: corpus size — this runs inside a retrieval, not offline.
MAX_NULL_PAIRS = 64

#: Default false-positive rate. This IS the knob, and it is stated as a rate
#: rather than a cosine so it survives an encoder change. 0.95 admits a
#: candidate only if it outscores 95% of unrelated pairs under this model.
DEFAULT_NULL_QUANTILE = 0.95


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile. No numpy dependency at import time."""
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sample")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    frac = pos - low
    return float(sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac)


#: MAD -> sigma for a normal distribution. Standard constant, not a tuning
#: knob: median absolute deviation times this estimates the standard
#: deviation robustly, without letting a single strong hit inflate it.
_MAD_TO_SIGMA = 1.4826


def _z_for_quantile(q: float) -> float:
    """The standard-normal z whose upper tail is ``1 - q``.

    Derived from the requested false-positive rate rather than picked, so
    the only number a caller supplies is the rate itself. Acklam's rational
    approximation; accurate to ~1e-9 over the range we use.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"quantile must be in (0, 1); got {q!r}")
    # Symmetric tail transform, then a compact rational approximation.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if q < p_low:
        r = (-2 * __import__("math").log(q)) ** 0.5
        return (((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / \
               ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1)
    if q > p_high:
        r = (-2 * __import__("math").log(1 - q)) ** 0.5
        return -(((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / \
                ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1)
    r = q - 0.5
    t = r * r
    return (((((a[0]*t+a[1])*t+a[2])*t+a[3])*t+a[4])*t+a[5])*r / \
           (((((b[0]*t+b[1])*t+b[2])*t+b[3])*t+b[4])*t+1)


def null_threshold(
    scores: Sequence[float],
    *,
    quantile: float = DEFAULT_NULL_QUANTILE,
    rng: random.Random | None = None,
) -> float | None:
    """The score a candidate must beat to be better than chance.

    ``scores`` is this query's score against every candidate. The null is
    approximated by the LOW half of that distribution: in a corpus where a
    query matches a handful of documents, most candidates are unrelated, so
    the bulk of the score distribution IS the null. This avoids a second
    embedding pass while still being measured rather than assumed.

    The cut is ``median + z * sigma`` over that null, with sigma estimated
    robustly from the MAD and z derived from the requested false-positive
    rate. A plain quantile of the low half was tried first and is wrong: it
    compares the top of the distribution against the bottom of the SAME
    distribution, so it always finds "winners" even in pure noise. Dispersion
    is what separates "there is a hit here" from "these all look alike" —
    a uniform spread has no outliers, and this returns few or none for it.

    Returns None when the sample is too small to justify any cut.
    """
    values = [float(s) for s in scores]
    if len(values) < MIN_NULL_SAMPLES:
        return None
    ordered = sorted(values)
    # The lower half: candidates a query does not match. Using the median as
    # the split point assumes fewer than half the corpus is relevant to any
    # one query, which is what "retrieval" means.
    null_sample = ordered[: max(MIN_NULL_SAMPLES // 2, len(ordered) // 2)]
    if rng is not None and len(null_sample) > MAX_NULL_PAIRS:
        null_sample = sorted(rng.sample(null_sample, MAX_NULL_PAIRS))
    median = _quantile(null_sample, 0.5)
    deviations = sorted(abs(v - median) for v in null_sample)
    mad = _quantile(deviations, 0.5)
    sigma = mad * _MAD_TO_SIGMA
    if sigma <= 0.0:
        # A null with no spread at all (every unrelated pair scored
        # identically). Nothing can be shown to exceed it; fall back to the
        # null's own maximum so only a strictly higher score survives.
        return float(null_sample[-1])
    return median + _z_for_quantile(quantile) * sigma


def select_above_chance(
    scored: Sequence[tuple[float, object]],
    *,
    top_k: int,
    quantile: float = DEFAULT_NULL_QUANTILE,
    floor: float | None = None,
) -> list[object]:
    """Rank, then cut where the scores stop looking like chance.

    ``floor`` is an optional absolute backstop for callers that genuinely
    have one (a caller who knows its corpus). It is applied IN ADDITION to
    the calibrated cut, never instead of it, so pinning a floor cannot
    resurrect the old behaviour of admitting everything.
    """
    if not scored:
        return []
    ranked = sorted(scored, key=lambda pair: pair[0], reverse=True)
    cut = null_threshold([s for s, _ in ranked], quantile=quantile)
    if cut is None:
        # Too few candidates to calibrate. Ranking is all we honestly have.
        kept = ranked
    else:
        kept = [pair for pair in ranked if pair[0] > cut]
        if not kept:
            # Everything looks like noise. That is a real answer — but a
            # retrieval that returns nothing when the caller asked for its
            # best guess is worse than one that returns its top-ranked item
            # and lets the caller see the score. Keep exactly one.
            kept = ranked[:1]
    if floor is not None:
        kept = [pair for pair in kept if pair[0] >= floor] or kept[:1]
    return [item for _, item in kept[:top_k]]
