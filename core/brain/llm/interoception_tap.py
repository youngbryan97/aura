"""core/brain/llm/interoception_tap.py — per-token substrate interoception (worker side).

Every decode step, the resident model computes a full log-probability
distribution over its vocabulary — its actual, momentary belief about what
comes next. Until now that signal was discarded in the worker loop, and the
parent's "surprise" feedback ran on a unique-word-ratio heuristic over the
finished text with ``logprobs=None``. This module samples the real signal at a
bounded cadence so observation cannot dominate generation latency.

The tap is a **pure observer** with three hard guarantees:

1. It can never alter generation — it only reads ``response.token`` /
   ``response.logprobs`` / ``response.text`` after the sampler has already run.
2. It can never raise into the token loop — every per-token failure is
   swallowed, counted in the payload as ``dropped``, and generation continues.
3. It is bounded — at most :data:`_HARD_TOKEN_CAP` per-token records, a
   fixed-size spike list, and a payload that stays a few KB of JSON no matter
   how long the generation ran.

What it measures, per sampled token:

* **surprisal** ``-log p(token)`` in nats — how unexpected the chosen word was
  to the model that chose it;
* **entropy** of the full next-token distribution — how open the moment was;
* **top-2 probability gap** — how contested the choice was (felt ambivalence);
* whether the sampled token was the argmax (exploration vs conviction).

Interpretation (fluency, felt confidence, strain) happens parent-side in
:mod:`core.being.thought_interoception`; the worker ships measurements only.

Kill switch: ``AURA_INTEROCEPTION=0`` disables the tap entirely
(:func:`maybe_build_tap` returns ``None``).
"""
from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, NamedTuple

logger = logging.getLogger("Aura.Brain.InteroceptionTap")

# Mirrors the worker's absolute generation cap; the tap never stores more.
_HARD_TOKEN_CAP = 8192
_DENSE_OPENING_TOKENS = 4

# A token only counts as a reportable "spike" above this surprisal (nats):
# 2.0 nats ⇒ the model gave the sampled word < ~14% probability. Below that,
# naming the word as "contested" would be noise, not sensation.
_SPIKE_FLOOR_NATS = 2.0

# Per-token failures we absorb (count + continue) rather than raise into the
# decode loop. Deliberately broad short of BaseException: a dropped
# measurement is always preferable to a damaged generation.
_STEP_RECOVERABLE = (
    AttributeError,
    IndexError,
    KeyError,
    OverflowError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    ZeroDivisionError,
)

PAYLOAD_VERSION = 1


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


def _finite(x: Any) -> float | None:
    try:
        f = float(x)
    # not a failure: a value that is not a number is not one this can read.
    except (TypeError, ValueError, OverflowError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


class StepStats(NamedTuple):
    """One decode step's measurements.

    A NamedTuple rather than a bare tuple: appending a field to a positional
    tuple silently rebinds callers that unpack with ``*_, last`` — which is
    precisely how adding the probability sum broke three call sites. Named
    access makes later measurements additive.
    """

    surprisal: float
    entropy: float
    top1: float
    top2: float
    argmax_hit: bool
    #: Total probability mass of the supplied distribution, used once per
    #: attempt to verify it really was normalized log-probabilities.
    prob_sum: float | None = None


def _step_stats(logprobs: Any, token_id: int) -> tuple[float, float, float, float, bool] | None:
    """Compute (surprisal, entropy, top1_logprob, top2_logprob, argmax_hit).

    Accepts an MLX array (live worker) or anything ``numpy.asarray`` can take
    (tests, CPU fallback). Returns ``None`` when the distribution is unusable;
    the caller counts it as a dropped measurement.
    """
    raw_stats: tuple[Any, Any, Any, Any, Any] | None = None
    try:
        import mlx.core as mx  # noqa: PLC0415 — worker-local heavy import

        if isinstance(logprobs, mx.array):
            lp = logprobs.reshape(-1)
            size = int(lp.shape[0])
            if not (0 <= int(token_id) < size):
                return None
            mlx_token_logprob = lp[int(token_id)]
            mlx_probs = mx.exp(lp)
            mlx_entropy = -mx.sum(mlx_probs * lp)
            # Total probability mass rides along in the same materialization, so
            # verifying normalization costs no extra device synchronization.
            mlx_prob_sum = mx.sum(mlx_probs)
            pair = mx.topk(lp, k=2) if size >= 2 else mx.stack([lp[0], lp[0]])
            arg = mx.argmax(lp)
            # One materialization for the whole step: five scalars.
            packed: Any = mx.stack(
                [
                    mlx_token_logprob,
                    mlx_entropy,
                    pair.reshape(-1)[0],
                    pair.reshape(-1)[1],
                    arg.astype(mx.float32),
                    mlx_prob_sum,
                ]
            ).tolist()
            if isinstance(packed, (list, tuple)) and len(packed) >= 6:
                raw_stats = (packed[0], packed[1], packed[2], packed[3],
                             packed[4], packed[5])
    # not a failure: no MLX here, so there are no raw stats to pack.
    except ImportError:
        raw_stats = None

    if raw_stats is None:
        import numpy as np  # noqa: PLC0415

        arr = np.asarray(logprobs, dtype=np.float64).reshape(-1)
        size = int(arr.shape[0])
        if size == 0 or not (0 <= int(token_id) < size):
            return None
        numpy_token_logprob = float(arr[int(token_id)])
        with np.errstate(over="ignore", invalid="ignore"):
            probs = np.exp(arr)
            numpy_entropy = float(-np.sum(probs * arr))
            numpy_prob_sum = float(np.sum(probs))
        if size >= 2:
            top_idx = np.argpartition(arr, -2)[-2:]
            numpy_top_a = float(arr[top_idx[0]])
            numpy_top_b = float(arr[top_idx[1]])
        else:
            numpy_top_a = numpy_top_b = numpy_token_logprob
        raw_stats = (
            numpy_token_logprob,
            numpy_entropy,
            numpy_top_a,
            numpy_top_b,
            float(int(np.argmax(arr))),
            numpy_prob_sum,
        )

    token_logprob = _finite(raw_stats[0])
    entropy_raw = _finite(raw_stats[1])
    top_a = _finite(raw_stats[2])
    top_b = _finite(raw_stats[3])
    arg_raw = _finite(raw_stats[4])
    prob_sum = _finite(raw_stats[5]) if len(raw_stats) > 5 else None
    if (
        token_logprob is None
        or entropy_raw is None
        or top_a is None
        or top_b is None
        or arg_raw is None
    ):
        return None
    top1, top2 = (
        (top_a, top_b) if top_a >= top_b else (top_b, top_a)
    )  # mx.topk does not guarantee order
    surprisal = max(0.0, -token_logprob)
    entropy = max(0.0, entropy_raw)
    argmax_hit = int(arg_raw) == int(token_id)
    return StepStats(surprisal, entropy, top1, top2, argmax_hit, prob_sum)


def _downsample_mean(values: list[float], points: int) -> list[float]:
    """Bucket-mean a series down to at most ``points`` values."""
    n = len(values)
    if n == 0 or points <= 0:
        return []
    if n <= points:
        return [round(v, 4) for v in values]
    out: list[float] = []
    for b in range(points):
        lo = (b * n) // points
        hi = max(lo + 1, ((b + 1) * n) // points)
        chunk = values[lo:hi]
        out.append(round(sum(chunk) / len(chunk), 4))
    return out


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


class InteroceptionTap:
    """Accumulates per-token substrate measurements for one generation attempt."""

    def __init__(
        self,
        *,
        spike_k: int | None = None,
        curve_points: int | None = None,
        sample_points: int | None = None,
        near_tie_gap: float = 0.10,
        step_stride: int = 1,
        request_id: str = "",
        model_id: str = "",
        provider: str = "",
    ) -> None:
        self._spike_k = spike_k if spike_k is not None else _env_int(
            "AURA_INTEROCEPTION_TOPK_SPIKES", 8, 0, 32
        )
        self._curve_points = curve_points if curve_points is not None else _env_int(
            "AURA_INTEROCEPTION_CURVE_POINTS", 32, 0, 128
        )
        self._sample_points = sample_points if sample_points is not None else _env_int(
            "AURA_INTEROCEPTION_SAMPLE_POINTS", 128, 8, 512
        )
        self._near_tie_gap = max(0.0, min(1.0, float(near_tie_gap)))
        self._step_stride = max(1, min(64, int(step_stride)))

        self._observed_tokens = 0
        self._skipped_stride = 0
        self._surprisals: list[float] = []
        self._entropies: list[float] = []
        self._prob_gaps: list[float] = []  # p(top1) - p(top2) per token
        self._argmax_hits = 0
        self._token_ids: list[int] = []
        self._token_lps: list[float] = []
        self._sampled_positions: list[int] = []
        self._texts: list[str] = []
        self._dropped = 0
        self._first_feed_at: float | None = None
        self._last_feed_at: float | None = None
        self._step_error_logged = False

        # Provenance, so the parent can bind a measurement to what produced it.
        self._request_id = str(request_id or "")[:64]
        self._model_id = str(model_id or "")[:128]
        self._provider = str(provider or "")[:64]
        self._created_at = time.time()
        #: What the observed array actually was. "assumed_log_softmax" is the
        #: honest default: the maths treats the input as normalized
        #: log-probabilities, and nothing upstream states that it is. Set to
        #: "verified_log_softmax" once a distribution passes the sum-to-one
        #: check below, or "unnormalized" when it demonstrably does not.
        self._logprob_stage = "assumed_log_softmax"
        self._normalization_checked = False

    def _provenance(self) -> dict[str, Any]:
        return {
            "request_id": self._request_id,
            "model_id": self._model_id,
            "provider": self._provider,
            "created_at": round(self._created_at, 3),
        }

    def _note_normalization(self, total_probability: float) -> None:
        """Record whether the supplied array really was a log-probability
        distribution.

        Entropy and surprisal are only meaningful over normalized
        log-probabilities. The tap exponentiates whatever it is handed, so raw
        logits or post-processor scores would produce numbers that LOOK like
        nats but are not. Checked once per attempt (the stage does not change
        mid-generation) to keep this off the hot path.
        """
        if self._normalization_checked:
            return
        self._normalization_checked = True
        if not math.isfinite(total_probability):
            self._logprob_stage = "non_finite"
        elif 0.90 <= total_probability <= 1.10:
            self._logprob_stage = "verified_log_softmax"
        else:
            self._logprob_stage = "unnormalized"
            logger.warning(
                "Interoception tap: supplied distribution sums to %.4f, not 1.0 — "
                "entropy/surprisal are not in nats. Reporting stage=unnormalized.",
                total_probability,
            )

    # ── ingestion ────────────────────────────────────────────────────────────
    def feed(self, token_id: Any, logprobs: Any, token_text: Any) -> None:
        """Record one decode step. Never raises."""
        try:
            position = self._observed_tokens
            self._observed_tokens += 1
            if position >= _HARD_TOKEN_CAP:
                self._dropped += 1
                return
            self._texts.append(str(token_text or ""))
            sample_step = (
                position < _DENSE_OPENING_TOKENS
                or position % self._step_stride == 0
            )
            if not sample_step:
                self._skipped_stride += 1
                return
            if logprobs is None:
                self._dropped += 1
                return
            stats = _step_stats(logprobs, int(token_id))
            if stats is None:
                self._dropped += 1
                return
            surprisal, entropy = stats.surprisal, stats.entropy
            top1, top2, argmax_hit = stats.top1, stats.top2, stats.argmax_hit
            if stats.prob_sum is not None:
                self._note_normalization(stats.prob_sum)
            now = time.monotonic()
            if self._first_feed_at is None:
                self._first_feed_at = now
            self._last_feed_at = now
            self._surprisals.append(surprisal)
            self._entropies.append(entropy)
            # exp of values ≤ 0 is safe; top1/top2 are logprobs of a normalized dist.
            self._prob_gaps.append(
                max(0.0, math.exp(min(0.0, top1)) - math.exp(min(0.0, top2)))
            )
            if argmax_hit:
                self._argmax_hits += 1
            self._token_ids.append(int(token_id))
            self._token_lps.append(-surprisal)
            self._sampled_positions.append(position)
        except _STEP_RECOVERABLE as exc:
            self._dropped += 1
            if not self._step_error_logged:
                self._step_error_logged = True
                logger.debug("Interoception step measurement failed (once-logged): %s", exc)

    @property
    def measured_tokens(self) -> int:
        return len(self._surprisals)

    # ── live snapshot (rides progress IPC messages) ─────────────────────────
    def live_snapshot(self) -> dict[str, Any] | None:
        """Cheap running summary for mid-generation pulses. Never raises."""
        try:
            n = len(self._surprisals)
            if n == 0:
                return None
            return {
                "token_count": self._observed_tokens,
                "measured_token_count": n,
                "mean_surprisal": round(sum(self._surprisals) / n, 4),
                "mean_entropy": round(sum(self._entropies) / n, 4),
            }
        # not a failure: a step that refuses has no reading to summarise.
        except _STEP_RECOVERABLE:
            return None

    # ── finalize ─────────────────────────────────────────────────────────────
    def finalize(self, *, attempt: int = 0, generation_tps: float | None = None) -> dict[str, Any] | None:
        """Distil the attempt into a compact, JSON-safe payload. Never raises.

        Returns ``None`` only when the tap genuinely observed nothing. If
        measurements were ATTEMPTED and all failed, a payload is still emitted
        carrying the drop count — see below.
        """
        try:
            n = len(self._surprisals)
            if n == 0:
                # The tap promises a `dropped` count, then used to throw it away
                # in exactly the case it matters: total measurement failure
                # returned None, making "the provider gave unusable
                # distributions on every token" indistinguishable from "no tap
                # ran at all". A silent instrument is the one failure mode an
                # observability surface must never have.
                if self._dropped <= 0 and self._observed_tokens <= 0:
                    return None
                return {
                    "version": PAYLOAD_VERSION,
                    "attempt": int(attempt),
                    "measured": False,
                    "token_count": self._observed_tokens,
                    "measured_token_count": 0,
                    "dropped": self._dropped,
                    "skipped_stride": self._skipped_stride,
                    "reason": (
                        "no_usable_distribution"
                        if self._dropped > 0
                        else "no_tokens_sampled"
                    ),
                    "logprob_stage": self._logprob_stage,
                    **self._provenance(),
                }
            duration = 0.0
            if self._first_feed_at is not None and self._last_feed_at is not None:
                duration = max(0.0, self._last_feed_at - self._first_feed_at)
            tps = generation_tps
            if tps is None and duration > 0 and n > 1:
                observed_span = self._sampled_positions[-1] - self._sampled_positions[0]
                tps = observed_span / duration

            sorted_surprisal = sorted(self._surprisals)
            near_ties = sum(1 for gap in self._prob_gaps if gap < self._near_tie_gap)
            tail = self._entropies[-16:]

            stride = max(1, (n + self._sample_points - 1) // self._sample_points)
            payload: dict[str, Any] = {
                "version": PAYLOAD_VERSION,
                "attempt": int(attempt),
                "measured": True,
                # Provenance: without these the parent can misattribute
                # measurements across attempts or models, since an integer
                # attempt is not enough to tell two generations apart.
                **self._provenance(),
                # Which stage of the sampler produced the array we read. The
                # maths below assumes normalized log-probabilities; if the
                # provider handed us raw logits or post-processor scores the
                # numbers are not entropy and surprisal at all, so what was
                # observed is reported rather than assumed.
                "logprob_stage": self._logprob_stage,
                "token_count": self._observed_tokens,
                "measured_token_count": n,
                "sampling_stride": self._step_stride,
                "sampling_coverage": round(
                    n / max(1, min(self._observed_tokens, _HARD_TOKEN_CAP)), 4
                ),
                "skipped_stride": self._skipped_stride,
                "dropped": self._dropped,
                "duration_s": round(duration, 3),
                "tokens_per_s": round(tps, 2) if tps is not None else None,
                "mean_surprisal": round(sum(self._surprisals) / n, 4),
                "p90_surprisal": round(_percentile(sorted_surprisal, 0.90), 4),
                "max_surprisal": round(sorted_surprisal[-1], 4),
                "mean_entropy": round(sum(self._entropies) / n, 4),
                "peak_entropy": round(max(self._entropies), 4),
                "tail_entropy": round(sum(tail) / len(tail), 4),
                "mean_top2_gap": round(sum(self._prob_gaps) / n, 4),
                "near_tie_rate": round(near_ties / n, 4),
                "argmax_rate": round(self._argmax_hits / n, 4),
                "curve": _downsample_mean(self._surprisals, self._curve_points),
                "spikes": self._top_spikes(),
                "token_ids_sample": self._token_ids[::stride][: self._sample_points],
                "logprob_sample": [
                    round(v, 4) for v in self._token_lps[::stride][: self._sample_points]
                ],
            }
            return payload
        except _STEP_RECOVERABLE as exc:
            logger.debug("Interoception finalize failed: %s", exc)
            return None

    def _top_spikes(self) -> list[dict[str, Any]]:
        """The K most surprising moments, with the words they landed on.

        Position 0 is excluded from *selection* (the first token reflects the
        prompt boundary, not content uncertainty) but stays in the aggregates.
        """
        if self._spike_k <= 0:
            return []
        candidates = sorted(
            (
                i for i in range(len(self._surprisals))
                if self._sampled_positions[i] != 0
                and self._surprisals[i] >= _SPIKE_FLOOR_NATS
            ),
            key=lambda i: self._surprisals[i],
            reverse=True,
        )[: self._spike_k]
        spikes = []
        for i in sorted(candidates):
            position = self._sampled_positions[i]
            context = "".join(self._texts[max(0, position - 3): position + 3])
            context = " ".join(context.split())[:60]
            spikes.append(
                {
                    "pos": position,
                    "text": self._texts[position].strip()[:24],
                    "context": context,
                    "surprisal": round(self._surprisals[i], 4),
                }
            )
        return spikes


def interoception_enabled() -> bool:
    return _env_flag("AURA_INTEROCEPTION", True)


def maybe_build_tap(
    *,
    request_id: str = "",
    model_id: str = "",
    provider: str = "",
) -> InteroceptionTap | None:
    """Env-gated constructor. Returns ``None`` (tap disabled) on any failure.

    Provenance is optional so existing callers keep working, but supplying it
    is what lets the parent bind a measurement to the request and model that
    produced it rather than guessing from an attempt number.
    """
    try:
        if not interoception_enabled():
            return None
        return InteroceptionTap(
            step_stride=_env_int("AURA_INTEROCEPTION_STEP_STRIDE", 4, 1, 64),
            request_id=request_id,
            model_id=model_id,
            provider=provider,
        )
    except _STEP_RECOVERABLE as exc:
        logger.debug("Interoception tap unavailable: %s", exc)
        return None
