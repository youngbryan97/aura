"""Admit work by what the machine has been measured to do, not by a formula.

The existing adaptive-compute allocator is a closed-form policy: weighted
sums of difficulty, uncertainty, stakes and pressure, compared against
hand-chosen thresholds. It is honest about being an uncalibrated heuristic
with no model-specific calibration, no uncertainty interval and no
comparison against a control policy. What it cannot do is answer the only
question that matters at admission time:

    Can THIS request finish inside the person's deadline, on THIS machine,
    in its CURRENT state — without destabilising the resident model?

A formula answers "is this number allowable". Those are different
questions, and the gap between them produced real incidents: a caller with
a ten-second timeout allocated fifteen seconds of work; expected
foreground contention escalated into a failure lockdown; a prompt cache
that was never built and then cleared every turn made conversation cost
grow quadratically while the policy saw nothing wrong.

So this measures. Every completed generation reports its prefill and
decode timings, and the estimator keeps a per-task-shape distribution of
what actually happened. Admission then predicts a completion time WITH an
uncertainty interval and compares it to the real deadline.

Three properties the formula could not have:

* **It knows when it does not know.** With no samples for a task shape,
  ``confidence`` is ``no_samples`` and the decision says so instead of
  producing a number with invented authority. An unmeasured shape is
  admitted cautiously, not confidently.
* **It gets less wrong over time.** The estimator is fed real outcomes, so
  a machine that is slower today allocates less today.
* **Background work consumes measured slack.** Not "runs and gets
  classified afterwards" — the slack is computed from the foreground
  lease and the queue before the work is allowed to start.

Admission happens BEFORE the expensive work. A request that cannot finish
is downgraded or refused while that is still cheap, rather than discovered
at the deadline when the only remaining move is to kill something.
"""
from __future__ import annotations

import enum
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

__all__ = [
    "AdmissionDecision",
    "AdmissionOutcome",
    "Confidence",
    "TaskShape",
    "ThroughputEstimator",
    "get_throughput_estimator",
    "recommended_completion_tokens",
    "recommended_foreground_deadline",
    "admit",
    "record_generation",
]


class AdmissionOutcome(str, enum.Enum):
    """What the scheduler decided, before any expensive work started."""

    #: Predicted to finish inside the deadline with margin.
    ADMIT = "admit"
    #: Cannot finish as requested, but a smaller version can. The caller
    #: gets a reduced token budget rather than a failure — a shorter answer
    #: on time beats a better one the person never sees.
    DOWNGRADE = "downgrade"
    #: Cannot finish even reduced, or admitting it would destabilise the
    #: resident model. Refused now, while refusing is still cheap.
    REFUSE = "refuse"


class Confidence(str, enum.Enum):
    """How much the prediction rests on measurement.

    Reported alongside every decision. A prediction from two samples and
    one from two hundred are not the same claim, and presenting them
    identically is how an uncalibrated number acquires authority.
    """

    NO_SAMPLES = "no_samples"
    SPARSE = "sparse"
    MEASURED = "measured"


#: Below this, the estimate is a guess with error bars we cannot trust.
_SPARSE_SAMPLES = 5
#: At or above this, the distribution is worth planning against.
_MEASURED_SAMPLES = 20
#: Ring size per shape. Long enough to be stable, short enough to track a
#: machine that has genuinely changed (thermal throttle, memory pressure).
_WINDOW = 64
#: Safety factor on the predicted duration when we have no measurements.
#: Applied to a conservative default rather than used to invent precision.
_UNMEASURED_CAUTION = 2.0
#: Fraction of the deadline reserved for everything that is not decode:
#: gates, assembly, delivery. Measured separately as overhead samples.
_MIN_MARGIN = 0.15


def _unmeasured_decode_seconds_per_token(model: str) -> float:
    """Conservative local decode prior until this process has measurements.

    A single 25ms/token default described a small accelerator model reasonably
    and a resident 32B on this host disastrously.  The latter was therefore
    admitted for 1,024 tokens inside a window that could physically decode only
    about 220.  Model size is not a measurement, so the resulting decision
    remains ``NO_SAMPLES``; it is only a safer prior than pretending all model
    sizes have the same throughput.
    """

    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*[bB](?![A-Za-z])", str(model or ""))
    if match is None:
        return 2.5e-2
    billions = float(match.group(1))
    if billions <= 3.0:
        return 1.8e-2
    if billions <= 8.0:
        return 3.5e-2
    if billions <= 16.0:
        return 6.5e-2
    if billions <= 40.0:
        return 1.6e-1
    return 2.5e-1


@dataclass(frozen=True)
class TaskShape:
    """The dimensions along which throughput genuinely differs.

    Deliberately coarse. A shape key so specific that every request is
    unique would give every request zero samples, which is the same as
    having no estimator at all.
    """

    model: str
    #: Prompt length bucket — prefill cost is dominated by this.
    prompt_bucket: str
    #: Whether the prompt cache was warm. The difference is large enough
    #: that mixing warm and cold samples makes both predictions wrong.
    cache_warm: bool
    foreground: bool

    @staticmethod
    def bucket_for(prompt_tokens: int) -> str:
        tokens = max(0, int(prompt_tokens))
        for limit in (512, 2048, 8192, 32768):
            if tokens <= limit:
                return f"<= {limit}"
        return "> 32768"

    def key(self) -> str:
        return (
            f"{self.model}|{self.prompt_bucket}|"
            f"{'warm' if self.cache_warm else 'cold'}|"
            f"{'fg' if self.foreground else 'bg'}"
        )


@dataclass
class _Samples:
    """A bounded ring of observations for one shape."""

    prefill_s_per_token: list[float] = field(default_factory=list)
    decode_s_per_token: list[float] = field(default_factory=list)
    overhead_s: list[float] = field(default_factory=list)
    generated_tokens: list[float] = field(default_factory=list)

    def add(
        self,
        prefill: float,
        decode: float,
        overhead: float,
        generated_tokens: int,
        *,
        completion_observed: bool,
    ) -> None:
        for series, value in (
            (self.prefill_s_per_token, prefill),
            (self.decode_s_per_token, decode),
            (self.overhead_s, overhead),
        ):
            if value >= 0.0 and math.isfinite(value):
                series.append(float(value))
                if len(series) > _WINDOW:
                    series.pop(0)
        if completion_observed:
            value = float(generated_tokens)
            if value >= 0.0 and math.isfinite(value):
                self.generated_tokens.append(value)
                if len(self.generated_tokens) > _WINDOW:
                    self.generated_tokens.pop(0)

    @property
    def count(self) -> int:
        return len(self.decode_s_per_token)

    @property
    def completion_count(self) -> int:
        return len(self.generated_tokens)


def _percentile(values: list[float], fraction: float) -> float:
    """Plain nearest-rank percentile. No interpolation, no numpy dependency.

    Percentile rather than mean throughout: admission cares about the slow
    case. A mean that hides a fat tail admits work that then misses.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(fraction * len(ordered))) - 1))
    return ordered[index]


@dataclass(frozen=True)
class AdmissionDecision:
    """The verdict, with the reasoning an operator needs to argue with it."""

    outcome: AdmissionOutcome
    confidence: Confidence
    reason: str
    #: Tokens the caller may actually generate. Equal to the request on
    #: ADMIT, reduced on DOWNGRADE, zero on REFUSE.
    granted_decode_tokens: int
    predicted_seconds: float
    deadline_seconds: float
    samples: int
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def admitted(self) -> bool:
        return self.outcome in (AdmissionOutcome.ADMIT, AdmissionOutcome.DOWNGRADE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "confidence": self.confidence.value,
            "reason": self.reason,
            "granted_decode_tokens": self.granted_decode_tokens,
            "predicted_seconds": round(self.predicted_seconds, 3),
            "deadline_seconds": round(self.deadline_seconds, 3),
            "samples": self.samples,
            **dict(self.detail),
        }


class ThroughputEstimator:
    """What this machine has actually been observed to do, per task shape.

    Thread-safe: generations complete on worker threads while admission
    runs on the event loop.
    """

    def __init__(self) -> None:
        self._lock = checked_lock("measured_admission.instance", reentrant=True)
        self._shapes: dict[str, _Samples] = {}

    def record(
        self,
        shape: TaskShape,
        *,
        prompt_tokens: int,
        generated_tokens: int,
        prefill_seconds: float,
        decode_seconds: float,
        overhead_seconds: float = 0.0,
        completion_observed: bool = True,
    ) -> None:
        """Feed one completed generation back in.

        Ignores degenerate observations rather than letting a divide-by-zero
        poison a shape's distribution for the next sixty-four requests.
        """
        if generated_tokens <= 0 or decode_seconds <= 0.0:
            return
        prompt_tokens = max(1, int(prompt_tokens))
        with self._lock:
            samples = self._shapes.setdefault(shape.key(), _Samples())
            samples.add(
                max(0.0, prefill_seconds) / prompt_tokens,
                max(0.0, decode_seconds) / max(1, int(generated_tokens)),
                max(0.0, overhead_seconds),
                generated_tokens,
                completion_observed=bool(completion_observed),
            )

    def confidence(self, shape: TaskShape) -> tuple[Confidence, int]:
        with self._lock:
            count = self._shapes.get(shape.key(), _Samples()).count
        if count == 0:
            return Confidence.NO_SAMPLES, 0
        if count < _SPARSE_SAMPLES:
            return Confidence.SPARSE, count
        if count < _MEASURED_SAMPLES:
            return Confidence.SPARSE, count
        return Confidence.MEASURED, count

    def predict_seconds(
        self, shape: TaskShape, *, prompt_tokens: int, decode_tokens: int
    ) -> tuple[float, Confidence, int]:
        """Predicted wall time at the slow end of what we have seen.

        p90, not mean. Admission is a promise about the deadline, and a
        promise made from an average is broken for a tenth of requests by
        construction.
        """
        confidence, count = self.confidence(shape)
        if confidence is Confidence.NO_SAMPLES:
            # No measurement. Say so, and be conservative rather than
            # inventing a rate that would look like knowledge.
            fallback = _UNMEASURED_CAUTION * (
                prompt_tokens * 1.5e-4
                + decode_tokens * _unmeasured_decode_seconds_per_token(shape.model)
            )
            return fallback, confidence, 0
        with self._lock:
            samples = self._shapes[shape.key()]
            prefill_rate = _percentile(samples.prefill_s_per_token, 0.90)
            decode_rate = _percentile(samples.decode_s_per_token, 0.90)
            overhead = _percentile(samples.overhead_s, 0.90)
        predicted = prompt_tokens * prefill_rate + decode_tokens * decode_rate + overhead
        if confidence is Confidence.SPARSE:
            # Few samples: widen rather than pretend the interval is tight.
            predicted *= 1.5
        return predicted, confidence, count

    def predict_completion_tokens(
        self,
        shape: TaskShape,
        *,
        maximum_tokens: int,
        prior_tokens: int,
    ) -> tuple[int, Confidence, int]:
        """Return observed p90 completion length, bounded by granted capacity."""

        maximum = max(1, int(maximum_tokens))
        prior = max(1, min(maximum, int(prior_tokens)))
        with self._lock:
            samples = self._shapes.get(shape.key())
            count = samples.completion_count if samples is not None else 0
            observed = (
                _percentile(samples.generated_tokens, 0.90)
                if samples is not None
                else 0.0
            )
        if count == 0:
            return prior, Confidence.NO_SAMPLES, 0
        confidence = (
            Confidence.MEASURED if count >= _MEASURED_SAMPLES else Confidence.SPARSE
        )
        predicted = max(prior, int(math.ceil(observed))) if confidence is Confidence.SPARSE else int(math.ceil(observed))
        return max(1, min(maximum, predicted)), confidence, count

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "shapes_measured": len(self._shapes),
                "total_samples": sum(s.count for s in self._shapes.values()),
                "shapes": {
                    key: {
                        "samples": samples.count,
                        "completion_samples": samples.completion_count,
                        "decode_s_per_token_p50": round(
                            _percentile(samples.decode_s_per_token, 0.50), 6
                        ),
                        "decode_s_per_token_p90": round(
                            _percentile(samples.decode_s_per_token, 0.90), 6
                        ),
                        "generated_tokens_p90": int(
                            _percentile(samples.generated_tokens, 0.90)
                        ),
                    }
                    for key, samples in self._shapes.items()
                },
            }


_ESTIMATOR: ThroughputEstimator | None = None
_ESTIMATOR_LOCK = checked_lock("measured_admission.estimator")


def get_throughput_estimator() -> ThroughputEstimator:
    global _ESTIMATOR
    with _ESTIMATOR_LOCK:
        if _ESTIMATOR is None:
            _ESTIMATOR = ThroughputEstimator()
            _publish_to_runtime_registry(_ESTIMATOR)
        return _ESTIMATOR


def recommended_foreground_deadline(
    *,
    model: str,
    prompt_tokens: int,
    decode_tokens: int,
    minimum_seconds: float,
    maximum_seconds: float,
) -> tuple[float, Confidence, int]:
    """Bound a foreground window by observed p90 throughput.

    The caller may not yet know whether its exact prefix will hit the KV cache.
    Plan against the slower warm/cold prediction rather than optimistically
    choosing one.  Natural EOS still returns early; this is permission to
    finish, not a sleep or a target duration.
    """

    estimator = get_throughput_estimator()
    predictions: dict[bool, tuple[float, Confidence, int]] = {}
    for cache_warm in (False, True):
        predictions[cache_warm] = estimator.predict_seconds(
            TaskShape(
                model=str(model or "unknown"),
                prompt_bucket=TaskShape.bucket_for(prompt_tokens),
                cache_warm=cache_warm,
                foreground=True,
            ),
            prompt_tokens=max(1, int(prompt_tokens)),
            decode_tokens=max(1, int(decode_tokens)),
        )
    # Cold-cache behavior is an upper-bound shape. Once it has observations,
    # an unmeasured warm bucket must not keep the static prior authoritative
    # forever. Conversely, warm-only evidence cannot prove cold behavior, so
    # retain the cold prior until a cold generation has completed.
    cold_prediction = predictions[False]
    if cold_prediction[1] is Confidence.NO_SAMPLES:
        candidates = list(predictions.values())
    else:
        candidates = [
            prediction
            for prediction in predictions.values()
            if prediction[1] is not Confidence.NO_SAMPLES
        ]
    predicted, confidence, samples = max(candidates, key=lambda item: item[0])
    # Admission itself reserves 15% for delivery. Invert that margin here and
    # retain four seconds for the HTTP projection and receipt binding.
    needed = (predicted / max(0.01, 1.0 - _MIN_MARGIN)) + 4.0
    deadline = max(float(minimum_seconds), min(float(maximum_seconds), needed))
    return deadline, confidence, samples


def recommended_completion_tokens(
    *,
    model: str,
    prompt_tokens: int,
    maximum_tokens: int,
    prior_tokens: int,
) -> tuple[int, Confidence, int]:
    """Plan expected EOS length from observed foreground generations."""

    estimator = get_throughput_estimator()
    predictions = [
        estimator.predict_completion_tokens(
            TaskShape(
                model=str(model or "unknown"),
                prompt_bucket=TaskShape.bucket_for(prompt_tokens),
                cache_warm=cache_warm,
                foreground=True,
            ),
            maximum_tokens=maximum_tokens,
            prior_tokens=prior_tokens,
        )
        for cache_warm in (False, True)
    ]
    measured = [item for item in predictions if item[1] is not Confidence.NO_SAMPLES]
    candidates = measured or predictions
    return max(candidates, key=lambda item: item[0])


def _publish_to_runtime_registry(estimator: ThroughputEstimator) -> None:
    """Make the estimator readable from the runtime layer.

    core/runtime may not import core.brain — that rule is why the
    foundation can boot and report on a mind that failed to start. So the
    health surface cannot reach in here; the estimator reaches out.
    """
    try:
        from core.runtime.service_registry import register_runtime_service

        register_runtime_service(
            "admission_throughput_estimator",
            estimator,
            required=False,
            owner="core/brain/llm/measured_admission.py",
            registered_by="get_throughput_estimator",
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "measured_admission",
            exc,
            severity="debug",
            action=(
                "admission throughput will not appear on the health surface; "
                "admission itself is unaffected"
            ),
        )


def record_generation(
    *,
    model: str,
    prompt_tokens: int,
    generated_tokens: int,
    prefill_seconds: float,
    decode_seconds: float,
    overhead_seconds: float = 0.0,
    cache_warm: bool = False,
    foreground: bool = True,
    completion_observed: bool = True,
) -> None:
    """Report a completed generation. Never raises into the response path."""
    try:
        get_throughput_estimator().record(
            TaskShape(
                model=str(model or "unknown"),
                prompt_bucket=TaskShape.bucket_for(prompt_tokens),
                cache_warm=bool(cache_warm),
                foreground=bool(foreground),
            ),
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            overhead_seconds=overhead_seconds,
            completion_observed=completion_observed,
        )
    except (ArithmeticError, TypeError, ValueError) as exc:
        record_degradation(
            "measured_admission",
            exc,
            severity="warning",
            action="dropped one throughput sample; admission falls back to a wider interval",
        )

    # The same counts, on the standard OpenTelemetry GenAI histograms.
    #
    # `core/observability/genai_semconv.py` defined gen_ai.client.token.usage
    # and a recorder for it, and nothing ever called that recorder — so the
    # only place in the process that knew a real token count was this
    # function, and it kept the number for admission and told no one. That is
    # the same half-wired shape as a writer with no reader, inverted: a reader
    # with no writer.
    #
    # Recorded here rather than at the model adapters because every lane that
    # completes a generation already reports it here, so one call site covers
    # them all and none can be forgotten.
    try:
        from core.observability.genai_semconv import record_token_usage

        record_token_usage(input_tokens=prompt_tokens, output_tokens=generated_tokens)
    except (ArithmeticError, AttributeError, ImportError, TypeError, ValueError) as exc:
        record_degradation(
            "measured_admission",
            exc,
            severity="debug",
            action="completed the generation without emitting its token-usage metric",
        )


def admit(
    *,
    model: str,
    prompt_tokens: int,
    requested_decode_tokens: int,
    deadline_seconds: float,
    foreground: bool = True,
    cache_warm: bool = False,
    model_loading: bool = False,
    queue_depth: int = 0,
    free_memory_bytes: int | None = None,
    model_footprint_bytes: int | None = None,
    foreground_lease_held_by_other: bool = False,
) -> AdmissionDecision:
    """Decide whether this request can finish, BEFORE it costs anything.

    The order of the refusals is the policy, and each one is a real
    incident that reached a person:

    1. The model is loading — admitting decode work now is what turned one
       slow answer into a cluster of 503s, because the next request timed
       out during the reload and killed the loading worker again.
    2. Memory cannot hold the request beside the resident model.
    3. Background work when the foreground holds the lease: it takes
       measured slack or it waits. It does not compete and get classified
       afterwards.
    4. The prediction does not fit the deadline — downgrade if a smaller
       answer fits, refuse if nothing does.
    """
    deadline_seconds = max(0.0, float(deadline_seconds or 0.0))
    requested_decode_tokens = max(0, int(requested_decode_tokens or 0))
    prompt_tokens = max(0, int(prompt_tokens or 0))

    shape = TaskShape(
        model=str(model or "unknown"),
        prompt_bucket=TaskShape.bucket_for(prompt_tokens),
        cache_warm=bool(cache_warm),
        foreground=bool(foreground),
    )
    estimator = get_throughput_estimator()
    predicted, confidence, samples = estimator.predict_seconds(
        shape, prompt_tokens=prompt_tokens, decode_tokens=requested_decode_tokens
    )

    def decide(
        outcome: AdmissionOutcome, reason: str, tokens: int, **detail: Any
    ) -> AdmissionDecision:
        return AdmissionDecision(
            outcome=outcome,
            confidence=confidence,
            reason=reason,
            granted_decode_tokens=max(0, int(tokens)),
            predicted_seconds=predicted,
            deadline_seconds=deadline_seconds,
            samples=samples,
            detail=detail,
        )

    if requested_decode_tokens <= 0:
        return decide(AdmissionOutcome.REFUSE, "no_decode_tokens_requested", 0)

    if model_loading:
        return decide(
            AdmissionOutcome.REFUSE,
            "model_is_loading",
            0,
            remedy="wait for ready; queueing decode against a loading worker is "
            "what turns one slow answer into a cluster of failures",
        )

    if free_memory_bytes is not None and model_footprint_bytes is not None:
        # The resident model must survive this request. Refusing here is
        # cheaper than an OOM that unloads ~20GB and reloads it.
        headroom = int(free_memory_bytes) - int(model_footprint_bytes)
        if headroom < 0:
            return decide(
                AdmissionOutcome.REFUSE,
                "insufficient_memory_beside_resident_model",
                0,
                free_bytes=int(free_memory_bytes),
                needed_bytes=int(model_footprint_bytes),
            )

    if not foreground and foreground_lease_held_by_other:
        return decide(
            AdmissionOutcome.REFUSE,
            "background_work_yields_to_the_foreground_lease",
            0,
            remedy="background cognition consumes measured slack; it does not "
            "compete with a user turn and get classified afterwards",
        )

    if deadline_seconds <= 0.0:
        return decide(
            AdmissionOutcome.REFUSE, "no_deadline_declared", 0,
            remedy="admission needs the caller's real deadline; guessing one is "
            "how fifteen seconds of work got allocated to a ten-second caller",
        )

    # Queue depth is real latency the caller will experience but this
    # request does not spend. Counted against the deadline, not ignored.
    queued_ahead = max(0, int(queue_depth)) * predicted
    usable = deadline_seconds * (1.0 - _MIN_MARGIN) - queued_ahead
    if usable <= 0.0:
        return decide(
            AdmissionOutcome.REFUSE,
            "queue_alone_exceeds_the_deadline",
            0,
            queue_depth=int(queue_depth),
        )

    if predicted <= usable:
        return decide(AdmissionOutcome.ADMIT, "fits_within_deadline", requested_decode_tokens)

    # Does not fit. Work out what would, from the measured decode rate
    # rather than by scaling the request down arbitrarily.
    per_token = predicted / max(1, requested_decode_tokens)
    affordable = int(usable / per_token) if per_token > 0 else 0
    if affordable >= 32:
        return decide(
            AdmissionOutcome.DOWNGRADE,
            "reduced_to_fit_the_deadline",
            affordable,
            requested=requested_decode_tokens,
        )

    return decide(
        AdmissionOutcome.REFUSE,
        "cannot_finish_within_the_deadline_even_reduced",
        0,
        affordable_tokens=affordable,
    )


def measured_slack_seconds(
    *, deadline_seconds: float, foreground_lease_held_by_other: bool, queue_depth: int
) -> float:
    """How much time background cognition may actually take right now.

    Zero while the foreground holds the lease. This is the number
    background work should ask for BEFORE starting, rather than starting
    and being classified as contention afterwards.
    """
    if foreground_lease_held_by_other:
        return 0.0
    slack = max(0.0, float(deadline_seconds or 0.0)) * (1.0 - _MIN_MARGIN)
    return slack / max(1, int(queue_depth) + 1)
