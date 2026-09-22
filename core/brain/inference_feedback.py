"""core/brain/inference_feedback.py
==================================
Closes the loop between LLM inference outputs and homeostatic states.
Calculates surprise (perplexity/log-prob based) and coherence, updates
the FreeEnergyEngine, LiquidSubstrate, and trains the logit projection.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np

from core.brain.homeostatic_modulator import InferenceModulation
from core.cognitive.sentiment_tracker import analyze_text_sentiment
from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service

logger = logging.getLogger("Aura.Brain.InferenceFeedback")

_FEEDBACK_RECOVERABLE_ERRORS = (
    AttributeError,
    TypeError,
    ValueError,
    RuntimeError,
    OSError,
    LookupError,
    TimeoutError,
)


def _record_feedback_degradation(exc: BaseException, *, action: str, severity: str = "warning") -> None:
    record_degradation("inference_feedback", exc, severity=severity, action=action)


def _finite(value: Any, default: float | None = None) -> float | None:
    """A finite float, or ``default`` when the value is not one.

    CP126 88125a35 / 5ba7dc20. NaN and infinity survived ``np.mean`` and
    ``np.clip`` — clip does not remove NaN, it propagates it — and reached
    the free-energy engine, the liquid substrate, the precision engine and
    the projection's learning rate. A NaN learning rate does not error; it
    quietly turns every subsequent weight into NaN.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def _finite_sequence(values: Any) -> list[float] | None:
    """Every element finite, or None. One bad token poisons the mean."""
    try:
        candidates = list(values)
    # not a failure: something that will not iterate is not a sequence of values.
    except TypeError:
        return None
    out: list[float] = []
    for item in candidates:
        number = _finite(item)
        if number is None:
            return None
        out.append(number)
    return out


def _modulation_fingerprint(modulation: Any) -> dict[str, Any]:
    """The controls that produced this generation, for attribution.

    CP126 150a1259: the modulation was accepted and never read, so feedback
    and projection learning could not attribute an outcome to temperature,
    sampling or substrate controls — nothing could learn whether a
    modulation HELPED. Recording it does not close that loop by itself, but
    it is the datum without which the loop cannot exist.
    """
    if modulation is None:
        return {}
    out: dict[str, Any] = {}
    for field in ("temperature", "top_p", "top_k", "repetition_penalty", "max_tokens"):
        value = getattr(modulation, field, None)
        number = _finite(value)
        if number is not None:
            out[field] = number
    return out


def _generation_identity(output_text: str, token_ids: Any) -> str:
    """A stable id for one generation, so a retry cannot double-dose.

    Derived from content rather than supplied, so callers that have no id
    still get the protection and an identical retry deduplicates.
    """
    digest = hashlib.sha256()
    digest.update(str(output_text or "").encode("utf-8", errors="replace"))
    try:
        digest.update(",".join(str(int(t)) for t in (token_ids or ())).encode("ascii"))
    except (TypeError, ValueError):
        digest.update(b"unhashable_token_ids")
    return digest.hexdigest()[:32]


def _read_substrate_channel(vector: Any, index: Any) -> float | None:
    """One channel of the substrate vector, or None if it cannot be read.

    CP126 fb465f7c: index checks rejected only values ABOVE the vector
    length, so a negative index silently read from the end of the vector
    and returned some unrelated channel as valence.
    """
    try:
        position = int(index)
    # not a failure: a value that is not a number is not one this can read.
    except (TypeError, ValueError):
        return None
    if position < 0 or position >= len(vector):
        return None
    return _finite(vector[position])


class InferenceFeedbackLoop:
    """Computes feedback signals from LLM inference outputs and feeds them back into
    the homeostatic substrate and free energy engine.
    """

    def __init__(self, substrate_dim: int = 512) -> None:
        # A genuine int, not merely something int-like. "512" and 512.0
        # both parse, and both mean a caller passed the wrong type — which
        # is worth surfacing here rather than as a shape mismatch deep in
        # the projection (CP126 fb465f7c).
        if isinstance(substrate_dim, bool) or not isinstance(substrate_dim, int):
            raise ValueError(
                f"substrate_dim must be an int, got {type(substrate_dim).__name__}"
            )
        if substrate_dim <= 0:
            raise ValueError(f"substrate_dim must be positive, got {substrate_dim}")
        self.substrate_dim = substrate_dim
        self._applied: list[str] = []

    #: Bounded ring of generation ids already applied to the engines. Bounded
    #: because a long-lived process must not accumulate one entry per turn
    #: forever; a retry arrives immediately after its original, so a short
    #: memory is sufficient to stop double-dosing.
    _APPLIED_RING = 512

    def _claim_generation(self, generation_id: str) -> bool:
        """True the first time this generation is seen, False on a repeat."""
        applied = getattr(self, "_applied", None)
        if applied is None:
            applied = []
            self._applied = applied
        if generation_id in applied:
            return False
        applied.append(generation_id)
        while len(applied) > self._APPLIED_RING:
            applied.pop(0)
        return True

    def _feed(self, service: str, apply: Any, receipt: dict[str, Any]) -> str:
        engine = get_runtime_service(service, default=None)
        if engine is None:
            return "unavailable"
        return self._feed_instance(service, engine, apply, receipt)

    def _feed_instance(
        self, service: str, engine: Any, apply: Any, receipt: dict[str, Any]
    ) -> str:
        """Apply one engine update and report whether it actually landed."""
        try:
            apply(engine)
            return "applied"
        except _FEEDBACK_RECOVERABLE_ERRORS as exc:
            _record_feedback_degradation(
                exc, action=f"continued response after {service} feedback injection failed"
            )
            receipt["degraded"].append(f"{service}_failed:{type(exc).__name__}")
            return f"failed:{type(exc).__name__}"


    def process_output(
        self,
        output_text: str,
        token_ids: list[int],
        logprobs: list[float] | None,
        modulation: InferenceModulation,
        modulator_projection: Any,
        feed_engines: bool = True,
    ) -> dict[str, Any]:
        """Process completed LLM response and update the homeostatic engines.

        Args:
            output_text: The string response from the model.
            token_ids: List of vocabulary token IDs that were produced.
            logprobs: Log probabilities for the generated tokens.
            modulation: The modulation configuration that was applied to this run.
            modulator_projection: The SubstrateLogitProjection instance to train.
            feed_engines: When False, skip the free-energy / liquid-substrate /
                precision-engine injections and only compute metrics + train the
                projection. Callers pass False when the thought-interoception
                organ has already fed those engines for this generation, so the
                same response never doses the substrate twice.

        Returns:
            Dictionary of calculated metrics: surprise, coherence, etc.
        """
        receipt: dict[str, Any] = {
            "generation_id": "",
            "token_count": len(token_ids or ()),
            "logprobs_available": bool(logprobs),
            "substrate_available": False,
            "engines_fed": {},
            "projection_trained": False,
            "degraded": [],
            # The modulation that produced this generation, recorded so an
            # outcome can be attributed to the controls that caused it
            # (CP126 150a1259: it was accepted as an argument and never read,
            # so nothing could learn whether a modulation helped).
            "modulation": _modulation_fingerprint(modulation),
        }

        # Once-only. CP126 1c32e107: four engines were dosed independently
        # with no generation identity, so a retried response applied the same
        # surprise and coherence twice — and a failure partway left some
        # engines updated and others not, with nothing recording which.
        generation_id = _generation_identity(output_text, token_ids)
        receipt["generation_id"] = generation_id
        if feed_engines and not self._claim_generation(generation_id):
            receipt["engines_fed"] = {"skipped": "already_applied"}
            feed_engines = False

        # 1. Surprise.
        #
        # CP126 d6fb5316: this is the MEAN NEGATIVE LOG PROBABILITY, clipped
        # to [0, 3]. It is not perplexity, and the clip discards every
        # distinction above 3.0, so a wildly surprised generation and a
        # merely surprised one arrive identical. The method is now reported
        # so a consumer knows which scale it is reading and never compares
        # this to a token perplexity.
        surprise_method = "unavailable"
        surprise = 0.0
        checked_logprobs = _finite_sequence(logprobs) if logprobs else None
        if checked_logprobs:
            mean_logprob = _finite(np.mean(checked_logprobs))
            if mean_logprob is not None:
                surprise = float(np.clip(-mean_logprob, 0.0, 3.0))
                surprise_method = "mean_negative_logprob_clipped_0_3"
                receipt["surprise_clipped"] = bool(-mean_logprob > 3.0)
        if surprise_method == "unavailable":
            if logprobs:
                receipt["degraded"].append("logprobs_not_finite")
            # Lexical fallback. CP126 e13e961c: the docstring promised
            # "length and punctuation volatility" and the code examined
            # neither — only case-sensitive whole-word uniqueness, which
            # makes repetitive boilerplate look surprising. It now measures
            # what it claims, on normalised words.
            words = [word.lower() for word in output_text.split()]
            unique_ratio = len(set(words)) / max(1, len(words))
            punctuation = sum(1 for char in output_text if char in ".,;:!?—-()[]\"'")
            punctuation_ratio = punctuation / max(1, len(output_text))
            repetition = 1.0 - unique_ratio
            surprise = float(np.clip(repetition + punctuation_ratio + 0.2, 0.1, 1.5))
            surprise_method = "lexical_repetition_and_punctuation"
        receipt["surprise_method"] = surprise_method

        # 2. Coherence with substrate state.
        substrate = get_runtime_service("liquid_substrate", default=None)
        valence: float | None = None
        arousal: float | None = None
        substrate_state: Any = None

        if substrate is not None:
            try:
                with substrate.sync_lock:
                    vector = substrate.x
                    substrate_state = vector.copy()
                    valence = _read_substrate_channel(vector, substrate.idx_valence)
                    arousal = _read_substrate_channel(vector, substrate.idx_arousal)
                if substrate_state is not None and len(substrate_state) != self.substrate_dim:
                    receipt["degraded"].append(
                        f"substrate_dim_mismatch:{len(substrate_state)}!={self.substrate_dim}"
                    )
                    substrate_state = None
            except _FEEDBACK_RECOVERABLE_ERRORS as exc:
                # CP126 af290c63: service resolution, lock acquisition, vector
                # copying and index access all ran outside any handler, so a
                # missing lock or malformed vector aborted response
                # finalisation instead of degrading the feedback receipt.
                _record_feedback_degradation(
                    exc, action="computed inference feedback without substrate state"
                )
                substrate_state, valence, arousal = None, None, None
                receipt["degraded"].append(f"substrate_read_failed:{type(exc).__name__}")

        substrate_available = substrate_state is not None and valence is not None
        receipt["substrate_available"] = bool(substrate_available)

        # Semantic output valence.  This is local-only and provenance-bearing:
        # the on-device learned analyzer handles compositional language and the
        # portable fallback handles negation, intensity, contrast and sarcasm.
        # Unsupported text abstains, so it cannot become durable projection
        # learning merely because a keyword happened to occur.
        sentiment = analyze_text_sentiment(output_text)
        output_valence = sentiment.valence
        receipt["sentiment_evidence"] = sentiment.as_dict()
        receipt["valence_evidence_hits"] = sentiment.evidence_units
        receipt["output_valence_grounded"] = sentiment.grounded

        # CP126 3c46ea8c, the critical one: when the substrate was absent
        # this substituted a zero vector, valence 0.0 and arousal 0.5, then
        # computed coherence from those invented numbers, TRAINED THE
        # PROJECTION on the synthetic vector, and returned substrate_valence
        # with no availability flag. Coherence came out 1.0 — perfect
        # alignment with a state nobody had observed.
        #
        # No substrate, no coherence. None, not 1.0, and no training.
        if substrate_available and sentiment.grounded:
            coherence = float(np.clip(((1.0 - abs(valence - output_valence)) * 2.0) - 1.0, -1.0, 1.0))
            coherence_grounded = True
        else:
            coherence = 0.0
            coherence_grounded = False
            receipt["degraded"].append(
                "coherence_ungrounded:no_substrate" if not substrate_available
                else "coherence_ungrounded:no_valence_evidence"
            )
        receipt["coherence_grounded"] = coherence_grounded

        # 3-4b. Engine injections, each recording whether it actually applied.
        #
        # SURPRISE and COHERENCE are grounded independently, and conflating
        # them would have been my own version of this file's bug. Surprise
        # comes from the model's own logprobs and is a real measurement with
        # or without a substrate; only coherence needs an observed state. So
        # the free-energy engine — which receives surprise alone — is fed
        # whenever surprise is measured, while the two engines that consume
        # coherence are not fed from a state nobody read.
        if feed_engines:
            surprise_grounded = surprise_method != "lexical_repetition_and_punctuation"
            if surprise_grounded:
                fe_surprise = float(np.clip(surprise / 3.0, 0.0, 1.0))
                receipt["engines_fed"]["free_energy_engine"] = self._feed(
                    "free_energy_engine",
                    lambda engine: engine.accept_surprise_signal(fe_surprise),
                    receipt,
                )
            else:
                receipt["engines_fed"]["free_energy_engine"] = "skipped:surprise_ungrounded"

            if coherence_grounded:
                if substrate is not None:
                    receipt["engines_fed"]["liquid_substrate"] = self._feed_instance(
                        "liquid_substrate",
                        substrate,
                        lambda engine: engine.accept_inference_feedback(
                            surprise=surprise, coherence=coherence
                        ),
                        receipt,
                    )
                receipt["engines_fed"]["precision_engine"] = self._feed(
                    "precision_engine",
                    lambda engine: engine.accept_inference_feedback(
                        surprise=surprise, coherence=coherence
                    ),
                    receipt,
                )
            else:
                receipt["engines_fed"]["coherence_consumers"] = (
                    "skipped:coherence_never_observed"
                )

        # 5. Train the projection — only on observed state.
        #
        # CP126 4803da5a: learn_step sat outside every handler and trusted
        # token ids, vector shape, coherence, surprise and the learning rate.
        # A failure here aborted the response AFTER other engines had already
        # been updated.
        if (
            modulator_projection is not None
            and token_ids
            and substrate_available
            and coherence_grounded
        ):
            # Arousal scales the learning rate directly, so an unbounded or
            # NaN arousal reverses or explodes online learning.
            safe_arousal = min(1.0, max(0.0, arousal if arousal is not None else 0.5))
            learning_rate = 0.002 * (1.0 + safe_arousal)
            try:
                modulator_projection.learn_step(
                    substrate_state=substrate_state,
                    token_ids=token_ids,
                    feedback_coherence=coherence,
                    surprise=surprise,
                    lr=learning_rate,
                )
                receipt["projection_trained"] = True
                receipt["learning_rate"] = learning_rate
            except _FEEDBACK_RECOVERABLE_ERRORS as exc:
                _record_feedback_degradation(
                    exc, action="served the response without training the logit projection"
                )
                receipt["degraded"].append(f"projection_train_failed:{type(exc).__name__}")
        elif modulator_projection is not None and token_ids:
            receipt["degraded"].append("projection_untrained:no_observed_substrate_state")

        return {
            "surprise": surprise,
            "coherence": coherence,
            "output_valence": output_valence,
            # None, not 0.0, when nothing was observed. CP126 f2e4b67a: a
            # caller could not tell a calculation from successfully applied
            # feedback, so the receipt now says which engines took it.
            "substrate_valence": valence,
            **receipt,
        }
