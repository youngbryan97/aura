"""What the client writes down about a generation.

Seven methods that take a worker response or a job event and record it —
throughput, interoception, a suppressed draft, a surface-control receipt, a
degraded event. They are about the RECORD, not about the generation, and
they change when a channel or a receipt changes rather than when decoding
does.


Lifted whole out of `mlx_client`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


class _RecordsWhatTheWorkerDid:
    """Lifted whole out of MLXLocalClient; see mlx_client.py."""

    def _record_throughput_sample(
        self,
        response: dict[str, Any],
        *,
        prompt: Any = None,
        foreground_request: bool = True,
    ) -> None:
        """Feed one real generation into the admission estimator.

        This is what makes admission MEASURED rather than a formula: the
        allocator's coefficients were hand-chosen with no model-specific
        calibration, so it could only ask "is this number allowable", never
        "can this finish on this machine as it is right now".

        Deliberately cheap and total. It runs on the completion path of
        every generation, so it does its own arithmetic, catches its own
        mistakes, and never lets a bad sample reach the caller — a
        telemetry write must not be able to fail a turn that worked.
        """
        from .mlx_client import (
            _BIG_ENOUGH_TO_TIME_TOKENS,
            _HOST_PREFILL_TPS,
            _HOST_RATES,
            _model_rate_key,
            _record_mlx_degradation,
            logger,
            record_generation,
        )

        try:
            started = float(getattr(self, "_current_request_started_at", 0.0) or 0.0)
            if started <= 0.0:
                return
            elapsed = max(0.0, time.time() - started)
            generated = max(0, int(response.get("tokens_used") or 0))
            if generated <= 0 or elapsed <= 0.0:
                return
            tokenization = response.get("prompt_tokenization")
            prompt_tokens = max(
                1,
                int(
                    (tokenization or {}).get("tokens")
                    if isinstance(tokenization, dict)
                    else 0
                )
                or len(str(prompt or "")) // 4,
            )
            # The worker reports prompt-cache retention, so a generation that
            # kept a cache is a WARM sample. Mixing warm and cold makes both
            # predictions wrong, which is why the shape carries it.
            cache_warm = bool(int(response.get("prompt_cache_reused_tokens") or 0) > 0)
            performance = response.get("generation_performance")
            exact_prefill = None
            exact_decode = None
            if isinstance(performance, dict):
                try:
                    measured_prefill = float(performance.get("prefill_seconds"))
                    measured_decode = float(performance.get("decode_seconds"))
                    if math.isfinite(measured_prefill) and measured_prefill >= 0.0:
                        exact_prefill = measured_prefill
                    if math.isfinite(measured_decode) and measured_decode > 0.0:
                        exact_decode = measured_decode
                except (TypeError, ValueError, OverflowError) as exc:
                    logger.debug("Prefill and decode seconds are not numbers, recording no sample: %s", exc)
                # Onto the turn's receipt, from the worker's own clock, so a
                # turn can be read back as where its time went and not only
                # as a rate the next deadline is built from.
                try:
                    from core.verify.turn_receipt import record_latency

                    if exact_prefill is not None:
                        record_latency("prefill", exact_prefill)
                    if exact_decode is not None:
                        record_latency("decode", exact_decode)
                except (ImportError, ValueError) as exc:
                    logger.debug("Turn latency not recorded: %s", exc)
                # The rate every deadline is built from. MLX timed this
                # inside the worker; the estimate this side keeps times how
                # often it was told, which is a different quantity and was
                # wrong by a factor of ten. See _measured_prefill_rate.
                try:
                    reported_tps = float(performance.get("prompt_tps") or 0.0)
                except (TypeError, ValueError, OverflowError) as exc:
                    logger.debug("Reported prompt_tps is not a number: %s", exc)
                    reported_tps = 0.0
                try:
                    measured_over = int(performance.get("prompt_tokens") or 0)
                except (TypeError, ValueError, OverflowError) as exc:
                    logger.debug("Reported prompt_tokens is not an integer: %s", exc)
                    measured_over = 0
                if measured_over < _BIG_ENOUGH_TO_TIME_TOKENS:
                    # Too small to be a rate. Setting a generation up costs
                    # the same whether it reads one token or a thousand, so
                    # over a short prompt that fixed cost IS the measurement.
                    #
                    # LIVE, 2026-09-08: the readiness probes send one and four
                    # token prompts, and MLX honestly reports 2.7 and 13.3
                    # tokens a second for them. Averaged into the rate the
                    # deadlines are built from, they took a 27B that reads at
                    # 116 down to single digits, and the answer clock then
                    # said a 9,360-character prompt would take 819 seconds to
                    # read and sized the turn at 1,022. The same reasoning is
                    # already written down one module over, where the read
                    # rate refuses to learn from a prompt under 400 characters.
                    reported_tps = 0.0
                if math.isfinite(reported_tps) and reported_tps > 0.0:
                    held = float(
                        getattr(self, "_worker_measured_prefill_tps", 0.0) or 0.0
                    )
                    # Averaged, so one generation under contention does not
                    # become the rule, and unaveraged for the first, so a
                    # fresh worker is not judged by a number nobody took.
                    self._worker_measured_prefill_tps = (
                        reported_tps
                        if held <= 0.0
                        else held * 0.7 + reported_tps * 0.3
                    )
                    _HOST_RATES["prefill"] = self._worker_measured_prefill_tps
                    key = _model_rate_key(getattr(self, "model_path", ""))
                    if key:
                        _HOST_PREFILL_TPS[key] = self._worker_measured_prefill_tps
            # Old workers do not report the split. Keep the bounded fallback
            # for rolling compatibility, but never overwrite MLX's measured
            # prompt and decode clocks with an estimate when they are present.
            prefill = (
                exact_prefill
                if exact_prefill is not None
                else min(elapsed * 0.25, prompt_tokens * 1.0e-3)
            )
            decode = (
                exact_decode
                if exact_decode is not None
                else max(1e-6, elapsed - prefill)
            )
            stop_reason = str(response.get("generation_stop_reason") or "").lower()
            surface_receipt = response.get("surface_control_receipt")
            semantic_complete = bool(
                isinstance(surface_receipt, dict)
                and surface_receipt.get("semantic_completion_satisfied") is True
            )
            completion_observed = semantic_complete or stop_reason in {
                "configured_stop",
                "eos",
                "semantic_contract_satisfied",
            }
            from .model_registry import runtime_model_measurement_key

            record_generation(
                model=runtime_model_measurement_key(self.model_path),
                prompt_tokens=prompt_tokens,
                generated_tokens=generated,
                prefill_seconds=prefill,
                decode_seconds=decode,
                cache_warm=cache_warm,
                foreground=bool(foreground_request),
                completion_observed=completion_observed,
            )
        except (ArithmeticError, AttributeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="skipped one admission throughput sample",
                severity="debug",
            )

    def _record_surface_control_receipt_from_response(self, response: dict[str, Any]) -> None:
        from .mlx_client import (
            _carry_decode_rate_across,
            _sanitize_surface_control_receipt,
            logger,
        )

        if isinstance(response, dict) and "prompt_cache_bytes" in response:
            try:
                self._last_prompt_cache_bytes = max(
                    0, int(response.get("prompt_cache_bytes") or 0)
                )
            except (TypeError, ValueError, OverflowError) as exc:
                logger.debug("prompt_cache_bytes is not an integer, leaving the last reading: %s", exc)
        receipt = _sanitize_surface_control_receipt(
            response.get("surface_control_receipt") if isinstance(response, dict) else None
        )
        if isinstance(response, dict) and "tokens_used" in response:
            try:
                receipt["generated_tokens"] = max(0, int(response.get("tokens_used") or 0))
            except (TypeError, ValueError, OverflowError) as exc:
                logger.debug("tokens_used is not an integer, leaving it off the receipt: %s", exc)
        if receipt:
            self._bind_surface_receipt_provenance(receipt, response)
        self._set_task_surface_control_receipt(receipt)
        if receipt:
            self._last_surface_control_receipt = receipt
            # The worker measures how fast it decodes; this process sizes the
            # deadline. Without carrying the number across, the deadline is
            # derived from the origin and the tier and can be shorter than the
            # budget the same request just computed.
            # Named where this client has an assignment, and unnamed where it
            # does not. `runtime_model_measurement_key("")` resolves to the
            # ACTIVE_MODEL alias, so passing an absent path would file the
            # brainstem's rate under the cortex — which is the very swap this
            # is here to stop. An unnamed reading is still read by every
            # caller that names no model.
            _bound = str(getattr(self, "model_path", "") or "")
            if _bound:
                from .model_registry import runtime_model_measurement_key

                _carry_decode_rate_across(
                    receipt, runtime_model_measurement_key(_bound)
                )
            else:
                _carry_decode_rate_across(receipt)

    def _record_suppressed_draft(
        self, text: str, reasons: tuple[str, ...] | list[str]
    ) -> None:
        """Put a gate-rejected draft on the bound turn, marked suppressed.

        Recoverable on purpose. These reasons are heuristics about SHAPE —
        "runtime_boilerplate", "too_thin_for_status_turn" — not findings of
        unsafety, and the recovery path only reaches for a suppressed draft
        when the turn would otherwise end with nothing. Between a draft a
        heuristic disliked and an apology that says nothing, the draft is the
        better answer, and the person can judge it themselves.

        Never raises: this is a salvage path, and a salvage path that can
        break the turn it is salvaging is worse than none.
        """
        from .mlx_client import (
            logger,
        )

        if not text:
            return
        try:
            from core.conversation.surface_disposition import UNSPEAKABLE_REASONS
            from core.runtime.turn_outcome import current_turn

            outcome = current_turn()
            if outcome is None:
                return
            bounded_reasons = [str(reason).strip()[:120] for reason in reasons if str(reason).strip()][:8]
            recoverable = not bool(set(bounded_reasons) & set(UNSPEAKABLE_REASONS))
            candidate_id = outcome.record_candidate(
                text,
                source="mlx_worker.surface_quality_rejected",
                metadata={"worker_quality_reasons": bounded_reasons},
            )
            outcome.suppress_candidate(
                candidate_id,
                gate="mlx_worker.surface_quality",
                reasons=bounded_reasons,
                recoverable=recoverable,
            )
        except Exception:  # noqa: BLE001 — salvage must never break the turn
            logger.debug("could not record suppressed worker draft", exc_info=True)

    def _record_interoception_from_response(
        self,
        response: dict[str, Any],
        *,
        foreground_request: bool,
        owner_label: str,
    ) -> None:
        """Capture the worker's felt-thought measurements and hand them to the
        thought-interoception organ. Observational only — never raises into the
        generation path."""
        from .mlx_client import (
            _bounded_interoception,
            _record_mlx_degradation,
            record_degradation,
        )

        try:
            payload = response.get("interoception") if isinstance(response, dict) else None
            if not isinstance(payload, dict) or not payload:
                return
            from core.being.thought_interoception import (
                get_thought_interoception,
                text_fingerprint,
            )

            # Fingerprint the payload to the text it measured, so consumers
            # (e.g. unified_inference feedback) can prove they are pairing the
            # right trace with the right response even under concurrent lanes.
            # Ingest FIRST: the engine is the validator. Storing beforehand
            # exposed malformed/unbounded worker data through the public
            # getter as the "most recent measurement" even when the engine
            # rejected it.
            #
            # CP126 093a2902, second half: ordering alone did not fix that,
            # because ingest() NEVER RAISES — it returns None for a payload it
            # dropped. The store ran regardless, so a rejected payload was
            # still published as the latest felt-thought measurement. Only an
            # accepted one is retained now, and only in bounded form.
            # CP126 0b3bbd3e: the trace and the response arrived as two
            # unrelated arguments, so under concurrent lanes a worker's
            # measurement could be filed against a different lane's answer.
            # The response carries the request id the worker echoed; handing
            # it over lets the organ PROVE the pairing rather than assume it.
            felt = get_thought_interoception().ingest(
                payload,
                origin=owner_label or "mlx",
                foreground=bool(foreground_request),
                response_text=str(response.get("text") or ""),
                generation_id=str(response.get("id") or ""),
            )
            if felt is None:
                _record_mlx_degradation(
                    ValueError("interoception payload rejected by the felt-thought engine"),
                    action="left the previous measurement in place rather than publishing a rejected payload",
                    severity="warning",
                )
                return
            stored = _bounded_interoception(payload)
            stored["_text_fingerprint"] = text_fingerprint(str(response.get("text") or ""))
            self._last_interoception = stored
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            record_degradation(
                "mlx_client_interoception",
                exc,
                severity="warning",
                action="continued generation return after interoception ingest failed",
            )

    def _record_worker_job_activity(self, payload: Mapping[str, Any]) -> None:
        """Accept request-bound inference activity from a worker heartbeat.

        Progress and terminal frames are consumed on the parent event loop. The
        heartbeat is produced by a separate worker thread and carries the model
        loop's own activity age. Reconstructing the worker-side event time keeps
        parent-loop starvation from turning queued progress into a false token
        stall, without treating process liveness as model progress.
        """
        from .mlx_client import (
            logger,
        )


        if payload.get("active_job") is not True:
            return
        request_id = str(payload.get("request_id") or "")
        current_request_id = str(getattr(self, "_current_request_id", "") or "")
        if not request_id or request_id != current_request_id:
            return
        try:
            emitted_at = float(payload.get("timestamp") or 0.0)
            activity_age_s = float(
                payload.get("job_progress_age_s", payload.get("job_age_s", -1.0))
            )
        except (TypeError, ValueError, OverflowError) as exc:
            logger.debug("Worker job timestamps are not numbers, recording no activity: %s", exc)
            return
        now = time.time()
        if (
            not math.isfinite(emitted_at)
            or not math.isfinite(activity_age_s)
            or emitted_at <= 0.0
            or emitted_at > now + 5.0
            or activity_age_s < 0.0
        ):
            return
        activity_at = emitted_at - activity_age_s
        request_started_at = float(
            getattr(self, "_current_request_started_at", 0.0) or 0.0
        )
        if request_started_at > 0.0 and activity_at < request_started_at - 1.0:
            return
        self._last_worker_job_activity_at = max(
            float(getattr(self, "_last_worker_job_activity_at", 0.0) or 0.0),
            activity_at,
        )

    def _record_degraded_event(
        self,
        reason: str,
        *,
        detail: str = "",
        severity: str = "warning",
        foreground_request: bool = False,
        classification: str | None = None,
    ) -> None:
        from .mlx_client import (
            _record_mlx_degradation,
            logger,
        )

        try:
            from core.health.degraded_events import record_degraded_event

            record_degraded_event(
                "mlx_client",
                reason,
                detail=detail,
                severity=severity,
                classification=self._classify_failure(
                    foreground_request=foreground_request,
                    reason=f"{reason}:{detail}",
                    classification=classification,
                ),
                context={
                    "model": os.path.basename(self.model_path),
                    "lane_state": self._lane_state,
                    "warmup_in_flight": self._warmup_in_flight,
                },
            )
        except Exception as exc:  # noqa: BLE001 - observation must not break the observed
            # This runs inside generation, warmup and reboot paths purely to
            # WRITE a diagnostic. The old clause caught three error types, so a
            # serialization or disk failure inside record_degraded_event
            # escaped and failed the very operation it was reporting on —
            # turning "we noticed something degraded" into an outage.
            try:
                _record_mlx_degradation(
                    exc,
                    action="kept lane-local degraded state after health event emission failed",
                )
            except Exception as fallback_exc:  # noqa: BLE001 - fail-closed fallback
                # Nothing further to try: the primary recorder and its
                # fallback have both failed. Swallowing it silently would make
                # a blind observability path indistinguishable from a quiet
                # one, so the loss is named even though it cannot be acted on.
                logger.debug(
                    "MLX degradation fallback also failed (%s: %s); the original "
                    "event is unrecorded.",
                    type(fallback_exc).__name__,
                    fallback_exc,
                )
            logger.debug("Failed to record MLX degraded event: %s", exc)

    def _record_worker_stream_progress(
        self,
        res: dict[str, Any],
        *,
        status: str | None,
        action: str | None,
    ) -> None:
        """Classify worker activity without mistaking compute for decoded output."""
        if action == "latent_reason":
            self._record_latent_progress(res)
        if status == "token":
            self._mark_token_progress(
                res.get("id"), generated_tokens=res.get("tokens_generated")
            )
        elif res.get("phase") == "prefill":
            # Reading the prompt is the model working on this request, and on
            # a long one it is the larger half. Counting only decoded tokens
            # made a turn look silent for the whole of it, so a wait that
            # defers to progress gave up during the one part of the turn where
            # nothing could have arrived yet.
            self._mark_prefill_progress(
                res.get("id"),
                processed=res.get("prompt_tokens_processed", 0),
                total=res.get("prompt_tokens_total", 0),
            )
        elif action == "latent_reason":
            # Branch selection proves liveness but is not decoded output. Treating
            # it as a token switches a healthy request onto the shorter token gap.
            self._mark_progress()
        elif res.get("tokens_generated") is not None:
            # A decoded token can be temporarily textless in the detokenizer.
            self._mark_token_progress(
                res.get("id"), generated_tokens=res.get("tokens_generated")
            )
        else:
            self._mark_progress()

