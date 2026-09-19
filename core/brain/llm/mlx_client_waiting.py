"""Waiting on a generation, and the progress marks that keep it alive.

Lifted whole out of `mlx_client`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .mlx_client import (
        Deadline,
        SharedFuture,
    )


class _WaitsForTheResult:
    """Lifted whole out of MLXLocalClient; see mlx_client.py."""

    def _mark_progress(self) -> None:
        from .mlx_client import (
            note_foreground_owner_progress,
        )

        self._last_progress_at = time.time()
        # The foreground lease's staleness is measured from PROGRESS, not from
        # acquisition, so every client-level progress mark is also a heartbeat
        # for whoever currently owns the foreground (CP126 6595b0e1).
        note_foreground_owner_progress()


    def _mark_generation_started(
        self,
        req_id: str,
        *,
        prompt_chars: int = 0,
        requested_max_tokens: int = 0,
        first_token_hard_ceiling_s: float = 0.0,
        request_seq: int = 0,
    ) -> None:
        from core.runtime.turn_progress import capture_progress

        from .mlx_client import (
            _PREFILL_HEADROOM,
            logger,
        )

        self._current_turn_progress = capture_progress()
        from core.runtime.chat_delivery_progress import capture_generation_progress

        self._current_delivery_progress = capture_generation_progress()
        now = time.time()
        self._current_request_id = str(req_id or "")
        self._current_request_seq = max(0, int(request_seq or 0))
        # A new generation supersedes any stale cooperative-cancel request.
        cancel_seq = getattr(self, "_cancel_seq", None)
        if cancel_seq is not None and int(getattr(cancel_seq, "value", 0)) not in (
            0,
            self._current_request_seq,
        ):
            cancel_seq.value = 0
        self._current_request_progress_baseline_at = max(
            self._last_heartbeat,
            self._last_progress_at,
            self._last_ready_at,
        )
        self._current_request_started_at = now
        self._current_first_token_at = 0.0
        #: Tokens observed for THIS request. The worker reports a count on the
        #: generations that carry a receipt and not on the ones that do not,
        #: and an answer's authorship is proven from that count.
        self._tokens_this_request = 0
        self._current_prompt_chars = max(0, int(prompt_chars or 0))
        self._current_requested_max_tokens = max(0, int(requested_max_tokens or 0))
        self._last_token_progress_at = 0.0
        self._last_worker_job_activity_at = 0.0
        self._current_request_prompt_chars = max(0, int(prompt_chars or 0))
        self._current_first_token_hard_ceiling_s = max(
            0.0,
            float(first_token_hard_ceiling_s or 0.0),
        )
        # A ceiling that does not account for the prompt is not a budget.
        #
        # LIVE 2026-08-26, two lines apart: "first-token ceiling 90.0s for a
        # 5-char prompt" and "first-token ceiling 4.0s for a 3431-char
        # prompt". Ninety seconds to read five characters, four to read nine
        # hundred tokens — the number had no relationship to the work, and
        # every decision she made while playing was cancelled by it. She chose
        # her moves from the consequence record alone and never held a plan,
        # which from outside looks exactly like a mind that is not thinking.
        #
        # The floor is the time THIS worker takes to read THIS prompt, at the
        # rate it has been measured at, with room for the queueing a shared
        # lane always has. It only ever raises a ceiling, never past the
        # livelock ceiling that catches a wedged worker.
        needed = min(
            self._prefill_floor_seconds(self._current_prompt_chars),
            # Never past the ceiling that catches a wedged worker. Said in
            # the comment before and not enforced in the code, which let one
            # bad rate reading ask for a ten-minute deadline.
            self._first_token_hard_ceiling(foreground_request=True),
        )
        if 0.0 < self._current_first_token_hard_ceiling_s < needed:
            logger.info(
                "⏱️ [MLX] first-token ceiling raised %.1fs → %.1fs: a %d-char prompt "
                "takes about %.1fs to read at %.0f tok/s",
                self._current_first_token_hard_ceiling_s,
                needed,
                self._current_prompt_chars,
                needed / _PREFILL_HEADROOM,
                self._measured_prefill_rate(),
            )
            self._current_first_token_hard_ceiling_s = needed
        # What the caller allowed, beside what the prompt will cost.
        #
        # A first-token ceiling is only meaningful next to the prompt it has
        # to read: 4 seconds is generous for a hundred tokens and impossible
        # for two thousand. Without both numbers in one line, a cancelled
        # request looks like a slow worker, and every decision she made while
        # playing was cancelled this way.
        if self._current_first_token_hard_ceiling_s > 0.0:
            logger.info(
                "⏱️ [MLX] first-token ceiling %.1fs for a %d-char prompt (%d max tokens)",
                self._current_first_token_hard_ceiling_s,
                self._current_prompt_chars,
                self._current_requested_max_tokens,
            )
        self._current_prefill_tokens_processed = 0
        self._current_prefill_tokens_total = 0
        self._prefill_observed_at = 0.0
        self._prefill_observed_tokens = 0
        self._mark_progress()

    def _mark_prefill_progress(
        self,
        req_id: str | None,
        *,
        processed: int,
        total: int,
    ) -> None:
        normalized_req_id = str(req_id or "")
        if (
            normalized_req_id
            and self._current_request_id
            and normalized_req_id != self._current_request_id
        ):
            return
        if not normalized_req_id and self._current_request_id:
            self._mark_progress()
            return
        done = max(0, int(processed or 0))
        # How fast this worker actually reads a prompt, from this worker.
        #
        # Measured rather than assumed, so a ceiling built on it is a ceiling
        # built on what the machine does. Averaged, so one slow chunk under
        # contention does not become the rule.
        now = time.time()
        # Between two observations of prefill, not since the request began.
        #
        # Measuring from the request start folds in queueing, admission and
        # whatever else happened before a single token was read: it reported
        # 4 tok/s on a worker doing 720, and asked for a ten-minute ceiling
        # on a prompt that takes a second and a half to read.
        last_at = float(getattr(self, "_prefill_observed_at", 0.0) or 0.0)
        last_done = int(getattr(self, "_prefill_observed_tokens", 0) or 0)
        if done > last_done and last_at > 0.0:
            spent = now - last_at
            if spent > 0.02:
                observed = (done - last_done) / spent
                previous = float(getattr(self, "_prefill_tokens_per_s", 0.0) or 0.0)
                self._prefill_tokens_per_s = (
                    observed if previous <= 0.0 else previous * 0.7 + observed * 0.3
                )
                # Not into _HOST_RATES. This is how often the parent was
                # TOLD about prefill, across an IPC queue onto a busy event
                # loop, and publishing it as the host's prefill rate is how a
                # 52,020-character prompt came to be budgeted at 1,082
                # seconds of reading. The host rate is set from what MLX
                # timed inside the worker.
        if done != last_done:
            self._prefill_observed_at = now
            self._prefill_observed_tokens = done
        if done > last_done and getattr(self, "_current_turn_progress", None) is not None:
            from core.runtime.turn_progress import note_progress

            note_progress(progress=self._current_turn_progress)
        delivery_progress = getattr(self, "_current_delivery_progress", None)
        if done > last_done and delivery_progress is not None:
            delivery_progress(phase="prefill", completed=done, total=max(0, int(total or 0)))
        self._current_prefill_tokens_processed = done
        self._current_prefill_tokens_total = max(0, int(total or 0))
        self._mark_progress()

    def _mark_token_progress(
        self, req_id: str | None = None, *, generated_tokens: int | None = None
    ) -> None:
        from .mlx_client import (
            _COLD_FIRST_TOKEN_S,
            _HOST_RATES,
            _PREFILL_HEADROOM,
            logger,
        )

        now = time.time()
        normalized_req_id = str(req_id or "")
        if (
            normalized_req_id
            and self._current_request_id
            and normalized_req_id != self._current_request_id
        ):
            return
        if not normalized_req_id and self._current_request_id:
            # Id-less progress cannot be ATTRIBUTED to the active request:
            # crediting it set first-token timestamps from unrelated or
            # malformed messages. It still proves the worker is alive.
            self._mark_progress()
            return
        previous_count = int(getattr(self, "_tokens_this_request", 0) or 0)
        if generated_tokens is None:
            delta = 1  # Legacy visible-token frames carry one token each.
        elif isinstance(generated_tokens, int) and not isinstance(generated_tokens, bool):
            delta = generated_tokens - previous_count
        else:
            delta = 0
        if delta <= 0:
            # Duplicate, reordered or malformed counters prove no new decoding.
            self._mark_progress()
            return
        delivery_progress = getattr(self, "_current_delivery_progress", None)
        if delivery_progress is not None:
            delivery_progress(phase="generating", completed=previous_count + delta)
        # Decode measured the same way prefill is: between two observations,
        # so queueing before the first token is not charged to writing.
        previous_at = float(getattr(self, "_last_token_progress_at", 0.0) or 0.0)
        if previous_at > 0.0 and self._current_first_token_at > 0.0:
            spent = now - previous_at
            if 0.005 < spent < 5.0:
                observed = delta / spent
                previous = _HOST_RATES["decode"]
                _HOST_RATES["decode"] = (
                    observed if previous <= 0.0 else previous * 0.8 + observed * 0.2
                )
        self._last_token_progress_at = now
        # Published where the layers above can read it. Five deadlines are
        # waiting on this one generation, and each of them was deciding
        # whether to end it from a stopwatch rather than from whether it was
        # still saying anything.
        try:
            from core.runtime.turn_progress import note_progress

            if getattr(self, "_current_turn_progress", None) is not None:
                note_progress(progress=self._current_turn_progress)
        except ImportError as exc:
            logger.debug("Turn progress unavailable, token progress not noted: %s", exc)
        if self._current_first_token_at <= 0.0:
            self._current_first_token_at = now
            # Request-to-token latency includes queueing and admission. Only
            # worker timings and advancing prefill frames measure reading.
            # What loading this model actually cost, from the one request
            # that pays for it. Everything before the first token of a
            # worker's life is weights coming off disk plus reading the
            # prompt; the prompt's share is already measured, so the rest is
            # the load. Averaged across workers, because the disk is one disk.
            if int(getattr(self, "_tokens_since_spawn", 0) or 0) == 0:
                gigabytes = self._weight_gigabytes()
                started = float(getattr(self, "_current_request_started_at", 0.0) or 0.0)
                spent = now - started if started > 0.0 else 0.0
                loading = spent - (
                    self._prefill_floor_seconds(self._current_prompt_chars)
                    / _PREFILL_HEADROOM
                )
                if gigabytes > 0.0 and loading > 0.1:
                    observed = gigabytes / loading
                    previous = float(_HOST_RATES.get("weight_load") or 0.0)
                    _HOST_RATES["weight_load"] = (
                        observed if previous <= 0.0 else previous * 0.7 + observed * 0.3
                    )
                # And the whole thing as a duration, which is what the next
                # cold start actually has to survive.
                #
                # A rate assumes the time is spent reading bytes. LIVE
                # 2026-08-29: a 0.8GB model derived a 3.2s allowance from its
                # size and took longer than the 8s SLA to speak — the rest of
                # it is framework import, tokenizer, and shader compile, none
                # of which scale with the weights. Measured whole, no model of
                # where it went is needed.
                if spent > 0.1:
                    name = os.path.basename(str(self.model_path or ""))
                    if name:
                        seen = _COLD_FIRST_TOKEN_S.get(name, 0.0)
                        _COLD_FIRST_TOKEN_S[name] = (
                            spent if seen <= 0.0 else seen * 0.7 + spent * 0.3
                        )
        self._tokens_since_spawn = int(getattr(self, "_tokens_since_spawn", 0) or 0) + delta
        self._tokens_this_request = previous_count + delta
        self._mark_progress()

    def _mark_generation_completed(self, *, user_facing: bool = False) -> None:
        self._last_generation_completed_at = time.time()
        if user_facing:
            self._last_user_facing_completed_at = self._last_generation_completed_at
            self._last_visible_readiness_at = self._last_generation_completed_at
        self._clear_active_generation_tracking()

    def _mark_runtime_unavailable(self, detail: str) -> None:
        reason = f"mlx_runtime_unavailable:{detail}"
        self._warmup_in_flight = False
        self._init_done = False
        self._set_lane_state("failed", reason)

    def _mark_healthy_generation_deadline(self, *, foreground_request: bool) -> None:
        """Publish a non-damaging no-text outcome and fence abandoned output."""
        self._deliberate_no_text_reason = "generation_deadline_worker_healthy"
        if foreground_request:
            self.soft_cancel_active_generation("abandoned_generation_deadline")

    def _wait_for_generation_result_remaining(self, deadline, foreground_request, hard_cap, progress_owned_completion, token_stall_after, wait_started):
        from .mlx_client import (
            logger,
        )

        remaining = deadline.remaining
        if not progress_owned_completion and remaining is not None and remaining <= 0.0:
            if not self._still_producing(
                within_s=token_stall_after, foreground_request=foreground_request
            ):
                raise TimeoutError
            # Tokens are still arriving, so the answer is being written.
            # Cancelling here throws away work that is going fine, and
            # what comes back instead is half a reply or an apology.
            #
            # This runtime serves one person on one laptop. Nothing is
            # queued behind this turn and nothing is being billed, so the
            # only thing a deadline buys is the illusion of control over
            # something that is already working. What actually needs
            # catching — a wedged worker, a decode looping forever — is
            # caught by the stall checks below and by the sentinel that
            # reads the output, neither of which asks what time it is.
            #
            # Still bounded: the hard cap above ends the wait for anything
            # pathological, and a generation that goes quiet fails on the
            # very next slice.
            if not self._said_it_is_taking_longer:
                self._said_it_is_taking_longer = True
                logger.info(
                    "⏳ [MLX] Past the deadline and still producing tokens; "
                    "waiting for the answer rather than cancelling it "
                    "(bounded at %.0fs).",
                    hard_cap,
                )

        # An expired soft deadline can still have an active decode. A
        # zero-second future wait spins the parent instead of observing it.
        slice_timeout = 2.0
        if not progress_owned_completion:
            if remaining and remaining > 0.0:
                slice_timeout = min(slice_timeout, remaining)
            slice_timeout = min(
                slice_timeout, max(0.001, hard_cap - (time.monotonic() - wait_started))
            )
        return slice_timeout

    async def _wait_for_generation_result_part_2(self):
        from .mlx_client import (
            _record_mlx_degradation,
            gc,
            get_memory_pressure_snapshot,
            logger,
        )

        self._rebase_after_system_sleep()

        # OBSERVATION and ENFORCEMENT are separated. They used to share
        # one try block, so a failure while ABORTING (queue cleanup,
        # future cancellation) was reported as "probe unavailable" and
        # the loop kept waiting with lifecycle state half-cleared —
        # the request neither aborted nor honestly failed.
        memory_snapshot = None
        try:
            memory_snapshot = await asyncio.to_thread(get_memory_pressure_snapshot)
            if memory_snapshot.should_gc:
                await asyncio.to_thread(gc.collect)
        except (OSError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            # Unobserved pressure is not observed headroom. Heavy lanes
            # are the allocation that pushes this host over, so a blind
            # probe is recorded rather than shrugged off at debug.
            if self._is_primary_or_deep_lane():
                _record_mlx_degradation(
                    exc,
                    action=(
                        "live memory-pressure probe unavailable during heavy "
                        "generation; abort decision could not be made"
                    ),
                    severity="warning",
                )
            else:
                logger.debug("MLX live memory pressure probe unavailable: %s", exc)
        return memory_snapshot

    def _wait_for_generation_result_part_3(self, foreground_request, future, memory_snapshot, req_id):
        from .mlx_client import (
            _cancel_shared_future,
            _record_mlx_degradation,
            logger,
        )

        logger.error(
            "🛑 [MLX] Aborting generation for %s under live memory pressure: %s",
            os.path.basename(self.model_path),
            memory_snapshot.reason,
        )
        self._pending_generations.pop(req_id, None)
        self._record_degraded_event(
            "generation_aborted_memory_pressure",
            detail=f"{os.path.basename(self.model_path)}:{memory_snapshot.reason}",
            severity="critical",
            foreground_request=foreground_request,
        )
        try:
            self.force_abort_active_generation("memory_pressure_during_generation")
            _cancel_shared_future(future)
        except (
            OSError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as abort_exc:
            # The abort itself failed. Critical pressure WAS
            # observed and cleanup cannot be proven, so the
            # request ends terminally with that on the record
            # instead of quietly resuming the wait.
            _record_mlx_degradation(
                abort_exc,
                action=(
                    "memory-pressure abort failed; generation state "
                    "could not be proven clean"
                ),
                severity="critical",
            )
            self._record_degraded_event(
                "generation_abort_failed_memory_pressure",
                detail=(
                    f"{os.path.basename(self.model_path)}:"
                    f"{type(abort_exc).__name__}"
                ),
                severity="critical",
                foreground_request=foreground_request,
            )

    def _wait_for_generation_result_current_runtime_progress(self, foreground_request, request_started_at):
        current_runtime_progress = max(
            self._last_heartbeat,
            self._last_progress_at,
            self._last_ready_at,
        )
        progress_baseline = float(
            getattr(self, "_current_request_progress_baseline_at", 0.0) or 0.0
        )
        has_runtime_progress_after_request = current_runtime_progress > max(
            request_started_at + 0.5,
            progress_baseline + 0.5,
        )
        # Heartbeats stretch the first-token SLA; they never waive
        # it. Round 14 live proof: a LIVELOCKED generation (worker
        # heartbeating, zero tokens) ran 185s to the endpoint
        # deadline because runtime progress exempted it forever.
        # Past the hard ceiling, silence is wedged no matter how
        # alive the worker claims to be.
        livelock_ceiling = self._first_token_hard_ceiling(
            foreground_request=foreground_request
        )
        # Reading a long question is not silence.
        #
        # The livelock ceiling asks how long a worker may go without
        # producing a token before it is wedged, and it was answered
        # without reference to how much there was to read. A prompt of
        # 8,618 characters takes about eighteen seconds to prefill at
        # the rate this host was measured at, and the ceiling was
        # twenty: LIVE 2026-08-26, "Cortex still sending heartbeats
        # (1.8s ago) but produced no token in 20.2s. Recycling the
        # lane." Every large question recycled a warm 20GB model,
        # which made the next one slower still.
        #
        # The same floor the request ceiling already uses. It only
        # ever raises this, and only by what the reading actually
        # costs.
        livelock_ceiling = max(
            livelock_ceiling,
            self._prefill_floor_seconds(self._current_prompt_chars),
            # A lane that has never produced a token is still reading
            # its weights. Silence there is the load, not a wedge.
            self._cold_lane_first_token_allowance(),
        )
        return has_runtime_progress_after_request, livelock_ceiling

    def _wait_for_generation_result_ceiling_about_fire(self, request_started_at):
        # Which ceiling is about to fire matters, because the two mean
        # opposite things about the worker.
        #
        # The LIVELOCK ceiling is the formula above — heartbeats with
        # zero tokens for far longer than any healthy generation. That
        # is a wedged worker and recycling it is correct.
        #
        # The DEADLINE ceiling is the caller's remaining wall-clock
        # minus a small reserve. Hitting it says nothing about the
        # worker's health; it says this turn ran out of time. The
        # abandonment branch below tests the two apart before it
        # decides whether to throw away a warm 20GB model.
        elapsed_without_token = time.time() - request_started_at
        # Reading the prompt is not silence.
        #
        # A first-token deadline asks "has anything come out yet",
        # and before the first token can exist the whole prompt has
        # to be read. On this host that is measured at about 720
        # tokens a second, so a two-thousand-token prompt spends
        # nearly three seconds in prefill by design — and a caller
        # whose budget is four seconds cancels the request at the
        # moment prefill finishes, every time, for reasons that have
        # nothing to do with the worker.
        #
        # LIVE 2026-08-26: every decision she made while playing was
        # cancelled this way. "her reasoning produced nothing (no
        # text came back)" over and over, so she chose her moves from
        # the consequence record alone and never held a plan — which
        # from outside looks exactly like a mind that is not
        # thinking.
        #
        # Prefill progress is progress, and stronger evidence than
        # the heartbeat already consulted here: a worker advancing
        # through the prompt is doing the work that produces the
        # first token. The livelock ceiling still applies, so a
        # genuinely wedged prefill is still caught.
        prefilling = (
            self._current_prefill_tokens_total > 0
            and self._current_prefill_tokens_processed
            < self._current_prefill_tokens_total
            and (time.time() - self._last_progress_at) < 5.0
        )
        return elapsed_without_token, prefilling

    def _wait_for_generation_result_ceiling_fired_decides(self, elapsed_without_token, first_token_sla, foreground_request, future, hard_first_token_ceiling, livelock_ceiling, req_id):
        # Which ceiling fired decides what this line is allowed to
        # claim. hard_first_token_ceiling is min(livelock, the
        # caller's deadline), so exceeding it usually means the
        # TURN ran out of budget, not that the worker is wedged.
        # The branch below already tested the two apart correctly
        # and kept the warm lane; only this message did not, and
        # it is the one a person reads. Live 2026-08-03, two
        # consecutive lines:
        #
        #   🛑 HARD CEILING exceeded (livelocked: heartbeats but
        #      zero tokens) ... 18.4s elapsed, hard=16.8s
        #   ⏱️ ...but is healthy (heartbeat 0.7s ago, livelock
        #      ceiling 20.0s). KEEPING the warm lane.
        #
        # 18.4s was under the 20.0s livelock ceiling. Nothing was
        # livelocked. Reporting a budget overrun as a wedged
        # worker sends someone hunting a fault that did not
        # happen — and at error severity it recruits the incident
        # machinery to hunt it too.
        from .mlx_client import (
            _cancel_shared_future,
            logger,
        )

        livelocked = elapsed_without_token > livelock_ceiling
        if livelocked:
            logger.error(
                "🛑 [MLX] First-token LIVELOCK for %s: heartbeats but zero tokens "
                "in %.1fs (livelock ceiling %.1fs, sla=%.1fs).",
                os.path.basename(self.model_path),
                elapsed_without_token,
                livelock_ceiling,
                first_token_sla,
            )
        elif elapsed_without_token > hard_first_token_ceiling:
            logger.warning(
                "⏱️ [MLX] First-token deadline exceeded for %s (%.1fs elapsed, "
                "turn budget %.1fs, sla=%.1fs). The worker is not wedged — the "
                "livelock ceiling is %.1fs.",
                os.path.basename(self.model_path),
                elapsed_without_token,
                hard_first_token_ceiling,
                first_token_sla,
                livelock_ceiling,
            )
        else:
            logger.warning(
                "⏱️ [MLX] First-token SLA exceeded for %s (%.1fs elapsed, "
                "sla=%.1fs) with no runtime progress.",
                os.path.basename(self.model_path),
                elapsed_without_token,
                first_token_sla,
            )
        self._pending_generations.pop(req_id, None)
        self._record_degraded_event(
            "first_token_sla_exceeded",
            detail=(
                f"{os.path.basename(self.model_path)}>{first_token_sla:.1f}s"
                f"{self._pressure_receipt_suffix()}"
            ),
            # A healthy worker that ran past this turn's budget is
            # expected backpressure, which CLAUDE.md says to record
            # below error. Only a real livelock is an error.
            severity="error" if livelocked else "warning",
            foreground_request=foreground_request,
        )
        # If we abandon a foreground generation, its eventual
        # output must never survive into the next turn. Fresh
        # heartbeats mean this is recoverable, not that the warm
        # lane is safe to keep carrying an orphaned request.
        heartbeat_age = (
            time.time() - self._last_heartbeat if self._last_heartbeat > 0 else 999.0
        )
        # LIVE DEFECT, 2026-07-25. Bryan asked a follow-up and got
        # nothing back. The trace:
        #
        #   First-token HARD CEILING exceeded (82.5s, hard=82.0s)
        #   Cortex still sending heartbeats (1.8s ago). Recycling...
        #   Abort ... arrived after the generation finished;
        #     nothing to abort, leaving the worker up.
        #
        # The 82.0s ceiling was not the livelock formula — that
        # computes ~450s here. It was the caller's deadline minus
        # the reserve, from an 86s inference-gate budget. And the
        # generation FINISHED, a few seconds after we stopped
        # waiting. The worker was never wedged; the turn was
        # simply slower than its budget under 80% RAM.
        #
        # Recycling it cost a 20GB reload, which made the NEXT
        # turn slower, which made the next deadline likelier to
        # expire. That is the cascade, and the recycle was the
        # part of it we chose.
        #
        # Orphaned output is already fenced three ways below and
        # above: the pending generation is dropped, the request id
        # no longer matches, and the worker is soft-cancelled
        # between tokens. Destroying a warm 20GB model was never
        # what kept late text out of the next turn.
        #
        # `livelocked` is computed once above, where it also picks
        # the wording of the line the operator reads.
        if heartbeat_age > 30.0:
            self._deferred_reboot_reason = "first_token_sla_exceeded"
        elif livelocked:
            logger.warning(
                "🛡️ [MLX] Cortex still sending heartbeats (%.1fs ago) but produced "
                "no token in %.1fs (livelock ceiling %.1fs). Recycling the lane.",
                heartbeat_age,
                elapsed_without_token,
                livelock_ceiling,
            )
            self._deferred_reboot_reason = "recoverable_first_token_sla_exceeded"
        else:
            logger.warning(
                "⏱️ [MLX] Cortex ran past this turn's deadline (%.1fs elapsed, "
                "budget %.1fs) but is healthy (heartbeat %.1fs ago, livelock "
                "ceiling %.1fs). Cancelling the request and KEEPING the warm lane.",
                elapsed_without_token,
                hard_first_token_ceiling,
                heartbeat_age,
                livelock_ceiling,
            )
            self._record_degraded_event(
                "first_token_deadline_exceeded_worker_healthy",
                detail=(
                    f"{os.path.basename(self.model_path)}"
                    f">{hard_first_token_ceiling:.1f}s"
                    f"{self._pressure_receipt_suffix()}"
                ),
                severity="warning",
                foreground_request=foreground_request,
            )
            # We chose to end this generation while the worker was
            # healthy. Publish that so the router scores the empty
            # result as our deferral rather than as Cortex damage.
            self._deliberate_no_text_reason = (
                "first_token_deadline_exceeded_worker_healthy"
            )
        # Ask the worker to drop the orphaned generation between
        # tokens — the abandoned output then never arrives at all,
        # instead of relying solely on a worker recycle.
        self.soft_cancel_active_generation("abandoned_first_token_sla")
        _cancel_shared_future(future)

    def _wait_for_generation_result_part_7(self, foreground_request, future, req_id, token_stall_after):
        from .mlx_client import (
            _cancel_shared_future,
            logger,
        )

        logger.error(
            "🛑 [MLX] Token progress stalled during generation for %s (>%.1fs).",
            os.path.basename(self.model_path),
            token_stall_after,
        )
        self._pending_generations.pop(req_id, None)
        self._record_degraded_event(
            "token_progress_stalled",
            detail=(
                f"{os.path.basename(self.model_path)}>{token_stall_after:.1f}s"
                f"{self._pressure_receipt_suffix()}"
            ),
            severity="error",
            foreground_request=foreground_request,
        )
        # Same principle as the first-token SLA: fresh heartbeats
        # keep this recoverable, but the abandoned generation must
        # be isolated from future foreground turns.
        heartbeat_age = (
            time.time() - self._last_heartbeat if self._last_heartbeat > 0 else 999.0
        )
        if heartbeat_age > 30.0:
            self._deferred_reboot_reason = "token_progress_stalled"
        else:
            logger.warning(
                "🛡️ [MLX] Cortex still sending heartbeats (%.1fs ago). "
                "Recycling after this abandoned foreground request so late text cannot bleed into the next turn.",
                heartbeat_age,
            )
            self._deferred_reboot_reason = "recoverable_token_progress_stalled"
        self.soft_cancel_active_generation("abandoned_token_stall")
        _cancel_shared_future(future)

    async def _wait_for_generation_result(
        self,
        req_id: str,
        future: SharedFuture,
        deadline: Deadline,
        *,
        foreground_request: bool = False,
        progress_owned_completion: bool = False,
    ) -> dict[str, Any] | None:
        """Wait in short slices so dead workers fail fast instead of hanging the UI."""
        from .mlx_client import (
            _await_shared_future,
            _cancel_shared_future,
            _generation_wait_hard_cap_s,
            logger,
        )

        stall_after = self._stale_after(
            during_generation=True, foreground_request=foreground_request
        )
        first_token_sla = self._first_token_sla(foreground_request=foreground_request)
        token_stall_after = self._token_stall_after(foreground_request=foreground_request)
        wait_started = time.monotonic()
        self._said_it_is_taking_longer = False
        # Finite-bounded: a malformed value previously RAISED through the
        # generation wait path, and infinity disabled the hard cap entirely.
        hard_cap = _generation_wait_hard_cap_s(
            deadline,
            foreground_request=foreground_request,
        )
        progress_owned_completion = progress_owned_completion and foreground_request
        while progress_owned_completion or (time.monotonic() - wait_started) <= hard_cap:
            slice_timeout = self._wait_for_generation_result_remaining(deadline, foreground_request, hard_cap, progress_owned_completion, token_stall_after, wait_started)
            try:
                return await _await_shared_future(future, timeout_s=slice_timeout)
            except TimeoutError:
                if future.done():
                    return future.result()

                memory_snapshot = await self._wait_for_generation_result_part_2()

                if (
                    memory_snapshot is not None
                    and memory_snapshot.refuse_heavy_local_generation
                    and self._is_primary_or_deep_lane()
                ):
                    from core.brain.llm.emergency_override import consume_override

                    live_override = consume_override(
                        "AURA_MLX_ALLOW_CRITICAL_MEMORY_GENERATION",
                        guard="live_memory_pressure_abort",
                        observed=(f"{os.path.basename(self.model_path)}:{memory_snapshot.reason}"),
                    )
                    if not live_override.active:
                        self._wait_for_generation_result_part_3(foreground_request, future, memory_snapshot, req_id)
                        return None

                if self._process is not None and not self._process.is_alive():
                    logger.error(
                        "🛑 [MLX] Worker died during generation. Deferring reboot until lock released."
                    )
                    self._pending_generations.pop(req_id, None)
                    self._record_degraded_event(
                        "worker_died_during_generation",
                        detail=os.path.basename(self.model_path),
                        severity="error",
                        foreground_request=foreground_request,
                    )
                    self._deferred_reboot_reason = "worker_died_during_generation"
                    _cancel_shared_future(future)
                    return None

                self._refresh_worker_job_activity()
                request_started_at = self._current_request_started_at
                has_runtime_progress_after_request, livelock_ceiling = self._wait_for_generation_result_current_runtime_progress(foreground_request, request_started_at)
                hard_first_token_ceiling = livelock_ceiling
                request_hard_ceiling = float(
                    getattr(self, "_current_first_token_hard_ceiling_s", 0.0) or 0.0
                )
                if request_hard_ceiling > 0.0:
                    hard_first_token_ceiling = min(
                        hard_first_token_ceiling,
                        request_hard_ceiling,
                    )
                elapsed_without_token, prefilling = self._wait_for_generation_result_ceiling_about_fire(request_started_at)
                advancing_prefill = (
                    self._current_prefill_tokens_total > 0
                    and self._current_prefill_tokens_processed < self._current_prefill_tokens_total
                    and (time.time() - self._prefill_progress_at()) < stall_after
                )
                worker_advancing = (
                    self._last_worker_job_activity_at > 0.0
                    and time.time() - self._last_worker_job_activity_at < token_stall_after
                )
                if progress_owned_completion:
                    hard_first_token_ceiling = livelock_ceiling
                elif prefilling and elapsed_without_token <= livelock_ceiling:
                    hard_first_token_ceiling = livelock_ceiling
                if (
                    req_id == self._current_request_id
                    and request_started_at > 0.0
                    and self._current_first_token_at <= 0.0
                    and not (
                        progress_owned_completion and (advancing_prefill or worker_advancing)
                    )
                    and (
                        (
                            elapsed_without_token > max(
                                first_token_sla,
                                # A lane that has never spoken is still coming
                                # up. The livelock ceiling learned this; the
                                # SLA did not, so the first generation of a
                                # worker's life was abandoned at 8 seconds —
                                # and it is the generation the measurement
                                # comes from, so the cold start could never be
                                # learned either.
                                self._cold_lane_first_token_allowance(),
                            )
                            and not has_runtime_progress_after_request
                        )
                        or elapsed_without_token > hard_first_token_ceiling
                    )
                ):
                    self._wait_for_generation_result_ceiling_fired_decides(elapsed_without_token, first_token_sla, foreground_request, future, hard_first_token_ceiling, livelock_ceiling, req_id)
                    return None

                # Reading is work here too, and this clock could not see it.
                #
                # Progress is emitted per token, so a generation that spends
                # twenty seconds inside one prefill step emits nothing and
                # reads as stalled. That case is real and already recorded in
                # the worker: "a measured 755-token recurrent prefill occupied
                # the inference thread for roughly 52 seconds". It happens
                # after the first token when a second pass re-reads the
                # context, which is exactly when this clock is watching.
                #
                # LIVE 2026-08-29: asked to work out why turns were slow, the
                # 27B produced tokens, went quiet for 40 seconds, and was
                # abandoned — "Token progress stalled during generation
                # (>40.0s)", "Cortex still sending heartbeats (2.2s ago)". The
                # person got the canned apology.
                #
                # Prefill progress was deliberately kept out of the first-token
                # clock, because there the question is whether reading has
                # begun at all. Here the question is whether the generation is
                # doing anything, and reading is doing something.
                last_token_progress = max(
                    self._last_token_progress_at,
                    self._current_first_token_at,
                    self._prefill_progress_at(),
                    self._last_worker_job_activity_at,
                )
                if (
                    req_id == self._current_request_id
                    and self._current_first_token_at > 0.0
                    and last_token_progress > 0.0
                    and (time.time() - last_token_progress) > token_stall_after
                ):
                    self._wait_for_generation_result_part_7(foreground_request, future, req_id, token_stall_after)
                    return None

                last_progress = max(
                    self._last_heartbeat, self._last_progress_at, self._last_ready_at,
                    self._last_worker_job_activity_at,
                )
                if last_progress and (time.time() - last_progress) > stall_after:
                    logger.error(
                        "🛑 [MLX] Worker heartbeat stalled during generation. Deferring reboot until lock released."
                    )
                    self._pending_generations.pop(req_id, None)
                    self._record_degraded_event(
                        "heartbeat_stalled_during_generation",
                        detail=f"{os.path.basename(self.model_path)} stalled for >{stall_after:.0f}s",
                        severity="error",
                        foreground_request=foreground_request,
                    )
                    self._deferred_reboot_reason = "heartbeat_stalled_during_generation"
                    self.soft_cancel_active_generation("abandoned_heartbeat_stall")
                    _cancel_shared_future(future)
                    return None
        raise TimeoutError

