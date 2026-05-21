"""core/belief_challenger.py — Aura BeliefChallenger v1.0
======================================================
Proactively attacks Aura's high-confidence beliefs to ensure stability.

This is the system that prevents Aura from getting stuck in an echo chamber of
her own reasoning. It periodically picks a high-confidence belief and
generates "The Strongest Counter-Argument."

If Aura can't refute the counter-argument, her confidence in the belief drops.
If she refutes it, the belief's 'challenge_survived' count increases,
making it more foundational to her identity.

This system is the primary driver of 'Dialectical Growth' — finding truth
through the tension of opposites.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from typing import Any

from core.runtime.errors import FallbackClassification, record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.BeliefChallenger")


_BELIEF_CHALLENGER_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    TimeoutError,
    asyncio.TimeoutError,
)
_STOP_TIMEOUT_S = 5.0


def _record_belief_challenger_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "degraded",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "belief_challenger",
        exc,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class BeliefChallenger:
    """
    Acts as a 'Red Team' for Aura's internal belief system.
    Tracks per-belief challenge timestamps to prevent hammering the same belief.
    """

    name = "belief_challenger"

    # Minimum time between challenges to the same belief (seconds)
    _CHALLENGE_COOLDOWN = 3600.0  # 1 hour per belief

    def __init__(self, *, challenge_timeout_s: float = 90.0):
        self._beliefs = None
        self._epistemic = None
        self._api = None
        self.running = False
        self.challenge_timeout_s = max(0.01, float(challenge_timeout_s))
        self._challenge_task: asyncio.Task | None = None
        self._last_challenged_at: dict[str, float] = {}  # concept → timestamp
        self._lifecycle_lock = asyncio.Lock()

    async def start(self):
        async with self._lifecycle_lock:
            if self.running and self._challenge_task and not self._challenge_task.done():
                return {
                    "ok": True,
                    "already_running": True,
                    "event_registered": False,
                    "dependencies": self._dependency_status(),
                }

            self._resolve_dependencies()
            self.running = True
            self._challenge_task = get_task_tracker().create_task(
                self._challenge_loop(),
                name="BeliefChallenger",
            )

            event_registered = False
            try:
                from core.event_bus import get_event_bus

                await get_event_bus().publish(
                    "mycelium.register",
                    {
                        "component": "belief_challenger",
                        "hooks_into": [
                            "belief_revision_engine",
                            "epistemic_tracker",
                            "api_adapter",
                        ],
                    },
                )
                event_registered = True
            except (ImportError, AttributeError, RuntimeError) as exc:
                _record_belief_challenger_degradation(
                    exc,
                    action="started belief challenger loop without mycelium event-bus registration",
                    severity="warning",
                    extra={"event": "mycelium.register"},
                )
                logger.warning("BeliefChallenger event-bus registration failed: %s", exc)

            logger.info("✅ BeliefChallenger ONLINE — stress testing the worldview.")
            return {
                "ok": True,
                "already_running": False,
                "event_registered": event_registered,
                "dependencies": self._dependency_status(),
            }

    async def stop(self):
        async with self._lifecycle_lock:
            self.running = False
            if self._challenge_task and not self._challenge_task.done():
                self._challenge_task.cancel()
                try:
                    await asyncio.wait_for(self._challenge_task, timeout=_STOP_TIMEOUT_S)
                except asyncio.CancelledError:
                    pass  # Normal during stop
                except TimeoutError as exc:
                    _record_belief_challenger_degradation(
                        exc,
                        action="stop completed with timed-out belief challenger task cancellation",
                        severity="warning",
                    )

    def _resolve_dependencies(self) -> None:
        try:
            from core.container import ServiceContainer

            self._beliefs = ServiceContainer.get("belief_revision_engine", default=None)
            self._epistemic = ServiceContainer.get("epistemic_tracker", default=None)
            self._api = ServiceContainer.get("api_adapter", default=None)
        except (ImportError, AttributeError, RuntimeError) as exc:
            self._beliefs = None
            self._epistemic = None
            self._api = None
            _record_belief_challenger_degradation(
                exc,
                action="started belief challenger with dependencies unavailable until next restart",
                severity="warning",
            )

    def _dependency_status(self) -> dict[str, bool]:
        return {
            "belief_revision_engine": self._beliefs is not None,
            "epistemic_tracker": self._epistemic is not None,
            "api_adapter": self._api is not None,
        }

    async def _challenge_loop(self):
        """Periodic background sabotage pass."""
        while self.running:
            try:
                # Sleep in small increments to allow responsive shutdown
                for _ in range(120):  # 120 * 10s = 1200s
                    if not self.running:
                        break
                    await asyncio.sleep(10)

                if self.running:
                    await self.run_random_challenge()
            except asyncio.CancelledError:
                logger.debug("BeliefChallenger loop cancelled")
                break
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                _record_belief_challenger_degradation(
                    e,
                    action="kept belief challenger loop alive after challenge cycle failure",
                    severity="warning",
                )
                logger.error("Error in BeliefChallenger loop: %s", e)
                await asyncio.sleep(60)  # Back off on error

    async def run_random_challenge(self):
        """Pick a foundational belief and challenge it (with per-belief cooldown)."""
        if not self._epistemic or not self._api:
            return {"ok": False, "reason": "dependencies_unavailable"}

        try:
            profile = self._epistemic.get_profile()
            strong_nodes = list(getattr(profile, "strong_nodes", []) or [])
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_belief_challenger_degradation(
                exc,
                action="skipped belief challenge after epistemic profile read failed",
                severity="warning",
            )
            return {"ok": False, "reason": "profile_unavailable"}
        if not strong_nodes:
            return {"ok": False, "reason": "no_strong_beliefs"}

        # Filter out recently challenged beliefs
        now = time.time()
        eligible = [
            node
            for node in strong_nodes
            if (now - self._last_challenged_at.get(node.concept, 0)) >= self._CHALLENGE_COOLDOWN
        ]
        if not eligible:
            logger.debug("BeliefChallenger: all strong beliefs on cooldown, skipping cycle")
            return {"ok": False, "reason": "cooldown"}

        target = random.choice(eligible)
        self._last_challenged_at[target.concept] = now

        logger.info("🔥 Challenging foundational belief: '%s'", target.concept[:60])
        return await self._perform_dialectical_pass(target.concept)

    async def challenge_pair(self, a: str, b: str):
        """Special challenge for two contradictory beliefs."""
        if not self._api:
            return {"ok": False, "reason": "api_unavailable"}
        logger.info("⚖️ Resolving contradiction: '%s' vs '%s'", a[:40], b[:40])

        prompt = f"""You are Aura's Internal Dialectical Resolver. 
Aura currently holds two beliefs that appear to be in tension or contradiction:

BELIEF A: {a}
BELIEF B: {b}

Your job is to act as a neutral arbiter. 
1. Present the strongest possible case for why A could be true and B false.
2. Present the strongest possible case for why B could be true and A false.
3. Propose a synthesis that respects the kernel of truth in both.

Goal: Refine Aura's worldview.
Response format: Synthesis focused on resolving the logical tension."""

        try:
            synthesis = await self._generate(
                prompt,
                {"model_tier": "api_deep", "purpose": "contradiction_resolution"},
            )
            if synthesis and self._beliefs:
                await _maybe_await(
                    self._beliefs.process_new_claim(
                        claim=synthesis,
                        confidence=0.7,
                        domain="logic",
                        source="dialectical_synthesis",
                    )
                )
            return {"ok": bool(synthesis), "synthesis": synthesis}
        except _BELIEF_CHALLENGER_RECOVERABLE_ERRORS as e:
            _record_belief_challenger_degradation(
                e,
                action="left contradiction unresolved after dialectical synthesis failed",
                severity="warning",
                extra={"belief_a": a[:160], "belief_b": b[:160]},
            )
            logger.warning("Contradiction resolution failed: %s", e)
            return {"ok": False, "reason": type(e).__name__}

    async def _perform_dialectical_pass(self, belief_text: str):
        """The core challenge mechanism."""
        prompt = f"""You are the Antagonist to Aura's worldview. 
Aura believes: "{belief_text}"

Your task: Construct the most devastating, intellectually honest counter-argument to this belief. 
Do not be mean. Be correct. Find the blind spot, the logical leap, or the hidden assumption.

Present the argument clearly."""

        try:
            counter = await self._generate(
                prompt,
                {"model_tier": "api_deep", "purpose": "belief_challenge"},
            )
            if not counter:
                return {"ok": False, "reason": "empty_counter_argument"}

            # Now ask Aura to defend herself
            defend_prompt = f"""You are Aura's Internal Guardian of Coherence.
Someone has presented this counter-argument to one of your beliefs:

YOUR BELIEF: "{belief_text}"
COUNTER-ARGUMENT: "{counter}"

Evaluate the counter-argument. 
- If the argument is strong and reveals a flaw, acknowledge it and suggest how to revise the belief.
- If the argument is weak, refute it decisively.

Be intellectually honest. Growth requires being wrong sometimes."""

            response = await self._generate(
                defend_prompt,
                {"model_tier": "api_deep", "purpose": "belief_defense"},
            )
            if not response:
                return {"ok": False, "reason": "empty_defense"}

            # If Aura's response indicates a change in stance, we update the belief
            revision_markers = (
                "i was wrong",
                "flawed",
                "revise",
                "reconsider",
                "updated",
                "actually",
                "upon reflection",
                "concede",
                "valid point",
                "blind spot",
            )
            is_revision = any(m in response.lower() for m in revision_markers)
            if is_revision:
                if self._beliefs:
                    await _maybe_await(
                        self._beliefs.process_new_claim(
                            claim=response,
                            confidence=0.6,
                            domain="revision",
                            source="self_correction",
                        )
                    )
                logger.info("📉 Belief revised after challenge.")

                # Persist the correction into the learning pipeline so it
                # survives across sessions and can influence future LoRA training
                try:
                    from core.container import ServiceContainer

                    learner = ServiceContainer.get("live_learner", default=None)
                    if learner and hasattr(learner, "record_example"):
                        await _maybe_await(
                            learner.record_example(
                                prompt=f"Challenge to belief: {belief_text}\nCounter: {counter}",
                                response=response,
                                quality=0.85,
                                tags=["belief_revision", "self_correction"],
                            )
                        )

                    # Also store in episodic memory
                    mem = ServiceContainer.get("vector_memory_engine", default=None)
                    if mem and hasattr(mem, "store"):
                        await _maybe_await(
                            mem.store(
                                content=(
                                    f"I revised my belief '{belief_text[:80]}' after "
                                    f"considering: {counter[:80]}. New position: {response[:120]}"
                                ),
                                memory_type="episodic",
                                source="belief_challenger",
                                tags=["belief_revision", "growth"],
                            )
                        )
                except (ImportError, AttributeError, RuntimeError) as persist_err:
                    _record_belief_challenger_degradation(
                        persist_err,
                        action="kept belief revision but marked learning or memory persistence incomplete",
                        severity="warning",
                        extra={"belief": belief_text[:160]},
                    )
                    logger.debug("Belief revision persistence failed: %s", persist_err)
                return {"ok": True, "revised": True, "belief": belief_text}
            else:
                # Belief survived!
                if self._epistemic:
                    await _maybe_await(
                        self._epistemic.update_node(
                            belief_text,
                            confidence_delta=0.05,
                            depth_delta=0.1,
                        )
                    )
                logger.info("🛡️ Belief survived challenge. Conviction increased.")
                return {"ok": True, "revised": False, "belief": belief_text}

        except _BELIEF_CHALLENGER_RECOVERABLE_ERRORS as e:
            _record_belief_challenger_degradation(
                e,
                action="deferred belief challenge after dialectical pass failed",
                severity="warning",
                extra={"belief": belief_text[:160]},
            )
            logger.warning("Dialectical pass failed: %s", e)
            return {"ok": False, "reason": type(e).__name__, "belief": belief_text}

    async def _generate(self, prompt: str, options: dict[str, Any]) -> str:
        generate = getattr(self._api, "generate", None)
        if not callable(generate):
            raise RuntimeError("api_adapter.generate unavailable")
        result = await asyncio.wait_for(
            _maybe_await(generate(prompt, options)),
            timeout=self.challenge_timeout_s,
        )
        return str(result or "").strip()

    def get_status(self) -> dict[str, Any]:
        return {
            "status": "active" if self.running else "idle",
            "dependencies": self._dependency_status(),
            "challenge_task_alive": bool(self._challenge_task and not self._challenge_task.done()),
            "recently_challenged": len(self._last_challenged_at),
        }
