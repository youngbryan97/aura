"""Her language organ in the subject-core driver: the stub, the live one, and a phase's time.

Lifted out of core/subject/driver.py when it passed its size ceiling. The stub
answers every prompt from the prompt alone, which is what every campaign ran
on before `--whole`; `bring_up_language` brings up the organ the desktop runs;
and `within_budget` gives a phase its time plus whatever it spent waiting on
that organ.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import logging
import time
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

from core.kernel.turn_door import USER_ORIGINS

logger = logging.getLogger("Aura.Subject.Driver")

#: How long her language organ may take to come up, the first time and after
#: its worker has stopped: loading her cortex is the slow part, and it is the
#: same load either way.
FOREGROUND_READY_S: float = 900.0

#: How often a lane that is still coming up is asked again.
_RECHECK_S: float = 10.0

__all__ = [
    "FOREGROUND_READY_S",
    "DeterministicMind",
    "bring_up_language",
    "install_mind",
    "keep_language_ready",
    "opens_the_turn",
    "person_turn",
    "phase_budget",
    "within_budget",
]


class DeterministicMind:
    """One answer, always, so two arms differ by the intervention and nothing else.

    Registered both as the kernel's `llm` organ and as the container's
    `llm_router`, because the phases prefer the router and fall back to the
    organ. Installed only as the organ, the response phase asked the real
    router — which has no model loaded offline — got nothing back, and raised
    on every single turn of every run. The reply never landed, so the exchange
    never reached memory consolidation, self-review had nothing to review, and
    the whole arc from a question to an answer was absent from a measurement of
    whether the parts of her reach each other.
    """

    #: Which turn this is. Set by the runtime before each turn and carried
    #: across the fork with everything else, so the two arms of a trial share
    #: it and the reply cannot differ between them — while two different turns
    #: get different replies, which is what stops the loop detector from
    #: reading the harness's constancy as her repeating herself.
    moment: int = 0

    #: The reply is deliberately bland and constant. Anything that varied would
    #: enter the state through several phases at once and appear as coupling.
    REPLY = "Continuity holds. The pipeline is executing over the shared state."

    #: What a caller that parses JSON gets. Several phases ask the model for a
    #: structured answer and raise on prose, so a stub that only speaks prose
    #: makes those phases fail on every turn — the inference phase did, for the
    #: whole of every run, and a phase that always fails is a phase absent from
    #: the measurement. Constant like the prose: the same shape every time, so
    #: two arms still differ by the intervention and nothing else.
    STRUCTURED = (
        '{"implicit_intent": "continue the exchange", "user_subtext": "steady", '
        '"momentum": "steady", "conversation_hooks": []}'
    )

    async def think(self, prompt: str, **_kwargs: Any) -> str:
        # A test double has to honour the interface its callers expect. Which
        # of the two constants comes back is decided by what the caller asked
        # for, not by anything about the state, so the answer is still a
        # function of the call site alone.
        if "json" in str(prompt or "").lower():
            return self.STRUCTURED
        # And the prose carries a mark of what was asked. A reply that is
        # byte-identical on every turn is a repetition, and the memory
        # consolidation phase is right to call it one: it degrades identity
        # stability to its floor and clears the pending initiatives that
        # produced it. So the harness pinned the self-state at its worst value
        # and suppressed deliberation's only output, on every turn of every
        # run, by being too constant.
        #
        # Still a function of the prompt alone, so two arms of a trial — which
        # share a snapshot and therefore a prompt — get the same answer, and
        # nothing about the displacement can reach the decoder.
        seed = f"{self.moment}|{prompt or ''}"
        mark = hashlib.blake2b(seed.encode("utf-8", "ignore"), digest_size=4)
        return f"{self.REPLY} [{mark.hexdigest()}]"

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return await self.think(prompt, **kwargs)

    async def classify(self, _prompt: str) -> str:
        return "CHAT"

    async def embed(self, _text: str) -> list[float]:
        return [0.0] * 8

    async def route(self, prompt: str, **kwargs: Any) -> str:
        return await self.think(prompt, **kwargs)

    async def chat(self, prompt: str, **kwargs: Any) -> str:
        return await self.think(prompt, **kwargs)

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        return await self.think(prompt, **kwargs)

    def get_stats(self) -> dict[str, Any]:
        # A count that does not move. The body reads token velocity off this,
        # and a growing count would be a clock in the interoception domain.
        return {"total_calls": 0}

    @property
    def high_pressure_mode(self) -> bool:
        return False


def install_mind(kernel: Any, engine: Any) -> None:
    """Make `engine` the language organ, under every name the phases ask for it by.

    The organ shape matters: phases test `organ.ready.is_set()` and read
    `organ.instance`, and a stub missing either makes the phase raise, which
    would be recorded as a phase that does nothing rather than as a hole in the
    harness. And under the names the phases ask for first: registered only as
    the organ, every phase that prefers `llm_router` reached a router with no
    model behind it.
    """
    import threading
    from types import SimpleNamespace

    from core.container import ServiceContainer

    ready = threading.Event()
    ready.set()
    kernel.organs["llm"] = SimpleNamespace(
        get_instance=lambda: engine, instance=engine, ready=ready, name="llm"
    )
    for name in ("llm_router", "local_llm"):
        try:
            ServiceContainer.register_instance(name, engine)
        except Exception as exc:  # noqa: BLE001 - a container that refuses is a datum
            logger.warning("could not register the language organ as %s: %s", name, exc)


async def bring_up_language(runtime: Any) -> dict[str, Any]:
    """Her language organ as the desktop boots it, then held to a greedy decode.

    The order is the boot's: the inference gate is built and initialised and
    registered first, and the router is built after it, so the router serves
    her cortex through the gate as it does on the desktop. Then the foreground
    lane is waited for, and each tier is asked once, because a lane still
    loading answers nothing and its circuit opens, and a measured turn must not
    be the one that finds it cold.
    """
    from core.brain.inference_gate import InferenceGate
    from core.brain.llm_health_router import build_router_from_config
    from core.config import config
    from core.container import ServiceContainer
    from core.subject.steady_mind import SteadyMind

    gate = InferenceGate(None)
    await gate.initialize()
    ServiceContainer.register_instance("inference_gate", gate)
    router = build_router_from_config(config)
    foreground = await gate.ensure_foreground_ready(timeout=FOREGROUND_READY_S)
    warmed: dict[str, bool] = {}
    for tier in ("primary", "tertiary"):
        answer = None
        for _ in range(30):
            try:
                answer = await router.think(
                    "Reply with the word ready.", prefer_tier=tier, max_tokens=8, temperature=0.0
                )
            except Exception as exc:  # noqa: BLE001 - a lane still loading is waited for
                logger.info("the %s lane is not answering yet: %s", tier, exc)
                answer = None
            if answer:
                break
            await asyncio.sleep(_RECHECK_S)
        warmed[tier] = bool(answer)
    mind = SteadyMind(router)
    install_mind(runtime.kernel, mind)
    return {
        "gate_initialized": bool(getattr(gate, "_initialized", False)),
        "foreground": {key: foreground.get(key) for key in ("state", "ready", "model") if isinstance(foreground, dict)},
        "warmed": warmed,
    }


async def keep_language_ready(runtime: Any) -> dict[str, Any]:
    """Start her language organ's worker again before a turn if it has stopped.

    The router's endpoint timeout stops a model worker as its way of aborting a
    generation. When inference comes back empty the router is meant to ask the
    gate to start the worker again (`_respawn_cortex_if_needed`), but it
    validates an endpoint first, and a stopped worker failed validation and
    opened the circuit, so nothing asked. The whole report run of 23 September
    lost her cortex on its first baseline turn this way, and every later turn
    ended in the failure sentence. The harness stops her background loops so
    that two arms see the same computation, and anything else that might have
    noticed is stopped with them.

    Only a stopped worker is acted on. The gate's own recovery is asked once and
    then watched, never asked again while it works: asking the gate to warm the
    lane on every recheck kept its startup quiet window open, and inside that
    window the warmup itself is refused. A worker that is alive and still coming
    up is left to the turn's own path.

    Checked before a turn's first frame, under the experiment clock, so the
    wait moves nothing in her. A turn her cortex could not serve still counts as
    one her cortex did not answer; this only keeps one stop from ending the run.
    """
    if not getattr(runtime, "whole", False):
        return {"checked": False}
    from core.container import ServiceContainer

    gate = ServiceContainer.get("inference_gate", default=None)
    client = getattr(gate, "_mlx_client", None)
    alive = getattr(client, "is_alive", None)
    respawn = getattr(gate, "_respawn_cortex_if_needed", None)
    if not callable(alive) or not callable(respawn) or alive():
        return {"checked": True, "recovered": False}
    started = time.monotonic()
    logger.warning("her language organ's worker has stopped; asking the gate to start it again")
    await respawn()
    while time.monotonic() - started < FOREGROUND_READY_S:
        if alive() and gate.get_conversation_status().get("conversation_ready"):
            seconds = round(time.monotonic() - started, 1)
            logger.warning("her language organ's worker was back in %.0fs", seconds)
            return {"checked": True, "recovered": True, "seconds": seconds}
        await asyncio.sleep(_RECHECK_S)
    logger.error("her language organ's worker did not come back within %.0fs", FOREGROUND_READY_S)
    return {"checked": True, "recovered": False, "seconds": round(time.monotonic() - started, 1)}


@contextlib.contextmanager
def person_turn(runtime: Any, origin: str) -> Iterator[Any]:
    """Open the turn a person's message opens on the desktop, for a whole run.

    On the desktop a person's message is bound as a turn before any phase runs
    (the chat route, and `cognitive_engine.think` on the kernel's path), and
    the runtime's readers ask `current_turn()` whether one is open: the
    router's endpoint watchdog, the gate, fact custody, turn effects. This
    driver runs the phases itself and never opened one, so in a whole run a
    generation somebody waited for was treated as unowned work. The router then
    armed its thread watchdog at the flat budget for a short reply, and on 23
    September it stopped her cortex's worker 105 s into a tool call that was
    still decoding. For an owned turn the desktop lets work that is still
    advancing run on.

    The reply is marked served, or nothing served, and the turn finalized
    under the name the desktop's chat route uses for a person's turn. The
    kernel's path names `cognitive_engine`, which is registered fail-closed:
    in a whole dry run on 24 September a turn that ended with an answer
    available but never served was escalated to a critical failure and raised,
    ending the run. On the desktop the chat route owns a person's turn and the
    same ending is recorded and survived.

    Only a person's turn in a whole run. The stub organ never reaches the
    router or the gate, and no path notes the stub's reply, so finalizing
    there would record a turn that served nothing on every conversation turn
    of a campaign: the stub organism's error signals changed for bookkeeping.
    That difference between the two organisms stays, and is said here.
    """
    if not getattr(runtime, "whole", False) or origin not in USER_ORIGINS:
        yield None
        return
    from core.runtime.turn_outcome import (
        TurnOutcome,
        UserVisibleState,
        bind_turn,
        finalize_turn,
    )

    outcome = TurnOutcome(origin=origin)
    try:
        with bind_turn(outcome):
            yield outcome
        reply = str(getattr(runtime.state.cognition, "last_response", "") or "").strip()
        if reply:
            outcome.mark_served(reply)
        else:
            outcome.mark_served("", state=UserVisibleState.NOTHING_SERVED)
    finally:
        finalize_turn(outcome, subsystem="chat")


def opens_the_turn(turn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Run a driver turn the way the desktop runs a message.

    A stopped language worker is brought back before anything else, outside
    any turn, so its warmup is not taken for a person's request. A person's
    turn in a whole run is then open for every phase (`person_turn`). A
    decorator rather than a second method, because the driver's class is at
    its method ceiling.
    """

    @functools.wraps(turn)
    async def turn_once(runtime: Any, condition: Any, **kwargs: Any) -> Any:
        await keep_language_ready(runtime)
        with person_turn(runtime, condition.origin):
            return await turn(runtime, condition, **kwargs)

    return turn_once


def phase_budget(runtime: Any, name: str, origin: str, *, fallback: float) -> float:
    """The phase's budget: live's when she runs whole, the harness's otherwise."""
    if not getattr(runtime, "whole", False):
        return fallback
    try:
        return float(
            runtime.kernel._phase_timeout_seconds(name, priority=origin in USER_ORIGINS)
        )
    except (AttributeError, TypeError, ValueError):
        return fallback


async def within_budget(work: Any, budget: float) -> Any:
    """Run a phase for its budget plus whatever it spent waiting on her language organ.

    A kept answer comes back at once and a fresh one takes seconds, so the
    arm that generated first would run out of time where the arm that read
    the kept answer did not. The wait on the organ is added back, which is
    why a campaign run whole measures her without the latency pressure the
    desktop puts on a turn, and says so.
    """
    from core.subject.steady_mind import waited

    task = asyncio.ensure_future(work)
    started = time.monotonic()
    organ_before = waited()
    while True:
        used = (time.monotonic() - started) - (waited() - organ_before)
        remaining = budget - used
        if remaining <= 0.0:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                # Ours, from the line above, unless this task is itself
                # being cancelled, in which case it is the caller's.
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
            raise TimeoutError(f"phase ran past its {budget:.0f}s budget")
        done, _ = await asyncio.wait({task}, timeout=min(remaining, 1.0))
        if done:
            return task.result()
