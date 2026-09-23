"""Her language organ in a campaign: the router the desktop uses, decoding greedily.

Every campaign so far ran with `DeterministicMind` in the language organ's
place. It answered every prompt with the same sentence and every structured
call with the same JSON, so every organ that reads what she said, and every
judgement the organ makes for the rest of her, was a constant. The battery was
measuring Aura with one organ replaced by a stub.

This puts the organ back. Calls go to the router the desktop runtime builds,
which serves her cortex and her faster tiers from the same workers and weights,
with the temperature set to zero so the decode is the greedy one. Two arms of a
paired trial that send the same call in the same state get the same answer: the
first answer is kept, keyed on everything the call carried and on the state her
steering hooks will read, and the second arm reads it. The state is in the key
because her feelings reach the forward pass through those hooks and not through
the call (core/consciousness/steering_channel.py); keyed on the call alone, an
arm that moved only her feelings would have been handed the other arm's answer.
When an intervention changes what reaches the organ, the organ answers
differently, and that difference is the intervention propagating through her
language, not sampling noise.

The router and the kept answers live here at module scope, in core.subject, on
purpose. The fork carries every organ attribute and every module global of the
organism by value, and this package is on its list of machinery that is never
carried: a fork must not copy the router's workers, and rewinding the kept
answers would make the second arm generate again.

`waited()` reports how long calls have spent waiting on the organ, so the driver
can give each phase its live budget plus that wait. Without it, the arm that
generated first could run out of time where the arm that read the kept answer
did not, and the difference would be the order the arms ran in.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from typing import Any

__all__ = ["SteadyMind", "forget_for_test", "kept", "waited"]

#: The router the organ speaks through, and every answer it has given.
_ROUTER: dict[str, Any] = {}
_KEPT: dict[str, Any] = {}
_WAITED: list[float] = [0.0]
_IN_FLIGHT: dict[int, float] = {}
_TICKETS = itertools.count()

#: Call arguments that carry a place to put things rather than a request.
_TRANSPORT = frozenset({"callback", "on_token", "stream_callback", "cancel_event"})


def _key(method: str, args: tuple[Any, ...], kwargs: dict[str, Any], steering: Any = None) -> str:
    request = {
        key: value
        for key, value in kwargs.items()
        if not key.startswith("_") and key not in _TRANSPORT and not callable(value)
    }
    blob = json.dumps(
        {"method": method, "args": list(args), "kwargs": request, "steering": steering},
        sort_keys=True,
        default=repr,
    )
    return hashlib.sha256(blob.encode("utf-8", "ignore")).hexdigest()


def kept() -> int:
    """How many distinct calls the organ has answered."""
    return len(_KEPT)


def waited() -> float:
    """Seconds spent waiting on the organ since the process began, counting calls still running."""
    now = time.monotonic()
    return _WAITED[0] + sum(now - started for started in _IN_FLIGHT.values())


def forget_for_test() -> None:
    _ROUTER.clear()
    _KEPT.clear()
    _IN_FLIGHT.clear()
    _WAITED[0] = 0.0


class SteadyMind:
    """The language organ, through the live router, greedy, answering a repeated call as before."""

    #: Set by the driver before each turn, as it sets the stub's. Nothing here
    #: reads it: the answer is a function of the call.
    moment: int = 0

    def __init__(self, router: Any) -> None:
        _ROUTER["router"] = router

    @staticmethod
    def _router() -> Any:
        router = _ROUTER.get("router")
        if router is None:
            raise RuntimeError("the steady mind has no router to speak through")
        return router

    async def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        from core.consciousness.steering_channel import steering_now

        key = _key(method, args, kwargs, steering=steering_now())
        if key in _KEPT:
            return _KEPT[key]
        target = getattr(self._router(), method)
        kwargs["temperature"] = 0.0
        started = time.monotonic()
        ticket = next(_TICKETS)
        _IN_FLIGHT[ticket] = started
        try:
            answer = await target(*args, **kwargs)
        finally:
            _IN_FLIGHT.pop(ticket, None)
            _WAITED[0] += time.monotonic() - started
        # An empty answer is a lane that did not answer, not an answer, and
        # keeping it would replay a transient failure for the rest of the run.
        if answer:
            _KEPT[key] = answer
        return answer

    async def think(self, prompt: Any = None, *args: Any, **kwargs: Any) -> Any:
        return await self._call("think", prompt, *args, **kwargs)

    async def generate(self, prompt: Any = None, *args: Any, **kwargs: Any) -> Any:
        return await self._call("generate", prompt, *args, **kwargs)

    async def generate_with_metadata(self, prompt: Any = None, *args: Any, **kwargs: Any) -> Any:
        return await self._call("generate_with_metadata", prompt, *args, **kwargs)

    async def classify(self, prompt: Any = None, *args: Any, **kwargs: Any) -> Any:
        return await self._call("classify", prompt, *args, **kwargs)

    # The stub's other names, which callers use and the router spells `think`.
    async def route(self, prompt: Any = None, **kwargs: Any) -> Any:
        return await self.think(prompt, **kwargs)

    async def chat(self, prompt: Any = None, **kwargs: Any) -> Any:
        return await self.think(prompt, **kwargs)

    async def complete(self, prompt: Any = None, **kwargs: Any) -> Any:
        return await self.think(prompt, **kwargs)

    async def embed(self, text: Any, **kwargs: Any) -> Any:
        router = self._router()
        if hasattr(router, "embed"):
            return await self._call("embed", text, **kwargs)
        return [0.0] * 8

    def get_stats(self) -> dict[str, Any]:
        # The body reads token velocity off this, and with the organ running
        # that is a real reading of how hard it worked.
        router = self._router()
        stats = router.get_stats() if hasattr(router, "get_stats") else {}
        return dict(stats or {})

    @property
    def high_pressure_mode(self) -> bool:
        return bool(getattr(self._router(), "high_pressure_mode", False))

    def __getattr__(self, name: str) -> Any:
        # Anything else a phase asks the organ for is the router's to answer.
        # Not kept: a method outside the ones above is not a generation.
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._router(), name)
