"""core/verify/lesion_registry.py

The system-wide protocol for "turn this faculty off and see".

:mod:`core.verify.causal_influence` can only reach a verdict about a channel if
somebody hands it paired trials, and the pairs only exist if the faculty can be
neutralized on demand. One engine in this codebase already worked that way —
``AffectiveValenceEngine.lesion()`` returns a flat neutral affect so its
contribution is fully ablatable. Everything else claiming causal influence
claimed it without offering any way to check.

This is that capability generalized. A faculty registers how to make itself
neutral; the registry hands the harness a uniform way to do it, so measurement
does not need to know anything about the subsystem it is measuring.

Neutral is not broken. A lesioned faculty must return its no-information
output — a flat affect, an identity transform, an empty context block, alpha
zero — and must not raise, hang, or degrade the run into an error path. A
lesion that crashes the turn measures the crash, not the faculty.

Layering: registration is inverted. Faculties import this; this imports no
faculty. See DEPS.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Verify.LesionRegistry")

#: Which channels are lesioned *for the current execution context*, and to what
#: depth. A ContextVar, not a plain dict, and the distinction is a safety
#: property rather than a style preference.
#:
#: A probe measuring the live runtime lesions a channel and generates. If that
#: flag were process-global, any user turn generating concurrently would be
#: served by the lesioned faculty — the measurement would degrade the very
#: system it is measuring, silently, for a real person mid-conversation.
#:
#: asyncio tasks inherit a copy of the context at creation and their mutations
#: never propagate back, so a lesion set inside a probe task covers that task
#: and everything it awaits, and is invisible to any turn running beside it.
#: ``asyncio.to_thread`` copies the context too. Where generation crosses into
#: a separate worker process, the lesion has already done its work: the values
#: are neutralized while the request is being assembled, and the request
#: carries them.
_ACTIVE_LESIONS: ContextVar[dict[str, int] | None] = ContextVar(
    "aura_active_lesions",
    default=None,
)

__all__ = [
    "LesionHandle",
    "LesionRegistry",
    "get_lesion_registry",
    "register_lesion",
    "lesionable",
    "lesioned",
    "reset_lesion_registry_for_test",
]


@dataclass(frozen=True)
class LesionHandle:
    """How to neutralize one channel, and who is answerable for it."""

    channel: str
    lesion: Callable[[], None]
    restore: Callable[[], None]
    owner: str
    #: What this channel's output becomes when lesioned. Recorded because a
    #: reader of a verdict needs to know what the counterfactual actually was:
    #: "steering alpha forced to 0" and "the whole block omitted from the
    #: prompt" are different questions with different answers.
    neutral_description: str
    #: True when the channel reaches the output as numbers (sampler settings,
    #: steering coefficients, gains) rather than as text in a prompt. The
    #: distinction is the point of the audit: a faculty whose only actuator is
    #: a sentence in the system prompt is doing prompt engineering, however
    #: much machinery sits behind the sentence.
    direct_actuation: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "owner": self.owner,
            "neutral": self.neutral_description,
            "direct_actuation": self.direct_actuation,
        }


class LesionUnavailable(RuntimeError):  # noqa: N818 - compatibility API
    """Raised when a channel is asked to lesion and has not registered how."""


class LesionRegistry:
    """Every faculty that can be turned off, and how."""

    def __init__(self) -> None:
        self._handles: dict[str, LesionHandle] = {}
        self._lock = checked_lock("lesion_registry.handles", reentrant=True)

    # Lesion depth lives in the ContextVar, never on the instance: two probes
    # in two tasks must not see each other's lesions, and neither may be seen
    # by a user turn.
    @staticmethod
    def _depths() -> dict[str, int]:
        return _ACTIVE_LESIONS.get() or {}

    @staticmethod
    def _set_depths(depths: dict[str, int]) -> None:
        _ACTIVE_LESIONS.set(depths)

    def register(self, handle: LesionHandle, *, replace: bool = False) -> None:
        with self._lock:
            existing = self._handles.get(handle.channel)
            if existing is not None and not replace:
                if existing.owner == handle.owner:
                    # Re-import or a second instance of the same owner. Idempotent.
                    return
                raise ValueError(
                    f"lesion channel {handle.channel!r} already registered by "
                    f"{existing.owner!r}; pass replace=True to take it over"
                )
            self._handles[handle.channel] = handle

    def unregister(self, channel: str) -> None:
        with self._lock:
            self._handles.pop(channel, None)

    def get(self, channel: str) -> LesionHandle | None:
        with self._lock:
            return self._handles.get(channel)

    def channels(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._handles))

    def is_registered(self, channel: str) -> bool:
        with self._lock:
            return channel in self._handles

    def is_lesioned(self, channel: str) -> bool:
        return self._depths().get(channel, 0) > 0

    def active_lesions(self) -> tuple[str, ...]:
        return tuple(sorted(name for name, depth in self._depths().items() if depth > 0))

    @contextmanager
    def lesion(self, channel: str) -> Iterator[LesionHandle]:
        """Neutralize ``channel`` for the duration of the block.

        Scoped to the calling execution context, so a trial never reaches a
        turn running beside it. Reentrant by depth: nested lesions of the same
        channel restore once, on the outermost exit. Restoration runs even when
        the body raises — a harness that leaves a faculty lesioned after a
        failed trial silently lobotomizes whatever comes next.
        """

        with self._lock:
            handle = self._handles.get(channel)
        if handle is None:
            raise LesionUnavailable(
                f"channel {channel!r} has no registered lesion; it cannot be "
                "measured, and nothing may claim it is causally influential"
            )

        # Copy-on-write: rebinding the ContextVar to a new dict rather than
        # mutating the one already there. A mutation would be visible through
        # every context that inherited the same object, which is the leak this
        # whole mechanism exists to prevent.
        depths = dict(self._depths())
        depth = depths.get(channel, 0)
        depths[channel] = depth + 1
        token = _ACTIVE_LESIONS.set(depths)
        should_lesion = depth == 0

        try:
            if should_lesion:
                # Stateful faculties (an engine with a real lesion() method)
                # are process-global whatever the ContextVar does. Only the
                # flag-based channels are genuinely concurrency-safe; a
                # stateful lesion is safe because the probe holds it briefly
                # and restores unconditionally, not because it is scoped.
                handle.lesion()
            yield handle
        finally:
            try:
                _ACTIVE_LESIONS.reset(token)
            except ValueError:
                # Token from a different context: the block was entered and
                # exited across a context boundary. Fall back to recomputing.
                remaining = dict(self._depths())
                if remaining.get(channel, 0) <= 1:
                    remaining.pop(channel, None)
                else:
                    remaining[channel] -= 1
                _ACTIVE_LESIONS.set(remaining)
            if should_lesion:
                try:
                    handle.restore()
                except Exception:
                    # A restore that fails leaves the faculty neutralized, which
                    # is far worse than a failed measurement. Say so at CRITICAL
                    # and let the caller's degradation path see the raise.
                    logger.critical(
                        "[LESION] restore FAILED for channel %s (owner %s): the "
                        "faculty is still neutralized",
                        channel,
                        handle.owner,
                    )
                    raise

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            handles = dict(self._handles)
        active = self.active_lesions()

        # Reported against the DECLARED channel set, not just against whatever
        # happens to have been imported. A registry that only knows about
        # modules loaded so far reports full coverage in a process that has
        # loaded nothing, which is the most flattering possible answer and the
        # least true one.
        from core.verify import influence_channels

        declared = set(influence_channels.ALL_CHANNELS)
        registered = set(handles)
        return {
            "registered": {name: h.as_dict() for name, h in sorted(handles.items())},
            "registered_count": len(handles),
            "direct_actuation_count": sum(1 for h in handles.values() if h.direct_actuation),
            "declared_count": len(declared),
            #: Declared channels nothing has offered a lesion for. Claims about
            #: these cannot be checked, and this is where that shows.
            "declared_without_lesion": tuple(sorted(declared - registered)),
            #: Lesions registered for channels outside the declared contract.
            #: Usually a test fixture; in production code it means an id was
            #: minted without being declared.
            "undeclared_registrations": tuple(sorted(registered - declared)),
            "active_lesions": active,
        }


_REGISTRY = LesionRegistry()


def get_lesion_registry() -> LesionRegistry:
    return _REGISTRY


def reset_lesion_registry_for_test() -> None:
    global _REGISTRY
    _REGISTRY = LesionRegistry()


def register_lesion(
    channel: str,
    *,
    lesion: Callable[[], None],
    restore: Callable[[], None],
    owner: str,
    neutral: str,
    direct_actuation: bool,
    replace: bool = False,
) -> LesionHandle:
    """Register how to neutralize ``channel``. The functional entry point."""

    handle = LesionHandle(
        channel=str(channel),
        lesion=lesion,
        restore=restore,
        owner=str(owner),
        neutral_description=str(neutral),
        direct_actuation=bool(direct_actuation),
    )
    _REGISTRY.register(handle, replace=replace)
    return handle


def lesionable(
    channel: str,
    *,
    owner: str,
    neutral: str,
    direct_actuation: bool,
    lesion_method: str = "lesion",
    restore_method: str = "restore",
) -> Callable[[type], type]:
    """Class decorator: register a singleton-ish faculty as lesionable.

    The decorated class must expose ``lesion()`` and ``restore()``. The registry
    binds to the most recently constructed instance, which is what a container
    singleton gives you::

        @lesionable(
            "affect.generation_controls",
            owner="core/being/affective_valence.py",
            neutral="flat affect: neutral valence, no control_effects",
            direct_actuation=True,
        )
        class AffectiveValenceEngine: ...
    """

    def decorate(cls: type) -> type:
        holder: dict[str, Any] = {"instance": None}
        original_init = cls.__init__

        def tracking_init(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            holder["instance"] = self

        cls.__init__ = tracking_init  # type: ignore[method-assign]

        def do(method_name: str) -> Callable[[], None]:
            def run() -> None:
                instance = holder.get("instance")
                if instance is None:
                    raise LesionUnavailable(
                        f"channel {channel!r} is registered against {cls.__name__} "
                        "but no instance has been constructed"
                    )
                getattr(instance, method_name)()

            return run

        register_lesion(
            channel,
            lesion=do(lesion_method),
            restore=do(restore_method),
            owner=owner,
            neutral=neutral,
            direct_actuation=direct_actuation,
            replace=True,
        )
        return cls

    return decorate


@contextmanager
def lesioned(*channels: str) -> Iterator[tuple[LesionHandle, ...]]:
    """Neutralize several channels at once, restoring all of them on exit."""

    registry = get_lesion_registry()
    with ExitStack() as stack:
        handles = tuple(stack.enter_context(registry.lesion(c)) for c in channels)
        yield handles


# ---------------------------------------------------------------------------
# The two-line adoption path
# ---------------------------------------------------------------------------


def register_flag_lesion(
    channel: str,
    *,
    owner: str,
    neutral: str,
    direct_actuation: bool,
) -> LesionHandle:
    """Register a channel whose lesion is just a flag the registry holds.

    Most faculties do not need ``lesion()``/``restore()`` methods and a mutable
    engine object. They need one place — where the channel's value reaches the
    thing it actuates — to substitute a neutral. This registers the flag; pair
    it with :func:`apply_channel` at that site and the channel is measurable.

    Deliberately free of any state of its own beyond the registry's depth
    counter, so a lesion cannot leak into a faculty's persistent state and
    outlive the trial that set it.
    """

    def noop() -> None:
        return None

    return register_lesion(
        channel,
        lesion=noop,
        restore=noop,
        owner=owner,
        neutral=neutral,
        direct_actuation=direct_actuation,
        replace=True,
    )


def apply_channel(channel: str, value: Any, *, neutral: Any) -> Any:
    """The value this channel contributes right now — or its neutral.

    Wrap the point where a faculty's output reaches what it actuates::

        alpha = apply_channel(
            influence_channels.LIVE_MIND_STEERING_ALPHA,
            derived_alpha,
            neutral=STEERING_OFF,
        )

    Outside a measurement trial this returns ``value`` and costs one dict
    lookup, so it is safe on the live generation path. Inside one it returns
    ``neutral``, which is what makes the counterfactual arm real rather than
    simulated: the trial runs the actual production code with the actual
    contribution removed, not a reconstruction of what that might look like.
    """

    if _REGISTRY.is_lesioned(channel):
        return neutral
    return value
