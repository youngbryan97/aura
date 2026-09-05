"""core/personality_integration.py - Personality Systems Integration
Bridge for enforcing Aura's technical personality across all modules.

CP126 found this reporting success it had not established. The function
returned True after calling two helpers that each succeed by doing nothing —
``_patch_orchestrator`` logged "output integrity filter active" whether or not
a reply queue existed — and ``verify_all_systems_aligned`` attested alignment
by comparing a hard-coded version string, ignoring its orchestrator argument
entirely. Repeated calls stacked filter closures with no marker or teardown,
and the reply filter rewrote the caller's own dict in place.

The rules now match the ones used for persona_integration:

* **Success means a hook was installed.** The result is a receipt naming what
  was installed and what was not; it is falsy when nothing was.
* **Installation is idempotent** and reversible, with an unwrap handle.
* **Attestation inspects the live wiring**, not a version constant.

CP126 fdbc444b / cd83e384 / 4504ac11 / 2ed17445 / e36030ad.
"""
from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Integration")

#: Attributes stamped on an installed wrapper so a second call recognizes its
#: own work rather than wrapping the wrapper (CP126 4504ac11).
_MARKER = "__aura_personality_filter__"
_ORIGINAL = "__aura_personality_original__"

_INTEGRATION_ERRORS = (
    AttributeError,
    ImportError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass
class PersonalityIntegrationReceipt:
    """What was actually installed.

    CP126 fdbc444b / cd83e384: the old function returned a bare bool that was
    True whether or not any hook existed, and False without quarantining
    anything — a caller ignoring the boolean simply ran unfiltered.
    """

    installed: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    engine_available: bool = False

    @property
    def ok(self) -> bool:
        """True only when at least one real filter is in place."""
        return bool(self.installed) and not self.errors

    def __bool__(self) -> bool:
        return self.ok

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "installed": list(self.installed),
            "skipped": list(self.skipped),
            "errors": list(self.errors),
            "engine_available": self.engine_available,
        }


def integrate_all_personality_systems(orchestrator) -> PersonalityIntegrationReceipt:
    """Integrate identity parameters across all system components.

    Returns a receipt rather than a bare bool. The receipt is falsy when no
    filter was installed, so a caller that treats it as a boolean gets the
    honest answer instead of an unconditional True.
    """
    logger.info("-" * 40)
    logger.info("Initiating personality integration")
    logger.info("-" * 40)

    receipt = PersonalityIntegrationReceipt()
    try:
        from .personality_engine import get_personality_engine

        engine = get_personality_engine()
        receipt.engine_available = engine is not None
    except _INTEGRATION_ERRORS as exc:
        record_degradation('personality_integration', exc)
        logger.error("Personality integration failed: %s", exc)
        receipt.errors.append(f"personality_engine_unavailable: {exc}")
        return receipt

    if engine is None or not callable(getattr(engine, "filter_response", None)):
        receipt.errors.append("personality_engine_has_no_filter_response")
        logger.error("Personality engine exposes no filter_response(); nothing installed.")
        return receipt

    # 1. Patch Proactive Communication
    comm = getattr(orchestrator, "proactive_comm", None)
    if comm is None:
        receipt.skipped.append("proactive_comm:absent")
    else:
        _install(receipt, "proactive_comm", lambda: _patch_proactive_comm(comm, engine))

    # 2. Patch Orchestrator Response Expresser
    _install(receipt, "reply_queue", lambda: _patch_orchestrator(orchestrator, engine))

    if receipt.ok:
        logger.info("Identity core integrated — filters: %s", receipt.installed)
    else:
        # CP126 cd83e384: say plainly that outgoing text is unfiltered, rather
        # than logging "integrity filter active" and returning a bare False.
        logger.error(
            "Personality integration installed NO output filter (skipped=%s errors=%s); "
            "outgoing messages are NOT identity-filtered.",
            receipt.skipped, receipt.errors,
        )
        record_degradation(
            'personality_integration',
            RuntimeError("no personality filter installed"),
            action="reported unfiltered outgoing messages to the caller",
            severity="error",
        )
    return receipt


def _install(
    receipt: PersonalityIntegrationReceipt, name: str, action: Callable[[], bool]
) -> None:
    try:
        if action():
            receipt.installed.append(name)
        else:
            receipt.skipped.append(f"{name}:no_hook_point")
    except _INTEGRATION_ERRORS as exc:
        record_degradation('personality_integration', exc)
        logger.error("Personality filter %s failed to install: %s", name, exc)
        receipt.errors.append(f"{name}: {exc}")


def _patch_orchestrator(orchestrator, engine) -> bool:
    """Add a final filter to all outgoing messages to catch 'Assistant' leaks.

    Returns whether a filter was actually installed. CP126 fdbc444b: this
    logged "Orchestrator output integrity filter active" unconditionally, even
    with no reply queue to filter.
    """
    queue = getattr(orchestrator, "reply_queue", None)
    put = getattr(queue, "put_nowait", None)
    if queue is None or not callable(put):
        logger.info("Orchestrator has no reply queue; no output filter installed.")
        return False
    if getattr(put, _MARKER, False):
        logger.debug("Orchestrator reply queue filter already installed")
        return True

    original_put = put

    @functools.wraps(original_put)
    def filtered_put(item):
        # CP126 2ed17445: the old filter assigned item['message'] on the
        # CALLER'S dict before enqueueing, so retries, audit code and other
        # consumers observed a personality-shaped value in place of the
        # original evidence. The filtered payload is a copy; the caller's
        # object is untouched and the pre-filter text is preserved.
        try:
            if isinstance(item, str):
                return original_put(engine.filter_response(item))
            if isinstance(item, dict) and "message" in item:
                shaped = dict(item)
                shaped["message"] = engine.filter_response(item.get("message"))
                shaped.setdefault("unfiltered_message", item.get("message"))
                return original_put(shaped)
        except _INTEGRATION_ERRORS as exc:
            record_degradation(
                'personality_integration',
                exc,
                action="enqueued the unfiltered message after the personality filter failed",
                severity="warning",
            )
        return original_put(item)

    setattr(filtered_put, _MARKER, True)
    setattr(filtered_put, _ORIGINAL, original_put)
    queue.put_nowait = filtered_put
    logger.info("Orchestrator reply queue filter active")
    return True


def _patch_proactive_comm(comm, engine) -> bool:
    """Ensure autonomous messages are also filtered."""
    original_queue = getattr(comm, "queue_message", None)
    if not callable(original_queue):
        return False
    if getattr(original_queue, _MARKER, False):
        logger.debug("Proactive communication filter already installed")
        return True

    @functools.wraps(original_queue)
    def filtered_queue(content, emotion, urgency, context=None):
        try:
            content = engine.filter_response(content)
        except _INTEGRATION_ERRORS as exc:
            record_degradation(
                'personality_integration',
                exc,
                action="queued the unfiltered autonomous message after the filter failed",
                severity="warning",
            )
        return original_queue(content, emotion, urgency, context)

    setattr(filtered_queue, _MARKER, True)
    setattr(filtered_queue, _ORIGINAL, original_queue)
    comm.queue_message = filtered_queue
    logger.info("Proactive communication aligned with personality core")
    return True


def uninstall_personality_systems(orchestrator) -> List[str]:
    """Remove installed filters and restore the original callables.

    CP126 4504ac11: there was no uninstall path at all, so a re-integration
    stacked another closure over the previous one and retained stale engine
    and communication objects for the process lifetime.
    """
    removed: List[str] = []
    queue = getattr(orchestrator, "reply_queue", None)
    put = getattr(queue, "put_nowait", None)
    original = getattr(put, _ORIGINAL, None)
    if original is not None:
        queue.put_nowait = original
        removed.append("reply_queue")

    comm = getattr(orchestrator, "proactive_comm", None)
    queue_message = getattr(comm, "queue_message", None)
    original_queue = getattr(queue_message, _ORIGINAL, None)
    if original_queue is not None:
        comm.queue_message = original_queue
        removed.append("proactive_comm")
    return removed


def personality_integration_status(orchestrator) -> Dict[str, Any]:
    """Which filters are installed right now, by inspection."""
    queue = getattr(orchestrator, "reply_queue", None)
    comm = getattr(orchestrator, "proactive_comm", None)
    return {
        "reply_queue_filtered": bool(
            getattr(getattr(queue, "put_nowait", None), _MARKER, False)
        ),
        "proactive_comm_filtered": bool(
            getattr(getattr(comm, "queue_message", None), _MARKER, False)
        ),
    }


def verify_all_systems_aligned(orchestrator) -> bool:
    """Check if the personality core is active and verified.

    CP126 e36030ad: this ignored its orchestrator argument and returned
    whether a separately constructed legacy kernel reported version "3.5.5" —
    so it could attest alignment while no integration existed at all. It now
    inspects the live wiring: a usable engine AND at least one installed
    filter.
    """
    status = personality_integration_status(orchestrator)
    if not any(status.values()):
        logger.warning("Personality alignment check failed: no output filter is installed")
        return False
    try:
        from .personality_engine import get_personality_engine

        engine = get_personality_engine()
    except _INTEGRATION_ERRORS as exc:
        logger.warning("Personality alignment check failed: engine unavailable (%s)", exc)
        return False
    if engine is None or not callable(getattr(engine, "filter_response", None)):
        logger.warning("Personality alignment check failed: engine cannot filter")
        return False
    return True
