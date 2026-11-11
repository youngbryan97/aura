"""Ask her cortex's worker to keep its weights pinned while voice mode is open, and to let them go after.

Counted, so two voice sessions open at once release only when the last one closes. Blocking (a control message to her
worker, answered within its timeout): call it off the event loop.
"""
from __future__ import annotations

import logging
from typing import Any

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.Voice.Pinned")

_SESSIONS_PINNING = 0
_PINNING_LOCK = checked_lock("voice.pinned_weights")


def _her_cortex_client() -> Any:
    from core.runtime.service_access import resolve_inference_gate

    return getattr(resolve_inference_gate(default=None), "_mlx_client", None)


def _ask(pinned: bool) -> dict[str, Any] | None:
    client = _her_cortex_client()
    send = getattr(client, "_send_worker_control_action", None)
    if not callable(send):
        return None
    try:
        return send("pin_weights" if pinned else "release_weights", timeout_s=10.0)
    except TimeoutError:
        # Her worker answers between jobs. The request stays in its queue and is acted on when it is reached.
        logger.info("📌 her worker had not answered %s within 10 s (busy with a job); it acts on it when it gets there",
                    "pin_weights" if pinned else "release_weights")
        return None
    except (OSError, RuntimeError, ValueError) as exc:  # a voice session never fails for want of a pin
        logger.info("📌 her cortex's weights were not %s: %s: %s", "pinned" if pinned else "released",
                    type(exc).__name__, exc)
        return None


def a_voice_session_opened() -> dict[str, Any] | None:
    """Pin her weights for the first voice session open."""
    global _SESSIONS_PINNING
    with _PINNING_LOCK:
        _SESSIONS_PINNING += 1
        first = _SESSIONS_PINNING == 1
    if not first:
        return None
    reply = _ask(True)
    if reply is not None:
        logger.info("📌 her cortex's weights pinned for voice mode: %s", reply)
    return reply


def a_voice_session_closed() -> dict[str, Any] | None:
    """Release her weights when the last voice session closes."""
    global _SESSIONS_PINNING
    with _PINNING_LOCK:
        _SESSIONS_PINNING = max(0, _SESSIONS_PINNING - 1)
        last = _SESSIONS_PINNING == 0
    if not last:
        return None
    reply = _ask(False)
    if reply is not None:
        logger.info("📌 her cortex's weights released after voice mode: %s", reply)
    return reply
