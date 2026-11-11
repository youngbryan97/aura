"""Her model's weights pinned in memory while a person is talking with her in voice mode.

mlx_lm wires a model's memory only while it generates (``mlx_lm.generate.wired_limit`` raises the limit, then puts
back the one it found), so between turns her weights are ordinary pages and macOS may page them out. R11, 11 November
2026: her prefill ran at 375 tokens a second a minute after a turn and at 144 after 25 idle minutes. With the limit
raised here, each generation's ``wired_limit`` puts back this one, so they stay resident until released. Bryan chose to
pin only while voice mode is open (about 15 GB the rest of the machine cannot have meanwhile).
"""
from __future__ import annotations

from typing import Any


def her_weights_pinned(pinned: bool, *, device: str, ipc_writer: Any, job: Any, mx: Any, model: Any) -> None:
    """Raise the wired limit to what this machine recommends (pinned) or back to none, and say what it is now."""
    reply: dict[str, Any] = {"id": job.get("id") if isinstance(job, dict) else None, "pinned": False}
    try:
        if device == "cpu" or mx is None or not mx.metal.is_available():
            reply.update(status="ok", detail="no Metal wired memory on this worker's device")
        else:
            limit = int(mx.device_info()["max_recommended_working_set_size"]) if pinned else 0
            previous = int(mx.set_wired_limit(limit))
            weights = 0
            try:
                from mlx.utils import tree_reduce

                weights = int(tree_reduce(lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0))
            except (ImportError, AttributeError, TypeError, ValueError):
                weights = 0  # the size is said when it can be read; the pin does not depend on it
            reply.update(status="ok", pinned=bool(pinned), wired_limit=limit, previous_wired_limit=previous,
                         model_bytes=weights)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        reply.update(status="error", error=f"{type(exc).__name__}: {exc}")
    ipc_writer.put(reply)
