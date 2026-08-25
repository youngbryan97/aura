"""The corpus the endogenous head is fitted on: her own state, her own words.

A head trained on invented pairs would measure nothing. So the pairs come from
what actually happened — the cognitive state Aura held when a generation
started, and the text that generation produced. Two decisions keep this cheap
enough to leave on:

* **The turn boundary, not the token.** Recording per token would put a write
  in the decode loop. The state does not change during one generation anyway,
  so one record per turn holds the same information.
* **Text now, tokens later.** The tokenizer lives in the worker process. The
  recorder stores the text; the trainer tokenizes when it runs, against the
  tokenizer the head will be bound to. Storing token ids here would bind the
  corpus to whichever model happened to be resident that day.

The store is bounded, rotated, and off by default for anything that is not a
real turn. Nothing is written without a governed scope, and nothing is written
from an event loop without the async path.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.brain.llm.endogenous_state import (
    STATE_DIM,
    EndogenousState,
    layout_digest,
)
from core.runtime.flags import FlagKind, declare

logger = logging.getLogger("Aura.EndogenousPairs")

#: Longest reply text kept per record. Long enough to hold a real answer,
#: short enough that a day of traffic is megabytes rather than gigabytes.
MAX_TEXT_CHARS = 4000

#: Roll the active file at this size, and keep this many rolled files. Both
#: bounds exist because an unbounded training corpus on the live host is an
#: outage waiting for a busy week.
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_ROLLED_FILES = 8

_PAIR_DIR_FLAG = declare(
    "AURA_ENDOGENOUS_PAIR_DIR",
    kind=FlagKind.STRING,
    default="",
    description="Directory the recorded (state, reply) corpus is written to",
    owner="core.brain.llm.endogenous_pair_recorder",
)
_RECORD_FLAG = declare(
    "AURA_ENDOGENOUS_RECORD",
    kind=FlagKind.BOOL,
    default=True,
    description="Whether live turns are recorded as training pairs for the endogenous head",
    owner="core.brain.llm.endogenous_pair_recorder",
)

#: Rotation and append are one storage transaction inside this process. Without
#: this lock, two completed turns can both rotate the same active file and one
#: can replace the other's generation before either appends.
_STORE_LOCK = threading.Lock()


def store_directory() -> Path:
    raw = str(_PAIR_DIR_FLAG.value() or "").strip()
    if raw:
        return Path(raw)
    return Path("data/endogenous_language")


def recording_enabled() -> bool:
    """Off means off. A corpus nobody asked for is a corpus nobody audited."""
    return bool(_RECORD_FLAG.value())


@dataclass(frozen=True)
class RecordedPair:
    """One turn: the state that held, and the words that came out."""

    values: np.ndarray
    present: np.ndarray
    text: str
    lane: str
    model: str
    recorded_at: float
    prompt_digest: str = ""

    @property
    def coverage(self) -> float:
        return float(np.mean(self.present)) if self.present.size else 0.0


def _active_path() -> Path:
    return store_directory() / "pairs.jsonl"


def _encode_present(present: np.ndarray) -> str:
    return "".join("1" if bool(p) else "0" for p in present)


def _decode_present(encoded: Any) -> np.ndarray | None:
    text = str(encoded or "")
    if len(text) != STATE_DIM or set(text) - {"0", "1"}:
        return None
    return np.asarray([c == "1" for c in text], dtype=bool)


def _record_payload(
    state: EndogenousState,
    text: str,
    *,
    lane: str,
    model: str,
    prompt: str,
) -> dict[str, Any]:
    return {
        "v": 1,
        "layout": layout_digest(),
        "t": round(time.time(), 3),
        "lane": str(lane or "")[:48],
        "model": os.path.basename(str(model or ""))[:96],
        "coverage": round(state.coverage, 4),
        "z": [round(float(v), 5) for v in state.values],
        "p": _encode_present(state.present),
        "prompt_sha": hashlib.blake2b(
            str(prompt or "").encode("utf-8", errors="ignore"), digest_size=8
        ).hexdigest(),
        "text": str(text or "")[:MAX_TEXT_CHARS],
    }


def _should_record(state: EndogenousState, text: str) -> bool:
    if not recording_enabled():
        return False
    if state.interventions:
        # An intervened state is an experiment. Training on it would fit the
        # head to conditions the runtime never actually held.
        return False
    if state.coverage <= 0.0:
        return False
    return bool(str(text or "").strip())


def record_pair(
    state: EndogenousState,
    text: str,
    *,
    lane: str = "",
    model: str = "",
    prompt: str = "",
) -> bool:
    """Append one pair. Returns whether anything was written."""
    if not _should_record(state, text):
        return False
    payload = _record_payload(state, text, lane=lane, model=model, prompt=prompt)
    line = json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        gateway = get_file_write_gateway()
        target = _active_path()
        with local_internal_governed_scope("endogenous_pair_recorder"):
            with _STORE_LOCK:
                gateway.ensure_directory(target.parent, source="endogenous_pair_recorder")
                _rotate_if_needed(
                    gateway,
                    target,
                    incoming_bytes=len(line.encode("utf-8")),
                )
                gateway.append_text(target, line, source="endogenous_pair_recorder")
        return True
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("endogenous pair not recorded: %s", exc)
        return False


async def record_pair_async(
    state: EndogenousState,
    text: str,
    *,
    lane: str = "",
    model: str = "",
    prompt: str = "",
) -> bool:
    """Run the complete bounded storage transaction away from the event loop."""
    return await asyncio.to_thread(
        record_pair,
        state,
        text,
        lane=lane,
        model=model,
        prompt=prompt,
    )


def _rotation_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + f"-{time.time_ns()}"


def _rotate_if_needed(gateway: Any, target: Path, *, incoming_bytes: int) -> None:
    """Roll the active file at the size bound and drop the oldest roll."""
    try:
        if (
            not target.exists()
            or target.stat().st_size + max(0, int(incoming_bytes)) <= MAX_FILE_BYTES
        ):
            return
    except OSError:
        return
    stamp = _rotation_stamp()
    gateway.move_path(target, target.with_name(f"pairs-{stamp}.jsonl"),
                      source="endogenous_pair_recorder")
    rolled = sorted(target.parent.glob("pairs-*.jsonl"))
    for stale in rolled[:-MAX_ROLLED_FILES]:
        gateway.delete_file(stale, source="endogenous_pair_recorder")


def iter_pairs(
    *,
    directory: Path | None = None,
    limit: int | None = None,
    require_layout: bool = True,
) -> Iterator[RecordedPair]:
    """Read the corpus back, newest files last, skipping anything malformed.

    Records written against an older channel layout are skipped by default.
    Mixing layouts would fit one matrix to two different meanings of the same
    column, which is the quiet way to get a head that measures well and means
    nothing.
    """
    root = Path(directory) if directory is not None else store_directory()
    current = layout_digest()
    files = sorted(root.glob("pairs-*.jsonl")) + [root / "pairs.jsonl"]
    seen = 0
    for path in files:
        if not path.exists():
            continue
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError as exc:
            logger.debug("endogenous corpus file unreadable (%s): %s", path, exc)
            continue
        with handle:
            for line in handle:
                if limit is not None and seen >= limit:
                    return
                pair = _parse_line(line, current=current, require_layout=require_layout)
                if pair is None:
                    continue
                seen += 1
                yield pair


def _parse_line(
    line: str, *, current: str, require_layout: bool
) -> RecordedPair | None:
    try:
        payload = json.loads(line)
    except ValueError:
        return None
    if not isinstance(payload, Mapping):
        return None
    if require_layout and str(payload.get("layout") or "") != current:
        return None
    present = _decode_present(payload.get("p"))
    if present is None:
        return None
    try:
        values = np.asarray(payload.get("z") or [], dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if values.shape != (STATE_DIM,) or not np.all(np.isfinite(values)):
        return None
    text = str(payload.get("text") or "")
    if not text.strip():
        return None
    return RecordedPair(
        values=values,
        present=present,
        text=text,
        lane=str(payload.get("lane") or ""),
        model=str(payload.get("model") or ""),
        recorded_at=float(payload.get("t") or 0.0),
        prompt_digest=str(payload.get("prompt_sha") or ""),
    )


def corpus_summary(directory: Path | None = None) -> dict[str, Any]:
    """How much corpus exists, and how much of it is usable at this layout.

    ``usable_records`` being far below ``total_records`` means the layout
    changed under the corpus, which is a reason to retrain and not a reason to
    worry.
    """
    root = Path(directory) if directory is not None else store_directory()
    total = 0
    usable = 0
    lanes: dict[str, int] = {}
    models: dict[str, int] = {}
    earliest = 0.0
    latest = 0.0
    current = layout_digest()
    for path in sorted(root.glob("pairs*.jsonl")):
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError:
            continue
        with handle:
            for line in handle:
                total += 1
                pair = _parse_line(line, current=current, require_layout=True)
                if pair is None:
                    continue
                usable += 1
                lanes[pair.lane] = lanes.get(pair.lane, 0) + 1
                models[pair.model] = models.get(pair.model, 0) + 1
                earliest = pair.recorded_at if not earliest else min(earliest, pair.recorded_at)
                latest = max(latest, pair.recorded_at)
    return {
        "directory": str(root),
        "total_records": total,
        "usable_records": usable,
        "layout": current,
        "lanes": lanes,
        "models": models,
        "earliest": earliest,
        "latest": latest,
        "recording_enabled": recording_enabled(),
    }


__all__ = [
    "MAX_FILE_BYTES",
    "MAX_ROLLED_FILES",
    "MAX_TEXT_CHARS",
    "RecordedPair",
    "corpus_summary",
    "iter_pairs",
    "record_pair",
    "record_pair_async",
    "recording_enabled",
    "store_directory",
]


# ──────────────────────────────────────────────────────────────────────────
# Pairing a request with the response that comes back for it.
#
# The state goes out on the request and the text comes back on the response,
# and the two arrive at different moments in different frames. This is the
# only thing that holds them together, so it is bounded: a request whose
# response never arrives must not keep its state alive forever.
# ──────────────────────────────────────────────────────────────────────────

_PENDING_LIMIT = 64
_PENDING: OrderedDict[str, tuple[dict[str, Any], str, str, float]] = OrderedDict()
_PENDING_LOCK = threading.Lock()

#: A request older than this lost its response. Dropped rather than paired
#: with whatever comes next.
_PENDING_TTL_S = 900.0


def remember_pending(
    request_id: str, payload: Mapping[str, Any], *, lane: str = "", model: str = ""
) -> None:
    """Hold the state that went out with one request until its reply lands."""
    key = str(request_id or "").strip()
    if not key or not payload:
        return
    now = time.time()
    with _PENDING_LOCK:
        _PENDING[key] = (dict(payload), str(lane), str(model), now)
        while len(_PENDING) > _PENDING_LIMIT:
            _PENDING.popitem(last=False)
        stale = [k for k, (_, _, _, at) in _PENDING.items() if now - at > _PENDING_TTL_S]
        for k in stale:
            _PENDING.pop(k, None)


def record_response(request_id: str, text: str) -> bool:
    """Pair a returned reply with the state that produced it, and store both."""
    key = str(request_id or "").strip()
    if not key:
        return False
    with _PENDING_LOCK:
        held = _PENDING.pop(key, None)
    if held is None:
        return False
    payload, lane, model, _at = held
    state = EndogenousState.from_payload(payload)
    if state is None:
        return False
    return record_pair(state, text, lane=lane, model=model)


def pending_depth() -> int:
    with _PENDING_LOCK:
        return len(_PENDING)


def reset_pending() -> None:
    """Drop every held request. For tests, and for a worker that was replaced."""
    with _PENDING_LOCK:
        _PENDING.clear()


__all__ += ["pending_depth", "record_response", "remember_pending", "reset_pending"]
