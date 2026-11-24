"""What has worked in the programs she made before: the mends that were kept, how often each sort of mend is, and the functions
of programs whose checks hold.

Each was kept only for what trying it showed, so it is something she knows and not something she was told (core/forge/
regression_memory.py and core/rebuilding/parts_that_worked.py keep the same of her own source and of rebuilt programs). A
failure like one she has mended is mended with that beside it; the sorts of mend that have paid are tried before those that
have not; and a program is written with the functions that worked in earlier ones like it to start from.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Making.WhatHasWorked")

__all__ = ["Kept", "functions_like", "get_kept"]

#: How many mends are held, the newest; and how much of a function is shown.
HELD, SHOWN = 400, 1400


class Kept:
    """The mends that were kept, and how often each sort of mend was tried and kept."""

    def __init__(self, path: Path | None = None) -> None:
        self._path, self._lock = path, checked_lock("making.what_has_worked")
        self._held: dict[str, Any] | None = None

    def _where(self) -> Path:
        if self._path is None:
            from core.runtime.state_ownership import state_root

            self._path = state_root() / "data" / "making" / "what_has_worked.json"
        return self._path

    def _load(self) -> dict[str, Any]:
        if self._held is None:
            held: Any = None
            try:
                held = json.loads(self._where().read_text(encoding="utf-8")) if self._where().is_file() else None
            except (OSError, ValueError) as exc:
                logger.warning("What has worked is not read (%s); it begins again.", exc)
            self._held = held if isinstance(held, dict) else {}
            self._held.setdefault("mends", [])
            self._held.setdefault("sorts", {})
        return self._held

    def tried(self, sort: str, *, kept: bool) -> None:
        """One mend of a sort tried, and whether it was kept."""
        with self._lock:
            row = self._load()["sorts"].setdefault(sort, [0, 0])
            row[0] += 1
            row[1] += int(kept)

    def pays(self, sort: str) -> float:
        """How likely a mend of a sort is to be kept, as it has gone: once in two where it has never been tried."""
        with self._lock:
            tried, kept = self._load()["sorts"].get(sort, (0, 0))
        return (kept + 1) / (tried + 2)

    def mended(self, *, kind: str, failing: Iterable[str], why: str, says: str, sort: str) -> None:
        """A mend that was kept: what had failed, and what was done."""
        with self._lock:
            held = self._load()["mends"]
            held.append({"kind": kind, "failing": [str(name)[:120] for name in failing][:4], "why": str(why)[:300], "says": str(says)[:400],
                         "sort": sort, "at": time.time()})
            del held[:-HELD]

    def like(self, kind: str, why: str, *, at_most: int = 2) -> list[str]:
        """What mended the failures most like this one, by her encoder where it is ready; none where it is not."""
        with self._lock:
            mends = [m for m in self._load()["mends"] if m.get("kind") == kind and m.get("says")]
        if not mends or not str(why).strip():
            return []
        from core.cognition.evidence_relevance import vectors_if_ready

        vectors = vectors_if_ready([str(why)[:400], *(m["why"] for m in mends)])
        if not vectors or len(vectors) != len(mends) + 1:
            return []
        import numpy as np

        v = np.asarray(vectors, dtype=np.float32)
        near = v[1:] @ v[0] / (np.linalg.norm(v[1:], axis=1) * np.linalg.norm(v[0]) + 1e-9)
        out: list[str] = []
        # Nearer than this failure's likeness to the rest of them, which is what says it is a likeness at all.
        enough = float(near.mean() + near.std()) if len(near) > 2 else -1.0
        for i in np.argsort(-near):
            if near[i] >= enough and (said := f"{mends[int(i)]['why'][:160]}: {mends[int(i)]['says'][:260]}") not in out:
                out.append(said)
            if len(out) >= at_most:
                break
        return out

    def keep(self) -> None:
        """Write what is held."""
        from core.rebuilding.her_build_writes import governed

        with self._lock:
            text = json.dumps(self._load(), indent=1, ensure_ascii=False)
        governed("ensure_directory", self._where().parent, source="making.what_has_worked")
        governed("write_text", self._where(), text, source="making.what_has_worked")


def functions_like(wanted: Sequence[str], where: Path | None, *, kind: str, at_most: int = 2, but: Path | None = None) -> list[str]:
    """The functions of earlier programs of this kind, whose every check held, that do what these say, the most alike first."""
    if where is None or not wanted or not Path(where).is_dir():
        return []
    found: list[tuple[str, str]] = []
    for path in sorted(Path(where).glob("*/program.json"), key=lambda p: -p.stat().st_mtime)[:60]:
        if but is not None and path.parent == but:
            continue
        try:
            made = json.loads(path.read_text(encoding="utf-8"))
            if made["design"]["kind"] != kind or not made.get("tried") or not all(row.get("ok") for row in made["tried"]):
                continue
            does = {line.split("(", 1)[0].strip(): line for line in made["design"].get("functions") or []}
            found += [(does[name], text) for name, text in made["parts"]["functions"].items() if name in does and len(text) <= SHOWN]
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue  # not a failure: a folder of another maker's is not a program of this one's
    if not found:
        return []
    from core.cognition.evidence_relevance import vectors_if_ready

    vectors = vectors_if_ready([*wanted, *(does for does, _text in found)])
    if not vectors or len(vectors) != len(wanted) + len(found):
        return []
    import numpy as np

    v = np.asarray(vectors, dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    mine, theirs = v[: len(wanted)], v[len(wanted):]
    best = (theirs @ mine.T).max(axis=1)
    return [found[int(i)][1] for i in np.argsort(-best)[:at_most]]


_KEPT: Kept | None = None
_LOCK = checked_lock("making.what_has_worked.singleton")


def get_kept() -> Kept:
    global _KEPT
    with _LOCK:
        if _KEPT is None:
            _KEPT = Kept()
        return _KEPT
