"""The coordinator's anchors on disk, each read when it is restored.

A 370-round anchor holds about 170 MB of her state once decoded, so every
process that decoded the whole bank of 128 held some 22 GB of it. On 23
October two workers and the coordinator came to 57 GB between them, and free
memory fell to 70 MB. Most of a bank is visited rarely: a cut draws anchors
from the first, eight or sixteen at a time, and most cuts are decided at the
first two looks.

So the bank is written in two parts and read on demand. Objects every
snapshot holds by reference (her module objects, and what they hold) are
written once, in a shared section, and each anchor's own state is pickled
apart, pointing into that section. The file is mapped, an anchor's snapshot is
decoded when something restores it, and the most recently restored few are
kept. Decoded, a bank shares what the single pickle shared, so a restore reads
the same state it read before; the earlier single-pickle bank is still read
whole.
"""

from __future__ import annotations

import collections
import io
import mmap
import pickle
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from core.subject.anchor_travel import _Reader, _Writer, fill_snapshot

__all__ = ["BANK_MAGIC", "AnchorBank", "BankedAnchor", "write_anchor_bank"]

BANK_MAGIC = b"AURA-ANCHOR-BANK-2\n"
_LENGTH = struct.Struct("<Q")
_SHARED = "aura-shared"
#: Values with no identity worth keeping: equal ones are interchangeable.
_ATOMIC = (str, bytes, int, float, bool, complex, type(None))


class _Discard:
    def write(self, data: Any) -> int:
        return len(data)


class _Tracker(_Writer):
    """Pickles to nowhere and notes every object with an identity it meets."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.met: dict[int, Any] = {}

    def persistent_id(self, obj: Any) -> Any:
        marker = super().persistent_id(obj)
        if marker is None and not isinstance(obj, _ATOMIC):
            self.met.setdefault(id(obj), obj)
        return marker


class _SharedWriter(_Writer):
    """Writes an anchor's own state, and a reference for anything in the shared section."""

    def __init__(self, *args: Any, shared: Mapping[int, int], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._shared = shared

    def persistent_id(self, obj: Any) -> Any:
        marker = super().persistent_id(obj)
        if marker is not None:
            return marker
        index = self._shared.get(id(obj))
        return None if index is None else (_SHARED, index)


class _SharedReader(_Reader):
    def __init__(self, *args: Any, shared: Sequence[Any], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._shared = shared

    def persistent_load(self, pid: Any) -> Any:
        if isinstance(pid, tuple) and len(pid) == 2 and pid[0] == _SHARED:
            return self._shared[pid[1]]
        return super().persistent_load(pid)


def _pickled(value: Any, writer: type[_Writer] = _Writer, **kwargs: Any) -> bytes:
    buffer = io.BytesIO()
    writer(buffer, protocol=pickle.HIGHEST_PROTOCOL, **kwargs).dump(value)
    return buffer.getvalue()


def _shared_by_reference(snapshots: Sequence[Any]) -> list[Any]:
    """The objects at least two of these snapshots hold by reference, in the order first met.

    Read off the first, the middle and the last snapshot. Something two of
    them share was made before either and is held, not copied; something only
    later anchors share is written with each of them, which costs space and
    changes nothing a restore reads, because a restore copies out of a snapshot.
    """
    picks = sorted({0, len(snapshots) // 2, len(snapshots) - 1})
    seen: collections.Counter[int] = collections.Counter()
    order: dict[int, Any] = {}
    for index in picks:
        tracker = _Tracker(_Discard(), protocol=pickle.HIGHEST_PROTOCOL)
        tracker.dump(snapshots[index])
        for key, obj in tracker.met.items():
            seen[key] += 1
            order.setdefault(key, obj)
    return [obj for key, obj in order.items() if seen[key] >= 2]


def write_anchor_bank(path: Path, *, anchors: Sequence[Any], **fields: Any) -> int:
    """Write the anchors and the fields beside them; the bytes written.

    The shared section first, then each anchor's own state as it is pickled,
    then the header that says where each one starts, then the header's
    length, so no more than one anchor's bytes are held at a time.
    """
    shared = _shared_by_reference([anchor.snapshot for anchor in anchors]) if len(anchors) > 1 else []
    index = {id(obj): place for place, obj in enumerate(shared)}
    partial = path.with_suffix(".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    with partial.open("wb") as out:
        out.write(BANK_MAGIC)
        shared_bytes = _pickled(shared)
        out.write(shared_bytes)
        shared_length = offset = len(shared_bytes)
        del shared_bytes
        for anchor in anchors:
            blob = _pickled(anchor.snapshot, _SharedWriter, shared=index)
            out.write(blob)
            entries.append({
                "current": np.asarray(anchor.current), "history": np.asarray(anchor.history),
                "source_condition": anchor.source_condition, "offset": offset, "length": len(blob),
            })
            offset += len(blob)
        header = _pickled({"fields": dict(fields), "shared_length": shared_length, "anchors": entries})
        out.write(header)
        out.write(_LENGTH.pack(len(header)))
    partial.replace(path)
    return path.stat().st_size


class BankedAnchor:
    """An anchor whose snapshot stays in the bank until something restores it."""

    __slots__ = ("current", "history", "source_condition", "_bank", "_index")

    def __init__(self, bank: AnchorBank, index: int, entry: Mapping[str, Any]) -> None:
        self.current = entry["current"]
        self.history = entry["history"]
        self.source_condition = entry["source_condition"]
        self._bank = bank
        self._index = index

    @property
    def snapshot(self) -> Any:
        return self._bank.snapshot(self._index)


class AnchorBank:
    """A bank written by `write_anchor_bank`, mapped and decoded an anchor at a time.

    `own` is this process's snapshot: each travelled snapshot is filled from it
    as it is decoded (core.subject.anchor_travel.fill_snapshot). `keep` is how
    many decoded snapshots stay; the caller sets it from its look schedule.
    """

    def __init__(self, path: Path, *, own: Any = None, keep: int = 1) -> None:
        self._handle = Path(path).open("rb")
        self._map = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
        if self._map[: len(BANK_MAGIC)] != BANK_MAGIC:
            raise ValueError(f"{path} is not an anchor bank in the shared-section format")
        end = len(self._map) - _LENGTH.size
        (length,) = _LENGTH.unpack(self._map[end:])
        header = _Reader(io.BytesIO(self._map[end - length:end])).load()
        self._base = len(BANK_MAGIC)
        self._shared_length = int(header["shared_length"])
        self._entries = list(header["anchors"])
        self.fields: dict[str, Any] = dict(header["fields"])
        self._shared: list[Any] | None = None
        self._own = own
        self._keep = max(1, int(keep))
        self._decoded: collections.OrderedDict[int, Any] = collections.OrderedDict()
        self.decodes = 0

    @staticmethod
    def is_one(path: Path) -> bool:
        with Path(path).open("rb") as handle:
            return handle.read(len(BANK_MAGIC)) == BANK_MAGIC

    def anchors(self) -> tuple[BankedAnchor, ...]:
        return tuple(BankedAnchor(self, index, entry) for index, entry in enumerate(self._entries))

    def keep(self, count: int) -> None:
        self._keep = max(1, int(count))
        while len(self._decoded) > self._keep:
            self._decoded.popitem(last=False)

    def snapshot(self, index: int) -> Any:
        if index in self._decoded:
            self._decoded.move_to_end(index)
            return self._decoded[index]
        if self._shared is None:
            self._shared = _Reader(io.BytesIO(self._map[self._base:self._base + self._shared_length])).load()
        entry = self._entries[index]
        start = self._base + int(entry["offset"])
        travelled = _SharedReader(io.BytesIO(self._map[start:start + int(entry["length"])]),
                                  shared=self._shared).load()
        snapshot = travelled if self._own is None else fill_snapshot(travelled, self._own)
        self.decodes += 1
        self._decoded[index] = snapshot
        while len(self._decoded) > self._keep:
            self._decoded.popitem(last=False)
        return snapshot

    def close(self) -> None:
        self._decoded.clear()
        self._map.close()
        self._handle.close()
