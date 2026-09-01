"""core/knowledge/atomspace_persistence.py — a store that survives the process.

The other half of :class:`core.knowledge.atomspace.AtomSpace`, in its own file
because persistence is its own concern and because the store was thirty-seven
methods with it inside. These reach the store's private records deliberately:
they are the same object's serialisation, not a client of it, and a snapshot
that could only see the public surface would lose the per-source attribution
that stops a reload double-counting a witness.

Card 098's second half. The first is the scale curve in
docs/evidence/atomspace_scale.json.
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any, Mapping

from core.knowledge.atomspace import (
    AttentionValue,
    Atom,
    Link,
    Node,
    TruthValue,
    Variable,
    _Record,
)

if TYPE_CHECKING:
    from core.knowledge.atomspace import AtomSpace

#: Snapshot format id. A version in the file rather than in a comment, so a
#: store written by an older build is refused instead of half-loaded.
SNAPSHOT_SCHEMA = "aura.atomspace.snapshot.v1"

__all__ = [
    "SNAPSHOT_SCHEMA",
    "snapshot",
    "save",
    "save_async",
    "restore",
    "load",
]


def encode_atom(atom: Atom) -> Any:
    """One atom as JSON, links nesting inside links."""
    if isinstance(atom, Node):
        return ["n", atom.atype, atom.name]
    if isinstance(atom, Variable):
        return ["v", atom.name]
    if isinstance(atom, Link):
        return ["l", atom.atype, [encode_atom(a) for a in atom.outgoing]]
    raise TypeError(f"cannot encode {type(atom).__name__}")


def decode_atom(row: Any) -> Atom:
    kind = row[0]
    if kind == "n":
        return Node(row[1], row[2])
    if kind == "v":
        return Variable(row[1])
    if kind == "l":
        return Link(row[1], tuple(decode_atom(a) for a in row[2]))
    raise ValueError(f"unknown atom encoding {kind!r}")



def snapshot(space: "AtomSpace") -> dict[str, Any]:
    """Everything the store holds, in a form that survives the process.

    Truth, attention and per-source attribution all travel. Dropping the
    sources would make a reloaded store forget that two assertions came
    from one witness, and the next revision would double-count evidence
    the original refused — a store that loses its provenance on restart is
    not the same store.
    """
    with space._lock:
        return {
            "schema": SNAPSHOT_SCHEMA,
            "sti_fund": space._sti_fund,
            "forgotten_total": space._forgotten_total,
            "derived_total": space._derived_total,
            "duplicate_assertions": space._duplicate_assertions,
            "unattributed_assertions": space._unattributed_assertions,
            "atoms": [
                {
                    "atom": encode_atom(rec.atom),
                    "tv": [rec.tv.strength, rec.tv.count],
                    "av": [rec.av.sti, rec.av.lti, rec.av.vlti],
                    "added_at": rec.added_at,
                    "sources": {
                        name: [tv.strength, tv.count]
                        for name, tv in rec.sources.items()
                    },
                    "unattributed": (
                        [rec.unattributed.strength, rec.unattributed.count]
                        if rec.unattributed is not None
                        else None
                    ),
                }
                for rec in space._records.values()
            ],
        }

def save(space: "AtomSpace", path: "os.PathLike[str] | str") -> int:
    """Write a snapshot atomically. Returns the atom count written.

    Blocking, and not only for the write: :func:`snapshot` holds the store's
    lock while it builds the payload, measured at 2.3 seconds for 750,000
    atoms on this host. Never call this from the event loop — use
    :func:`save_async`, which does both halves in a worker thread.

    Through the file write gateway, so the temp-then-rename is the same
    one every other durable write in this process uses: a crash part-way
    leaves the previous snapshot intact rather than a half file that loads
    as a smaller store.
    """
    import json as _json
    from pathlib import Path as _Path

    from core.runtime.file_write_gateway import get_file_write_gateway

    payload = snapshot(space)
    get_file_write_gateway().write_text(
        _Path(path), _json.dumps(payload, separators=(",", ":")) + "\n"
    )
    return len(payload["atoms"])

async def save_async(space: "AtomSpace", path: "os.PathLike[str] | str") -> int:
    """:func:`save` off the loop, both the snapshot and the write.

    The snapshot is the half that surprises: an on-loop fsync once froze this
    runtime for twenty minutes, and a lock held for seconds while a payload is
    built is the same shape of stall wearing a different name.
    """
    import asyncio
    import json as _json
    from pathlib import Path as _Path

    from core.runtime.file_write_gateway import get_file_write_gateway

    payload = await asyncio.to_thread(snapshot, space)
    text = await asyncio.to_thread(
        lambda: _json.dumps(payload, separators=(",", ":"))
    )
    await get_file_write_gateway().write_text_async(_Path(path), text + "\n")
    return len(payload["atoms"])


def restore(space: "AtomSpace", payload: Mapping[str, Any]) -> int:
    """Replace this store's contents with a snapshot. Returns atoms loaded.

    Refuses a payload it does not recognise rather than loading the part it
    understands: a store that silently comes back smaller is worse than
    one that refuses to come back.
    """
    if payload.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError(
            f"snapshot schema {payload.get('schema')!r} is not "
            f"{SNAPSHOT_SCHEMA!r}; refusing to load part of it"
        )
    rows = payload.get("atoms")
    if not isinstance(rows, list):
        raise ValueError("snapshot has no atom list")
    rebuilt: dict[Atom, _Record] = {}
    for row in rows:
        atom = decode_atom(row["atom"])
        strength, count = row["tv"]
        sti, lti, vlti = row["av"]
        unattributed = row.get("unattributed")
        rebuilt[atom] = _Record(
            atom=atom,
            tv=TruthValue(float(strength), float(count)),
            av=AttentionValue(float(sti), float(lti), bool(vlti)),
            added_at=float(row.get("added_at", time.time())),
            sources={
                name: TruthValue(float(s), float(c))
                for name, (s, c) in (row.get("sources") or {}).items()
            },
            unattributed=(
                TruthValue(float(unattributed[0]), float(unattributed[1]))
                if unattributed is not None
                else None
            ),
        )
    with space._lock:
        space._records = rebuilt
        space._by_type = {}
        space._incoming = {}
        for atom in rebuilt:
            if isinstance(atom, (Node, Link)):
                space._by_type.setdefault(atom.atype, set()).add(atom)
            if isinstance(atom, Link):
                for child in atom.outgoing:
                    space._incoming.setdefault(child, set()).add(atom)
        space._sti_fund = float(payload.get("sti_fund", space._sti_fund_capacity))
        space._forgotten_total = int(payload.get("forgotten_total", 0))
        space._derived_total = int(payload.get("derived_total", 0))
        space._duplicate_assertions = int(payload.get("duplicate_assertions", 0))
        space._unattributed_assertions = int(
            payload.get("unattributed_assertions", 0)
        )
        return len(rebuilt)

def load(space: "AtomSpace", path: "os.PathLike[str] | str") -> int:
    """Read a snapshot written by :meth:`save`. Returns atoms loaded."""
    import json as _json
    from pathlib import Path as _Path

    return restore(space, _json.loads(_Path(path).read_text(encoding="utf-8")))
