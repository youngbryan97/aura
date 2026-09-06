"""core/connectome/proofreading.py — the corrections, kept as a record.

No automated reconstruction is finished when it comes out of the pipeline. The
fly and the human volumes both took years of human proofreading on top of the
segmentation, and the reason the field trusts them is not that the automation
was good. It is that every correction is an edit with an author, and the
reconstruction is the base plus the edit list, replayable from either end.

This module gives Aura the same discipline for corrections to her own map.

* An edit is **applied to a version and produces the next one**. A ledger
  replayed from the same base gives the same connectome, byte for byte.
* An edit **carries its evidence**. A split repaired because the edge was seen
  firing 4,000 times is a different object from one repaired because a person
  thought it looked right, and the ledger keeps the difference.
* Nothing is deleted. Undo truncates the replay, so the record of a correction
  that turned out to be wrong survives the correction being withdrawn.

The queue is the other half. H01 and FlyEM both rank candidate edits rather
than working through them in file order, because the connectome barely moves
for most corrections and moves a lot for a few. Ranking by observed traffic
puts the edits that change what Aura believes about herself at the top.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .activity import ObservedEdges
from .types import Compartment, Connection, ConnectomeSnapshot, EdgeKind, stable_id

logger = logging.getLogger("Aura.Connectome.Proofreading")

__all__ = [
    "EditKind",
    "Edit",
    "EditLedger",
    "CandidateEdit",
    "focused_queue",
    "repair_observed_splits",
]


class EditKind(StrEnum):
    """The two errors a reconstruction makes, and the two ways to undo them."""

    #: A connection that exists was missing. Add it.
    JOIN = "join"
    #: A connection that does not exist was present. Remove it.
    CUT = "cut"
    #: A cell was assigned the wrong class.
    RECLASS = "reclass"
    #: A previously applied edit is withdrawn.
    WITHDRAW = "withdraw"


@dataclass(frozen=True)
class Edit:
    """One correction, with everything needed to judge it later."""

    kind: EditKind
    pre: str
    post: str
    author: str
    evidence: str
    at: float
    kind_detail: str = ""
    contacts: int = 1
    edge_kind: EdgeKind = EdgeKind.DRIVE
    observed_calls: int = 0
    target_edit_id: str = ""

    @property
    def edit_id(self) -> str:
        return stable_id(self.kind, self.pre, self.post, self.author, f"{self.at:.6f}")

    def to_json(self) -> dict[str, Any]:
        return {
            "edit_id": self.edit_id,
            "kind": str(self.kind),
            "pre": self.pre,
            "post": self.post,
            "author": self.author,
            "evidence": self.evidence,
            "at": self.at,
            "kind_detail": self.kind_detail,
            "contacts": self.contacts,
            "edge_kind": str(self.edge_kind),
            "observed_calls": self.observed_calls,
            "target_edit_id": self.target_edit_id,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> Edit:
        return cls(
            kind=EditKind(payload["kind"]),
            pre=str(payload["pre"]),
            post=str(payload["post"]),
            author=str(payload.get("author", "unknown")),
            evidence=str(payload.get("evidence", "")),
            at=float(payload.get("at", 0.0)),
            kind_detail=str(payload.get("kind_detail", "")),
            contacts=int(payload.get("contacts", 1)),
            edge_kind=EdgeKind(payload.get("edge_kind", EdgeKind.DRIVE)),
            observed_calls=int(payload.get("observed_calls", 0)),
            target_edit_id=str(payload.get("target_edit_id", "")),
        )


class EditLedger:
    """An append-only list of corrections, replayable onto a base snapshot."""

    def __init__(self, base_digest: str = "") -> None:
        self.base_digest = base_digest
        self._edits: list[Edit] = []

    def __len__(self) -> int:
        return len(self._edits)

    @property
    def edits(self) -> tuple[Edit, ...]:
        return tuple(self._edits)

    def add(self, edit: Edit) -> Edit:
        self._edits.append(edit)
        return edit

    def join(
        self,
        pre: str,
        post: str,
        *,
        author: str,
        evidence: str,
        contacts: int = 1,
        observed_calls: int = 0,
        edge_kind: EdgeKind = EdgeKind.DRIVE,
    ) -> Edit:
        return self.add(
            Edit(
                kind=EditKind.JOIN,
                pre=pre,
                post=post,
                author=author,
                evidence=evidence,
                at=time.time(),
                contacts=max(1, int(contacts)),
                observed_calls=int(observed_calls),
                edge_kind=edge_kind,
            )
        )

    def cut(
        self,
        pre: str,
        post: str,
        *,
        author: str,
        evidence: str,
        edge_kind: EdgeKind = EdgeKind.DRIVE,
    ) -> Edit:
        return self.add(
            Edit(
                kind=EditKind.CUT,
                pre=pre,
                post=post,
                author=author,
                evidence=evidence,
                at=time.time(),
                edge_kind=edge_kind,
            )
        )

    def reclass(self, uid: str, new_class: str, *, author: str, evidence: str) -> Edit:
        return self.add(
            Edit(
                kind=EditKind.RECLASS,
                pre=uid,
                post=uid,
                author=author,
                evidence=evidence,
                at=time.time(),
                kind_detail=new_class,
            )
        )

    def withdraw(self, edit_id: str, *, author: str, evidence: str) -> Edit:
        return self.add(
            Edit(
                kind=EditKind.WITHDRAW,
                pre="",
                post="",
                author=author,
                evidence=evidence,
                at=time.time(),
                target_edit_id=edit_id,
            )
        )

    # -- replay ---------------------------------------------------------

    def apply(self, snapshot: ConnectomeSnapshot) -> ConnectomeSnapshot:
        """Replay every live edit onto a base and return the corrected map.

        Withdrawals are resolved before anything is applied, so an edit that
        was later withdrawn never touches the connectome even though it stays
        in the record.
        """
        withdrawn = {
            edit.target_edit_id for edit in self._edits if edit.kind is EditKind.WITHDRAW
        }
        connections = dict(snapshot.connections)
        units = dict(snapshot.units)
        applied = 0
        for edit in self._edits:
            if edit.kind is EditKind.WITHDRAW or edit.edit_id in withdrawn:
                continue
            key = (edit.pre, edit.post, str(edit.edge_kind))
            if edit.kind is EditKind.JOIN:
                existing = connections.get(key)
                if existing is None:
                    connections[key] = Connection(
                        pre=edit.pre,
                        post=edit.post,
                        contacts=edit.contacts,
                        sign=1,
                        compartments=(Compartment.SOMA,),
                        kind=edit.edge_kind,
                    )
                else:
                    connections[key] = Connection(
                        pre=existing.pre,
                        post=existing.post,
                        contacts=existing.contacts + edit.contacts,
                        sign=existing.sign,
                        compartments=existing.compartments,
                        kind=existing.kind,
                    )
                applied += 1
            elif edit.kind is EditKind.CUT:
                if connections.pop(key, None) is not None:
                    applied += 1
            elif edit.kind is EditKind.RECLASS:
                unit = units.get(edit.pre)
                if unit is not None:
                    from dataclasses import replace as _replace

                    from .types import CellClass

                    try:
                        units[edit.pre] = _replace(unit, cell_class=CellClass(edit.kind_detail))
                        applied += 1
                    except ValueError:
                        logger.debug("reclass edit named an unknown class: %s", edit.kind_detail)
        corrected = ConnectomeSnapshot(
            version=snapshot.version + 1,
            units=units,
            connections=connections,
            neuropils=snapshot.neuropils,
            built_at=snapshot.built_at,
            source=snapshot.source,
            attrs=dict(snapshot.attrs),
        )
        corrected.attrs.update(
            {
                "proofread_edits": len(self._edits),
                "proofread_applied": applied,
                "proofread_withdrawn": len(withdrawn),
                "base_digest": self.base_digest or snapshot.digest(),
            }
        )
        return corrected

    # -- persistence ----------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "base_digest": self.base_digest,
            "edits": [edit.to_json() for edit in self._edits],
        }

    def dumps(self) -> str:
        return json.dumps(self.to_json(), sort_keys=True, indent=2)

    @classmethod
    def loads(cls, payload: str) -> EditLedger:
        data = json.loads(payload)
        ledger = cls(base_digest=str(data.get("base_digest", "")))
        for entry in data.get("edits", []):
            ledger.add(Edit.from_json(entry))
        return ledger


# ---------------------------------------------------------------------------
# Focused proofreading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateEdit:
    """A correction worth someone's attention, with why it is worth it."""

    kind: EditKind
    pre: str
    post: str
    impact: float
    observed_calls: int
    reason: str

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "pre": self.pre,
            "post": self.post,
            "impact": round(self.impact, 4),
            "observed_calls": self.observed_calls,
            "reason": self.reason,
        }


def focused_queue(
    snapshot: ConnectomeSnapshot,
    observed: ObservedEdges,
    *,
    limit: int = 200,
    min_caller_calls: int = 8,
) -> list[CandidateEdit]:
    """Rank corrections by how much the connectome would move if they landed.

    Splits are ranked by the traffic on the missing edge, because a path that
    carried four thousand calls and is absent from the map is a bigger error in
    what Aura believes about herself than one that carried three. Suspected
    merges come after every split, and stay suspicions: the branch may simply
    not have run.
    """
    present = {(c.pre, c.post) for c in snapshot.edges(EdgeKind.DRIVE)}
    total_calls = max(1, sum(observed.counts.values()))
    queue: list[CandidateEdit] = []
    for (pre, post), calls in observed.counts.items():
        if (pre, post) in present:
            continue
        if pre not in snapshot.units or post not in snapshot.units:
            continue
        queue.append(
            CandidateEdit(
                kind=EditKind.JOIN,
                pre=pre,
                post=post,
                impact=calls / total_calls,
                observed_calls=calls,
                reason="observed firing, absent from the reconstruction",
            )
        )
    fired: dict[str, int] = {}
    for (pre, _), count in observed.counts.items():
        fired[pre] = fired.get(pre, 0) + count
    for conn in snapshot.edges(EdgeKind.DRIVE):
        calls = fired.get(conn.pre, 0)
        if calls < min_caller_calls or (conn.pre, conn.post) in observed.pairs():
            continue
        queue.append(
            CandidateEdit(
                kind=EditKind.CUT,
                pre=conn.pre,
                post=conn.post,
                impact=-(conn.contacts / max(1, calls)),
                observed_calls=0,
                reason=f"caller fired {calls} times and never took this edge",
            )
        )
    queue.sort(key=lambda c: (-c.impact, c.pre, c.post))
    return queue[:limit]


def repair_observed_splits(
    snapshot: ConnectomeSnapshot,
    observed: ObservedEdges,
    *,
    author: str = "activity-recorder",
    ledger: EditLedger | None = None,
) -> EditLedger:
    """Write a join for every edge that was seen firing and is missing.

    This is the one class of correction that needs no judgement. The edge ran.
    """
    ledger = ledger or EditLedger(base_digest=snapshot.digest())
    present = {(c.pre, c.post) for c in snapshot.edges(EdgeKind.DRIVE)}
    for (pre, post), calls in sorted(observed.counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if (pre, post) in present:
            continue
        if pre not in snapshot.units or post not in snapshot.units:
            continue
        ledger.join(
            pre,
            post,
            author=author,
            evidence=f"observed {calls} calls during a recorded window",
            observed_calls=calls,
        )
    return ledger
