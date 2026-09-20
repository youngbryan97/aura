"""Importance, spreading, rent, forgetting: the economy itself.

Lifted whole out of `atomspace`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .atomspace import (
        Atom,
        Link,
    )


class _RunsTheAttentionEconomy:
    """Lifted whole out of AtomSpace; see atomspace.py."""

    def stimulate(self, atom: Atom, amount: float | None = None) -> float:
        """Pay STI to an atom from the fund (bounded by what the fund holds).

        Every stimulation also accrues a sliver of LTI — repeated relevance is
        what long-term importance *is*. Returns the STI actually granted.
        """

        amt = self._stimulus_size if amount is None else float(amount)
        with self._lock:
            rec = self._records.get(atom)
            if rec is None or amt <= 0.0:
                return 0.0
            grant = min(amt, self._sti_fund)
            self._sti_fund -= grant
            rec.av.sti += grant
            rec.av.lti += grant * 0.01
            return grant

    def attentional_focus(self, k: int | None = None) -> list[tuple[Atom, float]]:
        """The top-k atoms by STI with any attention at all — the focus."""

        limit = self._focus_size if k is None else int(k)
        with self._lock:
            ranked = sorted(
                ((rec.atom, rec.av.sti) for rec in self._records.values() if rec.av.sti > 0.0),
                key=lambda pair: pair[1],
                reverse=True,
            )
            return ranked[:limit]

    def _neighbors_locked(self, atom: Atom) -> set[Atom]:
        from .atomspace import (
            Link,
        )

        out: set[Atom] = set()
        if isinstance(atom, Link):
            out.update(atom.outgoing)
        for link in self._incoming.get(atom, ()):
            out.add(link)
            out.update(a for a in link.outgoing if a != atom)
        return out

    def spread_importance(self) -> float:
        """Diffuse a fraction of focus atoms' STI to their graph neighbors.

        This is ECAN's importance spreading: salience flows along structure,
        preferring strong Hebbian associations (learned co-activation) over
        plain syntactic adjacency, so what usually mattered *together with*
        the current focus becomes findable next. Returns the total STI moved.
        """
        from .atomspace import (
            HEBBIAN,
            Link,
        )

        moved = 0.0
        with self._lock:
            focus = [
                self._records[atom]
                for atom, sti in self.attentional_focus()
                if sti > 0 and atom in self._records
            ]
            for rec in focus:
                neighbors = [
                    self._records[n]
                    for n in self._neighbors_locked(rec.atom)
                    if n in self._records
                    and not (isinstance(n, Link) and n.atype == HEBBIAN)
                ]
                if not neighbors:
                    continue
                weights = [
                    1.0 + 4.0 * self._hebbian_weight_locked(rec.atom, n.atom)
                    for n in neighbors
                ]
                total_weight = sum(weights)
                share = rec.av.sti * self._spread_fraction
                rec.av.sti -= share
                for n, w in zip(neighbors, weights, strict=True):
                    n.av.sti += share * (w / total_weight)
                moved += share
        return moved

    def collect_rent(self) -> float:
        """Charge proportional rent on all STI back into the fund (decay)."""
        collected = 0.0
        with self._lock:
            for rec in self._records.values():
                if rec.av.sti <= 0.0:
                    continue
                rent = rec.av.sti * self._rent_rate
                rec.av.sti -= rent
                collected += rent
                if rec.av.sti < 0.01:
                    collected += rec.av.sti
                    rec.av.sti = 0.0
            self._sti_fund = min(self._sti_fund + collected, self._sti_fund_capacity)
        return collected

    def forget(self) -> list[Atom]:
        """Evict the least-important atoms when over capacity (ECAN forgetting).

        Only atoms with no incoming links (nothing else depends on them), not
        marked VLTI, ranked by (LTI, STI). Never touches the belief store —
        this is working-memory hygiene, not knowledge deletion.
        """

        with self._lock:
            overflow = len(self._records) - self._max_atoms
            if overflow <= 0:
                return []
            candidates = sorted(
                (
                    rec
                    for rec in self._records.values()
                    if not rec.av.vlti and not self._incoming.get(rec.atom)
                ),
                key=lambda r: (r.av.lti, r.av.sti),
            )
            evicted: list[Atom] = []
            for rec in candidates[:overflow]:
                self._evict_locked(rec.atom)
                evicted.append(rec.atom)
            self._forgotten_total += len(evicted)
            return evicted

    def _evict_locked(self, atom: Atom) -> None:
        from .atomspace import (
            Link,
            Node,
        )

        self._invalidate_dependents_locked(atom)
        existing = self._records.get(atom)
        if existing is not None:
            for key in tuple(existing.derivations):
                self._remove_derivation_locked(atom, key)
        rec = self._records.pop(atom, None)
        if rec is None:
            return
        atype = atom.atype if isinstance(atom, (Node, Link)) else "Unknown"
        self._by_type.get(atype, set()).discard(atom)
        self._incoming.pop(atom, None)
        if isinstance(atom, Link):
            for child in atom.outgoing:
                inc = self._incoming.get(child)
                if inc:
                    inc.discard(atom)

    def form_hebbian_links(self, *, max_pairs: int = 10) -> list[Link]:
        """ECAN Hebbian learning: link atoms that hold the focus *together*.

        Co-occurrence in the attentional focus is evidence of association;
        each co-focus tick revises the pair's Hebbian link upward. Spreading
        then prefers strong Hebbian paths, so attention learns the organism's
        actual co-activation structure instead of only static syntax.
        """
        from .atomspace import (
            HEBBIAN,
            Link,
            TruthValue,
        )

        formed: list[Link] = []
        focus = [
            atom
            for atom, _ in self.attentional_focus()
            if not (isinstance(atom, Link) and atom.atype == HEBBIAN)
        ]
        pairs = 0
        for i, a in enumerate(focus):
            for b in focus[i + 1:]:
                if pairs >= max_pairs:
                    return formed
                key = (a, b) if str(a) <= str(b) else (b, a)
                link = Link(HEBBIAN, key)
                self.add(link, TruthValue(1.0, 0.25))
                formed.append(link)
                pairs += 1
        return formed

    def _hebbian_weight_locked(self, a: Atom, b: Atom) -> float:
        from .atomspace import (
            HEBBIAN,
            Link,
        )

        key = (a, b) if str(a) <= str(b) else (b, a)
        rec = self._records.get(Link(HEBBIAN, key))
        if rec is None:
            return 0.0
        return rec.tv.strength * rec.tv.confidence

    def tick(self) -> dict[str, float]:
        """One attention-economy cycle: rent, spreading, Hebbian, forgetting."""
        rent = self.collect_rent()
        moved = self.spread_importance()
        hebbian = self.form_hebbian_links()
        evicted = self.forget()
        return {
            "rent_collected": rent,
            "sti_spread": moved,
            "hebbian_formed": float(len(hebbian)),
            "forgotten": float(len(evicted)),
        }

