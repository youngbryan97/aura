"""AtomSpace — Hyperon-style typed metagraph with economic attention.

OpenCog Hyperon's core organs, fused into Aura's knowledge layer:

- **Metagraph store**: immutable ``Node``/``Link`` atoms where links may point
  at links (not just vertices), typed by plain strings, deduplicated by value.
- **PLN truth values**: every atom carries a ``TruthValue`` (strength, count);
  repeated assertions merge by the PLN *revision* rule — evidence-weighted,
  convergent — replacing ad-hoc confidence blends. Chained implications are
  derived with the PLN *deduction* formula.
- **Unification queries**: MeTTa-style pattern matching with ``Variable``
  atoms, conjunctive multi-clause joins, and grounded predicates (Python
  callables evaluated during the match).
- **ECAN attention economy**: each atom carries STI/LTI attention values paid
  from a fixed fund; stimulation, rent, importance spreading along links,
  an attentional focus, and LTI-based forgetting. Inference is *economic*:
  the forward chainer only expands what is attentionally hot, so compute
  follows salience instead of scanning the whole graph.

Live wiring: ``BeliefRevisionEngine`` asserts every claim here (PLN revision
is now the confidence-update rule), stimulates claim atoms on access, ticks
the attention economy from its revision loop, and publishes derived
implications on the event bus (``atomspace.derived``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from core.runtime.lockdep import checked_lock

# ── PLN truth values ──────────────────────────────────────────────────────

# PLN's confidence lookahead: confidence = count / (count + _LOOKAHEAD).
_LOOKAHEAD = 1.0
_MAX_COUNT = 1000.0
# Confidence discount applied by inference rules (derived knowledge is never
# as certain as directly observed evidence).
_DEDUCTION_DISCOUNT = 0.9


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass(frozen=True)
class TruthValue:
    """PLN simple truth value: strength (probability) + count (evidence mass)."""

    strength: float = 0.5
    count: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "strength", _clamp01(float(self.strength)))
        object.__setattr__(self, "count", max(0.0, min(float(self.count), _MAX_COUNT)))

    @property
    def confidence(self) -> float:
        return self.count / (self.count + _LOOKAHEAD)

    def revise(self, other: "TruthValue") -> "TruthValue":
        """PLN revision: merge two independent estimates of the same atom.

        Evidence-weighted mean of strengths; counts add. Convergent — repeated
        confirmation raises confidence, contradictory evidence pulls strength
        toward the middle while the count keeps the disagreement visible.
        """
        total = self.count + other.count
        if total <= 0.0:
            return TruthValue((self.strength + other.strength) / 2.0, 0.0)
        s = (self.strength * self.count + other.strength * other.count) / total
        return TruthValue(s, total)

    def negation(self) -> "TruthValue":
        return TruthValue(1.0 - self.strength, self.count)

    def to_dict(self) -> dict[str, float]:
        return {"strength": self.strength, "count": self.count, "confidence": self.confidence}


def deduction_tv(ab: TruthValue, bc: TruthValue, b: TruthValue, c: TruthValue) -> TruthValue:
    """PLN deduction: from A→B and B→C derive A→C.

    Uses the standard independence-based PLN formula
    ``sAC = sAB·sBC + (1−sAB)·(sC − sB·sBC)/(1−sB)`` with node prevalences
    ``sB``/``sC`` (0.5 when unknown), clamped to [0,1]; the count is the
    weaker premise's count with a deduction discount.
    """
    s_ab, s_bc, s_b, s_c = ab.strength, bc.strength, b.strength, c.strength
    if s_b >= 0.9999:
        s_ac = s_ab * s_bc
    else:
        s_ac = s_ab * s_bc + (1.0 - s_ab) * (s_c - s_b * s_bc) / (1.0 - s_b)
    return TruthValue(_clamp01(s_ac), min(ab.count, bc.count) * _DEDUCTION_DISCOUNT)


def inversion_tv(ab: TruthValue, a: TruthValue, b: TruthValue) -> TruthValue:
    """PLN inversion (Bayes): from A→B derive B→A.

    ``sBA = sAB·sA/sB`` with node prevalences (0.5 when unknown). Inverted
    knowledge is weaker than observed knowledge, so the count is discounted
    harder than deduction's.
    """
    s_b = b.strength if b.strength > 1e-9 else 0.5
    s_ba = ab.strength * (a.strength if a.strength > 0 else 0.5) / s_b
    return TruthValue(_clamp01(s_ba), ab.count * _DEDUCTION_DISCOUNT * 0.8)


def abduction_tv(
    ac: TruthValue, bc: TruthValue, a: TruthValue, b: TruthValue, c: TruthValue
) -> TruthValue:
    """PLN abduction: from A→C and B→C derive A→B (invert B→C, then deduce)."""
    cb = inversion_tv(bc, b, c)
    return deduction_tv(ac, cb, c, b)


def induction_tv(
    ab: TruthValue, ac: TruthValue, a: TruthValue, b: TruthValue, c: TruthValue
) -> TruthValue:
    """PLN induction: from A→B and A→C derive B→C (invert A→B, then deduce)."""
    ba = inversion_tv(ab, a, b)
    return deduction_tv(ba, ac, a, c)


# ── Atoms: the typed metagraph vocabulary ─────────────────────────────────

class Atom:
    """Base class for metagraph atoms (immutable, hashable, value-identity)."""

    __slots__ = ()


@dataclass(frozen=True)
class Node(Atom):
    atype: str
    name: str

    def __str__(self) -> str:
        return f"({self.atype} \"{self.name}\")"


@dataclass(frozen=True)
class Variable(Atom):
    """A pattern variable — binds to any atom during unification."""

    name: str

    def __str__(self) -> str:
        return f"${self.name}"


@dataclass(frozen=True)
class Link(Atom):
    atype: str
    outgoing: tuple[Atom, ...]

    def __str__(self) -> str:
        inner = " ".join(str(a) for a in self.outgoing)
        return f"({self.atype} {inner})"


# Conventional type names (plain strings, like Hyperon's symbols).
CONCEPT = "Concept"
PREDICATE = "Predicate"
GROUNDED_PREDICATE = "GroundedPredicate"
IMPLICATION = "Implication"
INHERITANCE = "Inheritance"
EVALUATION = "Evaluation"
LIST = "List"
HEBBIAN = "Hebbian"


def concept(name: str) -> Node:
    return Node(CONCEPT, name)


def predicate(name: str) -> Node:
    return Node(PREDICATE, name)


def implication(a: Atom, b: Atom) -> Link:
    return Link(IMPLICATION, (a, b))


def evaluation(pred: Atom, *args: Atom) -> Link:
    return Link(EVALUATION, (pred, Link(LIST, tuple(args))))


# ── Attention values (ECAN) ───────────────────────────────────────────────

@dataclass
class AttentionValue:
    sti: float = 0.0        # short-term importance (what matters *now*)
    lti: float = 0.0        # long-term importance (what has mattered repeatedly)
    vlti: bool = False      # very-long-term: exempt from forgetting

    def to_dict(self) -> dict[str, object]:
        return {"sti": self.sti, "lti": self.lti, "vlti": self.vlti}


#: Per-atom cap on distinct tracked sources. Beyond it the oldest folds into
#: the unattributed pool: bookkeeping that grows without bound is a leak, and
#: an atom with this many witnesses is not the case deduplication protects.
_MAX_SOURCES_PER_ATOM = 256


@dataclass
class _Record:
    atom: Atom
    tv: TruthValue
    av: AttentionValue = field(default_factory=AttentionValue)
    added_at: float = field(default_factory=time.time)
    #: Each source's LATEST contribution to this atom's truth, keyed by the
    #: observation identity that produced it. A source restating itself
    #: replaces its own entry rather than adding a second one, which is what
    #: makes a mirror of an accumulating store idempotent.
    sources: dict[str, TruthValue] = field(default_factory=dict)
    #: Everything asserted without an identity, accumulated the old way.
    unattributed: TruthValue | None = None


def _fold(rec: "_Record") -> TruthValue:
    """Recompute an atom's truth from its per-source contributions.

    Each source contributes once, whatever it has said most recently. The
    unattributed pool joins as one more contribution, so a store still passing
    bare truth values is not silently dropped — it is just not deduplicated.
    """
    contributions = list(rec.sources.values())
    if rec.unattributed is not None:
        contributions.append(rec.unattributed)
    if not contributions:
        return TruthValue()
    folded = contributions[0]
    for extra in contributions[1:]:
        folded = folded.revise(extra)
    return folded


# ── Unification ───────────────────────────────────────────────────────────

Bindings = dict[str, Atom]


def unify(pattern: Atom, ground: Atom, bindings: Bindings | None = None) -> Bindings | None:
    """Match ``pattern`` (may contain Variables) against a ground atom.

    Returns the extended bindings on success, None on mismatch. Consistent:
    a variable bound earlier must match the same atom on reuse.
    """
    b = dict(bindings) if bindings else {}

    def walk(p: Atom, g: Atom) -> bool:
        if isinstance(p, Variable):
            bound = b.get(p.name)
            if bound is None:
                b[p.name] = g
                return True
            return bound == g
        if isinstance(p, Node):
            return isinstance(g, Node) and p == g
        if isinstance(p, Link):
            if not isinstance(g, Link) or p.atype != g.atype or len(p.outgoing) != len(g.outgoing):
                return False
            return all(walk(pc, gc) for pc, gc in zip(p.outgoing, g.outgoing))
        return False

    return b if walk(pattern, ground) else None


def substitute(pattern: Atom, bindings: Bindings) -> Atom:
    """Instantiate a pattern with bindings (unbound variables stay symbolic)."""
    if isinstance(pattern, Variable):
        return bindings.get(pattern.name, pattern)
    if isinstance(pattern, Link):
        return Link(pattern.atype, tuple(substitute(a, bindings) for a in pattern.outgoing))
    return pattern


def _pattern_is_ground(atom: Atom) -> bool:
    if isinstance(atom, Variable):
        return False
    if isinstance(atom, Link):
        return all(_pattern_is_ground(a) for a in atom.outgoing)
    return True


# ── The AtomSpace ─────────────────────────────────────────────────────────

class AtomSpace:
    """Thread-safe metagraph store with PLN truth values and ECAN attention."""

    def __init__(
        self,
        *,
        sti_fund: float = 10_000.0,
        stimulus_size: float = 20.0,
        rent_rate: float = 0.05,
        spread_fraction: float = 0.2,
        focus_size: int = 20,
        max_atoms: int = 50_000,
    ) -> None:
        self._lock = checked_lock("core.knowledge.atomspace.AtomSpace", reentrant=True)
        self._records: dict[Atom, _Record] = {}
        self._by_type: dict[str, set[Atom]] = {}
        self._incoming: dict[Atom, set[Link]] = {}
        self._grounded: dict[str, Callable[..., bool]] = {}
        # ECAN economy parameters
        self._sti_fund = float(sti_fund)
        self._sti_fund_capacity = float(sti_fund)
        self._stimulus_size = float(stimulus_size)
        self._rent_rate = float(rent_rate)
        self._spread_fraction = float(spread_fraction)
        self._focus_size = int(focus_size)
        self._max_atoms = int(max_atoms)
        self._forgotten_total = 0
        self._derived_total = 0
        # Evidence bookkeeping: revisions refused as duplicates, and revisions
        # that arrived with no identity to check.
        self._duplicate_assertions = 0
        self._unattributed_assertions = 0

    # ── store ─────────────────────────────────────────────────────────

    def add(
        self, atom: Atom, tv: TruthValue | None = None, *, source: str | None = None
    ) -> TruthValue:
        """Insert an atom (and its recursive outgoing set), revising the TV.

        Re-adding an existing atom merges truth by PLN revision — the Hyperon
        semantics for repeated assertion. Returns the atom's current TV.

        ``source`` names the observation this assertion comes from, and it is
        what stops one observation being heard ten times. PLN revision assumes
        its two arguments are independent estimates and has no way to check;
        asserting ``TruthValue(0.9, 4.0)`` ten times from one sensor reading
        used to take confidence from 0.44 to 0.98. When a source is given and
        this atom's truth already rests on it, the revision is skipped and the
        current value returned unchanged.

        Omitting ``source`` keeps the old behaviour, because most of the
        repository has no identity to give yet. That path is *unattributed*
        rather than independent, and ``evidence_report()`` counts it so the
        difference stays visible instead of being assumed away.
        """
        if not _pattern_is_ground(atom):
            raise ValueError("cannot add a pattern (Variable) to the AtomSpace")
        with self._lock:
            return self._add_locked(atom, tv, source=source)

    def _add_locked(
        self, atom: Atom, tv: TruthValue | None, *, source: str | None = None
    ) -> TruthValue:
        if isinstance(atom, Link):
            for child in atom.outgoing:
                if child not in self._records:
                    self._add_locked(child, None)
                self._incoming.setdefault(child, set()).add(atom)
        rec = self._records.get(atom)
        if rec is None:
            rec = _Record(atom=atom, tv=tv if tv is not None else TruthValue())
            if tv is not None:
                if source is not None:
                    rec.sources[source] = tv
                else:
                    rec.unattributed = tv
            self._records[atom] = rec
            atype = atom.atype if isinstance(atom, (Node, Link)) else "Unknown"
            self._by_type.setdefault(atype, set()).add(atom)
            return rec.tv
        if tv is None:
            return rec.tv
        if source is None:
            self._unattributed_assertions += 1
            rec.unattributed = tv if rec.unattributed is None else rec.unattributed.revise(tv)
        else:
            previous = rec.sources.get(source)
            if previous is not None:
                self._duplicate_assertions += 1
                if previous.strength == tv.strength and previous.count == tv.count:
                    return rec.tv
            rec.sources[source] = tv
            if len(rec.sources) > _MAX_SOURCES_PER_ATOM:
                # An atom with thousands of distinct witnesses is past the point
                # where per-source bookkeeping buys anything; fold the oldest
                # into the unattributed pool rather than growing without bound.
                oldest = next(iter(rec.sources))
                folded = rec.sources.pop(oldest)
                rec.unattributed = (
                    folded if rec.unattributed is None else rec.unattributed.revise(folded)
                )
        rec.tv = _fold(rec)
        return rec.tv

    def evidence_sources(self, atom: Atom) -> frozenset[str]:  # noqa: D402
        """The observation identities this atom's truth value rests on."""
        with self._lock:
            rec = self._records.get(atom)
            return frozenset(rec.sources) if rec else frozenset()

    def evidence_report(self) -> dict[str, Any]:
        """How much of this space's belief can name where it came from.

        ``unattributed_assertions`` is not an error count. It is the number of
        revisions that could not be checked for independence, which is the
        honest reading of a store still being migrated onto sourced evidence.
        """
        with self._lock:
            attributed = sum(1 for r in self._records.values() if r.sources)
            return {
                "atoms": len(self._records),
                "atoms_with_sources": attributed,
                "duplicate_assertions_refused": self._duplicate_assertions,
                "unattributed_assertions": self._unattributed_assertions,
            }

    def get_tv(self, atom: Atom) -> TruthValue | None:
        with self._lock:
            rec = self._records.get(atom)
            return rec.tv if rec else None

    def get_av(self, atom: Atom) -> AttentionValue | None:
        with self._lock:
            rec = self._records.get(atom)
            return AttentionValue(rec.av.sti, rec.av.lti, rec.av.vlti) if rec else None

    def set_vlti(self, atom: Atom, vlti: bool = True) -> None:
        with self._lock:
            rec = self._records.get(atom)
            if rec:
                rec.av.vlti = vlti

    def incoming(self, atom: Atom) -> list[Link]:
        with self._lock:
            return list(self._incoming.get(atom, ()))

    def atoms_of_type(self, atype: str) -> list[Atom]:
        with self._lock:
            return list(self._by_type.get(atype, ()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def __contains__(self, atom: Atom) -> bool:
        with self._lock:
            return atom in self._records

    # ── queries ───────────────────────────────────────────────────────

    def register_grounded(self, name: str, fn: Callable[..., bool]) -> None:
        """Register a grounded predicate callable, usable in queries as
        ``evaluation(Node(GROUNDED_PREDICATE, name), args…)``."""
        with self._lock:
            self._grounded[name] = fn

    def match(self, pattern: Atom, bindings: Bindings | None = None) -> list[Bindings]:
        """All bindings under which ``pattern`` unifies with a stored atom."""
        with self._lock:
            if isinstance(pattern, Variable):
                candidates: Iterable[Atom] = list(self._records)
            elif isinstance(pattern, (Node, Link)):
                candidates = list(self._by_type.get(pattern.atype, ()))
            else:
                candidates = ()
            out: list[Bindings] = []
            for cand in candidates:
                b = unify(pattern, cand, bindings)
                if b is not None:
                    out.append(b)
            return out

    def _grounded_clause(self, clause: Atom) -> tuple[str, tuple[Atom, ...]] | None:
        if (
            isinstance(clause, Link)
            and clause.atype == EVALUATION
            and len(clause.outgoing) == 2
            and isinstance(clause.outgoing[0], Node)
            and clause.outgoing[0].atype == GROUNDED_PREDICATE
            and isinstance(clause.outgoing[1], Link)
            and clause.outgoing[1].atype == LIST
        ):
            return clause.outgoing[0].name, clause.outgoing[1].outgoing
        return None

    def query(self, clauses: Sequence[Atom]) -> list[Bindings]:
        """Conjunctive pattern query: join clause matches on shared variables.

        Grounded-predicate clauses are evaluated (not matched) once their
        arguments are bound, acting as filters — MeTTa's grounded atoms.
        """
        results: list[Bindings] = [{}]
        # Evaluate structural clauses first, grounded filters last.
        structural = [c for c in clauses if self._grounded_clause(c) is None]
        grounded = [c for c in clauses if self._grounded_clause(c) is not None]
        for clause in structural:
            next_results: list[Bindings] = []
            for b in results:
                next_results.extend(self.match(substitute(clause, b), b))
            results = next_results
            if not results:
                return []
        for clause in grounded:
            gname, gargs = self._grounded_clause(clause)  # type: ignore[misc]
            with self._lock:
                fn = self._grounded.get(gname)
            if fn is None:
                return []
            kept: list[Bindings] = []
            for b in results:
                args = tuple(substitute(a, b) for a in gargs)
                if any(not _pattern_is_ground(a) for a in args):
                    continue
                try:
                    if fn(*args):
                        kept.append(b)
                except (ValueError, TypeError, ArithmeticError):
                    continue
            results = kept
            if not results:
                return []
        return results

    # ── ECAN: the attention economy ───────────────────────────────────

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
                for n, w in zip(neighbors, weights):
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

    # ── PLN inference: universal rewrite rules, economically controlled ─

    def apply_rules(
        self,
        rules: Sequence["InferenceRule"],
        *,
        max_derivations: int = 16,
        focus_only: bool = True,
    ) -> list[Atom]:
        """Fire pattern-based inference rules over the metagraph.

        Each rule is (premise patterns, conclusion template, truth formula) —
        the MeTTa/PLN architecture where inference *is* unification plus a
        rewrite. With ``focus_only``, a rule instance only fires when one of
        its bound premises (or a premise endpoint) is attentionally hot:
        ECAN's economic inference control. Derivations enter via revision and
        get a small stimulus so useful chains compound across cycles.
        """
        derived: list[Atom] = []
        hot: set[Atom] = set()
        if focus_only:
            for atom, _ in self.attentional_focus():
                hot.add(atom)
                if isinstance(atom, Link):
                    hot.update(atom.outgoing)
        for rule in rules:
            if len(derived) >= max_derivations:
                break
            for binding in self.query(list(rule.premises)):
                if len(derived) >= max_derivations:
                    break
                premise_atoms = [substitute(p, binding) for p in rule.premises]
                if focus_only and not any(
                    p in hot or (isinstance(p, Link) and any(o in hot for o in p.outgoing))
                    for p in premise_atoms
                ):
                    continue
                conclusion = substitute(rule.conclusion, binding)
                if not _pattern_is_ground(conclusion) or conclusion in premise_atoms:
                    continue
                if isinstance(conclusion, Link) and len(set(conclusion.outgoing)) < len(conclusion.outgoing):
                    continue  # degenerate (self-implication etc.)
                premise_tvs = tuple(
                    self.get_tv(p) or TruthValue() for p in premise_atoms
                )
                new_tv = rule.tv_fn(self, binding, premise_tvs)
                if new_tv is None or new_tv.count <= 0.0:
                    continue
                with self._lock:
                    already = self._records.get(conclusion)
                    if already is not None and already.tv.count >= new_tv.count:
                        continue
                    self._add_locked(conclusion, new_tv)
                    self._derived_total += 1
                derived.append(conclusion)
        for atom in derived:
            self.stimulate(atom, self._stimulus_size * 0.25)
        return derived

    def forward_chain(
        self,
        *,
        max_derivations: int = 16,
        focus_only: bool = True,
        rules: Sequence["InferenceRule"] | None = None,
    ) -> list[Atom]:
        """One bounded forward-chaining pass with the standard PLN rule set."""
        return self.apply_rules(
            rules if rules is not None else standard_pln_rules(),
            max_derivations=max_derivations,
            focus_only=focus_only,
        )

    def explain(
        self,
        target: Link,
        *,
        max_depth: int = 4,
        max_paths: int = 8,
    ) -> list[dict[str, Any]]:
        """Backward chaining: find implication chains that support ``target``.

        Goal-directed PLN — given ``Implication(a, c)``, search stored
        implications for chains ``a → … → c`` (depth-bounded), composing each
        chain's truth value with the deduction formula. Returns up to
        ``max_paths`` explanations sorted by evidential strength: the *why*
        behind a derived or queried link.
        """
        if not (isinstance(target, Link) and target.atype == IMPLICATION and len(target.outgoing) == 2):
            return []
        start, goal_atom = target.outgoing
        with self._lock:
            by_source: dict[Atom, list[Link]] = {}
            for link in self._by_type.get(IMPLICATION, ()):
                if isinstance(link, Link) and len(link.outgoing) == 2:
                    by_source.setdefault(link.outgoing[0], []).append(link)
        found: list[dict[str, Any]] = []

        def dfs(node: Atom, chain: list[Link], visited: frozenset[Atom]) -> None:
            if len(found) >= max_paths or len(chain) > max_depth:
                return
            if node == goal_atom and chain:
                tv = self.get_tv(chain[0]) or TruthValue()
                for link in chain[1:]:
                    mid = self.get_tv(link.outgoing[0]) or TruthValue()
                    end = self.get_tv(link.outgoing[1]) or TruthValue()
                    tv = deduction_tv(tv, self.get_tv(link) or TruthValue(), mid, end)
                found.append(
                    {
                        "chain": [str(link) for link in chain],
                        "strength": tv.strength,
                        "confidence": tv.confidence,
                        "hops": len(chain),
                    }
                )
                return
            for link in by_source.get(node, ()):
                nxt = link.outgoing[1]
                if nxt in visited or link == target:
                    continue
                dfs(nxt, chain + [link], visited | {nxt})

        dfs(start, [], frozenset({start}))
        return sorted(found, key=lambda e: (e["strength"] * e["confidence"]), reverse=True)

    # ── introspection ─────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "atoms": len(self._records),
                "by_type": {t: len(s) for t, s in self._by_type.items() if s},
                "sti_fund": self._sti_fund,
                "focus": [(str(a), round(s, 2)) for a, s in self.attentional_focus(5)],
                "forgotten_total": self._forgotten_total,
                "derived_total": self._derived_total,
            }


# ── Inference rules: unification + rewrite + truth formula ────────────────

@dataclass(frozen=True)
class InferenceRule:
    """A pattern-based inference rule (the MeTTa/PLN shape).

    ``premises`` are patterns (may share Variables), ``conclusion`` is a
    template over the same variables, and ``tv_fn`` computes the derived
    truth value from the space, the binding, and the premise TVs (returning
    None to veto the instance).
    """

    name: str
    premises: tuple[Atom, ...]
    conclusion: Atom
    tv_fn: Callable[["AtomSpace", Bindings, tuple[TruthValue, ...]], TruthValue | None]


def _node_tv(space: "AtomSpace", binding: Bindings, var: str) -> TruthValue:
    atom = binding.get(var)
    if atom is None:
        return TruthValue()
    return space.get_tv(atom) or TruthValue()


def _rule_deduction() -> InferenceRule:
    a, b, c = Variable("a"), Variable("b"), Variable("c")
    return InferenceRule(
        name="pln.deduction",
        premises=(Link(IMPLICATION, (a, b)), Link(IMPLICATION, (b, c))),
        conclusion=Link(IMPLICATION, (a, c)),
        tv_fn=lambda space, bind, tvs: deduction_tv(
            tvs[0], tvs[1], _node_tv(space, bind, "b"), _node_tv(space, bind, "c")
        ),
    )


def _rule_abduction() -> InferenceRule:
    a, b, c = Variable("a"), Variable("b"), Variable("c")
    return InferenceRule(
        name="pln.abduction",
        premises=(Link(IMPLICATION, (a, c)), Link(IMPLICATION, (b, c))),
        conclusion=Link(IMPLICATION, (a, b)),
        tv_fn=lambda space, bind, tvs: abduction_tv(
            tvs[0],
            tvs[1],
            _node_tv(space, bind, "a"),
            _node_tv(space, bind, "b"),
            _node_tv(space, bind, "c"),
        ),
    )


def _rule_induction() -> InferenceRule:
    a, b, c = Variable("a"), Variable("b"), Variable("c")
    return InferenceRule(
        name="pln.induction",
        premises=(Link(IMPLICATION, (a, b)), Link(IMPLICATION, (a, c))),
        conclusion=Link(IMPLICATION, (b, c)),
        tv_fn=lambda space, bind, tvs: induction_tv(
            tvs[0],
            tvs[1],
            _node_tv(space, bind, "a"),
            _node_tv(space, bind, "b"),
            _node_tv(space, bind, "c"),
        ),
    )


def standard_pln_rules() -> tuple[InferenceRule, ...]:
    """The default live rule set: deduction, abduction, induction."""
    return (_rule_deduction(), _rule_abduction(), _rule_induction())


# ── Claim bridge: NL beliefs → atoms (shared encoding with the prover) ────

def assert_claim(
    space: AtomSpace,
    claim: str,
    tv: TruthValue,
    *,
    domain: str = "world",
    stimulate: bool = True,
    source: str | None = None,
) -> tuple[Atom, TruthValue]:
    """Assert a natural-language claim into the space and return its revised TV.

    ``source`` names the observation behind the claim. Without one, repeated
    assertion of the same claim revises PLN-style and accumulates confidence;
    with one, a source restating itself replaces its own contribution. Callers
    mirroring a store that already accumulates evidence want the latter, or
    both sides count the same evidence.

    Uses the same propositional encoder as the deduction prover
    (``core/reasoning/belief_consistency.encode_belief``) so the atom
    namespace and the prover's atom namespace are one: implication-shaped
    claims become ``Implication`` links (fuel for PLN deduction), negated
    claims assert the concept with inverted strength (PLN negation).
    """
    from core.reasoning.belief_consistency import encode_belief
    from core.reasoning.natural_deduction import Implies, Not, Atom as PropAtom

    encoded = encode_belief(claim)
    formula = encoded.formula
    if isinstance(formula, Implies):
        def _leaf(f: Any) -> tuple[str, bool]:
            if isinstance(f, Not) and isinstance(f.f, PropAtom):
                return f.f.name, True
            if isinstance(f, PropAtom):
                return f.name, False
            return str(f), False

        ante, ante_neg = _leaf(formula.a)
        cons, cons_neg = _leaf(formula.b)
        # Negated endpoints keep their polarity in the concept name so the
        # implication link is faithful to the claim's logical content.
        a_node = concept(("¬" if ante_neg else "") + ante)
        c_node = concept(("¬" if cons_neg else "") + cons)
        atom: Atom = implication(a_node, c_node)
        out_tv = space.add(atom, tv, source=source)
    else:
        atom = concept(encoded.core_key)
        out_tv = space.add(atom, tv.negation() if encoded.negated else tv, source=source)
    ev = evaluation(predicate("claim_domain"), atom, concept(domain))
    # The domain link is a stipulation about the claim, not evidence for it;
    # it is asserted under its own identity so re-asserting a claim does not
    # accumulate confidence in what domain it belongs to.
    space.add(ev, TruthValue(1.0, 1.0), source=f"claim_domain:{domain}")
    if stimulate:
        space.stimulate(atom)
    return atom, out_tv


# ── Singleton ─────────────────────────────────────────────────────────────

_space_lock = checked_lock("core.knowledge.atomspace.spaces")
_space: AtomSpace | None = None


def _where_it_is_kept() -> Any:
    from pathlib import Path as _Path

    from core.runtime.state_ownership import state_root

    return _Path(state_root()) / "atomspace.json"


def _fill_it_from_disk(space: "AtomSpace") -> int:
    """Load the last snapshot into a fresh store. Returns atoms loaded.

    The store was built empty on every boot and never written, so the whole
    metagraph — the claims, the attention values, and the per-source
    attribution that stops a reload double-counting a witness — lasted exactly
    as long as the process. The other half of this class has had a save and a
    load in its own file the whole time and nothing outside its test called
    either.
    """

    place = _where_it_is_kept()
    if not place.exists():
        return 0
    try:
        from core.knowledge.atomspace_persistence import load

        return int(load(space, place))
    except Exception as exc:  # noqa: BLE001 — a bad snapshot is not a dead boot
        from core.runtime.errors import record_degradation

        record_degradation(
            "atomspace",
            exc,
            severity="warning",
            action="started from an empty metagraph rather than the snapshot",
        )
        return 0


def keep_the_atomspace() -> int:
    """Write the store to disk. Returns atoms written, or -1 if it could not.

    Blocking, deliberately: this is the shutdown path and the periodic one, and
    both want the write finished before they move on. Never from the event loop
    — ``atomspace_persistence.save_async`` is there for that.
    """

    with _space_lock:
        space = _space
    if space is None:
        return 0
    try:
        from core.governance_context import local_internal_governed_scope
        from core.knowledge.atomspace_persistence import save

        with local_internal_governed_scope("atomspace.keep", domain="state_mutation"):
            return int(save(space, _where_it_is_kept()))
    except Exception as exc:  # noqa: BLE001 — losing a snapshot is not fatal
        from core.runtime.errors import record_degradation

        record_degradation(
            "atomspace",
            exc,
            severity="warning",
            action="the metagraph was not written and will not survive the restart",
        )
        return -1


def get_atomspace() -> AtomSpace:
    global _space
    with _space_lock:
        if _space is None:
            _space = AtomSpace()
            fresh = _space
        else:
            return _space
    # Outside the lock: loading reaches back into the store, and a restore that
    # takes the same lock under the one that built it is how a boot wedges.
    _fill_it_from_disk(fresh)
    if _this_process_owns_the_state():
        import atexit

        atexit.register(keep_the_atomspace)
    return fresh


def _this_process_owns_the_state() -> bool:
    """Whether this process may write the metagraph it just read.

    Reading is safe from anywhere. Writing is not: a bare script or a test run
    that touches the store and exits would put its handful of atoms into the
    live instance's state root, and the next boot would load them as her own.
    That happened once, from a probe, before this guard existed.

    The profile is the answer already recorded elsewhere in the runtime — a
    live profile writes, and anything else reads what it finds and leaves it.
    """

    try:
        from core.runtime.state_ownership import RuntimeProfile, runtime_profile

        return runtime_profile() is RuntimeProfile.LIVE
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return False


def reset_atomspace_for_test(**kwargs: Any) -> AtomSpace:
    global _space
    with _space_lock:
        _space = AtomSpace(**kwargs)
        return _space


__all__ = [
    "Atom",
    "AtomSpace",
    "keep_the_atomspace",
    "AttentionValue",
    "Bindings",
    "CONCEPT",
    "EVALUATION",
    "GROUNDED_PREDICATE",
    "HEBBIAN",
    "IMPLICATION",
    "INHERITANCE",
    "InferenceRule",
    "LIST",
    "Link",
    "Node",
    "PREDICATE",
    "TruthValue",
    "Variable",
    "abduction_tv",
    "assert_claim",
    "concept",
    "deduction_tv",
    "evaluation",
    "get_atomspace",
    "implication",
    "induction_tv",
    "inversion_tv",
    "predicate",
    "reset_atomspace_for_test",
    "standard_pln_rules",
    "substitute",
    "unify",
]
