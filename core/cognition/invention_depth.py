"""core/cognition/invention_depth.py — is anything compounding.

Composing human-supplied primitives is not invention, however elaborate the
compositions get. The question that separates the two is whether what was
invented becomes available as material for the next invention. If generation
five can only build from the primitives a person wrote down, the system's
expressive power never grew; it only got busier.

Two things have to be true and both are easy to fake.

**The new thing has to be new.** A definition expressible as a composition of
what already exists is a macro. It may be useful — most abbreviations are —
and it adds nothing to what can be said. The check has to run against the
*closure* of the existing vocabulary and not against its primitives, which is
a mistake already made in this codebase once: a macro check that looked only
at the base set counted new syntax as new meaning, and the measured vocabulary
grew while the expressible set did not.

**It has to be known when it applies.** A primitive nobody has tried anywhere
has no domain, and that is a different state from applying everywhere. The two
were the same state here at first — an empty domain list meaning both — which
is the shape that lets an untested thing be reached for as though it were
general.

**It has to be reasoned with.** A primitive invented, kept, and never used in
a later derivation was stored rather than learned, and a vocabulary counting
those is counting its own filing.

**Invention has to feed invention.** A generation whose members all depend
only on generation zero is a wide vocabulary, not a deep one. Depth is the
longest chain where each link was invented on top of the one before it, and it
is the number that says whether anything is compounding.

Novelty is judged on extension rather than on form. A primitive is what it
does — the outputs it gives over a probe set — so two definitions that agree
everywhere are the same primitive whatever they look like, and a definition
that agrees with no composition of the existing vocabulary is genuinely new.
That makes the refusal checkable rather than a matter of opinion, and it is
why the probe set has to be fixed before the proposal rather than chosen with
it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Cognition.Invention")

#: How deep a composition search goes when checking whether a proposal is
#: already expressible. Bounded because the search is exponential; a proposal
#: that needs more than this to reconstruct is treated as new, and the bound
#: is reported so a reader knows what "new" meant.
MAX_COMPOSITION_DEPTH = 3

#: Probe inputs a proposal must be distinguishable on. Below this two
#: primitives can agree by accident and the extension says nothing.
MIN_PROBES = 4


class Verdict(StrEnum):
    """What a proposed primitive turned out to be."""

    #: Not expressible in the existing vocabulary. Genuinely new.
    INVENTED = "invented"
    #: Expressible as a composition of what exists. An abbreviation.
    MACRO = "macro"
    #: Agrees with an existing primitive everywhere probed.
    DUPLICATE = "duplicate"
    #: Too few probes to tell any of these apart.
    UNDECIDABLE = "undecidable"

    @property
    def grows_the_language(self) -> bool:
        return self is Verdict.INVENTED


@dataclass(frozen=True)
class Primitive:
    """One operation, and where it came from."""

    name: str
    #: What it does. Judged on what this returns, never on how it is written.
    fn: Callable[[Any], Any]
    #: 0 for what a person supplied; n for something invented on top of
    #: generation n-1.
    generation: int = 0
    #: Invented primitives this one is built from. Empty for generation zero.
    depends_on: tuple[str, ...] = ()
    #: Where it applies, learned from where it was used and worked. Empty
    #: means nobody has established a domain, which is different from
    #: applying everywhere — a primitive with no known domain is one nobody
    #: knows when to reach for.
    applies_where: tuple[str, ...] = ()
    #: How many times it was actually used in a later derivation. A primitive
    #: nothing reasons with is stored, not learned.
    used_in_reasoning: int = 0

    def extension(self, probes: Sequence[Any]) -> tuple[Any, ...]:
        """What it does over the probe set. Its identity."""
        out = []
        for probe in probes:
            try:
                out.append(self.fn(probe))
            except (TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
                out.append(_UNDEFINED)
        return tuple(out)

    @property
    def applicability_known(self) -> bool:
        return bool(self.applies_where)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "generation": self.generation,
            "depends_on": list(self.depends_on),
            "applies_where": list(self.applies_where),
            "used_in_reasoning": self.used_in_reasoning,
            "applicability_known": self.applicability_known,
        }


class _Undefined:
    """What a primitive returns where it does not apply. Its own value, so two
    primitives that both fail on a probe are not thereby the same."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<undefined>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Undefined)

    def __hash__(self) -> int:
        return hash("<undefined>")


_UNDEFINED = _Undefined()


@dataclass(frozen=True)
class Proposal:
    """A verdict on one proposed primitive."""

    name: str
    verdict: Verdict
    #: The composition that already does this, when there is one.
    equivalent_to: tuple[str, ...] = ()
    depth_searched: int = 0
    #: Declared dependencies that are not in the vocabulary. Reported rather
    #: than dropped: an invention standing on something that was refused is
    #: not standing on anything, and silence there would make a chain look
    #: deeper than it is.
    unknown_dependencies: tuple[str, ...] = ()
    generation: int = 0
    because: str = ""

    @property
    def accepted(self) -> bool:
        return self.verdict.grows_the_language

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "verdict": str(self.verdict),
            "accepted": self.accepted,
            "equivalent_to": list(self.equivalent_to),
            "depth_searched": self.depth_searched,
            "unknown_dependencies": list(self.unknown_dependencies),
            "generation": self.generation,
            "because": self.because,
        }


class Vocabulary:
    """Everything she can say, and how much of it she worked out herself."""

    def __init__(self, probes: Sequence[Any]) -> None:
        # Fixed at construction, before any proposal. A probe set chosen
        # alongside a proposal can be chosen to make it look new.
        self._probes = tuple(probes)
        self._primitives: dict[str, Primitive] = {}

    @property
    def probes(self) -> tuple[Any, ...]:
        return self._probes

    def supply(self, name: str, fn: Callable[[Any], Any]) -> Primitive:
        """Add a primitive a person wrote. Generation zero."""
        primitive = Primitive(name=name, fn=fn, generation=0)
        self._primitives[name] = primitive
        return primitive

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._primitives))

    def get(self, name: str) -> Primitive | None:
        return self._primitives.get(name)

    # ── the closure ──────────────────────────────────────────────────────

    def _compositions(self, depth: int) -> dict[tuple[Any, ...], tuple[str, ...]]:
        """Every extension reachable by composing up to `depth` primitives.

        The closure, not the base set. Checking a proposal against the
        primitives alone counts new syntax as new meaning, which is how a
        vocabulary grows while the expressible set stands still.
        """
        reachable: dict[tuple[Any, ...], tuple[str, ...]] = {}
        current: list[tuple[Callable[[Any], Any], tuple[str, ...]]] = [
            (p.fn, (p.name,)) for p in self._primitives.values()
        ]
        for _ in range(max(1, depth)):
            for fn, chain in current:
                extension = Primitive(name="", fn=fn).extension(self._probes)
                reachable.setdefault(extension, chain)
            nxt: list[tuple[Callable[[Any], Any], tuple[str, ...]]] = []
            for outer in self._primitives.values():
                for fn, chain in current:
                    if len(chain) >= depth:
                        continue
                    composed = _compose(outer.fn, fn)
                    nxt.append((composed, (outer.name, *chain)))
            if not nxt:
                break
            current = nxt
        return reachable

    def judge(
        self, name: str, fn: Callable[[Any], Any], *, depth: int = MAX_COMPOSITION_DEPTH
    ) -> Proposal:
        """Whether this proposal adds anything to what can be said."""
        if len(self._probes) < MIN_PROBES:
            return Proposal(
                name=name,
                verdict=Verdict.UNDECIDABLE,
                depth_searched=0,
                because=(
                    f"{len(self._probes)} probes; {MIN_PROBES} are needed before "
                    "two primitives agreeing means they are the same"
                ),
            )
        candidate = Primitive(name=name, fn=fn).extension(self._probes)
        for existing in self._primitives.values():
            if existing.extension(self._probes) == candidate:
                return Proposal(
                    name=name,
                    verdict=Verdict.DUPLICATE,
                    equivalent_to=(existing.name,),
                    depth_searched=1,
                    because=f"agrees with {existing.name} on every probe",
                )
        reachable = self._compositions(depth)
        chain = reachable.get(candidate)
        if chain is not None:
            return Proposal(
                name=name,
                verdict=Verdict.MACRO,
                equivalent_to=chain,
                depth_searched=depth,
                because=(
                    f"the composition {' ∘ '.join(chain)} already does this; "
                    "an abbreviation is useful and adds nothing to what can be said"
                ),
            )
        return Proposal(
            name=name,
            verdict=Verdict.INVENTED,
            depth_searched=depth,
            because=(
                f"no composition of {len(self._primitives)} primitives up to "
                f"depth {depth} produces this over {len(self._probes)} probes"
            ),
        )

    def invent(
        self,
        name: str,
        fn: Callable[[Any], Any],
        *,
        depends_on: Iterable[str] = (),
        depth: int = MAX_COMPOSITION_DEPTH,
    ) -> Proposal:
        """Judge a proposal and keep it if it adds something."""
        declared = tuple(sorted({str(d) for d in depends_on}))
        parents = tuple(d for d in declared if d in self._primitives)
        missing = tuple(d for d in declared if d not in self._primitives)
        verdict = self.judge(name, fn, depth=depth)
        if not verdict.accepted:
            return Proposal(
                name=verdict.name,
                verdict=verdict.verdict,
                equivalent_to=verdict.equivalent_to,
                depth_searched=verdict.depth_searched,
                unknown_dependencies=missing,
                because=verdict.because,
            )
        if missing:
            logger.info(
                "%s declares %d dependency it does not have: %s",
                name,
                len(missing),
                list(missing),
            )
        generation = 1 + max(
            (self._primitives[p].generation for p in parents), default=0
        )
        self._primitives[name] = Primitive(
            name=name, fn=fn, generation=generation, depends_on=parents
        )
        return Proposal(
            name=verdict.name,
            verdict=verdict.verdict,
            equivalent_to=verdict.equivalent_to,
            depth_searched=verdict.depth_searched,
            unknown_dependencies=missing,
            generation=generation,
            because=verdict.because,
        )

    # ── is anything compounding ──────────────────────────────────────────

    @property
    def depth(self) -> int:
        """The longest chain of inventions each built on the one before.

        One means every invention is composed of what a person supplied: a
        wide vocabulary, not a deep one. Above one, invention is feeding
        invention, which is the property the whole module is about.
        """
        invented = [p for p in self._primitives.values() if p.generation > 0]
        return max((p.generation for p in invented), default=0)

    def note_applies(self, name: str, domain: str, *, worked: bool) -> bool:
        """Record that a primitive was used somewhere and whether it helped.

        Applicability is learned rather than declared. A primitive nobody has
        tried anywhere has no known domain, which is a different state from
        applying everywhere, and the two used to be the same state — the
        empty tuple meaning both "untested" and "unrestricted".
        """
        primitive = self._primitives.get(name)
        if primitive is None:
            return False
        domains = set(primitive.applies_where)
        if worked:
            domains.add(str(domain))
        else:
            domains.discard(str(domain))
        self._primitives[name] = Primitive(
            name=primitive.name,
            fn=primitive.fn,
            generation=primitive.generation,
            depends_on=primitive.depends_on,
            applies_where=tuple(sorted(domains)),
            used_in_reasoning=primitive.used_in_reasoning + 1,
        )
        return True

    def stored_but_unused(self) -> tuple[str, ...]:
        """Invented, kept, and never reasoned with. Storage is not learning."""
        return tuple(
            sorted(
                p.name
                for p in self._primitives.values()
                if p.generation > 0 and p.used_in_reasoning == 0
            )
        )

    def lineage(self, name: str) -> tuple[str, ...]:
        """The chain of inventions this one stands on, oldest last."""
        primitive = self._primitives.get(name)
        if primitive is None:
            return ()
        chain: list[str] = []
        frontier = list(primitive.depends_on)
        seen: set[str] = set()
        while frontier:
            current = frontier.pop(0)
            if current in seen:
                continue
            seen.add(current)
            chain.append(current)
            parent = self._primitives.get(current)
            if parent is not None:
                frontier.extend(parent.depends_on)
        return tuple(chain)

    def snapshot(self) -> dict[str, Any]:
        by_generation: dict[int, int] = {}
        for primitive in self._primitives.values():
            by_generation[primitive.generation] = (
                by_generation.get(primitive.generation, 0) + 1
            )
        return {
            "primitives": len(self._primitives),
            "supplied": by_generation.get(0, 0),
            "invented": len(self._primitives) - by_generation.get(0, 0),
            "by_generation": {str(k): v for k, v in sorted(by_generation.items())},
            "depth": self.depth,
            "probes": len(self._probes),
            "with_known_domain": sum(
                1 for p in self._primitives.values()
                if p.generation > 0 and p.applicability_known
            ),
            "stored_but_unused": list(self.stored_but_unused()),
        }


def _compose(outer: Callable[[Any], Any], inner: Callable[[Any], Any]):
    def composed(value: Any) -> Any:
        return outer(inner(value))

    return composed


__all__ = [
    "MAX_COMPOSITION_DEPTH",
    "MIN_PROBES",
    "Primitive",
    "Proposal",
    "Verdict",
    "Vocabulary",
]
