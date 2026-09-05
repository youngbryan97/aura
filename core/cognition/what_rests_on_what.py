"""What rests on what, and what to do when the thing underneath goes.

Removing a head already takes the words written over it, which is a cascade and
is the least of the three things a dependency graph has to do. The other two
are the ones that decide whether a retraction is a repair or a second defect.

**Quarantine** rather than delete. A head suspected of being wrong is not the
same as a head known to be wrong, and throwing the first away loses the
evidence that would have settled which it was. A quarantined head stops being
offered and stops answering, and everything about it is still there.

**Rebuild** rather than mourn. A word written over a head that is gone is not
necessarily a word that cannot exist — the family it was derived from may be
sayable another way, and the honest thing is to try before writing it off. What
cannot be rebuilt goes inactive, and inactive is a state a person can read
rather than an absence they have to infer.

Read off the construction, never off the spelling
-------------------------------------------------
`one_algebra.what_it_rests_on` already does this for words: a word rests on the
words in its holes and on everything THOSE rest on, read from `built_from`
rather than from a name. The same discipline here. A word rests on a head when
its term mentions that head, which is a fact about the term; a head rests on
another head when its body contains that body, which is a fact about the term
too. Neither is a guess about what something is called.

What a floor term does not need
-------------------------------
A head written over another head needs no cascade at all, and that is a
property of the representation rather than an oversight. A floor term contains
its parts instead of pointing at them, so a descendant carries a copy of its
ancestor's body and keeps computing after the ancestor is gone. The dependency
is provenance, not linkage, and the distinction is worth keeping straight: a
graph that cascaded through it would be destroying things that still work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "QUARANTINED",
    "WhatWentWithIt",
    "quarantine",
    "rebuild",
    "release",
    "rests_on",
    "retract",
    "what_rests_on_it",
]

logger = logging.getLogger("Aura.WhatRestsOnWhat")

#: The heads set aside rather than removed, and the term each of them had. A
#: quarantined head answers nothing and is offered nowhere; releasing it puts
#: back exactly what was taken.
QUARANTINED: dict[str, Any] = {}


@dataclass(frozen=True)
class WhatWentWithIt:
    """What a retraction took, what it rebuilt, and what it left inactive."""

    head: str
    removed: bool
    #: Words whose term mentioned the head.
    words: tuple[str, ...] = ()
    #: Of those, the ones derived again from what remains.
    rebuilt: tuple[str, ...] = ()
    #: And the ones that could not be, which are gone rather than wrong.
    inactive: tuple[str, ...] = ()
    #: Heads whose body contains this one's. Recorded and not cascaded.
    rests_on_it: tuple[str, ...] = field(default_factory=tuple)

    def describes(self) -> str:
        if not self.removed:
            return f"{self.head!r} was not there"
        return (
            f"{self.head!r} went, and with it {len(self.words)} word(s): "
            f"{len(self.rebuilt)} rebuilt, {len(self.inactive)} inactive. "
            f"{len(self.rests_on_it)} head(s) carry a copy of it and keep working"
        )


def _mentions(term: Any, head: str) -> bool:
    """Whether this positional term uses that head, anywhere inside it."""
    edge = [term]
    while edge:
        here = edge.pop()
        if getattr(here, "head", None) == head:
            return True
        edge.extend(getattr(here, "parts", ()) or ())
    return False


def _contains(body: Any, piece: Any) -> bool:
    """Whether this floor term carries that one inside it."""
    edge = [body]
    while edge:
        here = edge.pop()
        if here == piece:
            return True
        edge.extend(getattr(here, "parts", ()) or ())
    return False


def rests_on(name: str) -> frozenset[str]:
    """Every head this word or head rests on, read off the construction."""
    from core.cognition.an_invented_kind import WHERE_FROM
    from core.cognition.one_algebra import DERIVED_HEADS

    word = WHERE_FROM.get(name)
    term = getattr(word, "term", None)
    if term is not None:
        return frozenset(one for one in DERIVED_HEADS if _mentions(term, one))
    head = DERIVED_HEADS.get(name)
    if head is None:
        return frozenset()
    return frozenset(
        one
        for one, other in DERIVED_HEADS.items()
        if one != name and _contains(head.body, other.body)
    )


def what_rests_on_it(head: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The words that would break without this head, and the heads that would not."""
    from core.cognition.an_invented_kind import WHERE_FROM
    from core.cognition.one_algebra import DERIVED_HEADS

    words = tuple(
        sorted(
            name
            for name, word in WHERE_FROM.items()
            if getattr(word, "term", None) is not None
            and _mentions(word.term, head)
        )
    )
    mine = DERIVED_HEADS.get(head)
    heads: tuple[str, ...] = ()
    if mine is not None:
        heads = tuple(
            sorted(
                name
                for name, other in DERIVED_HEADS.items()
                if name != head and _contains(other.body, mine.body)
            )
        )
    return words, heads


def quarantine(head: str) -> bool:
    """Set a head aside without losing it.

    Suspected wrong is not known wrong, and deleting the first loses whatever
    would have settled which it was.
    """
    from core.cognition.one_algebra import DERIVED_HEADS

    said = str(head)
    if said in QUARANTINED or said not in DERIVED_HEADS:
        return False
    QUARANTINED[said] = DERIVED_HEADS.pop(said)
    logger.info("quarantined %r; nothing is offered it and nothing answers it", said)
    return True


def release(head: str) -> bool:
    """Put back exactly what quarantine took."""
    from core.cognition.one_algebra import DERIVED_HEADS

    said = str(head)
    put_back = QUARANTINED.pop(said, None)
    if put_back is None:
        return False
    DERIVED_HEADS[said] = put_back
    logger.info("released %r", said)
    return True


def rebuild(
    words: tuple[str, ...],
    *,
    derive: Callable[[str], Any] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Try to derive each word again from what remains.

    ``derive`` is the caller's, because what it means to derive a word is a
    question about the family it came from and this module does not hold
    those. Without one it asks the developmental policy, which does hold them:
    losing a word is an ordinary developmental objective — say these families
    again without the part that went — and the ranking already knows how to
    answer that. Where even that gives nothing, nothing is rebuilt and
    everything is reported inactive, which is the honest answer rather than a
    silent success.
    """
    from core.cognition.an_invented_kind import WHERE_FROM

    if derive is None:
        derive = _ask_the_policy

    made: list[str] = []
    lost: list[str] = []
    for name in words:
        again = derive(name) if derive is not None else None
        if again is None:
            lost.append(name)
            continue
        WHERE_FROM[name] = again
        made.append(name)
    return tuple(made), tuple(lost)


def retract(
    head: str,
    *,
    derive: Callable[[str], Any] | None = None,
) -> WhatWentWithIt:
    """Take a head out, take what rested on it, and rebuild what can be.

    The order matters. What rests on it is read before anything is removed,
    because reading it afterwards reads a graph the removal has already
    changed.
    """
    from core.cognition.an_invented_kind import WHERE_FROM
    from core.cognition.one_algebra import DERIVED_HEADS

    said = str(head)
    if said not in DERIVED_HEADS:
        return WhatWentWithIt(head=said, removed=False)
    words, heads = what_rests_on_it(said)
    DERIVED_HEADS.pop(said, None)
    for name in words:
        WHERE_FROM.pop(name, None)
    made, lost = rebuild(words, derive=derive)
    found = WhatWentWithIt(
        head=said,
        removed=True,
        words=words,
        rebuilt=made,
        inactive=lost,
        rests_on_it=heads,
    )
    logger.info("retracted — %s", found.describes())
    return found


def _ask_the_policy(name: str) -> Any | None:
    """Derive a lost word by asking what is worth doing about losing it.

    A retraction leaves families that used to be sayable and are not. That is
    an occasion the ranking is already built for, and routing it there is what
    keeps rebuilding from being a second mechanism with its own idea of what a
    word is.
    """
    try:
        from core.cognition.she_decides_to_develop import she_develops_herself
        from core.cognition.the_record_of_her_own_work import the_record
    except ImportError:
        return None
    if not the_record().kept:
        return None
    try:
        _decided, came_of_it = she_develops_herself()
    except Exception:  # noqa: BLE001 - a rebuild that fails is a lost word
        logger.info("could not rebuild %s", name, exc_info=True)
        return None
    from core.cognition.an_invented_kind import WHERE_FROM

    return WHERE_FROM.get(name) if came_of_it else None
