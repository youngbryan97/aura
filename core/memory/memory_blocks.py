"""Memory blocks — the part of the context window Aura writes herself.

`IdentityCore` already tiers identity by mutability: an immutable base and
heartstone, semi-static legacy, and an evolved file that drifts. What it has
no notion of is *budget* or *address*. The evolved file grows without a
ceiling, nothing names its regions, and the only way it changes is a whole-file
rewrite from outside the turn. So Aura cannot say "this is what I now know
about Bryan, in the 2000 characters I have for it" — she can only have a file
that someone else edits.

A memory block is a labelled, bounded region of the context window with an
owner and an edit history. The label is an address the model can name in a tool
call; the limit is a budget it has to live within; the history is what stops a
self-editing memory from being an agent that can silently rewrite its own past.

Three positions this module takes, in order of how much they matter:

* **Overflow is refused, never truncated.** Silently trimming a self-edited
  block discards exactly the text the agent judged important enough to write,
  and leaves it believing the write succeeded. A refusal it can see and react
  to is strictly better than a success it cannot trust.
* **Immutable blocks refuse edits out loud.** Identity that can be edited by
  the thing whose identity it is has no anchoring property at all. But a
  silently ignored edit is worse than a refused one — the agent proceeds
  believing it changed something.
* **Shared blocks are compare-and-swap.** Sleep-time consolidation runs against
  the same blocks the live turn is editing. Last-write-wins across that seam
  loses whichever one finished first, silently, and the loser is usually the
  turn that had the user in front of it.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

logger = logging.getLogger("Aura.MemoryBlocks")

__all__ = [
    "MemoryBlock",
    "BlockEdit",
    "MemoryBlockSet",
    "BlockOverflow",
    "BlockImmutable",
    "BlockDerived",
    "BlockVersionConflict",
    "UnknownBlock",
]

DEFAULT_LIMIT = 2000


class BlockOverflow(ValueError):
    """The write would exceed the block's character limit."""


class BlockImmutable(PermissionError):
    """The block refuses edits — by anyone, including its owner."""


class BlockDerived(PermissionError):
    """The block is rendered from a structured source, and this author is not it.

    Distinct from immutability: a derived block changes constantly. What it
    refuses is being *rewritten as prose* by anyone other than the store whose
    records it displays.
    """


class BlockVersionConflict(RuntimeError):
    """Someone else edited the block since the version the caller read."""


class UnknownBlock(KeyError):
    """No block by that label."""


@dataclass(frozen=True)
class BlockEdit:
    """One recorded change. Attributed, timestamped, reversible."""

    label: str
    operation: str
    before: str
    after: str
    author: str
    reason: str = ""
    at: float = field(default_factory=time.time)


@dataclass
class MemoryBlock:
    """A labelled, bounded region of the context window.

    ``description`` is written for the model, not for us: it is rendered into
    the prompt so the agent knows what the block is *for*. A block whose purpose
    the agent has to infer gets filled with the wrong things.
    """

    label: str
    value: str = ""
    limit: int = DEFAULT_LIMIT
    description: str = ""
    immutable: bool = False
    version: int = 0
    #: The source that owns this block's text, if it is rendered from one.
    #:
    #: Some blocks are not prose at all — they are the display surface of a
    #: structured store, regenerated from records with fields. What she knows
    #: about a person is the case that forced this: summarising that block is
    #: exactly the operation that turns "seemed frustrated once, during a
    #: failing deploy" into "is easily frustrated", because compression drops
    #: adjectives before nouns and the qualifiers there are all adjectives.
    #:
    #: Naming the owner rather than setting a flag means the refusal can say
    #: who *should* be writing, and means the store itself is not locked out of
    #: its own block.
    derived_from: str = ""

    def __post_init__(self) -> None:
        if not self.label or not self.label.strip():
            raise ValueError("a block needs a label; it is how the model addresses it")
        if self.limit < 1:
            raise ValueError(f"block {self.label!r} needs a positive limit")
        if len(self.value) > self.limit:
            raise BlockOverflow(
                f"block {self.label!r} was constructed over its own limit "
                f"({len(self.value)} > {self.limit})"
            )

    @property
    def remaining(self) -> int:
        return self.limit - len(self.value)

    @property
    def utilization(self) -> float:
        return len(self.value) / self.limit

    def render(self) -> str:
        """The block as it appears in the prompt.

        The limit is shown deliberately. An agent that cannot see how much room
        it has left cannot budget, and will discover the ceiling only by
        hitting it.
        """
        header = f"<{self.label}>"
        if self.description:
            header += f"\n({self.description})"
        header += f"\n[{len(self.value)}/{self.limit} characters used]"
        return f"{header}\n{self.value}\n</{self.label}>"


class MemoryBlockSet:
    """A named collection of blocks, with edit history and CAS writes."""

    def __init__(self, blocks: Iterable[MemoryBlock] = ()) -> None:
        self._blocks: dict[str, MemoryBlock] = {}
        self._history: list[BlockEdit] = []
        for block in blocks:
            self.attach(block)

    # -- membership --------------------------------------------------------

    def attach(self, block: MemoryBlock) -> MemoryBlock:
        if block.label in self._blocks:
            raise ValueError(
                f"block {block.label!r} is already attached; labels are addresses "
                "and two blocks answering to one address is not resolvable"
            )
        self._blocks[block.label] = block
        return block

    def get(self, label: str) -> MemoryBlock:
        try:
            return self._blocks[label]
        except KeyError:
            raise UnknownBlock(
                f"no block {label!r}; attached: {sorted(self._blocks)}"
            ) from None

    def __contains__(self, label: object) -> bool:
        return label in self._blocks

    def __iter__(self) -> Iterator[MemoryBlock]:
        return iter(self._blocks.values())

    def __len__(self) -> int:
        return len(self._blocks)

    @property
    def labels(self) -> list[str]:
        return sorted(self._blocks)

    # -- the write path ----------------------------------------------------

    def _write(
        self,
        label: str,
        new_value: str,
        *,
        operation: str,
        author: str,
        reason: str,
        expected_version: int | None,
    ) -> MemoryBlock:
        block = self.get(label)

        if block.immutable:
            raise BlockImmutable(
                f"block {label!r} is immutable. It is refusing rather than "
                "ignoring you: proceeding as though this succeeded would be worse."
            )

        if block.derived_from and author != block.derived_from:
            raise BlockDerived(
                f"block {label!r} is rendered from {block.derived_from!r} and "
                f"{author!r} may not rewrite it. Its text is generated from "
                "records with fields; rewriting it as prose would discard the "
                "structure that keeps the qualifiers attached. Change the "
                "records and let it re-render."
            )

        if expected_version is not None and expected_version != block.version:
            raise BlockVersionConflict(
                f"block {label!r} is at version {block.version}, not "
                f"{expected_version} — someone edited it since you read it. "
                "Re-read and reapply rather than overwriting their work."
            )

        if len(new_value) > block.limit:
            raise BlockOverflow(
                f"block {label!r} holds {block.limit} characters; this write "
                f"needs {len(new_value)}. Refusing rather than truncating — "
                "a silent trim would drop exactly what you decided to keep."
            )

        before = block.value
        block.value = new_value
        block.version += 1
        self._history.append(
            BlockEdit(
                label=label, operation=operation, before=before,
                after=new_value, author=author, reason=reason,
            )
        )
        return block

    def append(
        self,
        label: str,
        text: str,
        *,
        author: str,
        reason: str = "",
        expected_version: int | None = None,
        separator: str = "\n",
    ) -> MemoryBlock:
        """Add to the end of a block."""
        block = self.get(label)
        new_value = f"{block.value}{separator}{text}" if block.value else text
        return self._write(
            label, new_value, operation="append", author=author,
            reason=reason, expected_version=expected_version,
        )

    def replace(
        self,
        label: str,
        old: str,
        new: str,
        *,
        author: str,
        reason: str = "",
        expected_version: int | None = None,
    ) -> MemoryBlock:
        """Substitute text within a block.

        A miss is an error, not a no-op. The agent believes it corrected
        something; letting that belief stand uncorrected is how a stale fact
        survives being 'fixed'.
        """
        block = self.get(label)
        if old not in block.value:
            raise ValueError(
                f"{old!r} does not appear in block {label!r}; nothing was "
                "changed. Read the block and retry against its actual text."
            )
        return self._write(
            label, block.value.replace(old, new), operation="replace",
            author=author, reason=reason, expected_version=expected_version,
        )

    def rewrite(
        self,
        label: str,
        text: str,
        *,
        author: str,
        reason: str = "",
        expected_version: int | None = None,
    ) -> MemoryBlock:
        """Replace a block's entire contents.

        The operation consolidation needs: re-derive the whole block from what
        is now known, rather than accreting corrections onto a growing tail.
        """
        return self._write(
            label, text, operation="rewrite", author=author,
            reason=reason, expected_version=expected_version,
        )

    def clear(self, label: str, *, author: str, reason: str = "") -> MemoryBlock:
        return self._write(
            label, "", operation="clear", author=author,
            reason=reason, expected_version=None,
        )

    # -- history -----------------------------------------------------------

    def history(self, label: str | None = None) -> list[BlockEdit]:
        """Edits, oldest first, optionally for one block."""
        if label is None:
            return list(self._history)
        return [edit for edit in self._history if edit.label == label]

    def revert(self, label: str, *, author: str, reason: str = "") -> MemoryBlock:
        """Undo the most recent edit to a block.

        Recorded as its own edit rather than by popping history: an undo that
        erases the evidence of what it undid is indistinguishable from the edit
        never having happened.
        """
        edits = self.history(label)
        if not edits:
            raise ValueError(f"block {label!r} has no edits to revert")
        return self._write(
            label, edits[-1].before, operation="revert", author=author,
            reason=reason or f"revert of {edits[-1].operation}",
            expected_version=None,
        )

    # -- prompt surface ----------------------------------------------------

    def render(self, *, labels: Iterable[str] | None = None) -> str:
        """Every attached block, in label order, as prompt text."""
        wanted = sorted(labels) if labels is not None else self.labels
        return "\n\n".join(self.get(label).render() for label in wanted)

    @property
    def total_characters(self) -> int:
        return sum(len(block.value) for block in self)

    @property
    def total_limit(self) -> int:
        return sum(block.limit for block in self)

    def pressure(self) -> dict[str, float]:
        """Per-block utilization, so consolidation can target the full ones.

        Without this, the only signal that a block is nearly full is a write
        failing — which is to say, the signal arrives after the loss.
        """
        return {block.label: block.utilization for block in self}
