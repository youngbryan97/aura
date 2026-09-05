"""Transaction boundary for speculative recurrent K/V cache mutations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from core.brain.llm.recurrent_depth import (
    _cache_entry_matches_snapshot,
    _snapshot_recurrent_caches,
)

if TYPE_CHECKING:
    from core.brain.llm.latent_cortex.kv_state_tree import KVStateTree


def _tree_error(message: str) -> RuntimeError:
    from core.brain.llm.latent_cortex.kv_state_tree import KVStateTreeError

    return KVStateTreeError(message)


class KVMutationTransaction:
    """One speculative or isolated cache child."""

    def __init__(
        self,
        tree: KVStateTree,
        *,
        cache: Sequence[Any],
        start: int,
        end: int,
        purpose: str,
        branch_index: int | None,
        parent_sha256: str,
        isolated: bool,
    ) -> None:
        self._tree = tree
        self._cache = cache
        self.start = start
        self.end = end
        self.purpose = purpose
        self.branch_index = branch_index
        self.parent_sha256 = parent_sha256
        self.isolated = isolated
        self._child_snapshots: list | None = None
        self._child_cache_sha256 = ""
        self._child_offsets_sha256 = ""
        self._mutation_observed = False
        self._appended_min: int | None = None
        self._appended_max: int | None = None
        self._execution_failed = False
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def observe_mutation(
        self,
        cache: Sequence[Any] | None = None,
        *,
        execution_failed: bool = False,
    ) -> None:
        from core.brain.llm.latent_cortex.kv_state_tree import (
            _cache_offsets,
            _offsets_sha256,
            _snapshot_commitment,
        )

        if self._closed:
            raise _tree_error("cannot observe a closed KV transaction")
        if self._child_snapshots is not None:
            raise _tree_error("KV transaction mutation was observed twice")
        target = self._cache if cache is None else cache
        self._execution_failed = bool(execution_failed)
        self._child_snapshots = _snapshot_recurrent_caches(
            target,
            0,
            self._tree.n_layers,
        )
        self._child_cache_sha256 = _snapshot_commitment(
            self._child_snapshots,
            salt=self._tree._salt,
        )
        child_offsets = _cache_offsets(target, 0, self._tree.n_layers)
        parent_offsets = self._tree._node_offsets(self.parent_sha256)
        self._child_offsets_sha256 = _offsets_sha256(child_offsets)
        if any(
            (child is None) != (parent is None)
            for child, parent in zip(child_offsets, parent_offsets, strict=True)
        ):
            raise _tree_error("speculative cache cursor contract changed")
        deltas = [
            child - parent
            for child, parent in zip(child_offsets, parent_offsets, strict=True)
            if child is not None and parent is not None
        ]
        if any(delta < 0 for delta in deltas):
            raise _tree_error("speculative KV child moved before its parent")
        parent_snapshots = self._tree._node(self.parent_sha256).snapshots
        changed = [
            not _cache_entry_matches_snapshot(item, snapshot)
            for item, snapshot in zip(target, parent_snapshots, strict=True)
        ]
        outside = [
            index
            for index, differs in enumerate(changed)
            if differs and not self.start <= index < self.end
        ]
        if outside:
            raise _tree_error("speculative KV mutation escaped its declared layer window")
        window_deltas = [
            child - parent
            for child, parent in zip(
                child_offsets[self.start : self.end],
                parent_offsets[self.start : self.end],
                strict=True,
            )
            if child is not None and parent is not None
        ]
        positive = [delta for delta in window_deltas if delta > 0]
        self._mutation_observed = any(changed)
        self._appended_min = min(positive, default=0) if window_deltas else None
        self._appended_max = max(positive, default=0) if window_deltas else None

    def observe_and_restore(
        self,
        cache: Sequence[Any] | None = None,
        *,
        execution_failed: bool = False,
    ) -> None:
        """Restore speculative writes even when their audit fails."""
        if self.isolated:
            raise _tree_error("isolated KV transaction must be discarded explicitly")
        try:
            self.observe_mutation(cache, execution_failed=execution_failed)
        finally:
            self.restore_parent(cache)

    def reject_after_restore(self, cache: Sequence[Any] | None = None) -> dict[str, Any]:
        if self.isolated:
            raise _tree_error("isolated KV transaction must be discarded explicitly")
        target = self._cache if cache is None else cache
        return self._tree._close_rejected(self, parent_cache=target, isolated=False)

    def restore_parent(self, cache: Sequence[Any] | None = None) -> None:
        if self._closed:
            raise _tree_error("cannot restore a closed KV transaction")
        target = self._cache if cache is None else cache
        self._tree.restore_boundary(target, self.parent_sha256)

    def discard_isolated(self, *, parent_cache: Sequence[Any]) -> dict[str, Any]:
        if not self.isolated:
            raise _tree_error("non-isolated KV transaction must restore its cache")
        return self._tree._close_rejected(
            self,
            parent_cache=parent_cache,
            isolated=True,
        )

    def commit(
        self,
        *,
        label: str,
        authority: str,
        latent_sha256: str,
        final: bool,
    ) -> str:
        return self._tree._commit_transaction(
            self,
            label=label,
            authority=authority,
            latent_sha256=latent_sha256,
            final=final,
        )


__all__ = ["KVMutationTransaction"]
