"""The prompt KV cache: a prefix trie, an LRU with a byte budget, resume handles.

Lifted out of ``mlx_worker`` unchanged. A KV prefix cache with its own eviction
policy, memory envelope and continuation capabilities is a subsystem, and it
was living inside a twelve-thousand-line worker loop where the ratchet counted
every line of it against the worker's budget.

Nothing here knows about the worker. Trimming is passed in, because only the
caller knows which mlx entry points this build has.
"""

from __future__ import annotations

import copy
import logging
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("MLXWorker")

__all__ = [
    "PromptCacheEntry",
    "PromptCacheSearchResult",
    "ArraysCacheMemberRollback",
    "PromptCacheOneTokenRollback",
    "PromptCacheResumeBinding",
    "PromptCacheLRU",
    "capture_prompt_cache_one_token_rollback",
    "rewind_hybrid_prompt_cache_one_token",
]




@dataclass
class PromptCacheEntry:
    prompt_cache: list[Any]
    count: int


@dataclass
class PromptCacheSearchResult:
    exact: list[int] | None
    shorter: list[int] | None
    longer: list[int] | None
    common_prefix: int


@dataclass(frozen=True)
class ArraysCacheMemberRollback:
    index: int
    state: tuple[Any, ...]
    left_padding: Any
    lengths: Any

    @property
    def nbytes(self) -> int:
        total = 0
        for value in (*self.state, self.left_padding, self.lengths):
            if value is not None:
                total += max(0, int(getattr(value, "nbytes", 0) or 0))
        return total


@dataclass(frozen=True)
class PromptCacheOneTokenRollback:
    members: tuple[ArraysCacheMemberRollback, ...]

    @property
    def nbytes(self) -> int:
        return sum(member.nbytes for member in self.members)


def capture_prompt_cache_one_token_rollback(
    prompt_cache: list[Any] | None,
) -> PromptCacheOneTokenRollback | None:
    """Retain the fixed-size state needed to rewind a hybrid cache one token.

    ``mlx_lm`` can trim KV caches, but Qwen3.5 combines those with
    ``ArraysCache`` recurrent states that intentionally have no inverse. The
    recurrent arrays are replaced on each model call rather than mutated in
    place, so retaining their immediately preceding array references is an
    exact, fixed-size rollback image. Unknown non-trimmable cache kinds are
    refused instead of being guessed compatible.
    """

    if not prompt_cache:
        return None
    try:
        from mlx_lm.models.cache import ArraysCache
    except ImportError:
        return None

    members: list[ArraysCacheMemberRollback] = []
    for index, cache_member in enumerate(prompt_cache):
        try:
            if bool(cache_member.is_trimmable()):
                continue
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
        if not isinstance(cache_member, ArraysCache):
            return None
        state = cache_member.state
        if not isinstance(state, list):
            return None
        members.append(
            ArraysCacheMemberRollback(
                index=index,
                state=tuple(state),
                left_padding=getattr(cache_member, "left_padding", None),
                lengths=getattr(cache_member, "lengths", None),
            )
        )
    if not members:
        return None
    return PromptCacheOneTokenRollback(tuple(members))


def rewind_hybrid_prompt_cache_one_token(
    prompt_cache: list[Any],
    rollback: PromptCacheOneTokenRollback | None,
) -> tuple[bool, str]:
    """Rewind mixed KV/recurrent cache state without reconstructing the prompt."""

    if rollback is None:
        return False, "hybrid_rollback_unavailable"
    try:
        from mlx_lm.models.cache import ArraysCache
    except ImportError:
        return False, "mlx_cache_types_unavailable"

    snapshots = {member.index: member for member in rollback.members}
    nontrimmable_indexes: set[int] = set()
    trimmable_members: list[Any] = []
    for index, cache_member in enumerate(prompt_cache):
        try:
            trimmable = bool(cache_member.is_trimmable())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False, "cache_trim_contract_unavailable"
        if trimmable:
            try:
                if int(cache_member.size()) < 1:
                    return False, "trimmable_cache_empty"
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return False, "trimmable_cache_size_unavailable"
            trimmable_members.append(cache_member)
            continue
        nontrimmable_indexes.add(index)
        if not isinstance(cache_member, ArraysCache) or index not in snapshots:
            return False, "nontrimmable_cache_not_snapshotted"

    if nontrimmable_indexes != set(snapshots):
        return False, "hybrid_cache_layout_changed"

    # Validation above completes before the first mutation. From this point all
    # operations are the declared one-token inverse for their cache kind.
    for cache_member in trimmable_members:
        if int(cache_member.trim(1)) != 1:
            return False, "trimmable_cache_rewind_failed"
    for index, snapshot in snapshots.items():
        cache_member = prompt_cache[index]
        cache_member.state = list(snapshot.state)
        cache_member.left_padding = snapshot.left_padding
        cache_member.lengths = snapshot.lengths
    return True, ""


@dataclass(frozen=True)
class PromptCacheResumeBinding:
    model_key: Any
    tokens: tuple[int, ...]
    prompt_cache: list[Any]
    one_token_rollback: PromptCacheOneTokenRollback | None
    context_digest: str
    created_at: float


def _why_it_will_not_trim(prompt_cache: Any) -> str:
    """Which cache objects refuse, so a permanent refusal reads as permanent.

    `can_trim_prompt_cache` is `all(c.is_trimmable())`, and a model whose
    `make_cache` returns even one object that always answers False can never
    have a prompt cache trimmed — not this turn, not any turn. That is a
    different fact from a rotating cache that has wrapped, and only the class
    names separate them.

    The resident model is one of the first kind. `Qwen3_5` is a hybrid linear
    attention model and its `make_cache` returns `ArraysCache` for the linear
    layers, whose `is_trimmable` is a bare `return False`. So the exact-hit and
    trimmed-hit paths are dead for it, and only a stored key that is a strict
    PREFIX of the new prompt can ever be reused — which is why an append-only
    prompt matters so much more here than the length of one.
    """

    try:
        refusing = sorted(
            {
                type(entry).__name__
                for entry in (prompt_cache or [])
                if not bool(getattr(entry, "is_trimmable", lambda: True)())
            }
        )
    except (AttributeError, TypeError, ValueError):
        return "unreadable"
    if not refusing:
        return "no object named itself"
    return ", ".join(refusing)


class PromptCacheLRU:
    def __init__(
        self,
        max_size: int = 12,
        max_entry_tokens: int = 0,
        max_total_tokens: int = 0,
        kv_bytes_per_token: int = 0,
        fixed_bytes_per_entry: int = 0,
        max_total_bytes: int = 0,
    ):
        self.max_size = max_size
        # 0 = uncapped. A positive cap refuses to RETAIN prompts longer than
        # this many tokens, bounding per-entry KV RAM on heavy models while
        # leaving generation itself untouched.
        self.max_entry_tokens = max_entry_tokens
        # An entry COUNT is not a memory bound. Twelve entries of unbounded
        # length is unbounded memory: once insertion actually worked, a
        # 31,718-token prompt was measured live and managed RSS grew
        # 73,963MB/h toward the 49GB ceiling. This bounds the total.
        self.max_total_tokens = max_total_tokens
        self.kv_bytes_per_token = kv_bytes_per_token
        self.fixed_bytes_per_entry = max(0, int(fixed_bytes_per_entry))
        self.max_total_bytes = max(0, int(max_total_bytes))
        self._cache: dict[Any, dict[Any, Any]] = {}
        # One eviction queue PER LANE, not one globally. A single global queue
        # meant Aura's internal lanes (loop ticks, enrichment, dreaming,
        # health probes — dozens of generations per minute) evicted the user
        # conversation's entry within seconds of it being written, so the one
        # entry whose reuse decides whether a conversation survives was always
        # the first one thrown away. Lane budgets still sum to max_size, so
        # per-entry KV RAM is bounded exactly as before.
        self._lru: dict[str, deque] = {}
        # A continuation capability names exact worker-owned KV state without
        # copying that state through IPC or reconstructing it from visible text.
        self._resume_bindings: dict[str, PromptCacheResumeBinding] = {}
        self._resume_ttl_s = 300.0
        self._resume_binding_limit = max(2, min(8, max_size))

    # ── introspection: what is actually retained right now ───────────────
    def retained_tokens(self) -> int:
        """Total cached tokens across every lane. The real memory driver."""
        reusable = sum(
            len(tokens) for queue in self._lru.values() for (_model_key, tokens) in queue
        )
        resumable = sum(len(binding.tokens) for binding in self._resume_bindings.values())
        return reusable + resumable

    def retained_entries(self) -> int:
        return sum(len(queue) for queue in self._lru.values()) + len(self._resume_bindings)

    def retained_bytes(self) -> int:
        """Approximate KV bytes held. Reported to the OOM ladder, not guessed."""
        token_bytes = (
            self.retained_tokens() * self.kv_bytes_per_token if self.kv_bytes_per_token > 0 else 0
        )
        rollback_bytes = sum(
            binding.one_token_rollback.nbytes
            for binding in self._resume_bindings.values()
            if binding.one_token_rollback is not None
        )
        fixed_bytes = self.retained_entries() * self.fixed_bytes_per_entry
        return token_bytes + fixed_bytes + rollback_bytes

    def shed(self) -> int:
        """Release everything and report the bytes freed.

        This is the OOM ladder's rung. The ladder had none — the verifier said
        so on every boot ("no organ exposes a shed hook, so the OOM ladder has
        no rungs: the only available response to memory pressure is a
        restart") — while this cache was the largest trivially-droppable
        allocation in the process.
        """
        freed = self.retained_bytes()
        self.clear()
        return freed

    def _enforce_total_token_budget(self) -> None:
        """Evict oldest entries until total retained tokens fit the budget.

        Per-lane entry budgets bound how many prefixes stay reusable; this
        bounds the MEMORY. The user-surface lane is drained last so a
        conversation keeps its prefix while internal lanes give theirs up.
        """
        if self.max_total_tokens <= 0:
            return
        lanes_by_drain_order = sorted(
            self._lru.keys(), key=lambda lane: (lane == "user_surface", lane)
        )
        byte_budget = self.max_total_bytes
        if byte_budget <= 0 and self.max_total_tokens > 0 and self.kv_bytes_per_token > 0:
            byte_budget = self.max_total_tokens * self.kv_bytes_per_token
        while self.retained_tokens() > self.max_total_tokens or (
            byte_budget > 0 and self.retained_bytes() > byte_budget
        ):
            evicted = False
            for lane in lanes_by_drain_order:
                queue = self._lru.get(lane)
                if queue:
                    evict_model_key, evict_tokens = queue.popleft()
                    self._delete(evict_model_key, list(evict_tokens))
                    evicted = True
                    break
            if evicted:
                continue
            oldest_resume = next(iter(self._resume_bindings), None)
            if oldest_resume is not None:
                self._resume_bindings.pop(oldest_resume, None)
            else:
                return

    def _lane_of(self, model_key: Any) -> str:
        # A single-entry budget cannot be split without overspending it, so
        # every lane shares one queue and eviction stays global.
        if self.max_size <= 1:
            return "shared"
        if isinstance(model_key, tuple) and len(model_key) >= 2:
            return str(model_key[1])
        return "default"

    def _lane_budget(self, lane: str) -> int:
        if self.max_size <= 1 or lane == "shared":
            return self.max_size
        # Asymmetric on purpose. The conversation is already protected by having
        # its OWN queue, and only its newest entry is ever reused — turn N+1
        # extends turn N, so older conversation entries are dead weight holding
        # the largest KV in the cache. The internal lane is the opposite: it
        # carries many DISTINCT prompt families (the reflective persona, the
        # pre-linguistic decision narrator, enrichment, dreaming), and with a
        # 50/50 split they evicted each other on every tick. Measured live,
        # repeatedly: "trimmed hit — reused 3/792 tokens", the same two families
        # taking turns destroying each other's prefix.
        reserved = max(1, min(3, self.max_size - 1))
        return reserved if lane == "user_surface" else self.max_size - reserved

    def _queue_for(self, lane: str) -> deque:
        queue_for_lane = self._lru.get(lane)
        if queue_for_lane is None:
            queue_for_lane = deque()
            self._lru[lane] = queue_for_lane
        return queue_for_lane

    def _forget_key(self, cache_key: tuple) -> None:
        queue_for_lane = self._lru.get(self._lane_of(cache_key[0]))
        if queue_for_lane is None:
            return
        try:
            queue_for_lane.remove(cache_key)
        except ValueError as exc:
            logger.debug("Prompt cache LRU entry already absent: %s", exc)

    def clear(self) -> None:
        self._cache.clear()
        self._lru.clear()
        self._resume_bindings.clear()

    def clear_model_key(self, model_key: Any) -> None:  # noqa: D401 - see log
        logger.info("🧊 [PROMPT CACHE] cleared everything under key=%s", model_key)
        return self._clear_model_key(model_key)

    def _clear_model_key(self, model_key: Any) -> None:
        """Discard one model/scope without erasing unrelated prompt state.

        Generation retries need a clean cache for the request that failed. They
        do not establish that every other lane is corrupt. In particular, a
        default-lane repair must not erase the user-surface prefix that keeps a
        live conversation warm. Weight changes and memory-pressure shedding
        still use ``clear()`` because those events invalidate every entry.
        """

        lane = self._lane_of(model_key)
        queue_for_lane = self._lru.get(lane)
        if queue_for_lane is not None:
            retained: deque = deque()
            for cache_key in list(queue_for_lane):
                cached_model_key, cached_tokens = cache_key
                if cached_model_key == model_key:
                    self._delete(cached_model_key, list(cached_tokens))
                else:
                    retained.append(cache_key)
            if retained:
                self._lru[lane] = retained
            else:
                self._lru.pop(lane, None)
        for handle, binding in list(self._resume_bindings.items()):
            if binding.model_key == model_key:
                self._resume_bindings.pop(handle, None)

    def _prune_resume_bindings(self, *, now: float | None = None) -> None:
        observed_at = time.monotonic() if now is None else float(now)
        expired = [
            handle
            for handle, binding in self._resume_bindings.items()
            if observed_at - binding.created_at > self._resume_ttl_s
        ]
        for handle in expired:
            self._resume_bindings.pop(handle, None)
        while len(self._resume_bindings) > self._resume_binding_limit:
            oldest = next(iter(self._resume_bindings), None)
            if oldest is None:
                break
            self._resume_bindings.pop(oldest, None)

    def bind_resume(
        self,
        model_key: Any,
        tokens: list[int],
        *,
        prompt_cache: list[Any] | None = None,
        one_token_rollback: PromptCacheOneTokenRollback | None = None,
        context_digest: str = "",
    ) -> str:
        """Move exact final KV state into a short-lived continuation capability.

        A resumable generation is a transaction boundary, not an ordinary cache
        hint.  The capability therefore owns the actual final cache object.  A
        later trie lookup allowed eviction or insertion-policy differences to
        turn a valid continuation into a silent full re-prefill.
        """

        if len(tokens) < 2:
            return ""
        if self.max_entry_tokens > 0 and len(tokens) > self.max_entry_tokens:
            return ""
        exact = self._search(model_key, tokens).exact
        if exact is not None:
            owned_cache = self._extract(model_key, exact).prompt_cache
        elif prompt_cache is not None:
            owned_cache = prompt_cache
        else:
            return ""
        self._prune_resume_bindings()
        handle = uuid.uuid4().hex
        self._resume_bindings[handle] = PromptCacheResumeBinding(
            model_key=model_key,
            tokens=tuple(int(token) for token in tokens),
            prompt_cache=owned_cache,
            one_token_rollback=one_token_rollback,
            context_digest=str(context_digest or ""),
            created_at=time.monotonic(),
        )
        self._prune_resume_bindings()
        self._enforce_total_token_budget()
        return handle if handle in self._resume_bindings else ""

    def fetch_resume(
        self,
        handle: str,
        model_key: Any,
        *,
        can_trim_prompt_cache: Any,
        trim_prompt_cache: Any,
        append_tokens: list[int] | None = None,
        context_digest: str = "",
    ) -> tuple[list[Any] | None, list[int], list[int], str]:
        """Consume exact worker state, replay its last token, then append a turn."""

        normalized = str(handle or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{32}", normalized):
            return None, [], [], "invalid_handle"
        self._prune_resume_bindings()
        binding = self._resume_bindings.pop(normalized, None)
        if binding is None:
            return None, [], [], "unknown_or_expired_handle"
        if binding.model_key != model_key:
            # A capability presented on the wrong lane must not disclose KV,
            # but it also must not let another lane destroy the rightful
            # continuation. Return ownership to the scoped general cache.
            self.insert_cache(
                binding.model_key,
                list(binding.tokens),
                binding.prompt_cache,
            )
            return None, [], [], "model_or_lane_mismatch"
        expected_digest = str(binding.context_digest or "")
        presented_digest = str(context_digest or "")
        if expected_digest and presented_digest != expected_digest:
            self.insert_cache(
                binding.model_key,
                list(binding.tokens),
                binding.prompt_cache,
            )
            return None, [], [], "context_mismatch"
        tokens = list(binding.tokens)
        prompt_cache = binding.prompt_cache
        if not can_trim_prompt_cache(prompt_cache):
            rewound, rewind_failure = rewind_hybrid_prompt_cache_one_token(
                prompt_cache,
                binding.one_token_rollback,
            )
            if not rewound:
                self.insert_cache(model_key, tokens, prompt_cache)
                return None, [], [], rewind_failure
            resume_kind = "hybrid recurrent/KV"
        else:
            trim_prompt_cache(prompt_cache, 1)
            resume_kind = "KV"
        appended = [int(token) for token in (append_tokens or [])]
        if append_tokens is not None:
            if not appended:
                self.insert_cache(model_key, tokens, prompt_cache)
                return None, [], [], "append_boundary_empty"
            if appended[0] != tokens[-1]:
                self.insert_cache(model_key, tokens, prompt_cache)
                return None, [], [], "append_boundary_final_token_mismatch"
            # The boundary renderer includes the assistant end token so it can
            # prove it is the same token the cache ended on. ``tokens`` already
            # owns it; after the one-token rewind it is replayed exactly once.
            appended = appended[1:]
        logical_tokens = [*tokens, *appended]
        replay_tokens = [tokens[-1], *appended]
        logger.info(
            "🎯 [PROMPT CACHE] %s exact resume — reused %d/%d tokens, %d to prefill.",
            resume_kind,
            len(tokens) - 1,
            len(logical_tokens),
            len(replay_tokens),
        )
        return prompt_cache, replay_tokens, logical_tokens, ""

    def _search(self, model_key: Any, tokens: list[int]) -> PromptCacheSearchResult:
        if model_key not in self._cache:
            return PromptCacheSearchResult(None, None, None, 0)

        current = self._cache[model_key]
        last_cache_index = -1
        index = 0

        while index < len(tokens) and tokens[index] in current:
            current = current[tokens[index]]
            if "cache" in current:
                last_cache_index = index
            index += 1

        if last_cache_index == len(tokens) - 1:
            return PromptCacheSearchResult(tokens, None, None, 0)

        # Index 0 is a valid one-token cached prefix; the old > 0 test threw
        # it away and forced an avoidable full prefill.
        shorter = tokens[: last_cache_index + 1] if last_cache_index >= 0 else None
        longer = None
        common_prefix = index
        if index > 0 and last_cache_index < 0:
            best = None
            stack = [(current, [])]
            while stack:
                node, extra = stack.pop()
                if "cache" in node:
                    if best is None or len(extra) < len(best):
                        best = extra
                else:
                    for tok in node:
                        stack.append((node[tok], extra + [tok]))
            if best is not None:
                longer = tokens[:index] + best

        return PromptCacheSearchResult(None, shorter, longer, common_prefix)

    def _get(self, model_key: int, tokens: list[int]) -> PromptCacheEntry:
        current = self._cache[model_key]
        for tok in tokens:
            current = current[tok]
        return current["cache"]

    def _delete(self, model_key: int, tokens: list[int]) -> None:
        path = [self._cache[model_key]]
        for tok in tokens:
            path.append(path[-1][tok])
        del path[-1]["cache"]
        for index in reversed(range(len(tokens))):
            prev_node, node, tok = path[index], path[index + 1], tokens[index]
            if len(node) > 0:
                break
            del prev_node[tok]

    def _extract(self, model_key: int, tokens: list[int]) -> PromptCacheEntry:
        cache_entry = self._get(model_key, tokens)
        if cache_entry.count == 1:
            self._delete(model_key, tokens)
            self._forget_key((model_key, tuple(tokens)))
            return cache_entry

        cache_entry.count -= 1
        return PromptCacheEntry(copy.deepcopy(cache_entry.prompt_cache), 1)

    def fetch_nearest_cache(
        self,
        model_key: Any,
        tokens: list[int],
        *,
        can_trim_prompt_cache: Any,
        trim_prompt_cache: Any,
        describe: Any = None,
    ) -> tuple[list[Any] | None, list[int]]:
        result = self._search(model_key, tokens)
        # Whether prefix reuse actually happens decides whether a conversation
        # survives: every turn that misses re-prefills the entire history, and
        # time-to-first-token climbs until it crosses the turn budget. Measured
        # live 2026-07-26, turns 5-7 of one conversation died that way with the
        # budget shrinking 81.1s -> 73.2s -> 55.8s against a first token that
        # never arrived in under 58s. None of that was visible: there was no hit/miss
        # signal anywhere, so reuse could only be inferred from latency.
        if result.exact is not None and len(tokens) > 1:
            # Never hand back an EMPTY remainder: mlx_lm has to be given at
            # least one token to run a decode step, so an exact hit reuses
            # everything but the final token and replays that one.
            cache_entry = self._extract(model_key, result.exact)
            if can_trim_prompt_cache(cache_entry.prompt_cache):
                trim_prompt_cache(cache_entry.prompt_cache, 1)
                logger.info(
                    "🎯 [PROMPT CACHE] exact hit — reused %d/%d tokens, 1 to prefill.",
                    len(tokens) - 1,
                    len(tokens),
                )
                return cache_entry.prompt_cache, tokens[-1:]
            # Untrimmable cache: putting it back keeps it available for the
            # prefix path instead of silently dropping a live entry.
            self.insert_cache(model_key, list(result.exact), cache_entry.prompt_cache)

        if result.shorter is not None:
            cache_entry = self._extract(model_key, result.shorter)
            prefix_len = len(result.shorter)
            logger.info(
                "🎯 [PROMPT CACHE] prefix hit — reused %d/%d tokens, %d to prefill.",
                prefix_len,
                len(tokens),
                len(tokens) - prefix_len,
            )
            return cache_entry.prompt_cache, tokens[prefix_len:]

        refused = ""
        if result.longer is not None:
            cache_entry = self._get(model_key, result.longer)
            if not can_trim_prompt_cache(cache_entry.prompt_cache):
                # A refusal, not an absence.
                #
                # The miss line below reports how far the prompt matched, and
                # a reader seeing "matched 5202 (54.7%)" concludes the trie
                # found nothing usable. What actually happened is that it
                # found the entry, and mlx_lm declined to trim it — so five
                # thousand tokens of reusable prefix were re-read with nothing
                # anywhere saying why.
                #
                # LIVE, 2026-09-07: every turn of a five-step tool loop, each
                # matching more than the last (17%, 35%, 47%, 55%), each
                # prefilling in full. The loop then exhausted its 178.8s turn
                # budget and the person got an unfinished answer.
                refused = (
                    "; the entry holding them refuses to trim ("
                    + _why_it_will_not_trim(cache_entry.prompt_cache)
                    + ")"
                )
            if can_trim_prompt_cache(cache_entry.prompt_cache):
                prefix = min(len(tokens) - 1, result.common_prefix)
                num_to_trim = len(result.longer) - prefix
                # _extract already copies when the entry is shared and hands
                # over the live object when it is not. Deep-copying here
                # unconditionally cost a second full KV allocation — ~1.5GB on
                # the 32B geometry — on the hot path of every diverging turn.
                trimmed = self._extract(model_key, result.longer)
                trim_prompt_cache(trimmed.prompt_cache, num_to_trim)
                logger.info(
                    "🎯 [PROMPT CACHE] trimmed hit — reused %d/%d tokens, %d to prefill.",
                    prefix,
                    len(tokens),
                    len(tokens) - prefix,
                )
                return trimmed.prompt_cache, tokens[prefix:]

        # A total miss was the one case with no diagnosis: the partial-hit
        # path named the divergent block, and the case that costs a FULL
        # prefill said only that it had happened. The trie walk already knows
        # how far the prompt matched before it left the tree, and that number
        # is the whole answer — a handful of tokens means something volatile
        # sits at the front of the prompt and no conversation will ever reuse
        # anything.
        matched = result.common_prefix
        divergent = ""
        if describe is not None and matched < len(tokens):
            try:
                divergent = describe(tokens[matched : matched + 24])
            except (AttributeError, RuntimeError, TypeError, ValueError):
                divergent = "<undecodable>"
        logger.info(
            "🧊 [PROMPT CACHE] miss — prefilling all %d tokens; key=%s known_keys=%d "
            "matched %d (%.1f%%) before diverging%s",
            len(tokens),
            model_key,
            len(self._cache),
            matched,
            100.0 * matched / max(1, len(tokens)),
            (f"; divergent text begins: {divergent[:160]!r}" if divergent else "")
            + refused,
        )
        return None, tokens

    def insert_cache(self, model_key: Any, tokens: list[int], prompt_cache: list[Any]) -> None:
        if self.max_entry_tokens > 0 and len(tokens) > self.max_entry_tokens:
            # Silent before: a prompt too long to retain made every later turn
            # re-prefill from zero, and nothing anywhere said so.
            logger.info(
                "🧊 [PROMPT CACHE] not retained: %d tokens over the %d-token "
                "per-entry cap.", len(tokens), self.max_entry_tokens,
            )
            return
        if model_key not in self._cache:
            self._cache[model_key] = {}
        current = self._cache[model_key]
        for tok in tokens:
            if tok not in current:
                current[tok] = {}
            current = current[tok]

        cache_key = (model_key, tuple(tokens))
        if "cache" in current:
            current["cache"].count += 1
            self._forget_key(cache_key)
        else:
            current["cache"] = PromptCacheEntry(prompt_cache, 1)

        lane = self._lane_of(model_key)
        queue_for_lane = self._queue_for(lane)
        queue_for_lane.append(cache_key)
        while len(queue_for_lane) > self._lane_budget(lane):
            evict_model_key, evict_tokens = queue_for_lane.popleft()
            # An eviction nobody can see is a cache that looks like it works.
            logger.info(
                "🧊 [PROMPT CACHE] evicted %d tokens from lane %s (budget %d)",
                len(evict_tokens), lane, self._lane_budget(lane),
            )
            self._delete(evict_model_key, list(evict_tokens))
        before = self.retained_tokens()
        self._enforce_total_token_budget()
        after = self.retained_tokens()
        if after < before:
            logger.info(
                "🧊 [PROMPT CACHE] budget drained %d tokens (%d -> %d, cap %d)",
                before - after, before, after, self.max_total_tokens,
            )


