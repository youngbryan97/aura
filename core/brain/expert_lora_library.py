"""Expert-LoRA library — the capacity loophole.

A 32B's weights hold a bounded amount of information (a real limit). But total
*reachable* expertise is not bounded by RAM — only *resident* expertise is. So we
keep a library of domain-specialist LoRA adapters on disk (you have ~813 GB free),
select the right one per task, and load only a few into memory at a time. Effective
capacity becomes ``base_model + on-disk adapter library``, accessed on demand.

This pairs with the self-improvement flywheel: promoted adapters (verifier-clean
reasoning distilled into a domain LoRA) register here, and the library serves them
back per task. The organism's expertise grows on disk without growing its RAM.

Scope (honest): this module owns the *registry, selection, and RAM-budgeted
residency* — the real, testable substrate. The actual MLX attach/detach of adapter
weights is delegated to a pluggable ``AdapterApplier`` so it can be wired to the live
worker later without this module ever touching model state itself. Default-off
(``AURA_EXPERT_LORA_LIBRARY``) so it never changes generation behavior implicitly.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.ExpertLoRALibrary")

_WORD_RE = re.compile(r"[a-z0-9]+")

#: Adapter artifacts must carry at least one of these to be loadable at all.
_ADAPTER_MARKERS = ("adapter_config.json", "adapters.safetensors", "adapters.npz")
#: Default resident memory budget (MB). The module always advertised a
#: "RAM-budgeted" residency; this is that budget (CP126 bddade3a).
_DEFAULT_MAX_RESIDENT_MB = 4096.0


def _flag_on(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "on", "yes", "enabled"}


def _budget_mb_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return float(default)
    if value != value or value in (float("inf"), float("-inf")) or value <= 0.0:
        return float(default)
    return value


def _tag_set(value: Any) -> set[str]:
    """Normalized tag set from untrusted data — never a set of characters."""
    if value is None:
        return set()
    if isinstance(value, str):
        # A bare string is ONE tag, not a sequence of letters.
        tag = value.strip().lower()
        return {tag} if tag else set()
    if isinstance(value, (list, tuple, set, frozenset)):
        return {
            str(item).strip().lower()
            for item in value
            if str(item).strip()
        }
    return set()


def _finite(value: Any, default: float, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if number != number or number in (float("inf"), float("-inf")):
        return float(default)
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def _applier_attaches_weights(applier: Any) -> bool:
    """Does this applier actually attach weights?

    CP126 20a12402: a no-op applier returning True made the library attest
    physical residency it had not produced. Appliers opt OUT by declaring
    ``attaches_weights = False``; anything that does not declare is assumed
    real (the live worker seams are).
    """
    return bool(getattr(applier, "attaches_weights", True))


def _attest_adapter_artifact(adapter: LoRAAdapter) -> tuple[bool, str]:
    """Prove an adapter artifact exists and looks like an adapter.

    CP126 70c50967: registration accepted any nonempty name/path, so the
    library could hand an applier a path that does not exist, is not an
    adapter, or escapes its directory. This is the minimum honest bar before
    anything may load it — a full signature/base-fingerprint chain is a
    separate promotion-receipt build.
    """
    raw = str(adapter.path or "").strip()
    if not raw:
        return False, "empty_path"
    try:
        path = Path(os.path.expanduser(raw)).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return False, "unresolvable_path"
    if not path.exists():
        return False, "artifact_missing"
    if path.is_dir():
        if not any((path / marker).exists() for marker in _ADAPTER_MARKERS):
            return False, "no_adapter_markers"
        return True, ""
    if path.is_file():
        if path.suffix.lower() not in {".safetensors", ".npz", ".json"}:
            return False, "unsupported_artifact_type"
        return True, ""
    return False, "not_a_regular_artifact"


@dataclass
class _Residency:
    """Who holds this adapter's weights, and what it costs."""

    applier_id: str
    applier: Any = None
    size_mb: float = 0.0
    last_used: float = field(default_factory=time.time)
    evicting: bool = False


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(str(text or "").lower()))


@dataclass
class LoRAAdapter:
    name: str
    path: str
    base_model: str = ""
    task_types: set[str] = field(default_factory=set)
    keywords: set[str] = field(default_factory=set)
    size_mb: float = 0.0
    quality: float = 0.5     # promotion score; higher wins ties
    source: str = ""         # e.g. "self_improvement", "manual"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "base_model": self.base_model,
            "task_types": sorted(self.task_types),
            "keywords": sorted(self.keywords),
            "size_mb": round(float(self.size_mb), 2),
            "quality": round(float(self.quality), 4),
            "source": self.source,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoRAAdapter:
        """Build an adapter from untrusted manifest data.

        CP126 079e8448: deserialization validated nothing — a STRING supplied
        for task_types/keywords became a set of single CHARACTERS, and
        NaN/Infinity could enter ranking and JSON output.
        """
        if not isinstance(data, dict):
            raise ValueError("adapter record is not a mapping")
        return cls(
            name=str(data.get("name", "")).strip(),
            path=str(data.get("path", "")).strip(),
            base_model=str(data.get("base_model", "")).strip(),
            task_types=_tag_set(data.get("task_types")),
            keywords=_tag_set(data.get("keywords")),
            size_mb=_finite(data.get("size_mb"), 0.0, minimum=0.0),
            quality=_finite(data.get("quality"), 0.5, minimum=0.0, maximum=1.0),
            source=str(data.get("source", "")).strip()[:64],
            created_at=_finite(data.get("created_at"), time.time(), minimum=0.0),
        )

    def copy(self) -> LoRAAdapter:
        """A detached snapshot.

        CP126 37e18573: registry reads handed out the INTERNAL instances, so a
        caller could mutate name, path, tags, quality, or base identity with no
        lock, persistence, duplicate check, or residency reconciliation.
        """
        return LoRAAdapter(
            name=self.name,
            path=self.path,
            base_model=self.base_model,
            task_types=set(self.task_types),
            keywords=set(self.keywords),
            size_mb=self.size_mb,
            quality=self.quality,
            source=self.source,
            created_at=self.created_at,
        )


@runtime_checkable
class AdapterApplier(Protocol):
    """Attaches/detaches adapter weights to/from the live model. MLX-specific."""

    def load(self, adapter: LoRAAdapter) -> bool: ...
    def unload(self, adapter: LoRAAdapter) -> bool: ...


class NoopApplier:
    """Safe default — attaches nothing, and SAYS so.

    CP126 20a12402: this returned True from both methods, so with the default
    applier ``activate`` recorded the adapter as resident and
    ``select_and_activate`` returned it as activated — an attestation that
    specialist weights were physically loaded when nothing had touched model
    state at all. A no-op reports failure to load, because it did not load.
    ``unload`` returns True because "no weights attached" IS the post-unload
    state it promises.
    """

    #: Marks an applier that cannot make weights resident. The library refuses
    #: to record residency for these instead of trusting a bare boolean.
    attaches_weights = False

    def load(self, adapter: LoRAAdapter) -> bool:
        logger.info(
            "📎 [ExpertLoRA] (noop) NOT loading adapter '%s' from %s — no applier is wired",
            adapter.name,
            adapter.path,
        )
        return False

    def unload(self, adapter: LoRAAdapter) -> bool:
        logger.info("📎 [ExpertLoRA] (noop) adapter '%s' was never attached", adapter.name)
        return True


@runtime_checkable
class AsyncAdapterApplier(Protocol):
    """Async applier for live seams (worker IPC) that must never block a loop."""

    async def load(self, adapter: LoRAAdapter) -> bool: ...
    async def unload(self, adapter: LoRAAdapter) -> bool: ...


class ExpertLoRALibrary:
    """Registry + per-task selection + RAM-budgeted residency for domain LoRAs."""

    _ERRORS = (OSError, ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError)

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        *,
        max_resident: int | None = None,
        applier: AdapterApplier | None = None,
    ) -> None:
        self._manifest_path = Path(
            manifest_path or str(state_root() / "data/adapters/library.json")
        )
        self._max_resident = int(max_resident if max_resident is not None
                                 else os.getenv("AURA_EXPERT_LORA_MAX_RESIDENT", "2"))
        self._max_resident = max(1, self._max_resident)
        # CP126 bddade3a: the module advertises RAM-BUDGETED residency and
        # records size_mb, but admission only ever compared len(_resident)
        # against max_resident — two 4GB adapters cost the same as two 20MB
        # ones. The count stays as a hard ceiling; this is the actual memory
        # budget that admission and eviction now respect.
        self._max_resident_mb = _budget_mb_env(
            "AURA_EXPERT_LORA_MAX_RESIDENT_MB", _DEFAULT_MAX_RESIDENT_MB
        )
        self._applier = applier or NoopApplier()
        self._lock = threading.RLock()
        self._adapters: dict[str, LoRAAdapter] = {}
        # name -> residency record. CP126 58bc5a8d: a bare name set could not
        # say WHICH applier holds the weights, so a second worker saw the name
        # and returned True without loading anything there, and evictions were
        # sent through whichever applier happened to be calling.
        self._resident: OrderedDict[str, _Residency] = OrderedDict()
        # In-flight async activations holding a reserved slot (CP126 be41d7a1).
        self._pending: dict[str, str] = {}
        self._load_manifest()

    # ── registry ────────────────────────────────────────────────────────────
    def register(self, adapter: LoRAAdapter) -> bool:
        if not adapter.name or not adapter.path:
            return False
        # CP126 70c50967: registration required only a nonempty name and path,
        # so an applier could be asked to load a path that does not exist, is
        # not an adapter, or was never promoted. Prove the artifact first.
        attested, reason = _attest_adapter_artifact(adapter)
        if not attested:
            record_degradation(
                "expert_lora_register",
                ValueError(f"adapter_artifact_unattested:{adapter.name}:{reason}"),
                action="refused to register an adapter whose artifact could not be attested",
                severity="warning",
            )
            logger.warning(
                "📚 [ExpertLoRA] refused '%s': %s (%s)", adapter.name, reason, adapter.path
            )
            return False
        with self._lock:
            existing = self._adapters.get(adapter.name)
            if existing is not None and adapter.name in self._resident:
                # CP126 c6f1e993: silently replacing a RESIDENT adapter left
                # the residency map naming the old loaded weights while later
                # eviction resolved the replacement object and asked the
                # applier to unload the wrong path. Retire the live one first.
                if existing.path != adapter.path:
                    logger.warning(
                        "📚 [ExpertLoRA] '%s' is resident and its artifact changed — "
                        "evicting the loaded adapter before re-registration.",
                        adapter.name,
                    )
                    if not self._evict(adapter.name):
                        record_degradation(
                            "expert_lora_register",
                            RuntimeError(f"resident_adapter_unload_failed:{adapter.name}"),
                            action="refused re-registration while the previous adapter is still attached",
                            severity="error",
                        )
                        return False
            # Store a detached copy so a caller keeping its reference cannot
            # mutate registry state behind the lock (CP126 37e18573).
            self._adapters[adapter.name] = adapter.copy()
            # CP126 d09b9a74: _persist swallowed every configured error and
            # returned nothing, so register/unregister reported success after
            # an in-memory-only mutation. Registration is durable or it failed.
            if not self._persist():
                self._adapters.pop(adapter.name, None)
                if existing is not None:
                    self._adapters[adapter.name] = existing
                record_degradation(
                    "expert_lora_persist",
                    RuntimeError(f"adapter_registration_not_durable:{adapter.name}"),
                    action="rolled back an adapter registration that could not be persisted",
                    severity="error",
                )
                return False
        logger.info("📚 [ExpertLoRA] registered '%s' (task_types=%s)", adapter.name, sorted(adapter.task_types))
        return True

    def unregister(self, name: str) -> bool:
        with self._lock:
            # CP126 4637a168: this popped the adapter FIRST and then called
            # _evict, which could no longer resolve the adapter object and so
            # skipped unload entirely — leaving weights attached permanently.
            # Unload while the adapter is still resolvable.
            if name in self._resident and not self._evict(name):
                record_degradation(
                    "expert_lora_unregister",
                    RuntimeError(f"resident_adapter_unload_failed:{name}"),
                    action="kept adapter registered because its weights are still attached",
                    severity="error",
                )
                return False
            existed = self._adapters.pop(name, None) is not None
            if existed:
                self._persist()
            return existed

    def list(self) -> list[LoRAAdapter]:
        # CP126 37e18573: hand out snapshots, not the live registry objects.
        with self._lock:
            return [adapter.copy() for adapter in self._adapters.values()]

    def get(self, name: str) -> LoRAAdapter | None:
        with self._lock:
            adapter = self._adapters.get(name)
            return adapter.copy() if adapter is not None else None

    # ── selection ─────────────────────────────────────────────────────────────
    def select_for(self, objective: str, task_type: str, *, base_model: str = "") -> LoRAAdapter | None:
        """Best adapter for a task: task_type must match; rank by keyword overlap × quality."""
        tt = str(task_type or "").strip().lower()
        obj_tokens = _tokens(objective)
        best: tuple[float, LoRAAdapter] | None = None
        with self._lock:
            for adapter in self._adapters.values():
                if base_model:
                    # CP126 a0cf1594: an adapter with an UNKNOWN base bypassed
                    # the mismatch check entirely, contradicting the stated
                    # "never apply an adapter trained on a different base".
                    # Unknown is not compatible — it is unverified.
                    if not adapter.base_model or adapter.base_model != base_model:
                        continue
                # CP126 9abdd2bf: this skipped only when task_types was
                # NONEMPTY, while scan() registers every discovered adapter
                # with an EMPTY set — so untagged adapters scored positively
                # for every objective and could be selected on nothing but
                # quality and insertion order.
                if not adapter.task_types or tt not in adapter.task_types:
                    continue
                overlap = len(obj_tokens & adapter.keywords) if adapter.keywords else 0
                # task_type match alone is worth a small base score so a tagged
                # specialist still wins over nothing even with no keyword overlap.
                relevance = (1.0 + overlap) * max(0.05, adapter.quality)
                if best is None or relevance > best[0]:
                    best = (relevance, adapter)
        return best[1].copy() if best else None

    # ── RAM-budgeted residency ──────────────────────────────────────────────
    def _applier_id(self, applier: Any) -> str:
        return f"{type(applier).__name__}:{id(applier):x}"

    def _resident_mb(self) -> float:
        # Caller holds lock.
        return sum(entry.size_mb for entry in self._resident.values()) + sum(
            float(self._adapters[n].size_mb) if n in self._adapters else 0.0
            for n in self._pending
        )

    def _needs_eviction(self, adapter: LoRAAdapter) -> bool:
        # Caller holds lock.
        if len(self._resident) + len(self._pending) >= self._max_resident:
            return True
        return (
            self._resident_mb() + float(adapter.size_mb or 0.0)
        ) > self._max_resident_mb

    def activate(self, name: str) -> bool:
        """Make an adapter resident (load on demand, LRU-evict over budget).

        CP126 0889dd98: this used to hold the registry RLock across every
        unload and load, so a multi-second worker swap blocked list, get,
        selection, registration, and unrelated activations for its full
        duration. Applier I/O now runs OUTSIDE the lock behind a reservation,
        the same discipline as ``activate_async``.
        """
        applier = self._applier
        if not _applier_attaches_weights(applier):
            # CP126 20a12402: never record residency behind an applier that
            # does not attach weights.
            logger.info(
                "📎 [ExpertLoRA] '%s' not activated: no weight-attaching applier is wired.",
                name,
            )
            return False
        applier_id = self._applier_id(applier)
        evictees: list[tuple[str, LoRAAdapter]] = []
        with self._lock:
            adapter = self._adapters.get(name)
            if adapter is None:
                return False
            entry = self._resident.get(name)
            if entry is not None:
                self._resident.move_to_end(name)
                entry.last_used = time.time()
                return True
            if name in self._pending:
                return False
            # Evict until BOTH the count ceiling and the memory budget admit
            # this adapter (CP126 bddade3a).
            while self._needs_eviction(adapter):
                evictable = [n for n, e in self._resident.items() if not e.evicting]
                if not evictable:
                    return False
                lru_name = evictable[0]
                lru_entry = self._resident[lru_name]
                lru_adapter = self._adapters.get(lru_name)
                if lru_adapter is None:
                    self._resident.pop(lru_name, None)
                    continue
                evictees.append((lru_name, lru_adapter))
                lru_entry.evicting = True
                break
            self._pending[name] = applier_id

        restore: list[tuple[str, _Residency]] = []
        try:
            for lru_name, lru_adapter in evictees:
                with self._lock:
                    lru_entry = self._resident.get(lru_name)
                unload_applier = (
                    lru_entry.applier if lru_entry is not None and lru_entry.applier else applier
                )
                try:
                    unloaded = bool(unload_applier.unload(lru_adapter))
                except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                    record_degradation("expert_lora_unload", exc)
                    unloaded = False
                with self._lock:
                    entry = self._resident.get(lru_name)
                    if entry is not None:
                        entry.evicting = False
                    if unloaded and entry is not None:
                        restore.append((lru_name, entry))
                        self._resident.pop(lru_name, None)
                if not unloaded:
                    # CP126 b022149a: a refused unload must not free budget.
                    record_degradation(
                        "expert_lora_unload",
                        RuntimeError(f"adapter_unload_refused:{lru_name}"),
                        action="refused new activation because the evictee is still attached",
                        severity="error",
                    )
                    return False
            # CP126 34f303bc: the sync load was the ONLY applier call without
            # an exception boundary, so a raise escaped after evictions had
            # already changed physical and logical state.
            try:
                ok = bool(applier.load(adapter))
            except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                record_degradation("expert_lora_load", exc)
                ok = False
            if ok:
                with self._lock:
                    self._resident[name] = _Residency(
                        applier_id=applier_id,
                        applier=applier,
                        size_mb=float(adapter.size_mb or 0.0),
                        last_used=time.time(),
                    )
                return True
            # CP126 fcc5ab93: a failed replacement load used to leave the
            # evicted adapters unloaded and forgotten — turning a failed
            # activation into destructive state loss. Put them back.
            self._restore_evicted(restore, applier)
            return False
        finally:
            with self._lock:
                self._pending.pop(name, None)

    def _restore_evicted(
        self,
        restore: list[tuple[str, _Residency]],
        applier: Any,
    ) -> None:
        """Reload adapters evicted for an activation that then failed."""
        for lru_name, entry in restore:
            adapter = self.get(lru_name)
            if adapter is None:
                continue
            reload_applier = entry.applier or applier
            try:
                reloaded = bool(reload_applier.load(adapter))
            except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                record_degradation("expert_lora_load", exc)
                reloaded = False
            if reloaded:
                with self._lock:
                    self._resident[lru_name] = _Residency(
                        applier_id=entry.applier_id,
                        applier=entry.applier,
                        size_mb=entry.size_mb,
                        last_used=entry.last_used,
                    )
            else:
                record_degradation(
                    "expert_lora_load",
                    RuntimeError(f"evicted_adapter_not_restored:{lru_name}"),
                    action="reported lost residency after a failed activation rollback",
                    severity="error",
                )

    def _evict(self, name: str) -> bool:
        """Unload an adapter and free its residency ONLY on confirmed unload.

        CP126 b022149a: both paths deleted residency first and ignored a False
        unload result (exceptions did not restore it either), so the library
        could claim capacity was free while the old adapter remained attached
        — and then load another adapter over budget.
        """
        # Caller holds lock.
        entry = self._resident.get(name)
        if entry is None:
            return True
        adapter = self._adapters.get(name)
        if adapter is None:
            # Nothing left to unload through; drop the stale record.
            self._resident.pop(name, None)
            return True
        # CP126 58bc5a8d: unload through the applier that actually loaded it.
        applier = entry.applier if entry.applier is not None else self._applier
        try:
            unloaded = bool(applier.unload(adapter))
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            record_degradation("expert_lora_unload", exc)
            return False
        if not unloaded:
            record_degradation(
                "expert_lora_unload",
                RuntimeError(f"adapter_unload_refused:{name}"),
                action="kept residency because the applier did not confirm unload",
                severity="error",
            )
            return False
        self._resident.pop(name, None)
        return True

    def deactivate(self, name: str) -> bool:
        """Release an adapter's weights. True when nothing is attached after."""
        with self._lock:
            return self._evict(name)

    def resident(self) -> list[str]:
        with self._lock:
            return list(self._resident.keys())

    def resident_for(self, applier: Any) -> list[str]:
        """Adapters attached BY THIS applier (CP126 58bc5a8d).

        A second worker must not conclude an adapter is loaded in ITS model
        just because another worker's applier loaded one with the same name.
        """
        target = self._applier_id(applier)
        with self._lock:
            return [n for n, e in self._resident.items() if e.applier_id == target]

    def select_and_activate(self, objective: str, task_type: str, *, base_model: str = "") -> LoRAAdapter | None:
        """One call for the generation path: pick the specialist and make it resident.

        Returns the activated adapter, or None when disabled / no match. Default-off
        via AURA_EXPERT_LORA_LIBRARY so it never alters generation implicitly.
        """
        if not _flag_on("AURA_EXPERT_LORA_LIBRARY"):
            return None
        adapter = self.select_for(objective, task_type, base_model=base_model)
        if adapter is None:
            return None
        return adapter if self.activate(adapter.name) else None

    # ── async residency (live worker seam) ────────────────────────────────────
    async def activate_async(self, name: str, applier: AsyncAdapterApplier) -> bool:
        """Make an adapter resident through an ASYNC applier (worker IPC).

        Same residency contract as ``activate`` but the attach/detach I/O is
        awaited without holding the registry lock, so a multi-second worker
        swap can never stall other registry readers. Residency maps update
        only from actual applier outcomes — ``resident()`` never claims an
        adapter the worker refused.
        """
        applier_id = self._applier_id(applier)
        evictees: list[tuple[str, LoRAAdapter, Any]] = []
        with self._lock:
            adapter = self._adapters.get(name)
            if adapter is None:
                return False
            entry = self._resident.get(name)
            if entry is not None:
                if entry.applier_id == applier_id:
                    self._resident.move_to_end(name)
                    entry.last_used = time.time()
                    return True
                # CP126 58bc5a8d: another worker/applier holds this name. It is
                # NOT loaded in this one, so claiming True would attest weights
                # that were never attached here.
                logger.info(
                    "📎 [ExpertLoRA] '%s' is resident on a different applier (%s); "
                    "not claiming residency for %s.",
                    name,
                    entry.applier_id,
                    applier_id,
                )
                return False
            # CP126 be41d7a1: no slot was reserved while the lock was released
            # for I/O, so two callers could both observe capacity, both load,
            # and both append residency — exceeding _max_resident (and loading
            # the same adapter twice).
            if name in self._pending:
                return False
            while self._needs_eviction(adapter):
                evictable = [
                    n for n, e in self._resident.items() if not e.evicting
                ]
                if not evictable:
                    # Capacity is held by in-flight activations (or by entries
                    # already being evicted). Refuse rather than exceed the
                    # budget — the previous code reserved nothing while it
                    # awaited I/O, so two callers each concluded there was room.
                    logger.info(
                        "📎 [ExpertLoRA] '%s' not activated: residency budget is "
                        "fully reserved (resident=%d pending=%d).",
                        name,
                        len(self._resident),
                        len(self._pending),
                    )
                    return False
                lru_name = evictable[0]
                lru_entry = self._resident[lru_name]
                lru_adapter = self._adapters.get(lru_name)
                if lru_adapter is None:
                    self._resident.pop(lru_name, None)
                    continue
                evictees.append(
                    (lru_name, lru_adapter, lru_entry.applier or applier)
                )
                # Reserve the eviction: hold the slot so no concurrent caller
                # re-admits it, but do NOT free the budget until the unload is
                # confirmed below (CP126 b022149a).
                lru_entry.evicting = True
                break
            self._pending[name] = applier_id

        restored: list[tuple[str, _Residency]] = []
        try:
            for lru_name, lru_adapter, lru_applier in evictees:
                try:
                    unloaded = bool(await lru_applier.unload(lru_adapter))
                except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                    record_degradation("expert_lora_unload", exc)
                    unloaded = False
                with self._lock:
                    entry = self._resident.get(lru_name)
                    if entry is not None:
                        entry.evicting = False
                    if unloaded and entry is not None:
                        restored.append((lru_name, entry))
                        self._resident.pop(lru_name, None)
                if not unloaded:
                    record_degradation(
                        "expert_lora_unload",
                        RuntimeError(f"adapter_unload_refused:{lru_name}"),
                        action="refused new activation because the evictee is still attached",
                        severity="error",
                    )
                    return False
            try:
                ok = bool(await applier.load(adapter))
            except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                record_degradation("expert_lora_load", exc)
                ok = False
            if ok:
                with self._lock:
                    self._resident[name] = _Residency(
                        applier_id=applier_id,
                        applier=applier,
                        size_mb=float(adapter.size_mb or 0.0),
                        last_used=time.time(),
                    )
                return True
            # CP126 fcc5ab93: restore what this activation evicted rather than
            # leaving a failed load as destructive state loss.
            await self._restore_evicted_async(restored, applier)
            return False
        finally:
            with self._lock:
                self._pending.pop(name, None)

    async def _restore_evicted_async(
        self,
        restore: list[tuple[str, _Residency]],
        applier: Any,
    ) -> None:
        for lru_name, entry in restore:
            adapter = self.get(lru_name)
            if adapter is None:
                continue
            reload_applier = entry.applier or applier
            try:
                reloaded = bool(await reload_applier.load(adapter))
            except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                record_degradation("expert_lora_load", exc)
                reloaded = False
            if reloaded:
                with self._lock:
                    self._resident[lru_name] = _Residency(
                        applier_id=entry.applier_id,
                        applier=entry.applier,
                        size_mb=entry.size_mb,
                        last_used=entry.last_used,
                    )
            else:
                record_degradation(
                    "expert_lora_load",
                    RuntimeError(f"evicted_adapter_not_restored:{lru_name}"),
                    action="reported lost residency after a failed async activation rollback",
                    severity="error",
                )

    async def select_and_activate_async(
        self,
        objective: str,
        task_type: str,
        applier: AsyncAdapterApplier,
        *,
        base_model: str = "",
    ) -> LoRAAdapter | None:
        """Async twin of ``select_and_activate`` for the live generation path."""
        if not _flag_on("AURA_EXPERT_LORA_LIBRARY"):
            return None
        adapter = self.select_for(objective, task_type, base_model=base_model)
        if adapter is None:
            return None
        return adapter if await self.activate_async(adapter.name, applier) else None

    # ── disk discovery ────────────────────────────────────────────────────────
    def scan(self, directory: str | Path, *, base_model: str = "", source: str = "scan") -> int:
        """Register adapters found under ``directory`` (dirs with adapter_config.json /
        adapters.safetensors). Returns count newly registered."""
        root = Path(os.path.expanduser(str(directory)))
        if not root.exists():
            return 0
        found = 0
        markers = _ADAPTER_MARKERS
        weight_markers = ("adapters.safetensors", "adapters.npz")
        for cfg in root.rglob("*"):
            try:
                if cfg.is_dir() and any((cfg / m).exists() for m in markers):
                    name = cfg.name
                    if name in self._adapters:
                        continue
                    # CP126 315080b7: a directory containing ONLY a config
                    # marker was registered as an adapter. Require actual
                    # weight material, refuse symlinked artifacts, and size
                    # the whole tree (immediate-file-only sizing omitted
                    # nested weights and so understated the RAM budget).
                    if cfg.is_symlink():
                        continue
                    if not any((cfg / m).is_file() for m in weight_markers):
                        logger.debug(
                            "📚 [ExpertLoRA] scan skipped '%s': no weight material", cfg
                        )
                        continue
                    size_mb = sum(
                        f.stat().st_size
                        for f in cfg.rglob("*")
                        if f.is_file() and not f.is_symlink()
                    ) / (1024 * 1024)
                    self.register(
                        LoRAAdapter(
                            name=name,
                            path=str(cfg),
                            base_model=base_model,
                            task_types=set(),
                            keywords=_tokens(name),
                            size_mb=size_mb,
                            source=source,
                        )
                    )
                    found += 1
            except self._ERRORS as exc:
                record_degradation("expert_lora_scan", exc)
        return found

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "adapters": len(self._adapters),
                "resident": list(self._resident.keys()),
                "max_resident": self._max_resident,
                # The advertised RAM budget, and what is actually spent on it.
                "max_resident_mb": round(self._max_resident_mb, 2),
                "resident_mb": round(self._resident_mb(), 2),
                "resident_by_applier": {
                    name: entry.applier_id for name, entry in self._resident.items()
                },
                "pending_activations": len(self._pending),
                "enabled": _flag_on("AURA_EXPERT_LORA_LIBRARY"),
            }

    # ── persistence ───────────────────────────────────────────────────────────
    def _load_manifest(self) -> None:
        if not self._manifest_path.exists():
            return
        try:
            raw = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            items = raw.get("adapters", {}) if isinstance(raw, dict) else {}
            if not isinstance(items, dict):
                items = {}
            with self._lock:
                for name, data in items.items():
                    try:
                        adapter = LoRAAdapter.from_dict(data)
                    except (ValueError, TypeError, AttributeError):
                        continue
                    key = str(name).strip()
                    if not key or not adapter.path:
                        continue
                    # CP126 68e1d462: the record was stored under the OUTER
                    # manifest key while from_dict read its own internal name,
                    # so selection could return an object whose .name did not
                    # match its registry key — activation then looked up
                    # adapter.name and missed (or hit a different adapter).
                    # The key is authoritative; a disagreement is recorded.
                    if adapter.name != key:
                        record_degradation(
                            "expert_lora_manifest",
                            ValueError(
                                f"manifest_key_name_mismatch:{key}!={adapter.name}"
                            ),
                            action="reconciled adapter name to its manifest key on load",
                            severity="warning",
                        )
                        adapter.name = key
                    self._adapters[key] = adapter
        except self._ERRORS as exc:
            record_degradation("expert_lora_load", exc)

    def _persist(self) -> bool:
        """Write the manifest. Returns whether the durable write succeeded."""
        try:
            self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 1,
                "saved_at": time.time(),
                "adapters": {n: a.to_dict() for n, a in self._adapters.items()},
            }
            fd, tmp = tempfile.mkstemp(prefix=".lora_lib_", suffix=".json", dir=str(self._manifest_path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False)
                os.replace(tmp, self._manifest_path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            return True
        except self._ERRORS as exc:
            record_degradation("expert_lora_persist", exc)
            return False


_singleton: ExpertLoRALibrary | None = None
_singleton_lock = threading.Lock()


def get_expert_lora_library() -> ExpertLoRALibrary:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = ExpertLoRALibrary()
    return _singleton


def reset_expert_lora_library() -> None:
    """Drop the singleton AFTER releasing whatever it made resident.

    CP126 21661496: this only cleared the reference, so live model
    modifications outlived the registry that tracked them — the next library
    had no record of adapters still attached to the worker.
    """
    global _singleton
    with _singleton_lock:
        library, _singleton = _singleton, None
    if library is None:
        return
    for name in library.resident():
        try:
            library.deactivate(name)
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            record_degradation(
                "expert_lora_reset",
                exc,
                action="library reset left an adapter attached to the model",
                severity="error",
            )
    remaining = library.resident()
    if remaining:
        record_degradation(
            "expert_lora_reset",
            RuntimeError(f"adapters_still_attached_after_reset:{','.join(remaining[:4])}"),
            action="reported adapters that outlived their registry",
            severity="error",
        )
