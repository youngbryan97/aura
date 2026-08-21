import asyncio
import hashlib
import json
import logging
import math
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from core.governance_context import local_internal_governed_scope
from core.memory import embedding_model
from core.memory.black_hole import BlackHoleDecodeError, decode_payload, encode_payload
from core.memory.horcrux import HorcruxManager
from core.memory.physics import bekenstein_check, hawking_decay
from core.memory.retention_policy import MemoryRetentionPolicy, black_hole_retention_policy
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway

try:
    from core.memory.rag import compute_term_freq, retrieve_memories, tokenize
except (ImportError, AttributeError, RuntimeError):
    from core.memory.rag import compute_term_freq, tokenize
    def retrieve_memories(query, memories, top_k=5, threshold=0.01, **kwargs):
        return []

logger = logging.getLogger("Aura.BlackHoleVault")

SEMANTIC_DECAY_LAMBDA_PER_DAY = 0.08
_VAULT_LOAD_ERRORS = (
    OSError,
    json.JSONDecodeError,
    RuntimeError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

class BlackHoleVault:
    """The central unified interface replacing VectorMemory.

    Search is hybrid semantic + lexical RAG (dense MiniLM cosine blended
    with TF-IDF — see core/memory/rag.py), Horcrux for keys, and Black
    Hole algorithms for storage.
    """

    memory_consolidation_backend = "black_hole_vault"
    collection_name = "black_hole_vault"
    embedding_metric = "cosine"
    embedding_version = "black-hole-tfhash384-v1"
    single_principal_collection = True

    def __init__(self, data_dir: str = "~/.aura/vault"):
        self.data_dir = os.path.expanduser(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        self.memories_file = os.path.join(self.data_dir, "event_horizon.json")
        
        self.horcrux = HorcruxManager(base_dir=os.path.dirname(self.data_dir))
        self.key = "fallback-locked-key"
        self.memories = []
        self._retention_policy = black_hole_retention_policy()
        self._max_memories = self._retention_policy.max_items
        self._dirty = False
        self._fallback_mode = False
        self._collection = self  # Compatibility surface, not a Chroma collection.
        self._mutation_lock = threading.RLock()
        self._initialized = False
        self._init_error: Optional[str] = None
        self._ensure_ready()
        
    async def on_start_async(self):
        """Standard lifecycle hook called by ServiceContainer."""
        await self.initialize()

    async def initialize(self) -> bool:
        """Async initialization for Horcrux and Vault."""
        if not await self.horcrux.initialize():
            logger.error("Horcrux failed to initialize! Black Hole Vault is locked.")
            return False
            
        self.key = self.horcrux.get_key_string()
        await asyncio.to_thread(self._load_vault)
        self._initialized = True
        self._init_error = None
        return True

    def _run_initialize_blocking(self) -> bool:
        """Run async Horcrux bootstrap from sync callers safely."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return bool(asyncio.run(self.initialize()))

        result: Dict[str, Any] = {}

        def _runner() -> None:
            try:
                result["ok"] = bool(asyncio.run(self.initialize()))
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:  # pragma: no cover - defensive fallback
                result["error"] = exc

        thread = threading.Thread(
            target=_runner,
            name="BlackHoleVaultInit",
            daemon=True,
        )
        thread.start()
        thread.join()

        if "error" in result:
            raise result["error"]
        return bool(result.get("ok"))

    def _ensure_ready(self) -> None:
        if self._initialized and self.horcrux.derived_key:
            return
        if self.horcrux.derived_key:
            self.key = self.horcrux.get_key_string()
            self._initialized = True
            self._init_error = None
            return
        try:
            if not self._run_initialize_blocking():
                self._init_error = "Horcrux initialization returned False"
                logger.warning("BlackHoleVault running in degraded mode: %s", self._init_error)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation('black_hole_vault', exc)
            self._init_error = str(exc)
            logger.warning("BlackHoleVault initialization degraded: %s", exc)
            
    def _load_vault(self):
        if not os.path.exists(self.memories_file):
            self.memories = []
            return
        try:
            with open(self.memories_file, "r", encoding="utf-8") as f:
                encrypted_data = f.read().strip()
            if not encrypted_data:
                self.memories = []
                return
            res = decode_payload(encrypted_data, self.key, strict=True)
            raw_json = res.get("decoded", "")
            self.memories = json.loads(raw_json) if raw_json else []
            self._ensure_memory_ids(persist=True)
        except BlackHoleDecodeError as exc:
            quarantined = self._quarantine_unreadable_vault(reason=type(exc).__name__)
            logger.warning(
                "BlackHoleVault quarantined unreadable encrypted memory file: %s",
                quarantined or "quarantine_failed",
            )
            self._fallback_mode = True
            self.memories = []
        except _VAULT_LOAD_ERRORS as e:
            record_degradation("black_hole_vault.load", e)
            logger.warning("Failed to load vault (falling back to empty): %s", e)
            self._fallback_mode = True
            self.memories = []

    def _quarantine_unreadable_vault(self, *, reason: str) -> str:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = f"{self.memories_file}.quarantine-{stamp}-{reason}"
        target = base
        counter = 1
        while os.path.exists(target):
            counter += 1
            target = f"{base}-{counter}"
        try:
            os.replace(self.memories_file, target)
            return target
        except OSError as exc:
            record_degradation("black_hole_vault.quarantine", exc)
            logger.warning("Failed to quarantine unreadable vault file: %s", exc)
            return ""
            
    def _save_vault(self):
        self._ensure_ready()
        if not self._dirty:
            return
        raw_json = json.dumps(self.memories)
        encoded = encode_payload(raw_json, self.key)
        with local_internal_governed_scope(
            "black_hole_vault.save_vault",
            domain="memory_write",
            receipt_prefix="black-hole-vault-save",
        ):
            get_file_write_gateway().write_text(
                self.memories_file,
                encoded["encoded"],
                source="black_hole_vault.save_vault",
            )
        self._dirty = False

    @staticmethod
    def _memory_id(memory: Dict[str, Any]) -> str:
        return str(memory.get("id") or memory.get("created") or "").strip()

    def _ensure_memory_ids(self, *, persist: bool = False) -> bool:
        """Migrate legacy timestamp identities to stable per-record identities."""
        changed = False
        occupied = {
            str(memory.get("id"))
            for memory in self.memories
            if str(memory.get("id") or "").strip()
        }
        for ordinal, memory in enumerate(self.memories):
            if str(memory.get("id") or "").strip():
                continue
            seed = json.dumps(
                {
                    "created": memory.get("created"),
                    "text": memory.get("text"),
                    "metadata": memory.get("metadata"),
                    "ordinal": ordinal,
                },
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            )
            candidate = f"bhv-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:24]}"
            suffix = 1
            unique = candidate
            while unique in occupied:
                suffix += 1
                unique = f"{candidate}-{suffix}"
            memory["id"] = unique
            occupied.add(unique)
            changed = True
        if changed:
            self._dirty = True
            if persist:
                self._save_vault()
        return changed
            
    def add_memory(
        self,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Standard interface matching VectorMemory and legacy content= callers."""
        self._ensure_ready()
        if text is None:
            text = kwargs.pop("content", None)
        if not text or not str(text).strip():
            return False

        text = str(text)
        metadata = metadata or {}
        current_bytes = len(json.dumps(self.memories).encode()) if self.memories else 0
        new_bytes = len(text.encode())
        
        # Physics bounds check
        check = bekenstein_check((current_bytes + new_bytes) * 8, radius_cm=10.0, energy_mj=50.0)
        if not check["fits"]:
            logger.warning("Bekenstein Bound Exceeded! Evaporating oldest memories...")
            self._evaporate()
            
        # One memory, one vector — unless it genuinely overflows the encoder.
        # The old fixed 800-word split predated a 256-token window; it shredded
        # every episode into pieces whose facts could never co-occur in a
        # single vector, so nothing downstream could score them together.
        chunks = embedding_model.chunk_for_embedding(text)
        now_ms = int(time.time() * 1000)
        semantic_meta = self._normalize_memory_metadata(metadata)
        
        for c in chunks:
            tokens = tokenize(c)
            vec = compute_term_freq(tokens)
            
            self.memories.append({
                "id": f"bhv-{uuid.uuid4().hex}",
                "text": c,
                "metadata": semantic_meta,
                "vec": vec,
                "created": now_ms,
                "access_count": 0
            })
            
        self._enforce_retention_cap()
            
        self._dirty = True
        self._save_vault()
        return True

    def _policy(self) -> MemoryRetentionPolicy:
        policy = getattr(self, "_retention_policy", None)
        if isinstance(policy, MemoryRetentionPolicy):
            return policy
        max_memories = int(getattr(self, "_max_memories", 5000) or 5000)
        return MemoryRetentionPolicy(max_items=max_memories, prune_keep_fraction=0.90, basis="legacy_fallback")

    def _enforce_retention_cap(self) -> None:
        policy = self._policy()
        self._max_memories = policy.max_items
        if len(self.memories) <= policy.max_items:
            return
        keep_count = policy.keep_count(len(self.memories))
        original_count = len(self.memories)
        self.memories = self._select_semantically_important(self.memories, keep_count)
        logger.info(
            "BlackHoleVault retention cap: kept %d of %d memories using %s policy.",
            len(self.memories),
            original_count,
            policy.basis,
        )

    def _normalize_memory_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(metadata or {})
        centrality = max(
            _safe_float(normalized.get("conceptual_centrality")),
            _safe_float(normalized.get("centrality")),
            _safe_float(normalized.get("identity_centrality")),
            1.0 if normalized.get("core_identity") or normalized.get("pinned") else 0.0,
        )
        affect = max(
            _safe_float(normalized.get("affect_intensity")),
            abs(_safe_float(normalized.get("valence"))),
            _safe_float(normalized.get("arousal")) * 0.5,
            _safe_float(normalized.get("importance")),
        )
        normalized["conceptual_centrality"] = max(0.0, min(1.0, centrality))
        normalized["affect_intensity"] = max(0.0, min(1.0, affect))
        return normalized

    def _memory_importance(self, memory: Dict[str, Any], *, now_ms: int | None = None) -> float:
        now_ms = now_ms or int(time.time() * 1000)
        metadata = memory.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        if metadata.get("pinned") or metadata.get("core_identity") or metadata.get("never_prune"):
            return float("inf")
        age_days = max(0.0, (now_ms - _safe_float(memory.get("created"), now_ms)) / 86_400_000.0)
        access_pressure = min(1.0, _safe_float(memory.get("access_count")) / 10.0)
        affective_amplitude = max(
            _safe_float(metadata.get("affect_intensity")),
            _safe_float(metadata.get("importance")),
            access_pressure,
        )
        conceptual_centrality = max(
            _safe_float(metadata.get("conceptual_centrality")),
            _safe_float(metadata.get("centrality")),
            _safe_float(metadata.get("identity_centrality")),
        )
        decayed_amplitude = affective_amplitude * math.exp(-SEMANTIC_DECAY_LAMBDA_PER_DAY * age_days)
        return decayed_amplitude + conceptual_centrality

    def _select_semantically_important(self, memories: List[Dict[str, Any]], keep_count: int) -> List[Dict[str, Any]]:
        now_ms = int(time.time() * 1000)
        return sorted(
            list(memories),
            key=lambda memory: self._memory_importance(memory, now_ms=now_ms),
            reverse=True,
        )[: max(0, keep_count)]
        
    def search_similar(self, query: str, limit: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Standard interface matching VectorMemory"""
        self._ensure_ready()
        if not self.memories:
            return []
        try:
            results = retrieve_memories(query, self.memories, top_k=limit, threshold=0.01)
        except TypeError:
            results = retrieve_memories(query, self.memories, limit, threshold=0.01)
        formatted = []

        for r in results:
            decay = hawking_decay(r["created"], self.key)
            if decay["fidelity"] < 0.1:
                continue 
                
            # Boost access count for Gravitational Queue
            original = next((m for m in self.memories if m["created"] == r["created"] and m["text"] == r["text"]), None)
            if original:
                # Sanitize access count retrieval for Pyre
                raw_count = original.get("access_count", 0)
                if isinstance(raw_count, (int, float)):
                    count_val = int(raw_count)
                else:
                    count_val = 0
                
                original["access_count"] = count_val + 1
                
                # Evolution 8: High-Gravity Pulse
                if count_val + 1 > 10:
                    try:
                        from core.container import ServiceContainer
                        mycelium = ServiceContainer.get("mycelium", default=None)
                        if mycelium:
                            mycelium.pulse_hypha("memory", "vault", success=True)
                    except (ImportError, AttributeError, RuntimeError) as _e:
                        record_degradation('black_hole_vault', _e)
                        logger.debug('Ignored Exception in black_hole_vault.py: %s', _e)
            
            formatted.append({
                "content": r["text"],
                "metadata": r.get("metadata", {}),
                "score": r["score"] * decay["fidelity"]
            })
            
        self._dirty = True
        # Note: We don't save immediately on searches to avoid excessive I/O.
        # It will be persisted on the next write or exit.
        return formatted

    # --- Legacy Compatibility Aliases ---
    async def index(self, content: str, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        """Async shim for MemoryManager compatibility."""
        import asyncio
        return await asyncio.to_thread(self.add_memory, content, metadata)
        
    def search(self, query: str, limit: int = 5, **kwargs):
        return self.search_similar(query, limit)
        
    def get(self, ids: Optional[List[str]] = None, limit: Optional[int] = None, include: Optional[List[str]] = None, **kwargs) -> Any:
        """Bulk retrieval for ChromaDB compatibility and SemanticDefragmenter support."""
        # If a single string is passed as the first positional arg (legacy behavior)
        if isinstance(ids, str) and not limit and not include:
            memory_id = ids
            for m in self.memories:
                if self._memory_id(m) == str(memory_id) or str(m.get("created")) == str(memory_id):
                    return m
            return None

        # Bulk retrieval logic
        found = []
        if ids:
            id_set = set(str(i) for i in ids)
            found = [
                m
                for m in self.memories
                if self._memory_id(m) in id_set or str(m.get("created")) in id_set
            ]
        else:
            found = self.memories
            
        if isinstance(found, list) and limit:
            sequence = list(found)
            found = sequence[:limit] if ids else sequence[-limit:]
            
        ret: Dict[str, Any] = {
            "ids": [self._memory_id(m) for m in found] if isinstance(found, list) else [],
            "documents": [str(m.get("text", "")) for m in found] if isinstance(found, list) else [],
            "metadatas": [m.get("metadata", {}) for m in found] if isinstance(found, list) else []
        }
        return ret

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Alias for get() to support various component integrations."""
        return self.get(memory_id)

    @property
    def total_mass_kb(self) -> float:
        """Returns the current mass of the vault in KB."""
        total_bytes = len(json.dumps(self.memories).encode()) if self.memories else 0
        return round(total_bytes / 1024, 2)

    def _evaporate(self):
        if not self.memories: return
        
        # Notify Mycelium of qualitative shift (Evolution 8)
        try:
            from core.container import ServiceContainer
            mycelium = ServiceContainer.get("mycelium", default=None)
            if mycelium:
                mycelium.log_hypha(
                    "memory", "vault", "EVAPORATION: Qualitative shift in history."
                )
                mycelium.pulse_hypha("memory", "vault", success=True)
        except (ImportError, AttributeError, RuntimeError) as _e:
            record_degradation('black_hole_vault', _e)
            logger.debug('Ignored Exception in black_hole_vault.py: %s', _e)

        keep_count = self._policy().keep_count(len(self.memories))
        self.memories = self._select_semantically_important(self.memories, keep_count)
        self._save_vault()

    def clear(self):
        """Standard interface: Reset the vault."""
        self.memories = []
        if os.path.exists(self.memories_file):
            os.remove(self.memories_file)
        logger.info("BlackHoleVault: Event horizon cleared.")

    def delete(self, ids: List[str]):
        """Standard interface: Delete memories by ID."""
        id_set = {str(memory_id) for memory_id in ids}
        self.memories = [
            m
            for m in self.memories
            if self._memory_id(m) not in id_set and str(m.get("created")) not in id_set
        ]
        self._dirty = True
        self._save_vault()
        logger.info("BlackHoleVault: Deleted %d memories.", len(ids))

    def delete_memories(
        self,
        ids: Optional[List[str]] = None,
        *,
        filter_metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> int:
        """VectorMemory-compatible deletion shim used by episodic pruning."""
        id_set = {str(memory_id) for memory_id in (ids or []) if str(memory_id)}
        metadata_filters = dict(filter_metadata or {})
        if not id_set and not metadata_filters:
            return 0

        normalized_filters: Dict[str, set[str]] = {}
        for key, raw_value in metadata_filters.items():
            if isinstance(raw_value, (list, tuple, set)):
                values = {str(value) for value in raw_value}
            else:
                values = {str(raw_value)}
            normalized_filters[str(key)] = values

        def _matches(memory: Dict[str, Any]) -> bool:
            if self._memory_id(memory) in id_set or str(memory.get("created")) in id_set:
                return True
            metadata = memory.get("metadata", {}) or {}
            if not isinstance(metadata, dict):
                return False
            return any(
                str(metadata.get(key)) in accepted
                for key, accepted in normalized_filters.items()
            )

        before = len(self.memories)
        self.memories = [memory for memory in self.memories if not _matches(memory)]
        deleted = before - len(self.memories)
        if deleted:
            self._dirty = True
            self._save_vault()
            logger.info("BlackHoleVault: Deleted %d memories via compatibility filter.", deleted)
        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """Standard interface: Return collection statistics."""
        return {
            "total_vectors": len(self.memories),
            "total_mass_kb": self.total_mass_kb,
            "max_vectors": self._policy().max_items,
            "retention_policy": self._policy().to_dict(),
            "engine": "black_hole_vault",
            "status": "active"
        }
