import os
import time
import json
import asyncio
import logging
import threading
from typing import Dict, Any, List, Optional
from core.memory.horcrux import HorcruxManager
from core.memory.black_hole import encode_payload, decode_payload
from core.memory.physics import bekenstein_check, hawking_decay, grav_queue_sort
try:
    from core.memory.rag import chunk_text, tokenize, compute_term_freq, retrieve_memories
except Exception:
    from core.memory.rag import chunk_text, tokenize, compute_term_freq
    def retrieve_memories(query, memories, top_k=5, threshold=0.01, **kwargs):
        return []

logger = logging.getLogger("Aura.BlackHoleVault")

class BlackHoleVault:
    """The central unified interface replacing VectorMemory.
    Uses TF-IDF RAG for search, Horcrux for keys, and Black Hole algorithms for storage.
    """
    def __init__(self, data_dir: str = "~/.aura/vault"):
        self.data_dir = os.path.expanduser(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        self.memories_file = os.path.join(self.data_dir, "event_horizon.json")
        
        self.horcrux = HorcruxManager(base_dir=os.path.dirname(self.data_dir))
        self.key = "fallback-locked-key"
        self.memories = []
        self._dirty = False
        self._fallback_mode = False
        self._collection = self # Shim for SemanticDefragmenter
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
            except Exception as exc:  # pragma: no cover - defensive fallback
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
        except Exception as exc:
            self._init_error = str(exc)
            logger.warning("BlackHoleVault initialization degraded: %s", exc)
            
    def _load_vault(self):
        if not os.path.exists(self.memories_file):
            self.memories = []
            return
        try:
            with open(self.memories_file, "r", encoding="utf-8") as f:
                encrypted_data = f.read().strip()
            res = decode_payload(encrypted_data, self.key)
            raw_json = res.get("decoded", "")
            self.memories = json.loads(raw_json) if raw_json else []
        except Exception:
            self._fallback_mode = True
            self.memories = []
            
    def _save_vault(self):
        self._ensure_ready()
        if not self._dirty:
            return
        tmp = self.memories_file + ".tmp"
        raw_json = json.dumps(self.memories)
        encoded = encode_payload(raw_json, self.key)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(encoded["encoded"])
        os.replace(tmp, self.memories_file)
        self._dirty = False
            
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
            
        chunks = chunk_text(text, chunk_size=800, overlap=80)
        now_ms = int(time.time() * 1000)
        
        for c in chunks:
            tokens = tokenize(c)
            vec = compute_term_freq(tokens)
            
            self.memories.append({
                "text": c,
                "metadata": metadata,
                "vec": vec,
                "created": now_ms,
                "access_count": 0
            })
            
        self._dirty = True
        self._save_vault()
        return True
        
    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Standard interface matching VectorMemory"""
        self._ensure_ready()
        if not self.memories:
            return []
        try:
            results = retrieve_memories(query, self.memories, top_k=limit, threshold=0.01)
        except TypeError:
            results = retrieve_memories(query, self.memories, limit, threshold=0.01)
        formatted = []
        now_ms = int(time.time() * 1000)
        
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
                            hypha = mycelium.get_hypha("memory", "vault")
                            if hypha: hypha.pulse(success=True)
                    except Exception as _e:
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
                if str(m.get("created")) == str(memory_id):
                    return m
            return None

        # Bulk retrieval logic
        found = []
        if ids:
            id_set = set(str(i) for i in ids)
            found = [m for m in self.memories if str(m.get("created")) in id_set]
        else:
            found = self.memories
            
        if isinstance(found, list) and limit:
            sequence = list(found)
            found = sequence[:limit] if ids else sequence[-limit:]
            
        ret: Dict[str, Any] = {
            "ids": [str(m.get("created")) for m in found] if isinstance(found, list) else [],
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
                hypha = mycelium.get_hypha("memory", "vault")
                if hypha:
                    hypha.log("EVAPORATION: Qualitative shift in history.")
                    hypha.pulse(success=True)
        except Exception as _e:
            logger.debug('Ignored Exception in black_hole_vault.py: %s', _e)

        sorted_mems = grav_queue_sort(self.memories)
        keep_count = int(len(self.memories) * 0.8)
        self.memories = sorted_mems[:keep_count]
        self._save_vault()

    def clear(self):
        """Standard interface: Reset the vault."""
        self.memories = []
        if os.path.exists(self.memories_file):
            os.remove(self.memories_file)
        logger.info("BlackHoleVault: Event horizon cleared.")

    def delete(self, ids: List[str]):
        """Standard interface: Delete memories by ID."""
        self.memories = [m for m in self.memories if str(m.get("created")) not in ids]
        self._dirty = True
        self._save_vault()
        logger.info("BlackHoleVault: Deleted %d memories.", len(ids))

    def get_stats(self) -> Dict[str, Any]:
        """Standard interface: Return collection statistics."""
        return {
            "total_vectors": len(self.memories),
            "total_mass_kb": self.total_mass_kb,
            "engine": "black_hole_vault",
            "status": "active"
        }
