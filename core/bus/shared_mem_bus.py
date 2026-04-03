import asyncio
import errno
import json
import logging
import mmap
import os
import tempfile
import uuid
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

logger = logging.getLogger("Bus.SharedMem")


def _fallback_root() -> Path:
    configured = os.environ.get("AURA_SHM_FALLBACK_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(tempfile.gettempdir()) / "aura_shm"


def _safe_segment_name(name: str) -> str:
    raw = str(name or "").strip().lstrip("/")
    if not raw:
        return "aura_shm"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)


class _FileBackedSharedMemory:
    """Fallback transport backed by a tmp file and mmap."""

    def __init__(self, name: str, path: Path, fd: int, mm: mmap.mmap):
        self.name = name
        self.path = path
        self._fd = fd
        self._mmap = mm

    @property
    def buf(self) -> memoryview:
        return memoryview(self._mmap)

    def close(self) -> None:
        try:
            self._mmap.close()
        finally:
            os.close(self._fd)

    def unlink(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except TypeError:
            if self.path.exists():
                self.path.unlink()


class SharedMemoryTransport:
    """
    High-performance, zero-copy transport for large state payloads.
    Uses named shared memory segments for cross-process synchronization.
    """
    
    def __init__(self, name: str, size: int = 8 * 1024 * 1024): # Default 8MB
        self.name = name
        self.size = size
        self.shm: Any | None = None
        self._is_owner = False
        self._bus_id = str(uuid.uuid4())
        self._backend = "posix_shm"

    @property
    def payload_capacity(self) -> int:
        """Bytes available for the JSON payload after the seqlock header."""
        return max(0, self.size - 13)

    def _fallback_path(self) -> Path:
        return _fallback_root() / f"{_safe_segment_name(self.name)}.bin"

    @staticmethod
    def _should_use_file_fallback(exc: Exception) -> bool:
        if isinstance(exc, (PermissionError, NotImplementedError)):
            return True
        if isinstance(exc, OSError):
            return getattr(exc, "errno", None) in {errno.EPERM, errno.EACCES, errno.ENOSYS}
        return False

    def _create_file_backed_segment(self) -> None:
        path = self._fallback_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError:
            self._attach_file_backed_segment()
            return

        try:
            os.ftruncate(fd, self.size)
            mm = mmap.mmap(fd, self.size, access=mmap.ACCESS_WRITE)
        except Exception:
            os.close(fd)
            path.unlink(missing_ok=True)
            raise

        self.shm = _FileBackedSharedMemory(self.name, path, fd, mm)
        self._is_owner = True
        self._backend = "file_mmap"

        buf = self.shm.buf
        try:
            buf[:self.size] = bytes(self.size)
            if hasattr(buf, "obj") and isinstance(buf.obj, mmap.mmap):
                buf.obj.flush()
        finally:
            try:
                buf.release()
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)

        logger.warning(
            "Shared memory %s is using file-backed mmap fallback at %s",
            self.name,
            path,
        )

    def _attach_file_backed_segment(self) -> None:
        path = self._fallback_path()
        if not path.exists():
            raise FileNotFoundError(path)

        fd = os.open(path, os.O_RDWR)
        try:
            size = max(16, int(path.stat().st_size))
            mm = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
        except Exception:
            os.close(fd)
            raise

        self.size = size
        self.shm = _FileBackedSharedMemory(self.name, path, fd, mm)
        self._backend = "file_mmap"
        logger.debug("Attached to file-backed shared memory: %s", path)

    async def create(self):
        """Create the shared memory segment."""
        try:
            shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
            self.shm = shm
            self._is_owner = True
            self._backend = "posix_shm"
            
            # [REAPER] Register SHM segment for post-mortem cleanup
            try:
                from core.reaper import register_reaper_shm
                register_reaper_shm(self.name)
            except Exception as _e:
                logger.debug('Ignored Exception in shared_mem_bus.py: %s', _e)

            # Zero out the memory correctly using bytes and memoryview
            buf = shm.buf
            if buf is not None:
                buf[:self.size] = bytes(self.size)
            logger.debug(f"Shared Memory Segment Created: {self.name} ({self.size} bytes)")
        except FileExistsError:
            # Already exists, just attach
            await self.attach()
        except Exception as e:
            if self._should_use_file_fallback(e):
                logger.warning(
                    "POSIX shared memory unavailable for %s (%s). Falling back to file-backed mmap.",
                    self.name,
                    e,
                )
                self._create_file_backed_segment()
                return
            logger.error(f"Failed to create shared memory {self.name}: {e}")
            raise

    async def attach(self):
        """Attach to an existing shared memory segment with retry resilience."""
        max_retries = 15 # Increase for slow actor starts
        retry_delay = 0.5 # Increase initial delay
        fallback_path = self._fallback_path()
        
        for attempt in range(max_retries):
            try:
                self.shm = shared_memory.SharedMemory(name=self.name)
                self.size = getattr(self.shm, "size", self.size) or self.size
                self._backend = "posix_shm"
                logger.debug(f"✓ Attached to Shared Memory: {self.name}")
                return
            except FileNotFoundError:
                if fallback_path.exists():
                    self._attach_file_backed_segment()
                    return
                if attempt < max_retries - 1:
                    logger.debug(f"⏳ SHM '{self.name}' not found. Retrying in {retry_delay}s... (Attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(retry_delay) # Async sleep
                    retry_delay = min(retry_delay * 1.5, 5.0) # Exponential backoff with cap
                else:
                    logger.error(f"❌ Shared memory segment not found after {max_retries} attempts: {self.name}")
                    raise
            except Exception as e:
                if self._should_use_file_fallback(e):
                    if fallback_path.exists():
                        self._attach_file_backed_segment()
                        return
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 1.5, 5.0)
                        continue
                logger.error(f"❌ Failed to attach to shared memory {self.name}: {e}")
                raise

    def write(self, data: Any):
        """Write serialized data to shared memory with atomic versioning (Seqlock pattern)."""
        if isinstance(data, dict) and "_bus_id" not in data:
            data["_bus_id"] = self._bus_id
        serialized = json.dumps(data).encode("utf-8")
        self._write_bytes(serialized)

    def write_serialized(self, data: Any):
        """Write an already-serialized JSON payload to shared memory."""
        if isinstance(data, str):
            serialized = data.encode("utf-8")
        elif isinstance(data, (bytes, bytearray, memoryview)):
            serialized = bytes(data)
        else:
            raise TypeError("write_serialized expects str or bytes-like data")
        self._write_bytes(serialized)

    def _write_bytes(self, serialized: bytes):
        """Write raw JSON bytes to shared memory with atomic versioning (Seqlock pattern)."""
        shm = self.shm
        if shm is None:
            raise RuntimeError("Not attached to shared memory")
        
        buf = shm.buf
        if buf is None:
            raise RuntimeError("Shared memory buffer is None")

        length = len(serialized)
        if length > self.payload_capacity: # 8 (version) + 4 (length) + 1 (unused)
            raise ValueError(f"Data too large for shared memory ({length} bytes)")

        # 2. Get current version and ensure it's EVEN
        current_ver = int.from_bytes(buf[0:8], byteorder='big')
        if current_ver % 2 != 0:
            logger.warning(f"SHM {self.name}: Pre-write version is ODD ({current_ver}) - repairing.")
            current_ver += 1
        
        try:
            # 3. Set version to ODD (indicates write in progress)
            buf[0:8] = (current_ver + 1).to_bytes(8, byteorder='big')
            
            # [ARM HARDENING] Memory barrier for Apple Silicon
            if hasattr(buf, 'obj') and isinstance(buf.obj, mmap.mmap):
                buf.obj.flush()

            # 4. Write length prefix (Bytes 8-12) and content
            buf[8:12] = length.to_bytes(4, byteorder='big')
            buf[12:12+length] = serialized
            
            # [ARM HARDENING] Ensure data write is flushed before final version bump
            if hasattr(buf, 'obj') and isinstance(buf.obj, mmap.mmap):
                buf.obj.flush()

            # 5. Increment version to EVEN (indicates write complete)
            buf[0:8] = (current_ver + 2).to_bytes(8, byteorder='big')
            
            if hasattr(buf, 'obj') and isinstance(buf.obj, mmap.mmap):
                buf.obj.flush()
                
            logger.debug(f"Wrote {length} bytes to {self.name} (Ver: {current_ver + 2})")
        except Exception as e:
            # [RECOVERY] Always restore to a known-even version on failure to unblock readers
            logger.error(f"Write failure on {self.name}: {e}. Restoring version {current_ver}.")
            try:
                buf[0:8] = current_ver.to_bytes(8, byteorder='big')
                if hasattr(buf, 'obj') and isinstance(buf.obj, mmap.mmap):
                    buf.obj.flush()
            except Exception as _e:
                logger.debug('Ignored Exception in shared_mem_bus.py: %s', _e)
            raise

    async def read(self) -> Any | None:
        """Read and deserialize data from shared memory using atomic version-check (Seqlock)."""
        shm = self.shm
        if shm is None:
            return None
        
        buf = shm.buf
        if buf is None:
            return None
        
        try:
            # Spinlock with exponential backoff
            for attempt in range(10):  # Retry on contention
                # 1. Read start version
                v1 = int.from_bytes(buf[0:8], byteorder='big')
                
                # 2. If version is ODD, a write is in progress. Wait and retry.
                if v1 % 2 != 0:
                    if attempt > 7:
                        logger.warning(f"⚠️ SHM {self.name}: Persistent ODD version ({v1}) detected. Possible dead writer.")
                        if self._is_owner:
                            logger.info(f"🛠️ SHM {self.name}: Owner repairing stale version lock.")
                            buf[0:8] = (v1 + 1).to_bytes(8, byteorder='big')
                            if hasattr(buf, 'obj') and isinstance(buf.obj, mmap.mmap):
                                buf.obj.flush()
                            continue # Retry immediately
                    await asyncio.sleep(0.001 * (2 ** attempt)) # Async sleep
                    continue
                
                # 3. Read length and data
                length = int.from_bytes(buf[8:12], byteorder='big')
                if length == 0:
                    return None
                
                content_raw = bytes(buf[12:12+length])
                
                # 4. Read end version
                v2 = int.from_bytes(buf[0:8], byteorder='big')
                
                # 5. If versions match, the read was atomic.
                if v1 == v2:
                    content = content_raw.decode('utf-8')
                    return json.loads(content)
                
                # Otherwise, a write occurred during our read. Retry.
                logger.debug(f"Torn read detected on {self.name} (v1={v1}, v2={v2}), retrying...")
                await asyncio.sleep(0.001 * (attempt + 1)) # Async sleep
                
            logger.warning(f"Failed to read atomic state from {self.name} after 10 attempts.")
            return None
        except Exception as e:
            logger.error(f"Read failure on {self.name}: {e}")
            return None

    def close(self):
        """Close the shared memory handle."""
        shm = self.shm
        if shm:
            if self._is_owner:
                try:
                    # [REAPER] Deregister before unlinking
                    try:
                        from core.reaper import ReaperManifest
                        ReaperManifest().deregister_shm(self.name)
                    except Exception as _e:
                        logger.warning("🚌 SharedMem: Failed to deregister from Reaper during close: %s", _e)
                    shm.unlink()
                    logger.debug(f"Shared Memory Segment Unlinked: {self.name}")
                except Exception as e:
                    logger.debug(f"Unlink failed (already gone?): {e}")
            shm.close()
            self.shm = None

    def __del__(self):
        self.close()
