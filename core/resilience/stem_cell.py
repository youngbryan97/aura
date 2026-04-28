"""core/resilience/stem_cell.py

Stem-Cell Reversion
===================
Immutable golden snapshots of Aura's *core organs*. When the immune system
detects corruption (incoherent belief storm, repeated subsystem crash,
self-modification gone wrong, identity-hash drift), the stem-cell engine
re-instantiates that organ from its golden version and lets it adapt back
to current state — instead of trying to keep patching a broken organ.

The golden snapshot is taken once, at boot, after the build hash and the
schema version line up. It is stored at:

    ~/.aura/data/stem_cells/{organ}_{schema_version}.signed

…with a HMAC signature using a key the runtime derives from the on-disk
``stem_cells.key`` file (created on first boot, mode 0600). Any tampered
snapshot fails verification and is refused; reversion then falls back to
the most recent prior known-good snapshot.

Organs that have stem-cell coverage:
  - UnifiedWill              (decision policy)
  - AgencyOrchestrator       (life-loop wiring)
  - MemoryFacade index       (subsystem registry; not the data itself)
  - Identity continuity hash (signature pattern)
  - SubstrateAuthority rules (governance baseline)

The engine never reverts memory contents — only structural / policy state.
That keeps autobiographical continuity intact across organ resets.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import hashlib
import hmac
import json
import logging
import os
import pickle
import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_bytes

logger = logging.getLogger("Aura.StemCell")


_STEM_DIR = Path.home() / ".aura" / "data" / "stem_cells"
_STEM_KEY_FILE = _STEM_DIR / "stem_cells.key"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.-]{1,96}$")
_MAX_HEADER_BYTES = 64 * 1024


def _validate_identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"invalid stem cell {field_name}: {value!r}")
    return value


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump())
    if hasattr(value, "__dict__"):
        return {
            str(k): _json_safe(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return str(value)


def _json_serialize(value: Any) -> bytes:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_deserialize(data: bytes) -> Any:
    return json.loads(data.decode("utf-8"))


def _key() -> bytes:
    _STEM_DIR.mkdir(parents=True, exist_ok=True)
    if _STEM_KEY_FILE.exists():
        return _STEM_KEY_FILE.read_bytes().strip()
    raw = secrets.token_bytes(32)
    _STEM_KEY_FILE.write_bytes(raw)
    try:
        os.chmod(_STEM_KEY_FILE, 0o600)
    except Exception:
        pass
    return raw


def _sign(data: bytes) -> bytes:
    return hmac.new(_key(), data, hashlib.sha256).digest()


def _verify(data: bytes, sig: bytes) -> bool:
    return hmac.compare_digest(_sign(data), sig)


@dataclass
class StemCellRecord:
    organ: str
    schema_version: str
    captured_at: float
    payload: bytes
    signature: bytes


class StemCellRegistry:
    """Per-organ snapshot store with HMAC-signed payloads."""

    def __init__(self) -> None:
        _STEM_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._serializers: dict[str, Callable[[Any], bytes]] = {}
        self._deserializers: dict[str, Callable[[bytes], Any]] = {}
        self._instantiators: dict[str, Callable[[Any], Any]] = {}

    # -------- registration --------

    def register(
        self,
        organ: str,
        *,
        serialize: Callable[[Any], bytes] | None = None,
        deserialize: Callable[[bytes], Any] | None = None,
        instantiate: Callable[[Any], Any] | None = None,
        allow_pickle: bool = False,
    ) -> None:
        organ = _validate_identifier(organ, field_name="organ")
        if allow_pickle and serialize is None and deserialize is None:
            logger.warning("pickle stem-cell serialization explicitly enabled for organ=%s", organ)
            serialize = pickle.dumps
            deserialize = pickle.loads
        self._serializers[organ] = serialize or _json_serialize
        self._deserializers[organ] = deserialize or _json_deserialize
        if instantiate is not None:
            self._instantiators[organ] = instantiate

    # -------- capture --------

    def capture(self, organ: str, instance: Any, *, schema_version: str = "1") -> StemCellRecord:
        organ = _validate_identifier(organ, field_name="organ")
        schema_version = _validate_identifier(schema_version, field_name="schema_version")
        with self._lock:
            ser = self._serializers.get(organ) or _json_serialize
            payload = ser(instance)
            sig = _sign(payload)
            record = StemCellRecord(
                organ=organ,
                schema_version=schema_version,
                captured_at=time.time(),
                payload=payload,
                signature=sig,
            )
            self._write(record)
            logger.info("🧬 stem cell captured: organ=%s schema=%s bytes=%d", organ, schema_version, len(payload))
            return record

    # -------- restore --------

    def latest(self, organ: str) -> StemCellRecord | None:
        organ = _validate_identifier(organ, field_name="organ")
        with self._lock:
            candidates = sorted(_STEM_DIR.glob(f"{organ}_*.signed"), reverse=True)
            for path in candidates:
                rec = self._read(path)
                if rec is None:
                    continue
                if _verify(rec.payload, rec.signature):
                    return rec
                logger.warning("🧬 stem cell signature mismatch on %s — skipping", path.name)
            return None

    def revert(self, organ: str, *, schema_version: str | None = None) -> Any:
        organ = _validate_identifier(organ, field_name="organ")
        if schema_version is not None:
            schema_version = _validate_identifier(schema_version, field_name="schema_version")
        rec = self.latest(organ)
        if rec is None:
            raise FileNotFoundError(f"no valid stem cell for organ={organ}")
        if schema_version is not None and rec.schema_version != schema_version:
            logger.warning(
                "🧬 stem cell schema mismatch organ=%s have=%s want=%s",
                organ, rec.schema_version, schema_version,
            )
        des = self._deserializers.get(organ) or _json_deserialize
        instance = des(rec.payload)
        instantiator = self._instantiators.get(organ)
        if instantiator is not None:
            instance = instantiator(instance)
        logger.warning("🧬 stem cell reverted: organ=%s captured_at=%s", organ, rec.captured_at)
        return instance

    # -------- io --------

    @staticmethod
    def _path(record: StemCellRecord) -> Path:
        organ = _validate_identifier(record.organ, field_name="organ")
        schema_version = _validate_identifier(record.schema_version, field_name="schema_version")
        return _STEM_DIR / f"{organ}_{schema_version}_{int(record.captured_at)}.signed"

    @classmethod
    def _write(cls, record: StemCellRecord) -> None:
        path = cls._path(record)
        header = json.dumps({
            "organ": record.organ,
            "schema_version": record.schema_version,
            "captured_at": record.captured_at,
            "signature_hex": record.signature.hex(),
        }).encode("utf-8")
        payload = len(header).to_bytes(4, "big") + header + record.payload
        atomic_write_bytes(path, payload)

    @staticmethod
    def _read(path: Path) -> StemCellRecord | None:
        try:
            with open(path, "rb") as fh:
                header_len = int.from_bytes(fh.read(4), "big")
                if header_len <= 0 or header_len > _MAX_HEADER_BYTES:
                    raise ValueError(f"invalid stem cell header length: {header_len}")
                header = json.loads(fh.read(header_len).decode("utf-8"))
                payload = fh.read()
            return StemCellRecord(
                organ=_validate_identifier(header["organ"], field_name="organ"),
                schema_version=_validate_identifier(
                    header["schema_version"], field_name="schema_version"
                ),
                captured_at=float(header["captured_at"]),
                payload=payload,
                signature=bytes.fromhex(header["signature_hex"]),
            )
        except Exception as exc:
            record_degradation('stem_cell', exc)
            logger.debug("stem cell read failed for %s: %s", path, exc)
            return None


_REGISTRY: StemCellRegistry | None = None


def get_registry() -> StemCellRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = StemCellRegistry()
    return _REGISTRY


__all__ = ["StemCellRegistry", "StemCellRecord", "get_registry"]
