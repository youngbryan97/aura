from __future__ import annotations
from core.runtime.errors import record_degradation


import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .cell import MorphogenCell, CellHandler
from .field import MorphogenField
from .organs import Organ, OrganStabilizer
from .types import CellLifecycle, CellManifest, MorphogenesisConfig, json_safe, stable_digest

logger = logging.getLogger("Aura.Morphogenesis.Registry")


def _default_root() -> Path:
    try:
        from core.config import config
        return Path(config.paths.data_dir) / "morphogenesis"
    except Exception:
        return Path.home() / ".aura" / "data" / "morphogenesis"


def _atomic_write_json(path: Path, payload: Dict[str, Any], *, schema_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from core.runtime.atomic_writer import atomic_write_json
        atomic_write_json(path, payload, schema_version=1, schema_name=schema_name)
        return
    except Exception as exc:
        record_degradation('registry', exc)
        logger.debug("canonical atomic_write_json unavailable for %s: %s", path, exc)

    import json, os, tempfile
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"schema_version": 1, "schema_name": schema_name, "payload": payload}, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if Path(tmp).exists():
                Path(tmp).unlink()
        except Exception:
            pass  # no-op: intentional


def _emit_state_receipt(path: Path, *, cause: str, key: str = "morphogenesis.registry") -> None:
    try:
        from core.runtime.receipts import StateMutationReceipt, get_receipt_store
        get_receipt_store().emit(
            StateMutationReceipt(
                receipt_id=f"state-{stable_digest(cause, path, time.time())}",
                cause=cause,
                domain="morphogenesis",
                key=key,
                schema_version=1,
                metadata={"path": str(path)},
            )
        )
    except Exception as exc:
        record_degradation('registry', exc)
        logger.debug("morphogenesis receipt skipped: %s", exc)


class MorphogenesisRegistry:
    """Persistent cell/organ registry with bounded durability.

    The registry is deliberately data-first: callable handlers are not
    serialized.  On reload, cells come back with local rules active; service-
    specific handlers can be reattached by integration.py.
    """

    def __init__(
        self,
        *,
        root: Optional[Path] = None,
        config: Optional[MorphogenesisConfig] = None,
    ):
        self.root = Path(root) if root is not None else _default_root()
        self.config = config or MorphogenesisConfig()
        self.state_path = self.root / "morphogenesis_state.json"
        self._lock = threading.RLock()
        self.cells: Dict[str, MorphogenCell] = {}
        self.organs: Dict[str, Organ] = {}

    def register_cell(self, manifest: CellManifest, *, handler: Optional[CellHandler] = None, replace: bool = False) -> MorphogenCell:
        cell = MorphogenCell(manifest, handler=handler)
        with self._lock:
            if cell.cell_id in self.cells and not replace:
                existing = self.cells[cell.cell_id]
                if handler is not None:
                    existing.handler = handler
                return existing
            if len(self.cells) >= self.config.max_cells and not manifest.protected:
                raise RuntimeError(f"morphogenesis registry capacity reached: {len(self.cells)} cells")
            self.cells[cell.cell_id] = cell
            return cell

    def reattach_handler(self, cell_id: str, handler: CellHandler) -> bool:
        with self._lock:
            cell = self.cells.get(cell_id)
            if not cell:
                return False
            cell.handler = handler
            return True

    def register_organ(self, organ: Organ) -> Optional[MorphogenCell]:
        with self._lock:
            if organ.organ_id in self.organs:
                return None
            if len(self.organs) >= self.config.max_organs:
                return None
            self.organs[organ.organ_id] = organ
            return self.register_cell(organ.to_manifest(), replace=False)

    def active_cells(self) -> List[MorphogenCell]:
        with self._lock:
            return [
                c for c in self.cells.values()
                if c.lifecycle not in {CellLifecycle.DEAD, CellLifecycle.APOPTOTIC}
            ]

    def get(self, cell_id: str) -> Optional[MorphogenCell]:
        with self._lock:
            return self.cells.get(cell_id)

    def prune_dead(self) -> int:
        with self._lock:
            dead = [
                cid for cid, c in self.cells.items()
                if c.lifecycle in {CellLifecycle.DEAD, CellLifecycle.APOPTOTIC}
                and not c.protected
            ]
            for cid in dead:
                self.cells.pop(cid, None)
            return len(dead)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "schema": "aura.morphogenesis.registry.v1",
                "created_at": time.time(),
                "config": self.config.to_dict(),
                "cells": {cid: cell.to_dict() for cid, cell in self.cells.items()},
                "organs": {oid: organ.to_dict() for oid, organ in self.organs.items()},
            }

    def save(self) -> None:
        payload = self.snapshot()
        _atomic_write_json(self.state_path, payload, schema_name="morphogenesis_registry")
        _emit_state_receipt(self.state_path, cause="morphogenesis.registry.persist")

    def load(self) -> bool:
        if not self.state_path.exists():
            return False
        try:
            import json
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            # AtomicWriter stores an envelope. Fallback stores same envelope shape.
            payload = data.get("payload", data)
            with self._lock:
                self.cells = {
                    cid: MorphogenCell.from_dict(cell_data)
                    for cid, cell_data in dict(payload.get("cells", {})).items()
                }
                self.organs = {
                    oid: Organ.from_dict(organ_data)
                    for oid, organ_data in dict(payload.get("organs", {})).items()
                }
            return True
        except Exception as exc:
            record_degradation('registry', exc)
            logger.warning("Morphogenesis registry load failed: %s", exc)
            return False

    def status(self) -> Dict[str, Any]:
        with self._lock:
            by_state: Dict[str, int] = {}
            by_role: Dict[str, int] = {}
            for c in self.cells.values():
                state = c.lifecycle.value if hasattr(c.lifecycle, "value") else str(c.lifecycle)
                by_state[state] = by_state.get(state, 0) + 1
                role = c.manifest.role.value if hasattr(c.manifest.role, "value") else str(c.manifest.role)
                by_role[role] = by_role.get(role, 0) + 1
            return {
                "cells": len(self.cells),
                "organs": len(self.organs),
                "active": by_state.get("active", 0),
                "dormant": by_state.get("dormant", 0),
                "hibernating": by_state.get("hibernating", 0),
                "quarantined": by_state.get("quarantined", 0),
                "apoptotic": by_state.get("apoptotic", 0),
                "dead": by_state.get("dead", 0),
                "by_state": by_state,
                "by_role": by_role,
                "state_path": str(self.state_path),
            }

