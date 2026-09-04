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
from .types import (
    CellLifecycle,
    CellManifest,
    CellState,
    MorphogenesisConfig,
    json_safe,
    stable_digest,
)
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.Morphogenesis.Registry")


def _default_root() -> Path:
    try:
        from core.config import config
        return Path(config.paths.data_dir) / "morphogenesis"
    except (ImportError, AttributeError, RuntimeError):
        return state_root() / "data" / "morphogenesis"


def _atomic_write_json(path: Path, payload: Dict[str, Any], *, schema_name: str) -> None:
    try:
        from core.runtime.atomic_writer import atomic_write_json
        atomic_write_json(path, payload, schema_version=1, schema_name=schema_name)
        return
    except (ImportError, AttributeError, RuntimeError) as exc:
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
        except (OSError, IOError):
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
    except (ImportError, AttributeError, RuntimeError) as exc:
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
        """Persist the registry, refusing to write an empty one over a full one.

        A bare ``MorphogeneticRuntime()`` constructed anywhere — a probe, a
        health check, a shutdown path that ran before registration — has zero
        cells, and its ``stop()`` used to overwrite the real population with
        ``{}``. That is how the live file came to hold ``cells: {}`` while
        twelve were registered at every boot.

        An empty registry is only written where the file is already empty or
        absent, so a genuine first run still persists.
        """
        with self._lock:
            empty = not self.cells and not self.organs
        if empty and self._stored_is_populated():
            logger.warning(
                "Refusing to persist an empty morphogenesis registry over %s, which holds cells.",
                self.state_path,
            )
            return
        payload = self.snapshot()
        _atomic_write_json(self.state_path, payload, schema_name="morphogenesis_registry")
        _emit_state_receipt(self.state_path, cause="morphogenesis.registry.persist")

    def _stored_is_populated(self) -> bool:
        if not self.state_path.exists():
            return False
        try:
            import json

            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            payload = data.get("payload", data) if isinstance(data, dict) else {}
            return bool(payload.get("cells")) or bool(payload.get("organs"))
        except (OSError, UnicodeDecodeError, ValueError, TypeError, AttributeError):
            # Unreadable is not populated; a first write should be allowed to
            # replace a file nothing can parse.
            return False

    def prune(self) -> dict[str, int]:
        """Housekeeping the loop owes the population.

        ``prune_dead`` existed and had no callers, so apoptotic cells stayed in
        the registry for the life of the process, counted in every status
        report and iterated on every tick. The dormancy and death windows in
        the config were read by nobody at all.
        """
        now = time.time()
        with self._lock:
            became_dormant = 0
            died = 0
            for cell in list(self.cells.values()):
                lifecycle = cell.lifecycle
                if lifecycle == CellLifecycle.APOPTOTIC:
                    idle = cell.state.age_ticks - cell.state.activation_count
                    if idle >= self.config.dead_after_apoptotic_ticks:
                        cell.state.lifecycle = CellLifecycle.DEAD
                        died += 1
                    continue
                if cell.protected or lifecycle != CellLifecycle.ACTIVE:
                    continue
                last = float(cell.state.last_activation_at or 0.0)
                if last and (now - last) > self.config.dormant_after_idle_ticks:
                    cell.state.lifecycle = CellLifecycle.DORMANT
                    became_dormant += 1
        removed = self.prune_dead()
        return {"dormant": became_dormant, "died": died, "removed": removed}

    def load(self) -> bool:
        """Merge persisted state over what is already registered.

        Two things here were wrong in production and between them the live
        registry held zero cells while the loop ticked once a second for
        months.

        It replaced ``self.cells`` outright. Cells are registered by
        ``register_morphogenesis_services`` *before* ``start()`` calls this, so
        an empty or stale file wiped the whole registered population.

        And it dropped every handler. ``MorphogenCell.from_dict`` builds a cell
        with ``handler=None``, so from the second boot onward no cell could do
        anything — ``reattach_handler`` existed and had no callers. State now
        merges onto the registered cell instead of replacing it, which keeps
        the handler that was registered with it.

        The exception list was wrong too: ``json.loads`` raises
        ``JSONDecodeError`` and ``read_text`` raises ``OSError``, neither of
        which was caught, so a corrupt state file raised out of a method whose
        contract is to return False.
        """
        if not self.state_path.exists():
            return False
        try:
            import json

            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            payload = data.get("payload", data) if isinstance(data, dict) else {}
            stored_cells = dict(payload.get("cells", {}))
            stored_organs = dict(payload.get("organs", {}))
        except (OSError, UnicodeDecodeError, ValueError, TypeError, AttributeError) as exc:
            record_degradation(
                "registry", exc, severity="degraded",
                action="kept the registered population after an unreadable morphogenesis state file",
            )
            logger.warning("Morphogenesis registry load failed: %s", exc)
            return False

        restored = 0
        with self._lock:
            for cell_id, cell_data in stored_cells.items():
                existing = self.cells.get(cell_id)
                if existing is not None:
                    # Keep the live object and its handler; take back only the
                    # state it earned last run.
                    try:
                        existing.state = CellState.from_dict(dict(cell_data.get("state", {})))
                        existing.neighbours = {
                            str(k): float(v)
                            for k, v in dict(cell_data.get("neighbours", {})).items()
                        }
                        restored += 1
                    except (TypeError, ValueError) as exc:
                        record_degradation(
                            "registry", exc, severity="warning",
                            action=f"kept live defaults for cell {cell_id} after unreadable stored state",
                        )
                    continue
                if len(self.cells) >= self.config.max_cells:
                    continue
                try:
                    self.cells[cell_id] = MorphogenCell.from_dict(cell_data)
                    restored += 1
                except (TypeError, ValueError) as exc:
                    record_degradation(
                        "registry", exc, severity="warning",
                        action=f"skipped unreadable stored cell {cell_id}",
                    )
            for organ_id, organ_data in stored_organs.items():
                try:
                    self.organs.setdefault(organ_id, Organ.from_dict(organ_data))
                except (TypeError, ValueError) as exc:
                    record_degradation(
                        "registry", exc, severity="warning",
                        action=f"skipped unreadable stored organ {organ_id}",
                    )
        logger.info(
            "Morphogenesis registry loaded: %d cell record(s) restored, %d cell(s) live.",
            restored, len(self.cells),
        )
        return True

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

