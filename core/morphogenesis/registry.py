from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock
from core.runtime.state_ownership import state_root

from .cell import CellHandler, MorphogenCell
from .organs import Organ
from .types import (
    CellLifecycle,
    CellManifest,
    CellState,
    MorphogenesisConfig,
    stable_digest,
)

logger = logging.getLogger("Aura.Morphogenesis.Registry")


#: Env var that moves the registry off the live instance's data directory.
#:
#: `config.paths.data_dir` resolves to ~/.aura/data whatever AURA_DATA_DIR
#: says, so a test or a sandbox run constructing a registry with no explicit
#: root wrote its state straight into the live instance's file. That happened
#: here on 2026-09-04: a probe run left 27 cells and 17 organs of test-derived
#: state where the live population's belongs.
#:
#: A test-shaped run already sets AURA_LOG_DIR by convention, so that is
#: honoured too rather than adding one more thing to remember.
_ROOT_ENV = "AURA_MORPHOGENESIS_DIR"


def _default_root() -> Path:
    override = os.environ.get(_ROOT_ENV, "").strip()
    if override:
        return Path(override)
    log_dir = os.environ.get("AURA_LOG_DIR", "").strip()
    if log_dir and not log_dir.startswith(str(Path.home() / ".aura")):
        # Redirected logs mean a run that is not the live instance.
        return Path(log_dir) / "morphogenesis"
    if os.environ.get("PYTEST_CURRENT_TEST"):
        # A test that builds a runtime without naming a root must not be able
        # to reach the live population's file. One did, and left 27 cells and
        # 17 organs of test-derived state where the live instance's belongs.
        # Env discipline is not enough on its own: the test that did it ran
        # with AURA_LOG_DIR set correctly and still landed here.
        return Path(tempfile.gettempdir()) / "aura-morphogenesis-tests"
    try:
        from core.config import config
        return Path(config.paths.data_dir) / "morphogenesis"
    except (ImportError, AttributeError, RuntimeError):
        return state_root() / "data" / "morphogenesis"


def _atomic_write_json(path: Path, payload: dict[str, Any], *, schema_name: str) -> None:
    try:
        from core.runtime.atomic_writer import atomic_write_json
        atomic_write_json(path, payload, schema_version=1, schema_name=schema_name)
        return
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation('registry', exc)
        logger.debug("canonical atomic_write_json unavailable for %s: %s", path, exc)

    import json
    import os
    import tempfile
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
        except OSError:
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
        root: Path | None = None,
        config: MorphogenesisConfig | None = None,
    ):
        self.root = Path(root) if root is not None else _default_root()
        self.config = config or MorphogenesisConfig()
        self.state_path = self.root / "morphogenesis_state.json"
        # Taken before the graph lock on the persistence path; both are
        # checked so the ordering is recorded rather than assumed.
        self._lock = checked_lock("morphogenesis.registry", reentrant=True)
        self.cells: dict[str, MorphogenCell] = {}
        self.organs: dict[str, Organ] = {}
        #: Topology, lineage and the motif library, attached by the runtime.
        #:
        #: These were held only in memory, so every restart threw away the
        #: shape the population had developed, who descended from whom, and
        #: every motif that had earned its credit. A developmental layer whose
        #: development does not survive a reboot is a layer that starts from
        #: the seed forever.
        self._graph: Any = None
        self._lineage: Any = None
        self._motifs: Any = None

    def attach_topology(self, *, graph: Any = None, lineage: Any = None, motifs: Any = None) -> None:
        """Hand the registry the state it should persist alongside the cells."""
        if graph is not None:
            self._graph = graph
        if lineage is not None:
            self._lineage = lineage
        if motifs is not None:
            self._motifs = motifs

    def register_cell(self, manifest: CellManifest, *, handler: CellHandler | None = None, replace: bool = False) -> MorphogenCell:
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

    def register_organ(self, organ: Organ) -> MorphogenCell | None:
        with self._lock:
            if organ.organ_id in self.organs:
                return None
            if len(self.organs) >= self.config.max_organs:
                return None
            self.organs[organ.organ_id] = organ
            return self.register_cell(organ.to_manifest(), replace=False)

    def active_cells(self) -> list[MorphogenCell]:
        with self._lock:
            return [
                c for c in self.cells.values()
                if c.lifecycle not in {CellLifecycle.DEAD, CellLifecycle.APOPTOTIC}
            ]

    def get(self, cell_id: str) -> MorphogenCell | None:
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

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": "aura.morphogenesis.registry.v1",
                "created_at": time.time(),
                "config": self.config.to_dict(),
                "cells": {cid: cell.to_dict() for cid, cell in self.cells.items()},
                "organs": {oid: organ.to_dict() for oid, organ in self.organs.items()},
                "graph": self._graph.to_dict() if self._graph is not None else None,
                "lineage": self._lineage.to_dict() if self._lineage is not None else None,
                "motifs": self._motifs.to_dict() if self._motifs is not None else None,
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
            stored_graph = payload.get("graph")
            stored_lineage = payload.get("lineage")
            stored_motifs = payload.get("motifs")
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
        self._restore_topology(stored_graph, stored_lineage, stored_motifs)
        logger.info(
            "Morphogenesis registry loaded: %d cell record(s) restored, %d cell(s) live, "
            "graph v%s with %d binding(s).",
            restored, len(self.cells),
            getattr(self._graph, "version", "-"),
            getattr(self._graph, "edge_count", 0),
        )
        return True

    def _restore_topology(self, graph_data: Any, lineage_data: Any, motif_data: Any) -> None:
        """Put back the developed shape, keeping only what still has a cell.

        A stored binding whose endpoint no longer exists is dropped rather than
        restored: the graph refuses a dangling edge, so restoring one wholesale
        would fail the whole load and cost the topology that was still good.
        """
        live = set(self.cells)
        if graph_data and self._graph is not None:
            try:
                from .graph import MorphEdge

                nodes = [n for n in graph_data.get("nodes", []) if n in live]
                edges = [
                    MorphEdge.from_dict(e) for e in graph_data.get("edges", [])
                    if e.get("source") in live and e.get("target") in live
                ]

                def restore(scratch: Any) -> None:
                    for node in nodes:
                        scratch.add_node(node)
                    for edge in edges:
                        scratch.add_edge(edge)

                self._graph.transaction(restore, cause="registry.load")
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    "registry", exc, severity="warning",
                    action="started with an unseeded topology after a stored graph would not restore",
                )
        if lineage_data and self._lineage is not None:
            try:
                from .lineage import Lineage

                restored = Lineage.from_dict(lineage_data)
                for cell_id, record in restored._records.items():
                    self._lineage._records.setdefault(cell_id, record)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    "registry", exc, severity="warning",
                    action="started with fresh lineage after a stored one would not restore",
                )
        if motif_data and self._motifs is not None:
            try:
                from .motifs import MotifLibrary

                restored_library = MotifLibrary.from_dict(motif_data)
                for motif_id, motif in restored_library._motifs.items():
                    self._motifs._motifs.setdefault(motif_id, motif)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    "registry", exc, severity="warning",
                    action="started with an empty motif library after a stored one would not restore",
                )

    def status(self) -> dict[str, Any]:
        with self._lock:
            by_state: dict[str, int] = {}
            by_role: dict[str, int] = {}
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

