"""core/self/mind_state_export.py — Mind State Export/Import
==================================================================
Export/import a mind state to a .aura-mind archive (ZIP of JSON).

**Read this before trusting a restore.** This module used to open by claiming
it exported "a complete mind state" and listing eight subsystems. It did not,
and could not. Every component was fetched behind ``hasattr(service, "...")``,
and for most of them the named method exists nowhere in the codebase:
``export_snapshot``, ``get_value_weights``, ``export_goals``,
``get_drive_state``, ``load_from_dict``, ``behavioral_scars`` and
``attachment_history`` were all fiction. A guard that is permanently False is
not a graceful degradation; it is a silent permanent no-op, and it reported
success.

Of nine advertised components, three could ever be written and one could ever
be read back. A restore-from-backup would have returned ``{"success": True}``
having reinstated a state vector and nothing else — no memory, no beliefs, no
values, no goals.

So capability is now **declared and probed**, never assumed:

* Each component names the service and the exact attributes it needs to export
  and to restore. If they are missing the component is reported ``unavailable``
  with the reason — in the manifest, in the return value, and in
  ``capability_report()`` — rather than vanishing.
* ``import_mind`` returns ``skipped`` for anything in the archive it cannot put
  back, so "restored" never silently means "partially restored".
* **Every** component is integrity-hashed. Three of nine used to be, and
  ``verify_integrity`` iterated only the hashes that existed and returned
  ``valid: True`` — so a tampered beliefs.json passed the tamper check. A
  component present in the archive with no hash is now *unverified*, which is
  not the same as valid.

``capability_report()`` is the machine-checkable version of the claim this
docstring used to make; ``tests/test_mind_state_export.py`` pins it, so the
list above cannot drift back into fiction.

Security: no private keys, API tokens, or stealth modules in export.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway

logger = logging.getLogger("Aura.MindStateExport")


@dataclass(frozen=True)
class Component:
    """One exportable slice of the mind, and what it actually needs to work.

    ``export_requires`` / ``restore_requires`` are the attribute names that must
    exist on the service. They are declared rather than discovered so that a
    missing one is a *reported* fact instead of a branch that never runs.
    """

    name: str
    service: str
    export: Callable[[Any], dict | None]
    export_requires: tuple[str, ...] = ()
    restore: Callable[[Any, dict], None] | None = None
    restore_requires: tuple[str, ...] = ()

    @property
    def filename(self) -> str:
        return f"{self.name}.json"

    def _missing(self, service: Any, required: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(attr for attr in required if not hasattr(service, attr))

    def export_readiness(self, service: Any) -> tuple[bool, str]:
        if service is None:
            return False, f"service {self.service!r} is not registered"
        missing = self._missing(service, self.export_requires)
        if missing:
            return False, (
                f"{self.service} is missing {', '.join(missing)} — nothing to export"
            )
        return True, ""

    def restore_readiness(self, service: Any) -> tuple[bool, str]:
        if self.restore is None:
            return False, "no restore path is implemented for this component"
        if service is None:
            return False, f"service {self.service!r} is not registered"
        missing = self._missing(service, self.restore_requires)
        if missing:
            return False, f"{self.service} is missing {', '.join(missing)} — cannot restore"
        return True, ""


class MindStateExporter:
    """Export/import a mind state for portability.

    Usage:
        exporter = get_mind_state_exporter()
        await exporter.export_mind("/path/to/aura.aura-mind")
        await exporter.import_mind("/path/to/aura.aura-mind")

    Call ``capability_report()`` first if the answer matters: it says which
    components this instance can genuinely round-trip, and why the rest cannot.
    """

    # Files to NEVER include in exports (security)
    EXCLUDED_PATTERNS = [
        "api_key", "token", "secret", "password", "credential",
        "stealth", "propagation", "sec_ops", "malware",
        "network_recon", ".env", "private_key",
    ]

    def __init__(self) -> None:
        self._started = False
        self._components = self._build_components()

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("mind_state_exporter", self, required=False)
        self._started = True
        logger.info("MindStateExporter ONLINE")

    # ------------------------------------------------------------------
    # The component table — the single place a component is declared
    # ------------------------------------------------------------------

    def _build_components(self) -> tuple[Component, ...]:
        return (
            Component(
                name="canonical_self",
                service="canonical_self",
                export=self._export_canonical_self,
                export_requires=("to_dict",),
                # load_from_dict does not exist on CanonicalSelf. Declared so
                # the gap is reported rather than silently skipped.
                restore=self._restore_canonical_self,
                restore_requires=("load_from_dict",),
            ),
            Component(
                name="substrate_state",
                service="conscious_substrate",
                export=self._export_substrate,
                export_requires=("get_state_summary", "get_state_vector", "get_state_dim"),
                restore=self._restore_substrate,
                restore_requires=("_state",),
            ),
            Component(
                name="memories",
                service="memory_system",
                export=self._export_memory,
                export_requires=("export_snapshot",),
                restore=self._restore_memory,
                restore_requires=("import_snapshot",),
            ),
            Component(
                name="beliefs",
                service="world_state",
                export=self._export_beliefs,
                export_requires=("_beliefs",),
                restore=self._restore_beliefs,
                restore_requires=("set_belief",),
            ),
            Component(
                name="values",
                service="heartstone",
                export=self._export_values,
                export_requires=("get_value_weights",),
                restore=self._restore_values,
                restore_requires=("set_value_weights",),
            ),
            Component(
                name="goals",
                service="goal_manager",
                export=self._export_goals,
                export_requires=("export_goals",),
                restore=self._restore_goals,
                restore_requires=("import_goals",),
            ),
            Component(
                name="drive_baselines",
                service="drive_engine",
                export=self._export_drives,
                export_requires=("get_drive_state",),
                restore=self._restore_drives,
                restore_requires=("set_drive_state",),
            ),
            Component(
                name="scars",
                service="canonical_self",
                export=self._export_scars,
                export_requires=("behavioral_scars",),
                restore=self._restore_scars,
                restore_requires=("behavioral_scars",),
            ),
            Component(
                name="attachments",
                service="canonical_self",
                export=self._export_attachments,
                export_requires=("attachment_history",),
                restore=self._restore_attachments,
                restore_requires=("attachment_history",),
            ),
        )

    @staticmethod
    def _service(name: str) -> Any:
        try:
            return ServiceContainer.get(name, default=None)
        except (ImportError, AttributeError, RuntimeError, KeyError):
            return None

    def capability_report(self) -> dict[str, Any]:
        """What this instance can actually round-trip, and why the rest cannot.

        The machine-checkable replacement for a docstring that listed eight
        subsystems it could not move.
        """
        exportable: dict[str, str] = {}
        restorable: dict[str, str] = {}
        for component in self._components:
            service = self._service(component.service)
            ok, reason = component.export_readiness(service)
            if not ok:
                exportable[component.name] = reason
            ok, reason = component.restore_readiness(service)
            if not ok:
                restorable[component.name] = reason
        names = [c.name for c in self._components]
        return {
            "components": names,
            "can_export": [n for n in names if n not in exportable],
            "cannot_export": exportable,
            "can_restore": [n for n in names if n not in restorable],
            "cannot_restore": restorable,
            "round_trips": [
                n for n in names if n not in exportable and n not in restorable
            ],
        }

    async def export_mind(self, output_path: str) -> dict[str, Any]:
        """Export a mind state to a .aura-mind archive.

        Unavailable components are named in the result and the manifest. They
        used to be omitted silently, which made a three-component archive
        indistinguishable from a nine-component one.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        manifest: dict[str, Any] = {
            "version": "2.0",
            "format": "aura-mind",
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "exported_at_unix": time.time(),
            "components": [],
            "integrity": {},
            "unavailable": {},
        }

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for component in self._components:
                service = self._service(component.service)
                ready, reason = component.export_readiness(service)
                if not ready:
                    manifest["unavailable"][component.name] = reason
                    continue

                data = component.export(service)
                if data is None:
                    manifest["unavailable"][component.name] = (
                        "the service returned nothing to export"
                    )
                    continue

                content = json.dumps(data, indent=2, default=str)
                zf.writestr(component.filename, content)
                manifest["components"].append(component.name)
                # Unconditional: a component written without a hash is one the
                # tamper check cannot see.
                manifest["integrity"][component.name] = hashlib.sha256(
                    content.encode()
                ).hexdigest()[:16]

            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        await get_file_write_gateway().write_bytes_async(
            path,
            buffer.getvalue(),
            source="mind_state_export.export_mind",
        )
        size = (await asyncio.to_thread(path.stat)).st_size

        logger.info(
            "Mind state exported: %s (%d of %d components, %d bytes)",
            path.name, len(manifest["components"]), len(self._components), size,
        )
        if manifest["unavailable"]:
            logger.warning(
                "Mind state export omitted %d component(s): %s",
                len(manifest["unavailable"]),
                "; ".join(f"{k}: {v}" for k, v in manifest["unavailable"].items()),
            )
        return {
            "success": True,
            "path": str(path),
            "size_bytes": size,
            "components": manifest["components"],
            "unavailable": manifest["unavailable"],
            "complete": not manifest["unavailable"],
        }

    async def import_mind(self, archive_path: str) -> dict[str, Any]:
        """Import a mind state from a .aura-mind archive.

        Returns ``imported`` *and* ``skipped``. The previous version returned
        only the former, so a restore that reinstated one component out of
        five reported success and said nothing about the other four.
        """
        path = Path(archive_path)
        if not await asyncio.to_thread(path.exists):
            return {"success": False, "error": f"Archive not found: {archive_path}"}

        by_name = {c.name: c for c in self._components}

        try:
            with zipfile.ZipFile(str(path), "r") as zf:
                names = set(zf.namelist())
                manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
                integrity = manifest.get("integrity", {})

                verification = self._verify_members(zf, names, integrity, by_name)
                if verification is not None:
                    return verification

                imported: list[str] = []
                skipped: dict[str, str] = {}

                for component in self._components:
                    if component.filename not in names:
                        continue
                    service = self._service(component.service)
                    ready, reason = component.restore_readiness(service)
                    if not ready:
                        skipped[component.name] = reason
                        continue
                    try:
                        component.restore(service, json.loads(zf.read(component.filename)))
                    except (ImportError, AttributeError, RuntimeError, TypeError,
                            ValueError, KeyError) as exc:
                        record_degradation(f"mind_import.{component.name}", exc)
                        skipped[component.name] = f"restore raised {type(exc).__name__}"
                        continue
                    imported.append(component.name)

            logger.info(
                "Mind state imported: %d component(s) from %s (%d skipped)",
                len(imported), path.name, len(skipped),
            )
            if skipped:
                logger.warning(
                    "Mind state import skipped: %s",
                    "; ".join(f"{k}: {v}" for k, v in skipped.items()),
                )
            return {
                "success": True,
                "imported": imported,
                "skipped": skipped,
                "complete": not skipped,
            }

        except (zipfile.BadZipFile, json.JSONDecodeError, KeyError) as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def _verify_members(
        zf: zipfile.ZipFile,
        names: set,
        integrity: dict[str, str],
        by_name: dict[str, Component],
    ) -> dict[str, Any] | None:
        """Refuse an archive whose contents do not match, or cannot be checked.

        An unhashed component is *unverified*, which is not the same as valid.
        Treating it as valid is what let a tampered beliefs.json through.
        """
        for component_name, expected in integrity.items():
            component = by_name.get(component_name)
            filename = component.filename if component else f"{component_name}.json"
            if filename not in names:
                continue
            actual = hashlib.sha256(zf.read(filename)).hexdigest()[:16]
            if actual != expected:
                return {
                    "success": False,
                    "error": f"Integrity check failed for {component_name}",
                }

        unhashed = sorted(
            component.name for component in by_name.values()
            if component.filename in names and component.name not in integrity
        )
        if unhashed:
            return {
                "success": False,
                "error": (
                    "Archive contains unverifiable component(s): "
                    f"{', '.join(unhashed)}. Refusing to restore state whose "
                    "integrity cannot be established."
                ),
            }
        return None

    async def verify_integrity(self, archive_path: str) -> dict[str, Any]:
        """Verify every component in the archive is hashed and matches."""
        path = Path(archive_path)
        if not await asyncio.to_thread(path.exists):
            return {"valid": False, "error": "Not found"}

        by_name = {c.name: c for c in self._components}
        try:
            with zipfile.ZipFile(str(path), "r") as zf:
                names = set(zf.namelist())
                manifest = json.loads(zf.read("manifest.json"))
                integrity = manifest.get("integrity", {})

                results: dict[str, Any] = {}
                all_ok = True
                for component_name, expected in integrity.items():
                    component = by_name.get(component_name)
                    filename = component.filename if component else f"{component_name}.json"
                    if filename not in names:
                        continue
                    actual = hashlib.sha256(zf.read(filename)).hexdigest()[:16]
                    ok = actual == expected
                    results[component_name] = {
                        "expected": expected, "actual": actual, "ok": ok,
                    }
                    if not ok:
                        all_ok = False

                unverified = sorted(
                    component.name for component in by_name.values()
                    if component.filename in names and component.name not in integrity
                )
                for component_name in unverified:
                    results[component_name] = {
                        "expected": None, "actual": None, "ok": False,
                        "reason": "present in archive with no recorded hash",
                    }
                    all_ok = False

                return {
                    "valid": all_ok,
                    "components": results,
                    "unverified": unverified,
                }
        except (zipfile.BadZipFile, json.JSONDecodeError) as e:
            return {"valid": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Component exporters
    #
    # Each takes the already-probed service. They no longer re-fetch it or
    # re-guard with hasattr: readiness was decided by the component table, so a
    # guard here would silently swallow the very gap the table exists to report.
    # ------------------------------------------------------------------

    def _export_canonical_self(self, cs: Any) -> dict | None:
        try:
            data = cs.to_dict()
            for key in list(data.keys()):
                if any(p in key.lower() for p in self.EXCLUDED_PATTERNS):
                    del data[key]
            return data
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("mind_export.canonical_self", exc)
        return None

    def _export_substrate(self, sub: Any) -> dict | None:
        try:
            state_vec = sub.get_state_vector()
            return {
                "summary": sub.get_state_summary(),
                "state_vector": state_vec.tolist() if hasattr(state_vec, "tolist") else [],
                "dimension": sub.get_state_dim(),
                "step_count": getattr(sub, "_step_count", 0),
            }
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("mind_export.substrate", exc)
        return None

    def _export_memory(self, mem: Any) -> dict | None:
        try:
            return mem.export_snapshot()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("mind_export.memory", exc)
        return None

    def _export_beliefs(self, ws: Any) -> dict | None:
        try:
            return {
                k: {"value": str(b.value), "confidence": b.confidence, "source": b.source}
                for k, b in ws._beliefs.items() if not b.expired
            }
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("mind_export.beliefs", exc)
        return None

    def _export_values(self, hs: Any) -> dict | None:
        try:
            return hs.get_value_weights()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("mind_export.values", exc)
        return None

    def _export_goals(self, goal_mgr: Any) -> dict | None:
        try:
            return goal_mgr.export_goals()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("mind_export.goals", exc)
        return None

    def _export_drives(self, de: Any) -> dict | None:
        try:
            return de.get_drive_state()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("mind_export.drives", exc)
        return None

    def _export_scars(self, cs: Any) -> dict | None:
        try:
            return {"scars": [str(s) for s in cs.behavioral_scars]}
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("mind_export.scars", exc)
        return None

    def _export_attachments(self, cs: Any) -> dict | None:
        try:
            return {"attachments": cs.attachment_history}
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("mind_export.attachments", exc)
        return None

    # ------------------------------------------------------------------
    # Component restorers
    #
    # These raise on failure rather than swallowing. import_mind catches and
    # records the component as skipped — a restore that quietly does nothing is
    # the defect this module was rebuilt to remove.
    # ------------------------------------------------------------------

    @staticmethod
    def _restore_canonical_self(cs: Any, data: dict) -> None:
        cs.load_from_dict(data)

    @staticmethod
    def _restore_substrate(sub: Any, data: dict) -> None:
        import numpy as np

        vec = data.get("state_vector", [])
        if not vec:
            return
        state = np.array(vec, dtype=np.float32)
        if state.shape != sub._state.shape:
            raise ValueError(
                f"substrate shape mismatch: archive {state.shape} vs live "
                f"{sub._state.shape}; refusing to reshape a mind state"
            )
        sub._state = state

    @staticmethod
    def _restore_memory(mem: Any, data: dict) -> None:
        mem.import_snapshot(data)

    @staticmethod
    def _restore_beliefs(ws: Any, data: dict) -> None:
        for key, belief in data.items():
            if not isinstance(belief, dict):
                continue
            ws.set_belief(
                key,
                belief.get("value"),
                confidence=float(belief.get("confidence", 0.7)),
                source=str(belief.get("source", "imported")),
            )

    @staticmethod
    def _restore_values(hs: Any, data: dict) -> None:
        hs.set_value_weights(data)

    @staticmethod
    def _restore_goals(gm: Any, data: dict) -> None:
        gm.import_goals(data)

    @staticmethod
    def _restore_drives(de: Any, data: dict) -> None:
        de.set_drive_state(data)

    @staticmethod
    def _restore_scars(cs: Any, data: dict) -> None:
        cs.behavioral_scars = list(data.get("scars", []))

    @staticmethod
    def _restore_attachments(cs: Any, data: dict) -> None:
        cs.attachment_history = data.get("attachments", [])

    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        return {"started": self._started}


_instance: MindStateExporter | None = None


def get_mind_state_exporter() -> MindStateExporter:
    global _instance
    if _instance is None:
        _instance = MindStateExporter()
    return _instance


__all__ = ["MindStateExporter", "get_mind_state_exporter"]
