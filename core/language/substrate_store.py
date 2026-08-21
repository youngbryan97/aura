"""Bounded persistence owner for Aura's learned language substrate."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.runtime.file_write_gateway import get_file_write_gateway

_MATCHER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True, slots=True)
class LanguageSubstrateStore:
    """Own the two fixed persistence namespaces used by language learning.

    Callers provide records, not paths. Production roots come from Aura's
    configuration and this module's repository location; tests may inject
    isolated roots without widening the production API.
    """

    data_root: Path
    project_root: Path

    @classmethod
    def configured(cls) -> LanguageSubstrateStore:
        from core.config import config

        return cls(
            data_root=Path(config.paths.data_dir),
            project_root=Path(__file__).resolve().parents[2],
        )

    def matcher_path(self, name: str) -> Path:
        identity = str(name or "").strip()
        if not _MATCHER_NAME.fullmatch(identity):
            raise ValueError("matcher name must be a bounded filesystem-safe identity")
        return self.data_root / "language" / f"{identity}.json"

    def write_matcher(self, name: str, payload: dict[str, Any]) -> Path:
        target = self.matcher_path(name)
        with local_internal_governed_scope("language.substrate_store.matcher"):
            gateway = get_file_write_gateway()
            gateway.ensure_directory(
                target.parent,
                source="language.substrate_store.matcher",
            )
            gateway.write_json(
                target,
                payload,
                schema_version=1,
                schema_name="aura.language.learned_matcher",
                source="language.substrate_store.matcher",
            )
        return target

    def write_measurement(self, receipt: dict[str, Any]) -> Path:
        target = (
            self.project_root
            / "artifacts"
            / "language_substrate"
            / "measurement.json"
        )
        with local_internal_governed_scope("language.substrate_store.measurement"):
            gateway = get_file_write_gateway()
            gateway.ensure_directory(
                target.parent,
                source="language.substrate_store.measurement",
            )
            gateway.write_json(
                target,
                receipt,
                schema_version=1,
                schema_name="aura.language.substrate_measurement",
                source="language.substrate_store.measurement",
            )
        return target


_STORE_LOCK = threading.Lock()
_STORE: LanguageSubstrateStore | None = None


def get_language_substrate_store() -> LanguageSubstrateStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = LanguageSubstrateStore.configured()
        return _STORE


__all__ = ["LanguageSubstrateStore", "get_language_substrate_store"]
