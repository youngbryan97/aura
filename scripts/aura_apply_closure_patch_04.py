#!/usr/bin/env python3
"""Idempotent helpers for Aura closure patch 04."""

from pathlib import Path

from core.runtime.atomic_writer import atomic_write_text


def patch_morphogenesis_registry(root: Path) -> bool:
    path = Path(root) / "core" / "morphogenesis" / "registry.py"
    text = path.read_text(encoding="utf-8")
    if '"quarantined": by_state.get("quarantined", 0)' in text:
        return False
    marker = '            "by_state": by_state,\n'
    replacement = (
        '            "by_state": by_state,\n'
        '            "quarantined": by_state.get("quarantined", 0),\n'
        '            "dead": by_state.get("dead", 0),\n'
    )
    if marker not in text:
        return False
    atomic_write_text(path, text.replace(marker, replacement, 1), encoding="utf-8")
    return True


def patch_morphogenesis_runtime(root: Path) -> bool:
    path = Path(root) / "core" / "morphogenesis" / "runtime.py"
    text = path.read_text(encoding="utf-8")
    if "fire_and_forget(" in text:
        return False
    old = "                        asyncio.ensure_future(record_organ_formation_episode(organ.to_dict()))"
    new = (
        "                        from core.runtime.task_ownership import fire_and_forget\n"
        "                        fire_and_forget(\n"
        "                            record_organ_formation_episode(organ.to_dict()),\n"
        '                            name="morphogenesis.organ_episode",\n'
        "                            bounded=True,\n"
        "                        )"
    )
    if old not in text:
        return False
    atomic_write_text(path, text.replace(old, new, 1), encoding="utf-8")
    return True


def patch_terminal_monitor(root: Path) -> bool:
    path = Path(root) / "core" / "terminal_monitor.py"
    text = path.read_text(encoding="utf-8")
    if "atomic_write_json" in text and 'schema_name="terminal_blacklist"' in text:
        return False
    old = "            BLACKLIST_PATH.write_text(json.dumps(list(self._blacklist)))"
    new = (
        "            from core.runtime.atomic_writer import atomic_write_json\n"
        "            atomic_write_json(\n"
        "                BLACKLIST_PATH,\n"
        "                list(self._blacklist),\n"
        '                schema_name="terminal_blacklist",\n'
        "                schema_version=1,\n"
        "            )"
    )
    if old not in text:
        return False
    atomic_write_text(path, text.replace(old, new, 1), encoding="utf-8")
    return True


__all__ = [
    "patch_morphogenesis_registry",
    "patch_morphogenesis_runtime",
    "patch_terminal_monitor",
]
