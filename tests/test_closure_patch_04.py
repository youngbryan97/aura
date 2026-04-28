from pathlib import Path
import importlib.util


def _load_installer():
    path = Path("scripts/aura_apply_closure_patch_04.py")
    spec = importlib.util.spec_from_file_location("patch04", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_patch04_registry_lifecycle_counters(tmp_path: Path):
    mod = _load_installer()
    registry = tmp_path / "core" / "morphogenesis" / "registry.py"
    get_task_tracker().create_task(get_storage_gateway().create_dir(registry.parent, cause='test_patch04_registry_lifecycle_counters'))
    registry.write_text(
        'def status(self):\n'
        '    with self._lock:\n'
        '        by_state = {}\n'
        '        by_role = {}\n'
        '        return {\n'
        '            "cells": len(self.cells),\n'
        '            "organs": len(self.organs),\n'
        '            "by_state": by_state,\n'
        '            "by_role": by_role,\n'
        '            "state_path": str(self.state_path),\n'
        '        }\n',
        encoding="utf-8",
    )

    assert mod.patch_morphogenesis_registry(tmp_path) is True
    text = registry.read_text(encoding="utf-8")
    assert '"quarantined": by_state.get("quarantined", 0)' in text
    assert '"dead": by_state.get("dead", 0)' in text


def test_patch04_runtime_uses_fire_and_forget(tmp_path: Path):
    mod = _load_installer()
    runtime = tmp_path / "core" / "morphogenesis" / "runtime.py"
    get_task_tracker().create_task(get_storage_gateway().create_dir(runtime.parent, cause='test_patch04_runtime_uses_fire_and_forget'))
    runtime.write_text(
        "                    try:\n"
        "                        from core.morphogenesis.hooks import record_organ_formation_episode\n"
        "                        asyncio.ensure_future(record_organ_formation_episode(organ.to_dict()))\n"
        "                    except Exception:\n"
        "                        pass\n",
        encoding="utf-8",
    )

    assert mod.patch_morphogenesis_runtime(tmp_path) is True
    text = runtime.read_text(encoding="utf-8")
    assert "fire_and_forget(" in text
    assert "morphogenesis.organ_episode" in text


def test_patch04_terminal_monitor_atomic_blacklist(tmp_path: Path):
    mod = _load_installer()
    terminal = tmp_path / "core" / "terminal_monitor.py"
    get_task_tracker().create_task(get_storage_gateway().create_dir(terminal.parent, cause='test_patch04_terminal_monitor_atomic_blacklist'))
    terminal.write_text(
        "    def _save_blacklist(self):\n"
        "        try:\n"
        "            BLACKLIST_PATH.parent.mkdir(parents=True, exist_ok=True)\n"
        "            BLACKLIST_PATH.write_text(json.dumps(list(self._blacklist)))\n"
        "        except Exception as e:\n"
        "            logger.error(f\"Failed to save blacklist: {e}\")\n",
        encoding="utf-8",
    )

    assert mod.patch_terminal_monitor(tmp_path) is True
    text = terminal.read_text(encoding="utf-8")
    assert "atomic_write_json" in text
    assert 'schema_name="terminal_blacklist"' in text


def test_task_codemod_reports_raw_task(tmp_path: Path):
    codemod_path = Path("scripts/aura_task_ownership_codemod.py")
    spec = importlib.util.spec_from_file_location("task_codemod", codemod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    src = tmp_path / "core" / "x.py"
    get_task_tracker().create_task(get_storage_gateway().create_dir(src.parent, cause='test_task_codemod_reports_raw_task'))
    src.write_text("import asyncio\nasyncio.create_task(foo())\n", encoding="utf-8")

    findings = mod.scan(tmp_path)
    assert findings
    assert findings[0].kind == "asyncio.create_task"
