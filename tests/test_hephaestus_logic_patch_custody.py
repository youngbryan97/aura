"""The deep forge must hand repair candidates a canonical source identity."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.skill_management import hephaestus


class _Brain:
    def __init__(self, original: str, replacement: str) -> None:
        self._content = json.dumps(
            {
                "original_snippet": original,
                "replacement_snippet": replacement,
            }
        )

    async def think(self, _prompt: str) -> SimpleNamespace:
        return SimpleNamespace(content=self._content)


class _Arbitrator:
    @asynccontextmanager
    async def evolution_context(self):
        yield True


def _install_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    root: Path,
    brain: _Brain,
    observed: dict[str, object],
) -> None:
    monkeypatch.setattr(
        hephaestus,
        "config",
        SimpleNamespace(paths=SimpleNamespace(project_root=root)),
    )
    monkeypatch.setattr(
        hephaestus.ServiceContainer,
        "get",
        lambda name, default=None: brain if name == "cognitive_engine" else default,
    )
    monkeypatch.setattr(
        hephaestus,
        "get_resource_arbitrator",
        lambda: _Arbitrator(),
    )

    class _Shadow:
        code_base = root

        async def test_mutation(self, **kwargs):
            observed.update(kwargs)
            return SimpleNamespace(passed=True, runtime_seconds=0.01, errors=[])

    from core.self_modification import shadow_runtime

    monkeypatch.setattr(shadow_runtime, "get_shadow_runtime", lambda _root: _Shadow())


@pytest.mark.asyncio
async def test_deep_forge_normalizes_absolute_target_before_shadow_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    target = root / "core" / "example.py"
    target.parent.mkdir(parents=True)
    target.write_text("value = 1\n", encoding="utf-8")
    observed: dict[str, object] = {}
    _install_dependencies(
        monkeypatch,
        root=root,
        brain=_Brain("value = 1", "value = 2"),
        observed=observed,
    )

    result = await hephaestus.HephaestusEngine().synthesize_logic_patch(
        str(target),
        "change the value",
    )

    assert result["ok"] is True
    assert result["fix"].target_file == "core/example.py"
    assert observed["file_path"] == "core/example.py"
    assert observed["original_code"] == "value = 1\n"
    assert observed["patched_code"] == "value = 2\n"
    assert target.read_text(encoding="utf-8") == "value = 1\n"


@pytest.mark.asyncio
async def test_deep_forge_rejects_ambiguous_source_anchor_before_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    target = root / "core" / "example.py"
    target.parent.mkdir(parents=True)
    target.write_text("value = 1\nvalue = 1\n", encoding="utf-8")
    observed: dict[str, object] = {}
    _install_dependencies(
        monkeypatch,
        root=root,
        brain=_Brain("value = 1", "value = 2"),
        observed=observed,
    )

    result = await hephaestus.HephaestusEngine().synthesize_logic_patch(
        str(target),
        "change one value",
    )

    assert result["ok"] is False
    assert "exactly once" in result["error"]
    assert observed == {}
    assert target.read_text(encoding="utf-8") == "value = 1\nvalue = 1\n"


@pytest.mark.asyncio
async def test_deep_forge_rejects_target_outside_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = tmp_path / "outside.py"
    target.write_text("value = 1\n", encoding="utf-8")
    observed: dict[str, object] = {}
    _install_dependencies(
        monkeypatch,
        root=root,
        brain=_Brain("value = 1", "value = 2"),
        observed=observed,
    )

    result = await hephaestus.HephaestusEngine().synthesize_logic_patch(
        str(target),
        "change the value",
    )

    assert result["ok"] is False
    assert "outside code base" in result["error"]
    assert observed == {}
