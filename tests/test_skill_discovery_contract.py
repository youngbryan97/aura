from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.capability_engine import CapabilityEngine
from core.skills.discovery import (
    SkillSourceRoot,
    build_skill_catalog,
    canonicalize_skill_candidates,
    default_skill_roots,
    validate_skill_catalog,
)


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_ast_catalog_covers_supported_declaration_shapes(tmp_path: Path):
    core_root = tmp_path / "fixture_core" / "skills"
    project_root = tmp_path / "project_skills"
    _write(core_root / "__init__.py", "")
    _write(
        core_root / "direct.py",
        """
from core.skills.base_skill import BaseSkill

class DirectSkill(BaseSkill):
    name = "direct"
    description = "Direct assignment skill."
    effect_scope = "pure_compute"
    async def execute(self, params, context):
        return {"ok": True}

class AnnotatedSkill(BaseSkill):
    name: str = "annotated"
    description: str = "Annotated assignment skill."
    effect_scope: str = "read_only"
    async def execute(self, params, context):
        return {"ok": True}

class Container:
    class NestedSkill(BaseSkill):
        name = "nested_must_not_escape"
        description = "Nested helper."
        effect_scope = "pure_compute"
        async def execute(self, params, context):
            return {"ok": True}
""",
    )
    _write(
        core_root / "inherited.py",
        """
from core.skills.base_skill import BaseSkill

class _AbstractCatalogSkill(BaseSkill):
    abstract = True
    description = "Inherited metadata skill."
    effect_scope = "sandboxed_compute"

class InheritedSkill(_AbstractCatalogSkill):
    name = "inherited"
    async def execute(self, params, context):
        return {"ok": True}
""",
    )
    _write(core_root / "package" / "__init__.py", "")
    _write(
        core_root / "package" / "decorated.py",
        """
from core.skills.base_skill import BaseSkill

def aura_skill(**_metadata):
    return lambda cls: cls

@aura_skill(name="decorated", description="Decorator metadata skill.", effect_scope="state_mutation")
class DecoratedSkill(BaseSkill):
    async def execute(self, params, context):
        return {"ok": True}
""",
    )
    _write(project_root / "__init__.py", "")
    _write(
        project_root / "custom.py",
        """
from core.skills.base_skill import BaseSkill

class ProjectSkill(BaseSkill):
    name = "project_skill"
    description = "Project-local capability."
    effect_scope = "read_write_artifacts"
    async def execute(self, params, context):
        return {"ok": True}
""",
    )
    _write(
        core_root / "tests" / "hidden.py",
        "class Hidden: name = 'hidden_test_skill'\n",
    )

    roots = (
        SkillSourceRoot(core_root, "fixture_core.skills", "core"),
        SkillSourceRoot(project_root, "project_skills", "project"),
    )
    catalog = build_skill_catalog(
        roots,
        rust_builder=canonicalize_skill_candidates,
    )

    assert catalog.ok is True
    assert catalog.backend == "rust-canonicalizer+python-discovery"
    assert catalog.parity_status == "canonicalizer_matched"
    assert {item.name for item in catalog.accepted} == {
        "annotated",
        "decorated",
        "direct",
        "inherited",
        "project_skill",
    }
    inherited = next(item for item in catalog.accepted if item.name == "inherited")
    assert inherited.inherited_metadata is True
    assert inherited.effect_scope == "sandboxed_compute"


def test_catalog_rejects_case_insensitive_duplicate_names(tmp_path: Path):
    root = tmp_path / "skills"
    _write(
        root / "duplicates.py",
        """
from core.skills.base_skill import BaseSkill
class FirstSkill(BaseSkill):
    name = "Collision"
    description = "First."
    effect_scope = "pure_compute"
    async def execute(self, params, context): return {"ok": True}
class SecondSkill(BaseSkill):
    name = "collision"
    description = "Second."
    effect_scope = "pure_compute"
    async def execute(self, params, context): return {"ok": True}
""",
    )
    catalog = build_skill_catalog(
        (SkillSourceRoot(root, "fixture", "project"),),
        try_rust=False,
    )

    assert catalog.ok is False
    assert catalog.duplicate_count == 1
    assert catalog.accepted == ()
    assert [issue.code for issue in catalog.blocking_issues] == ["duplicate_skill_name"]


def test_catalog_fails_closed_on_rust_python_divergence(tmp_path: Path):
    root = tmp_path / "skills"
    _write(
        root / "one.py",
        """
from core.skills.base_skill import BaseSkill
class OneSkill(BaseSkill):
    name = "one"
    description = "One."
    effect_scope = "pure_compute"
    async def execute(self, params, context): return {"ok": True}
""",
    )
    catalog = build_skill_catalog(
        (SkillSourceRoot(root, "fixture", "project"),),
        rust_builder=lambda _payload: '{"accepted":[],"duplicates":[]}',
    )

    assert catalog.ok is False
    assert catalog.parity_status == "diverged"
    assert catalog.blocking_issues[0].code == "rust_python_catalog_divergence"


def test_catalog_fails_closed_on_independent_rust_filesystem_divergence(tmp_path: Path):
    root = tmp_path / "skills"
    _write(
        root / "one.py",
        """
from core.skills.base_skill import BaseSkill
class OneSkill(BaseSkill):
    name = "one"
    description = "One."
    effect_scope = "pure_compute"
    async def execute(self, params, context): return {"ok": True}
""",
    )
    catalog = build_skill_catalog(
        (SkillSourceRoot(root, "fixture", "project"),),
        rust_builder=canonicalize_skill_candidates,
        rust_discoverer=lambda _roots: json.dumps(
            {
                "accepted": [],
                "candidates": [],
                "duplicates": [],
                "excluded": [],
                "issues": [],
                "source_file_count": 0,
            }
        ),
    )

    assert catalog.ok is False
    assert catalog.parity_status == "diverged"
    assert [issue.code for issue in catalog.blocking_issues] == [
        "rust_python_filesystem_catalog_divergence"
    ]


def test_discovery_does_not_import_and_probe_blocks_out_of_sandbox_write(tmp_path: Path):
    package = tmp_path / "fixture_skills"
    marker = tmp_path / "must_not_be_written.txt"
    _write(package / "__init__.py", "")
    _write(
        package / "unsafe_import.py",
        f"""
from pathlib import Path
from core.skills.base_skill import BaseSkill
Path({str(marker)!r}).write_text("side effect", encoding="utf-8")
class UnsafeImportSkill(BaseSkill):
    name = "unsafe_import"
    description = "Fixture with import-time mutation."
    effect_scope = "pure_compute"
    async def execute(self, params, context): return {{"ok": True}}
""",
    )
    catalog = build_skill_catalog(
        (SkillSourceRoot(package, "fixture_skills", "project"),),
        try_rust=False,
    )
    assert catalog.ok is True
    assert not marker.exists()

    validations = validate_skill_catalog(
        catalog,
        project_root=tmp_path,
        timeout_s=20,
        use_cache=False,
    )
    result = next(iter(validations.values()))
    assert result["status"] == "quarantined"
    assert result["stage"] == "import"
    assert "outside its sandbox" in result["error"]
    assert not marker.exists()


def test_live_repo_catalog_matches_rust_and_dry_runs_every_skill():
    catalog = build_skill_catalog(default_skill_roots())
    python_catalog = build_skill_catalog(default_skill_roots(), try_rust=False)
    validations = validate_skill_catalog(catalog, use_cache=False)
    engine = CapabilityEngine()

    assert catalog.ok is True
    assert catalog.backend == "rust-filesystem+python-parity"
    assert catalog.parity_status == "matched"
    # 77 shipped skills plus `reminder`, added 2026-08-19 because she said
    # she had set one when nothing was stored. A count that moves when a
    # capability is added is the point of the assertion.
    assert len(catalog.accepted) == 78
    assert len(catalog.excluded) == 10
    assert python_catalog.backend == "python"
    assert python_catalog.parity_status == "python_only"
    assert python_catalog.canonical_payload() == catalog.canonical_payload()
    assert all(item.get("status") == "valid" for item in validations.values())
    assert set(engine.skills) == {item.name for item in catalog.accepted}
    assert engine.get_catalog_health()["quarantined"] == []
    assert engine.is_ready() is True
    assert engine.dry_run_catalog()["ok"] is True


def test_runtime_registration_rejects_metadata_only_skill_claims():
    engine = CapabilityEngine()

    try:
        engine.register_skill({"name": "claimed_but_missing", "description": "No class."})
    except TypeError as exc:
        assert "requires a class" in str(exc)
    else:
        raise AssertionError("metadata-only skill registration must fail")
    assert "claimed_but_missing" not in engine.skills


def test_failed_catalog_reload_preserves_last_known_good_generation(monkeypatch: pytest.MonkeyPatch):
    import core.skills.discovery as discovery

    engine = CapabilityEngine()
    live_registry = engine.skills
    live_names = set(live_registry)
    live_digest = engine.get_catalog_health()["digest"]

    def fail_catalog_build(*_args, **_kwargs):
        raise RuntimeError("injected catalog build failure")

    monkeypatch.setattr(discovery, "build_skill_catalog", fail_catalog_build)

    engine.reload_skills()

    health = engine.get_catalog_health()
    assert engine.skills is live_registry
    assert set(engine.skills) == live_names
    assert health["digest"] == live_digest
    assert health["ready"] is False
    assert health["reason"] == "catalog_build_failed"
    assert health["serving_last_known_good"] is True
