from pathlib import Path
import importlib.util
import json


def test_persistence_ownership_round_trip(tmp_path: Path):
    from core.runtime.persistence_ownership import atomic_write_json_owned

    target = tmp_path / "state" / "x.json"
    atomic_write_json_owned(target, {"ok": True}, schema_name="test_schema", schema_version=1)

    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["schema_name"] == "test_schema"
    assert data["payload"]["ok"] is True


def test_persistence_audit_finds_write_text(tmp_path: Path):
    spec = importlib.util.spec_from_file_location(
        "persistence_audit",
        Path("scripts/aura_persistence_audit.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    prod = tmp_path / "core" / "state.py"
    prod.parent.mkdir(parents=True)
    prod.write_text('from pathlib import Path\nPath("x.json").write_text("{}")\n', encoding="utf-8")

    findings = mod.scan(tmp_path)
    assert findings
    assert findings[0].kind.endswith(".write_text")


def test_evidence_collector_writes_bundle(tmp_path: Path):
    spec = importlib.util.spec_from_file_location(
        "evidence",
        Path("scripts/aura_collect_flagship_evidence.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    root = tmp_path / "repo"
    root.mkdir()
    (root / "core" / "runtime").mkdir(parents=True)
    out = tmp_path / "evidence"

    evidence = mod.collect(root, out)

    assert (out / "flagship_evidence.json").exists()
    assert (out / "flagship_evidence.md").exists()
    assert "presence" in evidence
