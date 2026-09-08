"""Subject evidence retains gateway ownership without spelling-based false alarms."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.subject import archive, provenance
from tools.lint_governance import EffectVisitor


def effects(source):
    visitor = EffectVisitor(relative_path="core/example.py")
    visitor.visit(ast.parse(source))
    return visitor.calls


def test_local_write_text_wrapper_does_not_duplicate_its_owned_effect():
    calls = effects('''
from core.runtime.file_write_gateway import get_file_write_gateway
def write_text(path, text):
    get_file_write_gateway().write_text(path, text, source="test")
def serialize(path):
    write_text(path, "data")
''')
    assert [category for category, _, _ in calls] == ["file_write_gateway"]


def test_local_name_never_hides_raw_effect_inside_wrapper():
    calls = effects('''
def write_text(path, text):
    path.write_text(text)
write_text(target, "data")
''')
    assert [category for category, _, _ in calls] == ["raw_file_mutation"]


@pytest.mark.parametrize("source", [
    'def write_text(p, t): pass\nwrite_text = path.write_text\nwrite_text("data")',
    'def write_text(p, t): pass\ndef caller(write_text): write_text("data")',
    'def write_text(p, t): pass\nfrom external import write_text\nwrite_text("data")',
    'def outer():\n def write_text(p, t): pass\nwrite_text("data")',
    'target.write_text("data")',
    '@external\ndef write_text(p, t): pass\nwrite_text("data")',
    'def write_text(p, t): pass\nfrom external import *\nwrite_text("data")',
    'def write_text(p, t): pass\ndel write_text\nwrite_text("data")',
    'def write_text(p, t): pass\nclass write_text: pass\nwrite_text("data")',
])
def test_shadowed_or_unresolved_writers_still_require_ownership(source):
    assert any(category == "raw_file_mutation" for category, _, _ in effects(source))


def test_subject_modules_have_no_raw_effects():
    for module in (archive, provenance):
        calls = effects(Path(module.__file__).read_text())
        assert not [call for call in calls if call[0].startswith("raw_")]


def test_archive_round_trip_uses_gateway_and_preserves_existing_runs(tmp_path, monkeypatch):
    from core.runtime.file_write_gateway import get_file_write_gateway

    gateway = get_file_write_gateway()
    real_write = gateway.write_text
    writes = []
    def record(path, text, **kwargs):
        writes.append((Path(path), kwargs["source"]))
        real_write(path, text, **kwargs)
    monkeypatch.setattr(gateway, "write_text", record)
    first = provenance.next_run_directory(tmp_path / "runs")
    archive.write_json(first, "report.json", {"answer": 17})
    second = provenance.next_run_directory(tmp_path / "runs")
    archive.write_json(second, "report.json", {"answer": 23})
    assert first != second
    assert json.loads((first / "report.json").read_text()) == {"answer": 17}
    assert json.loads((second / "report.json").read_text()) == {"answer": 23}
    assert writes == [(first / "report.json", "subject_core.archive"), (second / "report.json", "subject_core.archive")]


def test_git_probe_declares_read_only_nonaccelerator_owner(monkeypatch):
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=" revision\n")
    monkeypatch.setattr(get_subprocess_gateway(), "run", run)
    assert provenance._git("rev-parse", "HEAD") == "revision"
    argv, options = calls[0]
    assert argv == ["git", "rev-parse", "HEAD"]
    assert options["cwd"] == provenance.REPO
    assert options["read_only"] is True
    assert options["accelerator_capability"] == "none"
    assert options["source"] == "subject_core.provenance.git"


def test_failed_git_probe_cannot_be_recorded_as_success(monkeypatch):
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    monkeypatch.setattr(get_subprocess_gateway(), "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="partial"))
    assert provenance._git("rev-parse", "HEAD") == ""
