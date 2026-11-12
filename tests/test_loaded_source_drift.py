"""Is this process running the code that is on disk?

`SourceBodyAwareness` watches the GIT DIRTY STATE, so committing an edit leaves
the tree clean while the running process stays exactly as stale. Nothing
measured the other quantity, so "are you running my latest fix?" had no
instrument behind it — and three separate defects on 2026-08-10 each cost a long
detour for want of it.
"""

from __future__ import annotations

import contextlib
import importlib.util
import marshal
import pathlib
import py_compile

import core.runtime.loaded_source_drift as loaded_source_drift
from core.runtime.dynamic_execution_gateway import DynamicExecutionGateway
from core.runtime.loaded_source_drift import scan_drift


def test_source_comparison_uses_the_governed_dynamic_compile_owner(
    monkeypatch, tmp_path
):
    source = tmp_path / "subject.py"
    source.write_bytes(b"VALUE = 1\n")
    # The fixture goes through the same owner the module under test uses.
    # A raw compile() here would be the one call site in this file that
    # bypasses the boundary the file exists to prove.
    compiled = DynamicExecutionGateway().compile_source(
        source.read_bytes(),
        filename=str(source),
        mode="exec",
        source="unit.source_comparison_fixture",
        dont_inherit=True,
    )
    cache = tmp_path / "subject.pyc"
    cache.write_bytes(b"\x00" * 16 + marshal.dumps(compiled))
    calls: list[dict] = []

    class _Gateway:
        def compile_source(self, source_code, **kwargs):
            calls.append({"source_code": source_code, **kwargs})
            return compiled

    monkeypatch.setattr(
        loaded_source_drift,
        "get_dynamic_execution_gateway",
        lambda: _Gateway(),
    )

    assert loaded_source_drift._compiled_bodies_differ(source, cache) is False
    assert calls == [
        {
            "source_code": b"VALUE = 1\n",
            "filename": str(source),
            "mode": "exec",
            "source": "loaded_source_drift.compare",
            "dont_inherit": True,
        }
    ]


def test_dynamic_compile_owner_preserves_byte_source_and_future_isolation():
    gateway = DynamicExecutionGateway()
    code = gateway.compile_source(
        b"VALUE = 1\n",
        filename="subject.py",
        mode="exec",
        source="unit.source_comparison",
        dont_inherit=True,
    )

    namespace = gateway.execute_code_object(
        code,
        globals_dict={},
        source="unit.source_comparison",
    )
    assert namespace["VALUE"] == 1


def test_a_freshly_imported_tree_reads_clean():
    import core.conversation.screen_reading_claim  # noqa: F401

    report = scan_drift()
    assert report.checked > 0
    # Pytest assertion rewriting produces bytecode that cannot equal ordinary
    # compilation. Test-only modules are not runtime surfaces and must be
    # excluded rather than misclassified as stale production code.
    assert not any(name.startswith("tests.") for name in report.unknown)
    assert not report.is_stale, report.narrative()


def test_assertion_rewritten_test_modules_are_not_runtime_drift_inputs():
    root = loaded_source_drift._project_root()
    loaded = tuple(loaded_source_drift._loaded_project_modules(root))

    assert not any(name.startswith("tests.") for name, _source, _cache in loaded)


@contextlib.contextmanager
def _compiled_as_the_interpreter_would(path: pathlib.Path):
    """The module's bytecode cache, written as an import writes it, where this process writes none.

    With PYTHONDONTWRITEBYTECODE set (this host's shell sets it) no cache exists, the scanner reports the module
    unknown rather than stale, and the edit below could not be seen at all: the test failed by the host, not by the
    code. A cache it writes is removed afterwards.
    """
    cache = pathlib.Path(importlib.util.cache_from_source(str(path)))
    made = not cache.exists()
    if made:
        py_compile.compile(str(path), cfile=str(cache), doraise=True)
    try:
        yield
    finally:
        if made:
            cache.unlink(missing_ok=True)


def test_a_real_edit_is_detected_and_clears():
    import core.conversation.screen_reading_claim as module

    path = pathlib.Path(module.__file__)
    original = path.read_bytes()
    with _compiled_as_the_interpreter_would(path):
        _an_edit_is_seen_and_clears(path, original)


def _an_edit_is_seen_and_clears(path: pathlib.Path, original: bytes) -> None:
    try:
        path.write_bytes(original + b"\n_DRIFT_PROBE = 1\n")
        report = scan_drift()
        assert "core.conversation.screen_reading_claim" in report.stale_modules
    finally:
        path.write_bytes(original)
    assert not scan_drift().is_stale


def test_cosmetic_rewrites_do_not_cry_stale():
    """A touch, a byte-identical restore, and a comment must all read clean.

    An instrument that fires on `git checkout` is one nobody trusts by the
    third time, so a timestamp mismatch is confirmed against the compiled code
    objects rather than believed.
    """
    import core.conversation.screen_reading_claim as module

    path = pathlib.Path(module.__file__)
    original = path.read_bytes()
    with _compiled_as_the_interpreter_would(path):
        try:
            for rewrite in (original, original + b"\n\n", original + b"\n# a comment\n"):
                path.write_bytes(rewrite)
                report = scan_drift()
                # Measured, not merely unmeasurable: a module the scanner cannot read is never a clean one.
                assert "core.conversation.screen_reading_claim" not in report.unknown
                assert not report.is_stale, rewrite[-20:]
        finally:
            path.write_bytes(original)


def test_dependencies_are_not_project_source():
    """The virtualenv lives INSIDE the repo root here.

    Scanning it swept in every installed package — 792 modules — and this
    instrument answers "am I running the code I just edited". Nobody edits
    their dependencies in place.
    """
    import torch  # noqa: F401  (ships modules under the in-repo .venv)

    report = scan_drift()
    assert not any("site-packages" in drift.path for drift in report.stale)
    assert all(not module.startswith("torch.") for module in report.stale_modules)


def test_a_relative_dunder_file_is_not_mistaken_for_project_source():
    """`torch.classes.__file__` is the bare string "_classes.py".

    `.resolve()` joins that to the current working directory — this repository —
    so two synthetic modules with no source file anywhere were reported as
    project files missing from disk.
    """
    import sys

    import torch  # noqa: F401

    synthetic = [
        name
        for name in ("torch.classes", "torch.ops")
        if name in sys.modules
        and isinstance(getattr(sys.modules[name], "__file__", None), str)
        and not pathlib.Path(sys.modules[name].__file__).is_absolute()
    ]
    if not synthetic:
        return
    report = scan_drift()
    for name in synthetic:
        assert name not in report.stale_modules
        assert name not in report.unknown
